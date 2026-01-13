import torch
from ..configs.configs_base import configs as configs_base
from ..configs.configs_data import data_configs
from ..configs.configs_inference import inference_configs
from ..protenix.config import parse_configs
from ..protenix.data.infer_data_pipeline import get_inference_dataloader
from ..runner.inference import InferenceRunner
from ..protenix.utils.torch_utils import to_device
from .io import AMINO_ACID_ATOMS_ORDER, ATOM_NAME_TO_ELEMENT, create_atom_mask, SEQUENCE_TYPE_TO_ATOM_DICTIONARY, SEQUENCE_TYPE_TO_RESIDUE_KIND, alignment_mask_by_chain
from ..protenix.data.utils import save_structure_cif
from tqdm import tqdm
import os 
from ..protenix.model.sample_confidence import logits_to_score
import sys
import gemmi
from .io import load_pdb_atom_locations, load_pdb_atom_locations_full, load_pdb_atom_locations_without_gaps
from ..protenix.metrics.rmsd import self_aligned_rmsd
from pykeops.torch import LazyTensor
from biotite.structure import AtomArray, array, Atom

def compare_dict(d_1, d_2):
    for key in d_1.keys():
        if not (d_1[key] == d_2[key]).all:
            return False
    return True
    
def clip_norm(tensor, norm):
    with torch.no_grad():
        tensor_norm = tensor.norm(dim=-1, keepdim=True)
        norm = norm.reshape(tensor_norm.shape)
        clipping_tensor = torch.ones_like(norm)
        clipping_mask = tensor_norm > norm
        clipping_tensor[clipping_mask] = norm[clipping_mask] / tensor_norm[clipping_mask]
        tensor.data *= clipping_tensor
    return tensor

class MSA:
    """
        This is a basic class that contains the 3 tensors that represent MSA embeddings
    """
    def __init__(self, s_inputs: torch.Tensor, s: torch.Tensor, z: torch.Tensor):
        self.s_inputs = s_inputs
        self.s = s
        self.z = z
    
    def __iter__(self):
        yield self.s_inputs
        yield self.s
        yield self.z
    
    def parameters(self):
        parameters = []
        for tensor in self:
            if tensor.requires_grad:
                parameters.append(tensor)
        return parameters     
    
    def zeros_like(self):
        return MSA(torch.zeros_like(self.s_inputs), torch.zeros_like(self.s), torch.zeros_like(self.z))
    
    def clone(self):
        return MSA(self.s_inputs.clone(), self.s.clone(), self.z.clone())
    
    def to(self, device):
        self.s_inputs = self.s_inputs.to(device)
        self.s = self.s.to(device)
        self.z = self.z.to(device)
    
    def requires_grad(self, required_grad, apply_to_s_inputs=True):
        self.s_inputs.requires_grad = apply_to_s_inputs and required_grad
        self.s.requires_grad = required_grad
        self.z.requires_grad = required_grad

    def detach(self):
        self.s_inputs = self.s_inputs.detach()
        self.s = self.s.detach()
        self.z = self.z.detach()
        return self
    
    def __add__(self, other):
        return MSA(*[item_1 + item_2 for item_1, item_2 in zip(self, other)])

def atom_mask_from_residue_mask(atom_array, residue_mask):
    activated_residues = set(residue_mask.nonzero().squeeze().tolist())
    atom_mask = torch.zeros(atom_array.shape[0], dtype=torch.bool, device="cpu")
    for i, atom in enumerate(atom_array):
        if (atom.res_id - 1) in activated_residues:
            atom_mask[i] = True
    atom_mask = atom_mask.to(residue_mask.device)
    return atom_mask

class ProtenixModelManager:
    def __init__(self, sequences_dictionary, pdb_id,
                should_align_to_chains=[0],
                assembly_identifier=None,
                chains_to_read=[0],
                ROI_residues = None,
                reference_pdb=None,
                pdb_contains_missing_atoms=False,
                N_cycle=10, chunk_size=256, diffusion_N=200,
                gamma0=0.8,
                gamma_min=1.0,
                noise_scale_lambda=1.003,
                step_scale_eta = 1,
                dtype="fp32",
                use_deepspeed_evo_attention=False,
                use_lma=False,
                msa_save_dir="./alignment_dir",
                msa_embedding_cache_dir="./msa_cache",
                pairformer_mixed_precision=False,
                model_checkpoint_path = "./src/af3-dev/release_model/model_v1.pt",
                dump_dir = "./output",
                use_msa=True,
                batch_size=1,
                device="cuda" if torch.cuda.is_available() else "cpu",
                should_concatenate_frozen_atoms=False,
                rmax_for_mask = 5.0,
                enable_memory_snapshot=False,
                ):
        self.pdb_id = pdb_id
        self.reference_pdb = reference_pdb
        self.should_align_to_chains = should_align_to_chains
        self.sequences_dictionary = sequences_dictionary
        self.assembly_identifier = assembly_identifier
        self.pdb_contains_missing_atoms = pdb_contains_missing_atoms
        self.full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in sequences_dictionary]
        # flattening the list of lists
        self.full_sequences = [item for sublist in self.full_sequences for item in sublist]
        self.chains_to_read = chains_to_read
        self.ROI_residues = ROI_residues
        self.sequence_types = [
            sequence_type
            for dictionary in sequences_dictionary
            for sequence_type in [dictionary.get("sequence_type", "proteinChain")] * dictionary["count"]
        ]

        if self.pdb_contains_missing_atoms:
            result = load_pdb_atom_locations_full(
                reference_pdb, 
                full_sequences_dict=self.sequences_dictionary,
                chains_to_read=chains_to_read,
                return_starting_indices=True,  # Get starting indices to preserve PDB residue numbering
            )
            self.reference_atom_locations = result[0]
            self.resolved_pdb_to_full_mask = result[1].to(device)
            # Store starting residue indices (1-indexed) for preserving PDB numbering when saving
            self.starting_residue_indices = result[-1] if len(result) > 2 else None
        else:
            self.reference_atom_locations = load_pdb_atom_locations(reference_pdb)
            self.resolved_pdb_to_full_mask = torch.ones(self.reference_atom_locations.shape[1], dtype=torch.bool, device=self.reference_atom_locations.device)
            self.starting_residue_indices = None  # No starting indices if PDB doesn't contain missing atoms (simple case)

        # TODO: Advaith
        if self.ROI_residues is not None:
            self.ROI_atom_mask = create_atom_mask(self.full_sequences, ROI_residues, sequence_types=self.sequence_types).to(device)
        else: 
            self.ROI_atom_mask = torch.zeros_like(self.resolved_pdb_to_full_mask, dtype=torch.bool, device=self.reference_atom_locations.device)

        self.reference_atom_locations = self.reference_atom_locations.to(device)

        if should_concatenate_frozen_atoms:
            # Only find frozen atoms around chain "B" (sequence index 0) to reduce computational complexity
            # Chain "1" is just an anchor, so we don't need frozen atoms around it
            frozen_atoms_chain_indices = [0]  # Sequence index 0 = chain "B"
            self.frozen_atoms_dict = _compute_frozen_atoms_and_concatenateable_masks_and_params(
                atom_array=self.reference_atom_locations[self.resolved_pdb_to_full_mask], # resolved atoms for distance calculation
                reference_pdb_path=self.reference_pdb,
                resolved_mask=self.resolved_pdb_to_full_mask,  # Mask to identify which atoms are resolved
                full_sequences_dict=self.sequences_dictionary,  # Needed to read all atoms correctly
                device=device,
                rmax_for_mask=rmax_for_mask,
                frozen_atoms_chain_indices=frozen_atoms_chain_indices,  # Only find frozen atoms around chain "B"
            )

        self.N_cycle = N_cycle
        self.diffusion_N = diffusion_N
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.use_deepspeed_evo_attention = use_deepspeed_evo_attention
        self.use_lma = use_lma
        self.msa_save_dir = msa_save_dir
        self.msa_embedding_cache_dir = msa_embedding_cache_dir

        if self.assembly_identifier is not None:
            self.msa_full_embedding_cache_dir = os.path.join(self.msa_embedding_cache_dir, self.pdb_id, self.assembly_identifier)
        else:
            self.msa_full_embedding_cache_dir = os.path.join(self.msa_embedding_cache_dir, self.pdb_id)

        self.pairformer_mixed_precision = pairformer_mixed_precision
        self.model_checkpoint_path = model_checkpoint_path
        self.dump_dir = dump_dir
        self.gamma0 = gamma0
        self.gamma_min = gamma_min
        self.noise_scale_lambda = noise_scale_lambda
        self.step_scale_eta = step_scale_eta
        self.use_msa = use_msa
        self.batch_size = batch_size
        self.enable_memory_snapshot = enable_memory_snapshot

        self.atom_array = None
        self.input_feature_dict = None
        self.eval_data_dict = None

        self.msa = None

        self.s_init = None
        self.z_init = None
        self.s_inputs_init = None

        self.protenix_model = None
        self.denoise_net = None
        self.noise_schedule = None

        self.device = device
        
        # Wrap _setup in exception handler to dump memory snapshot on OOM
        try:
            self._setup(device)
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            # RuntimeError is raised for CUDA OOM in some PyTorch versions
            if self.enable_memory_snapshot and ("out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError)):
                self._dump_memory_snapshot()
            raise
    
    def align_models_to_reference(self, strucutres, reduced_atom_mask=None):
        if reduced_atom_mask is not None:
            atom_mask = reduced_atom_mask
        else:
            atom_mask = torch.ones(strucutres.shape[1], device=strucutres.device, dtype=torch.bool)[None]

        # TODO: Fix advaith with residue range
        _, aligned_structure, _, _ = self_aligned_rmsd( strucutres, self.reference_atom_locations, atom_mask & (~self.ROI_atom_mask.to(device=atom_mask.device)) )
        return aligned_structure

    @staticmethod
    def _trim_unresolved_sequences(sequences_dictionary, pdb_path, chains_to_read: list[str] | None = None):
        """ 
        Reads the pdb file, and trims the aligned sequences in such a way 
        that the unresolved residues [at each of the ends of each chain] are removed. 
        Does not remove the missing regions inside the chains. Only on each end.
        Trimming is done such that the first and the last True value in the mask is included. Anything outside is removed from the sequence.
        """
        masks_per_chain_per_residue_per_atom = load_pdb_atom_locations_full(
            pdb_path,
            full_sequences_dict=sequences_dictionary,
            chains_to_read=chains_to_read,
            output_masks_per_chain=True,
        )
        masks_per_chain_per_residue = [
            [any(
                [nested_one_element_list[0] for nested_one_element_list in masks_per_atom] # have to flatten out due to the incoming shape of these lists (it's nested)
            ) for masks_per_atom in masks_per_chain]
            for masks_per_chain in masks_per_chain_per_residue_per_atom
        ]

        # Group chains by which sequence_dictionary entry they belong to
        # and OR the masks together for each group
        chain_idx = 0
        trimmed_sequences_dictionary = []
        
        for seq_dict in sequences_dictionary:
            count = seq_dict["count"]
            sequence = seq_dict["sequence"]
            
            # Get the masks for all chains corresponding to this sequence
            group_masks = masks_per_chain_per_residue[chain_idx:chain_idx + count]
            
            # OR the masks together element-wise for each residue position
            # All masks in the group should have the same length (same sequence)
            # zip(*group_masks) transposes: from [mask1, mask2, ...] to [(res0_mask1, res0_mask2, ...), (res1_mask1, res1_mask2, ...), ...]
            # Then any() checks if at least one mask has True for that residue position
            combined_mask = [any(masks_at_residue) for masks_at_residue in zip(*group_masks)] if group_masks else []
            
            # Find first and last True indices
            true_indices = [i for i, val in enumerate(combined_mask) if val]
            
            if true_indices:
                first_true_idx = true_indices[0]
                last_true_idx = true_indices[-1]
                
                # Trim the sequence to only include residues from first True to last True (inclusive)
                trimmed_sequence = sequence[first_true_idx:last_true_idx + 1]
            else:
                # No resolved residues found, keep original sequence (or empty?)
                trimmed_sequence = sequence
            
            # Create a new dictionary with the trimmed sequence, preserving other fields
            trimmed_dict = seq_dict.copy()
            trimmed_dict["sequence"] = trimmed_sequence
            trimmed_sequences_dictionary.append(trimmed_dict)
            
            chain_idx += count
        
        return trimmed_sequences_dictionary


    def _generate_msa_configuration(self):
        return [
            {
                "sequences": [
                    {
                        sequence_dict["sequence_type"]: {
                            "sequence": sequence_dict["sequence"],
                            "count": sequence_dict["count"],
                            **({
                                "msa": {
                                    "precomputed_msa_dir": f"{self.msa_save_dir}/msa/{i+1}",
                                    "pairing_db": "uniref100"
                                }
                            } if sequence_dict["sequence_type"] == "proteinChain" else {}),
                            "modifications": []
                        }
                    }
                    for i, sequence_dict in enumerate(self.sequences_dictionary) 
                ],
                "modelSeeds": [],
                "assembly_id": getattr(self, "assembly_identifier", "1"),
                "name": self.pdb_id
            }
            
        ]

    def _generate_configs(self, device=None):
        configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        configs.update(
            {
                "load_checkpoint_path": self.model_checkpoint_path,
                "dump_dir": self.dump_dir,
                "use_deepspeed_evo_attention": self.use_deepspeed_evo_attention,
                "use_lma": self.use_lma,
            }
        )
        configs["dtype"] = self.dtype
        configs["use_msa"] = self.use_msa
        if device is not None:
            configs["device"] = device
        configs["model"]["N_cycle"] = self.N_cycle
        configs["sample_diffusion"]["N_step"] = self.diffusion_N
        configs["sample_diffusion"]["N_sample"] = self.batch_size
        
        # parse_configs reads argv inside, and this breaks things because we want it to be empty, we we will cache it
        argv = sys.argv[:]
        sys.argv = argv[:1]
        configs = parse_configs(
            configs=configs,
            fill_required_with_null=True,
        )
        sys.argv = argv
        # this line makes the diffusion process run in parallel
        configs.infer_setting.sample_diffusion_chunk_size = None
        return configs
    
    def _setup_s_init_z_init(self, inplace_safe=True):
        
        # self.protenix_model.input_embedder.eval()
        # self.protenix_model.template_embedder.eval()
        # self.protenix_model.msa_module.eval()
        # self.protenix_model.pairformer_stack.eval()

        # Use input_feature_dict directly - autocast will handle dtype conversion
        input_feature_dict = self.input_feature_dict

        # Line 1-5
        s_inputs = self.protenix_model.input_embedder(
            input_feature_dict, inplace_safe=False, chunk_size=self.chunk_size
        )  # [..., N_token, 449]
        s_init = self.protenix_model.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.protenix_model.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.protenix_model.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.protenix_model.relative_position_encoding(input_feature_dict)
            z_init += self.protenix_model.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.protenix_model.relative_position_encoding(input_feature_dict)
            z_init = z_init + self.protenix_model.linear_no_bias_token_bond(
                input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        self.s_init = s_init
        self.z_init = z_init
        self.s_inputs_init = s_inputs
    
    def _load_cached_msa_embedding(self): # not configuration anymore, the prepared msa embeddings generated using msa configs etc...! 
        file_path = os.path.join(self.msa_full_embedding_cache_dir, f"msa.pt")
        if os.path.exists(file_path):
            self.msa = torch.load(file_path, weights_only=False)
            self.msa.to(self.device)
    
    def _save_cache_msa_embedding(self):
        file_path = os.path.join(self.msa_full_embedding_cache_dir, f"msa.pt")
        if not os.path.exists(file_path):
            os.makedirs(self.msa_full_embedding_cache_dir, exist_ok=True)
            self.msa.to("cpu")
            torch.save(self.msa, file_path)
            self.msa.to(self.device)

    def _load_cached_features(self):
        """Load cached input_feature_dict and atom_array if they exist."""
        file_path = os.path.join(self.msa_full_embedding_cache_dir, f"features.pt")
        if os.path.exists(file_path):
            cached_data = torch.load(file_path, weights_only=False)
            self.eval_data_dict = cached_data["eval_data_dict"]
            self.atom_array = cached_data["atom_array"]
            return True
        return False
    
    def _save_cached_features(self):
        """Save input_feature_dict and atom_array to cache."""
        file_path = os.path.join(self.msa_full_embedding_cache_dir, f"features.pt")
        if not os.path.exists(file_path):
            os.makedirs(self.msa_full_embedding_cache_dir, exist_ok=True)
            # Move tensors to CPU for saving (similar to MSA caching)
            eval_data_dict_cpu = {}
            for k, v in self.eval_data_dict.items():
                if isinstance(v, torch.Tensor):
                    eval_data_dict_cpu[k] = v.to("cpu")
                else:
                    eval_data_dict_cpu[k] = v
            
            cached_data = {
                "eval_data_dict": eval_data_dict_cpu,
                "atom_array": self.atom_array,  # AtomArray should be CPU-friendly
            }
            torch.save(cached_data, file_path)
            # Note: We don't move back to device here because device will be determined
            # by InferenceRunner later, and features will be moved to device after that

    def _selective_to_device(self, feature_dict, device, exclude_msa=True):
        """
        Move features to device, but exclude large MSA tensors to save memory.
        MSA features will be moved to device when msa_module needs them.
        """
        result = {}
        # MSA-related keys that are large and can be deferred
        msa_keys = {"msa", "has_deletion", "deletion_value"}
        
        for k, v in feature_dict.items():
            if isinstance(v, dict):
                result[k] = self._selective_to_device(v, device, exclude_msa)
            elif isinstance(v, torch.Tensor):
                if exclude_msa and k in msa_keys:
                    result[k] = v  # Keep on CPU for now
                elif v.device == device:
                    result[k] = v  # Already on device
                else:
                    result[k] = v.to(device)
            else:
                result[k] = v
        return result
    
    def _dump_memory_snapshot(self, filename="memory_snapshot.pickle"):
        """Dump PyTorch memory snapshot to file for debugging."""
        if not self.enable_memory_snapshot or not torch.cuda.is_available():
            return
        
        try:
            snapshot_path = os.path.join(os.getcwd(), filename)
            torch.cuda.memory._dump_snapshot(snapshot_path)
            print(f"\n{'='*80}")
            print(f"Memory snapshot saved to: {snapshot_path}")
            print(f"View it at: https://docs.pytorch.org/memory_viz")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Warning: Failed to dump memory snapshot: {e}")

    def _setup(self, device=None):
        # Enable memory snapshot recording if requested
        if self.enable_memory_snapshot and torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history()
            except AttributeError:
                # Fallback for older PyTorch versions
                if hasattr(torch.cuda.memory, 'snapshot'):
                    torch.cuda.memory.snapshot()
        
        configs = self._generate_configs(device)
        
        # Try to load cached features first
        features_cached = self._load_cached_features()
        
        if not features_cached:
            # Compute features if not cached
            dataloader = get_inference_dataloader(configs=configs, msa_configuration=self._generate_msa_configuration())
            self.eval_data_dict, self.atom_array, error_message = next(iter(dataloader))[0]

            if "input_feature_dict" not in self.eval_data_dict:
                raise ValueError(f"input_feature_dict not found in eval_data_dict. The error that lead to this is: {error_message}")
            
            # Save features to cache
            self._save_cached_features()

        self.runner = InferenceRunner(configs)
        device = self.runner.device
        self.protenix_model = self.runner.model
        self.denoise_net = self.protenix_model.diffusion_module
        self.noise_schedule = self.protenix_model.inference_noise_scheduler(
            N_step=self.diffusion_N, device=device, dtype=torch.float32
        )
        
        # Selectively move features to device: exclude large MSA tensors initially
        # MSA features are processed later by msa_module, so we can defer moving them
        input_feature_dict = self._selective_to_device(self.eval_data_dict["input_feature_dict"], device)
        self.input_feature_dict = input_feature_dict

        self._load_cached_msa_embedding()
        if self.msa is None:
            with torch.no_grad():
                # some oligomeric msa and pairformers are just too big..!
                # Use mixed precision for initialization too to save memory
                with torch.amp.autocast("cuda", dtype=torch.bfloat16 if self.pairformer_mixed_precision else torch.float32):
                    self._setup_s_init_z_init()
                    # Autocast handles dtype conversion during operations, no need for explicit .to() calls

                s = torch.zeros_like(self.s_init)
                z = torch.zeros_like(self.z_init)
                # saving msa because we need self.msa.s_inputs from the pairformer_cycle
                self.msa = MSA(self.s_inputs_init, s, z)

                # some oligomeric msa and pairformers are just too big..!
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16 if self.pairformer_mixed_precision else torch.float32):
                    for _ in tqdm(range(self.N_cycle), desc="Pairformer Cycle"):
                        s,z = self.pairformer_cycle(s,z)
                
                # Convert back to float32 only at the end for storage
                if self.pairformer_mixed_precision:
                    s = s.to(torch.float32)
                    z = z.to(torch.float32)
                    self.s_init = self.s_init.to(torch.float32)
                    self.z_init = self.z_init.to(torch.float32)
                    self.s_inputs_init = self.s_inputs_init.to(torch.float32)
                self.msa = MSA(self.s_inputs_init, s, z)
                self._save_cache_msa_embedding()
    
    def pairformer_cycle(self, s, z, s_inputs=None, s_init=None, z_init=None, input_feature_dict=None, inplace_safe=False):
        s_inputs = s_inputs if s_inputs is not None else self.msa.s_inputs
        s_init = s_init if s_init is not None else self.s_init
        z_init = z_init if z_init is not None else self.z_init
        input_feature_dict = input_feature_dict if input_feature_dict is not None else self.input_feature_dict

        # Move deferred MSA features to device lazily when needed
        if input_feature_dict is self.input_feature_dict:
            msa_keys = {"msa", "has_deletion", "deletion_value"}
            for k in msa_keys:
                if k in input_feature_dict and isinstance(input_feature_dict[k], torch.Tensor):
                    if input_feature_dict[k].device != self.device:
                        input_feature_dict[k] = input_feature_dict[k].to(self.device)

        z = z_init + self.protenix_model.linear_no_bias_z_cycle(self.protenix_model.layernorm_z_cycle(z))
        if inplace_safe and self.protenix_model.template_embedder.n_blocks > 0:
            z += self.protenix_model.template_embedder(
                input_feature_dict,
                z,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=False,
                use_lma=self.protenix_model.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=self.chunk_size,
            )
        elif self.protenix_model.template_embedder.n_blocks > 0:
            z = z + self.protenix_model.template_embedder(
                input_feature_dict,
                z,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=False,
                use_lma=self.protenix_model.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=self.chunk_size,
            )
        z = self.protenix_model.msa_module(
            input_feature_dict,
            z,
            s_inputs,
            pair_mask=None,
            use_memory_efficient_kernel=self.protenix_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protenix_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        s = s_init + self.protenix_model.linear_no_bias_s(self.protenix_model.layernorm_s(s))
        s, z = self.protenix_model.pairformer_stack(
            s,
            z,
            pair_mask=None,
            use_memory_efficient_kernel=self.protenix_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protenix_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        return s, z

    def get_confidance_scores(self, structures, inplace_safe=False):
        plddt_pred, pae_pred, pde_pred, resolved_pred =  self.protenix_model.run_confidence_head(
            input_feature_dict=self.input_feature_dict,
            s_inputs=self.msa.s_inputs,
            s_trunk=self.msa.s,
            z_trunk=self.msa.z,
            pair_mask=None,
            x_pred_coords=structures,
            use_memory_efficient_kernel=self.protenix_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protenix_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        score = logits_to_score(plddt_pred, 0, 100, 50)
        return score

    def get_t_hat(self, start_index=0, end_index=None):
        """
        end_index to support recycling steps
        """
        if end_index is None:
            end_index = start_index + 1
        c_tau_last, c_tau = self.noise_schedule[[start_index, end_index]]
        gamma = float(self.gamma0) if c_tau > self.gamma_min else 0
        t_hat = c_tau_last * (gamma + 1)
        return t_hat
    
    def get_x_start(self, batch_size=1, number_of_atoms=None):
        number_of_atoms = number_of_atoms or self.atom_array.shape[0]
        return self.noise_schedule[0] * torch.randn(
            size=(batch_size, number_of_atoms, 3), device=self.runner.device, dtype=torch.float32
        )
    
    def get_x_noisy(self, x_t, start_index=0, end_index=None):
        if end_index is None:
            end_index = start_index + 1

        if end_index > start_index:
            c_tau_last = self.noise_schedule[end_index]
            t_hat = self.get_t_hat(end_index)
            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            x_noisy = x_t + self.noise_scale_lambda * delta_noise_level * torch.randn(
                    size=x_t.shape, device=x_t.device, dtype=torch.float32
                )
            return x_noisy
        else:
            # Take forward steps
            sigma_start = self.noise_schedule[start_index]
            sigma_end = self.noise_schedule[end_index]
            std = (sigma_end**2 - sigma_start**2).sqrt()
            eps = torch.randn(size=x_t.shape, device=x_t.device, dtype=torch.float32)
            return x_t + self.noise_scale_lambda * std * eps

    # def get_x_t_from_x_0_hat(self, x_noisy, x_0_hat, start_index, end_index=None, guidance_direction=None, step_size=1.0, normalize_gradients=False):
    def get_x_t_from_x_0_hat(
            self, x_noisy, x_0_hat, start_index, end_index=None, 
            guidance_direction=None, step_size=1.0, normalize_gradients=False, normalize_by_current_gradient_strength=False,
            structures_gradient_norm=1.0, guidance_scale_gradually_increase=False,
        ):
        if end_index is None:
            end_index = start_index + 1
        is_backward_step = end_index > start_index

        if is_backward_step:
            c_tau = self.noise_schedule[end_index]
            t_hat = self.get_t_hat(start_index)
            delta = (x_noisy - x_0_hat) / t_hat[..., None, None]

            if guidance_direction is not None:
                if normalize_gradients:
                    # Normalize guidance to match delta norm (per batch element), then multiply by constants
                    # Same normalization logic for both scalar and per-atom constants
                    if isinstance(structures_gradient_norm, torch.Tensor):
                        # Per-atom constants: structures_gradient_norm shape [N_atoms]
                        # Expand to [1, N_atoms, 1] for broadcasting with [B, N_atoms, 3]
                        per_atom_constants = structures_gradient_norm.unsqueeze(0).unsqueeze(-1)  # [1, N_atoms, 1]
                        guidance_direction = guidance_direction * delta.norm(dim=(1,2), keepdim=True) / guidance_direction.norm(dim=(1, 2), keepdim=True) * per_atom_constants
                    else:
                        # Scalar constant: use original behavior
                        guidance_direction = guidance_direction * delta.norm(dim=(1,2), keepdim=True) / guidance_direction.norm(dim=(1, 2), keepdim=True) * structures_gradient_norm
                delta = delta + step_size * guidance_direction

            dt = c_tau - t_hat
            x_l = x_noisy + self.step_scale_eta * dt[..., None, None] * delta
            return x_l
        else:
            # Also, applying no guidance when adding the noise back (no need, will be applied during denoising anyway..!)
            return x_noisy
    
    def get_x_0_hat_from_x_noisy(self, x_noisy, t_hat=None, start_index=None,msa=None,input_feature_dict=None, inplace_safe=False):
        # make sure that t_hat is not None or t is not None, but not both
        assert not ((t_hat is None) and (start_index is None))
        assert not ((not t_hat is None) and (not start_index is None))
        if t_hat is None:
            t_hat = self.get_t_hat(start_index)
        msa = msa or self.msa
        input_feature_dict = input_feature_dict or self.input_feature_dict
        return self.denoise_net(
                    x_noisy=x_noisy,
                    t_hat_noise_level=t_hat.reshape([1]).repeat(x_noisy.shape[0]),
                    input_feature_dict=input_feature_dict,
                    s_inputs=msa.s_inputs,
                    s_trunk=msa.s,
                    z_trunk=msa.z,
                    chunk_size=self.chunk_size,
                    inplace_safe=inplace_safe,
                )
    
    def single_diffusion_step(self, x_t, start_index,end_index=None, inplace_safe=True, hook=None):
        x_noisy = self.get_x_noisy(x_t, start_index)
        x_0_hat = self.get_x_0_hat_from_x_noisy(x_noisy, start_index=start_index, inplace_safe=inplace_safe)
        if hook is not None:
            hook(x_0_hat, start_index)
        x_t_end = self.get_x_t_from_x_0_hat(x_noisy, x_0_hat, start_index,end_index=end_index)
        return x_t_end
                
    def save_structure_cif(self, structure, file_name):
        polytype = self.eval_data_dict["entity_poly_type"]
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        save_structure_cif(self.atom_array, structure, file_name, polytype, self.pdb_id)

    def save_structure_pdb(self, structure, atom_array, write_file_name):
        gemmi_structure = gemmi.read_structure(self.reference_pdb)
        chain = gemmi_structure[0][0]
        num_structures = len(gemmi_structure)
        for i in range(num_structures - 1, 0, -1):
            del gemmi_structure[i]
        num_models = len(gemmi_structure[0])
        for i in range(num_models - 1, 0, -1):
            del gemmi_structure[0][i]
        atom_index = 0
        structure = structure.cpu().detach().numpy()
        for res in chain:
            res_name = res.name
            atom_order = AMINO_ACID_ATOMS_ORDER[res_name]
            atoms = [[atom for atom in res if atom.name == atom_name][0] for atom_name in atom_order]
            if res[-1].name == "OXT":
                atoms.append(res[-1])
            for atom in atoms:
                atom.pos = gemmi.Position(*structure[atom_index])
                atom_index += 1

        os.makedirs(os.path.dirname(write_file_name), exist_ok=True)
        gemmi_structure.write_pdb(write_file_name)


def save_structure_full(structure, full_sequences, sequence_types, atom_array, write_file_name, bfactors=None, atom_mask=None, chain_names=None, starting_residue_indices=None):
    """
    Save structure with proper chain names matching original PDB chain names.
    
    Args:
        chain_names: Optional list of chain names. If provided, should match the order of full_sequences.
                    If None or shorter than full_sequences, defaults to A, B, C... for missing entries.
        starting_residue_indices: Optional list of starting residue indices (1-indexed) for each chain.
                                 If provided, residues will be numbered starting from these indices (preserving original PDB numbering).
                                 If None, residues start from 1.
    """
    gemmi_structure = gemmi.Structure()
    model = gemmi.Model("1") 
    
    chains = []
    if atom_mask is None:
        atom_mask = torch.ones(structure.shape[0], dtype=torch.bool)
    

    for i in range(len(full_sequences)):
        # Use provided chain name if available, otherwise default to A, B, C...
        if chain_names is not None and i < len(chain_names) and chain_names[i] is not None:
            chain_name = chain_names[i]
        else:
            chain_name = chr(ord("A") + i)
        chains.append(gemmi.Chain(chain_name)) 

    iter_index = 0
    
    structure = structure.cpu().detach().numpy()
    if bfactors is not None:
        # Convert bfactors to numpy array if it's a tensor (so indexing returns a scalar float)
        if isinstance(bfactors, torch.Tensor):
            bfactors = bfactors.cpu().detach().numpy()
        # Handle 2D bfactors (ensemble_size, N_atoms) - flatten to 1D by taking first ensemble member
        if bfactors.ndim > 1:
            bfactors = bfactors[0]  # Take first ensemble member's b-factors
    
    for chain_i, (chain, sequence_type) in enumerate(zip(chains, sequence_types)): 
        sequence = full_sequences[chain_i]
        # Determine starting residue number for this chain
        if starting_residue_indices is not None and chain_i < len(starting_residue_indices):
            start_residue_num = starting_residue_indices[chain_i]
        else:
            start_residue_num = 1  # Default to starting from 1
        
        for i, res_name_one_letter in enumerate(sequence): # creating the residue
            res = gemmi.Residue()
            res.name =  gemmi.expand_one_letter(res_name_one_letter, SEQUENCE_TYPE_TO_RESIDUE_KIND[sequence_type]) 
            res.seqid = gemmi.SeqId(start_residue_num + i, " ")  # Use preserved starting index
            residue_has_atoms = False  # Track if this residue has any atoms
            
            # Every first residue of a dna or rna chain should have the OP3 atom
            if sequence_type in ["rnaSequence", "dnaSequence"] and i == 0:
                if atom_mask[iter_index] == True:
                    atom = gemmi.Atom()
                    atom.name = "OP3"
                    atom.element = gemmi.Element("O")
                    atom.pos = gemmi.Position(*structure[iter_index])
                    if bfactors is not None:
                        atom.b_iso = bfactors[iter_index]
                    res.add_atom(atom)
                    residue_has_atoms = True
                iter_index += 1

            
            for atom_name in SEQUENCE_TYPE_TO_ATOM_DICTIONARY[sequence_type][res.name]: # appending all the atoms in the residue    
                if atom_mask[iter_index] == True:
                    # Save this atom - it's resolved in the structure
                    atom = gemmi.Atom()
                    atom.name = atom_name
                    atom.element = gemmi.Element(ATOM_NAME_TO_ELEMENT[atom_name])
                    atom.pos = gemmi.Position(*structure[iter_index])
                    if bfactors is not None:
                        atom.b_iso = bfactors[iter_index]

                    res.add_atom(atom)
                    residue_has_atoms = True
                
                # Always increment iter_index to move through the concatenated mask
                iter_index += 1
            
            # Handle OXT atom for the last residue of each chain
            if i == len(sequence) - 1 and sequence_type == "proteinChain":  # Last residue in chain
                if iter_index < len(atom_mask):
                    if atom_mask[iter_index] == True:
                        atom = gemmi.Atom()
                        atom.name = "OXT"
                        atom.element = gemmi.Element("O")
                        atom.pos = gemmi.Position(*structure[iter_index])
                        if bfactors is not None:
                            atom.b_iso = bfactors[iter_index]

                        res.add_atom(atom)
                        residue_has_atoms = True
                    
                    # Always increment if we're within bounds (whether OXT exists or not)
                    iter_index += 1

            # Only add residue to chain if it has at least one atom
            if residue_has_atoms:
                chain.add_residue(res)
    
        model.add_chain(chain)
    
    gemmi_structure.add_model(model)
    
    os.makedirs(os.path.dirname(write_file_name), exist_ok=True)
    gemmi_structure.write_pdb(write_file_name)
    # Return the in-memory structure so callers can reuse it (e.g., for merging/altloc)
    return gemmi_structure



def _compute_frozen_atoms_and_concatenateable_masks_and_params(
        atom_array: torch.Tensor, # resolved atoms for distance calculation
        reference_pdb_path: str,
        resolved_mask: torch.Tensor,  # Mask indicating which atoms are resolved (True = resolved, False = unresolved)
        full_sequences_dict: list[dict],  # Not used directly, but kept for compatibility
        device: torch.DeviceObjType = torch.device("cpu"),
        rmax_for_mask: float = 5.0,
        frozen_atoms_chain_indices: list[int] = None,  # Sequence indices (0-indexed) to use for finding frozen atoms. If None, uses all chains.
    ) -> dict[str, torch.Tensor]:
    """
    Compute frozen atoms: atoms that are NOT in the resolved mask but are within rmax_for_mask distance.
    These are atoms from the PDB that exist but are not being optimized (e.g., missing atoms, atoms from other chains, etc.)
    
    Strategy:
    1. Read ALL atoms from the PDB (all chains) using load_pdb_atom_locations_without_gaps
    2. Match atoms by position to identify which atoms are NOT in the resolved set (atom_array)
    3. Find unresolved atoms within rmax_for_mask distance of resolved atoms from specified chains only
    
    Args:
        frozen_atoms_chain_indices: List of sequence indices (0-indexed) to use for finding frozen atoms.
            Only atoms from these chains will be used for distance calculation. If None, uses all chains.
    """
    # Build full_sequences and sequence_types for creating chain masks
    full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in full_sequences_dict]
    full_sequences = [item for sublist in full_sequences for item in sublist]
    sequence_types = [
        sequence_type
        for dictionary in full_sequences_dict
        for sequence_type in [dictionary.get("sequence_type", "proteinChain")] * dictionary["count"]
    ]
    
    # Filter atom_array to only include atoms from specified chains (if provided)
    if frozen_atoms_chain_indices is not None and len(frozen_atoms_chain_indices) > 0:
        # Create mask for atoms from specified chains (shape: [N_all_atoms])
        chain_mask = alignment_mask_by_chain(
            full_sequences,
            chains_to_align=frozen_atoms_chain_indices,
            sequence_types=sequence_types
        ).to(device)
        # Get mask for atoms that are both resolved AND from specified chains
        # resolved_mask has shape [N_all_atoms], chain_mask has shape [N_all_atoms]
        filtered_resolved_mask = resolved_mask & chain_mask
        # atom_array is already filtered by resolved_mask, so we need to find which positions
        # in atom_array correspond to atoms from specified chains
        # We can do this by checking chain_mask at the positions where resolved_mask is True
        atom_array_chain_mask = chain_mask[resolved_mask]  # Shape: [N_resolved_atoms]
        atom_array_filtered = atom_array[atom_array_chain_mask]
    else:
        # Use all atoms if no chain filter specified
        atom_array_filtered = atom_array
    
    # Step 1: Read ALL atoms from the PDB (all chains, no filtering)
    all_atoms_from_pdb_positions, all_atoms_from_pdb_atomic_numbers, all_atoms_from_pdb_bfacs = \
        load_pdb_atom_locations_without_gaps(
            pdb_file=reference_pdb_path, 
            chains_to_read=None,  # Read all chains
            device=device,
        )
    
    if atom_array_filtered.shape[0] == 0:
        # No resolved atoms, return empty dict
        return {
            "close_to_relevant_chains_mask": torch.tensor([], dtype=torch.bool, device=device),
            "other_atoms_from_pdb_positions": torch.tensor([], dtype=torch.float32, device=device).reshape(0, 3),
            "other_atoms_from_pdb_atomic_numbers": torch.tensor([], dtype=torch.int32, device=device),
            "other_atoms_from_pdb_bfacs": torch.tensor([], dtype=torch.float32, device=device),
            "insertable_array": array([]),
        }
    
    # Step 2: Match atoms by position to find which atoms from all_atoms are NOT in the resolved set
    # For each atom in all_atoms, check if it's close to any resolved atom (within 0.1A tolerance)
    # If not close to any resolved atom, it's an unresolved atom
    # Tolerance: 0.1 Angstrom (atoms at same position should match)
    match_tolerance = 0.1
    distances_to_resolved = (
        LazyTensor(all_atoms_from_pdb_positions[:, None]) - LazyTensor(atom_array_filtered[None, :])
    ).square().sum(dim=2).sqrt()  # shape: [num_all_atoms, num_filtered_resolved_atoms]
    
    # Evaluate LazyTensor: check if any distance is within tolerance (using .sum() to evaluate)
    # An atom is "resolved" if it's within tolerance of any resolved atom
    is_resolved = (distances_to_resolved < match_tolerance).sum(dim=1).flatten() > 0  # shape: [num_all_atoms]
    unresolved_mask = ~is_resolved  # True for unresolved atoms
    
    unresolved_atoms_positions = all_atoms_from_pdb_positions[unresolved_mask]
    unresolved_atoms_elements = all_atoms_from_pdb_atomic_numbers[unresolved_mask]
    unresolved_atoms_bfacs = all_atoms_from_pdb_bfacs[unresolved_mask]
    
    if unresolved_atoms_positions.shape[0] == 0:
        # No unresolved atoms, return empty dict
        return {
            "close_to_relevant_chains_mask": torch.tensor([], dtype=torch.bool, device=device),
            "other_atoms_from_pdb_positions": torch.tensor([], dtype=torch.float32, device=device).reshape(0, 3),
            "other_atoms_from_pdb_atomic_numbers": torch.tensor([], dtype=torch.int32, device=device),
            "other_atoms_from_pdb_bfacs": torch.tensor([], dtype=torch.float32, device=device),
            "insertable_array": array([]),
        }
    
    # Step 3: Find unresolved atoms that are within rmax_for_mask distance of atoms from specified chains only
    distances = (
        LazyTensor(unresolved_atoms_positions[:, None]) - LazyTensor(atom_array_filtered[None, :])
    ).square().sum(dim=2).sqrt() # shape: [num_unresolved_atoms, num_filtered_resolved_atoms]
    
    # Find unresolved atoms that are within rmax_for_mask distance of any resolved atom
    close_to_relevant_chains_mask = (distances < rmax_for_mask).sum(dim=1).flatten() > 0  # True if within distance
    close_to_relevant_chains_mask = close_to_relevant_chains_mask.to(torch.bool)

    # Filter to only the unresolved atoms that are close to resolved atoms
    frozen_atoms_positions = unresolved_atoms_positions[close_to_relevant_chains_mask]
    frozen_atoms_elements = unresolved_atoms_elements[close_to_relevant_chains_mask]
    frozen_atoms_bfacs = unresolved_atoms_bfacs[close_to_relevant_chains_mask]
    
    # additionally, we need an atom array object in order to ensure correct concatenation of the atoms.
    atoms = [
        Atom(
            atomic_coordinates, 
            atomic_number = int(atomic_number.item()),
            element = gemmi.Element(int(atomic_number.item())).name,
            chain_id = "Z", # NOTE TODO: make sure this id will never match any of the other existing chain ids in the pdb 
        ) 
        for atomic_coordinates, atomic_number in zip(frozen_atoms_positions.cpu(), frozen_atoms_elements.cpu())
    ]
    insertable_array = array(atoms) # On purpose, contains no bonds. But, importantly, contains atomic numbers and element names -> needed for collision loss initialization..!

    output_dict = {
        "close_to_relevant_chains_mask": close_to_relevant_chains_mask,
        "other_atoms_from_pdb_positions": frozen_atoms_positions,
        "other_atoms_from_pdb_atomic_numbers": frozen_atoms_elements,
        "other_atoms_from_pdb_bfacs": frozen_atoms_bfacs,
        "insertable_array": insertable_array,
    }
    
    return output_dict

