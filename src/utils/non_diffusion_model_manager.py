import torch
from ..configs.configs_base import configs as configs_base
from ..configs.configs_data import data_configs
from ..configs.configs_inference import inference_configs
from ..protenix.config import parse_configs
from ..protenix.data.infer_data_pipeline import get_inference_dataloader
from ..runner.inference import InferenceRunner
from ..protenix.utils.torch_utils import to_device
from .io import AMINO_ACID_ATOMS_ORDER
from ..protenix.data.utils import save_structure_cif
from tqdm import tqdm
import os 
from ..protenix.model.sample_confidence import logits_to_score
import sys
import gemmi
from .io import load_pdb_atom_locations
from ..protenix.metrics.rmsd import self_aligned_rmsd


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

class ProteinxModelManager:
    def __init__(self, sequence, pdb_id,
                reference_pdb=None, # this is used for saving a pdb 
                N_cycle=10, chunk_size=256, diffusion_N=200,
                gamma0=0.8,
                gamma_min=1.0,
                noise_scale_lambda=1.003,
                step_scale_eta = 1,
                dtype="fp32",
                use_deepspeed_evo_attention=False,
                msa_save_dir="./alignment_dir",
                msa_embedding_cache_dir="./msa_cache",
                model_checkpoint_path = "./src/af3-dev/release_model/model_v1.pt",
                dump_dir = "./output",
                use_msa=True,
                batch_size=1,
                device=None,
                ):
        self.sequence = sequence
        self.pdb_id = pdb_id
        self.reference_pdb = reference_pdb
        self.reference_atom_locations = load_pdb_atom_locations(reference_pdb)
        self.reference_atom_locations = self.reference_atom_locations.to(device)

        self.N_cycle = N_cycle
        self.diffusion_N = diffusion_N
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.use_deepspeed_evo_attention = use_deepspeed_evo_attention
        self.msa_save_dir = msa_save_dir
        self.msa_embedding_cache_dir = msa_embedding_cache_dir
        self.model_checkpoint_path = model_checkpoint_path
        self.dump_dir = dump_dir
        self.gamma0 = gamma0
        self.gamma_min = gamma_min
        self.noise_scale_lambda = noise_scale_lambda
        self.step_scale_eta = step_scale_eta
        self.use_msa = use_msa
        self.batch_size = batch_size

        self.runnder = None
        self.atom_array = None
        self.input_feature_dict = None
        self.eval_data_dict = None

        self.msa = None

        self.s_init = None
        self.z_init = None
        self.s_inputs_init = None

        self.protein_x_model = None
        self.denoise_net = None
        self.noise_schedule = None

        self.device = device
        self._setup(device)
    
    def align_models_to_reference(self, strucutres):
        atom_mask = torch.ones(strucutres.shape[1], device=strucutres.device, dtype=torch.bool)[None]
        _, aligned_structure, _, _ = self_aligned_rmsd(strucutres, self.reference_atom_locations, atom_mask)
        return aligned_structure

    def _generate_msa_configuration(self):
        return [{
                "sequences": [
                    {
                        "proteinChain": {
                            "sequence": self.sequence,
                            "count": 1,
                            "msa": {
                                "precomputed_msa_dir": f"./{self.msa_save_dir}/{self.pdb_id}/msa",
                                "pairing_db": "uniref100"
                            }
                        }
                    }
                ],
                "modelSeeds": [],
                "assembly_id": "1",
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
        
        # self.protein_x_model.input_embedder.eval()
        # self.protein_x_model.template_embedder.eval()
        # self.protein_x_model.msa_module.eval()
        # self.protein_x_model.pairformer_stack.eval()

        # Line 1-5
        s_inputs = self.protein_x_model.input_embedder(
            self.input_feature_dict, inplace_safe=False, chunk_size=self.chunk_size
        )  # [..., N_token, 449]
        s_init = self.protein_x_model.linear_no_bias_sinit(s_inputs)
        z_init = (
            self.protein_x_model.linear_no_bias_zinit1(s_init)[..., None, :]
            + self.protein_x_model.linear_no_bias_zinit2(s_init)[..., None, :, :]
        )  #  [..., N_token, N_token, c_z]
        if inplace_safe:
            z_init += self.protein_x_model.relative_position_encoding(self.input_feature_dict)
            z_init += self.protein_x_model.linear_no_bias_token_bond(
                self.input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        else:
            z_init = z_init + self.protein_x_model.relative_position_encoding(self.input_feature_dict)
            z_init = z_init + self.protein_x_model.linear_no_bias_token_bond(
                self.input_feature_dict["token_bonds"].unsqueeze(dim=-1)
            )
        self.s_init = s_init
        self.z_init = z_init
        self.s_inputs_init = s_inputs
    
    def _load_cached_msa_embedding(self):
        file_path = os.path.join(self.msa_embedding_cache_dir, f"{self.pdb_id}.pt")
        if os.path.exists(file_path):
            self.msa = torch.load(file_path, weights_only=False, map_location=self.device)
            self.msa.to(self.device)
    
    def _save_cache_msa_embedding(self):
        file_path = os.path.join(self.msa_embedding_cache_dir, f"{self.pdb_id}.pt")
        if not os.path.exists(file_path):
            os.makedirs(self.msa_embedding_cache_dir, exist_ok=True)
            self.msa.to("cpu")
            torch.save(self.msa, file_path)
            self.msa.to(self.device)

    def _setup(self, device=None):
        configs = self._generate_configs(device)
        dataloader = get_inference_dataloader(configs=configs, msa_configuration=self._generate_msa_configuration())
        self.eval_data_dict, self.atom_array, _ = next(iter(dataloader))[0]
        self.runner = InferenceRunner(configs)
        device = self.runner.device
        self.protein_x_model = self.runner.model
        self.denoise_net = self.protein_x_model.diffusion_module
        self.noise_schedule = self.protein_x_model.inference_noise_scheduler(
            N_step=self.diffusion_N, device=device, dtype=torch.float32
        )
        input_feature_dict = to_device(self.eval_data_dict["input_feature_dict"], device)
        self.input_feature_dict = input_feature_dict

        self._load_cached_msa_embedding()
        if self.msa is None:
            with torch.no_grad():
                self._setup_s_init_z_init()

                s = torch.zeros_like(self.s_init)
                z = torch.zeros_like(self.z_init)
                self.msa = MSA(self.s_inputs_init, s, z) # saving msa because we need self.msa.s_inputs from the pairformer_cycle
                with torch.no_grad():
                    for _ in range(self.N_cycle):
                        s,z = self.pairformer_cycle(s,z)
                self.msa = MSA(self.s_inputs_init, s, z)
                self._save_cache_msa_embedding()
    
    def pairformer_cycle(self, s, z, s_inputs=None, s_init=None, z_init=None, input_feature_dict=None, inplace_safe=False):
        s_inputs = s_inputs if s_inputs is not None else self.msa.s_inputs
        s_init = s_init if s_init is not None else self.s_init
        z_init = z_init if z_init is not None else self.z_init
        input_feature_dict = input_feature_dict if input_feature_dict is not None else self.input_feature_dict

        z = z_init + self.protein_x_model.linear_no_bias_z_cycle(self.protein_x_model.layernorm_z_cycle(z))
        if inplace_safe and self.protein_x_model.template_embedder.n_blocks > 0:
            z += self.protein_x_model.template_embedder(
                input_feature_dict,
                z,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=False,
                use_lma=self.protein_x_model.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=self.chunk_size,
            )
        elif self.protein_x_model.template_embedder.n_blocks > 0:
            z = z + self.protein_x_model.template_embedder(
                input_feature_dict,
                z,
                use_memory_efficient_kernel=self.configs.use_memory_efficient_kernel,
                use_deepspeed_evo_attention=False,
                use_lma=self.protein_x_model.configs.use_lma,
                inplace_safe=inplace_safe,
                chunk_size=self.chunk_size,
            )
        z = self.protein_x_model.msa_module(
            input_feature_dict,
            z,
            s_inputs,
            pair_mask=None,
            use_memory_efficient_kernel=self.protein_x_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protein_x_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        s = s_init + self.protein_x_model.linear_no_bias_s(self.protein_x_model.layernorm_s(s))
        s, z = self.protein_x_model.pairformer_stack(
            s,
            z,
            pair_mask=None,
            use_memory_efficient_kernel=self.protein_x_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protein_x_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        return s, z

    def get_confidance_scores(self, structures, inplace_safe=False):
        plddt_pred, pae_pred, pde_pred, resolved_pred =  self.protein_x_model.run_confidence_head(
            input_feature_dict=self.input_feature_dict,
            s_inputs=self.msa.s_inputs,
            s_trunk=self.msa.s,
            z_trunk=self.msa.z,
            pair_mask=None,
            x_pred_coords=structures,
            use_memory_efficient_kernel=self.protein_x_model.configs.use_memory_efficient_kernel,
            use_deepspeed_evo_attention=False,
            use_lma=self.protein_x_model.configs.use_lma,
            inplace_safe=inplace_safe,
            chunk_size=self.chunk_size,
        )
        score = logits_to_score(plddt_pred, 0, 100, 50)
        return score

    def get_t_hat(self, start_index=0):
       c_tau_last, c_tau = self.noise_schedule[[start_index, start_index + 1]]
       gamma = float(self.gamma0) if c_tau > self.gamma_min else 0
       t_hat = c_tau_last * (gamma + 1)
       return t_hat
    
    def get_x_start(self, batch_size=1, number_of_atoms=None):
        number_of_atoms = number_of_atoms or self.atom_array.shape[0]
        return self.noise_schedule[0] * torch.randn(
            size=(batch_size, number_of_atoms, 3), device=self.runner.device, dtype=torch.float32
        )
    
    def get_x_noisy(self, x_t, start_index=0):
        c_tau_last = self.noise_schedule[start_index]
        t_hat = self.get_t_hat(start_index)
        delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
        x_noisy = x_t + self.noise_scale_lambda * delta_noise_level * torch.randn(
                size=x_t.shape, device=x_t.device, dtype=torch.float32
            )
        return x_noisy
    
    def get_x_t_from_x_0_hat(self, x_noisy, x_0_hat, start_index, end_index=None, guidance_direction=None, step_size=1.0, normalize_gradients=False):
        if end_index is None:
            end_index = start_index + 1
        c_tau = self.noise_schedule[end_index]
        t_hat = self.get_t_hat(start_index)
        delta = (x_noisy - x_0_hat) / t_hat[..., None, None]

        if guidance_direction is not None:
            if normalize_gradients:
                guidance_direction = guidance_direction * delta.flatten(1,-1).norm(dim=-1)[:, None, None]
            delta = delta + step_size * guidance_direction
        dt = c_tau - t_hat
        x_l = x_noisy + self.step_scale_eta * dt[..., None, None] * delta
        return x_l
    
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
