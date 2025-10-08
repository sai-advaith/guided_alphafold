import torch
import os
import wandb
from tqdm import tqdm
import json
import numpy as np
import gemmi
import random

from src.protenix.metrics.rmsd import self_aligned_rmsd
from src.utils.aa_bonded_pairs import find_bonded_pairs
from src.utils.relaxation import relax_pdb
from src.utils.io import load_pdb_atom_locations, get_sampler_pdb_inputs, delete_hydrogens, write_back_pdb_coordinates
from src.utils.io import load_config, namespace_to_dict, query_msa_server
from src.utils.non_diffusion_model_manager import ProtenixModelManager
from src.utils.non_diffusion_model_manager import save_structure_full
from src.losses import * 

class ExperimentManager:
    def __init__(self, config, device, config_file_path=None):
        # general
        self.device = device
        self.config_file_path = config_file_path
        self.config = config
        self.name = f"{self.config.general.name}_{self.config.protein.pdb_id}"

        self.msa_full_save_dir = None
        self.query_msa_server()
        self.model_manager = self._get_model_manager()
        self._setup_wandb()
        self.loss_function = self._get_loss_function()


        if self.config.protein.assembly_identifier is not None:
            self.experiment_save_dir = os.path.join(self.config.general.output_folder, self.config.protein.pdb_id, self.config.protein.assembly_identifier)
        else:
            self.experiment_save_dir = os.path.join(self.config.general.output_folder, self.config.general.name)

        os.makedirs(self.experiment_save_dir, exist_ok=True)

        # Deterministic settings
        np.random.seed(self.config.general.seed)
        torch.manual_seed(self.config.general.seed)
        torch.cuda.manual_seed(self.config.general.seed)
        random.seed(self.config.general.seed)

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Make torch.cdist deterministic
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Disable mixed precision for maximum determinism
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    def _setup_wandb(self):
        if self.config.wandb.login_key is not None:
            wandb.login(key=self.config.wandb.login_key)
            wandb.init(project=self.config.wandb.project, name=self.name, mode=self.config.wandb.mode, config=namespace_to_dict(self.config))

    def _get_model_manager(self):
        return ProtenixModelManager(
            sequences_dictionary=self.config.protein.sequences,
            pdb_id=self.config.protein.pdb_id,
            assembly_identifier=self.config.protein.assembly_identifier,
            chains_to_read=self.config.protein.chains_to_use,
            # TODO: Check with xray code about residue indices
            ROI_residues=self.config.protein.residue_range if self.config.protein.residue_range is not None else None,
            should_align_to_chains=self.config.protein.should_align_to_chains,
            reference_pdb=self.config.protein.reference_pdb,
            pdb_contains_missing_atoms=self.config.protein.contains_missing_atoms,
            N_cycle=self.config.model_manager.N_cycle,
            chunk_size=self.config.model_manager.chunk_size,
            diffusion_N=self.config.model_manager.diffusion_N,
            gamma0=self.config.model_manager.gamma0,
            gamma_min=self.config.model_manager.gamma_min,
            noise_scale_lambda=self.config.model_manager.noise_scale_lambda,
            step_scale_eta=self.config.model_manager.step_scale_eta,
            dtype="fp32",
            use_deepspeed_evo_attention=False,
            msa_save_dir=self.msa_full_save_dir,
            msa_embedding_cache_dir=self.config.model_manager.msa_embedding_cache_dir,
            model_checkpoint_path=self.config.model_manager.model_checkpoint_path,
            dump_dir=self.config.model_manager.dump_dir,
            use_msa=self.config.model_manager.use_msa,
            batch_size=self.config.general.batch_size,
            device=self.device,
            pairformer_mixed_precision=self.config.model_manager.pairformer_mixed_precision,
            )
    
    def _get_loss_function(self):
        loss_functions = []
        weights = []

        # Defining objects for the main loss functions!
        self.density_loss_function = None
        self.nmr_loss_function = None
        self.heavy_noe_loss_function = None
        self.relax_times_loss_function = None
        self.cryoesp_loss_function = None

        for loss_function_type in self.config.loss_function.loss_function_type:
            if loss_function_type == "density":
                density_config = self.config.loss_function.density_loss_function

                # Create density object
                loss_function = DensityGuidanceLossFunction(reference_pdbs=density_config.reference_pdbs,
                                                            full_pdb=self.config.protein.reference_raw_pdb,
                                                            chain=self.config.protein.reference_raw_pdb_chain,
                                                            aligned_density_file=density_config.density_file,
                                                            altloc_region=self.config.protein.residue_range[0],
                                                            rmax=self.config.loss_function.density_loss_function.rmax,
                                                            device=self.device,
                                                            batch_size=self.config.general.batch_size)
                self.density_loss_function = loss_function
                loss_functions.append(loss_function)
                loss_weight = getattr(density_config, "weight", 1)
                weights.append(loss_weight)

            elif loss_function_type == "nmr":
                noe_config = self.config.loss_function.nmr_loss_function
                loss_function = NMRLossFunction(restraint_file=noe_config.reference_nmr,
                                                        pdb_file=noe_config.pdb_file,
                                                        atom_array=self.model_manager.atom_array,
                                                        device=self.device, 
                                                        methyl_relax_file=noe_config.methyl_relax_file,
                                                        methyl_relax_scale=noe_config.methyl_relax_scale,
                                                        amide_rdc_file=noe_config.amide_rdc_file,
                                                        amide_rdc_scale=noe_config.amide_rdc_scale,
                                                        amide_relax_file=noe_config.amide_relax_file,
                                                        amide_relax_scale=noe_config.amide_relax_scale,
                                                        methyl_rdc_scale=noe_config.methyl_rdc_scale,
                                                        methyl_rdc_file=noe_config.methyl_rdc_file,
                                                        noe_scale=noe_config.noe_scale,
                                                        # op_n_bootstrap=noe_config.op_n_bootstrap,
                                                        iid_loss=noe_config.iid_loss)
                loss_functions.append(loss_function)
                self.nmr_loss_function = loss_function
                loss_weight = getattr(noe_config, "weight", 1)
                weights.append(loss_weight)
            elif loss_function_type == "relax_times":
                relax_times_config = self.config.loss_function.relax_times_loss_function
                loss_function = RelaxTimesLossFunction(atom_array=self.model_manager.atom_array,
                                                       coefficient_files=relax_times_config.coefficient_files,
                                                       data_file=relax_times_config.data_file,
                                                       device=self.device,
                                                       num_discretize_points=relax_times_config.num_discretize_points,
                                                       batch_size=self.config.general.batch_size)
                self.relax_times_loss_function = loss_function
                loss_functions.append(loss_function)
                loss_weight = getattr(relax_times_config, "weight", 1)
                weights.append(loss_weight)

            elif "cryoesp" == loss_function_type:
                cryoesp_config = self.config.loss_function.cryoesp_loss_function
                # TODO: Check with vova if anything can be removed or not.
                loss_function = CryoEM_ESP_GuidanceLossFunction(
                        cryoesp_config.reference_pdb, self.model_manager.resolved_pdb_to_full_mask, cryoesp_config.esp_file, 
                        emdb_resolution=cryoesp_config.emdb_resolution, device=self.device, is_assembled=(not cryoesp_config.is_assembled),
                        global_b_factor=cryoesp_config.global_b_factor, esp_gt_cutoff_value=cryoesp_config.esp_gt_cutoff_value,
                        reduced_D = cryoesp_config.reduced_D, use_Coloumb=cryoesp_config.use_Coloumb,
                        regions_of_interest = [
                            list(range(single_res_range[0],single_res_range[1]+1)) 
                            for single_res_range in self.config.protein.residue_range ],
                        sequences_dictionary=self.model_manager.sequences_dictionary, guide_only_ROI= cryoesp_config.guide_only_ROI, 
                        save_folder=None, # Will be set later in run() method with the new clean folder structure
                        aling_only_outside_ROI= cryoesp_config.aling_only_outside_ROI, 
                        should_add_b_factor_for_resolution_cutoff= getattr(cryoesp_config, "should_add_b_factor_for_resolution_cutoff", False),
                        optimize_b_factors= cryoesp_config.optimize_bfactor, should_align_to_chains=self.config.protein.should_align_to_chains,
                        chains_to_read=self.config.protein.chains_to_use,
                        to_convex_hull_of_ROI=self.config.protein.should_fill_mask, reapply_b_factor=cryoesp_config.reapply_b_factor, 
                        reapply_is_learnable=cryoesp_config.reapply_is_learnable,
                        sinkhorn_parameters={
                            "percentage": cryoesp_config.sinkhorn.percentage,
                            "p": cryoesp_config.sinkhorn.p,
                            "blur": cryoesp_config.sinkhorn.blur,
                            "reach": cryoesp_config.sinkhorn.reach,
                            "scaling": cryoesp_config.sinkhorn.scaling,
                            "turn_off_after": cryoesp_config.sinkhorn.turn_off_after,
                            "backend": cryoesp_config.sinkhorn.backend,
                            "debug_with_rmsd": cryoesp_config.sinkhorn.debug_with_rmsd, # whether to debug with rmsd or not. if True, then we compute the rmsd loss and use it for debugging
                            "guide_multimer_by_chains": cryoesp_config.sinkhorn.guide_multimer_by_chains, # whether to guide the multimer by chains or not
                            "debias": cryoesp_config.sinkhorn.debias, # whether to debias the sinkhorn loss or not
                        },
                        combinatorially_best_alignment=cryoesp_config.combinatorially_best_alignment,
                        alignment_strategy=cryoesp_config.alignment_strategy, # "global_density", "global_rmsd_to_gt", "cost_matrix_hungarian", "combinatorics", False
                        rmax_for_esp= cryoesp_config.rmax_for_esp, rmax_for_mask= cryoesp_config.rmax_for_mask, rmax_for_final_bfac_fitting=cryoesp_config.rmax_for_final_bfac_fitting, rmax_for_backbone=cryoesp_config.rmax_for_backbone,
                        reordering_every=cryoesp_config.reordering_every, # how often to reorder the multimer alignment
                        dihedrals_parameters={
                        "use_dihedrals": cryoesp_config.dihedrals.use_dihedrals, # whether to use dihedrals from the ground truth or not. if False, then we don't use dihedrals
                        "dihedral_loss_weight": cryoesp_config.dihedrals.dihedral_loss_weight, # weight of the dihedral loss
                        "dihedrals_file": cryoesp_config.dihedrals.dihedrals_file, # file with dihedrals to use
                        }, # dihedrals parameters
                        symmetry_parameters={
                            "symmetry_type": cryoesp_config.symmetry.symmetry_type, 
                            "reapply_symmetry_every": cryoesp_config.symmetry.reapply_symmetry_every, # how often to reapply the symmetry
                        },
                        gradient_ascent_parameters={
                            "steps": cryoesp_config.gradient_ascent_parameters.steps,
                            "lr_t_A": cryoesp_config.gradient_ascent_parameters.lr_t_A,
                            "lr_r_deg": cryoesp_config.gradient_ascent_parameters.lr_r_deg,
                            "reduction": cryoesp_config.gradient_ascent_parameters.reduction,
                            "per_step_t_cap_voxels": cryoesp_config.gradient_ascent_parameters.per_step_t_cap_voxels,
                            "Bfac": cryoesp_config.gradient_ascent_parameters.Bfac,
                            "bfactor_minimum": cryoesp_config.gradient_ascent_parameters.bfactor_minimum,
                            "n_random": cryoesp_config.gradient_ascent_parameters.n_random,
                            "t_init_box_edge_voxels": cryoesp_config.gradient_ascent_parameters.t_init_box_edge_voxels,
                        },
                        evaluate_only_resolved=getattr(cryoesp_config, "evaluate_only_resolved", False),
                    )
                self.cryoesp_loss_function = loss_function
                loss_functions.append(loss_function)
                loss_weight = getattr(cryoesp_config, "weight", 1)
                weights.append(loss_weight)
            else:
                raise ValueError(f"The loss function {loss_function_type} is not supported!")

        if self.config.loss_function.violation_loss_weight > 0:
            loss_functions.append(ViolationLossFunction(self.model_manager.atom_array))
            weights.append(self.config.loss_function.violation_loss_weight)
        if self.config.loss_function.bond_length_loss_weight > 0 or self.config.loss_function.cryoesp_loss_function.log_bond_length_loss:
            loss_functions.append(BondLengthLossFunction(self.model_manager.atom_array, self.device))
            weights.append(self.config.loss_function.bond_length_loss_weight)

        return MultiLossFunction(
            loss_functions, weights, 
            normalize_losses_by_main_loss=getattr(self.config.loss_function, "normalize_losses_by_main_loss", False)
        )

    def query_msa_server(self):
        msa_save_dir = os.path.join(self.config.model_manager.msa_save_dir, self.config.protein.pdb_id)
        # Check if the assembly identifier is provided (cryo specific)
        if self.config.protein.assembly_identifier is not None:
            msa_save_dir = os.path.join(msa_save_dir, self.config.protein.assembly_identifier)

        # Log the msa save dir
        self.msa_full_save_dir = msa_save_dir
        query_msa_server(self.msa_full_save_dir, self.config.protein.sequences)

    def save_state(self, structures, name, folder_path):
        os.makedirs(folder_path, exist_ok=True)

        # Density specific logging
        if "density" in self.config.loss_function.loss_function_type:
            structures = self.model_manager.align_models_to_reference(structures.detach())
            for i in range(structures.shape[0]):
                self.model_manager.save_structure_pdb(structures[i].cpu(), self.model_manager.atom_array, f"{folder_path}/{name}_{i}.pdb")

        # CryoESP specific logging
        elif "cryoesp" in self.config.loss_function.loss_function_type:
            esp_loss_function_obj = self.cryoesp_loss_function
            structures, _, _ = esp_loss_function_obj.align_structure(
                structures.detach(), esp_loss_function_obj.coordinates_gt.unsqueeze(0), 
                i=self.config.model_manager.diffusion_N-1, is_counted_down=False
            )
            for i in range(structures.shape[0]):
                save_structure_full(
                    structures[i].cpu(), self.model_manager.full_sequences, self.model_manager.atom_array, f"{folder_path}/{name}_{i}.pdb",
                    bfactors=esp_loss_function_obj.bfactor_gt
                )
        elif "nmr" in self.config.loss_function.loss_function_type:
            # Saving pdbs 
            for i in range(structures.shape[0]):
                save_structure_full(
                    structures[i].cpu(), self.model_manager.full_sequences, self.model_manager.atom_array, f"{folder_path}/{name}_{i}.pdb",
                    bfactors= None
                )
        else:
            raise ValueError(f"The loss function type {self.config.loss_function.loss_function_type} is not a valid option")

    def get_residue_mask(self):
        residue_range = self.config.protein.residue_range[0]
        residue_mask = [residue_range[0] <= i <= residue_range[1] for i in range(len(self.config.protein.sequence))]
        residue_mask = torch.tensor(residue_mask)
        return residue_mask

    def get_initial_latents(self):
        return self.model_manager.get_x_noisy(self.model_manager.get_x_start(self.config.general.batch_size))

    def run_full_diffusion_process(self, latents):
        structures = latents.clone()

        N = self.config.model_manager.diffusion_N
        if self.config.general.recycle_structures.should_recycle:
            A = self.config.general.recycle_structures.recycle_connection[0]
            B = self.config.general.recycle_structures.recycle_connection[1]
            R = self.config.general.recycle_structures.recycle_n_times
            
            total_steps = B + (R + 1) * (A - B) + (N - A) - 1
            steps_generator = tqdm(range(total_steps), "running diffusion process")
            schedule = list(range(0,B)) + (R+1)*list(range(B,A)) + list(range(A,N))
        else:
            steps_generator = tqdm(range(self.config.model_manager.diffusion_N), "running diffusion process")
            schedule = list(range(N))


        structures_gradient_norm = self.config.diffusion_process.guidance.guidance_direction_scale_factor
        normalize_gradients = self.config.diffusion_process.guidance.normalize_gradients
        guidance_direction, wandb_log = None, None
        start_guidance_from = self.config.general.denoiser_time_index
        for (step, i) in zip(steps_generator, schedule):
            if self.config.general.apply_diffusion_guidance and i > start_guidance_from:      
                structures.requires_grad = True

            x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy(structures, start_index=i, inplace_safe=False)
            guidance_direction = None

            start_idx = schedule[step]
            end_idx = schedule[step+1] if step + 1 < len(schedule) else None

            if self.config.general.apply_diffusion_guidance and i > start_guidance_from:
                wandb_log = {}
                if i != (self.config.model_manager.diffusion_N - 1):
                    loss_value, losses, new_x_0_hat = self.loss_function(x_0_hat, i / (self.config.model_manager.diffusion_N - 1), structures=structures, i=i, step=step)
                    wandb_log = self.loss_function.wandb_log(x_0_hat)
                    if new_x_0_hat is not None:
                        wandb_log = self.loss_function.wandb_log(new_x_0_hat)
                        x_0_hat = new_x_0_hat
                    loss_value.backward()
                    steps_generator.set_description(f"running diffusion process, loss: {loss_value.item():.5f}")                
                
                    with torch.no_grad():
                        guidance_direction = structures.grad if i > start_guidance_from else None
                        structures.grad = None

            structures = self.model_manager.get_x_t_from_x_0_hat(
                structures, x_0_hat, start_idx, end_idx, 
                guidance_direction=guidance_direction, step_size=self.config.diffusion_process.guidance.step_size, 
                normalize_gradients=normalize_gradients, structures_gradient_norm=structures_gradient_norm, guidance_scale_gradually_increase=self.config.diffusion_process.guidance.guidance_scale_gradually_increase,
            )
            if self.loss_function is not None:
                self.loss_function.post_optimization_step()
            if i < self.config.model_manager.diffusion_N - 1:
                # structures = self.model_manager.get_x_noisy(structures, i + 1)
                structures = self.model_manager.get_x_noisy(structures, start_index=start_idx, end_index=end_idx)
            structures = structures.detach().clone()
            if wandb_log is not None and self.config.wandb.login_key is not None:
                wandb.log(wandb_log)
        return structures

    def run(self):
        latents = self.get_initial_latents()

        structures = self.run_full_diffusion_process(latents)
        sub_folder_name = "diffusion_process"
        self.save_state(structures, self.config.protein.pdb_id[:4], os.path.join(self.experiment_save_dir, sub_folder_name))

        # Density specific!
        if self.density_loss_function is not None:
            metadata = {
                "pdb_id": self.config.protein.pdb_id[:4],
                "residue_range": self.config.protein.residue_range[0],
                "sequence": self.config.protein.sequences[0]["sequence"],
                "chain": self.config.protein.reference_raw_pdb_chain,
                "pdb_residue_range": self.config.protein.pdb_residue_range
            }
            with open(os.path.join(self.experiment_save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)
        elif self.cryoesp_loss_function is not None:
            metadata = {
                "pdb_id": self.config.protein.pdb_id[:4],
                "sequences": self.config.protein.sequences,
            }
            with open(os.path.join(self.experiment_save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

        elif self.nmr_loss_function is not None:
            metadata = {
                "pdb_id": self.config.protein.pdb_id[:4],
                "sequences": self.config.protein.sequences,
            }
            with open(os.path.join(self.experiment_save_dir, "metadata.json"), "w") as f:
                json.dump(metadata, f)

    def get_un_broken_samples(self, samples_directory, residue_range, bond_max_threshold):
        """
            this function will go over the files in self.samples_directory and return the files of the proteins which are not borken
        """
        pdb_files = [os.path.join(samples_directory, file) for file in os.listdir(samples_directory) if file.endswith(".pdb")]
        un_broken_files = []
        for pdb_file in pdb_files:
            structure = gemmi.read_pdb(pdb_file)
            model = structure[0]
            chain = model[0]
            bonded_atom_pairs = find_bonded_pairs(chain, residue_range)
            bond_distances = np.array([np.linalg.norm(np.array(list(pair[0][0].pos)) - np.array(list(pair[1][0].pos))) for pair in bonded_atom_pairs])
            if bond_distances.max() < bond_max_threshold:
                un_broken_files.append(pdb_file)
        return un_broken_files

    def relax_files(self, files):
        relaxed_file_names = []
        for file in files:
            folder_name = os.path.dirname(file)
            basename = os.path.basename(file)
            relaxed_file_name = os.path.join(folder_name, "relaxed", basename)
            os.makedirs(os.path.dirname(relaxed_file_name), exist_ok=True)
            relax_pdb(file, relaxed_file_name, use_gpu=False)
            delete_hydrogens(relaxed_file_name)
            relaxed_file_names.append(relaxed_file_name)
        return relaxed_file_names

    def align_relaxed_samples(self, files, reference_pdb_path, residue_range, device):
        aligned_files = []
        with torch.no_grad():
            structures = torch.cat([load_pdb_atom_locations(file, device) for file in files])
            coordinates_pdb, _, _, _, _ = get_sampler_pdb_inputs(reference_pdb_path, residue_range, device)

            ref_pose = coordinates_pdb.repeat(structures.shape[0], 1, 1)
            mask = torch.ones_like(ref_pose[0, :, 0])
            _, structures, _, _ = self_aligned_rmsd(structures, ref_pose, mask)

            for i in range(len(files)):
                structure_i, file_i = structures[i], files[i]
                aligned_file_i = write_back_pdb_coordinates(file_i, file_i, structure_i[None])
                aligned_files.append(aligned_file_i)

        return aligned_files

    def relax_structures(self, bond_max_threshold):
        un_broken_files = self.get_un_broken_samples(os.path.join(self.experiment_save_dir, "diffusion_process"), self.config.protein.residue_range[0], bond_max_threshold)
        relaxed_files = self.relax_files(un_broken_files)
        aligned_files = self.align_relaxed_samples(relaxed_files, self.config.protein.reference_pdb, self.config.protein.residue_range[0], self.device)
        return aligned_files
