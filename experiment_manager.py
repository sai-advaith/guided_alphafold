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
from src.utils.io import (
    load_pdb_atom_locations,
    get_sampler_pdb_inputs,
    delete_hydrogens,
    write_back_pdb_coordinates,
    load_config,
    namespace_to_dict,
    query_msa_server,
    find_starting_zero_indeces_of_alignment,
    merge_multiple_structures,
    alignment_mask_by_chain,
)
from src.utils.non_diffusion_model_manager import ProtenixModelManager
from src.utils.non_diffusion_model_manager import save_structure_full
from src.utils.phenix_manager import PhenixManager
from src.losses import * 

class ExperimentManager:
    def __init__(self, config, device, config_file_path=None):
        # general
        self.device = device
        self.config_file_path = config_file_path
        self.config = config
        self.name = f"{self.config.general.name}_{self.config.protein.pdb_id}"

        # Editing the sequences to trim the unresolved in the pdb if required. Should happen before the MSA server is queried.
        if getattr(self.config.protein, "trim_unresolved", False):
            self.config.protein.sequences = ProtenixModelManager._trim_unresolved_sequences(
                self.config.protein.sequences, self.config.protein.reference_pdb, getattr(self.config.protein, "chains_to_use", None),
            ) 

        self.msa_full_save_dir = None
        self.query_msa_server()
        self.model_manager = self._get_model_manager()
        self._setup_wandb()
        self.loss_function = self._get_loss_function()
        self.phenix_manager_path = None  # Will be set later from command line args
        
        # Create per-atom normalization constants tensor if per-chain constants are specified
        # Must be after model_manager is created (needs atom_array)
        self.per_atom_normalization_constants = self._create_per_atom_normalization_constants()


        if self.config.protein.assembly_identifier is not None:
            self.experiment_save_dir = os.path.join(self.config.general.output_folder, self.config.protein.pdb_id, self.config.protein.assembly_identifier)
        else:
            self.experiment_save_dir = os.path.join(self.config.general.output_folder, self.config.general.name)

        os.makedirs(self.experiment_save_dir, exist_ok=True)      
        
        # Set save_folder in loss function if it's a CryoESP loss function
        if self.cryoesp_loss_function is not None:
            self.cryoesp_loss_function.save_folder = self.experiment_save_dir      

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
            use_lma=getattr(self.config.model_manager, "use_lma", False),
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
            should_concatenate_frozen_atoms=getattr(self.config.protein, "should_concatenate_frozen_atoms", False),
            rmax_for_mask=self.config.loss_function.cryoesp_loss_function.rmax_for_mask, # NOTE: we can move these from the cryoesp loss function to some general config.
            enable_memory_snapshot=getattr(self.config.model_manager, "enable_memory_snapshot", False),
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
                        use_correlation_esp_loss=getattr(cryoesp_config, "use_correlation_esp_loss", False),
                        emdb_resolution=cryoesp_config.emdb_resolution, device=self.device, is_assembled=(not cryoesp_config.is_assembled),
                        global_b_factor=cryoesp_config.global_b_factor, esp_gt_cutoff_value=cryoesp_config.esp_gt_cutoff_value,
                        reduced_D = cryoesp_config.reduced_D, use_Coloumb=cryoesp_config.use_Coloumb,
                        ensemble_size=self.config.general.batch_size,  # B-factors are per ensemble member
                        regions_of_interest = [
                            list(range(single_res_range[0],single_res_range[1]+1)) if len(single_res_range) > 0 else []
                            for single_res_range in self.config.protein.residue_range ],
                        sequences_dictionary=self.model_manager.sequences_dictionary, guide_only_ROI= cryoesp_config.guide_only_ROI, 
                        save_folder=None, # Will be set later in run() method with the new clean folder structure
                        aling_only_outside_ROI= cryoesp_config.aling_only_outside_ROI, 
                        should_add_b_factor_for_resolution_cutoff= getattr(cryoesp_config, "should_add_b_factor_for_resolution_cutoff", False),
                        optimize_b_factors= cryoesp_config.optimize_bfactor, 
                        optimize_occupancies=getattr(cryoesp_config, "optimize_occupancies", False),
                        should_align_to_chains=self.config.protein.should_align_to_chains,
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
                            # ESP SE3 align ensemble parameters
                            "D_reduced": getattr(cryoesp_config.gradient_ascent_parameters, "D_reduced", None),
                            "volume_resolution_A": getattr(cryoesp_config.gradient_ascent_parameters, "volume_resolution_A", None),
                            "print_every": getattr(cryoesp_config.gradient_ascent_parameters, "print_every", 10),
                            "max_volumes_per_batch": getattr(cryoesp_config.gradient_ascent_parameters, "max_volumes_per_batch", 4),
                            "use_checkpointing": getattr(cryoesp_config.gradient_ascent_parameters, "use_checkpointing", False),
                            "pruning_iteration": getattr(cryoesp_config.gradient_ascent_parameters, "pruning_iteration", 4),
                            "n_keep_after_pruning": getattr(cryoesp_config.gradient_ascent_parameters, "n_keep_after_pruning", 3),
                            "second_pruning_iteration": getattr(cryoesp_config.gradient_ascent_parameters, "second_pruning_iteration", None),
                            "min_cc_for_convergence": getattr(cryoesp_config.gradient_ascent_parameters, "min_cc_for_convergence", 0.5),
                            "use_autocast": getattr(cryoesp_config.gradient_ascent_parameters, "use_autocast", False),
                            "min_cc_threshold": getattr(cryoesp_config.gradient_ascent_parameters, "min_cc_threshold", 0.15),
                            "max_reinit_attempts": getattr(cryoesp_config.gradient_ascent_parameters, "max_reinit_attempts", 3),
                            "overshoot_recovery_drop": getattr(cryoesp_config.gradient_ascent_parameters, "overshoot_recovery_drop", 0.02),
                            "use_so3_grid": getattr(cryoesp_config.gradient_ascent_parameters, "use_so3_grid", True),
                            "so3_grid_resolution": getattr(cryoesp_config.gradient_ascent_parameters, "so3_grid_resolution", None),
                            "use_pca_init": getattr(cryoesp_config.gradient_ascent_parameters, "use_pca_init", False),
                            "optimizer": getattr(cryoesp_config.gradient_ascent_parameters, "optimizer", "adam"),
                            "adam_betas": getattr(cryoesp_config.gradient_ascent_parameters, "adam_betas", (0.9, 0.999)),
                            "use_ema": getattr(cryoesp_config.gradient_ascent_parameters, "use_ema", False),
                            "ema_decay": getattr(cryoesp_config.gradient_ascent_parameters, "ema_decay", 0.95),
                            "use_lr_decay": getattr(cryoesp_config.gradient_ascent_parameters, "use_lr_decay", True),
                            # ESP-specific parameters (esp_* prefixed)
                            "esp_lr_t_A": getattr(cryoesp_config.gradient_ascent_parameters, "esp_lr_t_A", 1.0),
                            "esp_lr_r_deg": getattr(cryoesp_config.gradient_ascent_parameters, "esp_lr_r_deg", 1.0),
                            "esp_n_random": getattr(cryoesp_config.gradient_ascent_parameters, "esp_n_random", 4649),
                            "esp_t_init_box_edge_voxels": getattr(cryoesp_config.gradient_ascent_parameters, "esp_t_init_box_edge_voxels", 0.001),
                            # LR decay parameters
                            "lr_decay_factor": getattr(cryoesp_config.gradient_ascent_parameters, "lr_decay_factor", 0.9),
                            "lr_plateau_threshold": getattr(cryoesp_config.gradient_ascent_parameters, "lr_plateau_threshold", 5),
                            "lr_plateau_threshold_high_cc": getattr(cryoesp_config.gradient_ascent_parameters, "lr_plateau_threshold_high_cc", 10),
                            "lr_plateau_min_cc": getattr(cryoesp_config.gradient_ascent_parameters, "lr_plateau_min_cc", 0.3),
                            "lr_decay_warmup_steps": getattr(cryoesp_config.gradient_ascent_parameters, "lr_decay_warmup_steps", 10),
                            "lr_decay_cc_threshold": getattr(cryoesp_config.gradient_ascent_parameters, "lr_decay_cc_threshold", 0.5),
                            "lr_decay_cc_cooldown": getattr(cryoesp_config.gradient_ascent_parameters, "lr_decay_cc_cooldown", 12),
                            # Adaptive reinit parameters
                            "adaptive_reinit": getattr(cryoesp_config.gradient_ascent_parameters, "adaptive_reinit", False),
                            "adaptive_reinit_iterations": getattr(cryoesp_config.gradient_ascent_parameters, "adaptive_reinit_iterations", None),
                            "adaptive_reinit_fraction": getattr(cryoesp_config.gradient_ascent_parameters, "adaptive_reinit_fraction", 0.1),
                            "adaptive_reinit_cc_threshold": getattr(cryoesp_config.gradient_ascent_parameters, "adaptive_reinit_cc_threshold", None),
                            # Other parameters
                            "rmsd_regularization_weight": getattr(cryoesp_config.gradient_ascent_parameters, "rmsd_regularization_weight", 0.0),
                            "verbose": getattr(cryoesp_config.gradient_ascent_parameters, "verbose", False),
                        },
                        evaluate_only_resolved=getattr(cryoesp_config, "evaluate_only_resolved", False),
                        frozen_atoms_dict=getattr(self.model_manager, "frozen_atoms_dict", None),
                        save_aligned=getattr(cryoesp_config, "save_aligned", False),
                        integrate_gaussians_over_voxel=getattr(cryoesp_config, "integrate_gaussians_over_voxel", True),
                        guide_specific_chain=getattr(cryoesp_config, "guide_specific_chain", False),
                        cryoesp_chain_indices=getattr(cryoesp_config, "cryoesp_chain_indices", None),
                        cryoesp_residue_range_pdb=getattr(cryoesp_config, "cryoesp_residue_range_pdb", None),
                        loss_normalization_rmsd_phase=getattr(cryoesp_config, "loss_normalization_rmsd_phase", 1.0),
                        loss_normalization_cc_phase=getattr(cryoesp_config, "loss_normalization_cc_phase", 1.0),
                        use_old_esp_calculation=getattr(cryoesp_config, "use_old_esp_calculation", False),
                        per_chain_b_factors=getattr(cryoesp_config, "per_chain_b_factors", None),
                        chain_blurred_esp_loss_config=getattr(cryoesp_config, "chain_blurred_esp_loss", None),
                        esp_base_weight=getattr(cryoesp_config, "esp_base_weight", 1.0),  # Weight for base ESP loss (affects final loss but not normalization)
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
        
        has_frozen_atoms = getattr(self.config.protein, "should_concatenate_frozen_atoms", False)
        bond_length_loss_weight = getattr(self.config.loss_function, "bond_length_loss_weight", 0.0)
        if "cryoesp" in self.config.loss_function.loss_function_type:
            loss_functions.append(BondLengthLossFunction(
                self.model_manager.atom_array if not has_frozen_atoms \
                    else self.model_manager.atom_array + self.model_manager.frozen_atoms_dict["insertable_array"],
                self.device
            ))
            weights.append(bond_length_loss_weight)
        
        # Add RMSD loss function if configured
        if "cryoesp" in self.config.loss_function.loss_function_type:
            cryoesp_config = self.config.loss_function.cryoesp_loss_function
            rmsd_loss_config = getattr(self.config.loss_function, "rmsd_loss_function", None)
            if rmsd_loss_config is not None:
                rmsd_loss_sequence_indices = getattr(rmsd_loss_config, "rmsd_loss_sequence_indices", None)
                rmsd_loss_weight = getattr(rmsd_loss_config, "weight", 1.0)
                if rmsd_loss_sequence_indices is not None and len(rmsd_loss_sequence_indices) > 0:
                    loss_functions.append(RMSDLossFunction(
                        reference_pdb=cryoesp_config.reference_pdb,
                        mask=self.model_manager.resolved_pdb_to_full_mask,
                        sequences_dictionary=self.model_manager.sequences_dictionary,
                        chains_to_read=self.config.protein.chains_to_use,
                        rmsd_loss_sequence_indices=rmsd_loss_sequence_indices,
                        device=self.device,
                        should_align_to_chains=self.config.protein.should_align_to_chains,
                        frozen_atoms_dict=getattr(self.model_manager, "frozen_atoms_dict", None),
                    ))
                    weights.append(rmsd_loss_weight)

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

    def _get_chain_names_from_sequences_dict(self, sequences_dictionary):
        """
        Extract chain names from sequences_dictionary based on maps_to.
        Returns a list of chain names matching the order of full_sequences.
        
        Args:
            sequences_dictionary: Dictionary containing sequence info with 'maps_to' and 'count' keys
            
        Returns:
            List of chain names (or None if sequences_dictionary is not available)
        """
        if sequences_dictionary is None:
            return None
        
        chain_names = []
        for seq_dict in sequences_dictionary:
            maps_to = seq_dict.get("maps_to", [])
            count = seq_dict.get("count", 1)
            for copy_idx in range(count):
                if maps_to and len(maps_to) > 0:
                    chain_idx = copy_idx % len(maps_to)
                    chain_names.append(maps_to[chain_idx])
                else:
                    chain_names.append(None)
        return chain_names

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
            saved_gemmi_structures = []
            # Use the final step number (last iteration) to determine which alignment strategy to use
            # This ensures we use the density alignment strategy if we've passed turn_off_after
            final_step = self.config.model_manager.diffusion_N - 1
            structures, _, _ = esp_loss_function_obj.align_structure(
                structures.detach(), esp_loss_function_obj.coordinates_gt.unsqueeze(0), 
                i=final_step, step=final_step, is_counted_down=False
            )
            chain_names = self._get_chain_names_from_sequences_dict(
                getattr(esp_loss_function_obj, 'sequences_dictionary', None)
            )
            
            # Use stored starting residue indices (computed during initialization) to preserve original PDB numbering
            starting_residue_indices = getattr(self.model_manager, 'starting_residue_indices', None)
            
            for i in range(structures.shape[0]):
                # Select b-factors for this structure (or use first ensemble member if only one set)
                bfactors_to_use = esp_loss_function_obj.bfactor_gt[i] if esp_loss_function_obj.bfactor_gt.shape[0] > i else esp_loss_function_obj.bfactor_gt[0]
                gemmi_structure = save_structure_full(
                    structures[i].cpu(), self.model_manager.full_sequences, self.model_manager.sequence_types, self.model_manager.atom_array, f"{folder_path}/{name}_{i}.pdb",
                    bfactors=bfactors_to_use,
                    atom_mask=self.model_manager.resolved_pdb_to_full_mask.cpu(),  # Only save resolved atoms, filter out unresolved ones
                    chain_names=chain_names,
                    starting_residue_indices=starting_residue_indices  # Preserve original PDB residue numbering
                )
                saved_gemmi_structures.append(gemmi_structure)
            # Save merged ensemble with altlocs/occupancies for downstream evaluation
            if len(saved_gemmi_structures) > 0:
                try:
                    occupancy_list = None
                    if getattr(esp_loss_function_obj, "occupancy_gt", None) is not None:
                        occupancy_list = [
                            float(x) for x in esp_loss_function_obj.occupancy_gt.detach().cpu().tolist()
                        ]
                    merged_structure = merge_multiple_structures(saved_gemmi_structures, occupancies=occupancy_list)
                    merged_structure.write_pdb(os.path.join(folder_path, "ensemble.pdb"))
                except Exception as e:
                    print(f"Failed to create ensemble.pdb: {e}")
            # For now, only save for the CryoESP loss function.
            # Convert phenix_manager_path to PhenixManager object if path is provided
            phenix_manager_obj = None
            if self.phenix_manager_path is not None:
                phenix_manager_obj = PhenixManager(self.phenix_manager_path)
            # Call save_state on the cryoesp loss function directly (not on MultiLossFunction)
            cryoesp_config = self.config.loss_function.cryoesp_loss_function
            # Get B-factor fitting parameters from config (with defaults)
            bfactor_fitting_config = getattr(cryoesp_config, "bfactor_fitting", None)
            if bfactor_fitting_config is not None:
                b_factor_lr = getattr(bfactor_fitting_config, "b_factor_lr", 2.0)
                n_iterations = getattr(bfactor_fitting_config, "n_iterations", 500)
                bfactor_min = getattr(bfactor_fitting_config, "bfactor_min", 10.0)
                bfactor_max = getattr(bfactor_fitting_config, "bfactor_max", 800.0)
                use_cross_correlation = getattr(bfactor_fitting_config, "use_cross_correlation", True)
                bfactor_regularization = getattr(bfactor_fitting_config, "bfactor_regularization", 0.0)
                use_zero_b_values = getattr(bfactor_fitting_config, "use_zero_b_values", False)
                should_always_fit_gt = getattr(bfactor_fitting_config, "should_always_fit_gt", False)
            else:
                # Fallback to defaults if bfactor_fitting section is missing
                b_factor_lr = 2.0
                n_iterations = 500
                bfactor_min = 10.0
                bfactor_max = 800.0
                use_cross_correlation = True
                bfactor_regularization = 0.0
                use_zero_b_values = False
                should_always_fit_gt = False
            
            esp_loss_function_obj.save_state(
                structures,
                folder_path,
                phenix_manager_obj,
                skip_png=True,
                b_factor_lr=b_factor_lr,
                use_zero_b_values=use_zero_b_values,
                should_always_fit_gt=should_always_fit_gt,
                n_iterations=n_iterations,
                bfactor_min=bfactor_min,
                bfactor_max=bfactor_max,
                use_cross_correlation=use_cross_correlation,
                bfactor_regularization=bfactor_regularization,
                gt_bfactor_mode=getattr(cryoesp_config, "gt_bfactor_mode", "leave_pdb"),
            )
        
        elif "nmr" in self.config.loss_function.loss_function_type:
            # Saving pdbs 
            chain_names = self._get_chain_names_from_sequences_dict(
                getattr(self.model_manager, 'sequences_dictionary', None)
            )
            
            # Use stored starting residue indices (computed during initialization) to preserve original PDB numbering
            starting_residue_indices = getattr(self.model_manager, 'starting_residue_indices', None)
            
            for i in range(structures.shape[0]):
                save_structure_full(
                    structures[i].cpu(), self.model_manager.full_sequences, self.model_manager.sequence_types, self.model_manager.atom_array, f"{folder_path}/{name}_{i}.pdb",
                    bfactors=None,
                    atom_mask=self.model_manager.resolved_pdb_to_full_mask.cpu(),  # Only save resolved atoms, filter out unresolved ones
                    chain_names=chain_names,
                    starting_residue_indices=starting_residue_indices  # Preserve original PDB residue numbering
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

    def _create_per_atom_normalization_constants(self):
        """
        Create per-chain constant masks and values for later use.
        Returns None if per-chain constants are not specified (default behavior).
        
        Returns:
            dict | None: Dictionary with 'chain_masks' (list of masks) and 'chain_constants' (dict of chain_idx -> constant),
                        or None for default behavior
        """
        guidance_config = getattr(self.config.diffusion_process, 'guidance', None)
        if guidance_config is None:
            return None
        
        # Check if per-chain normalization constants are specified
        per_chain_constants = getattr(guidance_config, 'per_chain_normalization_constants', None)
        if per_chain_constants is None:
            return None
        
        # Convert SimpleNamespace to dict if needed
        if hasattr(per_chain_constants, '__dict__'):
            per_chain_constants = vars(per_chain_constants)
        elif not isinstance(per_chain_constants, dict):
            # If it's neither dict nor namespace, try to convert
            per_chain_constants = dict(per_chain_constants) if hasattr(per_chain_constants, '__iter__') else {}
        
        # Build full_sequences and sequence_types (same as in model_manager)
        full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in self.config.protein.sequences]
        full_sequences = [item for sublist in full_sequences for item in sublist]
        sequence_types = [
            sequence_type
            for dictionary in self.config.protein.sequences
            for sequence_type in [dictionary.get("sequence_type", "proteinChain")] * dictionary["count"]
        ]
        
        # Get number of atoms from model_manager
        n_atoms = self.model_manager.atom_array.shape[0]
        
        # Create masks for each chain with per-chain constants
        chain_masks = {}
        chain_constants_dict = {}
        
        for chain_idx, constant_value in per_chain_constants.items():
            if isinstance(chain_idx, str):
                # Convert chain name to sequence index if needed (e.g., "1" -> 1)
                chain_idx = int(chain_idx)
            
            if not (0 <= chain_idx < len(full_sequences)):
                raise ValueError(
                    f"Chain index {chain_idx} is out of range. "
                    f"Valid chain indices are 0 to {len(full_sequences) - 1}."
                )
            
            # Create mask for this chain
            chain_mask = alignment_mask_by_chain(
                full_sequences,
                chains_to_align=[chain_idx],
                sequence_types=sequence_types
            ).to(self.device)
            
            chain_masks[chain_idx] = chain_mask
            chain_constants_dict[chain_idx] = constant_value
        
        return {
            'chain_masks': chain_masks,
            'chain_constants': chain_constants_dict,
            'n_atoms': n_atoms
        }

    def _get_step_based_scale_factor(self, guidance_config, step_idx):
        """
        Get the step-based scale factor (scalar) for a given step index.
        This is a helper function that extracts the logic for getting step-based factors.
        
        Args:
            guidance_config: Configuration object for guidance settings
            step_idx: Current step index
            
        Returns:
            float: Scale factor for the current step
            
        Raises:
            ValueError: If neither step-based factors nor single value are provided in config
        """
        # Check for step-based scale factors first
        if hasattr(guidance_config, 'guidance_direction_scale_factors') and guidance_config.guidance_direction_scale_factors is not None:
            # Step-based format: list of {scale_factor: float, step_range: [start, end]}
            for entry in guidance_config.guidance_direction_scale_factors:
                # Handle both dict and object access (config might be dict or object)
                if isinstance(entry, dict):
                    scale_factor = entry['scale_factor']
                    step_range = entry['step_range']  # [start, end] - end is exclusive
                else:
                    scale_factor = entry.scale_factor
                    step_range = entry.step_range  # [start, end] - end is exclusive
                
                # Check if step_idx falls within this range
                if step_range[0] <= step_idx < step_range[1]:
                    return scale_factor
        
        # Fallback to single value if step-based not found
        if hasattr(guidance_config, 'guidance_direction_scale_factor'):
            return guidance_config.guidance_direction_scale_factor
        
        # Ultimate fallback
        return 1.0
    
    def _create_per_atom_normalization_tensor(self, step_based_factor, step_idx, guidance_config):
        """
        Create per-atom normalization tensor with step-based factor for default chains
        and per-chain constants for specified chains (which can be fixed or step-dependent).
        
        Args:
            step_based_factor: Scalar factor to use for chains without per-chain constants
            step_idx: Current step index (needed for step-dependent per-chain constants)
            guidance_config: Configuration object (needed for step-dependent per-chain constants)
            
        Returns:
            torch.Tensor: Per-atom tensor of shape [N_atoms] with normalization constants
        """
        n_atoms = self.per_atom_normalization_constants['n_atoms']
        per_atom_tensor = torch.ones(n_atoms, device=self.device, dtype=torch.float32) * step_based_factor
        
        # Apply per-chain constants (these override the step-based factor for specified chains)
        chain_masks = self.per_atom_normalization_constants['chain_masks']
        chain_constants = self.per_atom_normalization_constants['chain_constants']
        for chain_idx, chain_mask in chain_masks.items():
            constant_value = chain_constants[chain_idx]
            
            # Check if this is a step-based schedule (list/dict) or a fixed value (float)
            if isinstance(constant_value, (list, dict)) or (hasattr(constant_value, '__iter__') and not isinstance(constant_value, str)):
                # Step-based schedule: compute value for current step
                if isinstance(constant_value, dict):
                    # Single dict entry: {scale_factor: float, step_range: [start, end]}
                    if 'scale_factor' in constant_value and 'step_range' in constant_value:
                        step_range = constant_value['step_range']
                        if step_range[0] <= step_idx < step_range[1]:
                            per_atom_tensor[chain_mask] = constant_value['scale_factor']
                        continue
                # List of dict entries: [{scale_factor: float, step_range: [start, end]}, ...]
                for entry in constant_value:
                    if isinstance(entry, dict):
                        if 'scale_factor' in entry and 'step_range' in entry:
                            step_range = entry['step_range']
                            if step_range[0] <= step_idx < step_range[1]:
                                per_atom_tensor[chain_mask] = entry['scale_factor']
                                break
                    else:
                        # Object with scale_factor and step_range attributes
                        if hasattr(entry, 'scale_factor') and hasattr(entry, 'step_range'):
                            step_range = entry.step_range
                            if step_range[0] <= step_idx < step_range[1]:
                                per_atom_tensor[chain_mask] = entry.scale_factor
                                break
            else:
                # Fixed value (float): apply directly
                per_atom_tensor[chain_mask] = float(constant_value)
        
        return per_atom_tensor

    def _get_guidance_scale_factor(self, guidance_config, step_idx):
        """
        Get the guidance scale factor for a given step index.
        Returns either a scalar (float) or a per-atom tensor depending on configuration.
        
        IMPORTANT: step_idx should be the CUMULATIVE iteration count (not the diffusion time step).
        This ensures that when recycling/rerunning happens, normalization constants continue
        to progress and never reset. For example, if step 160 is reached and the constant
        switches from 0.7 to 0.05, it will never go back to 0.7 even if recycling occurs.
        
        Args:
            guidance_config: Configuration object for guidance settings
            step_idx: Current cumulative step index (from the main diffusion loop, not the diffusion time step)
            
        Returns:
            float | torch.Tensor: Scale factor for the current step (scalar) or per-atom tensor if per-chain constants are specified
            
        Raises:
            ValueError: If neither step-based factors nor single value are provided in config
        """
        # If per-chain constants are specified, create per-atom tensor
        if self.per_atom_normalization_constants is not None:
            step_based_factor = self._get_step_based_scale_factor(guidance_config, step_idx)
            return self._create_per_atom_normalization_tensor(step_based_factor, step_idx, guidance_config)
        
        # Default behavior: no per-chain constants, return scalar
        return self._get_step_based_scale_factor(guidance_config, step_idx)

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

        guidance_config = self.config.diffusion_process.guidance
        normalize_gradients = self.config.diffusion_process.guidance.normalize_gradients
        guidance_direction, wandb_log = None, None
        start_guidance_from = self.config.general.denoiser_time_index
        
        for (step, i) in zip(steps_generator, schedule):
            should_apply_guidance = self.config.general.apply_diffusion_guidance and i > start_guidance_from
            
            if should_apply_guidance:      
                structures.requires_grad = True

            x_0_hat = self.model_manager.get_x_0_hat_from_x_noisy(structures, start_index=i, inplace_safe=False)
            guidance_direction = None

            start_idx = schedule[step]
            end_idx = schedule[step+1] if step + 1 < len(schedule) else None

            # Always compute loss and wandb_log for logging, even when guidance is off
            wandb_log = {}
            if i != (self.config.model_manager.diffusion_N - 1):
                x_0_hat = self.loss_function.pre_optimization_step(x_0_hat, i=i, step=step) # Perform pre-optimization (adding concatenated bits.).

                if should_apply_guidance:
                    # With gradients for backprop
                    loss_value, losses, new_x_0_hat = self.loss_function(x_0_hat, i / (self.config.model_manager.diffusion_N - 1), structures=structures, i=i, step=step)
                    wandb_log = self.loss_function.wandb_log(x_0_hat)
                    if new_x_0_hat is not None:
                        wandb_log = self.loss_function.wandb_log(new_x_0_hat)
                        x_0_hat = new_x_0_hat # TODO Vova: rethink the x_0_hat logic since now this change is pre-baked in the preoptimization step! 
                    loss_value.backward()
                    steps_generator.set_description(f"running diffusion process, loss: {loss_value.item():.5f}")                
                
                    with torch.no_grad():
                        guidance_direction = structures.grad if i > start_guidance_from else None
                        structures.grad = None
                else:
                    # Without gradients, just for logging
                    with torch.no_grad():
                        loss_value, losses, new_x_0_hat = self.loss_function(x_0_hat, i / (self.config.model_manager.diffusion_N - 1), structures=structures, i=i, step=step)
                        wandb_log = self.loss_function.wandb_log(x_0_hat)
                        if new_x_0_hat is not None:
                            wandb_log = self.loss_function.wandb_log(new_x_0_hat)
                            x_0_hat = new_x_0_hat
                        steps_generator.set_description(f"running diffusion process, loss: {loss_value.item():.5f}")                
                
                x_0_hat = self.loss_function.post_optimization_step(x_0_hat) 
                # Perform post-optimization. I.e., any concatenated bits etc. are removed for the x_t to be calculated properly. 

            # Get scale factor for current step
            # NOTE: Using 'step' (cumulative iteration count) not 'i' (diffusion time step)
            # This ensures normalization constants progress correctly across recycling cycles
            structures_gradient_norm = self._get_guidance_scale_factor(guidance_config, step)

            structures = self.model_manager.get_x_t_from_x_0_hat(
                structures, x_0_hat, start_idx, end_idx, 
                guidance_direction=guidance_direction, step_size=self.config.diffusion_process.guidance.step_size, 
                normalize_gradients=normalize_gradients, structures_gradient_norm=structures_gradient_norm, guidance_scale_gradually_increase=self.config.diffusion_process.guidance.guidance_scale_gradually_increase,
            )
                
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

    @staticmethod
    def seed_experiment(seed):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)

        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False


# QoL change. Never to be pushed. 

def main(args):
    # For this branch, only Cryo stuff is assumed to happen.
    file_path = args.configuration_file
    device = args.device
    config = load_config(file_path)

    # Run the experiment 
    ExperimentManager.seed_experiment(config.general.seed)

    pipeline = ExperimentManager(config, device, file_path)
    pipeline.phenix_manager_path = args.phenix_manager
    pipeline.run()
    
    if config.loss_function.loss_function_type == "density": 
        pipeline.calculate_density_metrics()
    if config.loss_function.loss_function_type == "sf":
        pipeline.loss_function.save_structure_factors(pipeline.structures, os.path.join(pipeline.experiment_save_dir,"diffusion_guidance", "fc.ccp4"))

    # TODO: add the Cryo stuff evaluation runner. 
    

    

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--configuration_file', type=str, required=False, help="the path to the configuration file", default="pipeline_configurations/baseline.yaml")
    parser.add_argument('--device', type=str, required=False, default="cuda:0")
    parser.add_argument('--phenix_manager', type=str, required=False, default=None)
    args = parser.parse_args()
    main(args)
