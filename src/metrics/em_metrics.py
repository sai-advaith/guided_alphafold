
import importlib
import sys

import os
import numpy as np
import torch
import argparse
from ..utils.io import load_config, namespace_to_dict, load_pdb_atom_locations_full
from ..utils.phenix_manager import PhenixManager
from ..losses.em_loss_function import CryoEM_ESP_GuidanceLossFunction
import matplotlib.pyplot as plt
import pandas as pd

def load_pdb_structures(folder_path, cryoesp_loss_function, device="cuda:0"): 
    """
    Simple function to load PDB structures with PDBID_number.pdb naming pattern.
    Falls back to "full_structures" folder if coordinates are shorter than expected.
    
    Args:
        folder_path (str): Path to folder containing PDB files
        device (str): Device to load tensors on
        
    Returns:
        torch.Tensor: Structures tensor of shape [n_structures, n_atoms, 3]
    """
    import gemmi
    import torch
    import os
    import re
    
    print(f"Loading PDB structures from: {folder_path}")
    
    # Find PDB files with PDBID_number.pdb pattern (e.g., 6WO0_0.pdb, 6WO0_1.pdb)
    pdb_files = []
    for f in os.listdir(folder_path):
        if f.endswith('.pdb') and re.match(r'^[A-Za-z0-9]+_\d+\.pdb$', f):
            pdb_files.append(f)
    
    if not pdb_files:
        raise ValueError(f"No PDB files with PDBID_number.pdb pattern found in {folder_path}")
    
    # Sort by number
    pdb_files.sort(key=lambda x: int(x.split('_')[1].replace('.pdb', '')))
    print(f"Found {len(pdb_files)} PDB files: {pdb_files}")

    # Determine expected number of atoms directly from CryoESP loss object
    expected_coords_shape = cryoesp_loss_function.coordinates_gt.shape[0]
    
    structures = []
    for pdb_file in pdb_files:
        pdb_path = os.path.join(folder_path, pdb_file)
        print(f"  Loading {pdb_file}...")
        
        # Load and extract coordinates
        structure = gemmi.read_structure(pdb_path)
        coords = []
        for chain in structure[0]:
            for residue in chain:
                for atom in residue:
                    coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
        
        coords_tensor = torch.tensor(coords, dtype=torch.float32, device=device)
        structures.append(coords_tensor)
        print(f"    Shape: {coords_tensor.shape}")
        
        # Check if coordinates are shorter than expected and fall back to full_structures
        if expected_coords_shape is not None and coords_tensor.shape[0] < expected_coords_shape:
            print(f"    Warning: Coordinates shape {coords_tensor.shape[0]} is shorter than expected {expected_coords_shape}")
            full_structures_folder = os.path.join(folder_path, "full_structures")
            if os.path.exists(full_structures_folder):
                print(f"    Falling back to full_structures folder: {full_structures_folder}")
                full_pdb_path = os.path.join(full_structures_folder, pdb_file)
                if os.path.exists(full_pdb_path):
                    print(f"    Loading from full_structures: {pdb_file}...")
                    full_structure = gemmi.read_structure(full_pdb_path)
                    full_coords = []
                    for chain in full_structure[0]:
                        for residue in chain:
                            for atom in residue:
                                full_coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                    
                    full_coords_tensor = torch.tensor(full_coords, dtype=torch.float32, device=device)
                    structures[-1] = full_coords_tensor  # Replace the last added structure
                    print(f"    Full structure shape: {full_coords_tensor.shape}")
                else:
                    print(f"    Warning: Full structure file {full_pdb_path} not found, keeping original")
            else:
                print(f"    Warning: full_structures folder not found at {full_structures_folder}, keeping original")
    
    structures_tensor = torch.stack(structures, dim=0)
    print(f"Final structures tensor shape: {structures_tensor.shape}")
    return structures_tensor

def reload_em_loss():
    """
    Hot reload the EM loss function module for debugging.
    Call this function after making changes to the em_loss_function.py file.
    """
    print("Reloading EM loss function module...")
    
    # Remove the module from cache if it exists
    if 'src.losses.em_loss_function' in sys.modules:
        del sys.modules['src.losses.em_loss_function']
    
    # Also remove any related modules
    modules_to_remove = [k for k in sys.modules.keys() if 'em_loss_function' in k]
    for module in modules_to_remove:
        del sys.modules[module]
    
    # Reimport the module
    from src.losses.em_loss_function import CryoEM_ESP_GuidanceLossFunction
    
    print("✓ EM loss function module reloaded successfully!")
    return CryoEM_ESP_GuidanceLossFunction

def run_em_metrics(config_file_path, guided_model, phenix_env_path, rmax=None):
    # Set up plotting style
    plt.style.use('default')

    # Device setup
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load configuration
    config = load_config(config_file_path)
    print(f"Configuration loaded from: {config_file_path}")
    print(f"Protein PDB ID: {config.protein.pdb_id}")
    print(f"Loss function type: {config.loss_function.loss_function_type}")

    # Create sequences dictionary directly from config
    sequences_dictionary = config.protein.sequences
    print(f"Sequences dictionary: {sequences_dictionary}")

    # Create full sequences list (same logic as model manager)
    full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in sequences_dictionary]
    full_sequences = [item for sublist in full_sequences for item in sublist]
    print(f"Full sequences: {full_sequences}")

    # Create the mask using load_pdb_atom_locations_full (same as model manager)
    _, resolved_pdb_to_full_mask = load_pdb_atom_locations_full(
        config.protein.reference_pdb, 
        full_sequences=full_sequences,
        chains_to_read=config.protein.chains_to_use,
    )

    print(f"Mask shape: {resolved_pdb_to_full_mask.shape}")
    print(f"Number of atoms: {resolved_pdb_to_full_mask.sum().item()}")
    print("✓ Required components created successfully!")

    # Initialize CryoESP loss function (without model manager dependency)
    cryoesp_config = config.loss_function.cryoesp_loss_function
    CryoEM_ESP_GuidanceLossFunction = reload_em_loss()

    cryoesp_loss_function = CryoEM_ESP_GuidanceLossFunction(
        cryoesp_config.reference_pdb, 
        resolved_pdb_to_full_mask,  # Use our directly created mask
        cryoesp_config.esp_file, 
        emdb_resolution=cryoesp_config.emdb_resolution, 
        device=device, 
        is_assembled=(not cryoesp_config.is_assembled),
        global_b_factor=cryoesp_config.global_b_factor, 
        esp_gt_cutoff_value=cryoesp_config.esp_gt_cutoff_value,
        reduced_D=cryoesp_config.reduced_D, 
        use_Coloumb=cryoesp_config.use_Coloumb,
        regions_of_interest=[
            list(range(single_res_range[0], single_res_range[1]+1)) 
            for single_res_range in config.protein.residue_range
        ] if config.protein.residue_range is not None else None,
        sequences_dictionary=sequences_dictionary,  # Use our directly created sequences dictionary
        guide_only_ROI=cryoesp_config.guide_only_ROI, 
        save_folder=None,  # Will be set dynamically in rerun_evaluation
        aling_only_outside_ROI=cryoesp_config.aling_only_outside_ROI, 
        optimize_b_factors=cryoesp_config.optimize_bfactor, 
        should_align_to_chains=config.protein.should_align_to_chains,
        chains_to_read=config.protein.chains_to_use,
        to_convex_hull_of_ROI=config.protein.should_fill_mask, 
        reapply_b_factor=cryoesp_config.reapply_b_factor, 
        reapply_is_learnable=cryoesp_config.reapply_is_learnable,
        sinkhorn_parameters={
            "percentage": cryoesp_config.sinkhorn.percentage,
            "p": cryoesp_config.sinkhorn.p,
            "blur": cryoesp_config.sinkhorn.blur,
            "reach": cryoesp_config.sinkhorn.reach,
            "scaling": cryoesp_config.sinkhorn.scaling,
            "turn_off_after": cryoesp_config.sinkhorn.turn_off_after,
            "backend": cryoesp_config.sinkhorn.backend,
            "debug_with_rmsd": cryoesp_config.sinkhorn.debug_with_rmsd,
            "guide_multimer_by_chains": cryoesp_config.sinkhorn.guide_multimer_by_chains,
            "debias": cryoesp_config.sinkhorn.debias,
        },
        combinatorially_best_alignment=cryoesp_config.combinatorially_best_alignment,
        alignment_strategy=cryoesp_config.alignment_strategy,
        rmax_for_esp=cryoesp_config.rmax_for_esp, 
        rmax_for_mask=rmax if rmax is not None else cryoesp_config.rmax_for_mask,
        rmax_for_final_bfac_fitting= rmax if rmax is not None else cryoesp_config.rmax_for_final_bfac_fitting,
        reordering_every=cryoesp_config.reordering_every,
        dihedrals_parameters={
            "use_dihedrals": cryoesp_config.dihedrals.use_dihedrals,
            "dihedral_loss_weight": cryoesp_config.dihedrals.dihedral_loss_weight,
            "dihedrals_file": cryoesp_config.dihedrals.dihedrals_file,
        },
        symmetry_parameters={
            "symmetry_type": cryoesp_config.symmetry.symmetry_type, 
            "reapply_symmetry_every": cryoesp_config.symmetry.reapply_symmetry_every,
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
        evaluate_only_resolved=getattr(cryoesp_config, 'evaluate_only_resolved', False),
    )

    print("CryoESP loss function initialized successfully!")
    print(f"Reference PDB: {cryoesp_config.reference_pdb}")
    print(f"ESP file: {cryoesp_config.esp_file}")
    print(f"Global B-factor: {cryoesp_config.global_b_factor}")
    print(f"Ground truth coordinates shape: {cryoesp_loss_function.coordinates_gt.shape}")
    print(f"Ground truth B-factors shape: {cryoesp_loss_function.bfactor_gt.shape}")

    # Source = guided
    print(f"Using source_model: {guided_model}")

    # Load structures from guidance/unguided folder
    structures_folder = os.path.dirname(guided_model)
    output_folder = structures_folder
    structures = load_pdb_structures(structures_folder, cryoesp_loss_function, device)

    # Point CryoESP save_folder to the chosen output folder
    cryoesp_loss_function.save_folder = output_folder
    phenix_manager = PhenixManager(phenix_env_path)
    cryoesp_loss_function.save_state(
        structures,
        output_folder,
        phenix_manager,
        skip_png=True,
        b_factor_lr=2.0,
        use_zero_b_values=False,
        should_always_fit_gt=False,
        n_iterations=800,
        bfactor_min=10.0,  # Custom B-factor range
        bfactor_max=800.0,
        use_cross_correlation=True,
        bfactor_regularization=0.0,
    )
    print(f"✓ Evaluation completed! Results in: {output_folder}")

