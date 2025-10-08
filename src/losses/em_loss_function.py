import torch
import torch.nn.functional as F
from .abstract_loss_funciton import AbstractLossFunction
from ..utils.cryoesp_calculator import compute_elden_no_cycle_keops, initialize_lattice_coordinates, compute_Coloumb_stype_potential
from ..utils.io import get_sampler_pdb_inputs, get_atom_mask, alignment_mask_by_chain
import numpy as np
from ..protenix.metrics.rmsd import (
    self_aligned_rmsd,
    align_protein_to_density_pca,
    align_protein_to_protein_pca,
    align_multimeric_protein_to_multimeric_density_by_chain,
    blob_se3_align_adam_debug,
    blob_se3_align_adam_multi_start,
    interpolate_scalar_volume_at_points_fast_testing_corrected,
    run_batched_alignment
)
from ipdb import set_trace
import gemmi
from ..utils.io import (
    load_pdb_atom_locations_full,
    talos_to_dihedral_tensor,
    parse_dihedrals_csv,
    create_atom_mask,
    create_full_element_list,
    create_backbone_masks,
    create_cc_bfactor_tensor,
    parse_phenix_cc_log,
    parse_phenix_eval_log_to_csv,
)
from pykeops.torch import LazyTensor
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from math import sqrt
from copy import copy
from geomloss import SamplesLoss
import os
from scipy.spatial import ConvexHull
from matplotlib.patches import Patch
import itertools 
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import label
import subprocess
from ..utils.non_diffusion_model_manager import save_structure_full
from ..utils.fft_util import fft_downsample_3d, fft_upsample_3d, apply_resolution_cutoff, compute_freqs, apply_bfactor_to_map

# For saving images:
import imageio.v3 as iio
import re 
from pathlib import Path


from skimage.morphology import skeletonize
from skan import Skeleton # Import the main class
import skan
import scipy

class CryoEM_ESP_GuidanceLossFunction(AbstractLossFunction):
    def __init__(
            self, reference_pdb, mask, aligned_guiding_ESP_file, 
            emdb_resolution = 3.0, device="cpu", is_assembled=False, 
            global_b_factor = 50.0, esp_gt_cutoff_value = None, reduced_D=None, use_Coloumb=False,
            should_add_b_factor_for_resolution_cutoff = False,
            regions_of_interest=None, guide_only_ROI=False, 
            rmax_for_mask = 3.5, rmax_for_esp = 5.0, rmax_for_final_bfac_fitting=2.5, rmax_for_backbone=2.5, optimize_b_factors=False, save_folder:str = None, aling_only_outside_ROI=False,
            sequences_dictionary=None, should_align_to_chains=[0], chains_to_read=[0], to_convex_hull_of_ROI=False, reapply_b_factor=False,
            reapply_is_learnable=False, alignment_strategy="global_rmsd_to_gt",
            sinkhorn_parameters = None, combinatorially_best_alignment = False, reordering_every = 10,
            dihedrals_parameters = None, symmetry_parameters=None, gradient_ascent_parameters=None,
            evaluate_only_resolved=False, 
        ): 
        
        full_sequences = [[dictionary["sequence"],]*dictionary["count"] for dictionary in sequences_dictionary]
        full_sequences = [item for sublist in full_sequences for item in sublist]
        self.full_sequences = full_sequences 
        self.sequences_dictionary = sequences_dictionary
        self.masks_per_sequence = [alignment_mask_by_chain(full_sequences, [sequence_id]).to(device) for sequence_id, sequence in enumerate(full_sequences)]
        self.mask_indices_per_sequence = [
            torch.nonzero(mask, as_tuple=True)[0] for mask in self.masks_per_sequence
        ]

        # SAVING GLOBAL VARIABLES
        if sinkhorn_parameters is None:
            self.sinkhorn_parameters = {
                "blur": 0.5, 
                "p": 1, 
                "reach": None, 
                "scaling": 0.90, 
                "percentage": 0.0, 
                "turn_off_after": 100,
                "debug_with_rmsd": False,
                "debias": True,
                "guide_multimer_by_chains": False,
            }
        else: 
            sinkhorn_parameters = sinkhorn_parameters
        sinkhorn_parameters["counter"] = 0

        if dihedrals_parameters is None:
            self.dihedrals_parameters = {
                "use_dihedrals": False, # whether to use dihedrals from the ground truth or not. if False, dihedrals are not used
                "dihedral_loss_weight": 0.1, # weight of the dihedral loss
                "dihedrals_file": None, # file with dihedrals to use (if not from gt)
            }
        else:
            self.dihedrals_parameters = dihedrals_parameters
        if dihedrals_parameters["dihedrals_file"] is not None:
            if dihedrals_parameters["dihedrals_file"].endswith(".csv"):
                self.dihedrals_tensor, self.dihedrals_mask, self.chi1_dchi1_tensor, self.chi1_mask = parse_dihedrals_csv(dihedrals_parameters["dihedrals_file"], self.full_sequences[0], device=device)
            elif dihedrals_parameters["dihedrals_file"].endswith(".tab"):
                self.dihedrals_tensor, self.dihedrals_mask = talos_to_dihedral_tensor(
                    dihedrals_parameters["dihedrals_file"], 
                    self.full_sequences[0], 
                    device=device
                )
                # Initialize chi1 tensors as empty (TALOS doesn't provide chi1 angles)
                self.chi1_dchi1_tensor = torch.zeros((len(self.full_sequences[0]), 2), device=device)
                self.chi1_mask = torch.zeros(len(self.full_sequences[0]), device=device, dtype=torch.bool)
            else:
                raise ValueError(f"Dihedrals file {dihedrals_parameters['dihedrals_file']} is not a valid file.")

        if symmetry_parameters is None:
            self.symmetry_parameters = {
                "symmetry_type": None, # "custom_amyloid", "none", # Just do nothing by default..!
                "reapply_symmetry_every": 20, # how often to reapply the symmetry [or at which steps]
            }
        else:  
            self.symmetry_parameters = symmetry_parameters
        self.symmetry_parameters["counter"] = 0

        if gradient_ascent_parameters is None:
            self.gradient_ascent_parameters = {
                "steps": 30,
                "lr_t_A": 500,
                "lr_r_deg": 1000,
                "reduction": "mean",
                "per_step_t_cap_voxels": 10.0,
                "Bfac": 200,
                "bfactor_minimum": 200,
                "n_random": 10000,
                "t_init_box_edge_voxels": 25.0,
            }
        else:
            self.gradient_ascent_parameters = gradient_ascent_parameters


        self.sinkhorn_parameters = sinkhorn_parameters
        self.global_b_factor = global_b_factor 
        self.should_add_b_factor_for_resolution_cutoff = should_add_b_factor_for_resolution_cutoff
        self.esp_gt_cutoff_value = esp_gt_cutoff_value
        self.use_Coloumb = use_Coloumb # whether to use the Coloumb potential or not. if not, the elden potential is used
        self.guide_only_ROI = guide_only_ROI # whether to guide only the regions of interest or not. if True, density is computed only for the regions of interest
        self.device = device
        self.emdb_resolution = emdb_resolution; self.emdb_resolution_full = copy(emdb_resolution)
        self.percentage_of_rmsd_loss = sinkhorn_parameters["percentage"] # at the beginning, mostly computing the rmsd loss
        self.rmax_for_esp = rmax_for_esp
        self.rmax_for_mask = rmax_for_mask
        self.rmax_for_final_bfac_fitting = rmax_for_final_bfac_fitting
        self.rmax_for_backbone = rmax_for_backbone
        self.save_folder = save_folder
        self.evolution_plot_number = 0
        self._num_suffix = re.compile(r'_(\d+)\.png$', re.IGNORECASE)
        self.align_only_outside_ROI = aling_only_outside_ROI
        self.evaluate_only_resolved = evaluate_only_resolved 
        self.chains_to_read = chains_to_read 
        self.should_align_to_chains = should_align_to_chains # which chains to align to if the protein is symmetric (eigenvectors)
        self.align_to_chain_mask = alignment_mask_by_chain(full_sequences, chains_to_align=should_align_to_chains).to(device)
        self.reapply_b_factor = reapply_b_factor # if True, the b factor is reapplied to the coordinates of the ground truth structure. If False, it is not applied
        self.reapply_is_learnable = reapply_is_learnable # whether the reapply b factor is learnable or not. If True, it is optimized during training
        self.combinatorially_best_alignment = combinatorially_best_alignment
        self.alignment_strategy = alignment_strategy

        # Values to be saved and reported 
        self.last_loss_value = None 
        self.last_rmsd_value = None # potentially track the evolution of rmsd
        self.last_bond_length_value = None
        self.last_cosine_similarity = None
        self.last_OT_value = None
        self.last_density_loss_value = None
        if self.guide_only_ROI or self.align_only_outside_ROI:
            self.last_cosine_similarity_ROI = None
        self.last_dihedrals_loss_value = None
        
        self.reference_pdb_path = reference_pdb
        self.coordinates_gt, _, self.bfactor_gt, self.element_gt = \
            load_pdb_atom_locations_full(
                pdb_file=reference_pdb, 
                full_sequences=full_sequences,
                chains_to_read=chains_to_read,
                return_elements=True,
                return_bfacs=True,
                return_mask=True
            ) # the index is such that all the atoms get read!
        self.bfactor_gt_untouched = self.bfactor_gt.clone()

        self.full_pdb = self.coordinates_gt.clone().to(device)
        self.coordinates_gt = self.coordinates_gt.to(device) # coordinates of the ground truth structure
        self.bfactor_gt = self.bfactor_gt.to(device) # b-factors of the ground truth structure
        self.element_gt = self.element_gt.to(device) # elements of the ground truth structure
        
        self.AF3_to_pdb_mask = mask.to(device) # masking which of the AF3 entries are missing in the pdb file to align appropriately

        N_atoms = mask.shape[0]
        
        if global_b_factor is not None: # is required only in the case that there's no know b-factors. Like TET2 for example
            self.bfactor_gt[:] = global_b_factor

        # READING INTO THE DENSITY MAP AND SAVING CORRESPONDING VALUES
        self.esp_file = aligned_guiding_ESP_file
        density_map = gemmi.read_ccp4_map(aligned_guiding_ESP_file)
        self.density_map = density_map  

        self.D = density_map.grid.nu 
        self.D_full = density_map.grid.nu

        maxsize = density_map.grid.unit_cell.a

        self.pixel_size = density_map.grid.unit_cell.a / density_map.grid.nu
        self.pixel_size_full = density_map.grid.unit_cell.a / density_map.grid.nu 
        
        leftbottompoint =  np.array(list(density_map.get_extent().minimum)) * maxsize
        rightupperpoint =  np.array(list(density_map.get_extent().maximum)) * maxsize
        self.leftbottompoint = leftbottompoint; self.rightupperpoint = rightupperpoint

        self.fo = torch.from_numpy(density_map.grid.array).to(device) 
        self.fo_full = self.fo.clone() 
        self.fo_unthresholded = self.fo.clone()
        
        if self.esp_gt_cutoff_value is not None:
            self.fo = torch.where(self.fo >= self.esp_gt_cutoff_value, self.fo, self.esp_gt_cutoff_value) # deleting the values that are below the cutoff value as noise.
        
            self.fo_threshold_mask = torch.where(self.fo_full >= self.esp_gt_cutoff_value, 1.0, 0.0).to(torch.bool)
            self.fo_threshold_mask_full = torch.where(self.fo_full >= self.esp_gt_cutoff_value, 1.0, 0.0).to(torch.bool)
        else: 
            self.fo_threshold_mask = torch.ones_like(self.fo, dtype=torch.bool).to(self.device) # if no thresholding is needed, use the full mask

        # SCALE THE VOLUME IF NEEDED
        if reduced_D != self.D and reduced_D is not None:
            self.D_mininimal = 2 * self.D / emdb_resolution / self.pixel_size
            if reduced_D < self.D_mininimal:
                raise ValueError(f"Reduced D {reduced_D} is smaller than the minimal possible D {self.D_mininimal} for the given resolution {emdb_resolution} and pixel size {self.pixel_size}.")
            
            self.D = reduced_D # (for now 180)
            self.pixel_size = self.pixel_size_full * (self.D_full / self.D) 

            self.fo = fft_downsample_3d(self.fo, (self.D,)*3) 
            self.fo_unthresholded = fft_downsample_3d(self.fo_unthresholded, (self.D,)*3) 
            
            self.fo_threshold_mask = fft_downsample_3d(self.fo_threshold_mask, (self.D,)*3)
            self.fo_threshold_mask = torch.where(self.fo_threshold_mask < 0.5, 0.0, 1.0).to(torch.bool)

        # PREPARING THE GRIDS
        self.lattice_3d = initialize_lattice_coordinates(
            self.D, self.pixel_size, leftbottompoint=leftbottompoint, rightupperpoint=rightupperpoint
        ).to(device)

        self.lattice_3d_full = initialize_lattice_coordinates(
            self.D_full, self.pixel_size_full, leftbottompoint=leftbottompoint, rightupperpoint=rightupperpoint
        ).to(device)

        # CREATING MASKS 
        # 1) ROI mask (if plotting or only guiding in the ROI region)
        if full_sequences is not None and regions_of_interest is not None: 
            self.full_sequences = full_sequences
            self.regions_of_interest = regions_of_interest
            self.regions_of_interest_mask = create_atom_mask(full_sequences, regions_of_interest).to(device) # this is the mask of the atoms that are in the regions of interest

        # 2) ROI density mask (of the D**3 volume) 
        for_zone_of_interest = self.calculate_ESP(
            self.coordinates_gt.unsqueeze(0),
            single_b_fac=300.0, use_Coloumb=self.use_Coloumb, 
            should_align=False, atom_mask=self.regions_of_interest_mask, rmax=self.rmax_for_mask
        )
        self.density_zone_of_interest_mask = torch.where(for_zone_of_interest > 0.0, 1.0, 0.0).to(torch.bool).to(self.device) # this is the mask of the atoms that are in the regions of interest

        # 3) Density mask for where to calculate the loss (in case guidance is by the full protein structure)
        fo_for_mask = self.calculate_ESP(
            self.coordinates_gt.unsqueeze(0), atom_mask = self.AF3_to_pdb_mask,
            single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_mask, should_align=False 
        ) 
        self.density_mask = torch.where(fo_for_mask > 0.0, 1.0, 0.0).to(torch.bool).to(self.device) 
        
        fo_for_final_bfac_fitting = self.calculate_ESP(
            self.coordinates_gt.unsqueeze(0), atom_mask = self.AF3_to_pdb_mask,
            single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_final_bfac_fitting, should_align=False 
        ) 
        self.density_mask_for_final_bfac_fitting = torch.where(fo_for_final_bfac_fitting > 0.0, 1.0, 0.0).to(torch.bool).to(self.device) 

        fo_for_backbone = self.calculate_ESP(
            self.coordinates_gt.unsqueeze(0), atom_mask = self.AF3_to_pdb_mask,
            single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_backbone, should_align=False 
        ) 
        self.density_mask_for_backbone = torch.where(fo_for_backbone > 0.0, 1.0, 0.0).to(torch.bool).to(self.device) 

        if to_convex_hull_of_ROI:
            # Zone of interest hull
            try: 
                coords = self.lattice_3d[self.density_zone_of_interest_mask.flatten(), :].cpu()
                hull = ConvexHull(coords)
                A = torch.from_numpy(hull.equations[..., :-1]).to(torch.float32).to(self.device)
                b = torch.from_numpy(hull.equations[..., -1]).to(torch.float32).to(self.device).unsqueeze(-1)
                
                # Memory-efficient convex hull check using chunking
                new_mask = self._compute_convex_hull_mask_chunked(A, b)
                self.density_zone_of_interest_mask = new_mask.clone()
            except: 
                1 # fallback and don't fill in if rmax is too small for per-chain masks

            # Density mask hull
            coords = self.lattice_3d[self.density_mask.flatten(), :].cpu()
            hull = ConvexHull(coords)
            A = torch.from_numpy(hull.equations[..., :-1]).to(torch.float32).to(self.device)
            b = torch.from_numpy(hull.equations[..., -1]).to(torch.float32).to(self.device).unsqueeze(-1)
            
            # Memory-efficient convex hull check using chunking
            new_mask = self._compute_convex_hull_mask_chunked(A, b)
            self.density_mask = new_mask.clone()

            # Density mask for final b-factor fitting hull
            coords = self.lattice_3d[self.density_mask_for_final_bfac_fitting.flatten(), :].cpu()
            hull = ConvexHull(coords)
            A = torch.from_numpy(hull.equations[..., :-1]).to(torch.float32).to(self.device)
            b = torch.from_numpy(hull.equations[..., -1]).to(torch.float32).to(self.device).unsqueeze(-1)

            # Memory-efficient convex hull check using chunking
            new_mask = self._compute_convex_hull_mask_chunked(A, b)
            self.density_mask_for_final_bfac_fitting = new_mask.clone()

            # Density mask for backbone hull
            coords = self.lattice_3d[self.density_mask_for_backbone.flatten(), :].cpu()
            hull = ConvexHull(coords)
            A = torch.from_numpy(hull.equations[..., :-1]).to(torch.float32).to(self.device)
            b = torch.from_numpy(hull.equations[..., -1]).to(torch.float32).to(self.device).unsqueeze(-1)

            # Memory-efficient convex hull check using chunking
            new_mask = self._compute_convex_hull_mask_chunked(A, b)
            self.density_mask_for_backbone = new_mask.clone()

        self.sinkhorn_loss_function = SamplesLoss(
            loss="sinkhorn", 
            p=sinkhorn_parameters["p"], 
            blur=sinkhorn_parameters["blur"], 
            backend=sinkhorn_parameters["backend"], 
            debias=self.sinkhorn_parameters["debias"], 
            reach=sinkhorn_parameters["reach"], 
            scaling=sinkhorn_parameters["scaling"],
        )

        if self.sinkhorn_parameters["guide_multimer_by_chains"]: # Temporarily supports only the 
            self.density_masks_per_chain = []
            for mask in self.masks_per_sequence:
                temp_for_for_chain_mask  = self.calculate_ESP(
                    self.coordinates_gt.unsqueeze(0), atom_mask = self.AF3_to_pdb_mask & mask,
                    single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_mask, should_align=False 
                ) 
                self.density_masks_per_chain.append(
                    torch.where(temp_for_for_chain_mask > 0.0, 1.0, 0.0).to(torch.bool).to(self.device)
                ) 

            self.density_masks_per_chain_for_backbone = []
            for mask in self.masks_per_sequence:
                temp_for_for_chain_mask = self.calculate_ESP(
                    self.coordinates_gt.unsqueeze(0), atom_mask = self.AF3_to_pdb_mask & mask,
                    single_b_fac=300.0, use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_backbone, should_align=False 
                ) 
                self.density_masks_per_chain_for_backbone.append(
                    torch.where(temp_for_for_chain_mask > 0.0, 1.0, 0.0).to(torch.bool).to(self.device)
                ) 

        self.fc_from_gt = self.calculate_ESP(self.coordinates_gt.unsqueeze(0), use_Coloumb=False, single_b_fac=self.global_b_factor, should_align=False) 
        self.fc_from_gt_mean, self.fc_from_gt_std = self.fc_from_gt[self.density_mask].mean(), self.fc_from_gt[self.density_mask].std()

        self.optimize_b_factors = optimize_b_factors
        if optimize_b_factors:
            self.bfactor_gt.requires_grad = True 

            if reapply_is_learnable is True:
                self.reapply_b_factor = torch.tensor(reapply_b_factor, dtype=torch.float32, device=self.device) # reapply b factor to the coordinates of the ground truth structure
                self.reapply_b_factor.requires_grad = True 
                self.fo_unblurred = self.fo.clone()

            self.parameters_optimizer = torch.optim.Adam(self.get_optimizable_parameters(), lr=1) 

        if reapply_b_factor is not False: 
            if isinstance(reapply_b_factor, float):
                self.fo = apply_bfactor_to_map(
                    self.fo, pixel_size=self.pixel_size, B_blur=reapply_b_factor, device=self.device
                ).to(self.device)
            else:
                raise ValueError(
                    f"reapply_b_factor should be a float or False, but got {reapply_b_factor} instead."
                )
        
        self.previous_reordering = torch.arange(self.coordinates_gt.shape[0], device=self.device) # this is the previous reordering of the coordinates, which is the identity permutation at the beginning
        self.reordering_counter = reordering_every-1
        self.reordering_every = reordering_every-1 # how often to reorder the multimer alignment

        self.backbone_masks = create_backbone_masks(self.full_sequences, device=self.device)

        if self.sinkhorn_parameters["guide_multimer_by_chains"] is not False:
            self._initialize_interpolant_backbone_points_per_sequence()

    def _compute_convex_hull_mask_chunked(self, A, b, chunk_size=50000):
        """
        Memory-efficient computation of convex hull mask by processing lattice points in chunks.
        
        Args:
            A: Hull equations normal vectors [num_faces, 3]
            b: Hull equations constants [num_faces, 1]  
            chunk_size: Number of lattice points to process at once
            
        Returns:
            Boolean mask of shape [D, D, D] indicating points inside convex hull
        """
        total_points = self.lattice_3d.shape[0]  # D^3
        result_mask = torch.zeros(total_points, dtype=torch.bool, device=self.device)
        
        # Process lattice points in chunks to avoid memory issues
        for start_idx in range(0, total_points, chunk_size):
            end_idx = min(start_idx + chunk_size, total_points)
            
            # Get chunk of lattice points [chunk_size, 3]
            chunk_coords = self.lattice_3d[start_idx:end_idx]
            
            # Compute hull constraints for this chunk: A @ chunk_coords.T + b <= 0
            # A is [num_faces, 3], chunk_coords.T is [3, chunk_size]
            # Result is [num_faces, chunk_size]
            constraints = A @ chunk_coords.T + b
            
            # Point is inside hull if ALL constraints are satisfied (<=0)
            # Shape: [chunk_size]
            chunk_mask = (constraints <= 0).all(0)
            
            # Store result for this chunk
            result_mask[start_idx:end_idx] = chunk_mask
            
            # Clear chunk from GPU memory
            del chunk_coords, constraints, chunk_mask
            
        return result_mask.reshape(self.D, self.D, self.D)

    def replace_non_residue_range_atoms(self, x_0_hat):
        new_x_0_hat = x_0_hat.clone()
        new_x_0_hat[:, ~self.regions_of_interest_mask, :] = self.coordinates_gt[~self.regions_of_interest_mask, :]
        return new_x_0_hat # all atoms are resolved
        # otherwise overlap with self.AF3_to_pdb_mask
    
    def wandb_log(self, x_0_hat):
        to_return = {
            "loss": self.last_loss_value,
            "rmsd":self.last_rmsd_value,
            "OT_loss": self.last_OT_value,
            "bond_length": self.last_bond_length_value,
            "cosine_similarity": self.last_cosine_similarity,
            "density_loss": self.last_density_loss_value, 
            "dihedrals_loss": self.last_dihedrals_loss_value,
        }
        if self.guide_only_ROI or self.align_only_outside_ROI:
            to_return["cosine_similarity_ROI"] = self.last_cosine_similarity_ROI

        return to_return 

    def get_optimizable_parameters(self):
        to_return = []
        if self.optimize_b_factors:
            to_return.append(self.bfactor_gt)
            if self.reapply_is_learnable is not False:
                    to_return.append(self.reapply_b_factor)

        return to_return

    def post_optimization_step(self):
        if self.optimize_b_factors:
            self.bfactor_gt.data.clamp_(min=30.0, max=450.0) # clamping to some reasonable b factor values!
            self.parameters_optimizer.step()
            self.bfactor_gt.data.clamp_(min=30.0, max=450.0) # clamping to some reasonable b factor values!

            if self.reapply_is_learnable is not False:
                self.reapply_b_factor.data.clamp_(min=30.0, max=350.0)
                # Since the step has already been taken above, reapply the new blur value now
                self.fo = apply_bfactor_to_map(
                    self.fo_unblurred, pixel_size=self.pixel_size, B_blur=self.reapply_b_factor, device=self.device
                ).to(self.device)
            
    def calculate_ESP(
            self, x_0_hat, single_b_fac = None, use_Coloumb=False, should_align=True,
            atom_mask=None, rmax = sqrt(5.0), full_grid = False, bfactor = None
        ):

        D = self.D if not full_grid else self.D_full 
        lattice = self.lattice_3d if not full_grid else self.lattice_3d_full # choosing the lattice to use

        if atom_mask is None: # Means that the full mask is needed.
            atom_mask = torch.ones((x_0_hat.shape[1]), dtype=torch.bool) 

        #x_0_hat_reduced = x_0_hat[:, atom_mask, :]
        x_0_hat = x_0_hat[:, atom_mask, :]

        batch_size = x_0_hat.shape[0]
        if should_align:
            _, x_0_hat, _, _ = self_aligned_rmsd(
                x_0_hat, 
                self.coordinates_gt.repeat(batch_size, 1, 1), 
                self.AF3_to_pdb_mask 
            )

        bfac = self.bfactor_gt.unsqueeze(-1) if bfactor is None else bfactor.unsqueeze(-1)
        if self.should_add_b_factor_for_resolution_cutoff:
            bfac = bfac + (8 * torch.pi**2 * self.emdb_resolution**2) # The additive resolution cutoff b-factor
        if single_b_fac is not None:
            bfac = bfac.clone()
            bfac[:] = single_b_fac # used to be in place unsafe..! 
        bfac = bfac[atom_mask] 

        # Only computing the density around the atoms of the ensemble/model (choose the mask in non-differentiable way)
        with torch.no_grad():
            lattice_i = LazyTensor( lattice[:, None, :] )
            atom_positions_j = LazyTensor( x_0_hat.reshape(-1, 3)[None, :, :].detach() ) # flatten out all the atoms to ensure every atom fits into the grid appropriately
            D_ij =  (lattice_i - atom_positions_j).square().sum(dim=2, keepdim=True)  # Squared distance calculation  
            keep = D_ij < rmax**2 # if less than 7 angstroms, then no effect
            mask = keep.sum(1)
            mask = torch.where(mask > 0, 1.0, 0.0).detach().to(torch.bool).to(x_0_hat.device).flatten() # the mask of which lattice points are used for computations

        vol = torch.zeros(D**3, dtype=torch.float32, device=self.device)
        
        elden_function = compute_elden_no_cycle_keops if not use_Coloumb else compute_Coloumb_stype_potential
        
        elements = self.element_gt.unsqueeze(-1) if not use_Coloumb else self.element_gt.unsqueeze(-1).to(torch.float32) # if Coloumb is used, convert elements to float32 for keops computations
        elements = elements[atom_mask] # also take only the mask in the zone of interest

        vol[mask] = elden_function( # Computing full or limited grid (in the sense, either of shape DxDxD reduced, or D_full x D_full x D_full)
            D, lattice[mask,:], x_0_hat, 
            elements, 
            bfac, self.device
        ).flatten()
        vol = vol.reshape(D,D,D) 
        return vol 

    def align_multimer_by_hungarian_algo(self, x_0_hat, align_to, mask=None):
        
        #_, _, R, T = self_aligned_rmsd(
        #        x_0_hat[:, self.previous_reordering], align_to, 
        #        self.AF3_to_pdb_mask
        #    )
        
        #x_0_prealigned = (R[:,None] @ x_0_hat[...,None] + T[..., None] ).squeeze(-1)
        #x_0_prealigned, _, _ = align_protein_to_protein_pca(
        #    x_0_hat, align_to, 
        #    reduced_protein_mask=self.AF3_to_pdb_mask,
        #) # Pre-alignment using PCA function, then Hungarian algorithm to find best chain permutations
        x_0_prealigned, R, T = align_protein_to_density_pca(
            x_0_hat, self.lattice_3d[self.density_mask.flatten()].unsqueeze(0),
            self.element_gt[None, :, None], self.fo[self.density_mask][None, :, None],
            reduced_protein_mask=self.AF3_to_pdb_mask,
        )
        #x_0_prealigned = x_0_prealigned[:, self.previous_reordering, :] # prealign by the previous reordering, then get a new reordering, and order again

        chain_start_index = 0
        cost_matrices = []
        perm = []
        for dictionary_i, dictionary in enumerate(self.sequences_dictionary):
            cost = np.zeros( (dictionary["count"],)*2 )
            for i in range(chain_start_index, chain_start_index + dictionary["count"]):
                for j in range(chain_start_index, chain_start_index + dictionary["count"]):
                    mutual_resolved_mask = self.AF3_to_pdb_mask[self.masks_per_sequence[i]] & self.AF3_to_pdb_mask[self.masks_per_sequence[j]]
                    pred_chain_i =  x_0_prealigned[:, self.masks_per_sequence[i]][:, mutual_resolved_mask]
                    gt_chain_j = align_to[:, self.masks_per_sequence[j]][:, mutual_resolved_mask]
                    r = (pred_chain_i - gt_chain_j).square().sum(dim=-1).mean().sqrt()
                    #r = ( pred_chain_i.mean(1) - gt_chain_j.mean(1) ).square().sum().sqrt() # the distance between centroids
                    #r, aligned_reordered_x_0_hat, R, T = self_aligned_rmsd(
                    #    x_0_hat[:, self.masks_per_sequence[i]], align_to[:, self.masks_per_sequence[j]], # choosing the correct substructure and aligning them properly
                    #    self.AF3_to_pdb_mask[self.masks_per_sequence[i]] &  self.AF3_to_pdb_mask[self.masks_per_sequence[j]] # Ensure only resolved atoms are compared between chains
                    #) 
                    cost[i - chain_start_index,j - chain_start_index] = r
            cost_matrices.append(cost)
            perm.append(linear_sum_assignment(cost)[1] + chain_start_index) 
            chain_start_index += dictionary["count"]

        #print(cost[linear_sum_assignment(cost)].mean() for cost in cost_matrices)
        
        #perm = [linear_sum_assignment(cost)[1] for cost in cost_matrices]
        perm = np.concatenate(perm)
        print(perm)
        reordered_mask_indices = [
            self.mask_indices_per_sequence[i].to(dtype=torch.long, device=x_0_hat.device)
            for i in perm
        ]
        full_reordering_indices = torch.cat(reordered_mask_indices)
        self.previous_reordering = full_reordering_indices.clone()
        return full_reordering_indices.detach()

    def align_multimer_by_OT_hungarian(self, x_0_hat, mask=None):
        with torch.no_grad():
            loss_fn = SamplesLoss(
                "sinkhorn", p=1, blur=self.sinkhorn_parameters["blur"], scaling=0.70, debias = True
            )

            if mask is None:
                mask = torch.ones_like(self.AF3_to_pdb_mask, device=self.device, dtype=torch.bool)

            if True: 
                x_0_prealigned, R, T = align_multimeric_protein_to_multimeric_density_by_chain(
                    x_0_hat[:, self.previous_reordering, :],
                    [ self.lattice_3d[mask.flatten()].unsqueeze(0) for mask in self.density_masks_per_chain ],
                    self.element_gt[None, :, None],
                    [ self.fo[mask][None, :, None] for mask in self.density_masks_per_chain ],
                    reduced_protein_mask=self.AF3_to_pdb_mask,
                ) # prealign by the previous reordering, then get a new reordering, and order again..!
            else:  # potentailly add a flag to change prealignment strategy! 
                x_0_prealigned, R, T = align_protein_to_density_pca(
                    x_0_hat, self.lattice_3d[self.density_mask.flatten()].unsqueeze(0),
                    self.element_gt[None, :, None], self.fo[self.density_mask][None, :, None],
                    reduced_protein_mask=self.AF3_to_pdb_mask,
                )
            # x_0_prealigned, R, T = align_multimeric_protein_to_multimeric_density_by_chain(
            #     x_0_hat, 
            #     [ self.lattice_3d[mask.flatten()].unsqueeze(0) for mask in self.density_masks_per_chain ],
            #     self.element_gt[None, :, None], 
            #     [ self.fo[mask][None, :, None] for mask in self.density_masks_per_chain ],
            #     reduced_protein_mask=self.AF3_to_pdb_mask,
            # )

            chain_start_index = 0
            cost_matrices = []
            perm = []

            for dictionary_i, dictionary in enumerate(self.sequences_dictionary):
                cost = np.zeros((dictionary["count"],) * 2)
                for i in range(chain_start_index, chain_start_index + dictionary["count"]):
                    pred_chain_coords = x_0_prealigned[:, self.masks_per_sequence[i] & mask]  # shape (B, N_i, 3)
                    pred_chain_weights = self.element_gt[self.masks_per_sequence[i] & mask].repeat(x_0_hat.shape[0])  # shape (B*N_i,)
                    pred_chain_weights = pred_chain_weights / pred_chain_weights.sum()  # normalize

                    for j in range(chain_start_index, chain_start_index + dictionary["count"]):
                        # redo this cost_matrix to be the distance from weighted centroids now instead of what was had
                        voxel_mask = self.density_masks_per_chain[j]  # Boolean mask over full lattice
                        Y = self.lattice_3d[voxel_mask.flatten()]                     # shape (M_j, 3)
                        weightsY = self.fo[voxel_mask]                     # shape (M_j,)
                        weightsY = weightsY - weightsY.min()
                        weightsY = weightsY / weightsY.sum()
                        
                        centroid_pred = (pred_chain_coords * pred_chain_weights[:, None]).sum(dim=1)  # (B, 3)
                        centroid_Y = (Y * weightsY[:, None]).sum(dim=0)  # (3,)
                        dist = torch.norm(centroid_pred - centroid_Y[None, :], dim=1)
                        cost[i - chain_start_index, j - chain_start_index] = dist.item()

                        # # batch size is 1 here
                        # ot_loss = loss_fn(
                        #     pred_chain_weights[None, :], pred_chain_coords, 
                        #     weightsY[None, :], Y[None, :]
                        # )
                        # cost[i - chain_start_index, j - chain_start_index] = ot_loss.item()
                
                cost_matrices.append(cost)
                perm.append(linear_sum_assignment(cost)[1] + chain_start_index)
                chain_start_index += dictionary["count"]

            # Build the final permutation and indices
            perm = np.concatenate(perm)
            reordered_mask_indices = [
                self.mask_indices_per_sequence[i]
                for i in perm
            ]
            full_reordering_indices = torch.cat(reordered_mask_indices)
            self.previous_reordering = full_reordering_indices.clone() # save the last reordering indices for the next iteration
        #self.previous_reordering = full_reordering_indices.clone()
        print(perm)
        return full_reordering_indices.detach()

    def align_multimer_by_permutations(self, x_0_hat, align_to):    
        r_best = 1e10 # make sure that the first rmsd is definitely better than this
        iterable = all_groupwise_permutations( [sequence_dictionary["count"] for sequence_dictionary in self.sequences_dictionary] ) 
        for perm in iterable:

            # Step 2: Get and validate reordered indices
            reordered_mask_indices = [
                self.mask_indices_per_sequence[i].to(dtype=torch.long, device=x_0_hat.device)
                for i in perm
            ]
            all_indices = torch.cat(reordered_mask_indices)

            # Step 3: Reorder x_0_hat
            reordered_x_0_hat = x_0_hat[:, all_indices, :]

            
            r, aligned_reordered_x_0_hat, R, T = self_aligned_rmsd(
                reordered_x_0_hat, align_to, 
                (self.AF3_to_pdb_mask)  # consider this behaviour with should_align_to_chains
            )
            
            if r < r_best: 
                r_best = r
                best_reordering = all_indices.clone()

        return best_reordering.detach() 

    def align_structure(self, x_0_hat, align_to, i, step=None, use_saved_alignment=False, global_alignment_strategy = None, is_counted_down: bool = True, structures=None, time: float = 0.0):
        aligned_x_0_hat, R, T = self.align_structure_prealign(x_0_hat, align_to, use_saved_alignment, global_alignment_strategy, time=time)
        if self.symmetry_parameters["symmetry_type"] is not None:
            #print("Symmetry")
            aligned_x_0_hat = self.perform_symmetry_rotations(x_0_hat=aligned_x_0_hat, i=i, step=step, rigid_R=R.detach(), rigid_T=T.detach(), is_counted_down=is_counted_down, structures=structures, time=time) # flipping aligned x_0_hat if that suits the symmetry better.
        return aligned_x_0_hat, R, T

    def align_structure_prealign(self, x_0_hat, align_to, use_saved_alignment=False, global_alignment_strategy = None, time=0.0):
        global_alignment_strategy = global_alignment_strategy if global_alignment_strategy is not None else self.alignment_strategy

        # If global density or per_blob_centroid_alignment, then alignment is done and returned (no reordering and no Kabsch in the classical sense is needed)
        if global_alignment_strategy == "global_density":
            aligned_x_0_hat, R, T = align_protein_to_density_pca(
                x_0_hat, self.lattice_3d[self.density_mask.flatten()].unsqueeze(0),
                self.element_gt[None, :, None], self.fo[self.density_mask][None, :, None],
                reduced_protein_mask=self.AF3_to_pdb_mask,
            )
            return aligned_x_0_hat, R.detach(), T.detach()
        elif global_alignment_strategy == "per_blob_centroid_alignment":
            B = x_0_hat.shape[0]
            x_0_hat_centroids = (
                x_0_hat[:, self.AF3_to_pdb_mask, :] * self.element_gt[None, self.AF3_to_pdb_mask, None]
            ).reshape(B, len(self.full_sequences), -1, 3).sum(dim=2) / \
                self.element_gt[self.AF3_to_pdb_mask].reshape(1, len(self.full_sequences), -1, 1).sum(dim=2)

            fo_per_blob_positive = [
                self.fo[mask] - self.fo[mask].min()
                for mask in self.density_masks_per_chain
            ]
            blob_centroids = torch.stack([
                (self.lattice_3d[mask.flatten(), :] * fo_per_blob_positive[mask_i][..., None]).sum(dim=0) / fo_per_blob_positive[mask_i].sum() 
                for mask_i, mask in enumerate(self.density_masks_per_chain)
            ])[None]

            _, _, R, T = self_aligned_rmsd(
                x_0_hat_centroids, blob_centroids, atom_mask = torch.ones(len(self.full_sequences), device=x_0_hat.device, dtype=torch.bool)
            )
            x_0_hat_aligned = (R[:,None] @ x_0_hat[...,None] + T[..., None]).squeeze(-1)
            return x_0_hat_aligned, R.detach(), T.detach()
        elif global_alignment_strategy == "align_to_previous":
            if getattr(self, 'previous_coordinates', None) is not None:
                _, x_0_hat_aligned, R, T = self_aligned_rmsd(
                    x_0_hat, self.previous_coordinates,
                    atom_mask = self.AF3_to_pdb_mask
                )
                self.previous_coordinates = x_0_hat_aligned[0][None].clone().detach() # saving the B=0's file
                
            else:
                # If it is the first iteration and previous coordinates do not exist, align by the density centroids
                B = x_0_hat.shape[0]
                x_0_hat_centroids = (
                    x_0_hat[:, self.AF3_to_pdb_mask, :] * self.element_gt[None, self.AF3_to_pdb_mask, None]
                ).reshape(B, len(self.full_sequences), -1, 3).sum(dim=2) / \
                    self.element_gt[self.AF3_to_pdb_mask].reshape(1, len(self.full_sequences), -1, 1).sum(dim=2)
                print(x_0_hat_centroids[0, 0, :] - x_0_hat_centroids[0, 1, :])

                fo_per_blob_positive = [
                    self.fo[mask] - self.fo[mask].min()
                    for mask in self.density_masks_per_chain
                ]
                blob_centroids = torch.stack([
                    (self.lattice_3d[mask.flatten(), :] * fo_per_blob_positive[mask_i][..., None]).sum(dim=0) / fo_per_blob_positive[mask_i].sum() 
                    for mask_i, mask in enumerate(self.density_masks_per_chain)
                ])[None]
                _, _, R, T = self_aligned_rmsd(
                    x_0_hat_centroids, blob_centroids, atom_mask = torch.ones(len(self.full_sequences), device=x_0_hat.device, dtype=torch.bool)
                )
                x_0_hat_aligned = (R[:,None] @ x_0_hat[...,None] + T[..., None]).squeeze(-1)
                self.previous_coordinates = x_0_hat_aligned[0][None].clone().detach() # saving the B=0's file
            
            return x_0_hat_aligned, R.detach(), T.detach()

        elif global_alignment_strategy == "global_density_gradient_ascent":
            with torch.enable_grad():
                res = blob_se3_align_adam_multi_start(
                    coords=x_0_hat[0, self.AF3_to_pdb_mask, :].clone().detach(),
                    lattice_coords_3d=self.lattice_3d,
                    volume=self.fo,
                    mask3d=self.density_mask,    # [D,D,D] or [D^3]
                    voxel_size=float(self.pixel_size),
                    D=int(self.D),
                    steps=self.gradient_ascent_parameters["steps"],
                    lr_t_A=self.gradient_ascent_parameters["lr_t_A"] * float(self.pixel_size),
                    lr_r_deg=self.gradient_ascent_parameters["lr_r_deg"],
                    reduction=self.gradient_ascent_parameters["reduction"],
                    sampler_fn=interpolate_scalar_volume_at_points_fast_testing_corrected,
                    print_every=100,
                    per_step_t_cap_voxels=self.gradient_ascent_parameters["per_step_t_cap_voxels"],
                    Bfac=self.gradient_ascent_parameters["Bfac"],
                    bfactor_minimum=self.gradient_ascent_parameters["bfactor_minimum"],
                    n_random=self.gradient_ascent_parameters["n_random"],
                    seed=None,
                    return_all=False,
                    t_init_box_edge_voxels=self.gradient_ascent_parameters["t_init_box_edge_voxels"],
                    time=time,
                )
            R = res["best_R"][None].detach()
            T_global = res["T_global_best"][None, None].detach()
            #x_0_hat_aligned = x_0_hat.clone()
            x_0_hat_aligned = (
                x_0_hat @ R.transpose(-1, -2)
            ) + T_global
            #R[:,None] @ x_0_hat[...,None] + T_global[..., None]
            #).squeeze(-1)
            return x_0_hat_aligned, R, T_global
        
        elif global_alignment_strategy == "global_batched_density_alignment":
            from src.protenix.metrics.rmsd import batched_density_se3_align_adam_multi_start
            
            with torch.enable_grad():
                # Prepare ensemble from x_0_hat batch dimension
                # x_0_hat: [B, N_all, 3] where B is the ensemble size
                B_ensembles = x_0_hat.shape[0]
                ensemble_coords = x_0_hat[:, self.AF3_to_pdb_mask, :]  # [B_ensembles, N_masked, 3]
                
                # Prepare elements and b-factors for each ensemble member
                chain_elements = self.element_gt[self.AF3_to_pdb_mask].unsqueeze(0).expand(B_ensembles, -1)  # [B_ensembles, N]
                
                # Use trainable b-factors if available, otherwise use default
                if hasattr(self, 'bfactor_gt') and self.bfactor_gt is not None:
                    chain_b_factors = self.bfactor_gt[self.AF3_to_pdb_mask].unsqueeze(0).expand(B_ensembles, -1)  # [B_ensembles, N] - TRAINABLE!
                else:
                    chain_b_factors = torch.ones_like(chain_elements, dtype=torch.float32) * self.global_b_factor  # [B_ensembles, N] - fallback
                
                res = batched_density_se3_align_adam_multi_start(
                    coords=ensemble_coords,      # [B_ensembles, N_atoms, 3] - full ensemble
                    lattice_coords_3d=self.lattice_3d,
                    volume=self.fo,
                    mask3d=self.density_mask,    # [D,D,D] or [D^3]
                    elements=chain_elements,     # [B_ensembles, N] - ensemble elements
                    b_factors=chain_b_factors,   # [B_ensembles, N] - ensemble b-factors
                    voxel_size=float(self.pixel_size),
                    D=int(self.D),
                    steps=self.gradient_ascent_parameters["steps"],
                    lr_t_A=self.gradient_ascent_parameters["lr_t_A"] * float(self.pixel_size),
                    lr_r_deg=self.gradient_ascent_parameters["lr_r_deg"],
                    reduction=self.gradient_ascent_parameters["reduction"],
                    print_every=100,
                    per_step_t_cap_voxels=self.gradient_ascent_parameters["per_step_t_cap_voxels"],
                    Bfac=self.gradient_ascent_parameters["Bfac"],
                    bfactor_minimum=self.gradient_ascent_parameters["bfactor_minimum"],
                    n_random=self.gradient_ascent_parameters["n_random"],
                    seed=None,
                    return_all=False,
                    t_init_box_edge_voxels=self.gradient_ascent_parameters["t_init_box_edge_voxels"],
                    time=time,
                )
                R = res["best_R"][None].detach()
                T_global = res["T_global_best"][None, None].detach()
                x_0_hat_aligned = (x_0_hat @ R.transpose(-1, -2)) + T_global
                # x_0_hat_aligned = (R[:,None] @ x_0_hat[...,None] + T_global[..., None]).squeeze(-1)
            return x_0_hat_aligned, R, T_global
        
        elif global_alignment_strategy == "align_to_previous_and_density_momentum":
            with torch.enable_grad():
                res = blob_se3_align_adam_multi_start(
                        coords=x_0_hat[0, self.AF3_to_pdb_mask, :],
                        lattice_coords_3d=self.lattice_3d,
                        volume=self.fo,
                        mask3d=self.density_mask,    # [D,D,D] or [D^3]
                        voxel_size=float(self.pixel_size),
                        D=int(self.D),
                        steps=self.gradient_ascent_parameters["steps"],
                        lr_t_A=self.gradient_ascent_parameters["lr_t_A"] * float(self.pixel_size),
                        lr_r_deg=self.gradient_ascent_parameters["lr_r_deg"],
                        reduction=self.gradient_ascent_parameters["reduction"],
                        sampler_fn=interpolate_scalar_volume_at_points_fast_testing_corrected,
                        print_every=self.gradient_ascent_parameters["print_every"],
                        per_step_t_cap_voxels=self.gradient_ascent_parameters["per_step_t_cap_voxels"],
                        Bfac=self.gradient_ascent_parameters["Bfac"],
                        bfactor_minimum=self.gradient_ascent_parameters["bfactor_minimum"],
                        n_random=self.gradient_ascent_parameters["n_random"],
                        seed=None,
                        return_all=False,
                        t_init_box_edge_voxels=self.gradient_ascent_parameters["t_init_box_edge_voxels"],
                        time=time,
                    )
                R = res["best_R"][None].detach()
                T_global = res["T_global_best"][None, None].detach()
                    
        elif global_alignment_strategy == False:
            return x_0_hat, torch.eye(3).repeat(x_0_hat.shape[0], 1, 1).to(x_0_hat.device), torch.zeros((x_0_hat.shape[0], 3))[:,None,:].to(x_0_hat.device)

        if self.combinatorially_best_alignment == False:
            full_reordering_indices = torch.arange(x_0_hat.shape[1], device=x_0_hat.device)
        
        elif self.combinatorially_best_alignment == "combinatorics":
            full_reordering_indices = self.align_multimer_by_permutations(x_0_hat, align_to)

        elif self.combinatorially_best_alignment == "cost_matrix_hungarian":

            if self.reordering_counter >= self.reordering_every:
                for i in range(1):
                    full_reordering_indices = self.align_multimer_by_hungarian_algo(x_0_hat, align_to) # recompute the alignment + cost several times since prealignment cycle is done (to guarantee the best alignment possible)
                    self.full_reordering_indices = full_reordering_indices.clone() # save the last reordering indices for the next iteration
                    self.reordering_counter = 0
            else: 
                full_reordering_indices = self.full_reordering_indices
                self.reordering_counter += 1
        elif self.combinatorially_best_alignment == "density_to_chains_OT":
            for i in range(3):
                full_reordering_indices = self.align_multimer_by_OT_hungarian(x_0_hat, mask=self.AF3_to_pdb_mask)
            x_0_aligned, R, T = align_multimeric_protein_to_multimeric_density_by_chain(
                x_0_hat[:, full_reordering_indices, :],
                [ self.lattice_3d[mask.flatten()].unsqueeze(0) for mask in self.density_masks_per_chain ],
                self.element_gt[None, :, None],
                [ self.fo[mask][None, :, None] for mask in self.density_masks_per_chain ],
                reduced_protein_mask=self.AF3_to_pdb_mask,
            )
            return x_0_aligned, R, T
        else:
            raise ValueError(f"Unknown combinatorially_best_alignment value: {self.combinatorially_best_alignment}. Should be one of 'combinatorics', 'cost_matrix_hungarian' or False.")

        reordered_x_0_hat = x_0_hat[:, full_reordering_indices, :] # Reordering the x_0_hat according to the best permutation found

        # And then proceed with the reordering depending on the logic with ROI etc
        if self.guide_only_ROI or self.align_only_outside_ROI: 
                # Only take outside of ROI if explicitly said so or if only guiding by ROI
            r, aligned_reordered_x_0_hat, R, T = self_aligned_rmsd(
                reordered_x_0_hat, align_to, 
                (self.AF3_to_pdb_mask & (~self.regions_of_interest_mask) & self.align_to_chain_mask)  # consider this behaviour with should_align_to_chains
            )
        else: # Otherwise, align by the whole thing
            r, aligned_reordered_x_0_hat, R, T = self_aligned_rmsd(
                reordered_x_0_hat, align_to, 
                self.AF3_to_pdb_mask & self.align_to_chain_mask
            ) 

        return aligned_reordered_x_0_hat, R, T

    def __call__(self, x_0_hat:torch.Tensor, time, structures=None, i=None, step=None):
        # 0. Delete gradients of b-factors
        if self.optimize_b_factors:
            self.parameters_optimizer.zero_grad()
            if self.reapply_is_learnable is not False:
                self.reapply_b_factor.grad = None
        

        # 1. ALIGNMENT: aligning the region depending on ROI. Additionally, the alignment finds the best permutations per sequence 
        
        aligned_x_0_hat, R, T = self.align_structure(
            x_0_hat, self.coordinates_gt.unsqueeze(0),  i=i, step=step, use_saved_alignment=False, structures=structures, time=time
        ) 
        #if self.guide_only_ROI:
            # After alignment, replace the non-ROI atoms with the GT coordinates
        #    aligned_x_0_hat = self.replace_non_residue_range_atoms(aligned_x_0_hat)

        # 2. FO PART: Creating and masking the correct parts of the densities. In case of ROI guidance, compute only part of the density
        fc = self.calculate_ESP(
            x_0_hat = aligned_x_0_hat,
            use_Coloumb = self.use_Coloumb,
            rmax = self.rmax_for_esp, 
            should_align = False,
            atom_mask = self.regions_of_interest_mask if self.guide_only_ROI else None,
        ) 
        density_mask = self.density_zone_of_interest_mask if self.guide_only_ROI else self.density_mask # Limit ourselves to only plot and work with the ROI region
        
        fc_clone = fc.clone().detach() 

        fc = fc[density_mask] # extracting the masked regions in the densities for better comparison
        fo = self.fo[density_mask] 
        
        # 3. NORMALIZATION: Normalizing both to eliminate units issue (the quality of this depends on the quality of the mask. Visual inspection using plots below recommended)
        fo = (fo - fo.mean()) / (fo.std() + 1e-6) 
        fc = (fc - fc.mean()) / (fc.std() + 1e-6) 

        density_loss = (0.5*(fo - fc).abs()).mean() # L1 of the masked and normalized regions
        
        # 4: OT LOSS (if needed)
        rmsd_loss = (
            (
                aligned_x_0_hat[:, self.AF3_to_pdb_mask, :] - 
                self.coordinates_gt[self.AF3_to_pdb_mask, :].unsqueeze(0)
            ).square().sum(dim=-1)
        ).mean().sqrt()

        dihedrals_loss = self.compute_dihedrals_loss(aligned_x_0_hat) 

        optimal_transport_loss = self.compute_optimal_transport_loss(aligned_x_0_hat) # this also increments the counter for the sinkhorn loss function
        # REWRITING THE LOGIC OF COMBINED LOSS FUNCTIONS: THERE'S ONE 'MAIN' loss function. and the rest are 'secondary' losses that are equal to the portion of the main one.
        # Since in the CryoEM modality, denote the OT loss or the Density loss as the main loss depending on which is turned on
        # OT is the main if it is turned on. By default, ESP density + OT are not turned on simultaneously since there is no point; the second one is better for fine-level details due to being the direct forward model
        
        base_loss = density_loss if ((step is not None and step >= self.sinkhorn_parameters["turn_off_after"]) or (step is None and i >= self.sinkhorn_parameters["turn_off_after"]) or self.percentage_of_rmsd_loss == 0.0) else optimal_transport_loss
        dihedrals_loss_normalized = dihedrals_loss / dihedrals_loss.clone().detach().abs() * \
            base_loss.clone().detach().abs() if (dihedrals_loss is not None and dihedrals_loss != 0.0) else 0.0

        # If there are other losses, they are added here on top in the same way. The percentage of rmsd loss is not needed anymore
        loss = base_loss  + dihedrals_loss_normalized * self.dihedrals_parameters["dihedral_loss_weight"]

        # 5. VALUES FOR LOGGING 
        self.calculate_wandbi_logs(aligned_x_0_hat, loss, optimal_transport_loss=optimal_transport_loss, density_loss=density_loss, fc_clone=fc_clone, dihedrals_loss=dihedrals_loss)

        # 6. VIZUALIZATION: Save png evolution plots
        #with torch.no_grad():
        #    self.plot_evolution_plot_single_state(aligned_x_0_hat)
        #    if self.guide_only_ROI or self.align_only_outside_ROI: # Save plots for ROI context
        #        self.plot_ROI_evolution_plot_single_state(aligned_x_0_hat)

        # 7. RETURNING THE X_0_HAT THAT IS ALIGNED_BACK TO THE ORIGINAL STRUCTURE
        x_0_hat_aligned_back = (R.permute(0,2,1)[:,None] @ (aligned_x_0_hat - T)[...,None]).squeeze(-1)
        
        #if self.guide_only_ROI: # Currently disabled to avoid interfering with diffusion process
        #    x_0_hat.register_hook(lambda grad: grad * self.regions_of_interest_mask.unsqueeze(0).unsqueeze(-1))

        return loss, x_0_hat_aligned_back, None 

    def perform_symmetry_rotations(
        self, 
        x_0_hat: torch.Tensor, 
        i,
        step=None,
        rigid_R: torch.Tensor = None,
        rigid_T: torch.Tensor = None,
        is_counted_down: bool = True,
        apply_saved_transforms: bool = False,
        structures = None,
        time: float = 0.0,
    ) -> torch.Tensor:
        """
        Applies per-chain 180° flips (reflection across PCA axes) for 'custom_amyloid' symmetry.
        Chooses best flip based on Sinkhorn OT loss to matching density blob.
        Registers backward hooks to rotate gradients back to AF3's original coordinate frame.
        
        Returns:
            torch.Tensor: Modified x_0_hat with corrected symmetry and gradient hooks.
        """
        if not hasattr(self, "symmetry_parameters") or hasattr(self.symmetry_parameters, "symmetry_type") and self.symmetry_parameters["symmetry_type"] == None or hasattr(self.symmetry_parameters, "reapply_symmetry_every") and self.symmetry_parameters["reapply_symmetry_every"] == False:
            return x_0_hat
        
        #if is_counted_down:
        #    self.symmetry_parameters["counter"] += 1
        if self.symmetry_parameters["reapply_symmetry_every"] == True:
            1
        elif self.symmetry_parameters["reapply_symmetry_every"] == False:
            return x_0_hat
        elif step is not None and step not in self.symmetry_parameters["reapply_symmetry_every"]:
            return x_0_hat
        elif step is None and i not in self.symmetry_parameters["reapply_symmetry_every"]:
            return x_0_hat

        def perform_per_blob_alignment_symmetry(x_0_hat: torch.Tensor) -> torch.Tensor:
            """
            Perform alignment of chains per blobs. Must be done when chains are already well separated and aligned to blobs-ish according to 
            """
            transforms = {}  # Store transformations per chain
            
            for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                mask = self.AF3_to_pdb_mask & chain_mask
                if mask.sum() < 4:
                    continue

                coords_orig = x_0_hat[0, mask, :]
                _, R, T = align_protein_to_density_pca(
                    coords_orig[None], 
                    self.lattice_3d[self.density_masks_per_chain[chain_id].flatten()].unsqueeze(0),
                    self.element_gt[None, mask, None], 
                    self.fo[self.density_masks_per_chain[chain_id]][None, :, None],
                    #reduced_protein_mask=self.AF3_to_pdb_mask, # no reduced protein mask needed since already applied
                )
                #x_0_hat[0, mask, :] = coords_aligned # replace the coordinates with the aligned ones
                transforms[chain_id] = (R.detach(), T.detach())
            
            with torch.no_grad():
                for chain_id, (R, T) in transforms.items():
                    chain_mask = self.masks_per_sequence[chain_id]
                    x_0_hat[0, chain_mask, :] = (
                    #(R.permute(0, 2, 1) @ (x_0_hat[0, self.masks_per_sequence[chain_id], :] - T)[..., None]).squeeze(-1)
                        ( R @ x_0_hat[0, chain_mask].T[None] ).transpose(-1,-2) + T
                    )
            #x_0_hat.register_hook(lambda grad: grad * 0) # no gradient in this case since heavily changed by the alignment

            return x_0_hat

        def perform_per_blob_alignment_symmetry_gradient_ascent(x_0_hat: torch.Tensor) -> torch.Tensor:
            """
            Per-chain rigid alignment to EM density (blob) via multi-start gradient-ascent.
            Writes transforms per chain and applies them to all atoms of that chain.
            """
            transforms = {}

            with torch.enable_grad():
                for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                    mask = self.AF3_to_pdb_mask & chain_mask
                    if mask.sum().item() < 4:
                        continue

                    # Run the centroid-prealigned multi-start fit for this chain
                    with torch.enable_grad():
                        res = blob_se3_align_adam_multi_start(
                            coords=x_0_hat[0, mask],                          # [N_atoms, 3] for this chain
                            lattice_coords_3d=self.lattice_3d,
                            volume=self.fo,
                            mask3d=self.density_masks_per_chain[chain_id],    # [D,D,D] or [D^3]
                            voxel_size=float(self.pixel_size),
                            D=int(self.D),
                            steps=self.gradient_ascent_parameters["steps"],
                            lr_t_A=self.gradient_ascent_parameters["lr_t_A"] * float(self.pixel_size),
                            lr_r_deg=self.gradient_ascent_parameters["lr_r_deg"],
                            reduction=self.gradient_ascent_parameters["reduction"],
                            sampler_fn=interpolate_scalar_volume_at_points_fast_testing_corrected,
                            print_every=100,
                            per_step_t_cap_voxels=self.gradient_ascent_parameters["per_step_t_cap_voxels"],
                            Bfac=self.gradient_ascent_parameters["Bfac"],
                            bfactor_minimum=self.gradient_ascent_parameters["bfactor_minimum"],
                            n_random=self.gradient_ascent_parameters["n_random"],
                            seed=None,
                            return_all=False,
                            time=time,
                            t_init_box_edge_voxels=self.gradient_ascent_parameters["t_init_box_edge_voxels"],
                        )
                    #print(res["best_batch_index"])
                    R = res["best_R"]                         # [3,3]
                    T_global = res["T_global_best"]           # row-vector form: use with x @ R.T + T_global
                    transforms[chain_id] = (R.detach(), T_global.detach())

            # Apply the rigid transforms to ALL atoms of each chain (row-vector convention)
            with torch.no_grad():
                for chain_id, (R, T_global) in transforms.items():
                    cmask = self.masks_per_sequence[chain_id]
                    x_0_hat[0, cmask, :] = x_0_hat[0, cmask] @ R.T + T_global # ✔ row form
                    R = R[None]
                    T_global = T_global[None, None]
                    if structures is not None:
                        assert rigid_R is not None and rigid_T is not None, "Rigid transformation is required to align structures"
                        structures[0, cmask] = (
                            ( rigid_R.transpose(-1,-2) @ R @ rigid_R @ structures[0, cmask].T[None] ).transpose(-1, -2) +
                            ( rigid_R.transpose(-1,-2) @ (
                                T_global + (R @ rigid_T.transpose(-1,-2)).transpose(-1,-2) - rigid_T
                            ).transpose(-1,-2) ).transpose(-1,-2)
                        )


            return x_0_hat

        def perform_batched_density_alignment(x_0_hat: torch.Tensor) -> torch.Tensor:
            """
            Per-chain rigid alignment using batched density calculation with L1 loss objective.
            Uses your optimized compute_elden_for_density_calculation_batched approach.
            """
            from src.protenix.metrics.rmsd import batched_density_se3_align_adam_multi_start
            
            transforms = {}
            
            with torch.enable_grad():
                for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                    mask = self.AF3_to_pdb_mask & chain_mask
                    if mask.sum().item() < 4:
                        continue

                    # Prepare ensemble coordinates for this chain from x_0_hat batch dimension
                    B_ensembles = x_0_hat.shape[0]
                    chain_ensemble_coords = x_0_hat[:, mask, :]  # [B_ensembles, N_chain, 3]
                    
                    # Prepare elements and b-factors for each ensemble member of this chain
                    chain_elements = self.element_gt[mask].unsqueeze(0).expand(B_ensembles, -1)  # [B_ensembles, N_chain]
                    
                    # Use trainable b-factors if available, otherwise use default
                    if hasattr(self, 'bfactor_gt') and self.bfactor_gt is not None:
                        chain_b_factors = self.bfactor_gt[mask].unsqueeze(0).expand(B_ensembles, -1)  # [B_ensembles, N_chain] - TRAINABLE!
                    else:
                        chain_b_factors = torch.ones_like(chain_elements, dtype=torch.float32) * self.global_b_factor  # [B_ensembles, N_chain] - fallback
                    
                    # Run the batched density alignment
                    with torch.enable_grad():
                        res = batched_density_se3_align_adam_multi_start(
                            coords=chain_ensemble_coords,                     # [B_ensembles, N_chain, 3] - ensemble for this chain
                            lattice_coords_3d=self.lattice_3d,
                            volume=self.fo,
                            mask3d=self.density_masks_per_chain[chain_id],    # [D,D,D] or [D^3]
                            elements=chain_elements,                          # [B_ensembles, N]
                            b_factors=chain_b_factors,                        # [B_ensembles, N]
                            voxel_size=float(self.pixel_size),
                            D=int(self.D),
                            steps=self.gradient_ascent_parameters["steps"],
                            lr_t_A=self.gradient_ascent_parameters["lr_t_A"] * float(self.pixel_size),
                            lr_r_deg=self.gradient_ascent_parameters["lr_r_deg"],
                            reduction=self.gradient_ascent_parameters["reduction"],
                            print_every=100,
                            per_step_t_cap_voxels=self.gradient_ascent_parameters["per_step_t_cap_voxels"],
                            Bfac=self.gradient_ascent_parameters["Bfac"],
                            n_random=self.gradient_ascent_parameters["n_random"],
                            seed=None,
                            return_all=False,
                            time=time,
                            bfactor_minimum=self.gradient_ascent_parameters["bfactor_minimum"],
                            t_init_box_edge_voxels=self.gradient_ascent_parameters["t_init_box_edge_voxels"],
                        )
                    
                    R = res["best_R"]                         # [3,3]
                    T_global = res["T_global_best"]           # row-vector form: use with x @ R.T + T_global
                    transforms[chain_id] = (R.detach(), T_global.detach())

            # Apply the rigid transforms to ALL atoms of each chain (row-vector convention)
            with torch.no_grad():
                for chain_id, (R, T_global) in transforms.items():
                    cmask = self.masks_per_sequence[chain_id]
                    x_0_hat[0, cmask, :] = x_0_hat[0, cmask] @ R.T + T_global # ✔ row form
                    R = R[None]
                    T_global = T_global[None, None]
                    if structures is not None:
                        assert rigid_R is not None and rigid_T is not None, "Rigid transformation is required to align structures"
                        structures[0, cmask] = (
                            ( rigid_R.transpose(-1,-2) @ R @ rigid_R @ structures[0, cmask].T[None] ).transpose(-1, -2) +
                            ( rigid_R.transpose(-1,-2) @ (
                                T_global + (R @ rigid_T.transpose(-1,-2)).transpose(-1,-2) - rigid_T
                            ).transpose(-1,-2) ).transpose(-1,-2)
                        )

            return x_0_hat

        def perform_per_blob_alignment_symmetry_gradient_ascent_batched(x_0_hat: torch.Tensor) -> torch.Tensor:
            """
            Per-chain rigid alignment to EM density via a single batched multi-start gradient-ascent.
            This version is highly optimized to run all chains in parallel.
            It is a nested function and accesses class members via `self` from its parent scope.
            """
            coords_list, density_masks_list, chain_masks_to_update = [], [], []
            
            # Gather data from the parent's `self` context
            with torch.no_grad():
                for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                    mask = self.AF3_to_pdb_mask & chain_mask
                    if mask.sum().item() < 4:
                        continue
                    coords_list.append(x_0_hat[0, mask])
                    density_masks_list.append(self.density_masks_per_chain[chain_id])
                    chain_masks_to_update.append(chain_mask)

            if not coords_list:
                return x_0_hat

            # Call the batched alignment logic, passing necessary parameters from `self`
            results = run_batched_alignment(
                coords_list=coords_list,
                density_masks_list=density_masks_list,
                lattice_coords_3d=self.lattice_3d,
                volume=self.fo,
                voxel_size=float(self.pixel_size),
                D=int(self.D),
                sampler_fn=interpolate_scalar_volume_at_points_fast_testing_corrected, # Or self.sampler_fn
                n_random=1_500,
                start_batch_size=100,
                # Other kwargs like steps, lr, Bfac, etc. are passed here
            )

            # Apply the results, accessing `structures` and `rigid_R`/`T` from the parent scope
            with torch.no_grad():
                best_Rs, best_T_globals = results["best_R_b"], results["T_global_best_b"]
                for i, cmask in enumerate(chain_masks_to_update):
                    R, T_global = best_Rs[i], best_T_globals[i]
                    x_0_hat[0, cmask, :] = x_0_hat[0, cmask] @ R.T + T_global
                    
                    if structures is not None:
                        # Original logic for updating structures
                        assert rigid_R is not None and rigid_T is not None, "Rigid transformation is required"
                        R_exp, T_global_exp = R[None], T_global[None, None]
                        device = R.device
                        eye = torch.eye(3, device=device)[None]
                        structures[0, cmask] = (
                            (rigid_R.T @ R_exp @ rigid_R @ structures[0, cmask].T[None]).transpose(-1, -2) +
                            (rigid_R.T @ (R_exp - eye) @ rigid_T.T).transpose(-1, -2) +
                            (rigid_R.T @ T_global_exp.T).transpose(-1, -2)
                        )

            return x_0_hat

        def perform_global_axis_chainwise_flipping(x_0_hat: torch.Tensor) -> torch.Tensor:
            """
            Reflects each chain across the global PCA axis (1st PC of full x_0_hat), 
            centered at the chain's own centroid — so chain positions remain fixed.
            Gradients are transformed back via registered hooks.
            """
            grad_rotations = {}

            def flip_coordinates_around_axis(coords, mu, global_axis):
                """
                Reflects coords around the plane orthogonal to `global_axis`, centered at mu.
                """
                axis = global_axis / global_axis.norm()
                R = torch.eye(3, device=coords.device) - 2 * axis[:, None] @ axis[None, :]
                return (R @ (coords - mu).T).T + mu, R

            # -- Compute first PCA axis globally over the whole x_0_hat (mask-aware) --
            with torch.no_grad():
                global_mask = self.AF3_to_pdb_mask
                coords_global = x_0_hat[0, global_mask, :]  # [N, 3]
                coords_global_centered = coords_global - coords_global.mean(dim=0, keepdim=True)
                cov = coords_global_centered.T @ coords_global_centered / coords_global_centered.shape[0]
                U, _, _ = torch.linalg.svd(cov)
                global_pc1 = U[:, 0]  # First principal axis

                # -- Flip per chain using the same global axis --
                for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                    mask = self.AF3_to_pdb_mask & chain_mask
                    if mask.sum() < 4:
                        continue

                    coords_chain = x_0_hat[0, mask, :]  # [N_chain, 3]
                    mu = coords_chain.mean(dim=0, keepdim=True)

                    coords_flipped, R = flip_coordinates_around_axis(coords_chain, mu, global_pc1)

                    weights_atoms = self.element_gt[mask].clamp(min=1e-6)
                    weights_atoms = weights_atoms / weights_atoms.sum()

                    voxel_mask = self.density_masks_per_chain[chain_id]
                    Y = self.lattice_3d[voxel_mask.flatten(), :]
                    weightsY = (self.fo[voxel_mask] - self.fo[voxel_mask].min()).clamp(min=1e-6)
                    weightsY = weightsY / weightsY.sum()

                    loss_fn = self.sinkhorn_loss_function
                    loss_orig = loss_fn(weights_atoms[None, :], coords_chain[None, :], weightsY[None, :], Y[None, :])
                    loss_flip = loss_fn(weights_atoms[None, :], coords_flipped[None, :], weightsY[None, :], Y[None, :])

                    if loss_flip.item() < loss_orig.item():
                        x_0_hat[0, mask, :] = coords_flipped
                        grad_rotations[chain_id] = R.T  # Register backward transformation

            # --- Hook: rotate gradients back to unflipped frame ---
            def rotate_gradients_hook(grad: torch.Tensor) -> torch.Tensor:
                grad = grad.clone()
                for chain_id, R_T in grad_rotations.items():
                    mask = self.AF3_to_pdb_mask & self.masks_per_sequence[chain_id]
                    grad_chain = grad[0, mask, :]
                    grad[0, mask, :] = (R_T @ grad_chain.T).T
                return grad

            x_0_hat.register_hook(rotate_gradients_hook)
            return x_0_hat

        def per_chain_atomwise_rigid_alignment(x_0_hat) -> torch.Tensor:
            """
            Performs atom-to-atom rigid alignment (via RMSD) per chain to GT structure.
            Only the aligned version is returned — no gradient flow needed.
            """
            transforms = {}

            with torch.no_grad():
                for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                    mask = self.AF3_to_pdb_mask & chain_mask
                    if mask.sum() < 4:
                        continue

                    pred = x_0_hat[0, mask, :]                   # shape: [N_atoms, 3]
                    gt = self.coordinates_gt[mask, :]            # shape: [N_atoms, 3]

                    _, _, R, T = self_aligned_rmsd(
                        pred_pose=pred[None],
                        true_pose=gt[None],
                        atom_mask=torch.ones(pred.shape[0], device=pred.device) # no atom mask is needed since already masked..!
                    )
                    transforms[chain_id] = (R.detach(), T.detach())
            
            # Note: this no grad is to be experimented with, but apparently it is better like this 
            with torch.no_grad(): 
                for chain_id, (R, T) in transforms.items():
                    chain_mask = self.masks_per_sequence[chain_id]
                    x_0_hat[0, chain_mask, :] = (
                    #(R.permute(0, 2, 1) @ (x_0_hat[0, self.masks_per_sequence[chain_id], :] - T)[..., None]).squeeze(-1)
                        ( R @ x_0_hat[0, chain_mask].T[None] ).transpose(-1,-2) + T
                    )
                    # the structures have to be fixed in a similar fashion. The relative transformation is applied to structures (but in its original frame)
                    if structures is not None:
                        assert rigid_R is not None and rigid_T is not None, "Rigid transformation is required to align structures"
                        structures[0, chain_mask] = (
                            ( rigid_R.transpose(-1,-2) @ R @ rigid_R @ structures[0, chain_mask].T[None] ).transpose(-1, -2) + 
                            ( (rigid_R.transpose(-1,-2) @ (R - torch.eye(3, device=self.device)[None])) @ rigid_T.transpose(-1,-2)).transpose(-1,-2) + (rigid_R.transpose(-1,-2) @ T.transpose(-1,-2)).transpose(-1,-2)
                        )

            return x_0_hat

        def perform_per_blob_centroid_alignment(x_0_hat) -> torch.Tensor:
            B = x_0_hat.shape[0]
            x_0_hat_centroids = (
                x_0_hat[:, self.AF3_to_pdb_mask, :] * self.element_gt[None, self.AF3_to_pdb_mask, None]
            ).reshape(B, len(self.full_sequences), -1, 3).sum(dim=2, keepdim=True) / \
                self.element_gt[self.AF3_to_pdb_mask].reshape(1, len(self.full_sequences), -1, 1).sum(dim=2, keepdim=True)

            blob_centroids = torch.stack([
                (self.lattice_3d[mask.flatten(), :] * (self.fo[mask] - self.fo[mask].min())[..., None]).sum(dim=0) / (self.fo[mask] - self.fo[mask].min()).sum() 
                for mask in self.density_masks_per_chain
            ])[None, :, None, :]

            with torch.no_grad():
                aligned_x_0_hat = (
                    x_0_hat.reshape(B, len(self.full_sequences), -1, 3) - x_0_hat_centroids.detach() + blob_centroids.detach()
                ).reshape(B,-1,3)    

            return aligned_x_0_hat

        def align_to_previous_by_chains(x_0_hat):
            with torch.no_grad():
                if getattr(self, 'previous_coordinates_symmetry', None) is not None:
                    for chain_id, chain_mask in enumerate(self.masks_per_sequence):
                        mask = self.AF3_to_pdb_mask & chain_mask 
                        if mask.sum() < 4:
                            continue

                        pred = x_0_hat[0, mask, :]                   # shape: [N_atoms, 3]
                        gt = self.previous_coordinates_symmetry[0, mask, :]            # shape: [N_atoms, 3]

                        _, _, R, T = self_aligned_rmsd(
                            pred_pose=pred[None],
                            true_pose=gt[None],
                            atom_mask=torch.ones(pred.shape[0], device=pred.device) # no atom mask is needed since already masked..!
                        )
                        R = R.detach(); T = T.detach()
                        x_0_hat[0, chain_mask, :] = (R @ x_0_hat[0, chain_mask].T[None]).transpose(-1,-2) + T
                    self.previous_coordinates_symmetry = x_0_hat.clone().detach()
                else: 
                    x_0_hat = perform_per_blob_alignment_symmetry(x_0_hat)
                    self.previous_coordinates_symmetry = x_0_hat.clone().detach()
            
            return x_0_hat


            x_0_hat = perform_flipping_symmetry(x_0_hat)
        
        if self.symmetry_parameters["symmetry_type"] == "per_blob_alignment":
            x_0_hat = perform_per_blob_alignment_symmetry(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "global_axis_chainwise_flipping":
            x_0_hat = perform_global_axis_chainwise_flipping(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "debug_per_chain_alignment":
            x_0_hat = per_chain_atomwise_rigid_alignment(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "per_blob_gradient_alignment":
            x_0_hat = perform_per_blob_alignment_symmetry_gradient_ascent(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "per_blob_centroid_alignment":
            x_0_hat = perform_per_blob_centroid_alignment(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "previous_coordinates_symmetry":
            x_0_hat = align_to_previous_by_chains(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "debug_per_chain_alignment_batched":
            x_0_hat = perform_per_blob_alignment_symmetry_gradient_ascent_batched(x_0_hat)
        elif self.symmetry_parameters["symmetry_type"] == "batched_density_alignment":
            x_0_hat = perform_batched_density_alignment(x_0_hat)
        else:
            raise ValueError(f"Unknown symmetry type: {self.symmetry_parameters['symmetry_type']}. Supported: 'custom_amyloid', 'per_blob_alignment', 'global_axis_chainwise_flipping', 'per_chain_atomwise_rigid_alignment', 'batched_density_alignment'.")

        return x_0_hat

    def compute_optimal_transport_loss(self, aligned_x_0_hat):
    
        # Here, the OT loss is not computed differently depending on whether guidance is by the full volume or individual blobs of chains (amyloid-like structures with separated chains)

        self.sinkhorn_parameters["counter"] += 1 # increment the counter for the sink
        #mask_for_OT = torch.ones_like(self.AF3_to_pdb_mask, device=self.device, dtype=torch.bool)
        mask_for_OT = self.AF3_to_pdb_mask

        if not self.sinkhorn_parameters["guide_multimer_by_chains"]:
            X = aligned_x_0_hat[:, mask_for_OT, :].reshape(-1, 3) 
            density_mask_for_OT = self.density_mask.flatten()

            Y = self.lattice_3d[density_mask_for_OT, :] # voxel centers
            
            weightsX = self.element_gt[mask_for_OT] \
                .repeat( aligned_x_0_hat.shape[0]) # repeating Batch times..!
            weightsX = (weightsX / weightsX.sum(dim=-1)).flatten() # normalizing such that they add to one..!
             # for now leave those as default params

            weightsY = self.fo_unthresholded.flatten()[density_mask_for_OT] - self.fo_unthresholded.flatten()[density_mask_for_OT].min()   
            weightsY = (weightsY / weightsY.sum())#.unsqueeze(0) # normalize the weights to sum to 1 -> condition for the weights
        else: 
            # Realign by the blob centers 
            # Aligned x_0_hat is properly aligned in this example             
            #perm = self.align_multimer_by_OT_hungarian(aligned_x_0_hat) # this returns the best permutation of the chains in the multimer
            X = aligned_x_0_hat[:, mask_for_OT] # reordering the X according to the best permutation found
            # Align X to Y such that the loss is computed correctly
            X = X.reshape(len(self.masks_per_sequence), -1, 3) # since the atomic order is taken care of, it is fine by default
            weightsX = self.element_gt[mask_for_OT].reshape(len(self.masks_per_sequence), -1) 
            weightsX = (weightsX / weightsX.sum(dim=-1, keepdim=True))

            Y, weightsY = self.pad_per_chain_densities_to_one_tensor_for_OT()
        
        #weightsX = (self.element_gt[mask_for_OT] * (torch.pi*4 / self.bfactor_gt[mask_for_OT]).pow(3/2) ) \
        # For now no b-factor weighting to avoid signal degradation
        optimal_transport_loss = self.sinkhorn_loss_function(
            weightsX, X, weightsY, Y
        ) 
        if self.sinkhorn_parameters["guide_multimer_by_chains"]:
            optimal_transport_loss = optimal_transport_loss.sum() # now add them together 

        return optimal_transport_loss

    def pad_per_chain_densities_to_one_tensor_for_OT(self):
        lengths = [mask.sum() for mask in self.density_masks_per_chain]
        max_length = max(lengths)
        
        Ys = [
            torch.cat([
                    self.lattice_3d[mask.flatten(), :], torch.zeros((max_length - mask.sum(), 3), dtype=torch.float32, device=mask.device)
                ])
            for mask in self.density_masks_per_chain
        ]
        weightsYs = [
            torch.cat([
                self.fo[mask] - self.fo[mask].min(), 
                torch.zeros(max_length - mask.sum(), dtype=torch.float32, device=mask.device)
            ])
            for mask in self.density_masks_per_chain
        ]
        weightsYs = [weightsY / weightsY.sum() for weightsY in weightsYs] # normalizing the weights to sum to 1 -> condition for the weights

        return torch.stack(Ys), torch.stack(weightsYs)

    def compute_dihedrals_loss(self, x_0_hat):
        if self.dihedrals_parameters["use_dihedrals"] == False:
            return torch.tensor(0.0, device=self.device) # or better yet, None
        elif self.dihedrals_parameters["use_dihedrals"] == "from_gt":
            # This is a temporary loss function to compute the backbone loss
            # It is replaced with a proper loss function later on
            backbones_c = [
                backbone_dihedrals(
                    *[x_0_hat[0, backbone_mask & self.AF3_to_pdb_mask & chain_mask, :].unsqueeze(0)
                    for backbone_mask in self.backbone_masks],
                ) for chain_mask in self.masks_per_sequence
            ]

            backbones_o = [
                backbone_dihedrals(
                    *[self.coordinates_gt[mask & self.AF3_to_pdb_mask & chain_mask, :].unsqueeze(0) 
                    for mask in self.backbone_masks],
                ) for chain_mask in self.masks_per_sequence
            ]

            backbones_loss = [
                angle_diff(backbones_c[i][:, :, angle_idx], backbones_o[i][:, :, angle_idx]).square().mean()
                for i in range(len(backbones_c))
                for angle_idx in range(2)  # 0 for phi, 1 for psi
            ]
            return torch.stack(backbones_loss).mean()
        elif self.dihedrals_parameters["use_dihedrals"] == "from_nmr":
            # Per-chain application is needed; simplified here for now.
            # CRITICAL: Convert angles to radians, but keep half-widths in degrees for proper loss scaling
            phi_deg, psi_deg, dphi_deg, dpsi_deg = self.dihedrals_tensor.T.split(1)
            phi = phi_deg * torch.pi / 180.0  # Convert angles to radians
            psi = psi_deg * torch.pi / 180.0  # Convert angles to radians
            dphi = dphi_deg * torch.pi / 180.0  # Convert half-widths to radians to match angle units
            dpsi = dpsi_deg * torch.pi / 180.0  # Convert half-widths to radians to match angle units
            
            backbones_c = [
                backbone_dihedrals(
                    *[x_0_hat[:, backbone_mask & chain_mask, :]
                    for backbone_mask in self.backbone_masks],
                ) for chain_mask in self.masks_per_sequence
            ]
            
            dihedrals_loss_phi = torch.stack([
                #angle_diff(backbones_c[i][0], phi[0,1:len(backbones_c[i][0])+1]).square()[self.dihedrals_mask[1:len(backbones_c[i][0])+1]] / dphi[0,1:len(backbones_c[i][0])+1][self.dihedrals_mask[1:len(backbones_c[i][0])+1]].square()
                    angle_diff ( backbones_c[i][:,:,0], phi ).square()[:, self.dihedrals_mask] / dphi[:, self.dihedrals_mask].square()
                for i in range(len(self.masks_per_sequence))
            ])
            dihedrals_loss_psi = torch.stack([
                #angle_diff(backbones_c[i][0], phi[0,1:len(backbones_c[i][0])+1]).square()[self.dihedrals_mask[1:len(backbones_c[i][0])+1]] / dphi[0,1:len(backbones_c[i][0])+1][self.dihedrals_mask[1:len(backbones_c[i][0])+1]].square()
                    angle_diff ( backbones_c[i][:,:,1], psi ).square()[:, self.dihedrals_mask] / dpsi[:, self.dihedrals_mask].square()
                for i in range(len(self.masks_per_sequence))
            ])

            dihedrals_loss = (dihedrals_loss_phi.sum() + dihedrals_loss_psi.sum()) / (dihedrals_loss_phi.numel() + dihedrals_loss_psi.numel())
            
            return dihedrals_loss

        else: 
            raise ValueError(f"Unknown dihedrals_parameters['use_dihedrals'] value: {self.dihedrals_parameters['use_dihedrals']}. Should be one of 'from_gt', 'from_x_0_hat' or False.")


    def calculate_wandbi_logs(self, aligned_x_0_hat, loss, optimal_transport_loss, density_loss, fc_clone, dihedrals_loss):
        self.last_loss_value = loss.item()
        #aligned_reordered_x_0_hat_for_rmsd = self.align_structure(
        #    aligned_x_0_hat, self.coordinates_gt.unsqueeze(0), global_alignment_strategy="global_rmsd_to_gt"
        #)[0] # align the x_0_hat to the ground truth structure to compute the RMSD correctly
        if self.guide_only_ROI or self.align_only_outside_ROI:
            self.last_rmsd_value = (
                aligned_x_0_hat[:, self.AF3_to_pdb_mask & (~self.regions_of_interest_mask), :] - \
                self.coordinates_gt[self.AF3_to_pdb_mask & (~self.regions_of_interest_mask), :].unsqueeze(0)
            ).square().sum(dim=-1).mean().sqrt().item()
        else:
            self.last_rmsd_value = (
                aligned_x_0_hat[:, self.AF3_to_pdb_mask, :] - \
                self.coordinates_gt[self.AF3_to_pdb_mask, :].unsqueeze(0)
            ).square().sum(dim=-1).mean().sqrt().item()
        
        self.last_OT_value = optimal_transport_loss.item() if self.percentage_of_rmsd_loss != 0.0 else None 
        #self.last_bond_length_value = self.get_bond_loss(aligned_x_0_hat).item() / x_0_hat.shape[0] # this gets reported elsewhere 
        self.last_cosine_similarity = torch.nn.functional.cosine_similarity(
            torch.where(self.density_mask, self.fo, 0).flatten(), 
            torch.where(self.density_mask, fc_clone, 0).flatten(), dim=0 
            #torch.where(self.density_mask if len(self.full_sequences) > 1 else self.fo_threshold_mask, self.fo, 0).flatten(), 
            #torch.where(self.density_mask if len(self.full_sequences) > 1 else torch.ones_like(self.density_mask), fc_clone, 0).flatten(), dim=0 # only comparing volumes inside the region
        ).item() 
        self.last_density_loss_value = density_loss.item()

        self.last_dihedrals_loss_value = dihedrals_loss.sqrt().item() if self.dihedrals_parameters["use_dihedrals"] else None

        # Cosine similarities in case only ROI is chosen (have to recompute etc..!)
        if self.guide_only_ROI or self.align_only_outside_ROI:
            self.last_cosine_similarity_ROI = torch.nn.functional.cosine_similarity(
                torch.where(self.density_zone_of_interest_mask, self.fo, 0).flatten(), 
                torch.where(self.density_zone_of_interest_mask, fc_clone, 0).flatten(), dim=0
            ).item() 
            self.last_cosine_similarity = torch.nn.functional.cosine_similarity(
                torch.where(
                    self.density_mask, 
                    self.fo, 
                    0
                ).flatten(), 
                torch.where(
                    self.density_mask, 
                    self.calculate_ESP(
                        aligned_x_0_hat, use_Coloumb=self.use_Coloumb,
                        rmax=self.rmax_for_esp, should_align=False
                    ), # replacing fc_clone with a full computation of fc since in this mode it's never been computed..!
                    0 
                ).flatten(), dim=0
            ).item() 

    def save_state(
        self, 
        structures, 
        save_path,
        phenix_manager,
        skip_png: bool = False,
        b_factor_lr: float = 1.0,
        use_zero_b_values: bool = False,
        should_always_fit_gt: bool = False,
        n_iterations: int = 250,
        bfactor_min: float = 80.0,
        bfactor_max: float = 400.0,
        use_cross_correlation: bool = False,
        bfactor_regularization: float = 0.01,
    ):
        mrc_to_save = gemmi.read_ccp4_map(self.esp_file)

        fc_final = self.calculate_ESP(
            structures, 
            use_Coloumb=self.use_Coloumb, rmax=self.rmax_for_esp, full_grid=True,
        )

        fc_from_gt = self.calculate_ESP(
            self.coordinates_gt.unsqueeze(0), use_Coloumb=self.use_Coloumb,
            should_align=False, rmax=self.rmax_for_esp, full_grid=True,
            bfactor=self.bfactor_gt_untouched.to(device=self.device) if (self.bfactor_gt > 10.0).all() else torch.ones_like(self.bfactor_gt_untouched, device=self.device) * self.global_b_factor
        ) 

        mrc_to_save.grid.array[:] = fc_final.detach().cpu().numpy()
        mrc_to_save.write_ccp4_map(f"{save_path}/fc_from_guidance.mrc")
        
        mrc_to_save.grid.array[:] = fft_upsample_3d(self.fo.cpu(), out_shape= (self.D_full,)*3 ).detach().numpy() # save the fo that was deblurred
        mrc_to_save.write_ccp4_map(f"{save_path}/fo.mrc") # saving fc and fo to compare them later..!
        
        mrc_to_save.grid.array[:] = fc_from_gt.detach().cpu().numpy()
        mrc_to_save.write_ccp4_map(f"{save_path}/fc_from_gt.mrc") # Also saving the fc from the gt coordinates

        # Check if dealing with unguided structures and fit B-factors
        # Updated logic: protenix (or legacy af3) = unguided, anything else = guided
        base_name = os.path.basename(self.save_folder) if self.save_folder is not None else ""
        is_unguided = self.save_folder is not None and ("protenix" in base_name or "af3" in base_name)
        is_guided = self.save_folder is not None and not ("protenix" in base_name or "af3" in base_name)
        
        if is_unguided or is_guided:
            if is_guided:
                print("Detected guided structure - fitting B-factors with current guided b-factors as initialization...")
                # Use current guided b-factors as initialization
                initial_bfactors = self.bfactor_gt.clone().detach()
            else:
                print("Detected unguided structure - fitting B-factors...")
                # Use global b-factor as initialization
                initial_bfactors = torch.ones_like(self.bfactor_gt, device=self.device) * self.global_b_factor
            
            self.fit_bfactors_for_structures(
                structures, 
                save_path, 
                initial_bfactors=initial_bfactors,
                b_factor_lr=b_factor_lr,
                use_zero_b_values=use_zero_b_values,
                n_iterations=n_iterations,
                bfactor_min=bfactor_min,
                bfactor_max=bfactor_max,
                use_cross_correlation=use_cross_correlation,
                bfactor_regularization=bfactor_regularization
            )

        # Additionally, turn the .png plots into the .gif
        if not skip_png:
            self.convert_pngs_to_gif_plot(plot_folder="evolution_plots", save_gif_name="evolution.gif") 
            if self.guide_only_ROI or self.align_only_outside_ROI:
                self.convert_pngs_to_gif_plot(plot_folder="ROI_evolution_plots", save_gif_name="ROI_evolution.gif")
            
        # Run phenix for evaluations. Run over a loop of all the pdbs in the save_path (they were saved by now)
        gt_filename = "ground_truth_from_coordinates.pdb"  # Exclude the GT file from guided structure evaluation
        for pdb_file in os.listdir(save_path):
            if pdb_file.endswith(".pdb") and \
                not pdb_file.endswith("_rscc_painted.pdb") and \
                not os.path.basename(pdb_file) == os.path.basename(self.reference_pdb_path) and \
                pdb_file != gt_filename:   # Also exclude the saved GT file
                self.run_phenix_eval(os.path.join(save_path, pdb_file), save_path, phenix_manager)
        
        # Save GT coordinates as PDB and run evaluation on it (instead of reference file)
        # This ensures size consistency between GT and guided structures
        gt_pdb_path = self.save_gt_coordinates_as_pdb(
            save_path, 
            b_factor_lr=b_factor_lr, 
            use_zero_b_values=use_zero_b_values,
            should_always_fit_gt=should_always_fit_gt,
            n_iterations=n_iterations,
            use_cross_correlation=use_cross_correlation,
            bfactor_regularization=bfactor_regularization,
            bfactor_min=bfactor_min,
            bfactor_max=bfactor_max
        )
        print(f"Running phenix evaluation on saved GT: {gt_pdb_path}")
        self.run_phenix_eval(gt_pdb_path, save_path, phenix_manager)
        
        # Store the GT path for consistent use in all evaluations
        self._current_gt_pdb_path = gt_pdb_path
        
        # Debug: List all files in save_path to see what was created
        print("Files in save_path after evaluations:")
        for f in sorted(os.listdir(save_path)):
            if "cc_per_residue.log" in f:
                print(f"  CC log file: {f}")
        
        # Generate comparison plots after all evaluations are done
        self.plot_cc_vs_residue_comparison(save_path)
        self.plot_dihedral_comparison(save_path)
    
    def fit_bfactors_for_structures(
        self, 
        structures, 
        save_path=None, 
        initial_bfactors=None, 
        n_iterations=None, 
        b_factor_lr=1.0,
        use_zero_b_values=False,
        bfactor_min=None,
        bfactor_max=None,
        use_cross_correlation=False,
        bfactor_regularization=0.01
    ):
        """
        Unified B-factor fitting function using gradient descent on L1 loss or cross-correlation.
        This function optimizes B-factors to match the experimental density within the density mask.
        
        Args:
            structures: The structures to fit b-factors for [batch_size, n_atoms, 3]
                       Should already be in the correct reference frame (aligned to density map)
            save_path: Optional path to save the fitted structures. If None, only returns fitted B-factors.
            initial_bfactors: Optional tensor to initialize b-factors with. If None, uses global_b_factor.
            n_iterations: Optional number of iterations. If None, uses defaults (150 for normal, 250 for GT).
            b_factor_lr: Learning rate for B-factor optimization
            use_zero_b_values: Whether to use zero B-values for initialization
            bfactor_min: Minimum B-factor value for clamping (default: 60.0)
            bfactor_max: Maximum B-factor value for clamping (default: 400.0)
            use_cross_correlation: If True, use cross-correlation (Pearson) instead of L1 loss (default: False)
            bfactor_regularization: Penalty to discourage very small B-factors and encourage higher ones (default: 0.01)
            
        Returns:
            torch.Tensor: Fitted B-factors of shape [n_atoms]
        """
        # Set default values if None
        if bfactor_min is None:
            bfactor_min = 60.0
        if bfactor_max is None:
            bfactor_max = 400.0
            
        reduction_type = "cross-correlation" if use_cross_correlation else "L1 loss"
        print(f"Starting B-factor fitting for structures using {reduction_type}...")
        
        # Structures are already in the correct reference frame - no alignment needed
        aligned_structures = structures
        
        # Initialize learnable B-factors
        if initial_bfactors is not None:
            fitted_bfactors = initial_bfactors.clone()
        else:
            fitted_bfactors = torch.ones_like(self.bfactor_gt, device=self.device) * self.global_b_factor
        fitted_bfactors.requires_grad_(True)
        
        # Create optimizer for B-factor fitting
        optimizer = torch.optim.Adam([fitted_bfactors], lr=b_factor_lr)
        
        # Get density mask for loss computation
        density_mask = self.density_mask_for_final_bfac_fitting
        
        # Determine number of iterations
        if n_iterations is None:
            # Use 250 for GT coordinates, 150 for others
            n_iterations = 250 if (save_path and "ground_truth_from_coordinates" in save_path) else 250
        
        best_loss = float('inf')
        best_bfactors = fitted_bfactors.clone()
        
        print(f"Optimizing B-factors for {n_iterations} iterations...")
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Calculate ESP with current B-factors
            fc_predicted = self.calculate_ESP(
                aligned_structures,
                use_Coloumb=self.use_Coloumb,
                rmax=self.rmax_for_esp,
                should_align=False,
                bfactor=fitted_bfactors
            )
            
            # Apply density mask and extract masked regions
            fc_masked = fc_predicted[density_mask]
            fo_masked = self.fo[density_mask]
            
            # Keep raw values for cross-correlation (like phenix), normalize for L1 loss
            fo_masked_normalized = (fo_masked - fo_masked.mean()) / (fo_masked.std() + 1e-6)
            fc_masked_normalized = (fc_masked - fc_masked.mean()) / (fc_masked.std() + 1e-6)

            if use_cross_correlation:
                # Raw cross-correlation (like phenix): no z-score normalization
                # Calculate raw correlation coefficient
                fo_mean = fo_masked.mean()
                fc_mean = fc_masked.mean()
                fo_centered = fo_masked - fo_mean
                fc_centered = fc_masked - fc_mean
                
                # Raw correlation coefficient (like phenix)
                numerator = (fo_centered * fc_centered).mean()
                denominator = torch.sqrt((fo_centered * fo_centered).mean() * (fc_centered * fc_centered).mean())
                cc_raw = numerator / (denominator + 1e-6)
                
                # We minimize negative correlation to maximize positive correlation
                cc_loss = -cc_raw
                
                # Add penalty to discourage small B-factors and encourage higher ones
                # This prevents the optimization from making atoms "too sharp"
                bfactor_mean = fitted_bfactors.mean()
                bfactor_penalty = bfactor_regularization * torch.exp(-bfactor_mean / 80.0) * (120.0 - bfactor_mean)
                
                loss = cc_loss + bfactor_penalty
            else:
                # L1 loss between predicted and observed densities
                l1_loss = (0.5 * (fo_masked_normalized - fc_masked_normalized).abs()).mean()
                
                # Add penalty to discourage small B-factors and encourage higher ones
                bfactor_mean = fitted_bfactors.mean()
                bfactor_penalty = bfactor_regularization * torch.exp(-bfactor_mean / 80.0) * (120.0 - bfactor_mean)
                
                loss = l1_loss + bfactor_penalty
            
            # Backward pass
            loss.backward()
            
            # Clamp gradients to avoid instability
            if fitted_bfactors.grad is not None:
                torch.nn.utils.clip_grad_norm_([fitted_bfactors], max_norm=100.0)
            
            optimizer.step()
            
            # Clamp B-factors to reasonable range
            with torch.no_grad():
                fitted_bfactors.clamp_(min=bfactor_min, max=bfactor_max)
            
            # Track best B-factors
            current_loss = loss.item()
            if current_loss < best_loss:
                best_loss = current_loss
                best_bfactors = fitted_bfactors.clone().detach()
            
            if iteration % 10 == 0:
                loss_name = "CC loss" if use_cross_correlation else "L1 loss"
                print(f"  Iteration {iteration}: {loss_name} = {current_loss:.6f}, B-factor range = [{fitted_bfactors.min().item():.1f}, {fitted_bfactors.max().item():.1f}]")
        
        final_loss_name = "CC loss" if use_cross_correlation else "L1 loss"
        print(f"B-factor fitting completed. Final {final_loss_name}: {best_loss:.6f}")
        print(f"Fitted B-factor range: [{best_bfactors.min().item():.1f}, {best_bfactors.max().item():.1f}]")
        
        # Save structures with fitted B-factors if save_path is provided
        best_bfactors = best_bfactors if not use_zero_b_values else torch.zeros_like(best_bfactors) # this is for testing
        if save_path is not None:
            self.save_structures_with_bfactors(aligned_structures, best_bfactors, save_path)
        
        return best_bfactors
        
    def save_structures_with_bfactors(self, structures, fitted_bfactors, save_path):
        """
        Save structures with fitted B-factors, replacing the original files.
        """
        from src.utils.non_diffusion_model_manager import save_structure_full
        
        # Determine atom mask based on evaluate_only_resolved parameter
        if self.evaluate_only_resolved:
            atom_mask = self.AF3_to_pdb_mask.cpu()  # Only resolved atoms
            print(f"evaluate_only_resolved=True: Saving structures with only resolved atoms")
        else:
            atom_mask = None  # All atoms
            print(f"evaluate_only_resolved=False: Saving structures with all atoms")
        
        # Save each structure in the batch with fitted B-factors
        for i in range(structures.shape[0]):
            # Find original files to replace. This function is for saving only the structures, not the GT. The GT is saved elsewhere
            original_files = [f for f in os.listdir(save_path) if f.endswith('.pdb') and not f.endswith('_rscc_painted.pdb') and not f.endswith('from_coordinates.pdb')]
            
            if i < len(original_files) + 1:
                original_filename = original_files[i]
                original_path = os.path.join(save_path, original_filename)
                
                # Replace original with fitted version for phenix evaluation
                save_structure_full(
                    structure=structures[i].cpu(),
                    full_sequences=self.full_sequences,
                    atom_array=None,
                    write_file_name=original_path,
                    bfactors=fitted_bfactors.cpu().numpy(),
                    atom_mask=atom_mask  # Apply atom filtering based on evaluate_only_resolved
                )
                
                print(f"Replaced {original_filename} with B-factor fitted version")
        
            """
            if self.evaluate_only_resolved:
                                # Do NOT overwrite the original full PDB. Save a filtered copy instead.
                                filtered_filename = original_filename.replace('.pdb', '_resolved_only.pdb')
                                filtered_path = os.path.join(save_path, filtered_filename)
                                save_structure_full(
                                    structure=structures[i].cpu(),
                                    full_sequences=self.full_sequences,
                                    atom_array=None,
                                    write_file_name=filtered_path,
                                    bfactors=fitted_bfactors.cpu().numpy(),
                                    atom_mask=atom_mask
                                )
                                print(f"Saved resolved-only copy for evaluation: {filtered_filename}")
                            else:
                                # Overwrite original when evaluating full structures
                                save_structure_full(
                                    structure=structures[i].cpu(),
                                    full_sequences=self.full_sequences,
                                    atom_array=None,
                                    write_file_name=original_path,
                                    bfactors=fitted_bfactors.cpu().numpy(),
                                    atom_mask=atom_mask
                                )
                                print(f"Replaced {original_filename} with B-factor fitted version")
                    
            """

    def save_gt_coordinates_as_pdb(self, save_path, b_factor_lr=1.0, use_zero_b_values=False, should_always_fit_gt=False, n_iterations=250, use_cross_correlation=False, bfactor_regularization=0.01, bfactor_min=60.0, bfactor_max=400.0):
        """
        Save the coordinates_gt as a PDB file for evaluation.
        This ensures size consistency between GT and guided structures.
        Includes safety check for zero B-factors and fits them if needed.
        
        Args:
            save_path: Path to save the GT PDB file
            b_factor_lr: Learning rate for B-factor optimization
            use_zero_b_values: Whether to use zero B-values for initialization
            should_always_fit_gt: Whether to always fit B-factors regardless of current values
            n_iterations: Number of iterations for B-factor fitting
            use_cross_correlation: If True, use cross-correlation (Pearson) instead of L1 loss
            bfactor_regularization: Encourages B-factors to increase when fit is poor
        
        Returns:
            str: Path to the saved GT PDB file
        """
        gt_pdb_filename = "ground_truth_from_coordinates.pdb"
        gt_pdb_path = os.path.join(save_path, gt_pdb_filename)
        
        print(f"Saving GT coordinates as PDB: {gt_pdb_filename}")
        
        # Safety check for B-factors
        bfactors_to_use = self.bfactor_gt_untouched.cpu().numpy()
        
        # Check if B-factors are zero or small
        if np.any(bfactors_to_use[self.AF3_to_pdb_mask.cpu()] <= 10.0) or should_always_fit_gt:  # Using 0.1 as threshold for "effectively zero"
            print("Warning: Found zero or very small B-factors in GT structure. Fitting B-factors...")
            
            # Use the unified B-factor fitting function
            gt_structure = self.coordinates_gt.unsqueeze(0)  # Add batch dimension
            fitted_bfactors = self.fit_bfactors_for_structures(
                gt_structure, 
                save_path=None,  # Do not save, return fitted B-factors
                n_iterations=n_iterations,  # Use 250 iterations for GT as requested
                b_factor_lr=b_factor_lr,
                bfactor_min=bfactor_min,  # Use the same min/max as passed to save_state
                bfactor_max=bfactor_max,
                use_cross_correlation=use_cross_correlation,
                bfactor_regularization=bfactor_regularization,
            )
            bfactors_to_use = fitted_bfactors.cpu().numpy() if not use_zero_b_values else torch.zeros_like(fitted_bfactors, device=self.device).cpu().numpy()
            
            print(f"B-factors fitted for GT structure. Range: [{bfactors_to_use.min():.1f}, {bfactors_to_use.max():.1f}]")
        else:
            # Apply use_zero_b_values to original B-factors as well
            if use_zero_b_values:
                bfactors_to_use = np.zeros_like(bfactors_to_use)
            print(f"Using original B-factors. Range: [{bfactors_to_use.min():.1f}, {bfactors_to_use.max():.1f}]")
        
        # Save the GT coordinates using the same method as other structures
        save_structure_full(
            structure=self.coordinates_gt.cpu(),
            full_sequences=self.full_sequences, # check if sequence reduction is needed
            atom_array=None, 
            write_file_name=gt_pdb_path,
            bfactors=bfactors_to_use,
            atom_mask=self.AF3_to_pdb_mask.cpu()
        )
        
        print(f"GT coordinates saved to: {gt_pdb_path}")
        return gt_pdb_path

    def run_phenix_eval(self, pdb_full_path, save_path, phenix_manager):
        print(f"Running phenix eval for {os.path.basename(pdb_full_path)}...")

        # Determine the evaluation path and PDB filtering based on evaluate_only_resolved parameter
        if self.evaluate_only_resolved:
            # Create full_structures folder if it doesn't exist
            full_structures_path = os.path.join(save_path, "full_structures")
            os.makedirs(full_structures_path, exist_ok=True)
            
            # Main folder: save only resolved atoms (filtered) - THESE are for evaluation
            main_use_atom_mask = self.AF3_to_pdb_mask.cpu()  # Filter to only resolved parts
            # Full structures folder: save all atoms (no filtering) - THESE are for inspection only
            full_use_atom_mask = None  # Use full PDB structure (no filtering)
            
            print(f"evaluate_only_resolved=True: Main folder will have resolved atoms only (for evaluation), full_structures/ will have all atoms (for inspection)")
        else:
            # Default behavior: save full PDB files in main folder, no filtering
            main_use_atom_mask = None  # Use full PDB structure (no filtering)
            full_use_atom_mask = None  # Not used when evaluate_only_resolved=False
            print(f"evaluate_only_resolved=False: Using original behavior - full PDB files in main folder")

        # Load the PDB structure once
        pdb_structure, _, _, _ = load_pdb_atom_locations_full(
                pdb_file=pdb_full_path, 
                full_sequences=self.full_sequences,
                chains_to_read=list(range(len(self.full_sequences))),
                return_elements=True,
                return_bfacs=True,
                return_mask=True
            ) 

        if self.evaluate_only_resolved:
            # Only save guided structures (not ground truth) in full_structures folder
            is_ground_truth = "ground_truth_from_coordinates" in os.path.basename(pdb_full_path)
            
            if not is_ground_truth:
                # Save full PDB (all atoms) in full_structures folder - THESE are for inspection only
                full_pdb_path = os.path.join(full_structures_path, os.path.basename(pdb_full_path))
                save_structure_full( 
                    structure=pdb_structure, 
                    full_sequences=self.full_sequences, 
                    atom_array=None, 
                    write_file_name=full_pdb_path,
                    bfactors=None,  # No B-factors for the full PDB
                    atom_mask=full_use_atom_mask  # All atoms
                )
                print(f"Saved full PDB (all atoms) in full_structures folder (for inspection): {full_pdb_path}")
            else:
                print(f"Skipping full_structures save for ground truth file: {os.path.basename(pdb_full_path)}")

            # Use the original PDB file for phenix evaluation (it already has resolved atoms only)
            pdb_for_phenix = pdb_full_path
            evaluation_path = save_path
        else:
            # Original behavior: use the original PDB file for phenix evaluation
            pdb_for_phenix = pdb_full_path
            evaluation_path = save_path

        # Use the corrected density map (with positive B-factor applied) instead of original
        # Use the same pattern as in save_state method
        corrected_density_path = os.path.join(evaluation_path, "corrected_density_for_phenix.mrc")
        
        # Create mrc_to_save using the same pattern as save_state
        mrc_to_save = gemmi.read_ccp4_map(self.esp_file)
        mrc_to_save.grid.array[:] = fft_upsample_3d(self.fo.cpu(), out_shape=(self.D_full,)*3).detach().numpy()
        mrc_to_save.write_ccp4_map(corrected_density_path)
        
        print(f"Using corrected density map: {corrected_density_path}")
        print(f"Original density map: {self.esp_file}")

        # Running the phenix function itself and saving to the corresponding files 
        console_log_path = os.path.join(evaluation_path, os.path.basename(pdb_for_phenix)).replace(".pdb", "_phenix_eval.log")

        # Run phenix map model cc + save it
        phenix_output = phenix_manager.phenix_map_model_cc(pdb_for_phenix, corrected_density_path, self.emdb_resolution_full, os.path.join(evaluation_path, os.path.basename(pdb_for_phenix)).replace(".pdb", ""))
        with open(console_log_path, "w") as f:
            f.write(phenix_output)

        # Now parsing the results of the phenix eval into a [short] csv file 
        parse_phenix_eval_log_to_csv(console_log_path, console_log_path.replace(".log", "_short.csv"))

        # Parsing the CC scores 
        chain_data = parse_phenix_cc_log(console_log_path.replace("phenix_eval.log", "cc_per_residue.log"))
        rscc_tensor = create_cc_bfactor_tensor(chain_data, self.full_sequences)

        # Saving the RSCC painted pdb with appropriate atom filtering
        save_structure_full( 
            structure=pdb_structure, 
            full_sequences=self.full_sequences, 
            atom_array=None, 
            write_file_name=f"{evaluation_path}/{os.path.basename(pdb_for_phenix).replace('.pdb', '_rscc_painted.pdb')}",
            bfactors=rscc_tensor.cpu().numpy(),
            atom_mask=main_use_atom_mask  # Use the same filtering as the PDB used for phenix
        )

        # Clean up temporary corrected density file
        if os.path.exists(corrected_density_path):
            os.remove(corrected_density_path)
            print(f"Cleaned up temporary corrected density file: {corrected_density_path}")

        print(f"Phenix eval for {os.path.basename(pdb_for_phenix)} completed.")
    
    def plot_cc_vs_residue_comparison(self, save_path):
        """
        Create a comparison plot of CC values from reference PDB vs guided PDB.
        Directly reads the cc_per_residue.log files.
        
        Args:
            save_path (str): Directory containing the cc_per_residue.log files
        """
        # Find the reference cc file and all guided cc files
        reference_cc_file = None
        guided_cc_files = []
        
        # Look for files in the save_path
        # Use consistent GT filename from save_gt_coordinates_as_pdb
        gt_filename_base = "ground_truth_from_coordinates"
        
        print(f"Looking for reference file containing: '{gt_filename_base}'")
        
        for filename in os.listdir(save_path):
            if filename.endswith("_cc_per_residue.log"):
                print(f"Found CC log file: {filename}")
                if gt_filename_base in filename:
                    reference_cc_file = os.path.join(save_path, filename)
                    print(f"  -> This is the REFERENCE file: {filename}")
                else:
                    # Check if this is a guided structure (skip painted pdbs)
                    if "_rscc_painted" not in filename:
                        guided_cc_files.append(os.path.join(save_path, filename))
                        print(f"  -> This is a GUIDED file: {filename}")
        
        if not reference_cc_file:
            print("Could not find reference cc_per_residue.log file")
            raise FileNotFoundError("Could not find reference cc_per_residue.log file")
        
        if not guided_cc_files:
            print("Could not find any guided cc_per_residue.log files")
            raise FileNotFoundError("Could not find any guided cc_per_residue.log files")
        
        # Simple function to read CC values from log file
        def read_cc_log(log_file):
            residue_data = []
            with open(log_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) >= 4:
                            chain_id = parts[0]
                            residue_name = parts[1] 
                            residue_num = int(parts[2])
                            cc_value = float(parts[3])
                            residue_data.append((chain_id, residue_name, residue_num, cc_value))
            return residue_data
        
        # Read reference file
        ref_data = read_cc_log(reference_cc_file)
        if not ref_data:
            print("No data found in reference CC log file")
            return
        
        print(f"Reference file: {reference_cc_file}")
        print(f"Reference data points: {len(ref_data)}")
        
        # Sort reference data by chain alphabetically, then by residue number
        ref_data.sort(key=lambda x: (x[0], x[2]))  # (chain_id, residue_num)
        
        # Create a plot for each guided structure
        for guided_cc_file in guided_cc_files:
            guided_data = read_cc_log(guided_cc_file)
            
            if not guided_data:
                print(f"No data found in {os.path.basename(guided_cc_file)}")
                continue
            
            print(f"Guided file: {guided_cc_file}")
            print(f"Guided data points: {len(guided_data)}")
            
            # Sort guided data by chain alphabetically, then by residue number
            guided_data.sort(key=lambda x: (x[0], x[2]))  # (chain_id, residue_num)
            
            # Find intersection of residues present in both reference and guided data
            ref_residues = set((x[0], x[2]) for x in ref_data)  # (chain_id, residue_num) pairs
            guided_residues = set((x[0], x[2]) for x in guided_data)
            common_residues = ref_residues & guided_residues
            
            if not common_residues:
                continue
            
            # Filter both datasets to only include common residues
            ref_data_filtered = [x for x in ref_data if (x[0], x[2]) in common_residues]
            guided_data_filtered = [x for x in guided_data if (x[0], x[2]) in common_residues]
            
            # Sort filtered data consistently
            ref_data_filtered.sort(key=lambda x: (x[0], x[2]))
            guided_data_filtered.sort(key=lambda x: (x[0], x[2]))
            
            # Create the plot
            plt.figure(figsize=(14, 8), dpi=300)
            
            # Extract CC values for plotting from filtered data
            ref_cc_values = [data[3] for data in ref_data_filtered]
            guided_cc_values = [data[3] for data in guided_data_filtered]
            
            # Create x-axis positions
            x_positions = range(len(ref_cc_values))
            
            # Get guided structure name from filename
            guided_name = os.path.basename(guided_cc_file).replace('_cc_per_residue.log', '')
            
            # Plot both lines
            plt.plot(x_positions, ref_cc_values, 'o-', color='#d62728', linewidth=2.5, 
                    markersize=3, alpha=0.8, label='Reference PDB')
            plt.plot(x_positions, guided_cc_values, 'o-', color='#2ca02c', linewidth=2.5, 
                    markersize=3, alpha=0.8, label=f'Guided PDB ({guided_name})')
            
            # Add chain boundaries if multiple chains (using filtered data)
            chain_boundaries = []
            current_chain = ref_data_filtered[0][0] if ref_data_filtered else None
            for i, (chain_id, _, _, _) in enumerate(ref_data_filtered):
                if chain_id != current_chain:
                    chain_boundaries.append(i - 0.5)
                    current_chain = chain_id
            
            for boundary in chain_boundaries:
                plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add horizontal reference lines
            plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.7, linewidth=1.5, 
                       label='CC = 0.5 (Good)')
            plt.axhline(y=0.7, color='green', linestyle=':', alpha=0.7, linewidth=1.5, 
                       label='CC = 0.7 (Excellent)')
            
            # Styling
            plt.xlabel('Common Resolved Residue Position', fontsize=14, fontweight='bold')
            plt.ylabel('Cross-Correlation (CC)', fontsize=14, fontweight='bold')
            plt.title('Cross-Correlation: Reference vs Guided', 
                     fontsize=16, fontweight='bold', pad=20)
            
            # Set y-axis limits
            all_cc_values = ref_cc_values + guided_cc_values
            if all_cc_values:
                y_min, y_max = min(all_cc_values), max(all_cc_values)
                y_range = y_max - y_min
                plt.ylim(max(-0.1, y_min - 0.05 * y_range), min(1.1, y_max + 0.05 * y_range))
            
            # Grid and legend
            plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Extract overall CC statistics from CSV files
            def get_overall_cc_from_csv(cc_filename_base):
                csv_file = f"{save_path}/{cc_filename_base}_phenix_eval_short.csv"
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_file)
                    # Get CC_mask (overall mask correlation) as the main metric
                    cc_mask_row = df[df['metric'] == 'CC_mask']
                    if not cc_mask_row.empty:
                        return cc_mask_row.iloc[0]['value']
                except Exception as e:
                    print(f"Could not read {csv_file}: {e}")
                return None
            
            # Get overall CC values from CSV files
            # Use consistent GT filename from save_gt_coordinates_as_pdb
            ref_name = "ground_truth_from_coordinates"
            ref_overall_cc = get_overall_cc_from_csv(ref_name)
            guided_overall_cc = get_overall_cc_from_csv(guided_name)
            
            # Add statistics using overall CC values
            if ref_overall_cc is not None and guided_overall_cc is not None:
                improvement = guided_overall_cc - ref_overall_cc
                stats_text = f'Reference Overall CC: {ref_overall_cc:.3f}\nGuided Overall CC: {guided_overall_cc:.3f}\nImprovement: {improvement*100:+.1f}%'
            else:
                # Fallback to per-residue mean if CSV reading fails
                ref_mean = np.mean(ref_cc_values)
                guided_mean = np.mean(guided_cc_values)
                improvement = guided_mean - ref_mean
                stats_text = f'Reference Mean: {ref_mean:.3f}\nGuided Mean: {guided_mean:.3f}\nImprovement: {improvement*100:+.1f}%'
            plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                    fontsize=11, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            # Save the plot
            plot_filename = f"{save_path}/cc_comparison_{guided_name}.png"
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"CC comparison plot saved: {os.path.basename(plot_filename)}")
    
    def plot_dihedral_comparison(self, save_path):
        """
        Create comparison plots of dihedral angles from reference PDB vs guided PDbs.
        Shows experimental constraints as shaded regions.
        Only runs if dihedrals are used from NMR.
        
        Args:
            save_path (str): Directory to save the plots
        """
        import gemmi  # Import needed for helper function
        from ..utils.io import AMINO_ACID_ATOMS_ORDER  # Import needed for residue mapping
        
        # Check if dihedral evaluation is run
        if self.dihedrals_parameters["use_dihedrals"] != "from_nmr":
            print("Skipping dihedral evaluation - not using NMR dihedrals")
            return
        
        # Find all guided PDB files (skip painted ones and GT file)
        guided_pdb_files = []
        gt_filename = "ground_truth_from_coordinates.pdb"  # Consistent with save_gt_coordinates_as_pdb
        
        for filename in os.listdir(save_path):
            if (filename.endswith(".pdb") and 
                "_rscc_painted" not in filename and
                filename != gt_filename):
                # Check if it matches the pattern [pdb_id]_[number].pdb
                base_name = filename.replace('.pdb', '')
                if '_' in base_name and base_name.split('_')[-1].isdigit():
                    guided_pdb_files.append(os.path.join(save_path, filename))
        
        if not guided_pdb_files:
            print("No guided PDB files found for dihedral evaluation")
            return
        
        # Function to compute dihedrals for a PDB file
        def compute_pdb_dihedrals(pdb_path):
            # Load PDB coordinates and get its specific mask
            pdb_structure, pdb_mask, _, _ = load_pdb_atom_locations_full(
                pdb_file=pdb_path,
                full_sequences=self.full_sequences,
                chains_to_read=list(range(len(self.full_sequences))),
                return_elements=True,
                return_bfacs=True,
                return_mask=True
            )
            
            # Convert to tensor and apply masks (pdb_structure is already a tensor)
            coords_tensor = pdb_structure.to(self.device).unsqueeze(0)
            pdb_mask = pdb_mask.to(self.device)
            
            # Compute dihedrals for each chain using the specific PDB's mask
            chain_dihedrals = []
            for chain_mask in self.masks_per_sequence:
                phi_psi = backbone_dihedrals(
                    *[coords_tensor[:, backbone_mask & pdb_mask & chain_mask, :]
                      for backbone_mask in self.backbone_masks]
                )
                chain_dihedrals.append(phi_psi)
            
            return chain_dihedrals, pdb_mask
        
        # Compute reference dihedrals using existing coordinates_gt (much more efficient!)
        def compute_reference_dihedrals():
            chain_dihedrals = []
            for chain_mask in self.masks_per_sequence:
                phi_psi = backbone_dihedrals(
                    *[self.coordinates_gt[backbone_mask & self.AF3_to_pdb_mask & chain_mask, :].unsqueeze(0)
                      for backbone_mask in self.backbone_masks]
                )
                chain_dihedrals.append(phi_psi)
            return chain_dihedrals
        
        # Compute reference dihedrals
        ref_dihedrals = compute_reference_dihedrals()
        
        # Get experimental data (already in degrees)
        exp_phi, exp_psi, exp_dphi, exp_dpsi = self.dihedrals_tensor.T.split(1)  # Already in degrees
        exp_phi = exp_phi.squeeze().cpu().numpy()
        exp_psi = exp_psi.squeeze().cpu().numpy()
        exp_dphi = exp_dphi.squeeze().cpu().numpy()
        exp_dpsi = exp_dpsi.squeeze().cpu().numpy()
        exp_mask = self.dihedrals_mask.cpu().numpy()
        
        # Debug: Print some experimental values
        valid_exp_indices = np.where(exp_mask)[0][:3]  # First 3 valid residues
        #print(f"Debug - First few experimental psi values: {exp_psi[valid_exp_indices]}")
        #print(f"Debug - First few experimental psi half-widths: {exp_dpsi[valid_exp_indices]}")
        
        # The key insight: use the same mask for both reference and guided
        # to ensure they have the same tensor dimensions for comparison

        # Create plots for each guided structure
        for guided_pdb_path in guided_pdb_files:
            guided_name = os.path.basename(guided_pdb_path).replace('.pdb', '')
            
            try:
                guided_dihedrals, guided_pdb_mask = compute_pdb_dihedrals(guided_pdb_path)
                
                # Find intersection of resolved atoms between reference and guided structures
                common_resolved_mask = self.AF3_to_pdb_mask & guided_pdb_mask
                
                
                # Recompute both reference and guided dihedrals using the common mask
                # This ensures they have the same dimensions for comparison
                guided_coords = load_pdb_atom_locations_full(
                    pdb_file=guided_pdb_path,
                    full_sequences=self.full_sequences,
                    chains_to_read=list(range(len(self.full_sequences))),
                    return_elements=True,
                    return_bfacs=True,
                    return_mask=True
                )[0].to(self.device)
                
                # Compute dihedrals for both structures using the common mask
                ref_dihedrals_common = []
                guided_dihedrals_common = []
                
                for chain_mask in self.masks_per_sequence:
                    # Reference dihedrals with common mask
                    ref_phi_psi = backbone_dihedrals(
                        *[self.coordinates_gt[backbone_mask & common_resolved_mask & chain_mask, :].unsqueeze(0)
                          for backbone_mask in self.backbone_masks]
                    )
                    ref_dihedrals_common.append(ref_phi_psi)
                    
                    # Guided dihedrals with common mask
                    guided_phi_psi = backbone_dihedrals(
                        *[guided_coords[backbone_mask & common_resolved_mask & chain_mask, :].unsqueeze(0)
                          for backbone_mask in self.backbone_masks]
                    )
                    guided_dihedrals_common.append(guided_phi_psi)
                
                # Handle multimers: average dihedrals across all chains
                if len(ref_dihedrals_common) == 1:
                    # Single chain case
                    ref_phi_psi = ref_dihedrals_common[0].squeeze(0)  # Shape: (N_resolved, 2)
                    guided_phi_psi = guided_dihedrals_common[0].squeeze(0)  # Shape: (N_resolved, 2)
                else:
                    # Multi-chain case: average across chains
                    ref_phi_psi_list = [chain_dihedral.squeeze(0) for chain_dihedral in ref_dihedrals_common]
                    guided_phi_psi_list = [chain_dihedral.squeeze(0) for chain_dihedral in guided_dihedrals_common]
                    ref_phi_psi = torch.stack(ref_phi_psi_list).mean(dim=0)  # Average across chains
                    guided_phi_psi = torch.stack(guided_phi_psi_list).mean(dim=0)  # Average across chains
                
                # Extract phi and psi from the combined tensor
                ref_phi = ref_phi_psi[:, 0]  # Shape: (N,)
                ref_psi = ref_phi_psi[:, 1]  # Shape: (N,)
                guided_phi = guided_phi_psi[:, 0]  # Shape: (N,)
                guided_psi = guided_phi_psi[:, 1]  # Shape: (N,)
                
                # Convert to degrees
                ref_phi_deg = (ref_phi * 180.0 / np.pi).cpu().numpy()
                ref_psi_deg = (ref_psi * 180.0 / np.pi).cpu().numpy()
                guided_phi_deg = (guided_phi * 180.0 / np.pi).cpu().numpy()
                guided_psi_deg = (guided_psi * 180.0 / np.pi).cpu().numpy()
                
                # Determine residue indices (0-based) that are resolved in both PDB and guided and have experimental data
                chain_mask = self.masks_per_sequence[0]
                ca_all = torch.nonzero(self.backbone_masks[1] & chain_mask, as_tuple=False).squeeze(-1)
                ca_used = torch.nonzero(self.backbone_masks[1] & common_resolved_mask & chain_mask, as_tuple=False).squeeze(-1)
                if ca_used.numel() == 0:
                    continue
                if hasattr(torch, "isin"):
                    used_positions = torch.nonzero(torch.isin(ca_all, ca_used), as_tuple=False).squeeze(-1)
                else:
                    used_positions = torch.nonzero((ca_all.unsqueeze(1) == ca_used.unsqueeze(0)).any(dim=1), as_tuple=False).squeeze(-1)
                if used_positions.numel() == 0:
                    continue
                # Intersect with experimental dihedrals mask (per-residue)
                exp_mask_used = self.dihedrals_mask.index_select(0, used_positions.to(self.dihedrals_mask.device))
                if exp_mask_used.sum().item() == 0:
                    continue
                exp_mask_used_np = exp_mask_used.cpu().numpy()
                # Convert to degrees arrays already length == number of used residues
                ref_phi_arr = ref_phi_deg
                ref_psi_arr = ref_psi_deg
                guided_phi_arr = guided_phi_deg
                guided_psi_arr = guided_psi_deg
                # Apply aligned mask
                ref_phi_with_exp = ref_phi_arr[exp_mask_used_np]
                ref_psi_with_exp = ref_psi_arr[exp_mask_used_np]
                guided_phi_with_exp = guided_phi_arr[exp_mask_used_np]
                guided_psi_with_exp = guided_psi_arr[exp_mask_used_np]
                # Experimental values also aligned to residue indices
                used_positions_np = used_positions.cpu().numpy()
                exp_phi_for_plot = exp_phi[used_positions_np][exp_mask_used_np]
                exp_psi_for_plot = exp_psi[used_positions_np][exp_mask_used_np]
                exp_dphi_for_plot = exp_dphi[used_positions_np][exp_mask_used_np]
                exp_dpsi_for_plot = exp_dpsi[used_positions_np][exp_mask_used_np]
                # Residue indices for CSV/plotting (0-based)
                exp_residue_indices = used_positions_np[exp_mask_used_np]
                
                # Function to wrap angles individually to be closest to experimental values
                def wrap_angles_to_experimental(computed_angles, exp_angles):
                    """
                    For each computed angle, choose the representation (angle or angle±360) 
                    that is closest to the corresponding experimental angle.
                    """
                    wrapped = computed_angles.copy()
                    
                    for i in range(len(computed_angles)):
                        if not np.isnan(exp_angles[i]):
                            # Try the original angle and ±360 versions
                            options = [computed_angles[i], computed_angles[i] + 360, computed_angles[i] - 360]
                            distances = [abs(opt - exp_angles[i]) for opt in options]
                            # Choose the option with minimum distance to experimental value
                            wrapped[i] = options[np.argmin(distances)]
                    
                    return wrapped
                
                # Function to wrap experimental ranges to match wrapped computed angles
                def wrap_experimental_ranges(exp_center, exp_half_width, wrapped_computed_angle):
                    """
                    Wrap experimental center and ranges to be closest to the wrapped computed angle.
                    This ensures the shaded experimental regions appear in the right place.
                    """
                    if np.isnan(exp_center) or np.isnan(exp_half_width):
                        return exp_center, exp_half_width
                    
                    # Try different wrappings of the experimental center
                    center_options = [exp_center, exp_center + 360, exp_center - 360]
                    distances = [abs(opt - wrapped_computed_angle) for opt in center_options]
                    wrapped_center = center_options[np.argmin(distances)]
                    
                    return wrapped_center, exp_half_width
                
                # Apply angle wrapping to be closest to experimental values
                ref_phi_wrapped = wrap_angles_to_experimental(ref_phi_with_exp, exp_phi_for_plot)
                guided_phi_wrapped = wrap_angles_to_experimental(guided_phi_with_exp, exp_phi_for_plot)
                ref_psi_wrapped = wrap_angles_to_experimental(ref_psi_with_exp, exp_psi_for_plot)
                guided_psi_wrapped = wrap_angles_to_experimental(guided_psi_with_exp, exp_psi_for_plot)
                
                # Create residue labels
                def get_residue_labels(residue_indices):
                    """Get residue labels in format 'R123A' (residue number + amino acid code)"""
                    labels = []
                    sequence = self.full_sequences[0] if len(self.full_sequences) > 0 else ""
                    for res_idx in residue_indices:
                        res_num = res_idx + 1  # Convert to 1-indexed
                        if 0 <= res_idx < len(sequence):
                            aa_code = sequence[res_idx]
                            labels.append(f"{res_num}{aa_code}")
                        else:
                            labels.append(f"{res_num}")
                    return labels
                
                residue_labels = get_residue_labels(exp_residue_indices)
                
                # Save dihedral angles to CSV files for resolved residues only
                def save_dihedrals_csv(phi_angles, psi_angles, filename, residue_indices):
                    """Save dihedral angles for resolved residues only"""
                    import pandas as pd
                    
                    data = []
                    
                    # Add both phi and psi angles for each resolved residue
                    for i in range(len(phi_angles)):
                        # Use actual residue index from full sequence (1-indexed)
                        residue_num = residue_indices[i] + 1  # Convert to 1-indexed
                        
                        # Add phi angle
                        data.append({
                            'residue_idx': residue_num,
                            'angle_name': 'phi',
                            'target_angle': phi_angles[i],
                        })
                        
                        # Add psi angle
                        data.append({
                            'residue_idx': residue_num,
                            'angle_name': 'psi',
                            'target_angle': psi_angles[i],
                        })
                    
                    # Sort by residue number, then by angle name (phi before psi alphabetically)
                    data.sort(key=lambda x: (x['residue_idx'], x['angle_name']))
                    
                    df = pd.DataFrame(data)
                    df.to_csv(filename, index=False)
                
                # Save reference dihedrals (only once, not per guided structure)
                if guided_name == os.path.basename(guided_pdb_files[0]).replace('.pdb', ''):  # Only for first guided structure
                    ref_csv_filename = f"{save_path}/reference_dihedrals_with_exp_data.csv"
                    save_dihedrals_csv(ref_phi_wrapped, ref_psi_wrapped, ref_csv_filename, exp_residue_indices)
                
                # Save guided dihedrals
                guided_csv_filename = f"{save_path}/{guided_name}_dihedrals_with_exp_data.csv"
                save_dihedrals_csv(guided_phi_wrapped, guided_psi_wrapped, guided_csv_filename, exp_residue_indices)
                
                # Plot PHI angles with experimental data
                plt.figure(figsize=(14, 8), dpi=300)
                
                # Create arrays for the FULL sequence with NaN for missing data
                full_sequence_length = len(self.full_sequences[0])
                full_x_positions = range(full_sequence_length)
                
                # Initialize full arrays with NaN
                phi_full_ref = np.full(full_sequence_length, np.nan)
                phi_full_guided = np.full(full_sequence_length, np.nan)
                
                # Fill in values only where both computed dihedrals and experimental data exist
                for i, res_idx in enumerate(exp_residue_indices):
                    phi_full_ref[res_idx] = ref_phi_wrapped[i]
                    phi_full_guided[res_idx] = guided_phi_wrapped[i]
                
                # Plot with automatic handling of NaN (creates gaps/single dots)
                plt.plot(full_x_positions, phi_full_ref, 'o-', color='#d62728', 
                        linewidth=2.5, markersize=4, alpha=0.8, label='Reference PDB')
                plt.plot(full_x_positions, phi_full_guided, 'o-', color='#2ca02c', 
                        linewidth=2.5, markersize=4, alpha=0.8, label=f'Guided PDB ({guided_name})')
                
                # Add experimental constraints as shaded regions
                for i, res_idx in enumerate(exp_residue_indices):
                    if not np.isnan(exp_phi_for_plot[i]) and not np.isnan(exp_dphi_for_plot[i]):
                        # Get the average of reference and guided for experimental range wrapping
                        avg_wrapped = (ref_phi_wrapped[i] + guided_phi_wrapped[i]) / 2
                        
                        wrapped_exp_center, wrapped_exp_half = wrap_experimental_ranges(
                            exp_phi_for_plot[i], exp_dphi_for_plot[i], avg_wrapped)
                        
                        plt.fill_between([res_idx-0.4, res_idx+0.4], 
                                       wrapped_exp_center - wrapped_exp_half, 
                                       wrapped_exp_center + wrapped_exp_half, 
                                       alpha=0.3, color='orange', label='Experimental Range' if i == 0 else "")
                
                plt.xlabel('Residue', fontsize=14, fontweight='bold')
                plt.ylabel('Phi Angle (degrees)', fontsize=14, fontweight='bold')
                plt.title('Phi Dihedral Angles: Reference vs Guided', 
                         fontsize=16, fontweight='bold', pad=20)
                
                # Create labels for full sequence and set x-tick labels
                full_residue_labels = get_residue_labels(range(full_sequence_length))
                tick_spacing = max(1, full_sequence_length // 15)  # Show ~15 labels max
                tick_positions = full_x_positions[::tick_spacing]
                tick_labels = [full_residue_labels[i] for i in tick_positions]
                plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
                
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)
                plt.tight_layout()
                
                # Save phi plot
                phi_filename = f"{save_path}/dihedral_phi_comparison_{guided_name}.png"
                plt.savefig(phi_filename, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                # Plot PSI angles with experimental data
                plt.figure(figsize=(14, 8), dpi=300)
                
                # Initialize full arrays with NaN (same full sequence as phi plot)
                psi_full_ref = np.full(full_sequence_length, np.nan)
                psi_full_guided = np.full(full_sequence_length, np.nan)
                
                # Fill in values only where both computed dihedrals and experimental data exist
                for i, res_idx in enumerate(exp_residue_indices):
                    psi_full_ref[res_idx] = ref_psi_wrapped[i]
                    psi_full_guided[res_idx] = guided_psi_wrapped[i]
                
                # Plot with automatic handling of NaN (creates gaps/single dots)
                plt.plot(full_x_positions, psi_full_ref, 'o-', color='#d62728', 
                        linewidth=2.5, markersize=4, alpha=0.8, label='Reference PDB')
                plt.plot(full_x_positions, psi_full_guided, 'o-', color='#2ca02c', 
                        linewidth=2.5, markersize=4, alpha=0.8, label=f'Guided PDB ({guided_name})')
                
                # Add experimental constraints as shaded regions
                for i, res_idx in enumerate(exp_residue_indices):
                    if not np.isnan(exp_psi_for_plot[i]) and not np.isnan(exp_dpsi_for_plot[i]):
                        # Get the average of reference and guided for experimental range wrapping
                        avg_wrapped = (ref_psi_wrapped[i] + guided_psi_wrapped[i]) / 2
                        
                        wrapped_exp_center, wrapped_exp_half = wrap_experimental_ranges(
                            exp_psi_for_plot[i], exp_dpsi_for_plot[i], avg_wrapped)
                        
                        plt.fill_between([res_idx-0.4, res_idx+0.4], 
                                       wrapped_exp_center - wrapped_exp_half, 
                                       wrapped_exp_center + wrapped_exp_half, 
                                       alpha=0.3, color='orange', label='Experimental Range' if i == 0 else "")
                
                plt.xlabel('Residue', fontsize=14, fontweight='bold')
                plt.ylabel('Psi Angle (degrees)', fontsize=14, fontweight='bold')
                plt.title('Psi Dihedral Angles: Reference vs Guided', 
                         fontsize=16, fontweight='bold', pad=20)
                
                # Use the same tick positions and labels as phi plot
                plt.xticks(tick_positions, tick_labels, rotation=45, ha='right')
                
                plt.grid(True, alpha=0.3)
                plt.legend(fontsize=12)
                plt.tight_layout()
                
                # Save psi plot
                psi_filename = f"{save_path}/dihedral_psi_comparison_{guided_name}.png"
                plt.savefig(psi_filename, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                
                print(f"Dihedral plots saved: {os.path.basename(phi_filename)}, {os.path.basename(psi_filename)}")
                
            except Exception as e:
                print(f"Error processing {guided_name}: {e}")
                continue
        
    def plot_evolution_plot_single_state(self, aligned_x_0_hat):
        
        vmax, vmin = 35, -30 # the values for the colormap limits
        
        plot_folder = "evolution_plots"
        full_dir = os.path.join(self.save_folder, plot_folder)
        if not os.path.exists( full_dir ):
            os.makedirs(full_dir)

        test_density_mask = self.density_mask

        fc_clone = self.calculate_ESP(
            aligned_x_0_hat, use_Coloumb=self.use_Coloumb, should_align=False, # alignment has ben done already..!
            rmax=self.rmax_for_esp 
        ) 

        i,j,k=0,1,2

        extent=[self.leftbottompoint[j], self.rightupperpoint[j], self.leftbottompoint[k], self.rightupperpoint[k]]


        # Recompute ground truth ESP for comparison
        fc_from_gt = self.fc_from_gt = self.fc_from_gt
        fc_from_gt_mean, fc_from_gt_std = self.fc_from_gt_mean, self.fc_from_gt_std

        # These computations are performed each time but are necessary 
        fc_from_first_x0 = self.calculate_ESP(aligned_x_0_hat[0,:,:].unsqueeze(0), use_Coloumb=self.use_Coloumb, should_align=False) 
        fc_from_first_x0_mean, fc_from_first_x0_std = fc_from_first_x0[self.density_mask].mean(), fc_from_first_x0[self.density_mask].std()

        fig, ax = plt.subplots(3,2, figsize = (8,12))

        fc_mean = fc_clone[test_density_mask].mean()
        fc_std = fc_clone[test_density_mask].std() + 1e-5
        fo_mean = self.fo[test_density_mask].mean()
        fo_std = self.fo[test_density_mask].std() + 1e-5

        # Ensemble
        ax[0][0].imshow(
            torch.where(test_density_mask, (fc_clone - fc_mean)/fc_std, 0).sum(i).detach().cpu().T, 
            origin="lower",cmap="gray", extent=extent,
            vmax=vmax, vmin=vmin
        )

        # Desharpened Gt
        ax[0][1].imshow(
            torch.where(test_density_mask, (self.fo - fo_mean)/fo_std, 0).sum(i).detach().cpu().T, 
            origin="lower",cmap="gray", extent=extent,
            vmax=vmax, vmin=vmin
        )

        # Overlayed fc and fo 
        ax[1][0].imshow(
            torch.where(test_density_mask, (fc_clone - fc_mean)/fc_std, 0).sum(i).detach().cpu().T, 
            origin="lower",cmap="Reds", extent=extent,
            vmax=vmax, vmin=vmin, alpha=0.5, label="Ensemble Fc",
        )
        ax[1][0].imshow(
            torch.where(test_density_mask, (self.fo - fo_mean)/fo_std, 0).sum(i).detach().cpu().T, 
            origin="lower",cmap="Greens", extent=extent, label="Observed Fo",
            vmax=vmax, vmin=vmin, alpha=0.5
        )
        legend_elements = [
            Patch(facecolor="red", edgecolor="r", label="Ensemble Fc"),
            Patch(facecolor="green", edgecolor="g", label="Observed Fo")
        ]
        ax[1][0].legend(handles=legend_elements, loc="upper right", fontsize=8)

        # Fc from GT 
        ax[1][1].imshow(
            torch.where(test_density_mask, (fc_from_gt - fc_from_gt_mean)/fc_from_gt_std, 0).sum(i).detach().cpu().T, 
            origin="lower",cmap="gray", extent=extent,
            vmax = vmax, vmin =vmin
        )

        # Fo full
        ax[2][1].imshow(self.fo.sum(i).detach().cpu().T, origin="lower", cmap="gray", extent=extent)
        
        # Fc from first x0
        ax[2][0].imshow(
            torch.where(test_density_mask, (fc_from_first_x0 - fc_from_first_x0_mean)/fc_from_first_x0_std, 0 ).sum(i).detach().cpu().T, 
            origin="lower",cmap="gray", extent=extent,
            vmax=vmax, vmin=vmin
        )
        

        #ax[3][0].imshow(
        #    ((fc_clone - fc_mean)/fc_std).sum(i).T.detach().cpu(), origin="lower", cmap="gray",
        #    vmax=vmax, vmin=vmin, extent=extent
        #)

        ax[0][0].set_title("Ensemble Fc normalized")
        ax[0][1].set_title("Desharpened Fo normalized")
        ax[1][0].set_title("Fc and Fo normalized overlayed")
        ax[1][1].set_title("Fc from GT normalized")
        ax[2][0].set_title("Fc from first x0 normalized")
        ax[2][1].set_title("Desharpened Fo full")

        fig.suptitle(f"Step {self.evolution_plot_number}", fontsize=16)

        plt.savefig(full_dir + f"/evolution_plot_{self.evolution_plot_number}.png")
        plt.close()

        self.evolution_plot_number += 1 # Increment plot counter for unique filenames
    
    def plot_ROI_evolution_plot_single_state(self, aligned_x_0_hat):
        plot_folder = "ROI_evolution_plots"
        full_dir = os.path.join(self.save_folder, plot_folder)
        if not os.path.exists( full_dir ):
            os.makedirs(full_dir)

        fc_from_gt = self.calculate_ESP(
            self.coordinates_gt.unsqueeze(0), use_Coloumb=self.use_Coloumb,
            should_align=False
        ) 
        fc_from_gt_mean = fc_from_gt[self.density_zone_of_interest_mask].mean() # normalization happens only in the regions of interest as well
        fc_from_gt_std = fc_from_gt[self.density_zone_of_interest_mask].std() + 1e-5

        fc_unmasked = self.calculate_ESP(
            aligned_x_0_hat, should_align=False,
            use_Coloumb=self.use_Coloumb
        ) 
        fc_unmasked_mean = fc_unmasked[self.density_zone_of_interest_mask].mean()
        fc_unmasked_std = fc_unmasked[self.density_zone_of_interest_mask].std() + 1e-5

        fo_mean = self.fo[self.density_zone_of_interest_mask].mean()
        fo_std = self.fo[self.density_zone_of_interest_mask].std() + 1e-5

        reduce_window_slice = slice(self.D//2 - self.D//4, self.D//2 + self.D//4)

        fig, ax = plt.subplots(2,2, figsize=(12,12)); 
        extent = (self.leftbottompoint[1], self.rightupperpoint[1], self.leftbottompoint[2], self.rightupperpoint[2])
        ax[0][0].imshow(
            torch.where(self.density_zone_of_interest_mask == 1, (fc_unmasked - fc_unmasked_mean)/fc_unmasked_std, 0).detach().cpu().sum(0).T[reduce_window_slice,reduce_window_slice], 
            origin="lower", cmap="gray",
            extent = extent, vmax=15, vmin=-10, label="Normalized ensemble forward model"
        ) 
        ax[0][1].imshow(
            torch.where(self.density_zone_of_interest_mask == 1, (self.fo-fo_mean)/fo_std, 0).cpu().sum(0).T[reduce_window_slice,reduce_window_slice], 
            origin="lower", cmap="gray",
            extent = extent, vmax=15, vmin=-10, label = "Normalized ground truth"
        )
        ax[1][1].imshow(
            torch.where(self.density_zone_of_interest_mask == 1, (fc_from_gt - fc_from_gt_mean)/fc_from_gt_std, 0).cpu().sum(0).T[reduce_window_slice,reduce_window_slice], 
            origin="lower", cmap="gray",
            extent = extent, vmax=15, vmin=-10
        )

        ax[1][0].imshow(
            torch.where(self.density_zone_of_interest_mask == 1, (fc_unmasked - fc_unmasked_mean)/fc_unmasked_std, 0).detach().cpu().sum(0).T[reduce_window_slice,reduce_window_slice], 
            origin="lower", cmap="Reds", alpha=0.5,
            extent = extent, vmax=15, vmin=-10, label="Ensemble Fc"
        )
        ax[1][0].imshow(
            torch.where(self.density_zone_of_interest_mask == 1, (self.fo-fo_mean)/fo_std, 0).cpu().sum(0).T[reduce_window_slice,reduce_window_slice], 
            origin="lower", cmap="Greens",
            extent = extent, vmax=15, vmin=-10, label = "Observed Fo", alpha=0.5
        )
        legend_elements = [
            Patch(facecolor="red", edgecolor="r", label="Ensemble Fc"),
            Patch(facecolor="green", edgecolor="g", label="Observed Fo")
        ]
        ax[1][0].legend(handles=legend_elements, loc="upper right", fontsize=8)


        #ax[0].scatter(aligned_x_0_hat[0,self.regions_of_interest_mask,1].cpu().detach(), aligned_x_0_hat[0,self.regions_of_interest_mask,2].cpu().detach(),s=0.05,alpha=0.5)
        #ax[0].scatter(self.coordinates_gt[self.regions_of_interest_mask, 1].cpu(), self.coordinates_gt[self.regions_of_interest_mask, 2].cpu(),s=0.05,alpha=0.5)

        ax[0][0].set_title("Ensemble Fc normalized")
        ax[0][1].set_title("Desharpened Fo normalized")
        ax[1][0].set_title("Fc and Fo normalized overlayed")
        ax[1][1].set_title("Fc from GT normalized")
        
        #ax[1].scatter(self.coordinates_gt[self.regions_of_interest_mask, 1].cpu(), self.coordinates_gt[self.regions_of_interest_mask, 2].cpu(),s=0.05,alpha=0.2)
        fig.suptitle(f"Step {self.evolution_plot_number}", fontsize=16)

        #fig.legend()
        plt.savefig(full_dir + f"/ROI_evolution_plot_{self.evolution_plot_number}.png") # saving each plot like this..!
        plt.close()


    def convert_pngs_to_gif_plot(
            self, plot_folder="evolution_plots", save_gif_name="evolution.gif",
            fps:float = 1/24, loop: int = 0 # 0 is loop forever, 1 is loop once, etc.
        ):
        frames_full_dir = os.path.join(self.save_folder, plot_folder)
        save_full_dir = os.path.join(self.save_folder, save_gif_name)

        frame_dir = Path(frames_full_dir).expanduser().resolve()
        out_file = Path(save_full_dir).expanduser().resolve()

        # ── collect PNGs and sort numerically ──────────────────────────────────
        png_paths = sorted(
            (p for p in frame_dir.glob("*.png") if self._num_suffix.search(p.name)),
            key=lambda p: int(self._num_suffix.search(p.name).group(1)),
        )

        if not png_paths:
            # Just skip instead of an error. 
            return 
            #raise FileNotFoundError(
            #    f"No PNGs matching '*_<number>.png' found in {frame_dir}"
            #)

        # ── read, stack, and write in one call ─────────────────────────────────
        iio.imwrite(
            out_file,
            [iio.imread(p) for p in png_paths],
            duration=fps,
            loop=loop,
        )

        print(f"✅  Saved {len(png_paths)}-frame GIF → {out_file}")


    def _initialize_interpolant_backbone_points_per_sequence(self):
        
        binary_stacks = [
            torch.where(density_mask, 1.0, 0.0).to(torch.bool).cpu().numpy()
            for density_mask in self.density_masks_per_chain_for_backbone
        ]
        reinforced_stacks = [
            scipy.ndimage.binary_dilation(stack, iterations=2) for stack in binary_stacks
        ]
        
        skeletons = [skeletonize(binary_stack) for binary_stack in reinforced_stacks] # Creating a numpy skeleton

        skel_objs = [Skeleton(skeleton) for skeleton in skeletons] # Creating the object from skimage
        branch_data = [skan.summarize(skel_obj, separator='_') for skel_obj in skel_objs] # Creating the branch data

        # Extracting the path coordinates from the main branch for each chain 
        main_branch_ids = [branch_data['branch_distance'].idxmax() for branch_data in branch_data]
        longest_path_indices = [skel_obj.path_coordinates(main_branch_id) for skel_obj, main_branch_id in zip(skel_objs, main_branch_ids)] 

        # Converting the binary coordinates to the real coordinates
        path_coords = [
            self.lattice_3d.reshape(self.D, self.D, self.D, 3)[
                longest_path_indices[i][:, 0], 
                longest_path_indices[i][:, 1], 
                longest_path_indices[i][:, 2]
            ].cpu().numpy() 
            for i in range(len(longest_path_indices))
        ]

        # Interpolation to get the length of the path as desired
        skeleton_lengths = [len(path_coords[i]) for i in range(len(path_coords))]
        new_lengths = [mask.sum() for mask in self.masks_per_sequence]

        # Interpolation
        original_indices = [np.arange(skeleton_lengths[i]) for i in range(len(skeleton_lengths))]
        new_indices = [np.linspace(0, skeleton_lengths[i] - 1, new_lengths[i]) for i in range(len(skeleton_lengths))]  

        interpolators = [scipy.interpolate.interp1d(original_indices[i], path_coords[i].T) for i in range(len(path_coords))]
        final_backbone_points_per_sequence = [interpolators[i](new_indices[i]).T for i in range(len(interpolators))]

        self.backbones_concatenated = torch.from_numpy(np.concatenate(final_backbone_points_per_sequence, axis=0)).to(self.device).to(torch.float32)
        self.backbones_concatenated_centered = self.backbones_concatenated - self.backbones_concatenated.mean(dim=0)




# Other guidance/loss functions to be implemented.

class CryoEM_ImageAverage_GuidanceLossFunction(AbstractLossFunction):
    def __init__():
        raise NotImplementedError("This loss function is not implemented yet.")

class CryoEM_Images_GuidanceLossFunction(AbstractLossFunction):
    def __init__():
        raise NotImplementedError("This loss function is not implemented yet.")
    
        
def all_groupwise_permutations(sequence_counts):
    """
    Given sequence_counts like [2, 3], returns all permutations that 
    preserve group blocks but permute within each group.

    Example:
    [2, 3] means group 1 = [0, 1], group 2 = [2, 3, 4]
    Total permutations: 2! * 3! = 12
    Each output is a list of indices [i_0, i_1, ..., i_N-1] of length sum(sequence_counts)
    """
    groups = []
    start = 0
    for count in sequence_counts:
        group = list(range(start, start + count))
        groups.append(group)
        start += count

    # Generate permutations for each group
    group_perms = [list(itertools.permutations(g)) for g in groups]

    # Cartesian product of these permutations
    for product in itertools.product(*group_perms):
        yield [i for group in product for i in group]

def calc_dihedral2(
        v1: torch.Tensor, # Shape (N, 3)
        v2: torch.Tensor, # Shape (N, 3)
        v3: torch.Tensor, # Shape (N, 3)
        v4: torch.Tensor  # Shape (N, 3)
    ):
        """
        Calculates the dihedral angle defined by four arrays of 3d points (3,N)
        This is the angle between the plane defined by the first three
        points and the plane defined by the last three points.
        """
        b0 = v1 - v2
        b1 = v3 - v2
        b2 = v4 - v3

        # normalize b1 so that it does not influence magnitude of vector
        # rejections that come next
        b1_nrm = b1 / torch.norm(b1, dim=0, p=2)

        # v = projection of b0 onto plane perpendicular to b1
        #   = b0 minus component that aligns with b1
        # w = projection of b2 onto plane perpendicular to b1
        #   = b2 minus component that aligns with b1
        v = b0 - (b0*b1_nrm).sum(0) * b1_nrm
        w = b2 - (b2*b1_nrm).sum(0) * b1_nrm

        # angle between v and w in a plane is the torsion angle
        # v and w are not normalized but that's fine since tan is y/x
        x = (v * w).sum(0)
        y = (torch.cross(b1_nrm, v, dim=0) * w).sum(0)

        return torch.arctan2(y, x)

def calc_dihedral_batched(v1, v2, v3, v4, eps: float = 1e-8):
    """
    v1..v4: tensors of shape (..., 3) describing points p0..p3.
    Returns torsion angle (in radians) with shape (...,).
    """
    b0 = v1 - v2          # p0 - p1
    b1 = v3 - v2          # p2 - p1
    b2 = v4 - v3          # p3 - p2

    b1n = b1 / torch.clamp(torch.linalg.norm(b1, dim=-1, keepdim=True), min=eps)

    # project onto plane perpendicular to b1
    v = b0 - (b0 * b1n).sum(dim=-1, keepdim=True) * b1n
    w = b2 - (b2 * b1n).sum(dim=-1, keepdim=True) * b1n

    x = (v * w).sum(dim=-1)
    y = (torch.cross(b1n, v, dim=-1) * w).sum(dim=-1)
    return torch.atan2(y, x)  # in (-pi, pi]

def backbone_dihedrals_old(
        XN: torch.Tensor, # shape 
        XCA: torch.Tensor, 
        XC: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
    """
    XN, XCA and XC are coordinates of the N, CA and C atoms of the backbone.
    """
    PHI = calc_dihedral2(XC[:,0:-1], XN[:,1:], XCA[:,1:], XC[:,1:])
    PSI = calc_dihedral2(XN[:,0:-1], XCA[:,0:-1], XC[:,0:-1], XN[:,1:])
    return PHI, PSI

def backbone_dihedrals(
        XN: torch.Tensor, # Shape (B, N, 3), i.e., expects batched, even if B=1, then unsqueezed.
        XCA: torch.Tensor, 
        XC: torch.Tensor
    ) -> torch.Tensor:
    """
    XN, XCA and XC are coordinates of the N, CA and C atoms of the backbone.
    
    Returns:
        torch.Tensor: Shape (B, N, 2) where last dimension is [phi, psi] angles in radians
    """
    B = XN.shape[0]
    N = XN.shape[1]
    PHI_PSI = torch.zeros((B, N , 2), device=XN.device)
    PHI = calc_dihedral_batched(XC[:,0:-1], XN[:,1:], XCA[:,1:], XC[:,1:])
    PSI = calc_dihedral_batched(XN[:,0:-1], XCA[:,0:-1], XC[:,0:-1], XN[:,1:])

    PHI_PSI[:,1:,0] = PHI # Phi #i is defined for residues [i-1, i], hence the first residue is skipped.
    PHI_PSI[:,0:-1,1] = PSI # PSI #i is defined for residues [i, i+1], hence the last residue is skipped.
    return PHI_PSI

def angle_diff(a, b, positive_only=False):
    distance = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    return distance if not positive_only else torch.abs(distance)


"""
x_0_hat.retain_grad()

coordinates = self.coordinates_gt.clone().detach().requires_grad_(True)

testX = coordinates[self.AF3_to_pdb_mask]

testWX = self.element_gt[self.AF3_to_pdb_mask]
testWX = testWX / testWX.sum(dim=-1).flatten()

density_mask_for_OT = self.density_mask.flatten()

Y = self.lattice_3d[density_mask_for_OT, :]  # voxel centers

weightsY = self.fo_unthresholded.flatten()[density_mask_for_OT] - self.fo_unthresholded.flatten()[density_mask_for_OT].min()
weightsY = (weightsY / weightsY.sum())

testloss = SamplesLoss(
    loss="sinkhorn", p=1, blur=0.5, backend="online", debias=True, reach=0.08, scaling=0.9
)(
    testWX, testX, weightsY, Y
)
print(testloss)

testloss.backward(retain_graph=True)

print(coordinates.grad.norm()) 

# For debugging the sinkhorn loss and whether it's close enough to the local
"""


"""

i, j, k = 0, 1, 2
plt.imshow(
    skeleton.sum(i).T, origin="lower", 
    extent = (self.leftbottompoint[1], self.rightupperpoint[1], self.leftbottompoint[2], self.rightupperpoint[2]), 

); 
plt.scatter(self.coordinates_gt[self.masks_per_sequence[0]][:, j].cpu(), self.coordinates_gt[self.masks_per_sequence[0]][:, k].cpu(), alpha=0.01, c="orange") 
plt.show(); plt.savefig("test.png"); plt.close()

"""

"""

from skimage.morphology import skeletonize
from skan import Skeleton # Import the main class
import skan
import scipy

binary_stack = torch.where(self.density_masks_per_chain[0], 1.0, 0.0).to(torch.bool).cpu().numpy()
binary_stack = scipy.ndimage.binary_closing(binary_stack, iterations=30)
skeleton = skeletonize(binary_stack)

skel_obj = Skeleton(skeleton)
branch_data = skan.summarize(skel_obj, separator='_')

main_branch_id = branch_data['branch_distance'].idxmax()
path_coords = skel_obj.path_coordinates(main_branch_id)

# Interpolation to get the length of the path as desired
num_original_points = len(path_coords)
original_indices = np.arange(num_original_points)
interpolator = scipy.interpolate.interp1d(original_indices, path_coords.T)

N = 100
new_indices = np.linspace(0, num_original_points - 1, N)
final_backbone_points = interpolator(new_indices).T


"""

"""

1

"""



"""

# Assume 'skeleton' is the numpy array
labeled_skeleton, num_components = scipy.ndimage.label(skeleton)

print(f"Found {num_components} separate parts in the skeleton.")

# Different parts are also visualized with different colors
import matplotlib.pyplot as plt
plt.imshow(labeled_skeleton.sum(axis=0).T, cmap='nipy_spectral', origin='lower')
plt.title('Visualizing the Disconnected Skeleton Parts')
plt.colorbar()
plt.savefig('component_check.png')
plt.show()
plt.close() 
"""


"""

i, j, k = 0, 1, 2 # Your axis definitions

# Create the color sequence for the path_coords points
num_path_points = len(path_coords)
point_order_colors = np.arange(num_path_points)

plt.figure(figsize=(10, 8))

# 1. Plot the skeleton projection
plt.imshow(
    skeleton.sum(axis=i).T,
    origin="lower",
    cmap='gray_r'
    # extent=(self.leftbottompoint[j], self.rightupperpoint[j], self.leftbottompoint[k], self.rightupperpoint[k])
)

# 2. Plot the ORDERED SKELETON path_coords on top
scatter = plt.scatter(
    path_coords[:, j],           # Using path_coords here
    path_coords[:, k],           # Using path_coords here
    c=point_order_colors,        # Color by order
    cmap='viridis',              # Sequential colormap
    s=10,                        # Point size
    #edgecolor='black',           # Point edge color
    zorder=10,                   # Plot on top
    label='Ordered Skeleton Path (path_coords)'
)

# 3. Add a colorbar and labels
cbar = plt.colorbar(scatter)
cbar.set_label('Order of Points in path_coords')
plt.xlabel(f'Axis {j}')
plt.ylabel(f'Axis {k}')
plt.title('Confirmation of Ordered Skeleton Path')
plt.legend()
plt.savefig("test.png")
plt.show()
plt.close()

"""


"""

# Assume 'skeleton' and 'final_backbone_points' from the previous steps
# are already defined in the environment.

# --- Your plotting setup ---
i, j, k = 0, 1, 2 # Using axis definitions for projection

# Create the color sequence for the backbone points
num_backbone_points = len(final_backbone_points)
# This creates an array like [0, 1, 2, ..., N-1]
point_order_colors = np.arange(num_backbone_points)

# --- Generate the Plot ---
plt.figure(figsize=(10, 8))

# 1. Plot the skeleton projection (original code)
plt.imshow(
    skeleton.sum(axis=i).T,
    origin="lower",
    cmap='gray_r',
    # extent=(self.leftbottompoint[1], self.rightupperpoint[1], self.leftbottompoint[2], self.rightupperpoint[2])
)

# 2. Plot the original density cloud (original code)
# plt.scatter(
#     self.coordinates_gt[self.masks_per_sequence[0]][:, j].cpu(),
#     self.coordinates_gt[self.masks_per_sequence[0]][:, k].cpu(),
#     alpha=0.01, c="orange", label="Original Density"
# )

# 3. Plot the NEW ordered backbone points on top
scatter = plt.scatter(
    final_backbone_points[:, j], # Plotting the j-th coordinate
    final_backbone_points[:, k], # Plotting the k-th coordinate
    c=point_order_colors,        # Color each point by its order
    cmap='viridis',              # A good colormap for sequences (purple -> yellow)
    s=10,                        # Make the points bigger to see them clearly
    #edgecolor='black',           # Add a black edge to make them pop
    zorder=10,                   # Ensure they are plotted on top
    label='Ordered Backbone Points'
)

# 4. Add a colorbar to explain the color gradient
cbar = plt.colorbar(scatter)
cbar.set_label('Order of Points (Start -> End)')

plt.xlabel(f'Axis {j}')
plt.ylabel(f'Axis {k}')
plt.title('Skeleton with Ordered Backbone Overlay')
plt.legend()
plt.savefig("test.png")
plt.show()
plt.close()

"""


"""


# Assume 'skeleton' and 'final_backbone_points' from the previous steps
# are already defined in the environment.

# --- Your plotting setup ---
i, j, k = 0, 1, 2 # Using axis definitions for projection

# Create the color sequence for the backbone points
num_backbone_points = len(long_path_coords)
# This creates an array like [0, 1, 2, ..., N-1]
point_order_colors = np.arange(num_backbone_points)

# --- Generate the Plot ---
plt.figure(figsize=(10, 8))

# 1. Plot the skeleton projection (original code)
plt.imshow(
    skeleton.sum(axis=i).T,
    origin="lower",
    cmap='gray_r',
    # extent=(self.leftbottompoint[1], self.rightupperpoint[1], self.leftbottompoint[2], self.rightupperpoint[2])
)

# 2. Plot the original density cloud (original code)
# plt.scatter(
#     self.coordinates_gt[self.masks_per_sequence[0]][:, j].cpu(),
#     self.coordinates_gt[self.masks_per_sequence[0]][:, k].cpu(),
#     alpha=0.01, c="orange", label="Original Density"
# )

# 3. Plot the NEW ordered backbone points on top
scatter = plt.scatter(
    long_path_coords[:, j], # Plotting the j-th coordinate
    long_path_coords[:, k], # Plotting the k-th coordinate
    c=point_order_colors,        # Color each point by its order
    cmap='viridis',              # A good colormap for sequences (purple -> yellow)
    s=10,                        # Make the points bigger to see them clearly
    #edgecolor='black',           # Add a black edge to make them pop
    zorder=10,                   # Ensure they are plotted on top
    label='Ordered Backbone Points'
)

# 4. Add a colorbar to explain the color gradient
cbar = plt.colorbar(scatter)
cbar.set_label('Order of Points (Start -> End)')

plt.xlabel(f'Axis {j}')
plt.ylabel(f'Axis {k}')
plt.title('Skeleton with Ordered Backbone Overlay')
plt.legend()
plt.savefig("test.png")
plt.show()
plt.close()

"""