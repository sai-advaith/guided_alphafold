import numpy as np
import torch
import matplotlib.pyplot as plt

from .density_guidance.density_grid import XMap
import torch.nn.functional as F
from .density_guidance.density_estimator import AtomDensityEstimator
from .density_pre_processing import density_pre_processing

import os
from torch.utils.data import DataLoader

from .abstract_conditioner import LossFunctionGuidance
from ..protenix.metrics.rmsd import self_aligned_rmsd
from ..protenix.data.utils import save_structure_cif

# TODO
# 5. Save things in evaluation format -- main script
# 6. Bring the bfactor optimization code back -- conditioner script
# 7. Bring evaluation script back -- neither, but relevant
# 13. Spacegroup stuff
# 15. Log Fo and Fc
class DensityConditioner(LossFunctionGuidance):
    def __init__(self, input_density_map,
                 input_structure_coordinates,
                 bfactor,
                 elements,
                 resolution,
                 residue_range_mask=None,
                 dtype=torch.float32,
                 device=torch.device("cuda:0"),
                 batch_size=1,
                 step_size=50,
                 wandb_logger=None,
                 use_density_pre_processing=True, start_guidance_at_diffusion_time=200, density_pre_processing_level_set=0, density_pre_processing_epsilon=1e-7,
                 density_pre_processing_diff=0.1,
                 extraction_padding=2.5,
                 zone_of_interest_coords=None,
                 guide_x_t=False, recalculate_x_0_hat=False,
                 seed=0, pdb_id=None):
        super().__init__(guide_x_t, recalculate_x_0_hat, step_size)
        self.dtype, self.device = dtype, device
        self.resolution = resolution
        self.batch_size = batch_size

        # Read map + initialize the padding
        self.extraction_padding = extraction_padding
        fo_density_xmap = XMap.fromfile(input_density_map, resolution=self.resolution, label=None, dtype=self.dtype, device=self.device)
        self.fo_density_xmap = fo_density_xmap.canonical_unit_cell()
        # TODO: Replace
        self.fo_density_xmap.unit_cell.set_space_group("P1")
        assert self.fo_density_xmap.is_canonical_unit_cell()

        # Read the structure coordinates
        self.sliced_fo_xmap, self.grid_indices = self.fo_density_xmap.extract(zone_of_interest_coords, padding=self.extraction_padding)
        self.fo_density_xmap.array = self.fo_density_xmap.array.clamp(0)
        self.fo_locations = self.fo_density_xmap.get_voxels_cartisian_centeroids()[self.grid_indices]
        self.fo_gt_values = self.fo_density_xmap.array[self.grid_indices]

        # Mask for Atoms in relevant residue range
        self.residue_range_atom_mask = residue_range_mask

        # Apply pre-processing that wont clamp things at zero
        if use_density_pre_processing:
            self.processed_gt_fo = density_pre_processing(self.fo_gt_values.cpu(), level_set=density_pre_processing_level_set,epsilon=density_pre_processing_epsilon, diff=density_pre_processing_diff).to(device)
        else:
            self.processed_gt_fo = self.fo_gt_values
        self.density_estimator = AtomDensityEstimator(self.device)

        # Other important properties
        # TODO: Replace with bfactor optimization code
        self.structure_bfactor = bfactor.repeat(self.batch_size, 1)
        self.input_structure_coordinates = input_structure_coordinates.repeat(self.batch_size, 1, 1)

        # Get elements from somewhere
        self.structure_elements = np.repeat(elements, self.batch_size, axis=0)
        self.structure_elements_indicies = self.density_estimator.get_element_indexes(self.structure_elements)
        self.structure_occupancy = (self.structure_bfactor != 0).to(self.dtype)

        # Wandb logging + other hyperparameters
        self.start_guidance_at_diffusion_time = start_guidance_at_diffusion_time
        self.wandb_logger = wandb_logger

        # Create log dir
        dump_location = os.path.join("", self.pdb_id, f"seed_{self.seed}")
        self.prediction_save_dir = os.path.join(dump_location, "predictions")
        os.makedirs(self.prediction_save_dir, exist_ok=True)

    def compute_fo_fc(self, coordinates_all, bfactor, element_indecies, occupancy):
        fo = self.processed_gt_fo
        xyz = self.fo_locations.flatten(0,-2)[None].repeat(self.batch_size, 1 ,1)
        fc = self.density_estimator(coordinates_all, occupancy, element_indecies,bfactor, xyz, rmax=5)
        return fo.flatten()[None], fc

    def get_fo_fc_from_atom_coordinates(self, coordinates_all_atom):
        bfactor, element_indecies = self.structure_bfactor.clone(), self.structure_elements_indicies.clone()
        occupancy = self.structure_occupancy.clone()

        # Get density grid values
        fo_values, fc_values = self.compute_fo_fc(coordinates_all_atom, bfactor, element_indecies, occupancy)
        return fo_values, fc_values

    def energy_from_fo_fc(self, fo, fc):
        fc = fc.mean(dim=0, keepdims=True)
        mean = fo.mean()
        std = (fo.std() + 1e-6)

        fo = fo - mean
        fc = fc - mean

        fo = fo / std
        fc = fc / std
        U_density =  (0.5 * (fo - fc).square()).sum()
        U_conditioner = U_density.sum()
        return U_conditioner

    def log_density_files(self, fo, fc, diffusion_time):
        fo_density_slice, fc_density_slice = XMap.zeros_like(self.sliced_fo_xmap), XMap.zeros_like(self.sliced_fo_xmap)
        fo_density_slice.array, fc_density_slice.array = fo.clone(), fc.clone()

        # Save paths
        fo_output_fpath = os.path.join(self.prediction_save_dir, f"{self.pdb_id}_fo_t_{diffusion_time}.ccp4")
        fc_output_fpath = os.path.join(self.prediction_save_dir, f"{self.pdb_id}_fc_t_{diffusion_time}.ccp4")

        # Log the density files
        fo_density_slice.tofile(fo_output_fpath)
        fc_density_slice.tofile(fc_output_fpath)

    def calculate_fo_fc_energy(self, coordinates_all_atom, diffusion_time, verbose=False):
        fo_values, fc_values = self.get_fo_fc_from_atom_coordinates(coordinates_all_atom)

        fo_values, fc_values = fo_values.reshape(self.fo_gt_values.shape), fc_values.reshape([-1] + list(self.fo_gt_values.shape))

        # if diffusion_time >= 0:
        # self.log_density_files(fo_values, fc_values[0], diffusion_time)

        U_conditioner = self.energy_from_fo_fc(fo_values, fc_values)
        if verbose:
            cosine_similarity = torch.nn.functional.cosine_similarity(self.fo_gt_values.flatten()[None], fc_values.flatten(1,-1))
            print(f"cosine similarity anneal at t={diffusion_time}: {cosine_similarity.tolist()}")
        return U_conditioner

    def calculate_loss(self, x_t, x_0_hat, t):
        """
        Given coordinates, compute grad norm(Fo-Fc) wrt coordinates
        Coordinates: tensor of shape N x 3
        return the normalized Fo and Fc slices at that region
        """
        if t >= self.start_guidance_at_diffusion_time:
            return 0

        # Align to make sure we are in the density
        _, aligned_pose, _, _ = self_aligned_rmsd(x_0_hat, self.input_structure_coordinates, torch.ones_like(x_0_hat, dtype=torch.bool)[...,0])

        # Get density loss
        energy = self.calculate_fo_fc_energy(aligned_pose, t, verbose=True)

        # Log the cif file at every timestep of the diffusion process
        if t == 198:
            self.log_files(x_0_hat, t)

        # Log energy values in wandb
        self.wandb_logger.update("density_energy", energy)
        return energy