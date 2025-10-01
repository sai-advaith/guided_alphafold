import torch
import numpy as np 
from numpy import pi
from .density_atomsf import ATOM_TO_INDEX, ALL_ATOM_TENSOR
from pykeops.torch import LazyTensor

class AtomDensityEstimator:
    def __init__(self, device):
        # the mapping for each atom density
        self.device = device
        self.asf = ALL_ATOM_TENSOR.to(dtype=torch.float32, device=device)
        # this is a function that maps np.array of element chars to their element index
        self.element_indexes_mapper = np.vectorize(lambda x: ATOM_TO_INDEX.get(x, ATOM_TO_INDEX["C"]))
        
        self.current_element_indexes = None
        self.last_used_elements = None
        self.four_pi_sqaured = 4 * pi * pi

    def to(self, device):
        self.device = device
        self.asf = self.asf.to(self.device)
    
    def get_element_indexes(self, elements):
        """
            calculating the elements indexes is expensive becuase it is in numpy, this is a cache mechanism.
            when using the same protein, the elements shoulden't change, so this will give a significant speed up
        """
        if self.last_used_elements is None or self.last_used_elements.shape != elements.shape or (self.last_used_elements != elements).any():
            self.current_element_indexes = torch.tensor(self.element_indexes_mapper(elements), dtype=torch.int64, device=self.device)
            self.last_used_elements = elements.copy()
        return self.current_element_indexes.clone()
    
    def compute_density_from_distances(self, distance_squared, bfactors, element_indices):
        asf_data = self.asf[element_indices]
        divisor = asf_data[:, 1, :] + bfactors[:,None]

        bw = torch.where(divisor > 1e-4, -self.four_pi_sqaured / divisor, 0.0)
        aw = asf_data[:, 0, :] * (-bw / pi) ** 1.5

        exp_factor = distance_squared.unsqueeze(-1) * bw
        density = torch.sum(aw * torch.exp(exp_factor), dim=-1)
        return density
    
    def sample_fc_from_all_atom(self, atom_locations, occupancy, elements_indexes, bfactors, xyz, rmax=3.5, density_batch_size=60000):
        with torch.no_grad():
            no_grad_distances = (atom_locations[:,None] - xyz[:,:, None]).norm(dim=-1)
            mask = (no_grad_distances <= rmax)
            indexes = []
            for i, mask_chunk in enumerate(mask.split(density_batch_size, dim=1)):
                nonzero_indexes = mask_chunk.nonzero()
                nonzero_indexes[..., 1] += i * density_batch_size
                indexes.append(nonzero_indexes)
            applied_pairs = torch.cat(indexes, dim=0)
            pairing_batch, pairing_xyz, pairing_atoms = applied_pairs.unbind(dim=-1)

        # recalculating the distances to make sure that the big distnacaes matrix is not in the backpropagating graph, not using the sqrt in the end of the norm gives distances squared
        output_tensor = torch.zeros(xyz.shape[0], xyz.shape[1], dtype=torch.float32, device=self.device)
        if len(pairing_batch) == 0:
            return output_tensor
        distances_squared = (xyz[pairing_batch, pairing_xyz] - atom_locations[pairing_batch, pairing_atoms]).pow(2).sum(dim=-1)
        occupancy = occupancy[pairing_batch, pairing_atoms]
        bfactors = bfactors[pairing_batch, pairing_atoms]
        elements_indexes = elements_indexes[pairing_batch, pairing_atoms]
        density = self.compute_density_from_distances(distances_squared, bfactors, elements_indexes)
        density_occupancy = density * occupancy
        output_tensor.index_put_((pairing_batch, pairing_xyz), density_occupancy, accumulate=True)
        
        return output_tensor

    def sample_fc_from_all_atom_optimized(self, atom_locations, occupancy, elements_indexes, bfactors, xyz, rmax=3.5):
        xyz = xyz[0]
        output_volume = torch.zeros(xyz.shape[0], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            lattice_i = LazyTensor(xyz[:, None, :] )
            atom_positions_j = LazyTensor(atom_locations.reshape(-1, 3)[None, :, :])
            # D_ij =  (lattice_i - atom_positions_j).square().sum(dim=2, keepdim=True)
            D_ij = (lattice_i - atom_positions_j).norm(dim=2)
            keep = D_ij <= rmax
            mask = keep.sum(1)
            mask = torch.where(mask > 0, 1.0, 0.0).detach().to(torch.bool).to(atom_locations.device).flatten()

        if mask.sum() == 0:
            return output_volume

        B = atom_locations.shape[0]
        atom_locations = atom_locations.view(-1, 3)

        # Locations
        elements_indexes = elements_indexes.flatten()[:, None]
        bfactors = bfactors.flatten()[:, None]
        N = atom_locations.shape[0]

        lattice_i = LazyTensor(xyz[mask, None, :])
        atom_positions_j = LazyTensor(atom_locations[None, :, :])
        D_ij = (lattice_i - atom_positions_j).square().sum(dim=2, keepdim=True)

        scatter_factors = self.asf[elements_indexes]
        gaussian_amplitudes, gaussian_widths = scatter_factors[:, :, 0, :], scatter_factors[:, :, 1, :]
        divisor = gaussian_widths + (bfactors).unsqueeze(-1)

        a_jk = LazyTensor(gaussian_amplitudes.squeeze(1)[None])#(1, N, self.asf.shape[-1]))
        b_w = LazyTensor(torch.where(divisor > 1e-4, -self.four_pi_sqaured / divisor, 0.0).squeeze(1)[None])#.view(1, N, self.asf.shape[-1]))

        aw = a_jk * (-b_w / pi) ** 1.5
        exp_factor = (D_ij * b_w)
        volume = (aw * exp_factor.exp()).sum(dim=-1).sum(dim=1).squeeze()

        # Render output
        output_volume[mask] = volume / B
        return output_volume

    def __call__(self, atom_locations, occupancy, elements, bfactors, xyz, rmax=3.5):
        return self.sample_fc_from_all_atom(atom_locations, occupancy, elements, bfactors, xyz, rmax=rmax)
