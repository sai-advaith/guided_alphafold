import os
from ..protenix.metrics.rmsd import self_aligned_rmsd
from ..utils.io import get_sampler_pdb_inputs, get_atom_mask, get_non_missing_atom_mask
import torch
from ..utils.density_guidance.density_estimator import AtomDensityEstimator
import gemmi
import numpy as np
from .abstract_loss_funciton import AbstractLossFunction
from ..utils.symmetry import get_pdb_symmetries_R_T
# from geomloss import SamplesLoss


def get_density_map_voxel_centroids(density_map):
    density_extent = density_map.get_extent()
    extent_minimum, extent_maximum = [np.array(list(density_extent.minimum)), np.array(list(density_extent.maximum))]
    extent_size = extent_maximum - extent_minimum
    # density locations are between 0 and 1
    density_locations = np.mgrid[:density_map.grid.shape[0], :density_map.grid.shape[1], :density_map.grid.shape[2]].transpose(1,2,3,0) / np.array(density_map.grid.shape)[None, None, None]
    # density locations are between minimum extent and maximum extent
    density_locations = (density_locations * extent_size[None,None,None]) + extent_minimum[None,None,None]
    voxel_size = extent_size / np.array(density_map.grid.shape)
    # moving the density locations to the center of the voxel and not the bottom left
    density_locations = density_locations + voxel_size[None, None, None] / 2
    # projects the fractional coordinates to absoulte coordiantes
    density_locations = (np.array(density_map.grid.unit_cell.orth.mat)[None, None, None] @ density_locations[..., None]).squeeze(-1)
    density_locations = torch.tensor(density_locations, device="cpu", dtype=torch.float32)
    density_locations = density_locations.reshape(-1,3)
    return density_locations


def calculate_sliced_distance_matrix(density_calculation_locations, rmax, batch_size=10000):
    N = density_calculation_locations.shape[0]

    nonzero_indexes = []
    for row_indexes in torch.arange(N).split(batch_size):
        row_indexes = row_indexes.to(density_calculation_locations.device)
        index_points = density_calculation_locations[row_indexes]
        distances = (index_points[:, None] - density_calculation_locations[None]).norm(dim=-1)
        index_mask = distances < rmax
        nonzero = index_mask.nonzero()
        nonzero[...,0] = row_indexes[nonzero[...,0]]
        nonzero_indexes.append(nonzero.cpu())
    nonzero_indexes = torch.cat(nonzero_indexes, dim=0).to(density_calculation_locations.device)
    distances = (density_calculation_locations[nonzero_indexes[...,0]] - density_calculation_locations[nonzero_indexes[...,1]]).norm(dim=-1)
    return nonzero_indexes, distances

class SymmetryOperation:
    def __init__(self, atom_indexes, R, T):
        self.atom_indexes = atom_indexes
        self.R = R
        self.T = T
    
    def __call__(self, atom_locations):
        N = atom_locations.shape[0]
        transformed_atoms = (self.R[None] @ atom_locations[:, self.atom_indexes].reshape(N, -1, 3).permute(0,2,1)).permute(0,2,1) + self.T[None,None]
        return transformed_atoms, self.atom_indexes.reshape(-1)

class SymmetryOperations:
    def __init__(self, reference_atom_locations, zone_of_intereset, rmax, pdb, include_identity, device="cuda"):
        R_Ts = get_pdb_symmetries_R_T(gemmi.read_pdb(pdb), include_identity=include_identity)
        operations = [(torch.tensor(R, dtype=torch.float32, device=device), torch.tensor(T,
                    dtype=torch.float32, device=device)) for R,T in R_Ts]
        self.symmetry_operations = []
        for R,T in operations:
            transformed_atoms = (R @ reference_atom_locations.T).T + T
            distances = (transformed_atoms[:, None] - zone_of_intereset[None]).norm(dim=-1).min(dim=-1)[0]
            mask = distances < rmax
            if mask.any():
                atom_indexes = mask.nonzero().squeeze()
                self.symmetry_operations.append(SymmetryOperation(atom_indexes, R, T))
    
    def __call__(self, x_0_hat):
        if not len(self.symmetry_operations):
            return None, None
        indexes = []
        atom_locations = []
        for operation in self.symmetry_operations:
            transformed_atom_locations, operation_indexes = operation(x_0_hat)
            atom_locations.append(transformed_atom_locations)
            indexes.append(operation_indexes)

        return torch.cat(atom_locations, dim=1), torch.cat(indexes, dim=0)

class DensityGuidanceLossFunction(AbstractLossFunction):
    def __init__(self, reference_pdbs, full_pdb, chain, aligned_density_file, altloc_region=[], rmax=5, device="cpu", alignment_resiude_distance=8, batch_size=16):

        self.altloc_region = altloc_region

        ref_pdbs = []
        for ref_pdb in reference_pdbs:
            if os.path.exists(ref_pdb):
                ref_pdbs.append(ref_pdb)

        self.reference_pdbs = ref_pdbs
        self.chain = chain
        self.full_pdb = full_pdb
        self.aligned_density_file = aligned_density_file
        self.rmax = rmax
        self.device = device
        self.alignment_resiude_distance = alignment_resiude_distance
        self.batch_size = batch_size

        self.coordinates_gt, self.element_gt, self.bfactor, _, self.residue_range_atom_mask = get_sampler_pdb_inputs(reference_pdbs[0], altloc_region, device=device)
        self.rmsd_aligment_atom_mask = self._calcualte_rmsd_mask()

        self.zone_of_interest = self._calcualte_zone_of_intereset_atom_locations()

        # read and parse the density map
        self.density_map = gemmi.read_ccp4_map(aligned_density_file)
        self.density_locations = get_density_map_voxel_centroids(self.density_map).to(device)
        self.density_full_array = self.density_map.grid.array
        self.density_full_array = torch.tensor(self.density_full_array, device=device, dtype=torch.float32).flatten()

        self.density_calculation_locations, self.density_map_slicing_indexes, self.fo = self._get_density_calculation_locations_indexes_and_fo()

        self.density_estimator = AtomDensityEstimator(device)
        self.structure_elements_indicies = self.density_estimator.get_element_indexes(self.element_gt)
        self.structure_occupancy = (self.bfactor != 0).to(torch.float32)

        self.last_loss_value = None
        self.last_cosine_similarity = None
        self.last_density_aligment_loss = None
        self.last_substructure_loss = None

        # make this from the config file maybe
        self.bfactor[:,self.residue_range_atom_mask] *= 4 / batch_size

        # symmetry for the optimized chain
        reference_atom_locations = get_sampler_pdb_inputs(self.reference_pdbs[0], [0,-1], device)[0].squeeze()
        self.symmetry_operations = SymmetryOperations(reference_atom_locations, self.zone_of_interest, self.rmax + 10, self.full_pdb, False, self.device)
        self.fc_other_chains = self._get_fc_of_non_sampled_chains()

    def _get_fc_of_non_sampled_chains(self):
        pdb = gemmi.read_pdb(self.full_pdb)
        pdb.setup_entities()
        chain_index = next((i for i in range(len(pdb[0])) if pdb[0][i].name == self.chain))
        selected_chain = pdb[0][chain_index].clone()
        del pdb[0][chain_index]
        atom_locations = []
        elements = []
        bfactor = []
        occupancies = []
        # add only the ligands and waters from the optimized chain
        all_residues = [res for chain in pdb[0] for res in chain] + list(selected_chain.get_ligands()) + list(selected_chain.get_waters())
        for res in all_residues:
            for atom in res:
                atom_locations.append(list(atom.pos))
                elements.append(atom.element.name)
                bfactor.append(atom.b_iso)
                occupancies.append(atom.occ)
        atom_locations = torch.tensor(np.array(atom_locations), dtype=torch.float32, device=self.device)
        symmetry_operations = SymmetryOperations(atom_locations, self.zone_of_interest, self.rmax + 10, self.full_pdb, True, self.device)
        symetrized_atom_locations, indexes = symmetry_operations(atom_locations[None])
        elements = np.array(elements)[indexes.cpu()]
        bfactor = torch.tensor(np.array(bfactor), dtype=torch.float32, device=self.device)[indexes]
        occupancies = torch.tensor(np.array(occupancies), dtype=torch.float32, device=self.device)[indexes]
        element_indexes = self.density_estimator.get_element_indexes(elements)
        xyz = self.density_calculation_locations[None].repeat(1, 1, 1)
        fc = self.density_estimator(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)
        ##### OPTIMIZED CALL
        # fc = self.density_estimator.sample_fc_from_all_atom_optimized(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)[None]
        return fc

    def _calcualte_rmsd_mask(self):
        rmsd_aligment_atom_mask_residues = list(range(self.altloc_region[0] - self.alignment_resiude_distance, self.altloc_region[0])) + list(range(self.altloc_region[1] + 1, self.altloc_region[1] + self.alignment_resiude_distance + 1))
        return get_atom_mask(self.reference_pdbs[0], rmsd_aligment_atom_mask_residues).to(self.device)

    def _calcualte_zone_of_intereset_atom_locations(self):
        zones_of_interest = []
        for pdb in self.reference_pdbs:
            _, _, _, zone_of_interest, _ = get_sampler_pdb_inputs(pdb, self.altloc_region, device=self.device)
            zones_of_interest.append(zone_of_interest)
        return torch.cat(zones_of_interest, dim=0)

    def _get_density_calculation_locations_indexes_and_fo(self):
        distances_below_rmax = []
        for density_locations_batch in self.density_locations.split(1000000, 0):
            distances_below_rmax.append((density_locations_batch[:,None] - self.zone_of_interest[None]).norm(dim=-1).min(dim=-1)[0] < self.rmax)
        distances_below_rmax = torch.cat(distances_below_rmax)

        # a bug in mps where the tensor is too big, and the handling is wrong, https://discuss.pytorch.org/t/mps-large-tensor-handling-bug/214677
        device = distances_below_rmax.device
        distances_below_rmax = distances_below_rmax.cpu()
        indexes = distances_below_rmax.nonzero().squeeze()
        indexes = indexes.to(device)
        density_calculation_locations = self.density_locations[indexes]
        fo = self.density_full_array[indexes]
        fo = fo.clip(0)
        return density_calculation_locations, indexes, fo

    def save_state(self, structures, folder_path, **kwargs):
        with torch.no_grad():
            fc = self.calcualte_fc(structures)
        fc = fc.mean(dim=0)
        self.save_zone_of_interest_density(fc, f"{folder_path}/fc.ccp4")
        self.save_zone_of_interest_density(self.fo, f"{folder_path}/fo.ccp4")

    def save_zone_of_interest_density(self, density, file_path):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_map_slicing_indexes.cpu()] = density.cpu().numpy()
        self.density_map.grid.array[:] = array.reshape(self.density_map.grid.array.shape)
        self.density_map.write_ccp4_map(file_path)
        self.density_map.grid = original_grid
    
    def calcualte_fc(self, x_0_hat, xyz=None, occupancy=None, element_indecies=None, bfactor=None):
        batch_size = x_0_hat.shape[0]
        if occupancy is None:
            occupancy = self.structure_occupancy.repeat(batch_size, 1)
        if element_indecies is None:
            element_indecies = self.structure_elements_indicies.repeat(batch_size, 1)
        if xyz is None:
            xyz = self.density_calculation_locations[None].repeat(batch_size, 1, 1)
        if bfactor is None:
            bfactor = self.bfactor.repeat(x_0_hat.shape[0], 1)

        # add symmetry oprations
        transformed_atom_locations, indexes = self.symmetry_operations(x_0_hat)
        if transformed_atom_locations is not None:
            x_0_hat = torch.cat([x_0_hat, transformed_atom_locations], dim=1)
            occupancy = torch.cat([occupancy, occupancy[:,indexes]], dim=1)
            element_indecies = torch.cat([element_indecies, element_indecies[:,indexes]], dim=1)
            bfactor = torch.cat([bfactor, bfactor[:,indexes]], dim=1)

        fc = self.density_estimator(x_0_hat, occupancy, element_indecies, bfactor, xyz, rmax=self.rmax)
        
        #### OPTIMIZED CALL
        # fc = self.density_estimator.sample_fc_from_all_atom_optimized(x_0_hat, occupancy, element_indecies, bfactor, xyz, rmax=self.rmax)
        # if add_symmetry_chains:
        #     fc = fc + self.fc_other_chains
        return fc
    
    def wandb_log(self, x_0_hat):
        return ({"loss": self.last_loss_value, "cosine similarity": self.last_cosine_similarity, "bfactor": self.bfactor})

    def replace_non_residue_range_atoms(self, x_0_hat):
        """
            this function will replace the atoms that are not in the zone of intereset with the reference atoms
            this will also mask their gradients with respect to the density loss
            this makes it so they only way to change the densiy is to move the atoms in the zone of interest
        """
        new_x_0_hat = x_0_hat.clone()
        new_x_0_hat[:, ~self.residue_range_atom_mask] = self.coordinates_gt[:,~self.residue_range_atom_mask]
        return new_x_0_hat


    def __call__(self, x_0_hat, optimization_percetange):
        _, aligned_x_0_hat, R, T = self_aligned_rmsd(x_0_hat, self.coordinates_gt.repeat(x_0_hat.shape[0], 1, 1), (self.rmsd_aligment_atom_mask*0) + 1)
        aligned_x_0_hat = self.replace_non_residue_range_atoms(aligned_x_0_hat)
        fc = self.calcualte_fc(aligned_x_0_hat)

        fc_clone = fc.clone()
        fc = fc.mean(dim=0)

        mean = self.fo.mean()
        std = (self.fo.std() + 1e-6)

        fo = self.fo - mean
        fc = fc - mean

        fo = fo / std
        fc = fc / std

        loss = (0.5 * (fo[None] - fc).abs()).mean(0).sum()
        # print(torch.nn.functional.cosine_similarity(fo[None], fc_clone)) # Debug line

        self.last_loss_value = loss
        self.last_cosine_similarity = torch.nn.functional.cosine_similarity(fo, fc_clone.mean(dim=0)[None]).item()
        with torch.no_grad():
            new_x_0_hat = (R.permute(0,2,1)[:,None] @ (aligned_x_0_hat - T)[...,None]).squeeze(-1)

        return loss, new_x_0_hat

    ##### OPTIMIZED CALL
    # def __call__(self, x_0_hat, optimization_percetange):
    #     _, aligned_x_0_hat, R, T = self_aligned_rmsd(x_0_hat, self.coordinates_gt.repeat(x_0_hat.shape[0], 1, 1), (self.rmsd_aligment_atom_mask*0) + 1)
    #     aligned_x_0_hat = self.replace_non_residue_range_atoms(aligned_x_0_hat)
    #     # fc = self.calcualte_fc(aligned_x_0_hat, add_symmetry_chains=False)

    #     fc = self.calcualte_fc(aligned_x_0_hat, add_symmetry_chains=False)[None]
    #     # fc_clone = fc.clone()
    #     fc = fc.mean(dim=0)

    #     mean = self.fo.mean()
    #     std = (self.fo.std() + 1e-6)

    #     fo = self.fo - mean
    #     fc = fc - mean

    #     fo = fo / std
    #     fc = fc / std

    #     loss = (0.5 * (fo[None] - fc).abs()).mean(0).sum()
    #     # print(torch.nn.functional.cosine_similarity(fo[None], fc_clone))

    #     self.last_loss_value = loss
    #     # self.last_cosine_similarity = torch.nn.functional.cosine_similarity(fo, fc_clone.mean(dim=0)[None]).item()
    #     with torch.no_grad():
    #         new_x_0_hat = (R.permute(0,2,1)[:,None] @ (aligned_x_0_hat - T)[...,None]).squeeze(-1)

    #     return loss, new_x_0_hat
