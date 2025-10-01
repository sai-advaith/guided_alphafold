import torch
import os
from tqdm import tqdm
from src.protenix.metrics.rmsd import self_aligned_rmsd
from src.utils.density_guidance.density_estimator import AtomDensityEstimator
from src.utils.io import load_pdb_atom_locations, get_sampler_pdb_inputs, delete_hydrogens, extract_chain_ligand_water_tensors, AMINO_ACID_ATOMS_ORDER, remove_headers
import gemmi
from src.losses.density_loss_function import get_density_map_voxel_centroids, SymmetryOperations
import numpy as np
import gemmi
from biotite.structure.io.pdb import PDBFile
from src.utils.relaxation import relax_pdb
from src.utils.pdb_parsing import find_bonded_pairs
from typing import List
from src.utils.process_pipeline_inputs.extract_metadata import aa_map
from Bio.PDB.Polypeptide import is_aa

from src.utils.phenix_manager import PhenixManager
from src.utils.ccp4_manager import CCP4Manager

def is_water(residue):
    return residue.name in ['HOH', 'WAT'] and residue.het_flag

def is_ligand(residue):
    standard_residues = AMINO_ACID_ATOMS_ORDER.keys()  # standard amino acids and nucleotides
    return residue.het_flag and residue.name not in standard_residues

def get_full_pdb_residue_range(full_pdb, generated_pdb, residue_range, chain_name):
    reference_chain = gemmi.read_pdb(generated_pdb)[0][0]
    sub_sequence = "".join([aa_map[reference_chain[i].name] for i in range(residue_range[0] - 1, residue_range[1])])
    pdb = gemmi.read_pdb(full_pdb)
    chain_index = [i for i in range(len(pdb[0])) if pdb[0][i].name == chain_name][0]
    chain = pdb[0][chain_index]
    # chain_residue_range_start = "".join([aa_map[res.name] for res in chain]).find(sub_sequence)
    filtered_chain = [res for res in chain if (not is_water(res) and not is_ligand(res))]

    chain_residue_range_start = "".join([aa_map[res.name] for res in filtered_chain]).find(sub_sequence)
    chain_residue_range = [chain_residue_range_start, chain_residue_range_start + len(sub_sequence)]
    return chain_residue_range

def merge_pdbs(full_pdb, chain_name, pdb_files, occupancies, residue_range):
    reference_chains = [gemmi.read_pdb(file)[0][0] for file in pdb_files]
    chain_residue_range = get_full_pdb_residue_range(full_pdb, pdb_files[0], residue_range, chain_name)
    pdb = gemmi.read_pdb(full_pdb)
    chain_index = [i for i in range(len(pdb[0])) if pdb[0][i].name == chain_name][0]
    chain = pdb[0][chain_index]

    for i,residue in enumerate(chain[chain_residue_range[0]:chain_residue_range[1]]):
        atoms_to_delete = []
        for atom in residue:
            atoms_to_delete.append(atom.clone())
        for atom in atoms_to_delete:
            residue.remove_atom(atom.name, atom.altloc, atom.element)
        for altloc_index,(reference_chain, occupancy) in enumerate(zip(reference_chains, occupancies)):
            for atom in reference_chain[residue_range[0] - 1 + i]:
                new_atom = atom.clone()
                new_atom.occ = occupancy
                new_atom.altloc = chr(ord("A") + altloc_index)
                residue.add_atom(new_atom)
    return pdb

def get_density_voxel_center_locations(density_map):
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

def prod(*arguments):
    a = arguments[0]
    for other in arguments[1:]:
        a = a * other
    return a

class PDBManager:
    def __init__(self, pdb_file_path, residue_range, device="cuda"):
        self.pdb_file_path = pdb_file_path
        self.device = device
        self.pdb_object = PDBFile.read(self.pdb_file_path)
        self.atom_array = self.pdb_object.get_structure(
            model=1,
            altloc="all",
            extra_fields=["occupancy"]
        )
        self.atom_locations = torch.tensor(self.atom_array.coord, dtype=torch.float32, device=self.device)
        self.overall_bfactors = torch.tensor(self.pdb_object.get_b_factor().squeeze(), dtype=torch.float32, device=self.device)
        self.elements = np.array(self.atom_array.element)
        self.element_indexes = None
        self.residue_range_mask = torch.tensor(np.isin(self.atom_array.res_id, np.arange(residue_range[0], residue_range[1] + 1)), dtype=torch.bool, device=self.device)
        self.residue_range_bfactor = self.overall_bfactors[self.residue_range_mask].clone()
        self.occupancy = torch.tensor(self.atom_array.occupancy, dtype=torch.float32, device=self.device)

    def parameters(self):
        return [self.residue_range_bfactor]

    def detach(self):
        self.residue_range_bfactor = self.residue_range_bfactor.detach()

    def requries_grad(self, requries_grad=True):
        self.residue_range_bfactor.requires_grad_(requries_grad)

    @property
    def bfactors(self):
        bfactor = self.overall_bfactors.clone()
        bfactor[self.residue_range_mask] = self.residue_range_bfactor
        return bfactor

    def to(self, device):
        self.atom_locations = self.atom_locations.to(device)
        self.residue_range_mask = self.residue_range_mask.to(device)
    
    @property
    def region_of_interest_atoms(self):
        return self.atom_locations[self.residue_range_mask]

    def save_structure(self, file_path):
        self.atom_array.coord = self.atom_locations.cpu().detach().numpy()
        self.pdb_object.set_structure(self.atom_array, 1)
        self.pdb_object.write(file_path)

    @staticmethod
    def calculate_pairwise_deistances(managers):
        atom_locations = torch.stack([manager.region_of_interest_atoms for manager in managers])
        disatnces = (atom_locations[None] - atom_locations[:,None]).norm(dim=-1).mean(dim=-1)
        disatnces = disatnces - disatnces.mean()
        disatnces = disatnces / disatnces.std()
        return disatnces

class DensityLogger:
    def __init__(self, density_file, pdb_file_paths, residue_range, rmax=3, device="cuda", raw_pdb_file_path=None, chain_id=None, ref_pdb_path=None):
        self.density_file = density_file
        self.residue_range = residue_range
        self.device = device
        self.raw_pdb_file_path = raw_pdb_file_path
        self.chain_id = chain_id
        self.pdb_file_objs = [PDBManager(file, self.residue_range, device=self.device) for file in pdb_file_paths]

        self.rmax = rmax

        self.density_map = gemmi.read_ccp4_map(density_file)
        density_locations = torch.tensor(get_density_voxel_center_locations(self.density_map), dtype=torch.float32, device=self.device)
        density_full_array = self.density_map.grid.array
        density_full_array = torch.tensor(density_full_array, device=self.device, dtype=torch.float32).flatten()

        self.zone_of_interest = torch.cat([pdb.region_of_interest_atoms for pdb in self.pdb_file_objs])

        distances_below_rmax = []
        for density_locations_batch in density_locations.split(1000000, 0):
            distances_below_rmax.append((density_locations_batch[:,None] - self.zone_of_interest[None]).norm(dim=-1).min(dim=-1)[0] < self.rmax)
        distances_below_rmax = torch.cat(distances_below_rmax)

        device = distances_below_rmax.device
        distances_below_rmax = distances_below_rmax.cpu()
        indexes = distances_below_rmax.nonzero().squeeze()

        self.density_map_slicing_indexes = indexes
        indexes = indexes.to(device)
        self.density_calculation_locations = density_locations[indexes]
        self.density_estimator = AtomDensityEstimator(self.device)
        self.fc_other_chains = self._get_fc_of_non_sampled_chains()

        # Structure waters and ligands / small molecules
        het_atoms_properties = extract_chain_ligand_water_tensors(self.raw_pdb_file_path, self.chain_id, device=self.device)
        self.het_atoms_coordinates = het_atoms_properties[0]
        self.het_atoms_bfactor = het_atoms_properties[1]
        self.het_atoms_occupancy = het_atoms_properties[2]
        self.het_atoms_elements = het_atoms_properties[3]

        self.reference_atom_locations = get_sampler_pdb_inputs(ref_pdb_path, [0,-1], device)[0].squeeze()
        self.symmetry_operations = SymmetryOperations(self.reference_atom_locations, self.zone_of_interest, self.rmax + 10.0, ref_pdb_path, self.residue_range, self.device)

    def _get_fc_of_non_sampled_chains(self):
        pdb = gemmi.read_pdb(self.raw_pdb_file_path)
        pdb.setup_entities()
        chain_index = next((i for i in range(len(pdb[0])) if pdb[0][i].name == self.chain_id))
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
        symmetry_operations = SymmetryOperations(atom_locations, self.zone_of_interest, self.rmax + 10.0, self.raw_pdb_file_path, True, self.device)
        symetrized_atom_locations, indexes = symmetry_operations(atom_locations[None])
        if indexes is not None:
            elements = np.array(elements)[indexes.cpu()]
            bfactor = torch.tensor(np.array(bfactor), dtype=torch.float32, device=self.device)[indexes]
            occupancies = torch.tensor(np.array(occupancies), dtype=torch.float32, device=self.device)[indexes]
            element_indexes = self.density_estimator.get_element_indexes(elements)
            xyz = self.density_calculation_locations[None].repeat(1, 1, 1)
            fc = self.density_estimator(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)
            ##### OPTIMIZED CALL
            # fc = self.density_estimator.sample_fc_from_all_atom_optimized(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)[None]
            return fc
        else:
            return torch.tensor(0, device=self.device)

    def get_density_full_grid(self, density):
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_map_slicing_indexes] = density.cpu().numpy()
        self.density_map.grid = original_grid
        return array.reshape(self.density_map.grid.array.shape)

    def save_zone_of_interest_density(self, density, file_path):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_map_slicing_indexes] = density.cpu().numpy()
        self.density_map.grid.array[:] = array.reshape(self.density_map.grid.array.shape)
        self.density_map.write_ccp4_map(file_path)
        self.density_map.grid = original_grid

    def calculate_density(self, pdbs: List[PDBManager]):
        for pdb in pdbs:
            if pdb.element_indexes is None:
                pdb.element_indexes = self.density_estimator.get_element_indexes(pdb.elements)
        
        atom_locations = torch.stack([pdb.atom_locations for pdb in pdbs], dim=0)
        occupancy = torch.ones_like(atom_locations[..., 0])
        element_indexes = torch.stack([pdb.element_indexes for pdb in pdbs], dim=0)
        bfactors = torch.stack([pdb.bfactors for pdb in pdbs], dim=0)

        xyz = self.density_calculation_locations[None].repeat(len(pdbs), 1, 1)

        transformed_atom_locations, indices = self.symmetry_operations(atom_locations)
        if transformed_atom_locations is not None:
            atom_locations = torch.cat([atom_locations, transformed_atom_locations], dim=1)
            occupancy = torch.cat([occupancy, occupancy[:, indices]], dim=1)
            element_indexes = torch.cat([element_indexes, element_indexes[:, indices]], dim=1)
            bfactors = torch.cat([bfactors, bfactors[:,indices]], dim=1)

        if self.het_atoms_coordinates is not None:
            het_atom_element_indices = self.density_estimator.get_element_indexes(self.het_atoms_elements)

            het_atoms_coordinates = self.het_atoms_coordinates.repeat(atom_locations.shape[0], 1, 1)
            het_atoms_occupancy = self.het_atoms_occupancy.repeat(atom_locations.shape[0], 1)
            het_atom_element_indices = het_atom_element_indices.repeat(atom_locations.shape[0], 1)
            het_atoms_bfactor = self.het_atoms_bfactor.repeat(atom_locations.shape[0], 1)

            # Repeat on dim 0
            atom_locations = torch.cat([atom_locations, het_atoms_coordinates], dim=1)
            occupancy = torch.cat([occupancy, het_atoms_occupancy], dim=1)
            element_indexes = torch.cat([element_indexes, het_atom_element_indices], dim=1)
            bfactors = torch.cat([bfactors, het_atoms_bfactor], dim=1)

        fc = self.density_estimator(atom_locations, occupancy, element_indexes, bfactors, xyz, self.rmax)
        ##### OPTIMIZED CALL
        # fc = self.density_estimator.sample_fc_from_all_atom_optimized(atom_locations, occupancy, element_indexes, bfactors, xyz, self.rmax)[None]
        fc = fc + self.fc_other_chains
        return self.get_density_full_grid(fc)

    def save_density(self, density, file_path):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        self.density_map.grid.array[:] = density
        self.density_map.write_ccp4_map(file_path)
        self.density_map.grid = original_grid

class OMPMetric:
    def __init__(self,
                samples_directory,
                rmax,
                reference_density_file,
                residue_range,
                altloc_a_path,
                altloc_b_path=None,
                bond_max_threshold=2.1,
                device="cpu",
                raw_pdb_file_path=None,
                chain_id=None,
                mtz_file_path=None,
                reference_pdb_file_path=None,
                pdb_id=None,
                pdb_residue_range=None,
                ccp4_setup_sh=None,
                phenix_setup_sh=None,
                map_type="end"
        ):
        self.samples_directory = samples_directory
        self.reference_density_file = reference_density_file
        self.density_map = gemmi.read_ccp4_map(reference_density_file)
        self.rmax = rmax
        self.altloc_a_path = altloc_a_path
        if altloc_b_path is None or not os.path.exists(altloc_b_path):
            altloc_b_path = altloc_a_path
        self.altloc_b_path = altloc_b_path

        self.raw_pdb_file = raw_pdb_file_path
        self.chain_id = chain_id
        self.pdb_id = pdb_id
        self.reference_pdb_file_path = reference_pdb_file_path
        self.map_type = map_type

        het_atoms_properties = extract_chain_ligand_water_tensors(self.raw_pdb_file, self.chain_id, device=device)
        self.het_atoms_coordinates = het_atoms_properties[0]
        self.het_atoms_bfactor = het_atoms_properties[1]
        self.het_atoms_occupancy = het_atoms_properties[2]
        self.het_atoms_elements = het_atoms_properties[3]

        self.reference_pdbs = (self.altloc_a_path, self.altloc_b_path)
        self.residue_range = residue_range
        self.pdb_residue_range = pdb_residue_range
        self.bond_max_threshold = bond_max_threshold
        self.device = device

        coordinates_gt, element_gt, bfactor_gt, zone_of_interest, residue_range_atom_mask = get_sampler_pdb_inputs(altloc_a_path, residue_range, device=self.device)
        self.altloc_A = {
            "structure_coordinates": coordinates_gt,
            "elements": element_gt,
            "bfactor": bfactor_gt,
            "zone_of_interest": zone_of_interest,
            "residue_mask": residue_range_atom_mask
        }
        if altloc_b_path is not None:
            coordinates_gt, element_gt, bfactor_gt, zone_of_interest, residue_range_atom_mask = get_sampler_pdb_inputs(altloc_b_path, self.residue_range, device=self.device)
            self.altloc_B = {
                "structure_coordinates": coordinates_gt,
                "elements": element_gt,
                "bfactor": bfactor_gt,
                "zone_of_interest": zone_of_interest,
                "residue_mask": residue_range_atom_mask
            }
        else:
           self.altloc_B = None
        zone_of_interest = [self.altloc_A["zone_of_interest"]]
        if self.altloc_B is not None:
            zone_of_interest += [self.altloc_B["zone_of_interest"]]
        self.zone_of_interest = torch.cat(zone_of_interest, dim=0)

        self.density_calculation_locations, self.sliced_density = self.get_zone_of_interest_grid_params()

        self.density_estimator = AtomDensityEstimator(self.device)

        self.bfactor = self.altloc_A["bfactor"].mean()

        # self.symmetry_operations = SymmetryOperations(self.altloc_A["structure_coordinates"].squeeze(), self.zone_of_interest, self.rmax + 10.0, self.reference_pdbs[0], self.residue_range, self.device)
        self.symmetry_operations = SymmetryOperations(self.altloc_A["structure_coordinates"].squeeze(), self.zone_of_interest, self.rmax + 10.0, self.reference_pdbs[0], self.residue_range, self.device)
        self.fc_other_chains = self._get_fc_of_non_sampled_chains()

        # TODO:Parse this
        self.mtz_file_path = mtz_file_path
        self.ccp4_setup_sh = ccp4_setup_sh
        self.phenix_setup_sh = phenix_setup_sh
        # Managers
        self.ccp4_manager = CCP4Manager(self.ccp4_setup_sh)
        self.phenix_manager = PhenixManager(self.phenix_setup_sh)

    def _get_fc_of_non_sampled_chains(self):
        pdb = gemmi.read_pdb(self.raw_pdb_file)
        pdb.setup_entities()
        chain_index = next((i for i in range(len(pdb[0])) if pdb[0][i].name == self.chain_id))
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
        symmetry_operations = SymmetryOperations(atom_locations, self.zone_of_interest, self.rmax + 10, self.raw_pdb_file, True, self.device)
        symetrized_atom_locations, indexes = symmetry_operations(atom_locations[None])
        if indexes is not None:
            elements = np.array(elements)[indexes.cpu()]
            bfactor = torch.tensor(np.array(bfactor), dtype=torch.float32, device=self.device)[indexes]
            occupancies = torch.tensor(np.array(occupancies), dtype=torch.float32, device=self.device)[indexes]
            element_indexes = self.density_estimator.get_element_indexes(elements)
            xyz = self.density_calculation_locations[None].repeat(1, 1, 1)
            fc = self.density_estimator(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)
            ##### OPTIMIZED CALL
            # fc = self.density_estimator.sample_fc_from_all_atom_optimized(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)[None]
            return fc
        else:
            torch.tensor(0, device=self.device)
    def get_zone_of_interest_grid_params(self):
        # Get the density locations
        grid_locations = get_density_map_voxel_centroids(self.density_map).to(self.device)

        density_full_array = self.density_map.grid.array
        density_full_array = torch.tensor(density_full_array, device=self.device, dtype=torch.float32).flatten()
        
        distances_below_rmax = []
        for density_locations_batch in grid_locations.split(1000000, 0):
            distances_below_rmax.append((density_locations_batch[:,None] - self.zone_of_interest[None]).norm(dim=-1).min(dim=-1)[0] < self.rmax)
        distances_below_rmax = torch.cat(distances_below_rmax)

        # a bug in mps where the tensor is too big, and the handling is wrong, https://discuss.pytorch.org/t/mps-large-tensor-handling-bug/214677
        device = distances_below_rmax.device
        distances_below_rmax = distances_below_rmax.cpu()
        indexes = distances_below_rmax.nonzero().squeeze()
        
        self.density_map_slicing_indexes = indexes
        indexes = indexes.to(device)
        density_calculation_locations = grid_locations[indexes]
        sliced_density = density_full_array[indexes]
        sliced_density = sliced_density.clip(0)

        # self.save_zone_of_interest_density(sliced_density, "fo1.ccp4")

        return density_calculation_locations, sliced_density

    def get_r_value_r_free(self, pdb_file_path):
        return self.phenix_manager.calculate_rwork_rfree(pdb_file_path, self.mtz_file_path)

    def get_un_broken_samples(self):
        """
            this function will go over the files in self.samples_directory and return the files of the proteins which are not borken
        """
        pdb_files = [os.path.join(self.samples_directory, file) for file in os.listdir(self.samples_directory) if file.endswith(".pdb")]
        un_broken_files = []
        for pdb_file in pdb_files:
            structure = gemmi.read_pdb(pdb_file)
            model = structure[0]
            chain = model[0]
            bonded_atom_pairs = find_bonded_pairs(chain)
            bond_distances = np.array([np.linalg.norm(np.array(list(pair[0][0].pos)) - np.array(list(pair[1][0].pos))) for pair in bonded_atom_pairs])
            if bond_distances.max() < self.bond_max_threshold:
                un_broken_files.append(pdb_file)
        return un_broken_files

    def save_zone_of_interest_density(self, density, file_path):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_map_slicing_indexes] = density.cpu().numpy()
        self.density_map.grid.array[:] = array.reshape(self.density_map.grid.array.shape)
        self.density_map.write_ccp4_map(file_path)
        self.density_map.grid = original_grid

    def calculate_densities(self, structures):
        # Create element indices in the periodic table
        element_gt = self.altloc_A["elements"]
        element_indexes = self.density_estimator.get_element_indexes(element_gt)
        element_indexes = element_indexes.to(self.device)
        bfactor = self.altloc_A["bfactor"].clone()
        bfactor[:, self.altloc_A["residue_mask"]] = self.bfactor

        bfactor = bfactor.repeat(structures.shape[0], 1)
        element_indexes = element_indexes.repeat(structures.shape[0], 1)
        occupancy = torch.ones_like(structures[..., 0])

        xyz = self.density_calculation_locations[None].repeat(structures.shape[0], 1, 1)

        # Add symmetry operations
        transformed_atom_locations, indices = self.symmetry_operations(structures)
        if transformed_atom_locations is not None:
            structures = torch.cat([structures, transformed_atom_locations], dim=1)
            occupancy = torch.cat([occupancy, occupancy[:, indices]], dim=1)
            element_indexes = torch.cat([element_indexes, element_indexes[:, indices]], dim=1)
            bfactor = torch.cat([bfactor, bfactor[:,indices]], dim=1)

        # Append ligands and waters
        if self.het_atoms_coordinates is not None:
            het_atom_element_indices = self.density_estimator.get_element_indexes(self.het_atoms_elements)

            het_atoms_coordinates = self.het_atoms_coordinates.repeat(structures.shape[0], 1, 1)
            het_atoms_occupancy = self.het_atoms_occupancy.repeat(structures.shape[0], 1)
            het_atom_element_indices = het_atom_element_indices.repeat(structures.shape[0], 1)
            het_atoms_bfactor = self.het_atoms_bfactor.repeat(structures.shape[0], 1)

            # Repeat on dim 0
            structures = torch.cat([structures, het_atoms_coordinates], dim=1)
            occupancy = torch.cat([occupancy, het_atoms_occupancy], dim=1)
            element_indexes = torch.cat([element_indexes, het_atom_element_indices], dim=1)
            bfactor = torch.cat([bfactor, het_atoms_bfactor], dim=1)

        fc = self.density_estimator(structures, occupancy, element_indexes, bfactor, xyz, rmax=self.rmax)
        ##### OPTIMIZED CALL
        # fc = self.density_estimator.sample_fc_from_all_atom_optimized(structures, occupancy, element_indexes, bfactor, xyz, rmax=self.rmax)[None]
        if self.fc_other_chains is not None:
            return fc + self.fc_other_chains
        else:
            return fc

    def measure_structures_cosine_similarity(self, structures):
        fc = self.calculate_densities(structures)
        fc = fc.mean(dim=0)
        return torch.nn.functional.cosine_similarity(fc[None], self.sliced_density[None]).squeeze()

    def get_optimized_cosine_similarity(self, structures, optimize_bfactor=True):
        if optimize_bfactor:
            self.optimize_bfactor(structures)
        return self.measure_structures_cosine_similarity(structures)

    def normalized(self, density):
        density = density - density.mean()
        density = density / (density.std() + 1e-6)
        return density

    def get_shifted_cosine_similarity(self, structures):
        fc = self.calculate_densities(structures)
        fc = fc.mean(dim=0)
        fc_normalized = self.normalized(fc)
        fo_normalized = self.normalized(self.sliced_density)
        return torch.nn.functional.cosine_similarity(fc_normalized[None], fo_normalized[None]).squeeze()

    def optimize_bfactor(self, structures, steps=500, lr=1):
        self.bfactor = self.altloc_A["bfactor"].mean()
        self.bfactor.requires_grad = True
        optimizer = torch.optim.Adam([self.bfactor], lr=lr)
        for _ in tqdm(range(steps), "optimizing bfactor"):
            fc = self.calculate_densities(structures)
            fc = fc.mean(dim=0)
            loss = torch.nn.functional.mse_loss(fc, self.sliced_density)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.bfactor.grad = None
        self.bfactor = self.bfactor.detach()

    def select_optimal_subset(self, files, max_samples=5):
        temp_dir_name = os.path.join("density_omp_temp_dir", str(os.getpid()))
        os.makedirs(temp_dir_name, exist_ok=True)

        with torch.no_grad():
            structures = torch.cat([load_pdb_atom_locations(file, self.device) for file in files])
            ref_pose = self.altloc_A["structure_coordinates"].repeat(structures.shape[0], 1, 1)
            # mask = (~self.altloc_A["residue_mask"])
            mask = torch.ones_like(~self.altloc_A["residue_mask"]) # For full protein
            _, structures, _, _ = self_aligned_rmsd(structures, ref_pose, mask)

        # GT cosine similarity
        gt_cosine_similarity = self.get_optimized_cosine_similarity(torch.cat([self.altloc_A["structure_coordinates"], self.altloc_B["structure_coordinates"]])).item()

        possible_indexes = list(range(len(structures)))
        structures_cosine_similarity = [self.get_optimized_cosine_similarity(structures[index][None]).item() for index in possible_indexes]
        k = min(structures.shape[0], 10)
        _, topk_indices = torch.topk(torch.tensor(structures_cosine_similarity), k=k)
        possible_indexes = topk_indices.cpu().detach().numpy().tolist()


        selected_indexes = []
        r_free_history, r_work_history = [], []
        max_cosine_similarity = 0
        prev_r_free, prev_r_work = None, None
        for i in range(max_samples):
            if len(possible_indexes) == 0:
                break

            structures_cosine_similarity = [self.get_optimized_cosine_similarity(torch.cat([structures[i][None] for i in selected_indexes + [index]])).item() for index in possible_indexes]
            arg_max = np.argmax(structures_cosine_similarity)
            current_max_cosine_similarity = structures_cosine_similarity[arg_max]
            if current_max_cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = current_max_cosine_similarity
                index = possible_indexes[arg_max]

                # TODO: Check if selection index actually improves R-free (do it on the entire R-free set for now)
                intermediate_selection = selected_indexes[:]
                intermediate_selection.append(index)
                file_names = [files[i] for i in intermediate_selection]
                occupancies = [1/len(file_names)] * len(file_names)

                merged_pdb = merge_pdbs(self.raw_pdb_file, self.chain_id, file_names, occupancies, self.residue_range)
                merged_pdb_path = os.path.join(temp_dir_name, "merged.pdb")
                merged_pdb.write_pdb(merged_pdb_path)

                r_work_new, r_free_new = self.get_r_value_r_free(merged_pdb_path)

                # Either r free gets worse, or it stays the same and r work gets worse
                if prev_r_free is None or prev_r_work is None:
                    prev_r_free = r_free_new
                    prev_r_work = r_work_new
                else:
                    if r_free_new > prev_r_free or (r_free_new == prev_r_free and r_work_new > prev_r_work):
                        break

                prev_r_free = r_free_new
                prev_r_work = r_work_new

                r_free_history.append((prev_r_free))
                r_work_history.append((prev_r_work))

                # If it improves r free, include
                selected_indexes.append(index)
                possible_indexes.remove(index)
            else:
                break

        # Selection over
        selected_files = [files[i] for i in selected_indexes]
        selected_sturctures = torch.cat([structures[i][None] for i in selected_indexes])

        # Save best ensembles
        occupancies = [1/len(selected_files)] * len(selected_files)
        merged_pdb_obj = merge_pdbs(self.raw_pdb_file, self.chain_id, selected_files, occupancies, self.residue_range)
        ensemble_folder = os.path.dirname(files[0])
        merged_pdb_path = os.path.join(ensemble_folder, "omp", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_omp_{self.map_type}_guided.pdb")
        merged_pdb_obj.write_pdb(merged_pdb_path)

        self.optimize_bfactor(selected_sturctures)
        r_values = (r_work_history[-1], r_free_history[-1])

        gt_r_values = self.get_r_value_r_free(self.raw_pdb_file)
        return selected_files, selected_sturctures, max_cosine_similarity, gt_cosine_similarity, r_values, gt_r_values, merged_pdb_path

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

    def calculate_altlocs_assignment_distances(self, samples):
        """
            returns the distances to the closest altloc, and the assigment for that altloc
        """
        # Get relevant atom locations
        altloc_a_slice = self.altloc_A["zone_of_interest"].clone()
        altloc_b_slice = self.altloc_B["zone_of_interest"].clone()

        # RMSD from each altloc
        distances_x_a = []
        distances_x_b = []
        d_a_b = (altloc_b_slice - altloc_a_slice).norm(dim=-1).mean().item()
        
        distances = []
        for sample in samples:
            sample_slice = sample[self.altloc_A['residue_mask']]
            a_distance = (sample_slice - altloc_a_slice).norm(dim=-1).mean().item()
            b_distance = (sample_slice - altloc_b_slice).norm(dim=-1).mean().item()
            normalized_distance = -(1 - (a_distance / b_distance))
            distances.append(normalized_distance)
            distances_x_a.append(a_distance)
            distances_x_b.append(b_distance)
            
        return distances, distances_x_a, distances_x_b, d_a_b

    def get_altlocs_density(self):
        altloc_a_density = self.calculate_densities(self.altloc_A["structure_coordinates"].clone())
        if self.altloc_B is not None:
            altloc_b_density = self.calculate_densities(self.altloc_B["structure_coordinates"].clone())
            return (altloc_a_density + altloc_b_density) / 2
        else:
            return altloc_a_density

    def write_pdb_file(self, structure, file_name):
        i = 0
        new_structure = gemmi.read_structure(file_name).clone()
        for model in new_structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atom.pos = gemmi.Position(*structure[i].clone().cpu().detach().numpy())
                        i += 1
        # Overwrite it with aligned files
        new_structure.write_pdb(file_name)
        return file_name

    def align_relaxed_samples(self, files):
        aligned_files = []
        with torch.no_grad():
            structures = torch.cat([load_pdb_atom_locations(file, self.device) for file in files])
            ref_pose = self.altloc_A["structure_coordinates"].repeat(structures.shape[0], 1, 1)
            mask = torch.ones_like(ref_pose[0, :, 0])
            _, structures, _, _ = self_aligned_rmsd(structures, ref_pose, mask)

            for i in range(len(files)):
                structure_i, file_i = structures[i], files[i]
                aligned_file_i = self.write_pdb_file(structure_i, file_i)
                aligned_files.append(aligned_file_i)

        return aligned_files

    def get_density_all_grid(self, density):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_map_slicing_indexes] = density.cpu().numpy()
        self.density_map.grid = original_grid
        return array.reshape(self.density_map.grid.array.shape)

    def run_refmac_refinement(self, xyzin, hklout=None, xyzout=None):
        return self.ccp4_manager.run_refmac_refinement(xyzin, self.mtz_file_path, hklout, xyzout)
        # Refmac input string

    def get_rscc_metrics(self, pdb_file_path):
        return self.phenix_manager.get_rscc_metrics(pdb_file_path, self.mtz_file_path, self.chain_id, self.pdb_residue_range)

    def run(self, unguided=False):
        relaxed_files = []
        os.makedirs(os.path.join(self.samples_directory, "omp"), exist_ok=True)
        for file_name in os.listdir(self.samples_directory):
            if file_name.endswith(".pdb") and ("merged" not in file_name):
                relaxed_files.append(os.path.join(self.samples_directory, file_name))

        selected_files, _, sample_cosine_similarity, gt_cosine_similarity, r_values, gt_r_values, merged_pdb_path = self.select_optimal_subset(relaxed_files)

        if unguided:
            suffix = "_unguided"
        else:
            suffix = f"_{self.map_type}_guided"

        merged_mtz_refined_path = os.path.join(self.samples_directory, "omp", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_omp_ref{suffix}.mtz")
        merged_pdb_refined_path = os.path.join(self.samples_directory, "omp", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_omp_ref{suffix}.pdb")
        cif_path = os.path.join(self.samples_directory, "omp", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_omp_ref{suffix}.mmcif")
        merged_refined_metrics = self.run_refmac_refinement(merged_pdb_path, hklout=merged_mtz_refined_path, xyzout=merged_pdb_refined_path)

        # RSCC values
        ensemble_rscc = self.get_rscc_metrics(merged_pdb_refined_path)
        pdb_rscc = self.get_rscc_metrics(self.raw_pdb_file)

        # Remove headers
        remove_headers(merged_pdb_path, merged_pdb_path)
        remove_headers(merged_pdb_refined_path, merged_pdb_refined_path)

        if os.path.exists(cif_path):
            os.remove(cif_path)

        if np.isnan(merged_refined_metrics["initial_R"]):
            r_values_refined = (np.nan, np.nan)
        else:
            r_values = (merged_refined_metrics["initial_R"], merged_refined_metrics["initial_Rfree"])
            r_values_refined = (merged_refined_metrics["final_R"], merged_refined_metrics["final_Rfree"])

        pdb_refined_metrics = self.run_refmac_refinement(self.raw_pdb_file)
        if np.isnan(pdb_refined_metrics["initial_R"]):
            gt_r_values_refined = (np.nan, np.nan)
        else:
            gt_r_values = (pdb_refined_metrics["initial_R"], pdb_refined_metrics["initial_Rfree"])
            gt_r_values_refined = (pdb_refined_metrics["final_R"], pdb_refined_metrics["final_Rfree"])


        metrics_dict = {
            "pdb file names": [os.path.basename(sel_file) for sel_file in selected_files],
            "refined file": merged_pdb_refined_path,
            "cosine similarity": sample_cosine_similarity,
            "pdb cosine similarity": gt_cosine_similarity,
            "ensemble rscc": ensemble_rscc,
            "pdb rscc": pdb_rscc,
            "r_free_r_work": r_values,
            "r_free_r_work refined": r_values_refined,
            "pdb R-free": gt_r_values[1],
            "pdb R-work": gt_r_values[0],
            "pdb R-free refined": gt_r_values_refined[1],
            "pdb R-work refined": gt_r_values_refined[0],
        }
        print(f"selected files: {selected_files}")
        return metrics_dict
