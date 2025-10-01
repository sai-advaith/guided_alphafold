import torch 
import numpy as np 
import gemmi
import os
from tqdm import tqdm
from src.losses.density_loss_function import DensityGuidanceLossFunction
from src.utils.io import get_sampler_pdb_inputs, get_atom_mask, get_non_missing_atom_mask, AMINO_ACID_ATOMS_ORDER, remove_headers
from src.utils.density_guidance.density_estimator import AtomDensityEstimator
from src.losses.density_loss_function import SymmetryOperations

from src.utils.phenix_manager import PhenixManager
from src.utils.ccp4_manager import CCP4Manager

from biotite.structure.io.pdb import PDBFile
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import List
import json
from multiprocessing import Pool
import shutil
import json
from src.utils.process_pipeline_inputs.extract_metadata import aa_map
from src.utils.io import extract_chain_ligand_water_tensors

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

        reference_atom_locations = get_sampler_pdb_inputs(ref_pdb_path, [0,-1], device)[0].squeeze()
        self.symmetry_operations = SymmetryOperations(reference_atom_locations, self.zone_of_interest, self.rmax + 10.0, ref_pdb_path, self.residue_range, self.device)

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
        elements = np.array(elements)[indexes.cpu()]
        bfactor = torch.tensor(np.array(bfactor), dtype=torch.float32, device=self.device)[indexes]
        occupancies = torch.tensor(np.array(occupancies), dtype=torch.float32, device=self.device)[indexes]
        element_indexes = self.density_estimator.get_element_indexes(elements)
        xyz = self.density_calculation_locations[None].repeat(1, 1, 1)
        fc = self.density_estimator(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)

        ##### OPTIMIZED CALL
        # fc = self.density_estimator.sample_fc_from_all_atom_optimized(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)[None]
        return fc


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

class DensityCalculator:
    def __init__(self, density_file, reference_pdbs, raw_pdb_file_path, chain_id, residue_range, rmax=5, device="cuda"):
        self.residue_range = residue_range
        self.reference_pdbs = [PDBManager(file, self.residue_range, device=device) for file in reference_pdbs]
        self.rmax = rmax
        self.device = device
        self.raw_pdb_file_path = raw_pdb_file_path
        self.chain_id = chain_id

        self.density_map = gemmi.read_ccp4_map(density_file)
        density_locations = torch.tensor(get_density_voxel_center_locations(self.density_map), dtype=torch.float32, device=self.device)
        density_full_array = self.density_map.grid.array
        density_full_array = torch.tensor(density_full_array, device=self.device, dtype=torch.float32).flatten()

        self.zone_of_interest = torch.cat([pdb.region_of_interest_atoms for pdb in self.reference_pdbs])

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
        sliced_density = density_full_array[indexes]
        self.fo = sliced_density.clip(0)

        # self.density_loss_function = DensityGuidanceLossFunction(reference_pdbs, raw_pdb_file_path, chain_id, density_file, residue_range, rmax, device=device)
        self.density_estimator = AtomDensityEstimator(self.device)
        self.fc_other_chains = self._get_fc_of_non_sampled_chains()


        het_atoms_properties = extract_chain_ligand_water_tensors(self.raw_pdb_file_path, self.chain_id, device=self.device)
        self.het_atoms_coordinates = het_atoms_properties[0]
        self.het_atoms_bfactor = het_atoms_properties[1]
        self.het_atoms_occupancy = het_atoms_properties[2]
        self.het_atoms_elements = het_atoms_properties[3]

        ref_pdb_path = reference_pdbs[0]
        reference_atom_locations = get_sampler_pdb_inputs(ref_pdb_path, [0,-1], device)[0].squeeze()
        self.symmetry_operations = SymmetryOperations(reference_atom_locations, self.zone_of_interest, self.rmax + 10.0, ref_pdb_path, self.residue_range, self.device)

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
        elements = np.array(elements)[indexes.cpu()]
        bfactor = torch.tensor(np.array(bfactor), dtype=torch.float32, device=self.device)[indexes]
        occupancies = torch.tensor(np.array(occupancies), dtype=torch.float32, device=self.device)[indexes]
        element_indexes = self.density_estimator.get_element_indexes(elements)
        xyz = self.density_calculation_locations[None].repeat(1, 1, 1)
        fc = self.density_estimator(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)
        ##### OPTIMIZED CALL
        # fc = self.density_estimator.sample_fc_from_all_atom_optimized(symetrized_atom_locations, occupancies[None], element_indexes[None], bfactor[None], xyz, rmax=self.rmax)[None]
        return fc

    def calculate_density(self, reference_pdbs: List[PDBManager]):
        for reference_pdb in reference_pdbs:
            if reference_pdb.element_indexes is None:
                reference_pdb.element_indexes = self.density_estimator.get_element_indexes(reference_pdb.elements)

        atom_locations = torch.stack([pdb.atom_locations for pdb in reference_pdbs], dim=0)
        occupancy = torch.ones_like(atom_locations[..., 0])
        element_indexes = torch.stack([pdb.element_indexes for pdb in reference_pdbs], dim=0)
        bfactors = torch.stack([pdb.bfactors for pdb in reference_pdbs], dim=0)

        xyz = self.density_calculation_locations[None].repeat(len(reference_pdbs), 1, 1)

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
        return fc + self.fc_other_chains

    def save_zone_of_interest_density(self, density, file_path):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_map_slicing_indexes.cpu().numpy()] = density.cpu().numpy()
        self.density_map.grid.array[:] = array.reshape(self.density_map.grid.array.shape)
        self.density_map.write_ccp4_map(file_path)
        self.density_map.grid = original_grid

    def cosine_similarity(self, reference_pdbs: List[PDBManager]):
        fc = self.calculate_density(reference_pdbs).mean(dim=0)
        return torch.nn.functional.cosine_similarity(fc[None], self.fo[None])

class OCCMetric:
    def __init__(self, reference_pdbs, raw_pdb_file_path, chain_id, pdb_folder, density_file, residue_range, pdb_id, device="cuda", lr=5e-2, steps=300, grad_threshold=1e-3, regularization_weight=0.7, rmax=3.0, mtz_file=None, pdb_residue_range=None, phenix_setup_sh=None, ccp4_setup_sh=None):
        self.device = device
        self.density_file = density_file
        self.rmax = rmax
        self.pdb_id = pdb_id
        self.density_map = gemmi.read_ccp4_map(density_file)
        self.residue_range = residue_range
        self.reference_pdbs = reference_pdbs
        self.pdb_folder = pdb_folder
        self.sample_pdbs = self._load_pdb_managers(pdb_folder)
        self.N = len(self.sample_pdbs)
        self.I = torch.eye(self.N, dtype=torch.float32, device=self.device)
        self.raw_pdb_file_path = raw_pdb_file_path
        self.chain_id = chain_id
        self.density_calculator = DensityCalculator(density_file, reference_pdbs, self.raw_pdb_file_path, self.chain_id, residue_range, device=self.device, rmax=rmax)
        self.occupancy_logistics = torch.ones(self.N, dtype=torch.float32, device=self.device, requires_grad=True)

        with torch.no_grad():
            self.baseline_cosine_similarity = self.density_calculator.cosine_similarity(self.sample_pdbs)
        self.lr = lr 
        self.steps = steps
        self.grad_threshold = grad_threshold
        self.regularization_weight = regularization_weight
        
        # Cross-validation related attributes
        self.mtz_file = mtz_file
        self.pdb_residue_range = pdb_residue_range
        self.phenix_setup_sh = phenix_setup_sh
        self.ccp4_setup_sh = ccp4_setup_sh
        
        # Initialize managers if provided
        if self.phenix_setup_sh:
            self.phenix_manager = PhenixManager(self.phenix_setup_sh)
        else:
            self.phenix_manager = None
            
        if self.ccp4_setup_sh:
            self.ccp4_manager = CCP4Manager(self.ccp4_setup_sh)
        else:
            self.ccp4_manager = None

        self.occupancy_trajectory = []
    
    def _load_pdb_managers(self, pdb_folder):
        file_paths = sorted([os.path.join(pdb_folder, file_name) for file_name in os.listdir(pdb_folder) if file_name.endswith(".pdb") and (not "merged" in file_name)])
        pdb_managers = [PDBManager(file_path, residue_range=self.residue_range, device=self.device) for file_path in file_paths]
        return pdb_managers
    
    def calcualte_cosine_similarity_and_cost(self, pdbs_list=None, logistics=None):
        if pdbs_list is None:
            pdbs_list = self.sample_pdbs
        if logistics is None:
            logistics = self.occupancy_logistics

        densities = []
        for pdb in pdbs_list:
            densities.append(self.density_calculator.calculate_density([pdb]))

        densities = torch.cat(densities, dim=0)
        # TODO:
        # densities = self.density_calculator.calculate_density(pdbs_list)
        occupancy = logistics.softmax(dim=-1)
        fc = (densities * occupancy[:,None]).sum(dim=0)
        cosine_similarity = torch.nn.functional.cosine_similarity(fc[None], self.density_calculator.fo[None])

        cost = (torch.log(occupancy) * occupancy).mean()
        return cosine_similarity, -cost * self.regularization_weight
    
    def get_optimized_metrics(self, pdbs, logistics):
        # pdbs = [self.sample_pdbs[index] for index in selection_indexes]
        # logistics = self.occupancy_logistics[selection_indexes].detach().clone().requires_grad_(True)
        logistics = logistics.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([item for pdb in pdbs for item in pdb.parameters()] + [logistics], lr=self.lr)
        steps_generator = tqdm(range(1000))
        for _ in steps_generator:
            cosine_similarity, _ = self.calcualte_cosine_similarity_and_cost(pdbs, logistics)
            optimizer.zero_grad()
            (-cosine_similarity).backward()
            optimizer.step()
            steps_generator.set_description(f"optimizing occupanices, cosine similarity: {cosine_similarity.item():.4f}")
            if logistics.grad.norm() < 1e-7:
                break
        
        cosine_similarity, occupancies = cosine_similarity.item(), logistics.softmax(dim=-1).detach().cpu().numpy().tolist()

        print_string = [f"cosine similarity: {cosine_similarity:.4f}"]
        for pdb, occupancy in zip(pdbs, occupancies):
            print_string.append(f"{pdb.pdb_file_path} with occupancy: {occupancy:.4f}")
        print_string = "\n".join(print_string)
        print(print_string)

        return cosine_similarity, occupancies

    def get_density_all_grid(self, density):
        """
            this function will save the density map, where everything is 0 except for the region of interest which will have
            the given density array
        """
        original_grid = self.density_map.grid.clone()
        self.density_map.grid.array[:] = 0
        array = self.density_map.grid.array[:]
        array = array.flatten()
        array[self.density_calculator.density_map_slicing_indexes.cpu().numpy()] = density.cpu().numpy()
        self.density_map.grid = original_grid
        return array.reshape(self.density_map.grid.array.shape)

    def select_optimal_subset(self):
        for pdb in self.sample_pdbs:
            pdb.requries_grad()
        parameters = [item for pdb in self.sample_pdbs for item in pdb.parameters()] + [self.occupancy_logistics]
        optimizer = torch.optim.Adam(parameters, lr=self.lr)
        steps_generator = tqdm(range(self.steps))
        for _ in steps_generator:
            cosine_similarity, cost = self.calcualte_cosine_similarity_and_cost()
            optimizer.zero_grad()
            (cost - (cosine_similarity / self.baseline_cosine_similarity)).backward()
            optimizer.step()
            self.occupancy_trajectory.append(self.occupancy_logistics.detach().cpu().softmax(dim=-1).numpy())
            steps_generator.set_description(f"cosine simialrity: {cosine_similarity.item():.4f}")
            if self.occupancy_logistics.grad.norm() <= self.grad_threshold:
                break
        with torch.no_grad():
            selection_indexes = list((self.occupancy_logistics.softmax(dim=-1) > 0.1).nonzero().squeeze().cpu().numpy().reshape(-1))

        selected_pdb_objects = [self.sample_pdbs[index] for index in selection_indexes]
        selected_logistics = self.occupancy_logistics[selection_indexes].detach().clone().requires_grad_(True)
        selected_file_paths = [pdb.pdb_file_path for pdb in selected_pdb_objects]

        selected_cosine, occ_vals = self.get_optimized_metrics(selected_pdb_objects, selected_logistics)

        return selected_cosine, occ_vals, selected_file_paths, selected_pdb_objects
    
    def calculate_rwork_r_free(self, pdb_path, mtz_file):
        """Calculate R-work and R-free for a PDB structure."""
        if self.phenix_manager is None:
            raise ValueError("PhenixManager not initialized.")
        return self.phenix_manager.calculate_rwork_rfree(pdb_path, mtz_file)
    
    def run_refmac_refinement(self, hklin, hklout, xyzin, xyzout):
        """Run REFMAC refinement."""
        if self.ccp4_manager is None:
            raise ValueError("CCP4Manager not initialized.")
        return self.ccp4_manager.run_refmac_refinement(xyzin, hklin, hklout, xyzout)
    
    def get_rscc_metrics(self, pdb_file_path, mtz_file_path, chain, pdb_residue_range):
        """Get RSCC metrics for a PDB structure."""
        if self.phenix_manager is None:
            raise ValueError("PhenixManager not initialized.")
        return self.phenix_manager.get_rscc_metrics(pdb_file_path, mtz_file_path, chain, pdb_residue_range)
    
    def run(self, unguided=False, map_type="end"):
        """
        Run cross-validation to find the best regularization weight.
        
        Args:
            unguided: Whether this is unguided (affects output file naming)
            map_type: Type of map used for guidance (affects output file naming)
            
        Returns:
            dict: Best ensemble results with metrics
        """
        os.makedirs(os.path.join(self.pdb_folder, "occupancy_optim"), exist_ok=True)
        temp_dir_name = os.path.join("density_subset_selector_temp_dir", str(os.getpid()))
        os.makedirs(temp_dir_name, exist_ok=True)

        metrics_dict = dict()
        
        # Calculate reference metrics first
        reference_metrics = self.calculate_rwork_r_free(self.raw_pdb_file_path, self.mtz_file)
        metrics_dict["reference pdb r_work_r_free"] = reference_metrics

        # Run the selector
        cosine_similarity, occupancies, pdb_file_names, fc_values = self.select_optimal_subset()
        
        # Log to metrics dict
        metrics_dict["cosine similarity"] = cosine_similarity
        metrics_dict["occupancies"] = occupancies
        metrics_dict["pdb file names"] = [os.path.basename(file) for file in pdb_file_names]
        metrics_dict["fc"] = fc_values
        
        # Merge
        merged_pdb_object = merge_pdbs(self.raw_pdb_file_path, self.chain_id, pdb_file_names, occupancies, self.residue_range)
        merged_pdb_path = os.path.join(temp_dir_name, f"{self.regularization_weight}.pdb")
        merged_pdb_object.write_pdb(merged_pdb_path)

        # R work and R free of the sample
        metrics_dict["r_work_r_free"] = self.calculate_rwork_r_free(merged_pdb_path, self.mtz_file)

        # Refine merged file
        if unguided:
            suffix = "_unguided"
        else:
            suffix = f"_{map_type}_guided"

        # Refined file
        refined_mtz_file_path = os.path.join(self.pdb_folder, "occupancy_optim", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_occ_ref{suffix}.mtz")
        refined_pdb_file_path = os.path.join(self.pdb_folder, "occupancy_optim", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_occ_ref{suffix}.pdb")
        refinement_output = self.run_refmac_refinement(self.mtz_file, refined_mtz_file_path, merged_pdb_path, refined_pdb_file_path)

        # RSCC values
        ensemble_rscc = self.get_rscc_metrics(refined_pdb_file_path, self.mtz_file, self.chain_id, self.pdb_residue_range)
        pdb_rscc = self.get_rscc_metrics(self.raw_pdb_file_path, self.mtz_file, self.chain_id, self.pdb_residue_range)

        # Remove mmcif file
        if os.path.exists(os.path.join(self.pdb_folder, "occupancy_optim", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_occ_ref{suffix}.mmcif")):
            os.remove(os.path.join(self.pdb_folder, "occupancy_optim", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_occ_ref{suffix}.mmcif"))

        # R work and R free of the refined file
        if np.isnan(refinement_output["final_R"]):
            metrics_dict["r_work_r_free_refined"] = (np.nan, np.nan)
        else:
            metrics_dict["r_work_r_free"] = (refinement_output["initial_R"], refinement_output["initial_Rfree"])
            metrics_dict["r_work_r_free_refined"] = (refinement_output["final_R"], refinement_output["final_Rfree"])

        new_merged_file_path = os.path.join(self.pdb_folder, "occupancy_optim", f"{self.pdb_id}{self.chain_id}_{self.pdb_residue_range[0]}_{self.pdb_residue_range[1]}_occ_{map_type}_guided.pdb")
        shutil.copy(merged_pdb_path, new_merged_file_path)
        metrics_dict["reference pdb r_work_r_free"] = reference_metrics

        # Refine pdb too
        pdb_refinement_output = self.run_refmac_refinement(self.mtz_file, os.path.join(temp_dir_name, "temp.mtz"), self.raw_pdb_file_path, os.path.join(temp_dir_name, "temp.pdb"))

        # Remove headers
        remove_headers(new_merged_file_path, new_merged_file_path)
        remove_headers(refined_pdb_file_path, refined_pdb_file_path)

        if np.isnan(pdb_refinement_output["final_R"]):
            metrics_dict["reference pdb r_work_r_free_refined"] = (np.nan, np.nan)
        else:
            metrics_dict["reference pdb r_work_r_free"] = (pdb_refinement_output["initial_R"], pdb_refinement_output["initial_Rfree"])
            metrics_dict["reference pdb r_work_r_free_refined"] = (pdb_refinement_output["final_R"], pdb_refinement_output["final_Rfree"])

        metrics_dict["ensemble rscc"] = ensemble_rscc
        metrics_dict["pdb rscc"] = pdb_rscc
        metrics_dict["refined file"] = refined_pdb_file_path

        return metrics_dict
