"""
    This file contains the util functions for logging
"""
import wandb
import gemmi
import torch
import numpy as np
import yaml
from types import SimpleNamespace
from .mmseqs_query import run_mmseqs2
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from copy import deepcopy
from biotite.structure.io.pdb import PDBFile
import pandas as pd
import math

AMINO_ACID_ATOMS_ORDER = {
    "ALA": ["N", "CA", "C", "O", "CB"],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"],
    "CYS": ["N", "CA", "C", "O", "CB", "SG"],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"],
    "GLY": ["N", "CA", "C", "O"],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE"],
    "MSE": ["N", "CA", "C", "O", "CB", "CG", "SE", "CE"],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD"],
    "SER": ["N", "CA", "C", "O", "CB", "OG"],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2"],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH"],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2"],
    "CSO": ["N", "CA", "C", "O", "CB", "SG", "OD"]
}

ATOM_NAME_TO_ELEMENT = {
    "N": "N",
    "CA": "C",
    "C": "C",
    "O": "O",
    "CB": "C",
    "CG": "C",
    "CG1": "C",
    "CG2": "C",
    "CD": "C",
    "CD1": "C",
    "CD2": "C",
    "CE": "C",
    "CE1": "C",
    "CE2": "C",
    "CE3": "C",
    "CZ": "C",
    "CZ2": "C",
    "CZ3": "C",
    "CH2": "C",
    "NE": "N",
    "NE1": "N",
    "NE2": "N",
    "NZ": "N",
    "ND1": "N",
    "ND2": "N",
    "NH1": "N",
    "NH2": "N",
    "OD1": "O",
    "OD2": "O",
    "OE1": "O",
    "OE2": "O",
    "OG": "O",
    "OG1": "O",
    "OH": "O",
    "SD": "S",
    "SG": "S",
    "SE": "Se",
    "OXT": "O"
}

class WandbLogger:
    """
        This class will handle logging to wandb, to make sure that wanbd.log is called only once per iteration
    """
    def __init__(self):
        self.dict = {}
    
    def update(self, key, value):
        self.dict[key] = value

    def log(self):
        wandb.log(self.dict)
        self.dict.clear()

def filter_in_residue_bfactor(bfactor, residue_mask):
    bfactor_mean = bfactor.mean().item()
    bfactor[residue_mask] = bfactor_mean
    return bfactor

def load_pdb_atom_locations_full(
        pdb_file, full_sequences, chains_to_read=None, return_bfacs=False, return_mask=True, return_elements=False,
):
    # 0. Importing structures
    structure = gemmi.read_structure(pdb_file) 

    # 0..: Restructuring the case of just one sequence (old implementations)
    if isinstance(full_sequences, str):
        chains_to_read = [0,]
        full_sequences = [full_sequences,] 

    if chains_to_read is None:
        chains_to_read = list(range(len(full_sequences)))
    
    # 1.0 Cleaning if needed
    try: # hydrogens are being removed regardless.
        structure.remove_hydrogens()
        structure.remove_ligands_and_waters()
    except:
        print("Hydrogens or ligands failed to be removed. Continuing without removing them..!")

    # 3. Instantiating the positions and masked depending on the oligomeric state of the protein! We are still going from the fact that the sequence is an oligomer
    mask = [ 
        [
            [[0]] * len(AMINO_ACID_ATOMS_ORDER[ gemmi.expand_one_letter(one_letter_code, gemmi.ResidueKind.AA) ]) 
            for one_letter_code in full_sequence 
        ] for full_sequence in full_sequences
    ]
    atom_positions_full_perresidue = [ 
        [
            [[0,0,0]] * len(AMINO_ACID_ATOMS_ORDER[ gemmi.expand_one_letter(one_letter_code, gemmi.ResidueKind.AA)]) 
            for one_letter_code in full_sequence 
        ] for full_sequence in full_sequences
    ]
    for i in range(len(mask)):
        mask[i][-1].append([0]) # adding OXT to each last residue in each chain
        atom_positions_full_perresidue[i][-1].append([0,0,0]) 

    if return_bfacs: # Doing the same for b-factors..!
        b_factors = deepcopy(mask) 
    if return_elements: # and for elements too..!
        elements = deepcopy(mask) 

    # 4. Run the actual cycle to carefully import the atoms and chains
    chains = [structure[0][chain_i] for chain_i in chains_to_read] # we always take the first model, and the according chains of the first model!
    chains = [
        chains[sorted_i] for sorted_i in np.argsort([chain.name for chain in chains])
    ] # reordering by name: in the alphabetic order..!
    #chain_seqs = 

    for chain_i, chain in enumerate(chains):
        for residue_i, residue in enumerate(chain): 
            residue_index_in_pdb = residue.seqid.num - 1 # converting from the pdb 1-index to the proper 0-index (we go from the assimption that all the )
            if residue_index_in_pdb < 0:
                continue # Do not include residues with negative indices. 
            if residue_index_in_pdb >= len(full_sequences[chain_i]):
                continue

            for atom_i, atom in enumerate(residue):
                
                atom_index_in_pdb = AMINO_ACID_ATOMS_ORDER[residue.name].index(atom.name) \
                    if atom.name != "OXT" else -1 # in case of OXT, simply put at the last position in the residue array..!
                atom_positions_full_perresidue[chain_i][residue_index_in_pdb][atom_index_in_pdb] = (atom.pos.x, atom.pos.y, atom.pos.z)
                mask[chain_i][residue_index_in_pdb][atom_index_in_pdb] = [1]

                if return_bfacs:
                    b_factors[chain_i][residue_index_in_pdb][atom_index_in_pdb] = [atom.b_iso] if atom.b_iso is not None else [0.0] # bfactor is actually never zero..? 
                

    # 5. Flattening out the resulting arrays (NOW 3D instead of 2D arrays) and importing to torch tensors
    mask = [elem for sublist1 in mask for sublist2 in sublist1 for elem in sublist2]
    atom_positions_full_perresidue = [elem for sublist1 in atom_positions_full_perresidue for sublist2 in sublist1 for elem in sublist2]
    atom_positions_array = np.array(atom_positions_full_perresidue)
    atom_positions_tensor = torch.tensor(atom_positions_array, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool).flatten()

    to_return = [atom_positions_tensor, mask]
    if return_bfacs:
        b_factors = [elem for sublist1 in b_factors for sublist2 in sublist1 for elem in sublist2]
        b_factors_array = np.array(b_factors)
        b_factors_tensor = torch.tensor(b_factors_array, dtype=torch.float32).flatten()
        to_return.append(b_factors_tensor)

    if return_elements:
        elements = create_full_element_list(full_sequences)
        to_return.append(elements)

    return to_return

def create_full_element_list(
        sequences:str
):
    """
    Creating full element list in case some elements are unresolved in the original pdb file
    """
    residue_lists_per_sequence = [
        [gemmi.expand_one_letter(one_letter_code, gemmi.ResidueKind.AA) for one_letter_code in sequence]
        for sequence in sequences
    ]
    residue_lengths_per_sequence = [
        [len(AMINO_ACID_ATOMS_ORDER[residue]) for residue in residue_list]
        for residue_list in residue_lists_per_sequence
    ]
    atoms_per_residue_per_sequence = [
        [AMINO_ACID_ATOMS_ORDER[residue] for residue in residue_list]
        for residue_list in residue_lists_per_sequence
    ]

    atomic_numbers_per_sequence = [
        [[0,] * length for length in residue_lengths]
        for residue_lengths in residue_lengths_per_sequence
    ]

    for i, atoms_per_residue in enumerate(atoms_per_residue_per_sequence):
        for j, atomic_list in enumerate(atoms_per_residue):
            for k, atom_name in enumerate(atomic_list): 
                atomic_numbers_per_sequence[i][j][k] = gemmi.Element(ATOM_NAME_TO_ELEMENT[atom_name]).atomic_number
        # adding OXT to the last residue always! 
        atomic_numbers_per_sequence[i][-1].append(8)

    atomic_numbers = [item for sublist1 in atomic_numbers_per_sequence for sublist2 in sublist1 for item in sublist2]
    
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64).flatten()

    return atomic_numbers

def create_atom_mask(
    sequences: str, regions_of_interest_per_sequence # we need specific chains and full sequences in those chains!
):
    """
    Create the atom mask for the regions of interest given as a list of residue 1-indices per each full sequence. 
    The reason for requiring supplying the full sequence is due to many pdbs (especially in CryoEM) being uncompletely resolved.
    """
    residue_list_per_sequence = [
        [gemmi.expand_one_letter(one_letter_code, gemmi.ResidueKind.AA) for one_letter_code in sequence]
        for sequence in sequences
    ]
    residue_lengths_per_sequence = [
        [len(AMINO_ACID_ATOMS_ORDER[residue]) for residue in residue_list]
        for residue_list in residue_list_per_sequence
    ]
    
    masks_per_sequence = [
        [[0,] * length for length in residue_lengths]
        for residue_lengths in residue_lengths_per_sequence
    ]

    for i, regions_of_interest in enumerate(regions_of_interest_per_sequence):
        for interesting_residue in regions_of_interest:
            residue_index = interesting_residue - 1  # convert to 0-index
            masks_per_sequence[i][residue_index] = [1,] * residue_lengths_per_sequence[i][residue_index] # supplying the full 1-list as all the atoms in the residue are taken

    for i in range(len(sequences)):
        masks_per_sequence[i][-1] = masks_per_sequence[i][-1] + [0] # for OXT FIXME oxt can also be included, but ain't a big deal.

    mask = [elem for sublist1 in masks_per_sequence for sublist2 in sublist1 for elem in sublist2]  # flattening the mask
    
    mask = torch.tensor(mask, dtype=torch.bool).flatten()

    return mask

def load_pdb_atom_locations(pdb_file, device="cpu", single_model=True):
    structure = gemmi.read_structure(pdb_file)
    structure.remove_hydrogens()

    models_to_process = [structure[0]] if single_model else structure
    all_models_positions = []

    for model in models_to_process:
        model_positions = []
        chain = model[0]  # Assumes first chain is relevant
        for residue in chain:
            atoms_order = AMINO_ACID_ATOMS_ORDER.get(residue.name)
            if atoms_order is None:
                continue  # Skip unknown residues
            try:
                residue_atoms = [[atom for atom in residue if atom.name == atom_name][0] for atom_name in atoms_order]
            except IndexError:
                continue  # Skip residues with missing atoms
            if residue[-1].name == "OXT":
                residue_atoms.append(residue[-1])
            for atom in residue_atoms:
                model_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))

        if model_positions:
            all_models_positions.append(model_positions)

    atom_positions_array = np.array(all_models_positions)
    atom_positions_tensor = torch.tensor(atom_positions_array, dtype=torch.float32, device=device)
    return atom_positions_tensor  # shape: (1, N, 3) or (M, N, 3)

def write_back_pdb_coordinates(original_pdb_file, output_pdb_file, new_positions_tensor):
    # Step 1: Load the original structure
    structure = gemmi.read_structure(original_pdb_file)
    structure.remove_hydrogens()

    # Step 2: Flatten the new positions into a list
    if new_positions_tensor.dim() == 3:
        # If shape is (1, N, 3) or (M, N, 3), we pick the first model
        new_positions = new_positions_tensor[0].cpu().numpy()
    else:
        raise ValueError("Expected 3D tensor with shape (1, N, 3) or (M, N, 3)")

    # Step 3: Iterate through atoms in the same order as originally read
    model = structure[0]
    chain = model[0]  # assumes only one chain

    idx = 0
    for residue in chain:
        atoms_order = AMINO_ACID_ATOMS_ORDER.get(residue.name)
        if atoms_order is None:
            continue
        try:
            residue_atoms = [[atom for atom in residue if atom.name == atom_name][0] for atom_name in atoms_order]
        except IndexError:
            continue  # Skip residues with missing atoms
        if residue[-1].name == "OXT":
            residue_atoms.append(residue[-1])
        for atom in residue_atoms:
            new_x, new_y, new_z = new_positions[idx]
            atom.pos = gemmi.Position(new_x, new_y, new_z)
            idx += 1

    # Step 4: Save to new PDB
    structure.write_pdb(output_pdb_file)
    return output_pdb_file

def get_non_missing_atom_mask(pdb, atom_array):
    pdb_object = PDBFile.read(pdb)
    pdb_atom_array = pdb_object.get_structure(
        model=1,
        altloc="all",
    )
    atom_mask = []
    atom_array_index = 0
    pdb_atom_array_index = 0
    while atom_array_index < len(atom_array):
        if pdb_atom_array_index >= len(pdb_atom_array):
            atom_mask.append(False)
            atom_array_index += 1
            continue
        if (pdb_atom_array[pdb_atom_array_index].res_name == atom_array[atom_array_index].res_name) and (pdb_atom_array[pdb_atom_array_index].atom_name == atom_array[atom_array_index].atom_name):
            atom_mask.append(True)
            atom_array_index += 1
            pdb_atom_array_index += 1
        else:
            atom_mask.append(False)
            atom_array_index += 1



def get_atom_mask(pdb_file, residue_indices):
    """
        this function will return a boolean tensor that will be true for atoms that belong to residues that are in residue_indices
    """
    structure = gemmi.read_structure(pdb_file)
    model = structure[0]
    chain = model[0]
    mask = []
    for residue in chain:
        atoms_order = AMINO_ACID_ATOMS_ORDER[residue.name]
        residue_atoms = [[atom for atom in residue if atom.name == atom_name][0] for atom_name in atoms_order]
        if residue[-1].name == "OXT":
            residue_atoms.append(residue[-1])
        for atom in residue_atoms:
            altloc = atom.altloc
            # Append data to lists if altloc A or non-altloc
            if altloc == "A" or altloc == "\x00":
                mask.append(residue.seqid.num in residue_indices)
    return torch.tensor(mask)

def get_sampler_pdb_inputs(pdb_file, residue_indices, device):
    """
    We take the input pdb file and parse the atoms and their properties in the residue ranges
    """

    # TODO: fix this to get the correct order based on the atom array
    start_residue_number, end_residue_number = residue_indices
    structure =  gemmi.read_structure(pdb_file)

    # Lists to hold atom locations and their properties
    elements = []
    bfactors = []
    in_range_coordinates = []
    all_coordinates = []
    residue_range_mask = []

    chain = structure[0][0]
    # Iterate over all models, chains, and residues
    
    for residue in chain:
        res_num = residue.seqid.num

        # Skip residues without a numeric residue number
        if res_num is None:
            raise ValueError("Corrupted PDB!")

        # Check if residue is within the specified range
        in_range = start_residue_number <= res_num <= end_residue_number

        atoms_order = AMINO_ACID_ATOMS_ORDER[residue.name]
        # residue_atoms = [[atom for atom in residue if atom.name == atom_name][0] for atom_name in atoms_order]
        residue_atoms = []
        for atom_name in atoms_order:
            atoms_found = [atom for atom in residue if atom.name == atom_name]
            if atoms_found:
                residue_atoms.append(atoms_found[0])

        if residue[-1].name == "OXT":
            residue_atoms.append(residue[-1])
        for atom in residue_atoms:
            element = atom.element.name    # Element symbol
            bfactor = atom.b_iso           # B-factor
            coord = atom.pos               # Coordinates (Position object)
            coord_array = [coord.x, coord.y, coord.z]

            elements.append(element)
            bfactors.append(bfactor)
            residue_range_mask.append(in_range)
            all_coordinates.append(coord_array)

            if in_range:
                in_range_coordinates.append(coord_array)


    # Elements, bfactors and coordinates
    elements_array = np.array(elements, dtype=object)
    bfactors_array = np.array(bfactors, dtype=np.float32)
    in_range_coordinates_array = np.array(in_range_coordinates, dtype=np.float32)
    all_coordinates_array = np.array(all_coordinates, dtype=np.float32)
    mask_array = np.array(residue_range_mask, dtype=bool)

    # Convert it to torch
    bfactors_tensor = filter_in_residue_bfactor(torch.tensor(bfactors_array, device=device), mask_array)
    coordinates_tensor = torch.tensor(in_range_coordinates_array, device=device)
    mask_tensor = torch.tensor(mask_array, device=device, dtype=bool)
    all_coordinates_tensor = torch.tensor(all_coordinates_array, device=device)
    return all_coordinates_tensor[None], np.expand_dims(elements_array, axis=0), bfactors_tensor[None], coordinates_tensor, mask_tensor

def get_eval_precision(config_dtype_key):
    eval_precision = {
            "fp32": torch.float32,
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
        }[config_dtype_key]
    return eval_precision

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{key: dict_to_namespace(value) for key, value in d.items()})
    return d

def namespace_to_dict(namespace):
    if hasattr(namespace, "__dict__"):
        output = namespace.__dict__.copy()
    else:
        return namespace
    for key,value in output.items():
        output[key] = namespace_to_dict(value)
    return output

def load_config(yaml_file):
    with open(yaml_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return dict_to_namespace(config_dict)

def get_backbone_ca_mask(pdb_file, device=torch.device("cpu")):
    structure = gemmi.read_structure(pdb_file)
    model = structure[0]
    chain = model[0]
    backbone_mask = []
    ca_mask = []
    for residue in chain:
        for atom in residue:
            atom_name = atom.name
            # Append data to lists if altloc A or non-altloc
            if atom_name in ["N", "CA", "C", "O"]:
                backbone_mask.append(1)
            else:
                backbone_mask.append(0)

            if atom_name == "CA":
                ca_mask.append(1)
            else:
                ca_mask.append(0)

    return torch.tensor(backbone_mask, dtype=bool), torch.tensor(ca_mask, dtype=bool)


def query_msa_server(msa_full_save_dir, sequence_dictionary):
    if not os.path.exists(msa_full_save_dir):
        os.makedirs(msa_full_save_dir, exist_ok=True) 
        sequences = [dictionary["sequence"] for dictionary in sequence_dictionary]
        
        # Do paired ONLY if there's more than one unique sequence. 
        try:
            # From https://github.com/bjing2016/alphaflow/blob/02dc03763a016949326c2c741e6e33094f9250fd/scripts/mmseqs_search.py
            msa_unpaired = run_mmseqs2(
                sequences, prefix='tmp/', user_agent='sai-advaith/guided_alphafold', use_pairing=False
            )
        except Exception as e: 
            print(f"WARNING: Initial MSA failed. Retrying with nofilter mode. Error: {e}")
    
            # cleanup old tmp if anything was partially written
            import shutil
            if os.path.exists("tmp/_all"): shutil.rmtree("tmp/_all", ignore_errors=True)
            if os.path.exists("tmp/_env"): shutil.rmtree("tmp/_env", ignore_errors=True)
            if os.path.exists("tmp/_nofilter"): shutil.rmtree("tmp/_nofilter", ignore_errors=True)

            msa_unpaired = run_mmseqs2(
                sequences, prefix='tmp/', user_agent='sai-advaith/guided_alphafold', use_pairing=False, filter=False, use_env=False
            )

        if len(set(sequences)) > 1:
            msa_paired = run_mmseqs2(
                sequences, prefix='tmp/', user_agent='sai-advaith/guided_alphafold', use_pairing=True, pairing_strategy="complete" # pairing with the greedy didn't seem to work too nicely..!
            )

        os.system('rm -r tmp/_env')
        os.makedirs(os.path.join(msa_full_save_dir, "msa/"), exist_ok=True)

        # Non pairing a3m for each unique sequence!
        for i in range(len(sequences)):
            os.makedirs(os.path.join(msa_full_save_dir, f"msa/{i+1}"), exist_ok=True)

            # creating a subfolder for each unique sequence
            with open(os.path.join(msa_full_save_dir, f'msa/{i+1}/non_pairing.a3m'), 'w') as f:
                f.write(msa_unpaired[i])
            with open(os.path.join(msa_full_save_dir, f'msa/{i+1}/pairing.a3m'), 'w') as f:
                # if there are more than one unique sequence, we can do pairing
                if len(set(sequences)) > 1:
                    f.write(msa_paired[i])
                else:
                    continue

def delete_hydrogens(pdb_file):
    # Load the structure
    structure = gemmi.read_structure(pdb_file)
    structure.remove_hydrogens()
    # Save the modified structure back to the same path
    structure.write_pdb(pdb_file)

def is_water(residue):
    return residue.name in ['HOH', 'WAT'] and residue.het_flag

def is_ligand(residue):
    standard_residues = AMINO_ACID_ATOMS_ORDER.keys()  # standard amino acids and nucleotides
    return residue.het_flag and residue.name not in standard_residues

def extract_chain_ligand_water_tensors(pdb_path, target_chain_id, device="cpu"):
    structure = gemmi.read_structure(pdb_path)
    device = torch.device(device)
    model = structure[0]

    coords = []
    bfactors = []
    occupancies = []
    elements = []

    chain = model[target_chain_id]  # Get only the specified chain
    for residue in chain:
        if is_water(residue) or is_ligand(residue):
            for atom in residue:
                coords.append([atom.pos.x, atom.pos.y, atom.pos.z])
                bfactors.append(atom.b_iso)
                occupancies.append(atom.occ)
                elements.append(atom.element.name)

    # Convert to tensors and numpy array
    if len(coords) == 0 or len(bfactors) == 0 or len(occupancies) == 0 or len(elements) == 0:
        return None, None, None, None

    coordinates = torch.tensor(coords, dtype=torch.float32, device=device)
    bfactor_tensor = torch.tensor(bfactors, dtype=torch.float32, device=device)
    occupancy_tensor = torch.tensor(occupancies, dtype=torch.float32, device=device)
    elements_array = np.array(elements, dtype=str)

    return coordinates[None], bfactor_tensor[None], occupancy_tensor[None], np.expand_dims(elements_array, axis=0)

def remove_headers(pdb_file_path, write_pdb_file_path=None):
    keep_prefixes = ("CRYST1", "ATOM  ", "HETATM", "ANISOU", "TER", "END")

    # If no output path is given or matches input, use a temp file
    overwrite = write_pdb_file_path is None or write_pdb_file_path == pdb_file_path
    temp_path = pdb_file_path + ".tmp" if overwrite else write_pdb_file_path

    with open(pdb_file_path, "r") as infile, open(temp_path, "w") as outfile:
        for line in infile:
            if line.startswith(keep_prefixes):
                outfile.write(line)

    # If overwriting, replace original file
    if overwrite:
        os.replace(temp_path, pdb_file_path)
        return pdb_file_path
    else:
        return write_pdb_file_path

def create_backbone_masks(
        sequences: str, device=torch.device("cpu")
):
    """
    Outputs 3 masks: for N, CA, C atoms in the backbone of the protein sequences.
    """
    residue_lists_per_sequence = [
        [gemmi.expand_one_letter(one_letter_code, gemmi.ResidueKind.AA) for one_letter_code in sequence]
        for sequence in sequences
    ]
    residue_lengths_per_sequence = [
        [len(AMINO_ACID_ATOMS_ORDER[residue]) for residue in residue_list]
        for residue_list in residue_lists_per_sequence
    ]
    masks_per_sequence_N = [
        [[0,] * length for length in residue_lengths]
        for residue_lengths in residue_lengths_per_sequence
    ]
    masks_per_sequence_CA = [
        [[0,] * length for length in residue_lengths]
        for residue_lengths in residue_lengths_per_sequence
    ]
    masks_per_sequence_C = [
        [[0,] * length for length in residue_lengths]
        for residue_lengths in residue_lengths_per_sequence
    ]

    for chain_i in range(len(masks_per_sequence_CA)):
        for residue_i in range(len(masks_per_sequence_CA[chain_i])):
            masks_per_sequence_N[chain_i][residue_i][0] = 1  # N atom
            masks_per_sequence_CA[chain_i][residue_i][1] = 1  # CA atom
            masks_per_sequence_C[chain_i][residue_i][2] = 1  # C atom
        masks_per_sequence_N[chain_i][-1].append(0)  # OXT atom
        masks_per_sequence_CA[chain_i][-1].append(0)  # OXT atom
        masks_per_sequence_C[chain_i][-1].append(0)  # O
            
        
    masks_per_sequence_N = [elem for sublist1 in masks_per_sequence_N for sublist2 in sublist1 for elem in sublist2]
    masks_per_sequence_CA = [elem for sublist1 in masks_per_sequence_CA for sublist2 in sublist1 for elem in sublist2]
    masks_per_sequence_C = [elem for sublist1 in masks_per_sequence_C for sublist2 in sublist1 for elem in sublist2]

    masks_per_sequence_N = torch.tensor(masks_per_sequence_N, dtype=torch.bool, device=device).flatten()
    masks_per_sequence_CA = torch.tensor(masks_per_sequence_CA, dtype=torch.bool, device=device).flatten()
    masks_per_sequence_C = torch.tensor(masks_per_sequence_C, dtype=torch.bool, device=device).flatten()

    return masks_per_sequence_N, masks_per_sequence_CA, masks_per_sequence_C

def talos_to_dihedral_tensor(
    paths: Union[str, Sequence[str]], 
    full_sequence: str,
    *,
    device: str = "cpu",
    radians: bool = False,  # Changed default to False to match CSV parser behavior
    strict: bool = False,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Parse TALOS(-N) .tab files into tensors consistent with parse_dihedrals_csv format.
    
    Args:
        paths: Path(s) to TALOS .tab files
        full_sequence: Full protein sequence to create properly sized tensors
        device: Device for tensors
        radians: If True, convert to radians (default False to match CSV parser)
        strict: If True, raise errors on malformed lines
        dtype: Tensor data type
        
    Returns:
        tuple: (dihedrals_tensor, dihedrals_mask) where:
            - dihedrals_tensor: [len(full_sequence), 4] with [phi, psi, dphi, dpsi]
            - dihedrals_mask: [len(full_sequence)] boolean mask for valid residues
            
    CRITICAL INDEXING:
    - Output tensor[i] corresponds to 1-indexed residue i+1 (same as CSV parser)
    - This matches the expected format for backbone_dihedrals usage in loss function
    
    TALOS CLASS FILTERING:
    - Predictions with CLASS="Warn" are excluded (mask=0) as they have no consensus
    - Only "Strong", "Generous", and "Dyn" predictions are included in the mask
    - "None" predictions are already excluded due to 9999.0 sentinel values
    """

    if isinstance(paths, (str, bytes)):
        paths = [paths]

    # Initialize tensors like CSV parser
    phi_psi_dphi_dpsi_tensor = torch.zeros(len(full_sequence), 4, device=device, dtype=dtype) # there's no dihedral for the last residue! [as in, dihedrals are smaller than overall...!]
    phi_psi_mask = torch.zeros(len(full_sequence), device=device, dtype=torch.bool)

    def parse_one(path: str):
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                s = line.strip()
                if not s or s.startswith(("VARS", "FORMAT", "#", "REMARK", "DATA")):
                    continue
                parts = s.split()
                # Expect at least: RESID RESNAME PHI PSI DPHI DPSI DIST S2 COUNT CS_COUNT CLASS
                if len(parts) < 11:
                    if strict:
                        raise ValueError(f"{path}:{ln} malformed line: {line!r}")
                    continue

                try:
                    resid = int(parts[0])  # 1-indexed residue number from TALOS
                    phi  = float(parts[2])
                    psi  = float(parts[3])
                    dphi = float(parts[4])
                    dpsi = float(parts[5])
                    class_label = parts[10]  # CLASS column
                except ValueError:
                    if strict:
                        raise
                    continue

                # Skip warning predictions as they should not be used
                if class_label == "Warn":
                    continue

                # Handle TALOS missing sentinels
                if abs(phi) >= 9999.0:
                    phi = float('nan')
                    dphi = float('nan')
                if abs(psi) >= 9999.0:
                    psi = float('nan')
                    dpsi = float('nan')
                
                # Skip if both angles are missing
                if math.isnan(phi) and math.isnan(psi):
                    continue

                # Map 1-indexed TALOS residue to 0-indexed tensor position
                # CRITICAL: resid is 1-indexed, tensor is 0-indexed
                # So TALOS residue N goes to tensor[N-1]
                tensor_idx = resid - 1
                
                if 0 <= tensor_idx < len(full_sequence): # - 1: # there's no dihedral for the last residue! [as in, dihedrals are smaller than overall...!]
                    phi_psi_mask[tensor_idx] = True
                    phi_psi_dphi_dpsi_tensor[tensor_idx, 0] = phi if not math.isnan(phi) else 0.0
                    phi_psi_dphi_dpsi_tensor[tensor_idx, 1] = psi if not math.isnan(psi) else 0.0
                    phi_psi_dphi_dpsi_tensor[tensor_idx, 2] = dphi if not math.isnan(dphi) else 0.0
                    phi_psi_dphi_dpsi_tensor[tensor_idx, 3] = dpsi if not math.isnan(dpsi) else 0.0

    # Parse all files
    for path in paths:
        parse_one(path)

    # Convert to radians if requested (but default is False to match CSV parser)
    if radians:
        phi_psi_dphi_dpsi_tensor *= (math.pi / 180.0)

    return phi_psi_dphi_dpsi_tensor, phi_psi_mask

def parse_phenix_eval_log_to_csv(log_file_path, output_csv_path):
    """
    Parse Phenix evaluation log file and extract CC information to CSV.
    
    Args:
        log_file_path (str): Path to the Phenix evaluation log file
        output_csv_path (str): Path for output CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with all CC information
    """
    
    results = {
        'metric': [],
        'value': [],
        'b_factor': [],
        'occupancy': [],
        'n_atoms': [],
        'chain_id': [],
        'description': []
    }
    
    with open(log_file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse overall CC values
    for i, line in enumerate(lines):
        line = line.strip()
        
        # Overall CC metrics
        if line.startswith('CC_mask'):
            cc_value = float(line.split(':')[1].strip())
            results['metric'].append('CC_mask')
            results['value'].append(cc_value)
            results['b_factor'].append(None)
            results['occupancy'].append(None)
            results['n_atoms'].append(None)
            results['chain_id'].append('overall')
            results['description'].append('Overall mask correlation')
            
        elif line.startswith('CC_volume'):
            cc_value = float(line.split(':')[1].strip())
            results['metric'].append('CC_volume')
            results['value'].append(cc_value)
            results['b_factor'].append(None)
            results['occupancy'].append(None)
            results['n_atoms'].append(None)
            results['chain_id'].append('overall')
            results['description'].append('Overall volume correlation')
            
        elif line.startswith('CC_peaks'):
            cc_value = float(line.split(':')[1].strip())
            results['metric'].append('CC_peaks')
            results['value'].append(cc_value)
            results['b_factor'].append(None)
            results['occupancy'].append(None)
            results['n_atoms'].append(None)
            results['chain_id'].append('overall')
            results['description'].append('Overall peaks correlation')
            
        elif line.startswith('CC_box'):
            cc_value = float(line.split(':')[1].strip())
            results['metric'].append('CC_box')
            results['value'].append(cc_value)
            results['b_factor'].append(None)
            results['occupancy'].append(None)
            results['n_atoms'].append(None)
            results['chain_id'].append('overall')
            results['description'].append('Overall box correlation')
        
        # Per chain CC values
        elif 'chain ID  CC' in line and i+1 < len(lines):
            # Parse chain data lines
            j = i + 1
            while j < len(lines) and lines[j].strip() and not lines[j].startswith('Main chain'):
                chain_line = lines[j].strip()
                if len(chain_line.split()) >= 5:
                    parts = chain_line.split()
                    chain_id = parts[0]
                    cc_value = float(parts[1])
                    b_factor = float(parts[2])
                    occupancy = float(parts[3])
                    n_atoms = int(parts[4])
                    
                    results['metric'].append('CC_chain')
                    results['value'].append(cc_value)
                    results['b_factor'].append(b_factor)
                    results['occupancy'].append(occupancy)
                    results['n_atoms'].append(n_atoms)
                    results['chain_id'].append(chain_id)
                    results['description'].append(f'Per chain correlation - chain {chain_id}')
                j += 1
        
        # Main chain CC
        elif line.startswith('Main chain:') and i+2 < len(lines):
            main_chain_line = lines[i+2].strip()
            if len(main_chain_line.split()) >= 4:
                parts = main_chain_line.split()
                cc_value = float(parts[0])
                b_factor = float(parts[1])
                occupancy = float(parts[2])
                n_atoms = int(parts[3])
                
                results['metric'].append('CC_main_chain')
                results['value'].append(cc_value)
                results['b_factor'].append(b_factor)
                results['occupancy'].append(occupancy)
                results['n_atoms'].append(n_atoms)
                results['chain_id'].append('all')
                results['description'].append('Main chain correlation')
        
        # Side chain CC
        elif line.startswith('Side chain:') and i+2 < len(lines):
            side_chain_line = lines[i+2].strip()
            if len(side_chain_line.split()) >= 4:
                parts = side_chain_line.split()
                cc_value = float(parts[0])
                b_factor = float(parts[1])
                occupancy = float(parts[2])
                n_atoms = int(parts[3])
                
                results['metric'].append('CC_side_chain')
                results['value'].append(cc_value)
                results['b_factor'].append(b_factor)
                results['occupancy'].append(occupancy)
                results['n_atoms'].append(n_atoms)
                results['chain_id'].append('all')
                results['description'].append('Side chain correlation')
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Reorder chains to A, B, C, etc. for per-chain entries
    chain_entries = df[df['metric'] == 'CC_chain'].copy()
    if not chain_entries.empty:
        # Sort chain entries alphabetically
        chain_entries = chain_entries.sort_values('chain_id')
        
        # Remove old chain entries and add sorted ones
        df_no_chains = df[df['metric'] != 'CC_chain']
        df = pd.concat([df_no_chains, chain_entries], ignore_index=True)
    
    # Save to CSV
    df.to_csv(output_csv_path, index=False)
    print(f"Saved Phenix evaluation results to: {output_csv_path}")
    
    return df

def parse_phenix_cc_log(log_file_path):
    """
    Parse Phenix CC per residue log file.
    
    Args:
        log_file_path (str): Path to the Phenix CC log file
        
    Returns:
        dict: Dictionary with chain_id as key and list of (residue_num, residue_name, cc_value) as values
    """
    chain_data = {}
    
    with open(log_file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) >= 4:
                chain_id = parts[0]
                residue_name = parts[1]
                residue_num = int(parts[2])
                cc_value = float(parts[3])
                
                if chain_id not in chain_data:
                    chain_data[chain_id] = []
                
                chain_data[chain_id].append((residue_num, residue_name, cc_value))
    
    # Sort by residue number within each chain
    for chain_id in chain_data:
        chain_data[chain_id].sort(key=lambda x: x[0])
    
    return chain_data

def parse_dihedrals_csv(dihedrals_file, full_sequence, device="cpu"): # for now, we are mostly assuming the presence of only one unique sequence... [so either multimers or not....!]
    """
    The input is the .csv file in the format alex follows: 
    there's entries for [residue_number, dihedral_name] and it's corresponding [dihedral_value] [dihedral_half_width]. 
    This has to be transformed into tensors. And a mask for non-existent residues...!
    """
    phi_psi_dphi_dpsi_tensor = torch.zeros(len(full_sequence), 4, device=device)
    phi_psi_mask = torch.zeros(len(full_sequence), device=device, dtype=torch.bool)

    chi1_dchi1_tensor = torch.zeros((len(full_sequence), 2), device=device)
    chi1_mask = torch.zeros(len(full_sequence), device=device, dtype=torch.bool)

    dihedrals_df = pd.read_csv(dihedrals_file)

    for residue_i in range(0, len(full_sequence)): # 
        dataframe_for_residue = dihedrals_df[dihedrals_df["residue_num"] == residue_i+1] # the residues itself are saved as 1-indexed..!
        
        if len(dataframe_for_residue) == 0: # Do not change the masks from 0 to 1 etc. etc.
            continue 

        if "chi1" in dataframe_for_residue["angle_name"].values:
            chi1_mask[residue_i] = True
            chi1_dchi1_tensor[residue_i, 0] = dataframe_for_residue[dataframe_for_residue["angle_name"] == "chi1"]["target_angle"].values[0]
            chi1_dchi1_tensor[residue_i, 1] = dataframe_for_residue[dataframe_for_residue["angle_name"] == "chi1"]["half_width"].values[0]

        if "phi" in dataframe_for_residue["angle_name"].values and "psi" in dataframe_for_residue["angle_name"].values:
            phi_psi_mask[residue_i] = True
            
            # Note the correct order. First phi, psi, then dphi, dpsi..!
            phi_psi_dphi_dpsi_tensor[residue_i, 0] = dataframe_for_residue[dataframe_for_residue["angle_name"] == "phi"]["target_angle"].values[0]
            phi_psi_dphi_dpsi_tensor[residue_i, 2] = dataframe_for_residue[dataframe_for_residue["angle_name"] == "phi"]["half_width"].values[0] * 1.18 # converting half width half maximum to the std .. !

            phi_psi_dphi_dpsi_tensor[residue_i, 1] = dataframe_for_residue[dataframe_for_residue["angle_name"] == "psi"]["target_angle"].values[0]
            phi_psi_dphi_dpsi_tensor[residue_i, 3] = dataframe_for_residue[dataframe_for_residue["angle_name"] == "psi"]["half_width"].values[0] * 1.18 

    return phi_psi_dphi_dpsi_tensor, phi_psi_mask, chi1_dchi1_tensor, chi1_mask

def alignment_mask_by_chain(sequences, chains_to_align=[0]):
    residue_lists_per_sequence = [
        [gemmi.expand_one_letter(one_letter_code, gemmi.ResidueKind.AA) for one_letter_code in sequence]
        for sequence in sequences
    ]
    residue_lengths_per_sequence = [
        [len(AMINO_ACID_ATOMS_ORDER[residue]) for residue in residue_list]
        for residue_list in residue_lists_per_sequence
    ]

    atomic_lengths_per_sequence = [sum(lengths)+1 for lengths in residue_lengths_per_sequence] # +1 for OXT

    masks_per_sequence = [[0,]*length for length in atomic_lengths_per_sequence]

    for chain_to_align in chains_to_align:
        masks_per_sequence[chain_to_align] = [1,] * atomic_lengths_per_sequence[chain_to_align]

    mask = [elem for sublist in masks_per_sequence for elem in sublist]  # flattening the mask
    mask = torch.tensor(mask, dtype=torch.bool).flatten()
    return mask

def create_cc_bfactor_tensor(chain_data, full_sequences):
    """
    Create B-factor tensor from CC values with proper atom ordering.
    Handles chain reordering: log might have B,A,C but we need A,B,C order for sequences.
    Also handles residue numbering offset (log might start at 448, not 1).
    
    Args:
        chain_data (dict): Dictionary from parse_phenix_cc_log (chains might be in any order)
        full_sequences (list): List of full protein sequences (in A,B,C,... order)
        
    Returns:
        torch.Tensor: B-factor tensor with CC values, same length as total atoms including OXT
    """
    # Create CC lookup by chain and residue number
    cc_lookup = {}
    residue_number_ranges = {}
    
    for chain_id, residues in chain_data.items():
        cc_lookup[chain_id] = {}
        residue_numbers = [res_num for res_num, _, _ in residues]
        residue_number_ranges[chain_id] = (min(residue_numbers), max(residue_numbers))
        
        for res_num, res_name, cc_value in residues:
            cc_lookup[chain_id][res_num] = cc_value
    
    # Get the actual chains present in the log (e.g., ['A', 'B', 'C'])
    available_chains = sorted(chain_data.keys())
    #print(f"Available chains in log: {available_chains}")
    
    # Show residue number ranges for each chain
    for chain_id in available_chains:
        start, end = residue_number_ranges[chain_id]
        #print(f"Chain {chain_id}: residue numbers {start} to {end}")
    
    bfactor_list = []
    
    # Process sequences in A,B,C order, but map to available chains
    for seq_idx in range(len(full_sequences)):
        sequence = full_sequences[seq_idx]
        
        # Map sequence index to actual chain letter from log
        if seq_idx < len(available_chains):
            actual_chain_letter = available_chains[seq_idx]
            start_res_num = residue_number_ranges[actual_chain_letter][0]
            #print(f"Mapping sequence {seq_idx} (target chain {chr(ord('A') + seq_idx)}) to log chain {actual_chain_letter}, starting at residue {start_res_num}")
        else:
            #print(f"Warning: Sequence {seq_idx} has no corresponding chain in log")
            actual_chain_letter = None
            start_res_num = 1
        
        for residue_idx, res_name_one_letter in enumerate(sequence):
            # Use the actual residue numbering from the log
            actual_residue_number = start_res_num + residue_idx
            
            # Get CC value for this residue from the actual chain in the log
            if actual_chain_letter:
                cc_value = cc_lookup.get(actual_chain_letter, {}).get(actual_residue_number, 0.0)
            else:
                cc_value = 0.0
            
            # Get the 3-letter residue name
            res_name_three_letter = gemmi.expand_one_letter(res_name_one_letter, gemmi.ResidueKind.AA)
            
            # Add CC value for all atoms in this residue
            num_atoms = len(AMINO_ACID_ATOMS_ORDER[res_name_three_letter])
            bfactor_list.extend([cc_value] * num_atoms)
            
            # Add CC value for OXT atom if this is the last residue
            if residue_idx == len(sequence) - 1:
                bfactor_list.append(cc_value)
    
    bfactor_tensor = torch.tensor(bfactor_list, dtype=torch.float32)
    print(f"Created tensor with {len(bfactor_tensor)} values, range: {bfactor_tensor.min():.4f} - {bfactor_tensor.max():.4f}")
    
    return bfactor_tensor
