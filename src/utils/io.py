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
from biotite.structure.io.pdb import PDBFile


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


def query_msa_server(msa_save_dir, pdb_id, sequence):
    os.makedirs(msa_save_dir, exist_ok=True)
    if not os.path.exists(f"{msa_save_dir}/{pdb_id}"):
        msa = run_mmseqs2([sequence], prefix='/tmp/', user_agent='sai-advaith/guided_alphafold')[0]
        os.system('rm -r /tmp/_env')

        os.makedirs(f'{msa_save_dir}/{pdb_id}/msa/', exist_ok=True)

        # Non pairing a3m
        with open(f'{msa_save_dir}/{pdb_id}/msa/non_pairing.a3m', 'w') as f:
            f.write(msa)

        # ProteinX requirement
        with open(f"{msa_save_dir}/{pdb_id}/msa/pairing.a3m", "w"):
            pass

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