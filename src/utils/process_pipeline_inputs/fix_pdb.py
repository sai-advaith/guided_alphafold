import gemmi
import os
import pdbfixer
from openmm.app import PDBFile
from tqdm import tqdm
from .extract_metadata import get_chain_seq
from scipy.spatial.distance import cdist
import numpy as np

aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'SEC': 'U', 'PYL': 'O',
    'MSE': "M"
}

inverse_aa_map = {item:key for key,item in aa_map.items()}

def get_altloc_symbols_from_chain(chain: gemmi.Chain):
    symbols = {"A"}
    for res in chain:
        for atom in res:
            if atom.altloc != "\x00":
                symbols.add(atom.altloc)
    return symbols

def create_new_structure_like(structure: gemmi.Structure):
    new_structure = structure.clone()
    while len(new_structure) > 1:
        del new_structure[1]
    while len(new_structure[0]) > 0:
        del new_structure[0][0]
    return new_structure

def set_all_altlocs_to_none(structure: gemmi.Structure):
    model = structure[0]
    chain = model[0]
    for residue in chain:
        for atom in residue:
            atom.altloc = "\x00"


def search_for_seq_head_and_tail(structure, seq: str, window_size=8, min_matches=5):
    """
    Find where the modeled chain starts and ends within the SEQRES sequence.
    
    This function takes small windows from the beginning and end of the modeled
    chain and finds their best matches in the SEQRES sequence, accounting for
    potentially missing regions in the middle of the protein.
    
    Args:
        structure: gemmi.Structure object
        seq: SEQRES sequence string to search in
        window_size: size of substring to use for matching (default 8)
        min_matches: minimum consecutive matches required (default 5)
    
    Returns:
        tuple: (head_position, tail_position) in SEQRES coordinates
               Returns (None, None) if no good matches found
    """
    model = structure[0]
    chain = model[0]
    
    # Extract chain residue names
    chain_residues = [residue.name for residue in chain]
    
    if len(chain_residues) < min_matches:
        return None, None
    
    # Find head position
    head_pos = find_sequence_start(chain_residues, seq, window_size, min_matches)
    
    # Find tail position  
    tail_pos = find_sequence_end(chain_residues, seq, window_size, min_matches) + 1
    
    return head_pos, tail_pos


def find_sequence_start(chain_residues, seq, window_size=8, min_matches=5):
    """
    Find where the chain starts in the SEQRES sequence.
    
    Takes a window from the beginning of the modeled chain and finds
    the best match in the SEQRES sequence.
    """
    if len(chain_residues) < window_size:
        window_size = len(chain_residues)
    
    # Take window from start of modeled chain
    chain_window = chain_residues[:window_size]
    
    best_pos = None
    best_match_length = 0
    
    # Try all positions in the SEQRES sequence
    for seq_pos in range(len(seq) - window_size + 1):
        # Count consecutive matches from this position
        match_length = 0
        for i in range(min(window_size, len(seq) - seq_pos)):
            if seq_pos + i < len(seq) and chain_window[i] == seq[seq_pos + i]:
                match_length += 1
            else:
                break  # Stop at first mismatch for consecutive counting
        
        # Update best match if this one is better
        if match_length >= min_matches and match_length > best_match_length:
            best_match_length = match_length
            best_pos = seq_pos
    
    return best_pos



def find_sequence_end(chain_residues, seq, window_size=8, min_matches=5):
    """
    Find where the chain ends in the SEQRES sequence.
    
    Takes a window from the end of the modeled chain and searches
    backwards from the end of the SEQRES sequence for the best match.
    """
    if len(chain_residues) < window_size:
        window_size = len(chain_residues)
    
    # Take window from end of modeled chain
    chain_window = chain_residues[-window_size:]
    
    best_pos = None
    best_match_length = 0
    
    # Search backwards from the end of SEQRES sequence
    for seq_end_pos in range(len(seq) - 1, window_size - 2, -1):
        seq_start_pos = seq_end_pos - window_size + 1
        
        # Count consecutive matches from this position
        match_length = 0
        for i in range(window_size):
            if seq_start_pos + i >= 0 and chain_window[i] == seq[seq_start_pos + i]:
                match_length += 1
            else:
                break  # Stop at first mismatch
        
        # Update best match if this one is better
        if match_length >= min_matches and match_length > best_match_length:
            best_match_length = match_length
            best_pos = seq_end_pos  # Return end position
            
            # If we found a perfect match, we can stop (since we're searching from the end)
            if match_length == window_size:
                break
    
    return best_pos

def split_altlocs(structure: gemmi.Structure, chain_name, root="pipeline_inputs"):
    structure.setup_entities()
    structure.remove_ligands_and_waters()
    # structure.entities[0].full_sequence = [inverse_aa_map[res] for res in full_fasta_sequence]
    # structure.remove_hydrogens()
    model = structure[0]
    output_pdb_files = []
    chain = [chain for chain in model if (chain.name == chain_name) and (len(chain) > 0)][0]
    entity_sequence = [entity for entity in structure.entities if chain_name in [name[0] for name in entity.subchains]][0].full_sequence
    altloc_symbols = get_altloc_symbols_from_chain(chain)

    for symbol in altloc_symbols:
        new_structure = create_new_structure_like(structure)
        new_chain = chain.clone()
        for residue in new_chain:
            atoms_to_delete = set()
            for atom in residue:
                if atom.altloc not in [symbol, "\x00"]:
                    atoms_to_delete.add(atom.clone())
            for atom in atoms_to_delete:
                residue.remove_atom(atom.name, atom.altloc, atom.element)
        new_structure[0].add_chain(new_chain)
        structure_name = f"{structure.name[:4]}_chain_{chain.name}_altloc_{symbol}"
        set_all_altlocs_to_none(new_structure)

        # slicing tail
        seq_head, seq_tail = search_for_seq_head_and_tail(new_structure, entity_sequence)
        index_of_chain = [i for i in range(len(new_structure.entities)) if chain_name in [name[0] for name in new_structure.entities[i].subchains]][0]
        new_structure.entities[index_of_chain].full_sequence = new_structure.entities[index_of_chain].full_sequence[seq_head:seq_tail]

        new_structure.make_pdb_headers()
        os.makedirs(os.path.join(root,"pdbs", structure.name[:4]), exist_ok=True)
        pdb_file_path = os.path.join(root,"pdbs", structure.name[:4], f"{structure_name}.pdb")
        new_structure.write_pdb(pdb_file_path, gemmi.PdbWriteOptions(seqres_records=True))
        output_pdb_files.append(pdb_file_path)
    return output_pdb_files

def fix_bfactor_diff_seq(source_structure, target_structure, distance_threshold=2.0, default_bfactor=100.0):
    """
    Transfer B-factors from source to target structure based on spatial proximity.
    This handles cases where atom coordinates have shifted after adding missing residues.
    
    Parameters:
    - source_structure: Original structure with some missing residues
    - target_structure: Complete structure with filled residues (coordinates may have shifted)
    - distance_threshold: Maximum distance (Å) to consider atoms as matching (default: 2.0)
    - default_bfactor: B-factor value for newly added atoms/unmatched atoms (default: 100.0)
    
    Returns:
    - target_structure with updated B-factors
    """
    source_chain = source_structure[0][0]
    target_chain = target_structure[0][0]
    
    # Extract coordinates and B-factors from source
    source_atoms = []
    source_coords = []
    source_bfactors = []
    
    for residue in source_chain:
        for atom in residue:
            source_atoms.append((residue.seqid.num, residue.name, atom.name))
            source_coords.append([atom.pos[0], atom.pos[1], atom.pos[2]])
            source_bfactors.append(atom.b_iso)
    
    source_coords = np.array(source_coords)
    source_bfactors = np.array(source_bfactors)
    
    # Extract coordinates from target
    target_atoms = []
    target_coords = []
    
    for residue in target_chain:
        for atom in residue:
            target_atoms.append((residue.seqid.num, residue.name, atom.name))
            target_coords.append([atom.pos[0], atom.pos[1], atom.pos[2]])
    
    target_coords = np.array(target_coords)
    
    # Calculate distance matrix between all source and target atoms
    distances = cdist(source_coords, target_coords)
    
    # For each target atom, find the closest source atom
    target_idx = 0
    for residue in target_chain:
        for atom in residue:
            target_info = target_atoms[target_idx]
            
            # Find closest source atoms
            min_distances = distances[:, target_idx]
            closest_source_idx = np.argmin(min_distances)
            min_distance = min_distances[closest_source_idx]
            
            # Check if we have a reasonable match
            if min_distance <= distance_threshold:
                source_info = source_atoms[closest_source_idx]
                
                # Additional check: prefer matching atom names and residue types
                same_atom_name = (source_info[2] == target_info[2])
                same_residue_type = (source_info[1] == target_info[1])
                
                # If we have same atom name and residue type, use this match
                # Otherwise, look for better matches within threshold
                if not (same_atom_name and same_residue_type):
                    # Find all source atoms within threshold
                    within_threshold = np.where(min_distances <= distance_threshold)[0]
                    
                    # Prefer matches with same atom name and residue type
                    best_idx = closest_source_idx
                    best_score = 0
                    
                    for idx in within_threshold:
                        src_info = source_atoms[idx]
                        score = 0
                        if src_info[2] == target_info[2]:  # same atom name
                            score += 2
                        if src_info[1] == target_info[1]:  # same residue type
                            score += 1
                        
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    
                    closest_source_idx = best_idx
                
                # Assign B-factor from matched source atom
                atom.b_iso = source_bfactors[closest_source_idx]
            else:
                # No close match found - likely a newly added atom
                atom.b_iso = default_bfactor
            
            target_idx += 1
    
    return target_structure

def fix_bfactor_same_seq(source_structure, target_structure):
    source_chain = source_structure[0][0]
    target_chain = target_structure[0][0]
    bfactor_map = {}
    residue_index = 1
    for residue in source_chain:
        for atom in residue:
            bfactor_map[(residue_index, atom.name)] = atom.b_iso
        residue_index += 1
    # Target chain gets 1-indexed. Adjust indexing in source structure to match that indexing!
    for residue in target_chain:
        for atom in residue:
            atom.b_iso = bfactor_map.get((residue.seqid.num, atom.name), 100.0)
    return target_structure

def slice_structure_tail(source_structure, target_structure):
    """
        this function will slice the end of the target pdb to be the same as the source pdb
    """
    source_chain = source_structure[0][0]
    target_chain = target_structure[0][0]
    source_chain_last_index = source_chain[-1].seqid.num
    while target_chain[-1].seqid.num > source_chain_last_index:
        del target_chain[-1]

def mse_to_met(st: gemmi.Structure) -> None:
   for model in st:
       for chain in model:
           for residue in chain:
               if residue.name == 'MSE':
                   residue.name = 'MET'
                   for atom in residue:
                       if atom.name == 'SE':
                           atom.name = 'SD'
                           atom.element = gemmi.Element('S')

def main(pdb_id, chain, root="pipeline_inputs"):
    structure = gemmi.read_pdb(os.path.join(root, "pdbs", pdb_id, f"{pdb_id}.pdb"))
    # mse_to_met(structure)
    pdb_file_paths = split_altlocs(structure, chain, root)
    space_group, unit_cell = structure.spacegroup_hm, structure.cell
    file_paths = []
    for pdb_file_path in tqdm(pdb_file_paths, "fixing pdbs"):
        fixer = pdbfixer.PDBFixer(filename=pdb_file_path)

        # TODO: fix that, for some reason it doesn't add the missing residues at the end
        # Fix things, save it and then filter out the ligands and waters
        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        processed_file_path = f"{pdb_file_path.split('.')[0]}_fixed.pdb"
        PDBFile.writeFile(fixer.topology, fixer.positions, open(processed_file_path, 'w'))

        source_structure = gemmi.read_pdb(pdb_file_path)
        target_structure = gemmi.read_pdb(processed_file_path)

        if get_chain_seq(pdb_file_path) == get_chain_seq(processed_file_path):
            fix_bfactor_same_seq(source_structure, target_structure)
        else:
            fix_bfactor_diff_seq(source_structure, target_structure)
        target_structure.spacegroup_hm = space_group
        target_structure.cell = unit_cell
        target_structure.write_pdb(processed_file_path)
        for chain in target_structure[0]:
            for i,res in enumerate(chain):
                res.seqid.num = i + 1
        file_paths.append(processed_file_path)
    print("Done!")
    return file_paths
