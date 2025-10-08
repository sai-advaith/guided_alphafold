import gemmi
import os
import json
from tqdm import tqdm
import requests
from src.utils.io import AMINO_ACID_ATOMS_ORDER
from Bio.PDB import PDBParser, PDBIO, Select
from Bio.PDB.Polypeptide import PPBuilder

aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'SEC': 'U', 'PYL': 'O',
    'MSE': "M",
    # modified amino acids
    "CSO": "C",
    # not amino acids but can be in the cahin 
    "HEM": "%",
    "HOH": "&",
    "IMD": "^"
}

def get_chain_seq(pdb):
    # fasta_seq_file = f"https://www.rcsb.org/fasta/entry/{pdb.upper()}"
    # return requests.get(fasta_seq_file).text.split("\n")[1]
    return "".join([aa_map[res.name] for res in gemmi.read_pdb(pdb)[0][0]])

def save_metadata(pdb_id, idx_list, seq, root, pdb_range, chain=None):
    metadata_dir = os.path.join(root, "metadata", f"{pdb_id.lower()}")
    metadata_file_path = os.path.join(metadata_dir, f"{pdb_id.lower()}.json")
    metadata = {"pdb_id": pdb_id, "seq": seq, "residue_region": idx_list, "pdb_residue_range": pdb_range}
    if chain is not None:
        metadata["chain"] = chain

    # Create dir
    os.makedirs(metadata_dir, exist_ok=True)

    # Write to it
    with open(metadata_file_path, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)

def get_one_index_seq_range(pdb_file_path, region):
    idx_range = []
    seq = get_chain_seq(pdb_file_path)
    if region is not None:
        idx_start = seq.find(region) + 1
        idx_end = idx_start + len(region) - 1

        idx_range.append([idx_start, idx_end])
    return idx_range


def get_pdb_index_seq_range(pdb_file_path, chain_id, region):
    # Parse the local PDB file
    parser = PDBParser()
    try:
        structure = parser.get_structure("local_pdb", pdb_file_path)
    except FileNotFoundError:
        print(f"Error: PDB file not found at '{pdb_file_path}'")
        return None, None
    
    # Get the specified chain
    try:
        chain = structure[0][chain_id]
    except KeyError:
        print(f"Chain '{chain_id}' not found in the PDB file.")
        return None, None

    # Extract the full sequence of the chain
    all_residues = []
    for res in chain:
        if res.resname in AMINO_ACID_ATOMS_ORDER.keys():
            all_residues.append(aa_map[res.resname])
    full_seq = "".join(all_residues)

    # Find the substring within the full sequence
    if region in full_seq:
        start_index = full_seq.find(region)
        end_index = start_index + len(region) - 1 # end is inclusive
        
        # Map the sequence index to PDB residue indices
        residues = list(chain.get_residues())
        start_pdb_index = residues[start_index].get_full_id()[3][1]
        end_pdb_index = residues[end_index].get_full_id()[3][1]
        
        return [[start_pdb_index, end_pdb_index]]
    else:
        print(f"Substring '{region}' not found in chain '{chain_id}'.")
        return None, None

def main(pdb_id, chain, region=None, root="pipeline_inputs"):
    pdb_id = pdb_id.lower()
    pdb_file_path = os.path.join(root, "pdbs", pdb_id, f"{pdb_id}_chain_{chain}_altloc_A_fixed.pdb")
    raw_pdb_file_path = os.path.join(root, "pdbs", pdb_id, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_file_path):
        print(f"Warning: {pdb_id} PDB not found!")
        return

    # 1-index Seq
    seq = get_chain_seq(pdb_file_path)
    idx_range = get_one_index_seq_range(pdb_file_path, region)

    # PDB indexing
    pdb_range = get_pdb_index_seq_range(raw_pdb_file_path, chain, region)

    # Save
    save_metadata(pdb_id, idx_range, seq, root, pdb_range, chain)