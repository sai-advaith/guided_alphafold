import gemmi
import argparse
import os 
import json
aa_map = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLU': 'E', 'GLN': 'Q', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'ASX': 'B', 'GLX': 'Z', 'XAA': 'X', 'SEC': 'U', 'PYL': 'O',
    'MSE': "M"
}

def merge_ensamble_folder(ensamble_folder, reference_fixed_chain, residue_range, metrics_file=None) -> gemmi.Chain:
    if metrics_file:
        with open(metrics_file, "r") as f:
            pdb_files = json.load(f)["selected_files"]
    else:
        pdb_files = [os.path.join(ensamble_folder, pdb_file) for pdb_file in os.listdir(ensamble_folder) if pdb_file.endswith(".pdb")]
    chains = [gemmi.read_pdb(pdb_file)[0][0] for pdb_file in pdb_files]
    reference_chain = gemmi.read_pdb(reference_fixed_chain)[0][0]
    for residue_index in range(residue_range[0] - 1, residue_range[1]):
        while len(reference_chain[residue_index]) > 0:
            atom = reference_chain[residue_index][0]
            reference_chain[residue_index].remove_atom(atom.name, atom.altloc)

    num_chains = len(chains)
    for residue_index in range(residue_range[0] - 1, residue_range[1]):
        for altloc_index, chain in enumerate(chains):
            for atom in chain[residue_index]:
                atom_clone = atom.clone()
                atom_clone.altloc = chr(ord("A") + altloc_index)
                atom_clone.occ = 1/num_chains
                reference_chain[residue_index].add_atom(atom_clone)
    return reference_chain

def get_chain_seq(pdb_file_path):
    structure = gemmi.read_structure(pdb_file_path)
    seq = ""
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.name in aa_map:
                    seq = seq + aa_map[residue.name]
    return seq

def get_residue_range(pdb_from_the_ensamble, residue_pattern):
    seq = get_chain_seq(pdb_from_the_ensamble)
    idx_start = seq.find(residue_pattern) + 1
    idx_end = idx_start + len(residue_pattern)-1
    return (idx_start, idx_end)

def main(reference_raw_pdb, reference_fixed_chain_pdb, chain, residue_range, ensamble_folder, metrics_file):
    structure = gemmi.read_pdb(reference_raw_pdb)
    # residue_range = get_residue_range(os.path.join(ensamble_folder, [file for file in os.listdir(ensamble_folder) if file.endswith(".pdb")][0]), residue_pattern)
    merged_chain = merge_ensamble_folder(ensamble_folder, reference_fixed_chain_pdb, residue_range, metrics_file)
    chain_index = ord(chain) - ord("A")
    model = structure[0]
    chains = [chain.clone() for chain in model]
    chains[chain_index] = merged_chain
    while len(model) > 0:
        del model[0]
    for i,chain in enumerate(chains):
        model.add_chain(chain, i, True)
    merged_file_path = os.path.join(ensamble_folder, "merged.pdb")
    structure.write_pdb(merged_file_path)
    return merged_file_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference_pdb', type=str, default="pdbs/6kug/6kug.pdb", required=False)
    parser.add_argument('--reference_fixed_chain_pdb', type=str, default="split_pdbs/6kug/chain_A/6kug_chain_A_altloc_A.pdb", required=False)
    parser.add_argument('--chain', type=str, default="C", required=False)
    parser.add_argument('--residue_pattern', type=str, default="EIDEE", required=False)
    parser.add_argument("--ensamble_folder", type=str, default="relaxed", required=False)
    parser.add_argument("--metrics_file", type=str, default="relaxed/metrics.json", required=False)
    args = parser.parse_args()
    main(args.reference_pdb, args.reference_fixed_chain_pdb, args.main, args.residue_pattern, args.ensamble_folder, args.metrics_file)