from .aa_graphs import THREE_AAS_GRAPHS

from collections import defaultdict
import requests
import os
import pandas as pd
import pynmrstar
import re
import gemmi
import yaml
import json
from openmm.app import PDBFile
import pdbfixer
from copy import deepcopy
import shutil

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

def fix_pdb(pdb_file_path, pdb_id):
    # Extract first model, remove hydrogens and fix the pdb and save it!
    dir_name = os.path.dirname(pdb_file_path)
    structure = gemmi.read_structure(pdb_file_path)
    source_model = structure[0]

    # Remove hydrogens and create a new file
    source_model.remove_hydrogens()
    new = gemmi.Structure()
    new.name = structure.name
    new.spacegroup_hm = structure.spacegroup_hm
    new.cell = structure.cell

    # Create model (preserve original model name if present)
    model_name = source_model.name if source_model.name else "1"
    dst_model = gemmi.Model(model_name)

    for src_chain in source_model:
        dst_chain = gemmi.Chain(src_chain.name)
        for src_res in src_chain:
            dst_res = gemmi.Residue()
            dst_res.name = src_res.name
            dst_res.seqid = src_res.seqid
            dst_res.het_flag = src_res.het_flag
            for src_atom in src_res:
                dst_atom = gemmi.Atom()
                dst_atom.name = src_atom.name
                dst_atom.element = src_atom.element
                dst_atom.occ = src_atom.occ
                dst_atom.b_iso = src_atom.b_iso
                dst_atom.pos = src_atom.pos
                dst_atom.charge = src_atom.charge
                dst_atom.altloc = src_atom.altloc
                # copy anisotropic B if present
                if src_atom.aniso is not None:
                    dst_atom.aniso = src_atom.aniso
                dst_res.add_atom(dst_atom)
            dst_chain.add_residue(dst_res)
        dst_model.add_chain(dst_chain)

    new.add_model(dst_model)
    new_pdb_file_path = os.path.join(dir_name, f"{pdb_id}_fixed.pdb")
    new.setup_entities()
    new.write_pdb(new_pdb_file_path)

    # Fix it now
    fixer = pdbfixer.PDBFixer(filename=new_pdb_file_path)
    fixer.findMissingResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    PDBFile.writeFile(fixer.topology, fixer.positions, open(new_pdb_file_path, 'w'))
    return new_pdb_file_path

def download_nmr_restraints(pdb_id, save_dir):
    """
    Downloads the NMR restraints file (in mmCIF format) for a given PDB ID from the RCSB website.
    Attempts to download the uncompressed file first, and if that fails, tries the compressed version.
    If both fail, tries an alternative URL format.

    Args:
        pdb_id: The 4-character PDB ID.
        output_dir: The directory where the file should be saved.

    Returns:
        The path to the downloaded file if successful, otherwise None.
    """
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}_mr.str"  # URL for uncompressed file

    nmr_restraints_dir = os.path.join(save_dir, "restraints", pdb_id)
    os.makedirs(nmr_restraints_dir, exist_ok=True)
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_file_path = os.path.join(nmr_restraints_dir, f"{pdb_id}.str")
        with open(output_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"NMR restraints file for {pdb_id.upper()} downloaded successfully to {output_file_path}")
        return output_file_path

    except requests.exceptions.RequestException as e:
        print(f"Uncompressed restraints file not found for {pdb_id.upper()}: {e}")
        # Try downloading the compressed version
        gz_url = f"https://files.rcsb.org/download/{pdb_id.upper()}_mr.str.gz"
        try:
            gz_response = requests.get(gz_url, stream=True)
            gz_response.raise_for_status()
            gz_output_file_path = os.path.join(nmr_restraints_dir, f"{pdb_id}.str.cif.gz")
            with open(gz_output_file_path, 'wb') as f:
                for chunk in gz_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"Compressed NMR restraints file downloaded successfully to {gz_output_file_path}")
            return gz_output_file_path

        except requests.exceptions.RequestException as gz_e:
            print(f"Error downloading compressed restraints file for {pdb_id.upper()}: {gz_e}")
            
            # Try alternative URL format (using just the PDB ID as filename)
            alt_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.mr"
            try:
                alt_response = requests.get(alt_url, stream=True)
                alt_response.raise_for_status()
                alt_output_file_path = os.path.join(nmr_restraints_dir, f"{pdb_id}.str")
                with open(alt_output_file_path, 'wb') as f:
                    for chunk in alt_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                print(f"NMR restraints file downloaded using alternative URL for {pdb_id.upper()} to {alt_output_file_path}")
                return alt_output_file_path
                
            except requests.exceptions.RequestException as alt_e:
                print(f"Error downloading restraints using alternative URL for {pdb_id.upper()}: {alt_e}")
                
                if isinstance(gz_e, requests.exceptions.HTTPError) and gz_e.response.status_code == 404:
                    print("This may be because the structure does not have an associated restraints file or the PDB ID is incorrect.")

        return None

def download_pdb_file(pdb_id, save_dir):
    """
    Downloads the PDB file for a given PDB ID from the RCSB website.

    Args:
        pdb_id: The 4-character PDB ID.
        output_dir: The directory where the file should be saved.

    Returns:
        The path to the downloaded PDB file if successful, otherwise None.
    """
    pdb_id = pdb_id.lower()
    nmr_pdbs_dir = os.path.join(save_dir, "pdbs", pdb_id)
    os.makedirs(nmr_pdbs_dir, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        output_file_path = os.path.join(nmr_pdbs_dir, f"{pdb_id}.pdb")
        with open(output_file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                
        print(f"PDB file for {pdb_id.upper()} downloaded successfully to {output_file_path}")
        return output_file_path

    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDB file for {pdb_id.upper()}: {e}")
        return None

def guess_heavy_atom_old(hydrogen_name: str) -> str:
    """
    Given a hydrogen atom name (e.g., 'HA', 'HB2', 'HD21', 'HN', 'H',
    or old XPLOR-style 'MG1', 'MD2', etc.), guess the name of the heavy
    atom (usually carbon or nitrogen) to which it is attached, based on
    typical PDB or older NMR naming conventions.
    
    This version specifically ensures that 'HG12' becomes 'CG1'
    (i.e. it drops the final digit in the heavy-atom name).
    
    Real-world data can vary in naming, so this is just a heuristic.
    """

    # 1) Handle backbone amide hydrogen: N ↔ H / HN / H1 / H2 / H3
    if hydrogen_name in ['H', 'HN', 'H1', 'H2', 'H3']:
        return 'N'
    
    # 2) Handle alpha hydrogen(s): CA ↔ HA / HA2 / HA3
    if hydrogen_name.startswith('HA'):
        return 'CA'

    # 3) Regex to detect older methyl hydrogens: MG1, MD2, etc.
    #    We'll interpret 'M<letter><digits>' => 'C<letter>'
    #    For example, MG2 => CG, MD1 => CD, ME3 => CE, etc.
    match_methyl = re.match(r'^M([A-Z])\d*$', hydrogen_name)
    if match_methyl:
        letter_after_M = match_methyl.group(1)  # e.g. G, D, E
        return 'C' + letter_after_M             # e.g. CG, CD, CE

    # 4) Handle side-chain hydrogen names that begin with 'H' or 'D' (deuterons),
    #    BUT specifically parse out the final digit if it's a hydrogen index.
    #
    #    Example:
    #      - 'HG12' => side chain is 'G1', last digit '2' => return 'CG1'
    #      - 'HD21' => side chain is 'D2', last digit '1' => return 'CD2'
    #
    #    We'll look for the pattern:  ^H([A-Z0-9]+)(\d)$
    #    group(1) = the side-chain portion ('G1' or 'D2'), group(2) = the final digit (the proton index).
    
    match_sidechain = re.match(r'^H([A-Z0-9]+)(\d)$', hydrogen_name)
    if match_sidechain:
        # If we matched something like 'HG12',
        # group(1) = 'G1', group(2) = '2'
        # So the heavy atom is 'C' + 'G1' = 'CG1'
        return 'C' + match_sidechain.group(1)
    
    # 5) If it starts with 'H' (or 'D') but doesn't match the above pattern,
    #    fall back to a simpler replacement: 'H' -> 'C'.
    if hydrogen_name.startswith('H'):
        return 'C' + hydrogen_name[1:]

    # 6) If it's none of the above, return None or itself
    return None

def guess_heavy_atom(hydrogen_name: str, aa: str, three_aas_graphs: dict = THREE_AAS_GRAPHS) -> str:
    if aa not in three_aas_graphs:
        print(f"AA {aa} not found in three_aas_graphs")
        return None
    
    graph = three_aas_graphs[aa]
    if hydrogen_name in graph:
        return graph[hydrogen_name][0]
    else:
        print(f"Hydrogen {hydrogen_name} not found in {aa} graph")
        return None

# Extract loops and look for distance restraints
def extract_distance_restraints(star_file_path: str, verbose: bool = False) -> pd.DataFrame:
    # Load the STAR file
    try:
        entry = pynmrstar.Entry.from_file(star_file_path)
    except Exception as e:
        print(f"Error loading {star_file_path}: {e}")
        return pd.DataFrame()
    
    distance_restraints = []
    for saveframe in entry:
        # Check if this is a distance constraints saveframe
        if 'distance_constraints' in saveframe.category.lower():
            try:
                constraint_type = saveframe.get_tag('_Gen_dist_constraint_list.Constraint_type')[0]
            except:
                constraint_type = 'Unknown'
            
            # Find the loop containing the constraints
            for loop in saveframe:
                if '_Gen_dist_constraint' == loop.category:  # Exact match for main constraint loop
                    if verbose:
                        print("Tags:", loop.tags)  # Print the tags to see the structure
                    # Get index of seq_id_1
                    constrain_id = loop.tags.index("ID")
                    seq_id_1_index = loop.tags.index('Seq_ID_1')
                    comp_id_1_index = loop.tags.index('Comp_ID_1')
                    atom_id_1_index = loop.tags.index('Atom_ID_1')
                    seq_id_2_index = loop.tags.index('Seq_ID_2')
                    comp_id_2_index = loop.tags.index('Comp_ID_2')
                    atom_id_2_index = loop.tags.index('Atom_ID_2')
                    distance_val_index = loop.tags.index('Distance_val')
                    member_id_index = loop.tags.index('Member_ID')
                    member_logic_index = loop.tags.index('Member_logic_code')
                    lower_bound_index = loop.tags.index('Distance_lower_bound_val')
                    upper_bound_index = loop.tags.index('Distance_upper_bound_val')

                    for row in loop.data:
                        # Assuming standard column order based on NMR-STAR format
                        seq_id_1 = row[seq_id_1_index]    # Seq_ID_1
                        comp_id_1 = row[comp_id_1_index]   # Comp_ID_1
                        atom_id_1 = row[atom_id_1_index]   # Atom_ID_1
                        
                        seq_id_2 = row[seq_id_2_index]    # Seq_ID_2
                        comp_id_2 = row[comp_id_2_index]   # Comp_ID_2
                        atom_id_2 = row[atom_id_2_index]   # Atom_ID_2
                        
                        if (comp_id_1 == "THR") and (atom_id_1 == "HH"):
                            atom_id_1 = "HG1"
                            
                        if (comp_id_2 == "THR") and (atom_id_2 == "HH"):
                            atom_id_2 = "HG1"

                        # If atom1 is hydrogen, guess which heavy atom it’s attached to
                        if atom_id_1.startswith('H') or atom_id_1.startswith('M') or atom_id_1.startswith('Q'):
                            heavy_atom_1 = guess_heavy_atom(atom_id_1, comp_id_1)
                        else:
                            heavy_atom_1 = atom_id_1  # it's already heavy (C, N, etc.)

                        # If atom2 is hydrogen, guess which heavy atom it’s attached to
                        if atom_id_2.startswith('H') or atom_id_2.startswith('M') or atom_id_2.startswith('Q'):
                            heavy_atom_2 = guess_heavy_atom(atom_id_2, comp_id_2)
                        else:
                            heavy_atom_2 = atom_id_2
                        
                        distance_val = row[distance_val_index]    # Distance_val
                        lower_bound = row[lower_bound_index]     # Distance_lower_bound_val
                        upper_bound = row[upper_bound_index]     # Distance_upper_bound_val
                        
                        distance_restraints.append({
                            'type': constraint_type,
                            'constrain_id': row[constrain_id],
                            'member_id': row[member_id_index],
                            'member_logic': row[member_logic_index],
                            'residue1_num': seq_id_1,
                            'residue1_id': comp_id_1,
                            'atom1': atom_id_1,
                            'heavy_atom1': heavy_atom_1,
                            'residue2_num': seq_id_2,
                            'residue2_id': comp_id_2,
                            'atom2': atom_id_2,
                            'heavy_atom2': heavy_atom_2,
                            'distance': distance_val,
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound
                        })

    # Print the extracted distance restraints
    if verbose:
        for restraint in distance_restraints:
            print(
                f"Type: {restraint['type']}, "
                f"Residue 1: {restraint['residue1_num']}:{restraint['residue1_id']} {restraint['atom1']}, "
                f"Residue 2: {restraint['residue2_num']}:{restraint['residue2_id']} {restraint['atom2']}, "
                f"Heavy atoms: {restraint['heavy_atom1']}↔{restraint['heavy_atom2']}, "
                f"Distance: {restraint['distance']}, "
                f"Bounds: [{restraint['lower_bound']}, {restraint['upper_bound']}]"
            )

    # Save distance restraints to a pandas dataframe
    df = pd.DataFrame(distance_restraints)
    return df

def get_polymer_chain_sequences(pdb_file_path: str) -> list:
    """One-letter sequence per polymer chain in the first model."""
    sequences = []
    for chain in gemmi.read_pdb(pdb_file_path)[0]:
        sequence = "".join(aa_map[res.name] for res in chain if res.name in aa_map)
        # Skip ligand/solvent-only chains, which map to sentinel characters in aa_map.
        if sequence and not set(sequence) <= set("%&^"):
            sequences.append(sequence)
    return sequences


def get_amino_acid_sequence(pdb_file_path: str) -> str:
    return "".join([aa_map[res.name] for res in gemmi.read_pdb(pdb_file_path)[0][0]])

def save_metadata(pdb_id, sequence, root):
    metadata_dir = os.path.join(root, "metadata", f"{pdb_id.lower()}")
    os.makedirs(metadata_dir, exist_ok=True)
    metadata_file_path = os.path.join(metadata_dir, f"{pdb_id.lower()}.json")

    metadata = {"pdb_id": pdb_id, "seq": sequence}

    # Write to it
    with open(metadata_file_path, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)

def create_nmr_configuration_file_from_baseline(pdb_id, input_directory, output_directory,  wandb_key, wandb_project, methyl_rdc_file=None, amide_rdc_file=None, amide_relax_file=None, methyl_relax_file=None, baseline_config_file_path="pipeline_configurations/nmr_baseline.yaml"):
    configurations_folder = "generated_configurations"
    pdb_id = pdb_id.lower()
    os.makedirs(configurations_folder, exist_ok=True)

    with open(baseline_config_file_path, "r") as f:
        config = yaml.safe_load(f)

    # Metadata
    metadata_file_path = os.path.join(input_directory, "metadata", f"{pdb_id.lower()}", f"{pdb_id.lower()}.json")
    with open(metadata_file_path, "r") as f:
        metadata = json.load(f)

    config["general"]["name"] = f"{pdb_id}_nmr_guided"
    config["general"]["output_folder"] = f"{output_directory}"
    # Keep both MSA caches with the run's outputs rather than in the repo root, so a run is
    # self-contained and separate output directories do not share alignments or embeddings.
    config["model_manager"]["msa_save_dir"] = os.path.join(output_directory, "alignment_dir")
    config["model_manager"]["msa_embedding_cache_dir"] = os.path.join(output_directory, "msa_cache")

    # Protein parameters
    config["protein"]["sequences"] = [{"count": 1, "sequence": metadata["seq"], "sequence_type": "proteinChain"}]
    config["protein"]["chains_to_use"] = [0]
    config["protein"]["assembly_identifier"] = None
    config["protein"]["pdb_id"] = pdb_id
    config["protein"]["reference_raw_pdb_chain"] = "A"
    # Residue number the first sequence position carries in the restraint file. Output
    # PDBs are written in this numbering; the guidance loss shifts restraints to 1..n.
    config["protein"]["start_residue_from"] = int(start_residue_from or 1)
    config["protein"]["reference_raw_pdb"] = f"{input_directory}/pdbs/{pdb_id}/{pdb_id}.pdb"
    config["protein"]["reference_pdb"] = f"{input_directory}/pdbs/{pdb_id}/{pdb_id}.pdb"

    # Loss function parameters
    config["loss_function"]["nmr_loss_function"]["reference_nmr"] = f"{input_directory}/restraints/{pdb_id}/{pdb_id}.csv"
    config["loss_function"]["nmr_loss_function"]["pdb_file"] = f"{input_directory}/pdbs/{pdb_id}/{pdb_id}_fixed.pdb"

    # S2 scales. TODO set it
    methyl_rdc_scale = 0.0 if methyl_rdc_file is None else 0.5
    amide_rdc_scale = 0.0 if amide_rdc_file is None else 0.5
    amide_relax_scale = 0.0 if amide_relax_file is None else 0.5
    methyl_relax_scale = 0.0 if methyl_relax_file is None else 0.5

    # Order parameter scales + files
    config["loss_function"]["nmr_loss_function"]["methyl_rdc_file"] = methyl_rdc_file
    config["loss_function"]["nmr_loss_function"]["methyl_rdc_scale"] = methyl_rdc_scale
    config["loss_function"]["nmr_loss_function"]["amide_rdc_file"] = amide_rdc_file
    config["loss_function"]["nmr_loss_function"]["amide_rdc_scale"] = amide_rdc_scale
    config["loss_function"]["nmr_loss_function"]["amide_relax_file"] = amide_relax_file
    config["loss_function"]["nmr_loss_function"]["amide_relax_scale"] = amide_relax_scale
    config["loss_function"]["nmr_loss_function"]["methyl_relax_file"] = methyl_relax_file
    config["loss_function"]["nmr_loss_function"]["methyl_relax_scale"] = methyl_relax_scale

    # Wanbd key + wandb project
    if wandb_key and wandb_project:
        config["wandb"]["login_key"] = wandb_key
        config["wandb"]["mode"] = "online"
        config["wandb"]["project"] = wandb_project
    else:
        config["wandb"]["mode"] = "disabled"
        config["wandb"]["project"] = None
        config["wandb"]["login_key"] = None

    # Save config file path
    guided_config = deepcopy(config)
    guided_config["general"]["apply_diffusion_guidance"] = True
    guided_name = guided_config['general']['name']
    guided_config_file_path = os.path.join(configurations_folder, f"{guided_name}.yaml")
    with open(guided_config_file_path, "w") as f:
        yaml.safe_dump(guided_config, f)

    return guided_config_file_path

# Columns the guidance loss (NMRLossFunction) and the metrics (CalculateNOE) read by
# name. Anything else in the file is ignored. See README "Restraint file format".
REQUIRED_RESTRAINT_COLUMNS = [
    "type",
    "constrain_id",
    "residue1_num",
    "residue1_id",
    "atom1",
    "residue2_num",
    "residue2_id",
    "atom2",
    "lower_bound",
    "upper_bound",
]


def check_restraint_residue_numbering(df, sequences, start_residue_from=1, source=""):
    """
    Cross-check restraint residue numbering and names against the supplied sequence(s).

    A wrong --start_residue_from shows up as widespread residue-name disagreement, so
    that is treated as an error. Isolated disagreements are reported as warnings, since
    real restraint files do contain occasional mislabelled residues.
    """
    offset = int(start_residue_from) - 1
    longest = max(len(sequence) for sequence in sequences)
    where = f" in {source}" if source else ""

    observed = defaultdict(list)
    for _, row in df.iterrows():
        observed[int(row["residue1_num"])].append(str(row["residue1_id"]).upper())
        observed[int(row["residue2_num"])].append(str(row["residue2_id"]).upper())

    out_of_range = sorted(n for n in observed if not 1 <= n - offset <= longest)
    if out_of_range:
        raise ValueError(
            f"Restraint file{where} references residue number(s) {out_of_range[:8]}"
            f"{' ...' if len(out_of_range) > 8 else ''} which fall outside the sequence "
            f"after applying --start_residue_from {start_residue_from} "
            f"(valid file numbering is {1 + offset}..{longest + offset} for a "
            f"{longest}-residue sequence). Check the offset."
        )

    # Compare against every supplied sequence and keep the best agreement, so a
    # hetero-complex is not penalised for restraints belonging to its other chain.
    best_matches, best_mismatches = -1, []
    for sequence in sequences:
        matches, mismatches = 0, []
        for number, names in sorted(observed.items()):
            index = number - offset
            if not 1 <= index <= len(sequence):
                continue
            expected = gemmi.expand_one_letter(sequence[index - 1], gemmi.ResidueKind.AA)
            observed_name = max(set(names), key=names.count)
            if observed_name == expected:
                matches += 1
            else:
                mismatches.append((number, observed_name, index, expected))
        if matches > best_matches:
            best_matches, best_mismatches = matches, mismatches

    total = best_matches + len(best_mismatches)
    if total and best_matches / total < 0.5:
        detail = ", ".join(f"file {n}={g} but sequence[{i}]={e}" for n, g, i, e in best_mismatches[:5])
        raise ValueError(
            f"Restraint residue names disagree with the sequence for "
            f"{len(best_mismatches)}/{total} residues{where} using "
            f"--start_residue_from {start_residue_from}. This almost always means the "
            f"offset is wrong. Examples: {detail}."
        )
    if best_mismatches:
        print(
            f"NOTE: {len(best_mismatches)}/{total} restraint residues disagree with the "
            f"sequence name{where} (offset looks correct otherwise). Restraints are matched "
            f"by residue number and atom name, so these still apply to the intended residue:"
        )
        for number, got, index, expected in best_mismatches[:8]:
            print(f"  file residue {number} labelled {got}, sequence position {index} is {expected}")

    print(
        f"Residue numbering check: {best_matches}/{total} names agree with the sequence "
        f"(start_residue_from={start_residue_from})."
    )
    return best_matches, best_mismatches


def validate_restraint_csv(df, source=""):
    """
    Fail early and by name on a malformed restraint table.

    Without this, a missing or non-numeric column surfaces much later as an opaque
    torch.tensor dtype error from deep inside the loss constructor.
    """
    where = f" in {source}" if source else ""

    missing = [c for c in REQUIRED_RESTRAINT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Restraint file{where} is missing required column(s): {', '.join(missing)}. "
            f"Required columns are: {', '.join(REQUIRED_RESTRAINT_COLUMNS)}. "
            f"See the 'Restraint file format' table in the README."
        )

    if ("chain1" in df.columns) != ("chain2" in df.columns):
        raise ValueError(
            f"Restraint file{where} has only one of 'chain1'/'chain2'. Supply both "
            f"(for inter-chain restraints) or neither (all restraints within-chain)."
        )

    noe = df[df["type"] == "NOE"]
    if noe.shape[0] == 0:
        raise ValueError(
            f"Restraint file{where} contains no rows with type == 'NOE'. "
            f"Only NOE rows are used for guidance; found types: "
            f"{sorted(set(df['type'].astype(str)))}."
        )

    # constrain_id groups OR-alternatives and is cast to a numeric tensor downstream.
    if pd.to_numeric(noe["constrain_id"], errors="coerce").isna().any():
        raise ValueError(
            f"Restraint file{where} has non-numeric values in 'constrain_id'. "
            f"It must be an integer OR-group id."
        )

    # upper_bound has no "." handling downstream, unlike lower_bound.
    if pd.to_numeric(noe["upper_bound"], errors="coerce").isna().any():
        raise ValueError(
            f"Restraint file{where} has non-numeric values in 'upper_bound' "
            f"(note: '.' is accepted in 'lower_bound' but NOT in 'upper_bound'). "
            f"Every NOE row needs a numeric upper bound."
        )

    return noe.shape[0]


def prepare_custom_restraints(
    restraints_path, conformation_id, input_directory, sequences=None, start_residue_from=1
):
    """
    Normalize a user-supplied restraint file into the CSV the pipeline consumes.

    Accepts a CSV already in the documented schema, or a raw NMR-STAR restraint file
    (.str/.mr) converted with extract_distance_restraints(). Residue numbering is
    preserved as written; start_residue_from records how it maps onto the sequence.
    Returns the path to the CSV inside input_directory.
    """
    if not os.path.exists(restraints_path):
        raise FileNotFoundError(f"Restraint file not found: {restraints_path}")

    restraints_dir = os.path.join(input_directory, "restraints", conformation_id)
    os.makedirs(restraints_dir, exist_ok=True)
    csv_path = os.path.join(restraints_dir, f"{conformation_id}.csv")

    extension = os.path.splitext(restraints_path)[1].lower()
    if extension == ".csv":
        df = pd.read_csv(restraints_path)
    elif extension in (".str", ".mr"):
        print(f"Converting NMR-STAR restraints {restraints_path} -> {csv_path}")
        df = extract_distance_restraints(restraints_path, verbose=False)
        if df.shape[0] == 0:
            raise ValueError(
                f"No distance restraints could be extracted from {restraints_path}. "
                f"If this file is already in the pipeline CSV schema, pass it with a "
                f".csv extension instead."
            )
    else:
        raise ValueError(
            f"Unsupported restraint file '{restraints_path}' (extension '{extension}'). "
            f"Restraints must be a .csv in the schema documented in the README, or a "
            f".str/.mr NMR-STAR file. Convert other formats (CYANA .upl and similar) to "
            f"the CSV schema first."
        )

    n_noe = validate_restraint_csv(df, source=restraints_path)
    if sequences:
        check_restraint_residue_numbering(
            df[df["type"] == "NOE"], sequences,
            start_residue_from=start_residue_from, source=restraints_path,
        )
    df.to_csv(csv_path, index=False)
    print(f"Prepared {n_noe} NOE restraints for '{conformation_id}' -> {csv_path}")
    return csv_path


SUPPORTED_SEQUENCE_TYPES = ("proteinChain", "rnaSequence", "dnaSequence")


def normalize_chain_specification(sequences, counts=None, sequence_types=None):
    """
    Validate and normalize a multi-chain specification into config 'sequences' entries.

    Mirrors run_em.py: parallel lists of sequences, per-sequence copy counts, and
    molecule types. Defaults are one copy of each sequence, all protein chains.
    Returns (sequences_list, total_chains).
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    sequences = list(sequences)
    if not sequences:
        raise ValueError("At least one sequence must be supplied.")

    counts = [1] * len(sequences) if counts is None else [int(c) for c in counts]
    sequence_types = (
        ["proteinChain"] * len(sequences) if sequence_types is None else list(sequence_types)
    )

    if len(counts) != len(sequences):
        raise ValueError(
            f"--counts has {len(counts)} entries but --sequences has {len(sequences)}; "
            f"they must be parallel lists."
        )
    if len(sequence_types) != len(sequences):
        raise ValueError(
            f"--sequence_types has {len(sequence_types)} entries but --sequences has "
            f"{len(sequences)}; they must be parallel lists."
        )
    bad_types = [t for t in sequence_types if t not in SUPPORTED_SEQUENCE_TYPES]
    if bad_types:
        raise ValueError(
            f"Unsupported sequence type(s): {', '.join(bad_types)}. "
            f"Must be one of {', '.join(SUPPORTED_SEQUENCE_TYPES)}."
        )
    bad_counts = [c for c in counts if c < 1]
    if bad_counts:
        raise ValueError(f"--counts entries must be >= 1; got {bad_counts}.")

    sequences_list = [
        {"count": count, "sequence": sequence, "sequence_type": sequence_type}
        for sequence, count, sequence_type in zip(sequences, counts, sequence_types)
    ]
    return sequences_list, sum(counts)


def create_nmr_configuration_file_from_custom_inputs(
    conformation_id,
    sequence,
    restraints_csv_path,
    output_directory,
    wandb_key,
    wandb_project,
    reference_pdb=None,
    counts=None,
    sequence_types=None,
    start_residue_from=1,
    methyl_rdc_file=None,
    amide_rdc_file=None,
    amide_relax_file=None,
    methyl_relax_file=None,
    baseline_config_file_path="pipeline_configurations/nmr_baseline.yaml",
):
    """
    Build a guidance config from a sequence + restraints, with no deposited entry.

    Mirrors create_nmr_configuration_file_from_baseline() but takes the sequence and
    identifier directly instead of reading a scraped metadata/ file.
    """
    configurations_folder = "generated_configurations"
    os.makedirs(configurations_folder, exist_ok=True)

    with open(baseline_config_file_path, "r") as f:
        config = yaml.safe_load(f)

    config["general"]["name"] = f"{conformation_id}_nmr_guided"
    config["general"]["output_folder"] = f"{output_directory}"
    # Keep both MSA caches with the run's outputs rather than in the repo root, so a run is
    # self-contained and separate output directories do not share alignments or embeddings.
    config["model_manager"]["msa_save_dir"] = os.path.join(output_directory, "alignment_dir")
    config["model_manager"]["msa_embedding_cache_dir"] = os.path.join(output_directory, "msa_cache")

    # Chain parameters. `sequence` may be a single string (one chain) or a list of
    # sequences, optionally with per-sequence counts and molecule types.
    sequences_list, total_chains = normalize_chain_specification(
        sequence, counts=counts, sequence_types=sequence_types
    )
    config["protein"]["sequences"] = sequences_list
    config["protein"]["chains_to_use"] = list(range(total_chains))
    config["protein"]["should_align_to_chains"] = list(range(total_chains))
    config["protein"]["assembly_identifier"] = None
    config["protein"]["pdb_id"] = conformation_id
    config["protein"]["reference_raw_pdb_chain"] = "A"
    # Residue number the first sequence position carries in the restraint file. Output
    # PDBs are written in this numbering; the guidance loss shifts restraints to 1..n.
    config["protein"]["start_residue_from"] = int(start_residue_from or 1)

    # A reference structure is optional here: guidance needs only restraints, and the
    # topology comes from the sequence. When absent, metrics skip the MD comparison.
    config["protein"]["reference_pdb"] = reference_pdb
    config["protein"]["reference_raw_pdb"] = reference_pdb
    config["protein"]["contains_missing_atoms"] = reference_pdb is not None

    # Loss function parameters
    config["loss_function"]["nmr_loss_function"]["reference_nmr"] = restraints_csv_path
    config["loss_function"]["nmr_loss_function"]["pdb_file"] = reference_pdb

    # Order parameter scales + files
    methyl_rdc_scale = 0.0 if methyl_rdc_file is None else 0.5
    amide_rdc_scale = 0.0 if amide_rdc_file is None else 0.5
    amide_relax_scale = 0.0 if amide_relax_file is None else 0.5
    methyl_relax_scale = 0.0 if methyl_relax_file is None else 0.5

    config["loss_function"]["nmr_loss_function"]["methyl_rdc_file"] = methyl_rdc_file
    config["loss_function"]["nmr_loss_function"]["methyl_rdc_scale"] = methyl_rdc_scale
    config["loss_function"]["nmr_loss_function"]["amide_rdc_file"] = amide_rdc_file
    config["loss_function"]["nmr_loss_function"]["amide_rdc_scale"] = amide_rdc_scale
    config["loss_function"]["nmr_loss_function"]["amide_relax_file"] = amide_relax_file
    config["loss_function"]["nmr_loss_function"]["amide_relax_scale"] = amide_relax_scale
    config["loss_function"]["nmr_loss_function"]["methyl_relax_file"] = methyl_relax_file
    config["loss_function"]["nmr_loss_function"]["methyl_relax_scale"] = methyl_relax_scale

    if wandb_key and wandb_project:
        config["wandb"]["login_key"] = wandb_key
        config["wandb"]["mode"] = "online"
        config["wandb"]["project"] = wandb_project
    else:
        config["wandb"]["mode"] = "disabled"
        config["wandb"]["project"] = None
        config["wandb"]["login_key"] = None

    guided_config = deepcopy(config)
    guided_config["general"]["apply_diffusion_guidance"] = True
    guided_config_file_path = os.path.join(
        configurations_folder, f"{guided_config['general']['name']}.yaml"
    )
    with open(guided_config_file_path, "w") as f:
        yaml.safe_dump(guided_config, f)

    return guided_config_file_path


def main_from_custom_inputs(
    conformation_id,
    sequence,
    restraints_path,
    input_directory,
    output_directory,
    wandb_key,
    wandb_project,
    reference_pdb=None,
    counts=None,
    sequence_types=None,
    start_residue_from=1,
    methyl_rdc_file=None,
    amide_rdc_file=None,
    amide_relax_file=None,
    methyl_relax_file=None,
):
    """Custom-input entry point: conformation id + sequence + restraint file."""
    baseline_config_file_path = "pipeline_configurations/nmr_baseline.yaml"

    sequence_list = [sequence] if isinstance(sequence, str) else list(sequence)
    restraints_csv_path = prepare_custom_restraints(
        restraints_path, conformation_id, input_directory,
        sequences=sequence_list, start_residue_from=start_residue_from or 1,
    )

    order_parameters_dir = os.path.join(input_directory, "order_parameters", conformation_id)
    order_param_paths = {}
    for label, source in (
        ("methyl_rdc", methyl_rdc_file),
        ("amide_rdc", amide_rdc_file),
        ("amide_relax", amide_relax_file),
        ("methyl_relax", methyl_relax_file),
    ):
        if source is None:
            order_param_paths[label] = None
            continue
        os.makedirs(order_parameters_dir, exist_ok=True)
        destination = os.path.join(order_parameters_dir, os.path.basename(source))
        shutil.copy(source, destination)
        order_param_paths[label] = destination

    return create_nmr_configuration_file_from_custom_inputs(
        conformation_id,
        sequence,
        restraints_csv_path,
        output_directory,
        wandb_key,
        wandb_project,
        reference_pdb=reference_pdb,
        counts=counts,
        sequence_types=sequence_types,
        start_residue_from=start_residue_from or 1,
        methyl_rdc_file=order_param_paths["methyl_rdc"],
        amide_rdc_file=order_param_paths["amide_rdc"],
        amide_relax_file=order_param_paths["amide_relax"],
        methyl_relax_file=order_param_paths["methyl_relax"],
        baseline_config_file_path=baseline_config_file_path,
    )


def main(pdb_id, input_directory, output_directory, wandb_key, wandb_project, methyl_rdc_file=None, amide_rdc_file=None, amide_relax_file=None, methyl_relax_file=None):
    # Input files
    baseline_config_file_path="pipeline_configurations/nmr_baseline.yaml"
    reatraints_file_path = download_nmr_restraints(pdb_id, input_directory)
    pdb_file_path = download_pdb_file(pdb_id, input_directory)
    fixed_pdb_file_path = fix_pdb(pdb_file_path, pdb_id)
    amino_acid_sequence = get_amino_acid_sequence(fixed_pdb_file_path)

    # This path models the first chain only. Rather than silently discard the rest of a
    # complex, direct the user to the explicit multi-chain flags.
    chain_sequences = get_polymer_chain_sequences(fixed_pdb_file_path)
    if len(chain_sequences) > 1:
        raise ValueError(
            f"{pdb_id.upper()} has {len(chain_sequences)} polymer chains, but PDB-ID mode "
            f"builds a single-chain model and would silently keep only the first.\n"
            f"Re-run with the explicit multi-chain flags, for example:\n"
            f"  python3 run_nmr.py --conformation_id {pdb_id} \\\n"
            f"      --sequences {' '.join(chain_sequences)} \\\n"
            f"      --counts {' '.join(['1'] * len(chain_sequences))} \\\n"
            f"      --restraints {os.path.join(input_directory, 'restraints', pdb_id, pdb_id + '.csv')}\n"
            f"(Identical chains can be collapsed into one --sequences entry with a higher --counts value.)"
        )

    # Save metadata
    save_metadata(pdb_id, amino_acid_sequence, input_directory)

    # Save the order parameter files
    order_parameters_dir = os.path.join(input_directory, "order_parameters", pdb_id)
    if methyl_rdc_file is not None or amide_rdc_file is not None or amide_relax_file is not None or methyl_relax_file is not None:
        os.makedirs(order_parameters_dir, exist_ok=True)
    else:
        methyl_rdc_file_path = None
        amide_rdc_file_path = None
        amide_relax_file_path = None
        methyl_relax_file_path = None

    if methyl_rdc_file is not None:
        methyl_rdc_file_path = os.path.join(order_parameters_dir, os.path.basename(methyl_rdc_file))
        shutil.copy(methyl_rdc_file, methyl_rdc_file_path)
    if amide_rdc_file is not None:
        amide_rdc_file_path = os.path.join(order_parameters_dir, os.path.basename(amide_rdc_file))
        shutil.copy(amide_rdc_file, amide_rdc_file_path)
    if amide_relax_file is not None:
        amide_relax_file_path = os.path.join(order_parameters_dir, os.path.basename(amide_relax_file))
        shutil.copy(amide_relax_file, amide_relax_file_path)
    if methyl_relax_file is not None:
        methyl_relax_file_path = os.path.join(order_parameters_dir, os.path.basename(methyl_relax_file))
        shutil.copy(methyl_relax_file, methyl_relax_file_path)

    # Prepare restraints for guidance
    df = extract_distance_restraints(reatraints_file_path, verbose=False)
    restraints_csv_file_path = os.path.join(input_directory, "restraints", pdb_id, f"{pdb_id}.csv")
    if df.shape[0] > 0:
        print(f"Found {df.shape[0]} distance restraints for {pdb_id}")
        print(df.groupby('type').size())
        df.to_csv(restraints_csv_file_path, index=False)
    else:
        print(f"No distance restraints found for {pdb_id}")
        exit(0)

    # Update config file path
    config_file_path = create_nmr_configuration_file_from_baseline(pdb_id, input_directory, output_directory, wandb_key, wandb_project, methyl_rdc_file_path, amide_rdc_file_path, amide_relax_file_path, methyl_relax_file_path, baseline_config_file_path)
    return config_file_path
