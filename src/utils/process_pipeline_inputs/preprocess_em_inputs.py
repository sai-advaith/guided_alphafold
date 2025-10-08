import gzip
import requests
import os
import yaml
from copy import deepcopy
import shutil
import json
from tqdm import tqdm

class FileNotFoundError(Exception):
    """Exception raised when file is not found at the URL."""
    pass

def fetch_pdb(pdb_id, root):
    root = os.path.join(root, "pdbs", pdb_id)

    # Fetch the pdb from rcsb
    pdb_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    os.makedirs(root, exist_ok=True)
    pdb_file_name = os.path.join(root, f"{pdb_id}.pdb")
    
    try:
        pdb_response = requests.get(pdb_url)
        if pdb_response.status_code == 404:
            raise FileNotFoundError(f"PDB file not found at {pdb_url}")
        pdb_response.raise_for_status()
        with open(pdb_file_name, "wb") as pdb_file:
            pdb_file.write(pdb_response.content)
        return pdb_file_name
    except requests.exceptions.RequestException as e:
        if os.path.exists(pdb_file_name):
            os.remove(pdb_file_name)
        raise FileNotFoundError(f"Failed to fetch PDB file: {str(e)}")

def fetch_map(emdb_id, pdb_id, root):
    root = os.path.join(root, "maps", pdb_id)

    # Fetch the mtz from emdb
    map_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"
    os.makedirs(root, exist_ok=True)
    map_file_name = os.path.join(root, f"{emdb_id}.map.gz")

    # remote URL (gzipped)
    map_url = f"https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-{emdb_id}/map/emd_{emdb_id}.map.gz"

    # compressed local file
    gz_file = os.path.join(root, f"{emdb_id}.map.gz")

    # final uncompressed file
    map_file_name = os.path.join(root, f"emd_{emdb_id}.map")

    # download. can take some time depending on the connection speed......
    r = requests.get(map_url, stream=True)
    r.raise_for_status()
    with open(gz_file, "wb") as f:
        total_size = int(r.headers.get('content-length', 0))
        with tqdm(desc="fetching esp map", total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
    # gunzip into final .map file
    with gzip.open(gz_file, "rb") as f_in:
        with open(map_file_name, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # optional: remove the gzipped file
    os.remove(gz_file)

    return map_file_name


def create_em_configuration_file_from_baseline(pdb_id, sequences, counts, emdb_id, renumbered_file_path, assembly_identifier, input_directory, output_directory, wandb_key, wandb_project, reference_baseline, dihedrals_file=None, noe_restraints_file=None, noe_pdb_file=None):
    configurations_folder = "generated_configurations"
    os.makedirs(configurations_folder, exist_ok=True)

    # Reference yaml file
    with open(reference_baseline, "r") as f:
        config = yaml.safe_load(f)

    # General information
    config["general"]["name"] = f"{pdb_id}_assembly_{assembly_identifier}_em_guided"
    config["general"]["output_folder"] = f"{output_directory}"

    # Sequences and counts
    sequences_list, total_chains = [], 0
    for i in range(len(sequences)):
        total_chains += counts[i]
        sequences_list.append({"count": counts[i], "sequence": sequences[i]})

    # Protein parameters (update other hyperparameters in base yaml file)
    config["protein"]["sequences"] = sequences_list[:]
    config["protein"]["pdb_id"] = pdb_id
    config["protein"]["assembly_identifier"] = assembly_identifier
    config["protein"]["chains_to_use"] = [i for i in range(total_chains)]
    config["protein"]["reference_raw_pdb"] = renumbered_file_path
    config["protein"]["reference_pdb"] = renumbered_file_path
    config["protein"]["should_align_to_chains"] = [i for i in range(total_chains)]

    # NMR-restriaints loss
    if noe_restraints_file is not None:
        config["loss_function"]["loss_function_type"] = ["cryoesp", "nmr"]
        os.makedirs(os.path.join(input_directory, "nmr_restraints", pdb_id), exist_ok=True)
        shutil.copy(noe_restraints_file, os.path.join(input_directory, "nmr_restraints", pdb_id, os.path.basename(noe_restraints_file)))
        shutil.copy(noe_pdb_file, os.path.join(input_directory, "pdbs", pdb_id, os.path.basename(noe_pdb_file)))

        # Assign the file paths
        config["loss_function"]["nmr_loss_function"]["reference_nmr"] = os.path.join(input_directory, "nmr_restraints", pdb_id, os.path.basename(noe_restraints_file))
        config["loss_function"]["nmr_loss_function"]["pdb_file"] = os.path.join(input_directory, "pdbs", pdb_id, os.path.basename(noe_pdb_file))
    else:
        config["loss_function"]["loss_function_type"] = ["cryoesp"]

    # Dihedrals loss
    if dihedrals_file is not None:
        config["loss_function"]["cryoesp_loss_function"]["dihedrals"]["use_dihedrals"] = "from_nmr"
        config["loss_function"]["cryoesp_loss_function"]["dihedrals"]["dihedral_loss_weight"] = 1.0 # set weight
        os.makedirs(os.path.join(input_directory, "dihedrals", pdb_id), exist_ok=True)
        shutil.copy(dihedrals_file, os.path.join(input_directory, "dihedrals", pdb_id, os.path.basename(dihedrals_file)))
        config["loss_function"]["cryoesp_loss_function"]["dihedrals"]["dihedrals_file"] = os.path.join(input_directory, "dihedrals", pdb_id, os.path.basename(dihedrals_file))
    else:
        config["loss_function"]["cryoesp_loss_function"]["dihedrals"]["use_dihedrals"] = False
        config["loss_function"]["cryoesp_loss_function"]["dihedrals"]["dihedral_loss_weight"] = 0.0
        config["loss_function"]["cryoesp_loss_function"]["dihedrals"]["dihedrals_file"] = None

    config["loss_function"]["cryoesp_loss_function"]["reference_pdb"] = renumbered_file_path
    config["loss_function"]["cryoesp_loss_function"]["esp_file"] = f"{input_directory}/maps/{pdb_id}/emd_{emdb_id}.map"

    # wandb
    if wandb_key and wandb_project:
        config["wandb"]["login_key"] = wandb_key
        config["wandb"]["mode"] = "online"
        config["wandb"]["project"] = wandb_project
    else:
        config["wandb"]["mode"] = "disabled"
        config["wandb"]["project"] = None
        config["wandb"]["login_key"] = None

    # guided config copy
    guided_config = deepcopy(config)
    guided_config["general"]["apply_diffusion_guidance"] = True
    guided_config_file_path = os.path.join(configurations_folder, f"{pdb_id}_assembly_{assembly_identifier}_em_guided.yaml")
    with open(guided_config_file_path, "w") as f:
        yaml.safe_dump(guided_config, f)
    
    return guided_config_file_path

def save_metadata(pdb_id, assembly_identifier, sequences, counts, root):
    metadata_dir = os.path.join(root, "metadata", f"{pdb_id.lower()}")
    metadata_file_path = os.path.join(metadata_dir, f"{pdb_id.lower()}.json")

    metadata = {"pdb_id": pdb_id, "seq": sequences, "count": counts, "assembly_identifier": assembly_identifier}

    # Create dir
    os.makedirs(metadata_dir, exist_ok=True)

    # Write to it
    with open(metadata_file_path, 'w') as outfile:
        json.dump(metadata, outfile, indent=4)

def main(pdb_id, sequences, counts, emdb_id, renumbered_file_path, assembly_identifier, input_directory, output_directory, wandb_key, wandb_project, dihedrals_file=None, noe_restraints_file=None, noe_pdb_file=None):
    # Fetch the pdb file from rcsb
    fetch_pdb(pdb_id, input_directory)
    fetch_map(emdb_id, pdb_id, input_directory)

    # Move the renumbered file to the input directory
    shutil.copy(renumbered_file_path, os.path.join(input_directory, "pdbs", pdb_id))
    new_renumbered_file_path = os.path.join(input_directory, "pdbs", pdb_id, os.path.basename(renumbered_file_path))

    # Save metadata
    save_metadata(pdb_id, assembly_identifier, sequences, counts, input_directory)

    # Create the config file using template esp file
    config_file_path = create_em_configuration_file_from_baseline(pdb_id, sequences, counts, emdb_id, new_renumbered_file_path, assembly_identifier, input_directory, output_directory, wandb_key, wandb_project, reference_baseline="pipeline_configurations/cryo_baseline.yaml", dihedrals_file=dihedrals_file, noe_restraints_file=noe_restraints_file, noe_pdb_file=noe_pdb_file)
    return config_file_path