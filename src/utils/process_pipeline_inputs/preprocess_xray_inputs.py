from .fix_pdb import main as fix_pdb
from .ed_targets_density import get_end_maps, get_2fofc_maps
from .extract_metadata import main as extract_metadata
import requests
import os
import yaml
import json
from copy import deepcopy
import gemmi

class FileNotFoundError(Exception):
    """Exception raised when file is not found at the URL."""
    pass

def fetch_pdb(pdb_id, root):
    root = os.path.join(root, "pdbs", pdb_id)

    # Fetch the pdb from pdb-redo
    pdb_url = f"https://pdb-redo.eu/db/{pdb_id.lower()}/{pdb_id.lower()}_final.pdb"
    os.makedirs(root, exist_ok=True)
    pdb_file_name = os.path.join(root, f"{pdb_id}.pdb")
    
    try:
        pdb_response = requests.get(pdb_url)
        if pdb_response.status_code == 404 or "This entry was not found in PDB-Redo" in pdb_response.text:
            raise FileNotFoundError(f"PDB file not found at {pdb_url}")
        pdb_response.raise_for_status()
        with open(pdb_file_name, "wb") as pdb_file:
            pdb_file.write(pdb_response.content)
        return pdb_file_name
    except requests.exceptions.RequestException as e:
        if os.path.exists(pdb_file_name):
            os.remove(pdb_file_name)
        raise FileNotFoundError(f"Failed to fetch PDB file: {str(e)}")

def fetch_mtz(pdb_id, root):
    root = os.path.join(root, "mtzs", pdb_id)
    # Fetch the mtz from the pdb-redo
    mtz_url = f"https://pdb-redo.eu/db/{pdb_id.lower()}/{pdb_id.lower()}_final.mtz"
    os.makedirs(root, exist_ok=True)
    mtz_file_name = os.path.join(root, f"{pdb_id}.mtz")

    try:
        mtz_response = requests.get(mtz_url)
        if mtz_response.status_code == 404 or "This entry was not found in PDB-Redo" in mtz_response.text:
            raise FileNotFoundError(f"MTZ file not found at {mtz_url}")
        mtz_response.raise_for_status()
        with open(mtz_file_name, "wb") as mtz_file:
            mtz_file.write(mtz_response.content)
        return mtz_file_name
    except requests.exceptions.RequestException as e:
        if os.path.exists(mtz_file_name):
            os.remove(mtz_file_name)
        raise FileNotFoundError(f"Failed to fetch MTZ file: {str(e)}")

def create_density_configuration_file_from_baseline(pdb_id, chain, input_directory, output_directory, ccp4_env_path, phenix_env_path, map_type, wandb_key, wandb_project, reference_baseline="pipeline_configurations/xray_baseline.yaml"):
    configurations_folder = "generated_configurations"
    os.makedirs(configurations_folder, exist_ok=True)

    with open(reference_baseline, "r") as f:
        config = yaml.safe_load(f)

    with open(f"{input_directory}/metadata/{pdb_id}/{pdb_id}.json", "r") as f:
        metadata = json.load(f)

    # Residue range
    residue_range = metadata["residue_region"]
    pdb_residue_range = metadata["pdb_residue_range"]
    config["general"]["name"] = f"{pdb_id}{chain}_{pdb_residue_range[0][0]}_{pdb_residue_range[0][1]}_{map_type}_guided"
    config["general"]["output_folder"] = f"{output_directory}/{map_type}"

    # Protein parameters (only monomers supported for now)
    config["protein"]["sequences"] = [{"count": 1, "sequence": metadata["seq"]}]
    config["protein"]["chains_to_use"] = [0]
    config["protein"]["assembly_identifier"] = None
    config["protein"]["pdb_id"] = pdb_id
    config["protein"]["residue_range"] = residue_range
    config["protein"]["pdb_residue_range"] = pdb_residue_range
    config["protein"]["reference_raw_pdb_chain"] = chain
    config["protein"]["reference_raw_pdb"] = f"{input_directory}/pdbs/{pdb_id}/{pdb_id}.pdb"
    config["protein"]["reference_pdb"] = f"{input_directory}/pdbs/{pdb_id}/{pdb_id}_chain_{chain}_altloc_A_fixed.pdb"

    # Loss function parameters
    config["loss_function"]["density_loss_function"]["density_file"] = os.path.join(input_directory, "densities", pdb_id, f"{pdb_id}_chain_{chain}_{map_type}_carved.ccp4")
    config["loss_function"]["density_loss_function"]["mtz_file"] = f"{input_directory}/mtzs/{pdb_id}/{pdb_id}.mtz"
    config["loss_function"]["density_loss_function"]["ccp4_env_path"] = ccp4_env_path
    config["loss_function"]["density_loss_function"]["phenix_env_path"] = phenix_env_path
    config["loss_function"]["density_loss_function"]["map_type"] = map_type

    if wandb_key and wandb_project:
        config["wandb"]["login_key"] = wandb_key
        config["wandb"]["mode"] = "online"
        config["wandb"]["project"] = wandb_project
    else:
        config["wandb"]["mode"] = "disabled"
        config["wandb"]["project"] = None
        config["wandb"]["login_key"] = None

    reference_pdbs = []
    for altloc in ["A", "B"]:
        altloc_file = f"{input_directory}/pdbs/{pdb_id}/{pdb_id}_chain_{chain}_altloc_{altloc}_fixed.pdb"
        if os.path.exists(altloc_file):
            reference_pdbs.append(altloc_file)

    config["loss_function"]["density_loss_function"]["reference_pdbs"] = reference_pdbs
    # guided
    guided_config = deepcopy(config)
    guided_config["general"]["apply_diffusion_guidance"] = True
    guided_name = guided_config['general']['name']
    guided_config_file_path = os.path.join(configurations_folder, f"{guided_name}.yaml")
    with open(guided_config_file_path, "w") as f:
        yaml.safe_dump(guided_config, f)

    return guided_config_file_path

def main(pdb_id, chain, region, input_directory, output_directory, ccp4_setup_sh, phenix_setup_sh, wandb_key, wandb_project, map_type="end"):
    fetch_pdb(pdb_id, input_directory)
    fix_pdb(pdb_id, chain, input_directory)
    extract_metadata(pdb_id, chain, region, input_directory)
    try:
        fetch_mtz(pdb_id, input_directory)
        if map_type == "2fofc":
            get_2fofc_maps(f"{input_directory}/pdbs/{pdb_id}/{pdb_id}_chain_{chain}_altloc_A_fixed.pdb", chain, ccp4_setup_sh, phenix_setup_sh, [pdb_id], input_directory) # 2FoFc maps
        elif map_type == "end":
            get_end_maps(f"{input_directory}/pdbs/{pdb_id}/{pdb_id}_chain_{chain}_altloc_A_fixed.pdb", chain, ccp4_setup_sh, phenix_setup_sh, [pdb_id], input_directory)
        else:
            raise ValueError(f"Invalid map type: {map_type}")
    except:
        raise ValueError("Could not fetch density")
    config_file_path = create_density_configuration_file_from_baseline(pdb_id, chain, input_directory, output_directory, ccp4_setup_sh, phenix_setup_sh, map_type, wandb_key, wandb_project, reference_baseline="pipeline_configurations/xray_baseline.yaml")
    return config_file_path
