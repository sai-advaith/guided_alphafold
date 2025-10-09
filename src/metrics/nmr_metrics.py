
import torch

import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from biotite.structure.io import pdb
from loco_hd import *
from biotite.structure import AtomArray
from biotite.structure.atoms import stack
from biotite.structure.io import load_structure
from biotite.structure.chains import get_chain_starts
import argparse
import os

from ..utils.hydrogen_addition import add_hydrogen_to_pdb
from ..utils.relaxation import relax_pdb
from ..utils.io import load_pdb_atom_locations
from ..utils.calculate_noe import CalculateNOE
from ..utils.calculate_s2 import CalculateS2

miniconda_path = os.path.dirname(os.getcwd())
DSSP_PATH = f"{miniconda_path}/.conda/envs/proteinx/bin/mkdssp"

from biotite.application.dssp import DsspApp

def get_atom_array_from_pdb_file(path):
    if "7DAC_short_seqaligned" in path:
        # assumes dimer
        atom_arrays = load_structure(path)
        index_end_3_chain = get_chain_starts(atom_arrays)[3]
        atom_arrays = atom_arrays[:, :index_end_3_chain]
        # mask = np.isin(atom_arrays.chain_id, ["A-2"])
        # atom_arrays.chain_id[mask] = "B"
    else:
        file = pdb.PDBFile.read(path)
        # Convert to an AtomArray (for single model structures)
        model_count = file.get_model_count()
        atom_arrays = [file.get_structure(model=i) for i in range(1,model_count+1)]
        atom_arrays = stack(intersect_atom_array(atom_arrays))
    return atom_arrays[0] if len(atom_arrays) == 1 else atom_arrays

 

def intersect_atom_array(atom_arrays):
    """
    Return new AtomArrays with only atoms common to all inputs,
    based on (res_id, atom_name).
    """
    # Get (res_id, atom_name) pairs for each AtomArray
    key_sets = [set(zip(aa.res_id, aa.atom_name, aa.chain_id)) for aa in atom_arrays]

    # Find intersection of all key sets
    common_keys = sorted(set.intersection(*key_sets))

    # Filter each AtomArray using the common keys
    new_atom_arrays = []
    for atom_array in atom_arrays:
        mask = np.array([
            (res_id, atom_name, chain_id) in common_keys
            for res_id, atom_name, chain_id in zip(atom_array.res_id, atom_array.atom_name, atom_array.chain_id)
        ])
        new_atom_arrays.append(atom_array[mask])
    return new_atom_arrays



def ensure_hydrogens(structure_files):
    """Ensure each structure has hydrogens added."""
    updated_files = []
    for f in structure_files:
        hyd_file = f[:-4] + "_hyd_added.pdb"
        if not os.path.exists(hyd_file):
            add_hydrogen_to_pdb(f, hyd_file)
        updated_files.append(hyd_file)
    return updated_files


def ensure_relaxed(structure_files):
    """Ensure each structure has been relaxed."""
    updated_files = []
    for f in structure_files:
        relaxed_file = f[:-4] + "_colab_relaxed.pdb"
        if not os.path.exists(relaxed_file): 
            relax_pdb(f, relaxed_file, reorder_atoms=False, stiffness=100.0)
        updated_files.append(relaxed_file)
    return updated_files

def get_structures(folder, add_hydrogen, relax_colabfold):
    structures_files = [f"{folder}/{structure}" for structure in os.listdir(folder) if "hyd" not in structure and "colab" not in structure and ".pdb"in structure]
    structures_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    if add_hydrogen:
        structures_files = ensure_hydrogens(structures_files)
    if relax_colabfold:
        structures_files = ensure_relaxed(structures_files)
    atom_array = [get_atom_array_from_pdb_file(f) for f in structures_files]
    if relax_colabfold:
        # colabfold_relax adds/removes hydrogens, should make sure all are the same per file (single configuration)
        atom_array = intersect_atom_array(atom_array)
    structures = [torch.tensor(atom_array[i].coord, dtype=torch.float32) for i in range(len(structures_files))]
    structures = torch.stack(structures, dim=0)
    return structures, structures_files, atom_array


def generate_config_list(folder_path, pdb_id, add_hydrogen, relax_colabfold):
    configs_lst = []
    print(len(os.listdir(folder_path)))

    model_folder = "diffusion_process"
    if not os.path.exists(f"{folder_path}/{model_folder}"):
        print(f"Metrics for {pdb_id} failed")
        exit(0)

    structures, structures_files, atom_array = get_structures(f"{folder_path}/{model_folder}", add_hydrogen, relax_colabfold)
    assert intersect_atom_array(atom_array) == atom_array
    config_dict = {"folder": f"{folder_path}/{model_folder}", "structures": structures, "structures_files": structures_files, "atom_array": atom_array}
    config_dict["name"] = f"{pdb_id}_guided"
    configs_lst += [config_dict]

    return configs_lst


def fix_atom_arrays(configs_lst):
    atom_arrays = []
    batch_sizes = [0]
    for config in configs_lst:
        atom_arrays.extend(config["atom_array"])
        batch_sizes.append(len(config["atom_array"])+batch_sizes[-1])
    atom_arrays = intersect_atom_array(atom_arrays)
    for i,config in enumerate(configs_lst):
        config["intersected_atom_array"] = atom_arrays[batch_sizes[i]: batch_sizes[i+1]]
        structures = [torch.tensor(config["intersected_atom_array"][i].coord, dtype=torch.float32) for i in range(len(config["intersected_atom_array"]))]
        structures = torch.stack(structures, dim=0)
        config["intersected_structures"] = structures
    return configs_lst
        

def reorder_atom_arrays(configs_lst, non_alph_struct):
    def atom_key(atom):
        # Define a unique key for each atom — adjust as needed
        return (atom.atom_name, atom.res_id, atom.chain_id)

    # Get reference atom order
    ref_atom_array = configs_lst[0]["intersected_atom_array"][0]
    ref_keys = [atom_key(atom) for atom in ref_atom_array]

    for j in range(1, non_alph_struct+1):
        reordered_arrays = []
        j = -1*j
        for scrambled_array in configs_lst[j]["intersected_atom_array"]:
            scrambled_keys = [atom_key(atom) for atom in scrambled_array]
            key_to_atom = {key: atom for key, atom in zip(scrambled_keys, scrambled_array)}

            # Reorder to match reference
            atom_list = [key_to_atom[key] for key in ref_keys]
            atom_array = AtomArray(len(atom_list))
            for i, atom in enumerate(atom_list):
                atom_array[i] = atom
            reordered_arrays.append(atom_array)

        # Update configs_lst[-1]["atom_array"] in-place (optional)
        configs_lst[j]["intersected_atom_array"] = reordered_arrays
        structures = [torch.tensor(configs_lst[j]["intersected_atom_array"][i].coord, dtype=torch.float32) for i in range(len(configs_lst[j]["intersected_atom_array"]))]
        structures = torch.stack(structures, dim=0)
        configs_lst[j]["intersected_structures"] = structures
    
    return configs_lst


    

def process_file(file, add_hydrogen, relax_colabfold, evaluations, additional_protein_files, pdbs_output_folder, md_file_path, order_params_files=None):
    pdb_id = os.path.basename(file).split(".")[0]
    
    print(f"Processing {pdb_id}")
    
    gt_file = md_file_path
    gt_atom_stack = get_atom_array_from_pdb_file(gt_file)
    gt_atom_array = [gt_atom_stack] if isinstance(gt_atom_stack, AtomArray) else [gt_atom_stack[i] for i in range(gt_atom_stack.stack_depth())]
    gt_structure = torch.from_numpy(np.stack([a.coord for a in gt_atom_array])).to(torch.float32)
    gt_structure = gt_structure if len(gt_structure.shape)==3 else gt_structure[None]
    
    configs_lst = generate_config_list(pdbs_output_folder, pdb_id, add_hydrogen, relax_colabfold)
    configs_lst.append({"name": "MD", "folder": gt_file,"structures": gt_structure, "structures_files": [gt_file], "atom_array": gt_atom_array})

    non_alph_struct = 1

    if additional_protein_files:
        for f in additional_protein_files:
            print(f"Adding structure from {f}")
            atom_stack = get_atom_array_from_pdb_file(f)
            atom_array = [atom_stack] if isinstance(gt_atom_stack, AtomArray) else [atom_stack[i] for i in range(atom_stack.stack_depth())]
            structure = load_pdb_atom_locations(f,single_model=False)
            
            configs_lst.append({"name": f, "folder": f, "structures": structure, "structures_files": [f], "atom_array": atom_array})
            
            non_alph_struct += 1


    # colabfold_relax adds/removes hydrogens, should make sure all are the same for all configs so they will be compared according to the same restraints
    configs_lst = fix_atom_arrays(configs_lst)

    # ordering gt atomarray the same way like the rest    
    configs_lst = reorder_atom_arrays(configs_lst, non_alph_struct)    

    print("configs_lst len",len(configs_lst))

    results = []
    for config in tqdm(configs_lst):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        intersected_structures = config["intersected_structures"].to(device)        

        res_dict = {
                    "name": config["name"],
                    "pdb_id": pdb_id,
                    }
        for k,v in evaluations.items():
            if not v:
                continue
            print(f'key: {k}  pdb_id: {pdb_id}  name: {config["name"]}') 
            
            if k == "noe":
                calculator = CalculateNOE(file, config["intersected_atom_array"][0], device)      
                res_dict.update(calculator.run(intersected_structures))
            elif k == "s2":
                if not order_params_files:
                    raise ValueError("Please provide order parameters files for evaluation.")
                calculator = CalculateS2()
                res_dict.update(calculator.run(config["intersected_atom_array"], intersected_structures, order_params_files))

        results.append(res_dict)
    
    return results



def run_nmr_metrics(pdb_output_folder, md_file, restraint_file, add_hydrogen, relax_colabfold, results_path, additional_protein_files=None, order_params_files=None, noe=True, order_params=False):
    evaluations = {"noe": noe, "s2": order_params}
    results = process_file(restraint_file, add_hydrogen, relax_colabfold, evaluations, additional_protein_files, pdb_output_folder, md_file, order_params_files)
    results = pd.DataFrame(results)
    results.to_csv(results_path, index=False)
