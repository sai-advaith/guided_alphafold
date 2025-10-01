import re
import subprocess
import numpy as np
import gemmi
import subprocess
from typing import List
import os
import gemmi
from src.utils.gemmi_ccp4_utils import *
import shutil
from src.utils.phenix_manager import PhenixManager
from src.utils.ccp4_manager import CCP4Manager

END_LINE = "@@@@@@@ThisIsTheEndOfTheCommands@@@@@@@@@\n"
bash_process = None

def wait_for_output():
    output_lines = []
    output_line = None
    while output_line != END_LINE:
        output_line = bash_process.stdout.readline()
        output_lines.append(output_line)
    return "\n".join(output_lines[:-1])

def run_commands(commands: List[str]):
    for command in commands:
        bash_process.stdin.write(command)
    bash_process.stdin.write(f"echo {END_LINE}")
    bash_process.stdin.flush()
    return wait_for_output()

def compute_volume(pdb_path):
    """
    Make sure input is in degrees
    """
    structure = gemmi.read_structure(pdb_path)
    
    # Extract unit cell
    cell = structure.cell
    a, b, c = cell.a, cell.b, cell.c
    alpha, beta, gamma = cell.alpha, cell.beta, cell.gamma

    # Volume calculation
    init_volume = a * b * c
    alpha, beta, gamma = np.radians(alpha), np.radians(beta), np.radians(gamma)
    angle_scaling = np.sqrt(1 - np.cos(alpha)**2 - np.cos(beta)**2 - np.cos(gamma)**2 + 2 * np.cos(gamma) * np.cos(alpha) * np.cos(beta))
    return init_volume * angle_scaling

def make_full_unit_cell(file_path, output_path):
    density_object = gemmi.read_ccp4_map(file_path)
    density_object.setup(0)
    density_object.grid.symmetrize_max()
    density_object.write_ccp4_map(output_path)

def end_cleanup(pdb, chain, current_dir, density_dir):
    """
    Remove verbose output from end map and copy the end map to the density directory
    """
    shutil.copy(f"{pdb}_chain_{chain}_end_carved.ccp4", density_dir)

    # Remove extensions
    extensions_to_remove = [".map", ".ccp4", ".eff", ".geo", ".mtz", ".def", ".pdb", ".cif", ".txt", ".log"]
    for extension in extensions_to_remove:
        for file_name in os.listdir(current_dir):
            if file_name.endswith(extension):
                os.remove(file_name)

    # Remove map vacuum level
    os.remove(os.path.join(current_dir, "map_vacuum_level.com"))

def cleanup_2fofc(pdb, chain, current_dir, density_dir):
    shutil.copy(f"{pdb}_chain_{chain}_2fofc_carved.ccp4", density_dir)

    # Remove box files
    # Remove extensions
    extensions_to_remove = [".ccp4", ".pdb"]
    for extension in extensions_to_remove:
        for file_name in os.listdir(current_dir):
            if file_name.endswith(extension):
                os.remove(file_name)

def get_end_maps(carve_pdb_path, chain, ccp4_setup_sh, phenix_setup_sh, pdbs_list=["3ohe"], root="pipeline_inputs"):
    global bash_process
    bash_process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    current_dir = os.getcwd()

    # Define managers
    ccp4_manager = CCP4Manager(ccp4_setup_sh)
    phenix_manager = PhenixManager(phenix_setup_sh)

    for pdb in pdbs_list:
        density_dir = f"{root}/densities/{pdb}"
        os.makedirs(density_dir, exist_ok=True)
        pdb_path = f"{root}/pdbs/{pdb}/{pdb}.pdb"
        mtz_path = f"{root}/mtzs/{pdb}/{pdb}.mtz"

        # Phenix refine
        print("RUNNING PHENIX REFINE")
        phenix_manager.phenix_refine(pdb_path, mtz_path)

        # Run END Rapid
        print("GENERATING END MAP")
        run_commands([f"source {ccp4_setup_sh}\n", f"source {phenix_setup_sh}\n", f"./END_RAPID.com {pdb}_refine_001.eff -norapid\n"])

        # Make full unit cell
        print("CARVING END MAP")
        make_full_unit_cell(f"2FoFc_END.map", f"{pdb}_2FoFc_END.map")

        # Carve around the pdb file
        phenix_manager.phenix_map_box(carve_pdb_path, f"{pdb}_2FoFc_END.map")
        carve_pdb_prefix = carve_pdb_path.split("/")[-1].split(".")[0]
        ccp4_manager.run_mapmask(f"{carve_pdb_prefix}_box.ccp4", f"{pdb}_chain_{chain}_end_carved.ccp4")

        # Remove verbouse output files
        end_cleanup(pdb, chain, current_dir, density_dir)
    bash_process.communicate()

def get_2fofc_maps(carve_pdb_path, chain, ccp4_setup_sh, phenix_setup_sh, pdbs_list=["3ohe"], root="pipeline_inputs"):
    global bash_process
    bash_process = subprocess.Popen(["/bin/bash"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    current_dir = os.getcwd()

    # Define managers
    ccp4_manager = CCP4Manager(ccp4_setup_sh)
    phenix_manager = PhenixManager(phenix_setup_sh)

    for pdb in pdbs_list:
        density_dir = f"{root}/densities/{pdb}"
        os.makedirs(density_dir, exist_ok=True)
        pdb_path = f"{root}/pdbs/{pdb}/{pdb}.pdb"
        mtz_path = f"{root}/mtzs/{pdb}/{pdb}.mtz"

        # 2Fofc file path change format to end
        map_output_path = f"{root}/densities/{pdb}/{pdb}_2fofc.map"

        pdb_path = os.path.abspath(pdb_path)
        mtz_path = os.path.abspath(mtz_path)
        map_output_path = os.path.abspath(map_output_path)

        f000, volume = phenix_manager.phenix_f000(pdb_path), gemmi.read_mtz_file(mtz_path).get_cell().volume

        # 2FOFC Flag
        f1, phi, sig1 = "FWT", "PHWT", "SIGFP"

        # Add to ccp4
        ccp4_manager.run_fft(mtz_path, map_output_path, f1, phi, sig1, f000, volume)
        run_commands([f"cd {root}/densities/{pdb}\n"])

        phenix_manager.phenix_map_box(carve_pdb_path, map_output_path)
        carve_pdb_prefix = carve_pdb_path.split("/")[-1].split(".")[0]
        ccp4_manager.run_mapmask(f"{carve_pdb_prefix}_box.ccp4", f"{pdb}_chain_{chain}_2fofc_carved.ccp4")

        cleanup_2fofc(pdb, chain, current_dir, density_dir)

        run_commands(["cd ../../..\n"])
    bash_process.communicate()
