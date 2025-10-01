import os
import subprocess
import re
from typing import Tuple
import gemmi
import pandas as pd

class PhenixManager:
    def __init__(self, phenix_setup_sh):
        self.phenix_setup_sh = phenix_setup_sh

    def run_shell_command(self, command: str, cwd: str = None) -> str:
        full_cmd = f"source {self.phenix_setup_sh} && {command}"
        result =  subprocess.run(
            ["/bin/bash", "-c", full_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
            check=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Phenix failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        return result.stdout

    def phenix_f000(self, pdb):
        output = self.run_shell_command(f"phenix.f000 {pdb}")
        f000 = float(re.search(r"F\(0,0,0\)=(\d+\.?\d*)", output)[1])
        return f000
    
    def phenix_refine(self, pdb, mtz, strategy="rigid_body", number_of_macro_cycles=1):
        return self.run_shell_command(f"phenix.refine {pdb} {mtz} strategy={strategy} main.number_of_macro_cycles={number_of_macro_cycles}")

    def calculate_rwork_rfree(self, pdb: str, mtz: str) -> Tuple[float, float]:
        f_obs_label = "FP"
        r_free_flags_label = "FREE"

        output = self.run_shell_command(f"phenix.model_vs_data {pdb} {mtz} f_obs_label={f_obs_label} r_free_flags_label={r_free_flags_label}")
        r_work = float(re.search(r"r_work:\s*([0-9]+\.[0-9]+)", output)[1])
        r_free = float(re.search(r"r_free:\s*([0-9]+\.[0-9]+)", output)[1])
        return r_work, r_free

    def phenix_real_space_correlation(self, pdb, mtz):
        return self.run_shell_command(f"phenix.real_space_correlation {pdb} {mtz} detail=residue")

    def parse_phenix_rscc_output(self, output_text):
        """Parse phenix.real_space_correlation output."""
        lines = output_text.splitlines()
        
        # Extract overall map correlation coefficient
        overall_cc = None
        for line in lines:
            if "Overall map cc(Fc,2mFo-DFc):" in line:
                overall_cc = float(line.split(":")[1].strip())
                break
        
        # Find the start of the residue data table
        data_start_idx = None
        for i, line in enumerate(lines):
            if "<id string>" in line and "occ" in line and "CC" in line:
                data_start_idx = i + 1
                break
        
        if data_start_idx is None:
            raise ValueError("Could not find residue data table in phenix output")
        
        # Parse residue data
        residue_data = []
        
        for line in lines[data_start_idx:]:
            line = line.strip()
            if not line:
                continue
                
            # Split by whitespace and handle cases with and without altloc
            parts = line.split()
            
            if len(parts) < 6:  # Need at least chain, resname, resid, occ, adp, cc
                continue
                
            # Determine if there's an altloc identifier
            # Check if the second field is a single character (altloc) or a 3-letter residue name
            if len(parts) >= 8 and len(parts[1]) == 1 and parts[1].isalpha():
                # Format with altloc: chain altloc resname resid occ adp cc rho1 rho2
                chain = parts[0]
                altloc = parts[1]
                resname = parts[2]
                resid = parts[3]
                occ = parts[4]
                adp = parts[5]
                cc = parts[6]
                rho1 = parts[7] if len(parts) > 7 else "0"
                rho2 = parts[8] if len(parts) > 8 else "0"
            elif len(parts) >= 7:
                # Format without altloc: chain resname resid occ adp cc rho1 rho2
                chain = parts[0]
                altloc = None
                resname = parts[1]
                resid = parts[2]
                occ = parts[3]
                adp = parts[4]
                cc = parts[5]
                rho1 = parts[6] if len(parts) > 6 else "0"
                rho2 = parts[7] if len(parts) > 7 else "0"
            else:
                continue
                
            try:
                residue_data.append({
                    'chain': chain,
                    'altloc': altloc,
                    'resname': resname,
                    'resid': int(resid),
                    'occupancy': float(occ),
                    'adp': float(adp),
                    'correlation': float(cc),
                    'rho1': float(rho1),
                    'rho2': float(rho2)
                })
            except ValueError:
                continue
        
        df = pd.DataFrame(residue_data)
        return overall_cc, df


    def get_rscc_for_residue_range(self, df, chain, start_resid, end_resid):
        """
        Get RSCC values organized by altloc using the exact logic requested:
        1. Check residue range for presence of altlocs
        2. Initialize list of lists with that size  
        3. Go through dataframe: if residue has altloc, add to corresponding list, otherwise append to both
        """
        target_data = df[(df['chain'] == chain) & 
                        (df['resid'] >= start_resid) & 
                        (df['resid'] <= end_resid)]
        
        if target_data.empty:
            return []
        
        # Step 1: Check for presence of altlocs in the range
        altlocs_in_range = sorted([alt for alt in target_data['altloc'].unique() if alt is not None])
        
        # Step 2: Initialize list of lists based on altloc presence
        if not altlocs_in_range:
            # No altlocs found - single list
            num_lists = 1
            list_labels = ['no_altloc']
        else:
            # Altlocs found - create list for each altloc
            num_lists = len(altlocs_in_range)
            list_labels = altlocs_in_range
        
        # Initialize the lists
        result_lists = [[] for _ in range(num_lists)]
        
        # Step 3: Go through each residue in the range
        for resid in range(start_resid, end_resid + 1):
            # Get all entries for this residue
            residue_entries = df[(df['chain'] == chain) & (df['resid'] == resid)]
            
            if residue_entries.empty:
                # No data found for residue - add None to all lists
                for i in range(num_lists):
                    result_lists[i].append(None)
                continue
            
            # Check if this residue has altlocs
            residue_altlocs = residue_entries[residue_entries['altloc'].notna()]
            residue_non_altlocs = residue_entries[residue_entries['altloc'].isna()]
            
            if not residue_altlocs.empty:
                # This residue HAS altlocs - add to corresponding lists
                altloc_values = {}
                for _, row in residue_altlocs.iterrows():
                    altloc_values[row['altloc']] = row['correlation']
                
                # Add to corresponding lists
                for i, list_label in enumerate(list_labels):
                    if list_label in altloc_values:
                        result_lists[i].append(altloc_values[list_label])
                    else:
                        result_lists[i].append(None)
                        
            else:
                # This residue does NOT have altlocs - add to ALL lists
                if not residue_non_altlocs.empty:
                    value = residue_non_altlocs['correlation'].iloc[0]
                    # Add to ALL lists
                    for i in range(num_lists):
                        result_lists[i].append(value)
                else:
                    # No data - add None to all lists
                    for i in range(num_lists):
                        result_lists[i].append(None)
        
        return result_lists

    def get_rscc_metrics(self, pdb, mtz, chain, pdb_residue_range):
        raw_output = self.phenix_real_space_correlation(pdb, mtz)
        _, raw_df = self.parse_phenix_rscc_output(raw_output)
        raw_rscc = self.get_rscc_for_residue_range(raw_df, chain, pdb_residue_range[0], pdb_residue_range[1])
        return raw_rscc

    def phenix_map_box(self, carve_pdb_path, map_path, wrapping=True, selection_radius=10, symmetry="P1"):
        shell_command = f"phenix.map_box {carve_pdb_path} {map_path} wrapping={wrapping} selection_radius={selection_radius} symmetry={symmetry}"
        return self.run_shell_command(shell_command)