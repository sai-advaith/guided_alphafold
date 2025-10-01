import os
import subprocess
import re
from typing import Tuple
import gemmi
import numpy as np

class CCP4Manager:
    def __init__(self, ccp4_setup_sh):
        self.ccp4_setup_sh = ccp4_setup_sh

    def run_shell_command(self, command: str, cwd: str = None, input_string: str = None) -> str:
        full_cmd = f"source {self.ccp4_setup_sh} && {command}"
        result =  subprocess.run(
            ["/bin/bash", "-c", full_cmd],
            input=input_string,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            text=True,
            check=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Phenix failed:\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}")
        return result.stdout

    def run_refmac_refinement(self, xyzin, hklin, hklout=None, xyzout=None):
        # Refmac input string
        refmac_input = f"ncyc {5}\nend\n"

        if hklout is None:
            hklout = "temp.mtz"
        if xyzout is None:
            xyzout = "temp.pdb"

        # Build a shell command that sources CCP4, then runs REFMAC5
        shell_command = f"refmac5 hklin {hklin} hklout {hklout} xyzin {xyzin} xyzout {xyzout}"

        refmace_output = self.run_shell_command(shell_command, input_string=refmac_input)
        default_dict = {
                "initial_R": np.nan,
                "final_R": np.nan,
                "initial_Rfree": np.nan,
                "final_Rfree": np.nan,
                "log": None  # Optional: return full output for logging/debug

        }

        # --- Extract the "$TEXT:Result: $$ Final results $$" block only ---
        block_match = re.search(r"\$TEXT:Result: \$\$ Final results \$\$(.*?)\$\$", refmace_output, re.DOTALL)
        if not block_match:
            return default_dict

        final_block = block_match.group(1)

        # --- Extract initial and final R-factor and R-free from that block ---
        r_factor_match = re.search(r"R factor\s+([\d.]+)\s+([\d.]+)", final_block)
        r_free_match   = re.search(r"R free\s+([\d.]+)\s+([\d.]+)", final_block)

        if r_factor_match and r_free_match:
            initial_r, final_r = map(float, r_factor_match.groups())
            initial_rf, final_rf = map(float, r_free_match.groups())

            return {
                "initial_R": initial_r,
                "final_R": final_r,
                "initial_Rfree": initial_rf,
                "final_Rfree": final_rf,
                "log": refmace_output  # Optional: return full output for logging/debug
            }
        else:
            return default_dict
    
    def run_mapmask(self, mapin, mapout, axis="X Y Z"):
        shell_command = f"mapmask mapin {mapin} mapout {mapout} << EOF\nAXIS {axis}\nEOF\n"
        return self.run_shell_command(shell_command)

    def run_fft(self, hklin, mapout, f1, phi, sig1, f100, volume):
        shell_command = f"fft hklin {hklin} mapout {mapout} << EOF\nLABIN F1={f1} PHI={phi} SIG1={sig1}\nVF000 {volume} {f100}\nEOF\n"
        return self.run_shell_command(shell_command)