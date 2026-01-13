from ..utils.openfold_violations.violations import find_structural_violations, get_atom14_positions
import torch
from biotite.structure import connect_via_residue_names
import numpy as np
import gemmi

class AbstractLossFunction:  
    def wandb_log(self, x_0_hat):
        pass

    def __call__(self, x_0_hat, time, structures=None, i=None):
        pass
    
    def post_optimization_step(self, x_0_hat):
        return x_0_hat
        
    def pre_optimization_step(self, x_0_hat, i=None, step=None):
        return x_0_hat

    def save_state(self, structures, folder_path, **kwargs):
        pass