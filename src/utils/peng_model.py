import torch
import numpy as np 
import os 

def _load_peng_element_scattering_factor_parameter_table():
    # TODO: do not use abolute path
    path = os.path.join(os.path.dirname(__file__), "peng1996_element_params.npy")
    atom_scattering_factor_params = np.load(path)
    return atom_scattering_factor_params# torch.tensor(atom_scattering_factor_params)

peng_element_scattering_factor_parameter_table = _load_peng_element_scattering_factor_parameter_table()

class ScatteringAttributes:
    def __init__(self, device):
        self.peng_element_scattering_factor_parameter_table = torch.from_numpy(
            _load_peng_element_scattering_factor_parameter_table()
        ).to(torch.float32).to(device=device)
        
    
    def __call__(self, atom_identities):
        return self.peng_element_scattering_factor_parameter_table[:, atom_identities]