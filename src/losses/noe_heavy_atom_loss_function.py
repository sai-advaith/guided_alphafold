
from matplotlib import pyplot as plt
from tqdm import tqdm


from ..protenix.metrics.rmsd import self_aligned_rmsd

from ..utils.io import load_pdb_atom_locations
from .abstract_loss_funciton import AbstractLossFunction
import torch
import pandas as pd

EPSILON=1e-5

def q_group_heavy_atoms(nmr_name, residue_name):
    if residue_name == "TYR" and nmr_name == "QD":
        return "CD1"           # for HD1
    if residue_name == "TYR" and nmr_name == "QE":
        return "CE1"           # for HE1
    # if residue_name == "PHE" and nmr_name == "QD":
    #     return ["CD1", "CD2"]     # for HD1, HD2
    # if residue_name == "PHE" and nmr_name == "QE":
    #     return ["CE1", "CE2"]     # for HE1, HE2
    if residue_name == "TRP" and nmr_name == "QD":
        return "CD1"            # for HD1
    if residue_name == "TRP" and nmr_name == "QE":
        return "CE2"            # for HE1
    if residue_name == "TRP" and nmr_name == "QZ":
        return "CZ2"            # for HZ2
    if residue_name == "LYS" and nmr_name == "QZ":
        return "NZ" # for HZ1, HZ2, HZ3
    return -1

def calculate_within_chain_clash(
    coordinates: torch.Tensor,
    threshold: float = 1.1
) -> torch.Tensor:
    # Get pairwise distances between all atoms
    distances = torch.cdist(coordinates, coordinates)
    # Pick only the upper triangular part of the distance matrix ignoring the diagonal
    distances = distances.triu(diagonal=1)
    # Slice out upper triangular part
    distances = distances[distances > 0]
    # Penalize heavily if the distance is below the threshold
    loss = (torch.relu(threshold - distances)/0.25).exp()
    num_violations = (distances < threshold).sum()
    return loss.mean(), num_violations


   
class NOEHeavyAtomLossFunction(AbstractLossFunction):
    def __init__(self, restraint_file, pdb_file, atom_array=None, device="cpu", iid_loss=False):
        self.nmr_data = pd.read_csv(restraint_file)
        self.nmr_data = self.nmr_data[self.nmr_data["type"]=="NOE"]
        self.pdb_id = pdb_file.split("/")[1]
        
        self.last_loss = None
        self.atom_array = atom_array
        self.reference_atom_locations = load_pdb_atom_locations(pdb_file).to(device)
        self.device = device
        
        self.iid_loss = iid_loss
        self.num_violations = None
        self.within_chain_clash = None
        self.ub_loss_val = None
        self.lb_loss_val = None
        self.constraints_satisfied_ub = None
        self.constraints_satisfied_lb = None
        
        # gets indices of atoms that are going constrained to be between bounds, the mask is because some of the constraints maybe missing form the atom array
        self.guidance_params, mask = self.get_guidance_params()
        # self.nmr_data = self.nmr_data[mask].reset_index(drop=True)
        
        # get bounds and or conditions after masking the missing constraints
        # If a lower bound is equal to '.' set it to 0
        self.nmr_data["lower_bound"] = self.nmr_data["lower_bound"].apply(lambda x: 0 if x == '.' else x)
        self.lower_bound = torch.tensor(self.nmr_data["lower_bound"], dtype=torch.float32, device=device)[mask]
        self.upper_bound = torch.tensor(self.nmr_data["upper_bound"], dtype=torch.float32, device=device)[mask]
        # or conds
        or_cond = torch.tensor(self.nmr_data["constrain_id"], dtype=torch.float32, device=device)
        self.unique_or, self.inverse_or_indices = torch.unique(or_cond, return_inverse=True)
        self.inverse_or_indices = self.inverse_or_indices[mask]


    def get_guidance_params(self):
        atom_lookup = {(atom.res_id, atom.atom_name): i for i, atom in enumerate(self.atom_array)}
        guidance_params = []
        mask = []
        
        for _, row in tqdm(self.nmr_data.iterrows(), total=len(self.nmr_data), desc="generating comparison indices"):
            key1 = (row['residue1_num'], row['heavy_atom1'])
            key2 = (row['residue2_num'], row['heavy_atom2'])
            
            matching_atom1 = atom_lookup.get(key1, -1)
            matching_atom2 = atom_lookup.get(key2, -1)
            
            if matching_atom1 == -1 and "Q" in row['heavy_atom1']:
                a = q_group_heavy_atoms(row['heavy_atom1'], row['residue1_id'])
                matching_atom1 = atom_lookup.get((row['residue1_num'], a), -1)
            
            if matching_atom2 == -1 and "Q" in row['heavy_atom2']:
                a = q_group_heavy_atoms(row['heavy_atom2'], row['residue2_id'])
                matching_atom1 = atom_lookup.get((row['residue2_num'], a), -1)

            if matching_atom1 != -1 and matching_atom2 != -1:
                guidance_params.append(torch.tensor([matching_atom1, matching_atom2], dtype=torch.int))
                mask.append(True)
            else:
                print(f"Atom match not found for residues: {row['residue1_num']} - {row['residue2_num']} with atoms: {row['heavy_atom1']} - {row['heavy_atom2']}")
                mask.append(False)
        
        guidance_params = torch.stack(guidance_params, dim=0).to(self.device)
        return guidance_params, mask
    

    
    def wandb_log(self, x_0_hat):
        _, aligned_structure, _, _ = self_aligned_rmsd(x_0_hat, self.reference_atom_locations, torch.ones_like(x_0_hat, dtype=torch.bool)[...,0])
        rmsd_loss = (aligned_structure - self.reference_atom_locations[None]).norm(dim=-1).mean()
        
        return ({"loss": self.last_loss,
                 "rmsd_loss": rmsd_loss,
                 "num_violations": self.num_violations,
                 "within_chain_clash": self.within_chain_clash,
                 "ub_loss": self.ub_loss_val,
                 "lb_loss": self.lb_loss_val,
                 "constraints_satisfied_ub": self.constraints_satisfied_ub,
                 "constraints_satisfied_lb": self.constraints_satisfied_lb,
                 })
        
    def integrate_or_conditions(self, curr_loss):
        min_values = torch.zeros((curr_loss.shape[0], len(self.unique_or),),dtype=curr_loss.dtype, device=curr_loss.device)
        inverse_or_indices = self.inverse_or_indices[None].repeat(curr_loss.shape[0],1)
        dim=1
        min_values = torch.scatter_reduce(min_values, dim, inverse_or_indices, curr_loss, reduce="amin", include_self=False)
        
        return min_values

    def _compute_loss_bounds(self, dist):
        loss_ub = torch.relu(dist - self.upper_bound[None])
        loss_lb = torch.relu(self.lower_bound[None] - dist)
        # take or conditions into account
        loss_ub_or = self.integrate_or_conditions(loss_ub)
        loss_lb_or = self.integrate_or_conditions(loss_lb)
        
        loss_ub = loss_ub_or.mean()
        loss_lb = loss_lb_or.mean()
        
        return loss_ub, loss_lb, loss_ub_or, loss_lb_or

    def __call__(self, x_0_hat, time):
        x_0_hat = x_0_hat.to(self.device)
        
        atoms_to_compare_1 = x_0_hat[:,self.guidance_params[:,0]]
        atoms_to_compare_2 = x_0_hat[:,self.guidance_params[:,1]]
    
        model_distances = (atoms_to_compare_1 - atoms_to_compare_2).norm(dim=-1)
        model_distances = model_distances if self.iid_loss else model_distances.mean(dim=0)[None]
        
        self.ub_loss_val, self.lb_loss_val, loss_ub_or, loss_lb_or = self._compute_loss_bounds(model_distances)

        loss = self.ub_loss_val
        
        
        # logs
        self.constraints_satisfied_ub = ((loss_ub_or==0).sum().item() / len(self.unique_or)) 
        self.constraints_satisfied_lb = ((loss_lb_or==0).sum().item() / len(self.unique_or)) 
        within_chain_clash, num_violations = calculate_within_chain_clash(x_0_hat, 1.2)
        self.num_violations = num_violations
        self.within_chain_clash = within_chain_clash.item()
        self.last_loss = loss
        
        
        return loss, None
    

