import glob
from matplotlib import pyplot as plt
from tqdm import tqdm

from ..protenix.metrics.rmsd import self_aligned_rmsd

from .abstract_loss_funciton import AbstractLossFunction
import gemmi
import wandb
import torch
import pandas as pd
import numpy as np
import biotite.structure as struc
import numbers

from biotite.structure.io.pdb import PDBFile
   
EPSILON = 1e-5

def is_valid_float(x):
    try:
        return not pd.isna(float(x))
    except (ValueError, TypeError):
        return False

class S2LossFunction(AbstractLossFunction):
    def __init__(self, atom_array, s_2_file, device, type="amide_relax"):
        s_2_guidance_values = pd.read_csv(s_2_file)
        s_2_guidance_values = s_2_guidance_values[s_2_guidance_values['s2'].apply(is_valid_float)]
        s_2_guidance_values["s2"] = s_2_guidance_values["s2"].astype(float).values
        self.guid_res_ids = torch.tensor(s_2_guidance_values["res_id"].values, device=device)
        self.guid_s2_values = torch.tensor(s_2_guidance_values["s2"].values, device=device, dtype=torch.float32)
        self.atom_array = atom_array    
        self.last_loss = None
        # could be "methyl_rdc" "amide_rdc" or "amide_relax" or "methyl_relax"
        self.type=type
        self.s2_weights = torch.ones_like(self.guid_s2_values).to(device)
        self.guid_atom1 = None
        self.guid_atom2 = None
        if "amide" in type:
            if "s2_error" in s_2_guidance_values.columns:
                s2_std = torch.tensor(s_2_guidance_values["s2_error"].values, device=device, dtype=torch.float32)
                self.s2_weights = 1 / (s2_std**2 + EPSILON)
                # self.s2_weights = self.s2_weights / self.s2_weights.mean() # normalizing
                self.s2_weights = 1.0 + (self.s2_weights - self.s2_weights.min()) * (3.0 - 1.0) / (self.s2_weights.max() - self.s2_weights.min())
        if type == "methyl_rdc":
            mod_error = s_2_guidance_values["mod_error"].values
            s2_std = torch.tensor(mod_error.astype(float), device=device, dtype=torch.float32)
            self.s2_weights = 1 / (s2_std**2 + EPSILON)
            self.guid_atom1 = s_2_guidance_values["atom1"].values
            self.guid_atom2 = s_2_guidance_values["atom2"].values
            self.methyl_indices = self.get_carbon_indices()
            self.s2_weights = 1.0 + (self.s2_weights - self.s2_weights.min()) * (3.0 - 1.0) / (self.s2_weights.max() - self.s2_weights.min())
        if type == "methyl_relax":
            self.guid_atom1 = s_2_guidance_values["atom1"].values
            self.guid_atom2 = s_2_guidance_values["atom2"].values
            self.methyl_indices = self.get_carbon_indices()


            
    def get_carbon_indices(self):
        indices = []
        for res_id, atom1, atom2 in zip(self.guid_res_ids, self.guid_atom1, self.guid_atom2):
            res_id = res_id.item()
            idx1 = np.where((self.atom_array.atom_name == atom1) & (self.atom_array.res_id == res_id))[0][0]
            idx2 = np.where((self.atom_array.atom_name == atom2) & (self.atom_array.res_id == res_id))[0][0] 
            indices.append((idx1, idx2))
        return indices

    
    def wandb_log(self, x_0_hat):
        return ({"loss": self.last_loss
                 })
    
    def second_legendre(self, cos_theta):
        """
        Compute the second Legendre polynomial P2(x) = (3x^2 - 1)/2.
        """
        return 0.5 * (3.0 * cos_theta**2 - 1.0)

    

    def gather_nh_vectors(self, batch_coord, hyodrogens_coords, hydrogen_names):
        """
        For each residue, collect all N->H unit vectors across the ensemble
        of superimposed PDB files and their models.
        """
        residue_vectors = {}
        
        for res_id in np.unique(self.atom_array.res_id):
            mask = (self.atom_array.res_id == res_id) & (self.atom_array.atom_name == "N")
            # Find the backbone N coordinate
            coords_N = batch_coord[:,mask]
            
            # Find amide proton(s)
            mask = torch.tensor([(entry[0] == res_id) and (entry[2] in ["H", "HN", "H1", "H2", "H3", "HN1", "HN2"]) for entry in hydrogen_names], device=batch_coord.device)
            coords_H = hyodrogens_coords[:,mask]

            
            if coords_N.shape[1] == 1 and coords_H.shape[1] >= 1:
                # Take the first amide proton as representative
                # vector = coords_H[:,0] - coords_N[:,0] #TODO
                vector = coords_H[:,0] - coords_N[:,0]
                norm = torch.linalg.norm(vector, dim=-1)
                mask = norm > 1e-8
                vector = torch.where(mask.unsqueeze(-1), vector / norm.unsqueeze(-1), vector)
                residue_vectors.setdefault(res_id, []).append(vector)
                    
        return residue_vectors
    
    def gather_carbon_vectors(self, batch_coord):
        residue_vectors = {}

        for res_id, at1, at2, (idx1, idx2) in zip(self.guid_res_ids, self.guid_atom1, self.guid_atom2, self.methyl_indices):
            res_id = res_id.item()
            
            coord_C1 = batch_coord[:, idx1]
            coord_C2 = batch_coord[:, idx2]

            vector = coord_C1 - coord_C2
            norm = torch.linalg.norm(vector, dim=-1)
            mask = norm > 1e-8
            vector = torch.where(mask.unsqueeze(-1), vector / norm.unsqueeze(-1), vector)
            residue_vectors.setdefault(f"{res_id},{at1},{at2}", []).append(vector)
                    
        return residue_vectors
            
        

    def compute_order_parameters(self, residue_vectors):
        """
        Compute S^2 for each residue given a dictionary of N–H unit vectors
        across an ensemble.

        Uses the formula:
        S^2 = 1/[N(N-1)] * sum_{i != j}[ P2( d_i dot d_j ) ]
        where d_i are unit vectors, and P2 is the second Legendre polynomial.
        """
        results = {}
        for key, vectors in residue_vectors.items():
            if "," in str(key):
                res_id = int(key.split(",")[0])
            else:
                res_id = key
            if res_id == 1:
                continue
            vectors = torch.cat(vectors, dim=0)
            n_vec = len(vectors)
            if n_vec < 2:
                # Not enough snapshots to do a pairwise calculation
                results[key] = torch.nan
                continue

            # Dot product between all pairs
            dot_matrix = torch.einsum('ik,jk->ij', vectors, vectors)  # shape (n_vec, n_vec)
            p2_matrix = self.second_legendre(dot_matrix)

            # Sum over i != j
            sum_off_diag = torch.sum(p2_matrix) - torch.sum(torch.diag(p2_matrix))
            denom = n_vec * (n_vec - 1)
            s2 = sum_off_diag / denom
            results[key] = s2

        return results
    
    def plot_s2_values(self, pred_order_params, s_2_loss_val):
        # Plot the predicted and estimated s2
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        keys = [k.replace(",", "\n") if isinstance(k,str) else k for k in pred_order_params.keys()]
        pred_order_params = torch.stack(list(pred_order_params.values())).squeeze().cpu().detach().numpy()
        if not isinstance(keys[0], str):
            # if it's string then it's methyl and the keys are difenetly the same
            filtered = [(k, v) for k, v in zip(keys, pred_order_params) \
                    if k in self.guid_res_ids]
            keys, pred_order_params = zip(*filtered)
        ax.plot(keys, pred_order_params, label="predicted", marker="o")
        guid_s2_values = self.guid_s2_values[1:] if self.guid_res_ids[0]==1 else self.guid_s2_values
        ax.scatter(keys, guid_s2_values.squeeze().cpu().detach().numpy(), label="estimated", marker="x", color="red")
        ax.legend()
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("S2")
        # ax.set_xlim(0, 75)
        ax.set_ylim(0, 1)
        ax.set_title("S2 values. Error: {:.2f}".format(s_2_loss_val.item()))
        plt.tight_layout()
        return fig
    

    def __call__(self, x_0_hat, time, hyodrogens_coords, hydrogen_names):
        # align batch
        # Align with the first element of the batch
        x_0_ref = x_0_hat[[0]]
        _, x_0_hat, rot, trans = self_aligned_rmsd(x_0_hat, x_0_ref, torch.ones(x_0_hat.shape[1], dtype=torch.bool).to(x_0_hat.device))
        
        hyodrogens_coords = torch.matmul(hyodrogens_coords, rot.transpose(-1, -2)) + trans

        if "methyl" in self.type:
            residue_vectors = self.gather_carbon_vectors(x_0_hat)
        else:
            # Gather the N->H vectors from each superimposed structure
            residue_vectors = self.gather_nh_vectors(x_0_hat, hyodrogens_coords, hydrogen_names)
        
        # Compute the ensemble-derived S^2
        order_params = self.compute_order_parameters(residue_vectors)
        
        order_values = torch.stack(list(order_params.values())).to(self.guid_res_ids.device)  
        if isinstance(list(order_params.keys())[0], numbers.Integral) or "," not in list(order_params.keys())[0]:
            res_ids = torch.tensor(list(order_params.keys())).to(self.guid_res_ids.device)  
            indices = torch.where(self.guid_res_ids[:,None] == res_ids)[1]
            order_values = order_values[indices] 
        # do not compare to res_id 1 
        mask = self.guid_res_ids > 1
        guid_s2 = self.guid_s2_values[mask]
        loss = (order_values - guid_s2) ** 2
        loss = loss * self.s2_weights[mask]
        loss = torch.mean(loss)
        # loss = torch.norm((order_values - self.guid_s2_values), p=1)
        
        self.last_loss = loss
        return loss, order_params
    
