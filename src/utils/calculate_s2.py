import json
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from src.protenix.metrics.rmsd import self_aligned_rmsd
import torch

def is_valid_float(x):
    try:
        return not pd.isna(float(x))
    except (ValueError, TypeError):
        return False
    

def second_legendre(cos_theta):
    """
    Compute the second Legendre polynomial P2(x) = (3x^2 - 1)/2.
    """
    return 0.5 * (3.0 * cos_theta**2 - 1.0)
    
class CalculateS2:
    def __init__(self, ):
        pass
    
    def get_carbon_indices(self, atom_array, guid_res_ids, guid_atom1, guid_atom2):
        indices = []
        for res_id, atom1, atom2 in zip(guid_res_ids, guid_atom1, guid_atom2):
            res_id = res_id.item()
            idx1 = np.where((atom_array.atom_name == atom1) & (atom_array.res_id == res_id))[0][0]
            idx2 = np.where((atom_array.atom_name == atom2) & (atom_array.res_id == res_id))[0][0] 
            indices.append((idx1, idx2))
        return indices
        
    def gather_carbon_vectors(self, batch_coord, atom_array, path):
        residue_vectors = {}
        s_2_guidance_values = self.get_gt_s2(path)
        guid_res_ids = torch.tensor(s_2_guidance_values["res_id"].values, device=batch_coord.device)
        guid_atom1 = s_2_guidance_values["atom1"].values
        guid_atom2 = s_2_guidance_values["atom2"].values
        methyl_indices = self.get_carbon_indices(atom_array[0], guid_res_ids, guid_atom1, guid_atom2)

        for res_id, at1, at2, (idx1, idx2) in zip(guid_res_ids, guid_atom1, guid_atom2, methyl_indices):
                res_id = res_id.item()
                
                coord_C1 = batch_coord[:, idx1]
                coord_C2 = batch_coord[:, idx2]

                vector = coord_C1 - coord_C2
                norm = torch.linalg.norm(vector, dim=-1)
                mask = norm > 1e-8
                vector = torch.where(mask.unsqueeze(-1), vector / norm.unsqueeze(-1), vector)
                residue_vectors.setdefault(f"{res_id},{at1},{at2}", []).append(vector)
                    
        return residue_vectors
    
    def gather_nh_vectors(self, structures, atom_array):
        """
        For each residue, collect all N->H unit vectors across the ensemble
        of superimposed PDB files and their models.
        """
        residue_vectors = {}
        
        for res_id in np.unique(atom_array[0].res_id):
            residue_mask = (atom_array[0].res_id == res_id)
            residue_atoms = atom_array[0][residue_mask]

            # Find the backbone N coordinate
            mask = residue_atoms.atom_name == "N"
            coords_N = structures[:,residue_mask][:, mask]

            # Find amide proton(s)
            coords_H = structures[:,residue_mask][:, np.isin(
                residue_atoms.atom_name, ["H", "HN", "H1", "H2", "H3", "HN1", "HN2"]
            )]

            
            if coords_N.shape[1] == 1 and coords_H.shape[1] >= 1:
                # Take the first amide proton as representative
                # vector = coords_H[:,0] - coords_N[:,0] #TODO
                vector = coords_H[:,0] - coords_N[:,0]
                norm = torch.linalg.norm(vector, dim=-1)
                mask = norm > 1e-8
                vector = torch.where(mask.unsqueeze(-1), vector / norm.unsqueeze(-1), vector)
                residue_vectors.setdefault(str(res_id), []).append(vector)
                    
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
            if "," in key:
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
            p2_matrix = second_legendre(dot_matrix)

            # Sum over i != j
            sum_off_diag = torch.sum(p2_matrix) - torch.sum(torch.diag(p2_matrix))
            denom = n_vec * (n_vec - 1)
            s2 = sum_off_diag / denom
            results[key] = s2.item()

        return results
        
    def calculate_s2_loss(self, structures, atom_array, s2_type, path):
        ref = structures[[0]]
        _, structures, _, _ = self_aligned_rmsd(structures, ref, torch.ones(structures.shape[1], dtype=torch.bool).to(structures.device))
        if "methyl" in s2_type:
            residue_vectors = self.gather_carbon_vectors(structures, atom_array, path)
        else:
            # Gather the N->H vectors from each superimposed structure
            residue_vectors = self.gather_nh_vectors(structures, atom_array)
        
        if "ubi" in path:
            residue_vectors = {k:v for k,v in residue_vectors.items() if int(k.split(",")[0]) <= 70}
            
        # Compute the ensemble-derived S^2
        order_params = self.compute_order_parameters(residue_vectors)
        # move residue_vectors to numpy cpu
        updated_residue_vectors = {}
        for key in residue_vectors.keys():
            updated_residue_vectors[str(key)] = residue_vectors[key][0].cpu().numpy().tolist()
        return order_params, updated_residue_vectors

    def get_gt_s2(self, path):
        gt = pd.read_csv(path)
        gt = gt[gt['s2'].apply(is_valid_float)]
        gt["s2"] = gt["s2"].astype(float).values
        return gt

    
    def compute_salmon(self, predicted_order_parameters):
        salmon = pd.read_csv("src/metrics/s2/Salmon.txt", header=None, delimiter=" ")
        salmon_res_id = salmon.iloc[:, 0].astype(str)
        salmon_s2 = salmon.iloc[:, 1]
        salmon_dict = dict(zip(salmon_res_id, salmon_s2))
        salmon_dict = {k: v for k, v in salmon_dict.items() if int(k) <= 70}
        salmon_diff = [salmon_dict[res_id] - predicted_order_parameters[res_id] for res_id in predicted_order_parameters.keys() if res_id in salmon_dict]
        filtered_salmon_order_params = [predicted_order_parameters[res_id] for res_id in predicted_order_parameters.keys() if res_id in salmon_dict]
        salmon_order_params_list = [salmon_dict[res_id] for res_id in predicted_order_parameters.keys() if res_id in salmon_dict]
        salmon_corr = np.corrcoef(np.array(salmon_order_params_list), np.array(filtered_salmon_order_params))[0, 1]
        salmon_error = np.nanmean((np.array(salmon_diff))**2)
        
        return salmon_corr, salmon_error, salmon_dict
    
    def compute_s2_correlation(self, predicted_order_parameters,s2_type,path):
        gt = self.get_gt_s2(path)
        if "methyl" in s2_type:
            gt_res_id = [f"{res_id},{a1},{a2}" for res_id,a1,a2 in zip(gt["res_id"].values, gt["atom1"].values, gt["atom2"].values)]
        else:
            gt_res_id = [str(r) for r in gt["res_id"].values]
        gt_s2 = np.array(gt["s2"])
        gt_dict = dict(zip(gt_res_id, gt_s2))
        
        
        diff = [gt_dict[res_id] - predicted_order_parameters[res_id] for res_id in predicted_order_parameters.keys() if res_id in gt_dict]
        filtered_order_params = [predicted_order_parameters[res_id] for res_id in predicted_order_parameters.keys() if res_id in gt_dict]
        order_params_list = [gt_dict[res_id] for res_id in predicted_order_parameters.keys() if res_id in gt_dict]
        corr = np.corrcoef(np.array(order_params_list), np.array(filtered_order_params))[0, 1]
        error = np.nanmean((np.array(diff))**2)
        
        data_error = np.nan
        if "amide" in s2_type:
            data_error = gt[gt["res_id"].astype(str).isin(gt_dict.keys())]["s2_error"].tolist()
        if s2_type == "methyl_rdc":
            data_error = gt["mod_error"].astype(float).tolist()
        if s2_type == "amide_relax":
            salmon_corr, salmon_error, salmon_dict = self.compute_salmon(predicted_order_parameters)
            return corr,salmon_corr,error, salmon_error, gt_dict, salmon_dict, data_error, gt_res_id
        
        return corr,np.nan,error, np.nan, gt_dict, np.nan, data_error, gt_res_id
    
    
    
    def run(self, atom_arrays, structures, s2_types_dict):
        s2_results = {t: None for t in s2_types_dict.keys()}
        for s2_type in s2_types_dict.keys():
            s2_order_params, residue_vectors = self.calculate_s2_loss(structures, atom_arrays, s2_type, s2_types_dict[s2_type])
            s2_corr,s2_salmon_corr,s2_error, s2_salmon_error, gt_dict, salmon_dict, data_error, gt_res_id = self.compute_s2_correlation(s2_order_params, s2_type, s2_types_dict[s2_type])
            s2_results[s2_type] = {"order_params": s2_order_params, "residue_vectors": residue_vectors, "corr": s2_corr, "salmon_corr": s2_salmon_corr, "error":s2_error, "salmon_error":s2_salmon_error, "gt": gt_dict, "salmon": salmon_dict, "data_error":data_error, "gt_res_id":gt_res_id}
            if s2_type == "amide_relax":
                self.save_s2_graphs("test2", s2_order_params, s2_corr, s2_type, s2_types_dict[s2_type])
        return {"ensemble_s2_results": json.dumps(s2_results)}
        
     
     
    def save_s2_graphs(self, name, pred_order_params, s2_corr, s2_type, path):
        gt = self.get_gt_s2(path)
        if "methyl" in s2_type:
            gt_res_id = [f"{res_id},{a1},{a2}" for res_id,a1,a2 in zip(gt["res_id"].values, gt["atom1"].values, gt["atom2"].values)]
        else:
            gt_res_id = [str(r) for r in gt["res_id"].values]
        gt_s2 = np.array(gt["s2"])
        # Plot the predicted and estimated s2
        fig, ax = plt.subplots(1, 1, figsize=(15, 5))
        keys = pred_order_params.keys()
        pred_order_params = [v for k,v in zip(keys,list(pred_order_params.values())) if k in gt_res_id]
        keys = [k for k in keys if k in gt_res_id]
        original_keys = keys.copy()
        keys = [k.replace(",", "\n") for k in keys]
        ax.plot(keys, pred_order_params, label="predicted", marker="o")
        guid_s2_values = [v for k,v in zip(gt_res_id, gt_s2) if k in original_keys]
        ax.scatter(keys, guid_s2_values, label="estimated", marker="x", color="red")
        ax.legend()
        ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
        ax.set_xlabel("Residue ID")
        ax.set_ylabel("S2")
        # ax.set_xlim(0, 75)
        ax.set_ylim(0, 1)
        ax.set_title("S2 values. Error: {:.2f}".format(s2_corr.item()))
        plt.tight_layout()
        # Log the image to wandb
        path = ""#"nature_sweep/s2_figures"
        # os.makedirs(path, exist_ok=True)
        # plt.savefig(f"{path}/{name}.png")
        plt.savefig(f"{name}.png")
        plt.close(fig)
