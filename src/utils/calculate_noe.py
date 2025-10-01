

import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


def adjust_nmr_data_atom_array(nmr_data, atom_array):
    # in the case of colabfold_relax some of the hydrogens disappear, remove their restraints
    atom_pairs = set(zip(atom_array[0].res_id, atom_array[0].atom_name))
    mask = nmr_data.apply(lambda row: 
                          ((row["residue1_num"], row["atom1"]) in atom_pairs and 
                          (row["residue2_num"], row["atom2"]) in atom_pairs) or
                          ('M' in row["atom1"] or 'M' in row["atom2"]), axis=1)
    # assuming none of the methyl hydrogens get deleted,  might not be true but if it is this is the easiest solutions, if it's not the script will fail
    
    return nmr_data[mask].reset_index(drop=True)


def load_distance_constraints(nmr_data) -> tuple[torch.Tensor, torch.Tensor]:
    nmr_data["lower_bound"] = nmr_data["lower_bound"].apply(lambda x: 0 if x == '.' else float(x))
    lower_bound = torch.tensor(nmr_data["lower_bound"].values, dtype=torch.float32)
    upper_bound = torch.tensor(nmr_data["upper_bound"].values, dtype=torch.float32)
    return upper_bound, lower_bound

def methyl_group_names(nmr_name, residue_name):
    if residue_name == "LEU" and nmr_name == "MD1":
        return ["HD11", "HD12", "HD13"]
    if residue_name == "LEU" and nmr_name == "MD2":
        return ["HD21", "HD22", "HD23"]
    if residue_name == "ALA" and nmr_name == "MB":
        return ["HB1","HB2","HB3"]
    if residue_name == "MET" and nmr_name == "ME":
        return ["HE1", "HE2", "HE3"]
    if residue_name == "ILE" and nmr_name == "MG":
        return ["HG21","HG22","HG23"]
    if residue_name == "ILE" and nmr_name == "MD":
        return ["HD11","HD12","HD13"]
    if residue_name == "VAL" and nmr_name == "MG1":
        return ["HG11","HG12","HG13"]
    if residue_name == "VAL" and nmr_name == "MG2":
        return ["HG21","HG22","HG23"]
    if residue_name == "THR" and nmr_name == "MG":
        return ["HG21","HG22","HG23"]
    raise ValueError("Unknown methyl hydrogen to pdb conversion")

def q_group_names(nmr_name, residue_name):
    if residue_name == "TYR" and nmr_name == "QD":
        return ["HD1"]
    if residue_name == "TYR" and nmr_name == "QE":
        return ["HE1"]
    if residue_name == "PHE" and nmr_name == "QD":
        return ["HD1","HD2"]
    if residue_name == "PHE" and nmr_name == "QE":
        return ["HE1", "HE2"]
    if residue_name == "TRP" and nmr_name == "QD":
        return ["HD1"]
    if residue_name == "TRP" and nmr_name == "QE":
        return ["HE1"]
    if residue_name == "TRP" and nmr_name == "QZ":
        return ["HZ2"]
    if residue_name == "LYS" and nmr_name == "QZ":
        return ["HZ1","HZ2","HZ3"]
    raise ValueError("Unknown q hydrogen to pdb conversion")

class CalculateNOE:
    def __init__(self, restraint_file, atom_arrays, device):
        nmr_data = pd.read_csv(f"pipeline_inputs/nmr_restraints/{restraint_file}")
        nmr_data = nmr_data[nmr_data["type"] == "NOE"]
        nmr_data = adjust_nmr_data_atom_array(nmr_data, atom_arrays)
        self.nmr_data = nmr_data
        
        self.atom_arrays = atom_arrays
        
        upper_bound, lower_bound = load_distance_constraints(self.nmr_data)
        self.upper_bound, self.lower_bound = upper_bound.to(device), lower_bound.to(device) 
        or_cond = torch.tensor(self.nmr_data["constrain_id"], dtype=torch.float32, device=device)
        self.unique_or, self.inverse_or_indices = torch.unique(or_cond, return_inverse=True)
        
        
        
    def _get_atom_coord(self, row, tag_num, structures):
        atom_name = row[f"atom{tag_num}"]
        residue_name = row[f"residue{tag_num}_id"]
        residue_id = row[f"residue{tag_num}_num"]

        if "M" in atom_name:  # Special case for methyl groups
            return self._compute_group_coord(atom_name, residue_name, residue_id, structures, "M")
        if "Q" in atom_name:
            return self._compute_group_coord(atom_name, residue_name, residue_id, structures, "Q")
        try:
            idx = np.where((self.atom_arrays[0].atom_name == atom_name) & (self.atom_arrays[0].res_id == residue_id))[0][0]
        except IndexError:
            print(f"missing atom {residue_id, atom_name}")
            return None
        return structures[:, idx]
    
    

    def _compute_group_coord(self, atom_name, residue_name, residue_id, structures, group_type):
        if group_type=="M":
            names = methyl_group_names(atom_name, residue_name)
        if group_type=="Q":
            names = q_group_names(atom_name, residue_name)
        indices = [np.where((self.atom_arrays[0].atom_name == name) & (self.atom_arrays[0].res_id == residue_id))[0][0] for name in names]
        return structures[:, indices].mean(dim=1)

        
    def get_comparison_atoms(self, structures):
        atoms_to_compare_1, atoms_to_compare_2  = [], []
        mask = []
        for i, row in tqdm(self.nmr_data.iterrows(), total=len(self.nmr_data), desc="generating comparison atoms"):
            idx1 = self._get_atom_coord(row, 1, structures)
            idx2 = self._get_atom_coord(row, 2, structures)
            if idx1 is None or idx2 is None:
                mask.append(False)
                continue
            atoms_to_compare_1.append(idx1)
            atoms_to_compare_2.append(idx2)
            mask.append(True)
            # else:
            #     raise ValueError(f"Atom match not found for residues: {row['residue1_num']} - {row['residue2_num']} with atoms: {row['atom1']} - {row['atom2']}")
            #     # print(f"Atom match not found for residues: {row['residue1_num']} - {row['residue2_num']} with atoms: {row[tag1]} - {row[tag2]}")
        atoms_to_compare_1 = torch.stack(atoms_to_compare_1).permute(1,0,2)
        atoms_to_compare_2 = torch.stack(atoms_to_compare_2).permute(1,0,2)
        return atoms_to_compare_1, atoms_to_compare_2, mask
        
        
    def get_total_constraints(self):
        return len(self.unique_or)
    
    def integrate_or_conditions(self, curr_loss_lb, curr_loss_ub, hyd_mask):
        min_values = torch.zeros((len(self.unique_or)),dtype=curr_loss_lb.dtype, device=curr_loss_lb.device)
        min_values = torch.scatter_reduce(min_values, 0, self.inverse_or_indices[...,hyd_mask], curr_loss_lb+curr_loss_ub, reduce="amin", include_self=False)
        argmin_indices = torch.full(min_values.shape, -1, dtype=torch.long, device=self.inverse_or_indices.device)
        # For each element in the sequence
        for i in range(self.inverse_or_indices[hyd_mask].shape[0]):
            cluster_idx = self.inverse_or_indices[i]
            # If this source value is smaller than what's currently at the target location
            if curr_loss_lb[i]+curr_loss_ub[i] == min_values[cluster_idx] and argmin_indices[cluster_idx] == -1:
                argmin_indices[cluster_idx] = i

        min_values_lb = curr_loss_lb[argmin_indices]
        min_values_ub = curr_loss_ub[argmin_indices]
        return min_values_ub, min_values_lb, argmin_indices
    
    
    def get_losses(self, atoms_to_compare_1, atoms_to_compare_2, hyd_mask):
        model_distances = (atoms_to_compare_1 - atoms_to_compare_2).norm(dim=-1)
        model_distances = model_distances.mean(dim=0)
        loss_ub = torch.relu(model_distances - self.upper_bound[...,hyd_mask])
        loss_lb = torch.relu(self.lower_bound[...,hyd_mask] - model_distances)
        # take or conditions into account
        loss_ub, loss_lb, argmin_loss = self.integrate_or_conditions(loss_lb, loss_ub, hyd_mask) 
        
        return loss_ub, loss_lb, argmin_loss
    
    
    def calculate_violations(self, atoms_to_compare_1, atoms_to_compare_2 ,hyd_mask):
        loss_ub_val, loss_lb_val, argmin_loss = self.get_losses(atoms_to_compare_1, atoms_to_compare_2, hyd_mask)
        loss_ub_val, loss_lb_val = loss_ub_val.cpu().numpy(), loss_lb_val.cpu().numpy()
        loss_ub = loss_ub_val > 0
        loss_lb = loss_lb_val > 0
        ub_violations = loss_ub.sum()
        lb_violations = loss_lb.sum()

        return ub_violations + lb_violations, loss_ub, loss_lb, loss_ub_val, loss_lb_val, argmin_loss
    
    def calculate_ensembles_additive(self, atoms_to_compare_1, atoms_to_compare_2, hyd_mask, verbose=False):
        num_structures = atoms_to_compare_1.shape[0]
        remaining_indices = set(range(num_structures))
        best_indices = []
        best_score = float('inf')
        
        for i in range(num_structures):
            initial_score= self.calculate_violations(
                atoms_to_compare_1[i][None],
                atoms_to_compare_2[i][None],
                hyd_mask,
            )[0]
            if initial_score < best_score:
                best_score = initial_score
                best_indices = [i]

        remaining_indices.remove(best_indices[0])

        # Iteratively add structures to the ensemble
        while len(best_indices) < 5:
            best_improvement = float('inf')
            best_candidate = None

            for idx in remaining_indices:
                current_score = self.calculate_violations(
                    atoms_to_compare_1[best_indices + [idx]],
                    atoms_to_compare_2[best_indices + [idx]],
                    hyd_mask,
                )[0]

                improvement = current_score - best_score

                if improvement < best_improvement:
                    best_improvement = improvement
                    best_candidate = idx

            if best_candidate is not None and best_improvement <= 0:
                best_indices.append(best_candidate)
                best_score += best_improvement
                remaining_indices.remove(best_candidate)
            else:
                break

        if verbose:
            print(f"Additive: Structure indices: {best_indices}, Score: {best_score}")

        return best_indices, *self.calculate_violations(atoms_to_compare_1[best_indices],
                    atoms_to_compare_2[best_indices],
                    hyd_mask,)
        
    
    def run(self, structures):        
        atoms_to_compare_1, atoms_to_compare_2, hyd_mask = self.get_comparison_atoms(structures) 
        ensemble_score, ensemble_ub, ensemble_lb, ensemble_ub_soft, ensemble_lb_soft, ensemble_or_atom_indices = self.calculate_violations(atoms_to_compare_1, atoms_to_compare_2, hyd_mask)
        additive_indexes, omp_score_additive, omp_additive_ub, omp_additive_lb, omp_additive_ub_soft, omp_additive_lb_soft, additive_or_atom_indices = self.calculate_ensembles_additive(atoms_to_compare_1, atoms_to_compare_2, hyd_mask)

        return {
            "total_constraints": self.get_total_constraints(),
            
            "ensemble_score": ensemble_score,
            "omp_score_additive": omp_score_additive,
            "omp_score_additive_indexes": additive_indexes,
            
            "ensemble_ub_values": json.dumps(ensemble_ub_soft.tolist()),
            "ensemble_lb_values": json.dumps(ensemble_lb_soft.tolist()),
            "omp_additive_ub_values": json.dumps(omp_additive_ub_soft.tolist()),
            "omp_additive_lb_values": json.dumps(omp_additive_lb_soft.tolist()),
            
            "omp_additive_ub": json.dumps(omp_additive_ub.tolist()),
            "omp_additive_lb": json.dumps(omp_additive_lb.tolist()),
            
            "ensemble_ub": json.dumps(ensemble_ub.tolist()),
            "ensemble_lb": json.dumps(ensemble_lb.tolist()),
            
            "omp_additive_comp_indices": json.dumps(additive_or_atom_indices.tolist()),
            "ensemble_or_atom_indices": json.dumps(ensemble_or_atom_indices.tolist()),
            
            "constraint_mask": json.dumps(hyd_mask),
        }
