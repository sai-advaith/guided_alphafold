
import json
import numpy as np
from biotite.structure.io.pdb import PDBFile
from ..protenix.metrics.rmsd import self_aligned_rmsd
import torch.nn.functional as F

import pandas as pd
import torch
import os
import gemmi
from biotite.structure import connect_via_residue_names

EPSILON = 1e-5


class CalculateEPR:
    def __init__(self, atom_arrays, constraints_file="pipeline_inputs/epr/deer_distances.csv", rotamers_statistics_file="pipeline_inputs/epr/rotamer1_R1A_298K_2015.csv", rotamers_structures_folder="pipeline_inputs/epr/rotamer1_R1A_298K_2015", device="cuda", batch_size=16, sample_points=1000):
        constraints_df = pd.read_csv(constraints_file)
        self.constraints = {(k1,k2):torch.tensor(v, dtype=torch.float32, device=device) for k1,k2,v in zip(constraints_df['res1'], constraints_df['res2'], constraints_df['exp_dist'])}
        self.guidance_values = torch.stack(list(self.constraints.values()))
        rotamers_statistics_df = pd.read_csv(rotamers_statistics_file)
        self.rotamers_statistics = torch.from_numpy(rotamers_statistics_df["prob"].values).to(device).to(torch.float32)
        self.sample_points = sample_points
        self.device = device
        
        std = torch.tensor(constraints_df["sigma"].values, device=device, dtype=torch.float32)
        self.guidance_weights = 1 / (std**2 + EPSILON)
        self.guidance_weights = 1.0 + (self.guidance_weights - self.guidance_weights.min()) * (3.0 - 1.0) / (self.guidance_weights.max() - self.guidance_weights.min())
       
        
        self.constraint_res_ids = np.unique(constraints_df[['res1', 'res2']].values)
        self.atom_array = atom_arrays[0]
        
        self.rotamer_atom_arrays = []
        for f in os.listdir(rotamers_structures_folder):
            pdb_file = PDBFile.read(os.path.join(rotamers_structures_folder, f))
            self.rotamer_atom_arrays.append(pdb_file.get_structure()[0])
            
        # remove hydrogens
        hyd_mask = self.rotamer_atom_arrays[0].element == 'H'    
        self.rotamer_atom_arrays = [arr[~hyd_mask] for arr in self.rotamer_atom_arrays]
            
        self.atom_names = ['N', 'CA', 'C', 'O', 'CB']
        self.res_indices = {res:np.where((self.atom_array.res_id == res))[0] for res in self.constraint_res_ids}
        self.rotamers_structures = torch.from_numpy(np.stack([rot.coord for rot in self.rotamer_atom_arrays])).to(device)
        self.rotamer_indices_by_atom_name = self.ordered_index_atom_names(1, self.rotamer_atom_arrays[0])
        self.rotamers_coord_by_atom_name = torch.stack([torch.from_numpy(rot.coord[self.rotamer_indices_by_atom_name]) 
                                              for rot in self.rotamer_atom_arrays], dim=0).to(device)
        
        self.atom_indices_by_res = {res: self.ordered_index_atom_names(res, self.atom_array) for res in self.constraint_res_ids}
        self.exp_rotamers_statistics = self.rotamers_statistics.unsqueeze(0).expand(batch_size, -1)
        self.atom_array_with_rotamers = self.replace_and_add_atoms()
        self.collision_distances = {res:self._initialize_collision_distances(self.atom_array_with_rotamers[res]) for res in self.constraint_res_ids}
        
        self.o1_mask = {res:self.atom_array_with_rotamers[res].atom_name=="O1" for res in self.constraint_res_ids}

    
    def _initialize_collision_distances(self, atom_array):
        atom_covalent_r = np.array([gemmi.Element(atom_array[index].element).covalent_r for index in range(len(atom_array))])
        atom_covalent_r = torch.tensor(atom_covalent_r, device=self.device)
        return (atom_covalent_r[None] + atom_covalent_r[:, None])
    
    def _initilize_topolgy_bonds(self, atom_array):
        # bonds and bond types, target_atom[i] = indexes of atoms bonded to atom i, padded with -1
        target_atom, _ = atom_array.bonds.get_all_bonds()
        bonds = set()
        for i in range(target_atom.shape[0]):
            bonded_atoms = target_atom[i]
            for index in bonded_atoms:
                if index != -1:
                    bonds.add(tuple(sorted((i, index))))
        bonded_atoms = np.array(list(bonds))
        bonded_atoms = torch.tensor(bonded_atoms, device=self.device)
        return bonded_atoms
   
    def ordered_index_atom_names(self, res_id, array):
        mask = (array.res_id == res_id)
        replace_indices = []
        used_names = []
        for name in self.atom_names:
            idx = np.where(mask & (array.atom_name == name))[0]
            if len(idx) > 0:
                replace_indices.append(idx[0])
                used_names.append(name)
        return torch.tensor(replace_indices)
    
    def replace_and_add_atoms(self):
        updated_atom_arrays = {}
        for res in self.constraint_res_ids:
            new_rotamer_atoms = self.rotamer_atom_arrays[0]
            new_rotamer_atoms.set_annotation("res_id", np.full(len(new_rotamer_atoms),res))
            new_rotamer_atoms.set_annotation("chain_id", np.full(len(new_rotamer_atoms),self.atom_array.chain_id[0]))
            new_rotamer_atoms.set_annotation("res_name", np.full(len(new_rotamer_atoms),"CYS"))
            new_rotamer_atoms.set_annotation("is_protein", np.full(len(new_rotamer_atoms),1))
            new_rotamer_atoms.set_annotation("hetero", np.full(len(new_rotamer_atoms),0))
            merged_atom_array = self.atom_array[:self.res_indices[res][0]] + new_rotamer_atoms + self.atom_array[self.res_indices[res][-1]+1:]
            updated_atom_arrays[res] = merged_atom_array
        return updated_atom_arrays
    
    def update_atoms_locations(self, res, x_0_hat, aligned_rotamers):   
        x_exp = x_0_hat[:, None, :, :].expand(-1, aligned_rotamers.shape[1], -1, -1).clone()
        updated_x_0_hat = torch.cat([x_exp[:,:,:self.res_indices[res][0]], aligned_rotamers, x_exp[:,:,self.res_indices[res][-1]+1:]], dim=2)
        return updated_x_0_hat

    
    def get_non_collision_mask(self, res, atom_locations, bonded_atoms, atom_array, padding = 0.4):        
        res_mask = torch.from_numpy((atom_array.res_id == res)).to(self.device)  # shape (A,)
        res_idx = torch.nonzero(res_mask, as_tuple=False).squeeze(-1)  # shape (M,)
        other_idx = torch.nonzero(~res_mask, as_tuple=False).squeeze(-1)  # shape (A-M,)
        
        struct = atom_locations.flatten(0,1)
        # for struct in atom_locations:
        coords_res = struct[:, res_idx] 
        coords_other = struct[:, other_idx]
        distances = (coords_res[:, :, None] - coords_other[:, None]).norm(dim=-1)  # (N, M, A-M)
        # Prepare allowed distances
        cd_sub = self.collision_distances[res][res_idx[:, None], other_idx] # (M, A-M)
        # Initialize bond mask: shape (M, A-M), filled with 0 (check) or 3 (skip)
        bond_mask = torch.zeros((len(res_idx), len(other_idx)), device=atom_locations.device)
        # Add 3 to bonded pairs involving residue atoms
        bonded_set = set(map(tuple, bonded_atoms.cpu().numpy()))
        for i_res_local, i_res_global in enumerate(res_idx):
            for j_other_local, j_other_global in enumerate(other_idx):
                if (i_res_global.item(), j_other_global.item()) in bonded_set or \
                (j_other_global.item(), i_res_global.item()) in bonded_set:
                    bond_mask[i_res_local, j_other_local] = 3.0

        # Add bond mask to distances
        distances = distances + bond_mask[None]  # (N, M, A-M)

        # Compute collision margin
        margin = cd_sub[None] + padding - distances  # (N, M, A-M)
        collision_mask = (margin > 0).any(dim=(1, 2)) # positive = collision  
        print(f"Collision mask {res}: {collision_mask.sum()} collisions detected")
        return ~collision_mask.view(atom_locations.shape[0], atom_locations.shape[1]) 
    
        
    def run(self, structures):
        x_0_hat = structures
        # align first
        alignemnt_x_0_hat = [x_0_hat[:,self.atom_indices_by_res[res]] for res in self.constraint_res_ids]
        alignemnt_x_0_hat = torch.stack(alignemnt_x_0_hat, dim=1)
        _, _, rot, trans = self_aligned_rmsd(self.rotamers_coord_by_atom_name[None,None], alignemnt_x_0_hat[:,:,None], torch.ones(self.rotamers_coord_by_atom_name.shape[-2], dtype=torch.bool).to(x_0_hat.device))
           
        update_x_0_rotamers = {}
        updated_probs = {}
        o1_locations = {}
        collision_masks = {}
        for i,res in enumerate(self.constraint_res_ids):
            aligned_rotamers = torch.matmul(self.rotamers_structures[None], rot[:,i].transpose(-1, -2)) + trans[:,i]
            updated_x_0_hat = self.update_atoms_locations(res, x_0_hat, aligned_rotamers)
            self.atom_array_with_rotamers[res].coord = updated_x_0_hat[0][0].detach().cpu().numpy()
            self.atom_array_with_rotamers[res].bonds = connect_via_residue_names(self.atom_array_with_rotamers[res])
            bonded_atoms = self._initilize_topolgy_bonds(self.atom_array_with_rotamers[res])
            with torch.no_grad():
                mask = self.get_non_collision_mask(res, updated_x_0_hat, bonded_atoms, self.atom_array_with_rotamers[res])
                # mask = torch.ones_like(mask, dtype=torch.bool)
            collision_masks[res] = mask
            updated_probs[res] = torch.where(mask, self.exp_rotamers_statistics, torch.zeros_like(self.exp_rotamers_statistics))
            # re-normalize statistics
            row_sums = updated_probs[res].sum(dim=1, keepdim=True)
            nonzero_mask = row_sums != 0
            updated_probs[res] = torch.where(nonzero_mask, updated_probs[res] / row_sums, updated_probs[res])
            update_x_0_rotamers[res] = updated_x_0_hat
            o1_locations[res] = updated_x_0_hat[:,:,self.o1_mask[res]]
            # self.save_tensor_to_pdb(updated_x_0_hat[0][0], self.atom_array_with_rotamers[res], "test.pdb")
        
                        
        distances = []
        distances_dict = {}
        probs_dict = {}
        guidance_mask = torch.zeros(self.guidance_values.shape[0], dtype=torch.bool, device=self.device)
        for i, key in enumerate(self.constraints.keys()):
            # both have values
            key_dist = []
            probs = []
            if updated_probs[key[0]].sum() != 0 and updated_probs[key[1]].sum() != 0:
                key0_probs = updated_probs[key[0]]
                key1_probs = updated_probs[key[1]]
                # the keys are for the same structure
                for j in range(len(key0_probs)):
                    if key0_probs[j].sum() != 0 and key1_probs[j].sum() != 0:
                        prob0 = key0_probs[j]
                        prob1 = key1_probs[j]
                        indices0 = torch.multinomial(prob0, self.sample_points, replacement=True)
                        indices1 = torch.multinomial(prob1, self.sample_points, replacement=True)
                        loc0 = o1_locations[key[0]][j][indices0]
                        loc1 = o1_locations[key[1]][j][indices1]
                        dist = (loc0 - loc1).norm(dim=-1)
                        key_dist.append(dist)
                        probs.append(prob0*prob1)
            if len(key_dist) > 0:
                distances_dict[key] = torch.stack(key_dist, dim=0)
                probs_dict[key] = torch.stack(probs, dim=0)
                distances.append(torch.stack(key_dist, dim=0).mean()[None])
                guidance_mask[i] = True
                
                
        distances = torch.cat(distances, dim=0) if len(distances)!=0 else []
        loss = x_0_hat.sum()*0.0
        if len(distances)!=0:
            loss=(((distances-self.guidance_values[guidance_mask]).abs())*self.guidance_weights[guidance_mask]).mean()

        return {"epr_loss": loss.item(),
                "distances": json.dumps({str(k):v.detach().cpu().tolist() for k,v in distances_dict.items()}),
                "distances_probs": json.dumps({str(k):v.detach().cpu().tolist() for k,v in probs_dict.items()}),
                "o1_probs": json.dumps({int(k): v.detach().cpu().tolist() for k, v in updated_probs.items()}),
                "o1_locations": json.dumps({int(k): v.detach().cpu().tolist() for k, v in o1_locations.items()})
                }