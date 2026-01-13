from .abstract_loss_funciton import AbstractLossFunction
import torch
from biotite.structure import connect_via_residue_names
import numpy as np
import gemmi

class BondLengthLossFunction(AbstractLossFunction):
    def __init__(self, atom_array_object, device):
        self._device = device
        self._atom_array_object = atom_array_object
        self._bonds, self._bond_lengths = self._initialize_topolgy_bonds()
        self._collision_distances = self._initialize_collision_distances()

        self._last_bond_length_loss = None
        self._last_collision_loss = None

    def _initialize_topolgy_bonds(self):
        # bonds and bond types, target_atom[i] = indexes of atoms bonded to atom i, padded with -1
        target_atom, _ = connect_via_residue_names(self._atom_array_object).get_all_bonds()
        bonds = set()
        for i in range(target_atom.shape[0]):
            bonded_atoms = target_atom[i]
            for index in bonded_atoms:
                if index != -1:
                    bonds.add(tuple(sorted((i, index))))
        bond_lengths = []
        bonded_atoms = np.array(list(bonds))
        for atom_1_index, atom_2_index in bonded_atoms:
            atom_1_element = gemmi.Element(self._atom_array_object[atom_1_index].element)
            atom_2_element = gemmi.Element(self._atom_array_object[atom_2_index].element)
            bond_length = atom_1_element.covalent_r + atom_2_element.covalent_r
            bond_lengths.append(bond_length)
        bonded_atoms = torch.tensor(bonded_atoms, device=self._device)
        bond_lengths = torch.tensor(bond_lengths, device=self._device, dtype=torch.float32)
        return bonded_atoms, bond_lengths
    
    def _initialize_collision_distances(self):
       atom_covalent_r = np.array([gemmi.Element(self._atom_array_object[index].element).covalent_r for index in range(len(self._atom_array_object))])
       atom_covalent_r = torch.tensor(atom_covalent_r, device=self._device)
       return (atom_covalent_r[None] + atom_covalent_r[:, None])

    # Using van der waals radii for collision detection instead of the previou approach with covalent radii (covalent is too permissive. atom collision is not defined as nuclear collision!)
    # def _initialize_collision_distances(self):
    #     # Use 0.5 for Hydrogen (allows H-bonds), otherwise standard vdW radius
    #     # 0.5 (H) + 1.52 (O) - 0.4 (pad) = 1.62 Threshold. Safe for 1.8A bonds.
    #     atom_vdw_r = torch.tensor([
    #         0.5 if self._atom_array_object[i].element.upper() == "H" else gemmi.Element(self._atom_array_object[i].element).vdw_r 
    #         for i in range(len(self._atom_array_object))
    #     ], device=self._device)
    
    #     return (atom_vdw_r[None] + atom_vdw_r[:, None])
    
    def bond_length_loss(self, atom_locations, threshold=0.2 , l=2):
        bond_lengths = (atom_locations[:, self._bonds[...,0]] - atom_locations[:, self._bonds[...,1]]).norm(dim=-1)
        bond_lengths_diff = ((bond_lengths - self._bond_lengths).abs() - threshold).relu()
        return bond_lengths_diff.pow(l).sum() / atom_locations.shape[0]
    
    def collision_loss(self, atom_locations, padding = 0.4):
    #def collision_loss(self, atom_locations, padding = -0.4):
        distances = (atom_locations[:,:, None] - atom_locations[:,None]).norm(dim=-1)
        mask = torch.zeros_like(distances)
        mask[:, self._bonds[...,0], self._bonds[...,1]] = 100.0
        mask[:, self._bonds[...,1], self._bonds[...,0]] = 100.0
        mask[:, self._bonds[...,0], self._bonds[...,0]] = 100.0
        mask[:, self._bonds[...,1], self._bonds[...,1]] = 100.0
        distances = distances + mask
        #threshold_loss = (self._collision_distances + padding - distances).max(dim=0)[0].relu()
        # redone to take all the collisions and not just the worst of a pair per structure in the ensemble
        self.last_thresholds = (self._collision_distances + padding - distances).relu()
        threshold_loss = self.last_thresholds.sum(dim=(1,2)) 
        return threshold_loss.sum() / atom_locations.shape[0] # average collision loss per structure
    
    def get_bond_loss(self, atom_locations):
        bond_length_loss = self.bond_length_loss(atom_locations)
        collision_loss = self.collision_loss(atom_locations)
        loss_values = bond_length_loss + collision_loss
        self._last_bond_length_loss = bond_length_loss.item()
        self._last_collision_loss = collision_loss.item()
        return loss_values 
    
    def __call__(
        self, x_0_hat, time, structures=None, i=None, step=None, 
        contains_padded_atoms = False, padded_atoms_elements = None, # For now, leave unchanged..!
    ):
        return self.get_bond_loss(x_0_hat), None
    
    def wandb_log(self, x_0_hat):
        return {
            "bond length loss": self._last_bond_length_loss, 
            "collision loss": self._last_collision_loss, # average collision loss per structure
            "normalized collision loss": self._last_collision_loss / x_0_hat.shape[1], # normalizing not only per number of structures but also what's the average collision per atom pair
            "max collision loss of an atom pair": self.last_thresholds.max().item(), # max collision loss of an atom pair
            "total bond loss": self._last_bond_length_loss + self._last_collision_loss
        }