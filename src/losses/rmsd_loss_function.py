from .abstract_loss_funciton import AbstractLossFunction
import torch
import gemmi
from ..protenix.metrics.rmsd import self_aligned_rmsd
import numpy as np

def load_pdb_atom_locations(pdb_file, ignored_residues=[]):
    # TODO: fix this to get the correct order
    structure = gemmi.read_structure(pdb_file)
    model = structure[0]
    chain = model[0]
    atom_positions = []
    mask = []
    for i,residue in enumerate(chain):
        for atom in residue:
            atom_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
            if i in ignored_residues:
                mask.append(0)
            else:
                mask.append(1)
    atom_positions_array = np.array(atom_positions)
    atom_positions_tensor = torch.tensor(atom_positions_array, dtype=torch.float32)
    mask = torch.tensor(mask, dtype=torch.bool)
    return atom_positions_tensor, mask

class MultiRMSDLossFunction(AbstractLossFunction):
    def __init__(self, reference_files, top_k=10,mean_loss_weight=4, distance_loss_weight=1, device="cpu"):
        self.reference_atom_positions = [load_pdb_atom_locations(reference_file)[0].to(device)[None] for reference_file in reference_files]
        self.reference_atom_positions = torch.stack(self.reference_atom_positions, dim=0)
        self.top_k = top_k
        self.mean_loss_weight = mean_loss_weight
        self.distance_loss_weight = distance_loss_weight
        self.last_loss_value = None
    
    def __call__(self, x_0_hat, _):
        repeats = [x_0_hat.shape[0], 1, 1]
        rmsd_mask = torch.ones_like(x_0_hat[...,0])
        _, aligned_structure, _, _ = self_aligned_rmsd(x_0_hat, self.reference_atom_positions[0].repeat(*repeats), rmsd_mask)
        distances = (aligned_structure[None] - self.reference_atom_positions).norm(dim=-1).topk(self.top_k, dim=-1)[0].mean(dim=-1)
        distances_loss = distances.min(dim=-1)[0].sum()
        mean_loss = (aligned_structure.mean(dim=0) - self.reference_atom_positions.flatten(0,1).mean(dim=0)).norm(dim=-1).topk(self.top_k, dim=-1)[0].mean()
        loss_value = distances_loss * self.distance_loss_weight + mean_loss * self.mean_loss_weight
        self.last_loss_value = loss_value.detach()
        return loss_value, None
    
    def wandb_log(self, x_0_hat):
        return ({"rmsd loss": self.last_loss_value})