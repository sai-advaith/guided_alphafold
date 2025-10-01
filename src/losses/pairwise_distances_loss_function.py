from .abstract_loss_funciton import AbstractLossFunction
import gemmi
import torch

def get_distance_matrix_mask(pdb_file_path, atom_type="N"):
    structure = gemmi.read_structure(pdb_file_path)
    model = structure[0]
    chain = model[0]
    atom_positions = []
    atom_mask = []
    for i,residue in enumerate(chain):
        for atom in residue:
            atom_positions.append((atom.pos.x, atom.pos.y, atom.pos.z))
            atom_mask.append(atom.element.name == atom_type)
    atom_positions = torch.tensor(atom_positions, dtype=torch.float32)
    atom_mask = torch.tensor(atom_mask, dtype=torch.bool)
    distance_matrix = torch.cdist(atom_positions, atom_positions, p=2)
    distance_matrix[atom_mask] = 0
    distance_matrix[:, atom_mask] = 0
    return distance_matrix
            
class PairwiseDistancesLossFunction(AbstractLossFunction):
    def __init__(self, distance_matrix, rmax=100000):
        self.distance_matrix = distance_matrix
        self.distance_matrix[self.distance_matrix >= rmax] = 0
        # symetric matrix, we can remove the bottom half triangle
        self.distance_matrix = torch.triu(self.distance_matrix)
        self.indexes = self.distance_matrix.nonzero()
        print(f"comparing {self.indexes.shape[0]} atom locations")

        self.distances = self.distance_matrix[self.indexes.unbind(dim=-1)]
        self.last_loss = None
    
    def wandb_log(self, x_0_hat):
        return ({"pairwise distance loss": self.last_loss})

    def __call__(self, x_0_hat, time):
        # [batch_size, pair_index, first_atom_in_pair_or_second, xyz]
        atoms_to_compare = x_0_hat[:,self.indexes]
        # subtract atoms in the same pair to calculate pair distance
        model_distances = (atoms_to_compare[..., 0,:] - atoms_to_compare[..., 1,:]).norm(dim=-1)
        # get the mean distances in the ensamble
        model_distances = model_distances.mean(dim=0)
        loss = torch.nn.functional.mse_loss(model_distances, self.distances)
        self.last_loss = loss
        return loss, None