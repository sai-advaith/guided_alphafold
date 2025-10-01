

from .abstract_loss_funciton import AbstractLossFunction
import wandb
import torch
import pandas as pd
import numpy as np
from ..utils.hydrogen_addition import FragmentLibrary, AtomNameLibrary, get_hydrogen_names
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer
import torch.nn.functional as F


def specden(tau, w, T):
    """
        Spectral density function. 

        Inputs:
            tau - time in seconds
            w - frequency in rad/s
            T - temperature in K

        Outputs:
            spectral density function with a single timescale t, evaluated at frequency w.
    """

    R = 8.3145  
    taueff = tau * torch.exp((40000 / R) * (300 - T) / (300 * T))
    return (2 / 5.) * taueff / (1 + (taueff * w) ** 2)

def calc_detector_from_output_file(fn, tau):
    data = torch.from_numpy(np.loadtxt(fn)).float()
    
    c = data[:, 0]  # shape [N]
    omega = data[:, 1].unsqueeze(1)  # shape [N, 1]
    temp = data[:, 2].unsqueeze(1)   # shape [N, 1]

    tau = tau[None, :]  # make it shape [1, N_tau]

    spec = specden(tau, omega, temp)  # shape [N, N_tau]
    sens = torch.matmul(c, spec)      # shape [N_tau]
    
    return sens

   
class RelaxTimesLossFunction(AbstractLossFunction):
    def __init__(self, atom_array, coefficient_files, data_file, device, num_discretize_points = 50, batch_size=5):

        self.time_groups = len(coefficient_files)
        self.tau_calc = torch.logspace(start=-13, end=-3, steps=num_discretize_points, base=10)

        self.atom_array = atom_array
        self.last_loss = None
        self.device = device
        
        self.fragment_library = FragmentLibrary.standard_library()
        self.name_library = AtomNameLibrary.standard_library()
        
        data_df = pd.read_csv(data_file)
        column_names = data_df.columns[0].split('\t')[:-1]
        data_df = data_df.iloc[:, 0].str.split('\t', expand=True)
        data_df.columns = column_names
        data_df = data_df[[col for col in data_df.columns if "50%" in col]]
        self.gt_relaxation = torch.tensor(data_df.astype('float32').values.T).to(device)
        
        self.sens_values = []
        for file in coefficient_files:
            self.sens_values.append(calc_detector_from_output_file(file, self.tau_calc))
        self.sens_values = torch.stack(self.sens_values).to(device)
        self.tau_calc = self.tau_calc.to(device)
        
        self.exps = torch.stack([
            torch.exp(-1 * group / (10 ** self.tau_calc)) for group in range(self.time_groups)
        ])
        self.cvxpy_layer = self._set_up_opt(batch_size, self.exps.shape[1], self.gt_relaxation.shape[-1])

    
    def wandb_log(self, x_0_hat):
        
        return ({"loss": self.last_loss,
                 })
        
    def gather_nh_vectors(self, batch_coord, hyodrogens_coords, hydrogen_names):
        """
        For each residue, collect all N->H unit vectors across the ensemble
        of superimposed PDB files and their models.
        """
        residue_vectors = []
        
        for res_id in np.unique(self.atom_array.res_id):
            mask = (self.atom_array.res_id == res_id) & (self.atom_array.atom_name == "N")
            # Find the backbone N coordinate
            coords_N = batch_coord[:,mask]
            
            # Find amide proton(s)
            mask = torch.tensor([(entry[0] == res_id) and (entry[2] in ["H", "HN", "H1", "H2", "H3", "HN1", "HN2"]) for entry in hydrogen_names], device=batch_coord.device)
            coords_H = hyodrogens_coords[:,mask]
            
            if coords_N.shape[1] == 1 and coords_H.shape[1] >= 1:
                # residue_vectorsTake the first amide proton as representative
                # vector = coords_H[:,0] - coords_N[:,0] #TODO
                vector = coords_H[:,0] - coords_N[:,0]
                norm = torch.linalg.norm(vector, dim=-1)
                mask = norm > 1e-8
                vector = torch.where(mask.unsqueeze(-1), vector / norm.unsqueeze(-1), vector)
                residue_vectors.append(vector)
                    
        return torch.stack(residue_vectors, dim=0).permute(1,0,2)
    
    def _set_up_opt(self, bs, n, m):
        dz_tensor = (self.tau_calc[1:] - self.tau_calc[:-1]).cpu()[None, None].repeat(bs,1,1)
        
        theta = cp.Variable((bs, m,n))
        dz = cp.Constant(dz_tensor)
        e = cp.Parameter((bs,n))
        p_target = cp.Parameter((bs, m))
        
        integrand = cp.multiply(e[:,None], theta)
        avg_integrand = cp.multiply(0.5, (integrand[...,1:] + integrand[...,:-1]))
        avg_integrand = cp.multiply(avg_integrand, dz)
        p_pred = cp.sum(avg_integrand, axis=2)
        objective = cp.Minimize(cp.sum_squares(p_pred - p_target))
        # Constraints: θ ≥ 0 and normalized integral
        
        # Trapezoidal normalization: ∫ θ(z) dz = 1 for each row
        avg_theta = 0.5 * (theta[..., 1:] + theta[..., :-1])  # shape: (m, n-1)
        integral_constraint = cp.sum(cp.multiply(avg_theta, dz), axis=2) == 1
        constraints = [
            theta >= 0,
            integral_constraint,
        ]
        problem = cp.Problem(objective, constraints)
        cvxpylayer = CvxpyLayer(problem, parameters=[e, p_target], variables=[theta])
        return cvxpylayer
    
        
        
    def __call__(self, x_0_hat, time):
        num_per_group = x_0_hat.shape[0] // self.time_groups
        
        # add hydrogens
        hydrogen_atoms_batch, naming_sample = self.fragment_library.calculate_hydrogen_coord_batch(x_0_hat, self.atom_array.bonds, 
                                                                                                   self.atom_array.atom_name, self.atom_array.element, 
                                                                                                   self.atom_array.res_name, self.device)
        hydrogen_names = get_hydrogen_names(self.atom_array, naming_sample, self.name_library)
        
        
        amide_vectors = self.gather_nh_vectors(x_0_hat, hydrogen_atoms_batch, hydrogen_names)
        reference = amide_vectors[0][None]
        amide_dot_products = (amide_vectors * reference).sum(dim=-1)
        
        theta, = self.cvxpy_layer(self.exps, amide_dot_products)
        
        dz_tensor = (self.tau_calc[1:] - self.tau_calc[:-1])[None,None]
        integrand = theta*self.sens_values[:,None]
        avg_integrand = 0.5 * (integrand[...,1:] + integrand[...,:-1])

        predicted_relaxation = torch.sum(avg_integrand * dz_tensor, dim=-1)
         
        loss = F.mse_loss(self.gt_relaxation, predicted_relaxation)
        self.last_loss = loss
        
        return loss, None
    
    
    
    