##########################################
# ELDen Flow

###############################
# Imports
import torch
import numpy as np
from .peng_model import ScatteringAttributes

from pykeops.torch import LazyTensor

###############################
# Actual implementation


def initialize_lattice_coordinates(D, voxel_size, leftbottompoint=None, rightupperpoint=None):
    boundary = voxel_size * D
    lattice_center_angstroms = torch.tensor([D/2 * voxel_size, ] * 3, dtype=torch.float32)#.to(device)

    if leftbottompoint is not None and rightupperpoint is not None:
        x_left, y_left, z_left = leftbottompoint
        x_right, y_right, z_right = rightupperpoint
    else:
        x_left, x_right = 0, boundary
        y_left, y_right = 0, boundary
        z_left, z_right = 0, boundary

    x2, x1, x0 = np.meshgrid(
        np.linspace(x_left, x_right, D, endpoint=True),
        np.linspace(y_left, y_right, D, endpoint=True),
        np.linspace(z_left, z_right,D, endpoint=True),
        indexing='ij'
    ) 
    lattice = np.stack([x2.ravel(), x1.ravel(), x0.ravel()], axis=1).astype(np.float32)
    lattice = torch.from_numpy(lattice) #.to(device) # NOTE: think of whether we wanna be moving the lattice to the device here or outside of this function

    return lattice

def create_1d_lattice(D, voxel_size):
    boundary = voxel_size * D
    x = np.linspace(0, boundary, D, endpoint=True)
    x = torch.from_numpy(x)
    #x = torch.from_numpy
    return x

def create_2d_lattice(D, voxel_size):
    boundary = voxel_size * D
    x1, x0 = np.meshgrid(
        np.linspace(0,boundary,D, endpoint=True),
        np.linspace(0,boundary,D, endpoint=True),
        indexing='ij'
    ) 
    lattice = np.stack([x0.ravel(), x1.ravel()], axis=1).astype(np.float32)
    lattice = torch.from_numpy(lattice) #.to(device) 

    return lattice

def compute_elden_no_cycle_keops( 
        D, lattice, atom_positions, atom_identities, b_factors, device
    ): 
    B = atom_positions.shape[0] if len(atom_positions.shape) > 2 else 1 # if the atom positions are not in a batch, then we need to add a batch dimension
    atom_positions = atom_positions.view(-1, 3) # flattening out all the atom positions..! [don't want the batches anymore..!]

    N = atom_positions.shape[0]; D3 = lattice.shape[0]; Kparam = 5 

    atom_identities = atom_identities.repeat(B, 1) 
    b_factors = b_factors.repeat(B, 1) # repeat the b factors for all the batches

    # preparing the LazyTensors
    lattice_i = LazyTensor(lattice[:, None, :]) # Shape (D**3, 1, 3), where D = square slice lattice side length
    atom_positions_j = LazyTensor(
        atom_positions[None, :, :]
    ) # Shape (1, N, 3), where N = number of atoms
    D_ij = ((lattice_i - atom_positions_j) ** 2).sum(dim=2, keepdim=True)  # Shape (D**2, N). 

    scattering_attributes = ScatteringAttributes(device) 
    gaussian_amplitudes, gaussian_widths = scattering_attributes(atom_identities) # a_jk and b_jk respectively, Shape (N, Kparam)
    gaussian_widths = 1 / (gaussian_widths + b_factors.unsqueeze(-1)) 
    a_jk = LazyTensor(gaussian_amplitudes.view(1, N, Kparam)); b_jk = LazyTensor(gaussian_widths.view(1, N, Kparam)) # Shape (1, N, Kparam) to ensure D_ij can be broadcasted

    vol = (
        a_jk * (4 * torch.pi)**(3/2) * b_jk**(3/2) * \
        (-4 * torch.pi**2 * D_ij * b_jk).exp() # The shape is (D**3, N, Kparam)
    ).sum(dim=-1).sum(1) # the on-the-fly calculated elden! 
    return vol / B

def compute_Coloumb_stype_potential(
        D, lattice, atom_positions, atom_identities, b_factors, device
    ):
    """
    Computing electrostatic potential maps without the X-Ray scattering factors as they are much 'wider' and are much harder to control in terms of thickness/broadness of the volume.
    """
    B = atom_positions.shape[0] if len(atom_positions.shape) > 2 else 1 # if the atom positions are not in a batch, then we need to add a batch dimension
    atom_positions = atom_positions.view(-1, 3) # flattening out all the atom positions..! [don't want the batches anymore..!]

    N = atom_positions.shape[0]; D3 = lattice.shape[0]; 

    atom_identities = atom_identities.repeat(B, 1)
    b_factors = b_factors.repeat(B, 1) # repeat the b factors for all the batches

    # preparing the LazyTensors
    lattice_i = LazyTensor(lattice[:, None, :]) # Shape (D**3, 1, 3), where D = square slice lattice side length
    atom_positions_j = LazyTensor(
        atom_positions[None, :, :]
    ) # Shape (1, N, 3), where N = number of atoms
    D_ij = ((lattice_i - atom_positions_j) ** 2).sum(dim=2, keepdim=True)  # Shape (D**2, N).

    sigmas_squared_j = LazyTensor( (b_factors / (8*torch.pi**2)).reshape(1,N,1) )
    atom_identities_j = LazyTensor( atom_identities.reshape(1,N,1) ) # Shape (1, N, 1)

    vol = (
        atom_identities_j * (1 / (2 * torch.pi * sigmas_squared_j)).power(3/2) * \
        (- D_ij / (2 * sigmas_squared_j)).exp() 
    ).sum(1)

    return vol / B


def compute_projection_with_keops(
        D, lattice_2d, atom_positions, atom_identities, b_factors, device
    ):

    N = atom_positions.shape[0]; D2 = lattice_2d.shape[0]; Kparam = 5 

    lattice_i = LazyTensor(lattice_2d[:, None, :].to(device)) # Shape (D**3, 1, 3), where D = square slice lattice side length
    atom_positions_xy_j = LazyTensor(atom_positions[None, :, 0:2].to(device)) # Shape (1, N, 3), where N = number of atoms
    D_ij = ((lattice_i - atom_positions_xy_j) ** 2).sum(dim=2, keepdim=True)  # Shape (D**2, N). 

    scattering_attributes = ScatteringAttributes("cpu") 
    gaussian_amplitudes, gaussian_widths = scattering_attributes(atom_identities) # a_jk and b_jk respectively, Shape (N, Kparam)
    gaussian_widths = 1 / (gaussian_widths + b_factors.unsqueeze(-1)) 
    a_jk = LazyTensor(gaussian_amplitudes.view(1, N, Kparam).to(device)); b_jk = LazyTensor(gaussian_widths.view(1, N, Kparam).to(device)) # Shape (1, N, Kparam) to ensure D_ij can be broadcasted
    
    image = ( # no more roots since we are on the 2D lattice now! the rest in the formulas stays the same really...!
        a_jk * (4 * torch.pi) * b_jk * \
        (-4 * torch.pi**2 * D_ij * b_jk).exp() # The shape is (D**3, N, Kparam)
    ).sum(dim=-1).sum(1).reshape(D, D) # the on-the-fly calculated elden! 
    return image
