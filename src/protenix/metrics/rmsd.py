# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Dict

from torch.nn.utils.rnn import pad_sequence
from typing import Callable, List

import torch
import math

def rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    mask: torch.Tensor = None,
    eps: float = 0.0,
    reduce: bool = True,
):
    """
    compute rmsd between two poses, with the same shape
    Arguments:
        pred_pose/true_pose: [...,N,3], two poses with the same shape
        mask: [..., N], mask to indicate which atoms/pseudo_betas/etc to compute
        eps: add a tolerance to avoid floating number issue
        reduce: decide the return shape of rmsd;
    Return:
        rmsd: if reduce = true, return the mean of rmsd over batches;
            else return a tensor containing each rmsd separately
    """

    # mask [..., N]
    assert pred_pose.shape == true_pose.shape  # [..., N, 3]

    if mask is None:
        mask = torch.ones(true_pose.shape[:-1], device=true_pose.device)

    # [...]
    err2 = (torch.square(pred_pose - true_pose).sum(dim=-1) * mask).sum(
        dim=-1
    ) / mask.sum(dim=-1)
    rmsd = err2.add(eps).sqrt()
    if reduce:
        rmsd = rmsd.mean()
    return rmsd


def align_pred_to_true(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: Optional[torch.Tensor] = None,
    weight: Optional[torch.Tensor] = None,
    allowing_reflection: bool = False,
):
    """Find optimal transformation, rotation (and reflection) of two poses.
    Arguments:
        pred_pose: [...,N,3] the pose to perform transformation on
        true_pose: [...,N,3] the target pose to align pred_pose to
        atom_mask: [..., N] a mask for atoms
        weight: [..., N] a weight vector to be applied.
        allow_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_pose: [...,N,3] the transformed pose
        rot: optimal rotation
        translate: optimal translation
    """
    if atom_mask is not None:
        pred_pose = pred_pose * atom_mask.unsqueeze(-1)
        true_pose = true_pose * atom_mask.unsqueeze(-1)
    else:
        atom_mask = torch.ones(*pred_pose.shape[:-1]).to(pred_pose.device)

    if weight is None:
        weight = atom_mask
    else:
        weight = weight * atom_mask

    weighted_n_atoms = torch.sum(weight, dim=-1, keepdim=True).unsqueeze(-1)
    pred_pose_centroid = (
        torch.sum(pred_pose * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    pred_pose_centered = pred_pose - pred_pose_centroid
    true_pose_centroid = (
        torch.sum(true_pose * weight.unsqueeze(-1), dim=-2, keepdim=True)
        / weighted_n_atoms
    )
    true_pose_centered = true_pose - true_pose_centroid
    H_mat = torch.matmul(
        (pred_pose_centered * weight.unsqueeze(-1)).transpose(-2, -1),
        true_pose_centered * atom_mask.unsqueeze(-1),
    )
    u, s, v = torch.svd(H_mat)
    u = u.transpose(-1, -2)

    if not allowing_reflection:

        det = torch.linalg.det(torch.matmul(v, u))

        diagonal = torch.stack(
            [torch.ones_like(det), torch.ones_like(det), det], dim=-1
        )
        rot = torch.matmul(
            torch.diag_embed(diagonal).to(u.device),
            u,
        )
        rot = torch.matmul(v, rot)
    else:
        rot = torch.matmul(v, u)
    translate = true_pose_centroid - torch.matmul(
        pred_pose_centroid, rot.transpose(-1, -2)
    )

    pred_pose_translated = (
        torch.matmul(pred_pose_centered, rot.transpose(-1, -2)) + true_pose_centroid
    )

    return pred_pose_translated, rot, translate


def partially_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    align_mask: torch.Tensor,
    atom_mask: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    """RMSD when aligning parts of the complex coordinate, does NOT take permutation symmetricity into consideration
    Arguments:
        pred_pose: native predicted pose, [..., N,3]
        true_pose: ground truth pose, [..., N, 3]
        align_mask: a mask representing which coordinates to align [..., N]
        atom_mask: a mask representing which coordinates to compute loss [..., N]
        weight: a weight tensor assining weights in alignment for each atom [..., N]
        eps: add a tolerance to avoid floating number issue in sqrt
        reduce: decide the return shape of rmsd;
        allowing_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_part_rmsd: the rmsd of part being align_masked
        unaligned_part_rmsd: the rmsd of unaligned part
        transformed_pred_pose:
        rot: optimal rotation
        trans: optimal translation
    """
    _, rot, translate = align_pred_to_true(
        pred_pose,
        true_pose,
        atom_mask=atom_mask * align_mask,
        weight=weight,
        allowing_reflection=allowing_reflection,
    )
    transformed_pose = torch.matmul(pred_pose, rot.transpose(-1, -2)) + translate
    err_atom = torch.square(transformed_pose - true_pose).sum(dim=-1) * atom_mask
    aligned_mask, unaligned_mask = atom_mask * align_mask.float(), atom_mask * (
        1 - align_mask.float()
    )
    aligned_part_err_square = (err_atom * aligned_mask).sum(dim=-1) / aligned_mask.sum(
        dim=-1
    )
    unaligned_part_err_square = (err_atom * unaligned_mask).sum(
        dim=-1
    ) / unaligned_mask.sum(dim=-1)
    aligned_part_rmsd = aligned_part_err_square.add(eps).sqrt()
    unaligned_part_rmsd = unaligned_part_err_square.add(eps).sqrt()
    if reduce:
        aligned_part_rmsd = aligned_part_rmsd.mean()
        unaligned_part_rmsd = unaligned_part_rmsd.mean()
    return aligned_part_rmsd, unaligned_part_rmsd, transformed_pose, rot, translate


def self_aligned_rmsd(
    pred_pose: torch.Tensor,
    true_pose: torch.Tensor,
    atom_mask: torch.Tensor,
    eps: float = 0.0,
    reduce: bool = True,
    allowing_reflection: bool = False,
):
    """RMSD when aligning one molecule with ground truth and compute rmsd.
    Arguments:
        pred_pose: native predicted pose, [..., N,3]
        true_pose: ground truth pose, [..., N, 3]
        atom_mask: a mask representing which coordinates to compute loss [..., N]
        eps: add a tolerance to avoid floating number issue in sqrt
        reduce: decide the return shape of rmsd;
        allowing_reflection: whether to allow reflection when finding optimal alignment
    return:
        aligned_rmsd: the rmsd of part being align_masked
        transformed_pred_pose: the aligned pose
        rot: optimal rotation matrix
        trans: optimal translation
    """
    aligned_rmsd, _, transformed_pred_pose, rot, trans = partially_aligned_rmsd(
        pred_pose=pred_pose,
        true_pose=true_pose,
        align_mask=atom_mask,
        atom_mask=atom_mask,
        eps=eps,
        reduce=reduce,
        allowing_reflection=allowing_reflection,
    )
    return aligned_rmsd, transformed_pred_pose, rot, trans


def weighted_rigid_align(
    x: torch.Tensor,
    x_target: torch.Tensor,
    atom_weight: torch.Tensor,
    stop_gradient: bool = True,
) -> tuple[torch.Tensor]:
    """Implements Algorithm 28 in AF3. Wrap `align_pred_to_true`.

    Args:
        x (torch.Tensor): input coordinates, it will be moved to match x_target.
            [..., N_atom, 3]
        x_target (torch.Tensor): target coordinates for the input to match.
            [..., N_atom, 3]
        atom_weight (torch.Tensor): weights for each atom.
            [..., N_atom] or [N_atom]
        stop_gradient (bool): whether to detach the output. If true, will run it with no_grad() ctx.

    Returns:
        x_aligned (torch.Tensor): rotated, translated x which should be closer to x_target.
            [..., N_atom, 3]
    """

    if len(atom_weight.shape) == len(x.shape) - 1:
        assert atom_weight.shape[:-1] == x.shape[:-2]
    else:
        assert len(atom_weight.shape) == 1 and atom_weight.shape[-1] == x.shape[-2]

    if stop_gradient:
        with torch.no_grad():
            x_aligned, rot, trans = align_pred_to_true(
                pred_pose=x,
                true_pose=x_target,
                atom_mask=None,
                weight=atom_weight,
                allowing_reflection=False,
            )
            return x_aligned.detach()
    else:
        x_aligned, rot, trans = align_pred_to_true(
            pred_pose=x,
            true_pose=x_target,
            atom_mask=None,
            weight=atom_weight,
            allowing_reflection=False,
        )
        return x_aligned


def pca_frame(
    coords: torch.Tensor, # Shape [B, N_atoms, 3]
    weights: torch.Tensor | None = None, # Shape [1, N_atoms, 1]. Weights are assumed to be the same, can be changed
):
    weights = weights if weights is not None else torch.ones((1, coords.shape[1], 1), device=coords.device)
    if torch.any(weights < 0):
        weights = weights - weights.min() # make sure weights are non-negative
    weights = weights / weights.sum(dim=1, keepdim=True) # we divide by the voxel values..!

    mu = (weights * coords).sum(dim=1, keepdim=True)
    centered = coords - mu

    cov = (weights * centered).transpose(-2, -1) @ centered

    U, _, _ = torch.linalg.svd(cov)

    dets = U.det()
    U[dets < 0, :, -1] = -U[dets<0, :, -1]

    return U.detach(), mu.detach()

# NOTE: the only difference between align_protein_to_protein and align_protein_to_protein_or_density_pca 
# is that the latter only masks the protein to be aligned, while the former masks both proteins (if required)
def align_protein_to_protein_pca(
    protein_to_be_aligned: torch.Tensor, # Shape [B, N_atoms, 3]
    protein_reference: torch.Tensor, # Shape [1, N_atoms, 3]. We align several to one batch
    protein_to_be_aligned_weights: torch.Tensor | None = None, # Shape [1, N_atoms, 1]
    protein_reference_weights: torch.Tensor | None = None, # Shape [1, M_atoms, 1] 
    reduced_protein_mask: torch.Tensor | None = None, # Shape [ N_atoms ] (optional)
):
    assert protein_to_be_aligned.shape[:-1] == protein_reference.shape[:-1], \
        f"Shapes of protein_to_be_aligned {protein_to_be_aligned.shape} and protein_reference {protein_reference.shape} do not match!"

    if reduced_protein_mask is None:
        reduced_protein_mask = torch.ones(protein_to_be_aligned.shape[1], device=protein_to_be_aligned.device, dtype=torch.bool)

    U_to_be_aligned, mu_to_be_aligned = pca_frame(
        protein_to_be_aligned[:, reduced_protein_mask, :], protein_to_be_aligned_weights[:, reduced_protein_mask, :] if protein_to_be_aligned_weights is not None else None
    )
    U_reference, mu_reference = pca_frame(
        protein_reference[:, reduced_protein_mask, :], (protein_reference_weights[:, reduced_protein_mask, :] if protein_reference_weights is not None else None)
    )

    R = U_reference @ U_to_be_aligned.transpose(-2, -1) # Shape [B, 3, 3]
    t = mu_reference - mu_to_be_aligned @ R.transpose(-2, -1) # Shape [B, 1, 3]

    return protein_to_be_aligned @ R.transpose(-2, -1) + t, R, t # Nice...!

def align_protein_to_density_pca(
    protein_to_be_aligned: torch.Tensor,  # Shape [B, N_atoms, 3]
    density_reference: torch.Tensor,  # Shape [1, M_atoms, 3]
    protein_to_be_aligned_weights: torch.Tensor | None = None,  # Shape [1, N_atoms, 1]
    density_reference_weights: torch.Tensor | None = None,  # Shape [1, M_atoms, 1]
    reduced_protein_mask: torch.Tensor | None = None,  # Shape [N_atoms] (optional)
):
    """
    Aligns a protein to another density or density using PCA.
    """

    if reduced_protein_mask is None:
        reduced_protein_mask = torch.ones(protein_to_be_aligned.shape[1], device=protein_to_be_aligned.device, dtype=torch.bool)

    U_to_be_aligned, mu_to_be_aligned = pca_frame(
        protein_to_be_aligned[:, reduced_protein_mask, :], (protein_to_be_aligned_weights[:, reduced_protein_mask, :] if protein_to_be_aligned_weights is not None else None)
    )
    U_reference, mu_reference = pca_frame( # no masking here, we always take the full density that was provided
        density_reference, density_reference_weights
    )

    R = U_reference @ U_to_be_aligned.transpose(-2, -1) # Shape [B, 3, 3]
    t = mu_reference - mu_to_be_aligned @ R.transpose(-2, -1) # Shape [B, 1, 3]

    return protein_to_be_aligned @ R.transpose(-2, -1) + t, R, t # Nice...!


def align_multimeric_protein_to_multimeric_density_by_chain(
    protein_to_be_aligned: torch.Tensor,  # Shape [1, N_chains * N_atoms_per_chain, 3] # NOTE FOR NOW ONLY SUPPORT PER BATCH 1 I think..!
    density_reference_blobs: list[torch.Tensor],  # A list of N_chains Tensors of Shape [1, M_voxels_centers, 3]
    protein_weights: torch.Tensor | None = None,  # Shape [1, N_chains * N_atoms_per_chain, 1]
    density_reference_blobs_weights: list[torch.Tensor] | None = None,  # A list of N_chains Tensors of Shape [1, M_voxels_centers, 1]
    reduced_protein_mask: torch.Tensor | None = None,  # Shape [N_chains * N_atoms_per_chain] (optional)
):
    """
    NOTE: this alignment algorithm assumes that the protein to be aligned is already in the same order as the density blobs..!!
    """
    N_total_atoms = protein_to_be_aligned.shape[1]
    N_total_atoms_reduced = reduced_protein_mask.sum() if reduced_protein_mask is not None else N_total_atoms
    N_chains = len(density_reference_blobs)
    assert N_total_atoms % N_chains == 0, f"Number of atoms {N_total_atoms} is not divisible by number of chains {N_chains}!"
    N_atoms_per_chain = N_total_atoms // N_chains
    N_atoms_per_chain_reduced = N_total_atoms_reduced // N_chains

    with torch.no_grad():
        protein_weights = protein_weights if protein_weights is not None else torch.ones(
            (1, N_total_atoms, 1), device=protein_to_be_aligned.device
        )
        protein_weights = protein_weights[:, reduced_protein_mask, :] if reduced_protein_mask is not None else protein_weights
        protein_weights = protein_weights.reshape(N_chains, -1, 1)
        protein_weights = protein_weights / protein_weights.sum(dim=1, keepdim=True)  # Normalize weights to sum to 1

        density_reference_blobs_weights = (
            density_reference_blobs_weights
            if density_reference_blobs_weights is not None
            else [torch.ones((1, blob.shape[1], 1), device=blob.device) for blob in density_reference_blobs]
        ) 
        density_reference_blobs_weights = [
            blob_weights - (blob_weights.min() if (blob_weights < 0).any() else 0)
            for blob_weights in density_reference_blobs_weights
        ]
        density_reference_blobs_weights = [
            blob_weights / blob_weights.sum(dim=1, keepdim=True) 
            for blob_weights in density_reference_blobs_weights
        ]  # Normalize weights to sum to 1

        protein_centers_per_chain = (
            protein_to_be_aligned[:, reduced_protein_mask, :].reshape(N_chains, -1, 3) * protein_weights
        ).sum(dim=1).unsqueeze(0)

        density_centers_per_chain = torch.cat([
            (blob * blob_weights).sum(dim=1)
            for blob, blob_weights in zip(density_reference_blobs, density_reference_blobs_weights)
        ]).unsqueeze(0)  # List of [1, 3] tensors

        _, _, R, T = self_aligned_rmsd(
            pred_pose=protein_centers_per_chain,
            true_pose=density_centers_per_chain,
            atom_mask = torch.ones((N_chains), device=protein_to_be_aligned.device, dtype=torch.bool),
        )
        R = R.detach()
        T = T.detach()

    return (R[:,None] @ protein_to_be_aligned[...,None] + T[..., None] ).squeeze(-1), R, T



import math
import torch

def compute_freqs(vol, pixel_size):
    # Perform FFT and calculate amplitude
    fft_data = torch.fft.fftshift(torch.fft.fftn(vol))
    amplitude = fft_data.abs()
    
    # Freqs
    freqs_flat = [torch.fft.fftfreq(shape, d=pixel_size) for shape in vol.shape]
    freqs = torch.meshgrid(*freqs_flat, indexing='ij')
    freqs = [torch.fft.fftshift(f) for f in freqs]

    # Calculate s² values
    s2_full  = (freqs[0]**2 + freqs[1]**2 + freqs[2]**2)

    return freqs, amplitude, s2_full

def apply_bfactor_to_map(vol, pixel_size, B_blur, device):
    """
    vol: torch.Tensor (3D) - input map
    pixel_size: float - voxel spacing in Å
    B_blur: float - positive B-factor to apply (Å²)
    returns: blurred torch.Tensor (3D)
    """
    N = vol.shape
    # FFT
    fft_vol = torch.fft.fftshift(torch.fft.fftn(vol))
    
    # Frequencies grid
    freqs, amplitude, s2_full = compute_freqs(vol, pixel_size)
    
    # Apply B-factor attenuation in Fourier space
    s2_full = s2_full.to(device)
    attenuation = torch.exp(- (B_blur / 4.0) * s2_full).to(device)
    fft_blur = fft_vol * attenuation
    
    # Inverse FFT
    fft_blur = torch.fft.ifftshift(fft_blur)
    blurred = torch.fft.ifftn(fft_blur).real
    return blurred

def _skew(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v.unbind(-1)
    z = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([ z,  -vz,  vy]),
        torch.stack([ vz,   z, -vx]),
        torch.stack([-vy,  vx,   z]),
    ])

def rodrigues_from_axis_angle(w: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(w) + 1e-12
    k = w / theta
    K = _skew(k)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


# ---------- main ----------
def blob_se3_align_adam_debug(
    coords: torch.Tensor,                     # [N,3]
    lattice_coords_3d: torch.Tensor,          # [D,D,D,3] or [D^3,3]
    volume: torch.Tensor,                     # [D,D,D]  (RAW map)
    mask3d: torch.Tensor,                     # [D^3] or [D,D,D] (bool or 0/1)
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,                      # start conservative
    lr_r_deg: float = 1.0,
    reduction: str = "mean",
    center_torque: bool = True,               # kept for API
    sampler_fn=None,                          # e.g. interpolate_scalar_volume_at_points_fast
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0 # cap |t| per step (~1 voxel) or None
):
    assert sampler_fn is not None, "Provide sampler_fn (e.g., interpolate_scalar_volume_at_points_fast)."

    device = volume.device
    dtype  = volume.dtype

    # break upstream graph; keep only (t,w) learnable
    coords = coords.to(device=device, dtype=dtype).detach()
    vol    = volume.to(device=device, dtype=dtype).contiguous().detach()

    # ---- soft mask volume (gentle blur/dilate so weights don't vanish) ----
    if mask3d.dim() == 3:
        mv = mask3d.to(device=device, dtype=vol.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,D,D,D]
    else:
        mv = mask3d.to(device=device, dtype=vol.dtype).reshape(1,1,D,D,D)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mask_vol = mv.squeeze(0).squeeze(0)
    mx = mask_vol.max()
    if mx > 0: mask_vol = mask_vol / mx
    mask_vol = mask_vol.contiguous().detach()            # [D,D,D]

    # centroid-only prealignment (centers of coords & masked lattice)
    mask_flat = (mask_vol.reshape(-1) > 0.5)
    if lattice_coords_3d.dim() == 4:
        lattice_flat = lattice_coords_3d.reshape(-1, 3).to(device=device, dtype=dtype)
    else:
        lattice_flat = lattice_coords_3d.to(device=device, dtype=dtype).reshape(-1, 3)
    coords_centroid = coords.mean(dim=0)
    blob_centroid   = lattice_flat[mask_flat].mean(dim=0)
    T_init = (blob_centroid - coords_centroid).detach()
    coords_shifted = coords + T_init

    # sample the MASKED MAP directly (linear op -> better gradients than sampling vol & mask separately)
    masked_map = (vol * mask_vol).contiguous().detach()
    masked_map = apply_bfactor_to_map(masked_map, voxel_size, 300, device) # NOTE: SINCE WE NEED ROUGH ALIGNMENT, it's very important to blur to make sure grads are not zero..!

    def objective(qp: torch.Tensor) -> torch.Tensor:
        vals = sampler_fn(
            lattice_coords_3d=lattice_coords_3d.detach(),  # keep constants out of graph
            voxel_size=float(voxel_size), D=int(D),
            volume=masked_map, query_points=qp              # ONLY qp carries grads
        )  # [N]
        return vals.mean() if reduction == "mean" else vals.sum()

    # params & optimizer
    t = torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))
    w = torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))
    opt = torch.optim.SGD(
        [{"params":[t], "lr": lr_t_A},
         {"params":[w], "lr": math.radians(lr_r_deg)}],
        momentum=0.0
    )

    # diagnostics: lattice origin for in-bounds check
    if lattice_coords_3d.dim() == 4:
        lattice_min_dbg = lattice_coords_3d[0,0,0].to(device=device, dtype=dtype)
    else:
        lattice_min_dbg = lattice_coords_3d.to(device=device, dtype=dtype).min(dim=0).values
    vsize_t = torch.as_tensor(voxel_size, device=device, dtype=dtype)

    print(f"[DEBUG] Align start: steps={steps}, lr_t_A={lr_t_A:.4f}Å, lr_r_deg={lr_r_deg:.2f}°")

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        R  = rodrigues_from_axis_angle(w)
        qp = coords_shifted @ R.T + t
        obj  = objective(qp)       # we MAXIMIZE this
        loss = -obj                # ascent via minimizing negative
        loss.backward()

        # diagnostics
        with torch.no_grad():
            ijk = (qp - lattice_min_dbg) / vsize_t
            inb = (
                (ijk[...,0] >= 0) & (ijk[...,0] <= D-1) &
                (ijk[...,1] >= 0) & (ijk[...,1] <= D-1) &
                (ijk[...,2] >= 0) & (ijk[...,2] <= D-1)
            ).float().mean().item()

            # finite-diff along ascent directions (-grad loss)
            dObj_T = float('nan'); dObj_R = float('nan')
            if t.grad is not None and t.grad.norm() > 0:
                dT = (-t.grad) / (t.grad.norm() + 1e-12)
                epsT = 0.05 * float(voxel_size)
                obj_T = objective((coords_shifted @ R.T) + (t + epsT * dT)).item()
                dObj_T = obj_T - obj.item()
            if w.grad is not None and w.grad.norm() > 0:
                dW = (-w.grad) / (w.grad.norm() + 1e-12)
                R_eps = rodrigues_from_axis_angle(w + math.radians(0.5) * dW)
                obj_R = objective((coords_shifted @ R_eps.T) + t).item()
                dObj_R = obj_R - obj.item()

        opt.step()

        # optional per-step translation cap (~1 voxel)
        if per_step_t_cap_voxels is not None:
            with torch.no_grad():
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                n = t.norm()
                if n > cap:
                    t.mul_(cap / (n + 1e-12))

        if it == 1 or it % print_every == 0 or it == steps:
            gT = t.grad.norm().item() if t.grad is not None else 0.0
            gW = w.grad.norm().item() if w.grad is not None else 0.0
            print(
                f"[DEBUG] Iter {it:02d}/{steps} obj={obj.item():+.6f} "
                f"||grad_T||={gT:.6f} ||grad_w||={gW:.6f} "
                f"|T|={t.norm().item():.6f}Å |w|={(w.norm().item()*180.0/math.pi):.6f}° "
                f"inb={inb:.3f} dObj(+T)={dObj_T:+.4e} dObj(+R)={dObj_R:+.4e}"
            )

    with torch.no_grad():
        R_final = rodrigues_from_axis_angle(w).detach()
        aligned_coords = coords_shifted @ R_final.T + t
        T_full = (T_init + t).detach()

    return aligned_coords, R_final, T_full





import math
import torch
from typing import Callable, Tuple


def random_rotation_matrices_haar(n: int, *, device=None, dtype=None, seed: int|None=None) -> torch.Tensor:
    """
    Haar(SO(3)) via unit quaternions: sample q ~ N(0,I)^4 / ||.||, convert to R. Returns [n,3,3].
    """
    if dtype is None:
        dtype = torch.float32
    g = torch.Generator(device=device)
    if seed is not None:
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        q = torch.randn(n, 4, generator=g, device=device, dtype=dtype)
    else:
        q = torch.randn(n, 4, device=device, dtype=dtype)  # uses global RNG state (randomized)


    q = q / q.norm(dim=1, keepdim=True)                         # q = (w,x,y,z)
    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]

    R = torch.empty(n,3,3, device=device, dtype=dtype)
    R[:,0,0] = 1 - 2*(y*y + z*z)
    R[:,0,1] = 2*(x*y - z*w)
    R[:,0,2] = 2*(x*z + y*w)
    R[:,1,0] = 2*(x*y + z*w)
    R[:,1,1] = 1 - 2*(x*x + z*z)
    R[:,1,2] = 2*(y*z - x*w)
    R[:,2,0] = 2*(x*z - y*w)
    R[:,2,1] = 2*(y*z + x*w)
    R[:,2,2] = 1 - 2*(x*x + y*y)
    return R



def rot_pi_about_pca_axes(U_ref: torch.Tensor):
    """
    Return the three 180° rotation matrices (det=+1) about the PCA axes in U_ref.
    """
    device, dtype = U_ref.device, U_ref.dtype
    Ds = [
        torch.diag(torch.tensor([ 1., -1., -1.], device=device, dtype=dtype)),  # π about axis 0
        torch.diag(torch.tensor([-1.,  1., -1.], device=device, dtype=dtype)),  # π about axis 1
        torch.diag(torch.tensor([-1., -1.,  1.], device=device, dtype=dtype)),  # π about axis 2
    ]
    Rxs = [U_ref @ D @ U_ref.T for D in Ds]  # conjugate into world frame

    # sanity: ensure proper rotations
    one = torch.tensor(1., device=device, dtype=dtype)
    for R in Rxs:
        assert torch.allclose(torch.det(R), one, atol=1e-5), "π-rotation must have det=+1"
    return Rxs[0], Rxs[1], Rxs[2]

def pca_frame(
    coords: torch.Tensor, # Shape [B, N_atoms, 3]
    weights: torch.Tensor | None = None, # Shape [1, N_atoms, 1]. Weights are assumed to be the same, can be changed
):
    weights = weights if weights is not None else torch.ones((1, coords.shape[1], 1), device=coords.device)
    if torch.any(weights < 0):
        weights = weights - weights.min() # make sure weights are non-negative
    weights = weights / weights.sum(dim=1, keepdim=True) # we divide by the voxel values..!

    mu = (weights * coords).sum(dim=1, keepdim=True)
    centered = coords - mu

    cov = (weights * centered).transpose(-2, -1) @ centered

    U, _, _ = torch.linalg.svd(cov)

    dets = U.det()
    U[dets < 0, :, -1] = -U[dets<0, :, -1]

    return U.detach(), mu.detach()

def rodrigues_batch(w: torch.Tensor) -> torch.Tensor:
    """
    Batched axis-angle to rotation, w:[B,3] -> R:[B,3,3]
    """
    B = w.shape[0]
    theta = torch.linalg.norm(w, dim=-1).clamp_min(1e-12)           # [B]
    k = w / theta.unsqueeze(-1)                                     # [B,3]
    kx, ky, kz = k.unbind(-1)
    zeros = torch.zeros(B, device=w.device, dtype=w.dtype)
    K = torch.stack([
        zeros, -kz,   ky,
           kz, zeros, -kx,
          -ky,  kx, zeros
    ], dim=-1).reshape(B,3,3)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand(B,3,3)
    s = torch.sin(theta).view(B,1,1)
    c = torch.cos(theta).view(B,1,1)
    return I + s*K + (1.0 - c) * (K @ K)  

def _pca_axes_from_points(points_1xNx3: torch.Tensor) -> torch.Tensor:
    """
    Returns a right-handed PCA frame U_ref [3,3] for density/reference points.
    points_1xNx3: [1, N, 3]
    """
    U_ref, _ = pca_frame(points_1xNx3)     # uses your function; returns [1,3,3]
    return U_ref[0]

def _pca_rotation_for_protein_vs_density(protein_1xNx3: torch.Tensor,
                                         density_1xMx3: torch.Tensor) -> torch.Tensor:
    """
    Compute R_pca = U_ref @ U_src^T (same convention as your align_*_pca).
    """
    U_src, _ = pca_frame(protein_1xNx3)    # [1,3,3]
    U_ref, _ = pca_frame(density_1xMx3)    # [1,3,3]
    R = U_ref @ U_src.transpose(-2, -1)    # [1,3,3]
    # enforce right-handedness (pca_frame already fixes det>0, so this is mostly redundant)
    det = torch.det(R[0])
    if det < 0:
        R = R.clone()
        R[0,:,2] = -R[0,:,2]
    return R[0]                             # [3,3]


def interpolate_scalar_volume_at_points_fast_testing_corrected(
    lattice_coords_3d: torch.Tensor,  # [D,D,D,3] or [D^3,3]
    voxel_size: float,
    D: int,
    volume: torch.Tensor,             # [D,D,D] (float), layout [x,y,z]
    query_points: torch.Tensor,       # [N,3] or [B,N,3]
) -> torch.Tensor:
    """
    Trilinear interpolation of `volume` at real-space `query_points`.
    Differentiable w.r.t. query_points. Assumes axis-aligned, uniform voxels.
    """
    vol = volume.contiguous()
    device, dtype = vol.device, vol.dtype

    # lattice origin (voxel centers)
    if lattice_coords_3d.dim() == 4:
        lattice_min = lattice_coords_3d[0, 0, 0].to(device=device, dtype=dtype)
    else:
        lattice_min = lattice_coords_3d.to(device=device, dtype=dtype).min(dim=0).values

    vsize = torch.as_tensor(voxel_size, device=device, dtype=dtype)

    batched = (query_points.dim() == 3)
    qp = query_points.to(device=device, dtype=dtype)
    if not batched:
        qp = qp[None, ...]  # [1,N,3]

    # real → voxel index coords
    coords = (qp - lattice_min) / vsize  # [B,N,3]
    x, y, z = coords.unbind(-1)

    # Clamp the floored indices, not the continuous coordinates
    # This ensures x1, y1, z1 are always <= D-1 (in bounds)
    x0 = torch.floor(x).clamp(0, D - 2).to(torch.long)
    y0 = torch.floor(y).clamp(0, D - 2).to(torch.long)
    z0 = torch.floor(z).clamp(0, D - 2).to(torch.long)
    #print(x0)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    dx = (x - x0.float()).unsqueeze(-1)  # [B,N,1]
    dy = (y - y0.float()).unsqueeze(-1)
    dz = (z - z0.float()).unsqueeze(-1)

    # correct flat indexing for [x,y,z] layout using strides
    s0, s1, s2 = vol.stride()  # elements
    def lin(i, j, k):  # i=x, j=y, k=z
        return i * s0 + j * s1 + k * s2

    flat = vol.view(-1)

    idx000 = lin(x0, y0, z0); idx001 = lin(x0, y0, z1)
    idx010 = lin(x0, y1, z0); idx011 = lin(x0, y1, z1)
    idx100 = lin(x1, y0, z0); idx101 = lin(x1, y0, z1)
    idx110 = lin(x1, y1, z0); idx111 = lin(x1, y1, z1)

    c000 = flat.take(idx000).unsqueeze(-1)
    c001 = flat.take(idx001).unsqueeze(-1)
    c010 = flat.take(idx010).unsqueeze(-1)
    c011 = flat.take(idx011).unsqueeze(-1)
    c100 = flat.take(idx100).unsqueeze(-1)
    c101 = flat.take(idx101).unsqueeze(-1)
    c110 = flat.take(idx110).unsqueeze(-1)
    c111 = flat.take(idx111).unsqueeze(-1)

    #c00 = c000*(1-dz) + c001*dz
    #c01 = c010*(1-dz) + c011*dz
    #c10 = c100*(1-dz) + c101*dz
    #c11 = c110*(1-dz) + c111*dz
    #c0  = c00 *(1-dy) + c01 *dy
    #c1  = c10 *(1-dy) + c11 *dy
    #out = (c0*(1-dx) + c1*dx).squeeze(-1)  # [B,N]
    wx0, wx1 = 1 - dx, dx
    wy0, wy1 = 1 - dy, dy
    wz0, wz1 = 1 - dz, dz

    out = (
        wx0*wy0*wz0*c000 + wx1*wy0*wz0*c100 +
        wx0*wy1*wz0*c010 + wx1*wy1*wz0*c110 +
        wx0*wy0*wz1*c001 + wx1*wy0*wz1*c101 +
        wx0*wy1*wz1*c011 + wx1*wy1*wz1*c111
    ).squeeze(-1)

    return out if batched else out[0]

def compute_freqs(vol, pixel_size):
    # Perform FFT and calculate amplitude
    fft_data = torch.fft.fftshift(torch.fft.fftn(vol))
    amplitude = fft_data.abs()
    
    # Freqs
    freqs_flat = [torch.fft.fftfreq(shape, d=pixel_size) for shape in vol.shape]
    freqs = torch.meshgrid(*freqs_flat, indexing='ij')
    freqs = [torch.fft.fftshift(f) for f in freqs]

    # Calculate s² values
    s2_full  = (freqs[0]**2 + freqs[1]**2 + freqs[2]**2)

    return freqs, amplitude, s2_full

def apply_bfactor_to_map(vol, pixel_size, B_blur, device):
    """
    vol: torch.Tensor (3D) - input map
    pixel_size: float - voxel spacing in Å
    B_blur: float - positive B-factor to apply (Å²)
    returns: blurred torch.Tensor (3D)
    """
    N = vol.shape
    # FFT
    fft_vol = torch.fft.fftshift(torch.fft.fftn(vol))
    
    # Frequencies grid
    freqs, amplitude, s2_full = compute_freqs(vol, pixel_size)
    
    # Apply B-factor attenuation in Fourier space
    s2_full = s2_full.to(device)
    attenuation = torch.exp(- (B_blur / 4.0) * s2_full).to(device)
    fft_blur = fft_vol * attenuation
    
    # Inverse FFT
    fft_blur = torch.fft.ifftshift(fft_blur)
    blurred = torch.fft.ifftn(fft_blur).real
    return blurred

def _skew(v: torch.Tensor) -> torch.Tensor:
    vx, vy, vz = v.unbind(-1)
    z = torch.zeros((), device=v.device, dtype=v.dtype)
    return torch.stack([
        torch.stack([ z,  -vz,  vy]),
        torch.stack([ vz,   z, -vx]),
        torch.stack([-vy,  vx,   z]),
    ])

def rodrigues_from_axis_angle(w: torch.Tensor) -> torch.Tensor:
    theta = torch.norm(w) + 1e-12
    k = w / theta
    K = _skew(k)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    return I + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def blob_se3_align_adam_debug(
    coords: torch.Tensor,                     # [N,3]
    lattice_coords_3d: torch.Tensor,          # [D,D,D,3] or [D^3,3]
    volume: torch.Tensor,                     # [D,D,D]  (RAW map)
    mask3d: torch.Tensor,                     # [D^3] or [D,D,D] (bool or 0/1)
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,                      # Å per step
    lr_r_deg: float = 1.0,                    # degrees per step
    reduction: str = "mean",
    center_torque: bool = True,               # kept for API; not used
    sampler_fn: Callable = None,              # e.g. interpolate_scalar_volume_at_points_fast
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0, # cap ||Δt|| per step (~1 voxel)
    Bfac: float = 300,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Gradient-ascent alignment of atom coords to a masked density map.
    Rotation is applied about a fixed protein centroid computed after pre-centering.
    Translation cap applies to *increment* Δt each step.
    """
    assert sampler_fn is not None, "Provide sampler_fn (e.g., interpolate_scalar_volume_at_points_fast)."

    device = volume.device
    dtype  = volume.dtype

    # --- constants (no grad) ---
    X = coords.to(device=device, dtype=dtype).detach()               # [N,3]
    vol = volume.to(device=device, dtype=dtype).contiguous().detach()

    # Soft mask (light blur/dilate)
    if mask3d.dim() == 3:
        mv = mask3d.to(device=device, dtype=vol.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,D,D,D]
    else:
        mv = mask3d.to(device=device, dtype=vol.dtype).reshape(1,1,D,D,D)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mask_vol = mv.squeeze(0).squeeze(0).contiguous()
    mx = mask_vol.max()
    if mx > 0: mask_vol = mask_vol / mx
    mask_vol = mask_vol.detach()                                      # [D,D,D]

    # Centroid-only prealignment (centers of coords & masked lattice)
    if lattice_coords_3d.dim() == 4:
        lattice_flat = lattice_coords_3d.reshape(-1, 3).to(device=device, dtype=dtype)
    else:
        lattice_flat = lattice_coords_3d.to(device=device, dtype=dtype).reshape(-1, 3)
    mask_flat = (mask_vol.reshape(-1) > 0.5)
    coords_centroid = X.mean(dim=0)
    blob_centroid   = lattice_flat[mask_flat].mean(dim=0)
    T_init = (blob_centroid - coords_centroid).detach()
    X_shifted = (X + T_init).detach()

    # Fixed rotation pivot = protein centroid after pre-shift
    m0 = X_shifted.mean(0).detach()
    Xc = (X_shifted - m0).contiguous()                                # centered coords (constant)

    # Masked (and blurred) map for smoother grads
    masked_map = (vol * mask_vol).contiguous().detach()
    masked_map = apply_bfactor_to_map(masked_map, voxel_size, Bfac, device)  # assumes provided

    # Objective (only qp carries grads)
    def objective(qp: torch.Tensor) -> torch.Tensor:
        vals = sampler_fn(
            lattice_coords_3d=lattice_coords_3d.detach(),
            voxel_size=float(voxel_size), D=int(D),
            volume=masked_map, query_points=qp
        )  # [N]
        return vals.mean() if reduction == "mean" else vals.sum()

    # Learnable params & optimizer
    t = torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))
    w = torch.nn.Parameter(torch.zeros(3, device=device, dtype=dtype))
    opt = torch.optim.SGD(
        [{"params":[t], "lr": lr_t_A},
         {"params":[w], "lr": math.radians(lr_r_deg)}],
        momentum=0.0
    )

    # Diagnostics for in-bounds (use lattice min corner)
    if lattice_coords_3d.dim() == 4:
        lattice_min_dbg = lattice_coords_3d.reshape(-1,3).to(device=device, dtype=dtype).min(dim=0).values
    else:
        lattice_min_dbg = lattice_coords_3d.to(device=device, dtype=dtype).min(dim=0).values
    vsize_t = torch.as_tensor(voxel_size, device=device, dtype=dtype)

    print(f"[DEBUG] Align start: steps={steps}, lr_t_A={lr_t_A:.4f}Å, lr_r_deg={lr_r_deg:.2f}°")

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        R  = rodrigues_from_axis_angle(w)              # recompute from current w
        qp = Xc @ R.T + m0 + t                         # rotate about m0, then translate

        obj  = objective(qp)                           # ascent via minimizing negative
        loss = -obj
        loss.backward()

        # Diagnostics (finite differences along ascent directions)
        with torch.no_grad():
            ijk = (qp - lattice_min_dbg) / vsize_t
            inb = (
                (ijk[...,0] >= 0) & (ijk[...,0] <= D-1) &
                (ijk[...,1] >= 0) & (ijk[...,1] <= D-1) &
                (ijk[...,2] >= 0) & (ijk[...,2] <= D-1)
            ).float().mean().item()

            dObj_T = float('nan'); dObj_R = float('nan')
            if t.grad is not None and t.grad.norm() > 0:
                dT = (-t.grad) / (t.grad.norm() + 1e-12)
                epsT = 0.05 * float(voxel_size)
                obj_T = objective(Xc @ R.T + m0 + (t + epsT * dT)).item()
                dObj_T = obj_T - obj.item()
            if w.grad is not None and w.grad.norm() > 0:
                dW = (-w.grad) / (w.grad.norm() + 1e-12)
                R_eps = rodrigues_from_axis_angle(w + math.radians(0.5) * dW)
                obj_R = objective(Xc @ R_eps.T + m0 + t).item()
                dObj_R = obj_R - obj.item()

        # Step
        t_prev = t.detach().clone()
        opt.step()

        # Cap the *increment* in translation (~ one voxel)
        if per_step_t_cap_voxels is not None:
            with torch.no_grad():
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                delta = t - t_prev
                n = delta.norm()
                if n > cap:
                    t.copy_(t_prev + delta * (cap / (n + 1e-12)))

        if it == 1 or it % print_every == 0 or it == steps:
            gT = t.grad.norm().item() if t.grad is not None else 0.0
            gW = w.grad.norm().item() if w.grad is not None else 0.0
            print(
                f"[DEBUG] Iter {it:02d}/{steps} obj={obj.item():+.6f} "
                f"||grad_T||={gT:.6f} ||grad_w||={gW:.6f} "
                f"|T|={t.norm().item():.6f}Å |w|={(w.norm().item()*180.0/math.pi):.6f}° "
                f"inb={inb:.3f} dObj(+T)={dObj_T:+.4e} dObj(+R)={dObj_R:+.4e}"
            )

    with torch.no_grad():
        R_final = rodrigues_from_axis_angle(w).detach()
        aligned_coords = Xc @ R_final.T + m0 + t
        T_full = (T_init + t).detach()

    return aligned_coords, R_final, T_full


def blob_se3_align_adam_multi_start(
    coords: torch.Tensor,                     # [N,3]
    lattice_coords_3d: torch.Tensor,          # [D,D,D,3] or [D^3,3]
    volume: torch.Tensor,                     # [D,D,D]
    mask3d: torch.Tensor,                     # [D^3] or [D,D,D] (bool/0-1)
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,
    lr_r_deg: float = 1.0,
    reduction: str = "mean",
    sampler_fn: Callable = None,
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0,
    Bfac: float = 300.0,
    n_random: int = 16,
    seed: int | None = None,
    return_all: bool = False,
    time: float = 0.0, # Time for b-factor annealing
    bfactor_minimum: float = 100.0,
    t_init_box_edge_voxels: float | None = None
):

    """
    Multi-start variant of your solver:
      - PCA pre-alignment frame
      - 6 extra 180° starts around density PCA axes (pre & post)
      - n_random Haar(SO(3)) starts composed with R_pca
      - Batched Adam-like ascent over all starts
    Returns dict with best and optionally all batch results.
    """
    assert sampler_fn is not None, "Provide sampler_fn (e.g., interpolate_scalar_volume_at_points_fast_testing_corrected)."

    device = volume.device
    dtype  = volume.dtype

    # --------- Prepare map, mask, and density points (same logic as your single-run) ----------
    vol = volume.to(device=device, dtype=dtype).contiguous().detach()

    if mask3d.dim() == 3:
        mv = mask3d.to(device=device, dtype=vol.dtype).unsqueeze(0).unsqueeze(0)  # [1,1,D,D,D]
    else:
        mv = mask3d.to(device=device, dtype=vol.dtype).reshape(1,1,D,D,D)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mask_vol = mv.squeeze(0).squeeze(0).contiguous()
    mx = mask_vol.max()
    if mx > 0: mask_vol = mask_vol / mx
    mask_vol = mask_vol.detach()                                      # [D,D,D]

    # lattice flatten + masked density points
    if lattice_coords_3d.dim() == 4:
        lattice_flat = lattice_coords_3d.reshape(-1, 3).to(device=device, dtype=dtype)
    else:
        lattice_flat = lattice_coords_3d.to(device=device, dtype=dtype).reshape(-1, 3)
    mask_flat = (mask_vol.reshape(-1) > 0.5)
    density_pts = lattice_flat[mask_flat].reshape(1, -1, 3)           # [1,M,3]

    # --------- PCA frame and initial rotations ----------
    # Base PCA rotation mapping protein -> density
    protein_1xNx3 = coords.to(device=device, dtype=dtype).reshape(1, -1, 3)
    R_pca = _pca_rotation_for_protein_vs_density(protein_1xNx3, density_pts)  # [3,3]
    U_ref = _pca_axes_from_points(density_pts)         # keep this line
    Rx, Ry, Rz = rot_pi_about_pca_axes(U_ref)          # NEW

    R_rand = random_rotation_matrices_haar(n_random, device=device, dtype=dtype, seed=seed)
    R_inits = torch.cat([
        R_pca.unsqueeze(0),
        torch.stack([R_pca @ Rx, R_pca @ Ry, R_pca @ Rz,   # post (local)
                    Rx @ R_pca, Ry @ R_pca, Rz @ R_pca],  # pre  (global)
                    dim=0),
        R_rand @ R_pca,
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    ], dim=0)
    B = R_inits.shape[0]        # <-- ADD THIS LINE (must be before t/w/opt)

    #perm = torch.randperm(R_inits.shape[0], device=R_inits.device)
    #R_inits = R_inits[perm] 

    # --------- Pre-centering (keep your original translation init logic) ----------
    X = coords.to(device=device, dtype=dtype).detach()                     # [N,3]
    coords_centroid = X.mean(dim=0)
    blob_centroid   = lattice_flat[mask_flat].mean(dim=0)
    T_init = (blob_centroid - coords_centroid).detach()

    X_shifted = (X + T_init).detach()
    m0 = X_shifted.mean(0).detach()
    Xc = (X_shifted - m0).contiguous()                                     # [N,3] centered

    # --------- Smoothed masked map (unchanged) ----------
    masked_map = (vol * mask_vol).contiguous().detach()
    # Bfactor decreases with time..!
    Bfac_annealed = bfactor_minimum + (1.0 - time) * (Bfac - bfactor_minimum)
    masked_map = apply_bfactor_to_map(masked_map, voxel_size, Bfac_annealed, device)

    vsize_t = torch.as_tensor(voxel_size, device=device, dtype=dtype)

    # Objective now supports batched qp: [B,N,3] -> [B] (mean/sum per batch)
    def objective_batched(qp: torch.Tensor) -> torch.Tensor:
        """
        qp: [B,N,3]
        returns: [B] scores to MAXIMIZE
        """
        vals = sampler_fn(
            lattice_coords_3d=lattice_coords_3d.detach(),
            voxel_size=float(voxel_size), D=int(D),
            volume=masked_map, query_points=qp
        )  # [B,N] (your sampler supports batched inputs)
        if reduction == "mean":
            return vals.mean(dim=-1)                    # [B]
        else:
            return vals.sum(dim=-1)                     # [B]

    # --------- Learnable batch params ---------
    t_inits = ( torch.rand(B, 3, device=device, dtype=dtype) - 0.5 ) * 2.0 * t_init_box_edge_voxels
    t = torch.nn.Parameter(t_inits)
    w = torch.nn.Parameter(torch.zeros(B, 3, device=device, dtype=dtype))
    opt = torch.optim.SGD(
        [{"params":[t], "lr": lr_t_A},
         {"params":[w], "lr": math.radians(lr_r_deg)}],
        momentum=0.0
    )

    #print(f"[DEBUG] Align start (batched): B={B} starts | steps={steps}, lr_t_A={lr_t_A:.4f}Å, lr_r_deg={lr_r_deg:.2f}°")

    traj = [] if return_all else None

    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)

        # Compose current rotations with init rotations
        R_delta = rodrigues_batch(w)                             # [B,3,3]
        R_curr  = torch.matmul(R_delta, R_inits)                 # [B,3,3]

        # Transform atoms: X' = Xc R^T + m0 + t
        Xb = Xc.unsqueeze(0).expand(B, -1, -1)                   # [B,N,3]
        qp = torch.matmul(Xb, R_curr.transpose(1,2)) + t[:,None,:]

        # Add the fixed pivot back
        qp = qp + m0.unsqueeze(0).unsqueeze(0)                   # [B,N,3]

        # Objective and loss
        scores = objective_batched(qp)                           # [B], maximize
        loss = -scores.sum()                                     # minimize negative total
        loss.backward()

        # Translation increment cap (~ per-step voxel)
        t_prev = t.detach().clone()
        opt.step()
        if per_step_t_cap_voxels is not None:
            with torch.no_grad():
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                delta = t - t_prev
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scale = torch.clamp(cap / norms, max=1.0)
                t.copy_(t_prev + delta * scale)

        if it == 1 or it % print_every == 0 or it == steps:
            gT = t.grad.norm().item() if t.grad is not None else 0.0
            gW = w.grad.norm().item() if w.grad is not None else 0.0
            s_max = scores.max().item()
            s_mean = scores.mean().item()
            #print(f"[DEBUG] Iter {it:02d}/{steps} best={s_max:+.6f} mean={s_mean:+.6f} "
            #      f"||grad_T||={gT:.6f} ||grad_w||={gW:.6f} |T|max={t.norm(dim=-1).max().item():.6f}Å "
            #      f"|w|max={(w.norm(dim=-1).max().item()*180.0/math.pi):.6f}°")

        if return_all:
            traj.append({
                "it": it,
                "scores": scores.detach().clone(),
                "t": t.detach().clone(),
                "w": w.detach().clone(),
            })

    # --------- Finalize and pick best ---------
    with torch.no_grad():
        R_delta = rodrigues_batch(w)                             # [B,3,3]
        R_best_all = torch.matmul(R_delta, R_inits)              # [B,3,3]
        Xb = Xc.unsqueeze(0).expand(B, -1, -1)
        qp_final = torch.matmul(Xb, R_best_all.transpose(1,2)) + t[:,None,:] + m0.unsqueeze(0).unsqueeze(0)

        final_scores = objective_batched(qp_final)               # [B]
        best_idx = torch.argmax(final_scores)

        best_idx = torch.argmax(final_scores)
        #reverse_perm = torch.argsort(perm)
        #best_idx = reverse_perm[best_idx]

        # Compose outputs to match your original return style
        aligned_coords_best = qp_final[best_idx]                 # [N,3]
        R_final_best = R_best_all[best_idx]                      # [3,3]
        T_full_all = T_init + t                                  # [B,3]  (same definition as your single-run)
        T_full_best = T_full_all[best_idx]                       # [3]

        T_global_all = torch.einsum('i,bij->bj', (T_init - m0), R_best_all.transpose(1,2)) + m0 + t  # [B,3]
        T_global_best = T_global_all[best_idx]

        out = {
            "best_aligned_coords": aligned_coords_best,          # [N,3]
            "best_R": R_final_best,                              # [3,3]
            "best_T": T_full_best,                               # [3]
            "best_score": final_scores[best_idx].item(),
            "best_batch_index": int(best_idx.item()),
            "all_scores": final_scores.detach(),                 # [B]
            "all_R": R_best_all.detach(),                        # [B,3,3]
            "all_T": T_full_all.detach(),                        # [B,3]
            "T_global_best": T_global_best.detach(),            # [3]
        }
        if return_all:
            out["trajectory"] = traj
        return out



def density_se3_align_adam_multi_start(
    coords: torch.Tensor,                     # [N,3] protein coordinates
    lattice_coords_3d: torch.Tensor,          # [D,D,D,3] or [D^3,3] lattice coordinates
    volume: torch.Tensor,                     # [D,D,D] observed density (fo)
    mask3d: torch.Tensor,                     # [D^3] or [D,D,D] density mask
    elements: torch.Tensor,                   # [N] element types
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,
    lr_r_deg: float = 1.0,
    reduction: str = "mean",
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0,
    Bfac: float = 300.0,
    n_random: int = 16,
    seed: int | None = None,
    return_all: bool = False,
    time: float = 0.0,
    bfactor_minimum: float = 100.0,
    t_init_box_edge_voxels: float | None = None,
    apply_bfactor_smoothing_to_fo: bool = False,
    rmax: float = 5.0,
    use_Coloumb: bool = False
):
    """
    Density-based SE3 alignment using ESP calculation instead of trilinear interpolation.
    
    Similar to blob_se3_align_adam_multi_start but:
    1. Computes protein density using ESP calculation
    2. Compares with observed density using L1 loss (same as B-factor fitting)
    3. Optimizes rotation and translation to minimize density difference
    
    Args:
        coords: [N,3] protein atom coordinates
        lattice_coords_3d: [D,D,D,3] or [D^3,3] lattice coordinates
        volume: [D,D,D] observed density map (fo)
        mask3d: [D^3] or [D,D,D] density mask
        elements: [N] element types for ESP calculation
        apply_bfactor_smoothing_to_fo: If True, apply B-factor smoothing to fo and compensate in ESP
        rmax: Maximum radius for ESP calculation
        use_Coloumb: Whether to use Coulomb potential in ESP
    
    Returns:
        Dict with best alignment results similar to blob_se3_align_adam_multi_start
    """
    from src.losses.em_loss_function import apply_bfactor_to_map
    
    device = volume.device
    dtype = volume.dtype
    
    # --------- Prepare map, mask, and density points ----------
    vol = volume.to(device=device, dtype=dtype).contiguous().detach()
    
    if mask3d.dim() == 3:
        mask_vol = mask3d.to(device=device, dtype=torch.bool)
    else:
        mask_vol = mask3d.to(device=device, dtype=torch.bool).reshape(D, D, D)
    
    # Lattice coordinates
    if lattice_coords_3d.dim() == 4:
        lattice_flat = lattice_coords_3d.reshape(-1, 3).to(device=device, dtype=dtype)
    else:
        lattice_flat = lattice_coords_3d.to(device=device, dtype=dtype).reshape(-1, 3)
    
    mask_flat = mask_vol.reshape(-1)
    density_pts = lattice_flat[mask_flat].reshape(1, -1, 3)  # [1,M,3]
    
    # --------- PCA frame and initial rotations ----------
    protein_1xNx3 = coords.to(device=device, dtype=dtype).reshape(1, -1, 3)
    R_pca = _pca_rotation_for_protein_vs_density(protein_1xNx3, density_pts)
    U_ref = _pca_axes_from_points(density_pts)
    Rx, Ry, Rz = rot_pi_about_pca_axes(U_ref)
    
    R_rand = random_rotation_matrices_haar(n_random, device=device, dtype=dtype, seed=seed)
    R_inits = torch.cat([
        R_pca.unsqueeze(0),
        torch.stack([R_pca @ Rx, R_pca @ Ry, R_pca @ Rz,
                    Rx @ R_pca, Ry @ R_pca, Rz @ R_pca], dim=0),
        R_rand @ R_pca,
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    ], dim=0)
    B = R_inits.shape[0]
    
    # --------- Pre-centering ----------
    X = coords.to(device=device, dtype=dtype).detach()
    coords_centroid = X.mean(dim=0)
    blob_centroid = lattice_flat[mask_flat].mean(dim=0)
    T_init = (blob_centroid - coords_centroid).detach()
    
    X_shifted = (X + T_init).detach()
    m0 = X_shifted.mean(0).detach()
    Xc = (X_shifted - m0).contiguous()
    
    # --------- Prepare observed density ----------
    # B-factor annealing
    Bfac_annealed = bfactor_minimum + (1.0 - time) * (Bfac - bfactor_minimum)
    
    # Apply B-factor smoothing to fo if requested
    if apply_bfactor_smoothing_to_fo:
        fo_smoothed = apply_bfactor_to_map(vol, voxel_size, Bfac_annealed, device)
        # Compensate by increasing protein B-factors
        protein_bfac = Bfac_annealed + Bfac_annealed
    else:
        fo_smoothed = vol
        protein_bfac = Bfac_annealed
    
    # Extract masked observed density
    fo_masked = fo_smoothed[mask_vol]  # [M]
    
    # Normalize observed density (same as in B-factor fitting)
    fo_normalized = (fo_masked - fo_masked.mean()) / (fo_masked.std() + 1e-6)
    
    # Elements tensor
    elements_tensor = elements.to(device=device, dtype=dtype)
    
    def objective_batched(qp: torch.Tensor) -> torch.Tensor:
        """
        qp: [B,N,3] protein coordinates for each batch
        returns: [B] scores to MAXIMIZE (negative L1 loss)
        """
        batch_scores = []
        
        for b in range(B):
            # Calculate ESP for current protein position using PyKeOps for scalability
            try:
                fc_predicted = calculate_ESP_at_lattice_points_keops(
                    coords=qp[b],  # [N,3]
                    lattice_coords=lattice_flat[mask_flat],  # [M,3]
                    elements=elements_tensor,  # [N]
                    bfactor=torch.ones_like(elements_tensor, dtype=dtype) * protein_bfac,
                    voxel_size=voxel_size,
                    rmax=rmax,
                    use_Coloumb=True,  # Always use Coulomb for PyKeOps efficiency
                    device=device
                )  # [M]
            except ImportError:
                # Fallback to standard implementation if PyKeOps not available
                fc_predicted = calculate_ESP_at_lattice_points(
                    coords=qp[b],  # [N,3]
                    lattice_coords=lattice_flat[mask_flat],  # [M,3]
                    elements=elements_tensor,  # [N]
                    bfactor=torch.ones_like(elements_tensor, dtype=dtype) * protein_bfac,
                    voxel_size=voxel_size,
                    rmax=rmax,
                    use_Coloumb=use_Coloumb,
                    device=device
                )  # [M]
            
            # Normalize predicted density
            fc_normalized = (fc_predicted - fc_predicted.mean()) / (fc_predicted.std() + 1e-6)
            
            # L1 loss (same as B-factor fitting) - minimize, so negate for maximization
            l1_loss = (0.5 * (fo_normalized - fc_normalized).abs()).mean()
            score = -l1_loss  # Negate to maximize
            batch_scores.append(score)
        
        return torch.stack(batch_scores)  # [B]
    
    # --------- Learnable batch params ---------
    if t_init_box_edge_voxels is None:
        t_init_box_edge_voxels = 1.0
    
    t_inits = (torch.rand(B, 3, device=device, dtype=dtype) - 0.5) * 2.0 * t_init_box_edge_voxels
    t = torch.nn.Parameter(t_inits)
    w = torch.nn.Parameter(torch.zeros(B, 3, device=device, dtype=dtype))
    opt = torch.optim.SGD(
        [{"params": [t], "lr": lr_t_A},
         {"params": [w], "lr": math.radians(lr_r_deg)}],
        momentum=0.0
    )
    
    print(f"[DEBUG] Density align start: B={B} starts | steps={steps}, lr_t_A={lr_t_A:.4f}Å, lr_r_deg={lr_r_deg:.2f}°")
    
    traj = [] if return_all else None
    
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        
        # Compose current rotations with init rotations
        R_delta = rodrigues_batch(w)  # [B,3,3]
        R_curr = torch.matmul(R_delta, R_inits)  # [B,3,3]
        
        # Transform atoms: X' = Xc R^T + m0 + t
        Xb = Xc.unsqueeze(0).expand(B, -1, -1)  # [B,N,3]
        qp = torch.matmul(Xb, R_curr.transpose(1, 2)) + t[:, None, :]
        
        # Add the fixed pivot back
        qp = qp + m0.unsqueeze(0).unsqueeze(0)  # [B,N,3]
        
        # Objective and loss
        scores = objective_batched(qp)  # [B], maximize
        loss = -scores.sum()  # minimize negative total
        loss.backward()
        
        # Translation increment cap
        t_prev = t.detach().clone()
        opt.step()
        if per_step_t_cap_voxels is not None:
            with torch.no_grad():
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                delta = t - t_prev
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scale = torch.clamp(cap / norms, max=1.0)
                t.copy_(t_prev + delta * scale)
        
        if it == 1 or it % print_every == 0 or it == steps:
            gT = t.grad.norm().item() if t.grad is not None else 0.0
            gW = w.grad.norm().item() if w.grad is not None else 0.0
            s_max = scores.max().item()
            s_mean = scores.mean().item()
            print(f"[DEBUG] Iter {it:02d}/{steps} best={s_max:+.6f} mean={s_mean:+.6f} "
                  f"||grad_T||={gT:.6f} ||grad_w||={gW:.6f}")
        
        if return_all:
            traj.append({
                "it": it,
                "scores": scores.detach().clone(),
                "t": t.detach().clone(),
                "w": w.detach().clone(),
            })
    
    # --------- Finalize and pick best ---------
    with torch.no_grad():
        R_delta = rodrigues_batch(w)  # [B,3,3]
        R_best_all = torch.matmul(R_delta, R_inits)  # [B,3,3]
        Xb = Xc.unsqueeze(0).expand(B, -1, -1)
        qp_final = torch.matmul(Xb, R_best_all.transpose(1, 2)) + t[:, None, :] + m0.unsqueeze(0).unsqueeze(0)
        
        final_scores = objective_batched(qp_final)  # [B]
        best_idx = torch.argmax(final_scores)
        
        # Compose outputs
        aligned_coords_best = qp_final[best_idx]  # [N,3]
        R_final_best = R_best_all[best_idx]  # [3,3]
        T_full_all = T_init + t  # [B,3]
        T_full_best = T_full_all[best_idx]  # [3]
        
        T_global_all = torch.einsum('i,bij->bj', (T_init - m0), R_best_all.transpose(1, 2)) + m0 + t  # [B,3]
        T_global_best = T_global_all[best_idx]
        
        out = {
            "best_aligned_coords": aligned_coords_best,  # [N,3]
            "best_R": R_final_best,  # [3,3]
            "best_T": T_full_best,  # [3]
            "best_score": final_scores[best_idx].item(),
            "best_batch_index": int(best_idx.item()),
            "all_scores": final_scores.detach(),  # [B]
            "all_R": R_best_all.detach(),  # [B,3,3]
            "all_T": T_full_all.detach(),  # [B,3]
            "T_global_best": T_global_best.detach(),  # [3]
        }
        if return_all:
            out["trajectory"] = traj
        return out


def calculate_ESP_at_lattice_points_keops(
    coords: torch.Tensor,        # [N,3]
    lattice_coords: torch.Tensor, # [M,3]
    elements: torch.Tensor,      # [N]
    bfactor: torch.Tensor,       # [N]
    voxel_size: float,
    rmax: float = 5.0,
    use_Coloumb: bool = True,    # Default to True for PyKeOps efficiency
    device: torch.device = None
) -> torch.Tensor:
    """
    PyKeOps-based ESP calculation at lattice points for scalable density alignment.
    
    Uses the same PyKeOps implementation as calculate_ESP() with use_Coloumb=True
    for maximum efficiency and scalability with large protein/mask combinations.
    
    Returns:
        torch.Tensor: [M] ESP values at lattice points
    """
    from pykeops.torch import LazyTensor
    
    if device is None:
        device = coords.device
    
    coords = coords.to(device)
    lattice_coords = lattice_coords.to(device) 
    elements = elements.to(device).to(torch.float32)  # PyKeOps needs float32
    bfactor = bfactor.to(device)
    
    N = coords.shape[0]
    M = lattice_coords.shape[0]
    
    # Create LazyTensors for memory-efficient computation
    lattice_i = LazyTensor(lattice_coords[:, None, :])  # [M, 1, 3]
    coords_j = LazyTensor(coords[None, :, :])           # [1, N, 3]
    
    # Distance squared calculation
    D_ij = ((lattice_i - coords_j) ** 2).sum(dim=2, keepdim=True)  # [M, N, 1]
    
    if use_Coloumb:
        # Use same Coulomb potential as calculate_ESP function
        sigmas_squared_j = LazyTensor((bfactor / (8*torch.pi**2)).reshape(1, N, 1))  # [1, N, 1]
        elements_j = LazyTensor(elements.reshape(1, N, 1))  # [1, N, 1]
        
        # Apply distance cutoff mask efficiently
        cutoff_mask = (D_ij < rmax**2).float()
        
        # Coulomb-style potential with B-factor Gaussian
        esp_values = (
            elements_j * (1 / (2 * torch.pi * sigmas_squared_j)).power(3/2) * 
            (- D_ij / (2 * sigmas_squared_j)).exp() * cutoff_mask
        ).sum(1)  # Sum over atoms, result: [M, 1]
        
    else:
        # Simple electron density (less efficient without PyKeOps optimization)
        elements_j = LazyTensor(elements.reshape(1, N, 1))
        bfactor_j = LazyTensor(bfactor.reshape(1, N, 1))
        
        cutoff_mask = (D_ij < rmax**2).float()
        bfac_factor = (-bfactor_j * D_ij / (8 * torch.pi**2)).exp()
        
        esp_values = (
            elements_j * (-D_ij).exp() * bfac_factor * cutoff_mask
        ).sum(1)  # [M, 1]
    
    return esp_values.squeeze(-1)  # [M]


def calculate_ESP_at_lattice_points(
    coords: torch.Tensor,        # [N,3]
    lattice_coords: torch.Tensor, # [M,3]
    elements: torch.Tensor,      # [N]
    bfactor: torch.Tensor,       # [N]
    voxel_size: float,
    rmax: float = 5.0,
    use_Coloumb: bool = False,
    device: torch.device = None
) -> torch.Tensor:
    """
    Fallback ESP calculation using standard PyTorch operations.
    Less memory efficient but works without PyKeOps.
    """
    if device is None:
        device = coords.device
    
    coords = coords.to(device)
    lattice_coords = lattice_coords.to(device)
    elements = elements.to(device)
    bfactor = bfactor.to(device)
    
    # Distance calculation: [M, N]
    distances = torch.cdist(lattice_coords, coords)  # [M, N]
    
    # Apply distance cutoff for efficiency
    mask = distances <= rmax  # [M, N]
    
    # B-factor contribution: exp(-B * r^2 / (8*pi^2))
    bfac_factor = torch.exp(-bfactor[None, :] * distances**2 / (8 * torch.pi**2))  # [M, N]
    
    if use_Coloumb:
        # Coulomb potential: Z * Gaussian
        esp_values = elements[None, :] * (1 / (2 * torch.pi * (bfactor[None, :] / (8*torch.pi**2))))**(3/2) * bfac_factor
    else:
        # Simple electron density: Z * exp(-r^2)
        esp_values = elements[None, :] * torch.exp(-distances**2) * bfac_factor
    
    # Apply distance mask
    esp_values = esp_values * mask.float()  # [M, N]
    
    # Sum over atoms
    result = esp_values.sum(dim=1)  # [M]
    
    return result


def batched_density_se3_align_adam_multi_start(
    coords: torch.Tensor,                     # [B_ensembles, N, 3] ensemble coordinates
    lattice_coords_3d: torch.Tensor,          # [D,D,D,3] or [D^3,3] lattice coordinates
    volume: torch.Tensor,                     # [D,D,D] observed density (fo)
    mask3d: torch.Tensor,                     # [D^3] or [D,D,D] density mask
    elements: torch.Tensor,                   # [B_ensembles, N] element types for each ensemble
    b_factors: torch.Tensor,                  # [B_ensembles, N] b-factors for each ensemble
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,
    lr_r_deg: float = 1.0,
    reduction: str = "mean",
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0,
    Bfac: float = 300.0,
    n_random: int = 16,
    seed: int | None = None,
    return_all: bool = False,
    time: float = 0.0,
    bfactor_minimum: float = 100.0,
    t_init_box_edge_voxels: float | None = None,
):
    """
    Batched density-based SE3 alignment using your optimized compute_elden_for_density_calculation_batched approach.
    
    Two-stage optimization for gradient stability:
    - First (steps-5) iterations: Use smoothed volume and compensated protein b-factors  
    - Last 5 iterations: Use original volume and original protein b-factors
    
    This function aligns the ENTIRE ENSEMBLE of proteins to create the full density volume Fc that matches
    the observed density. The ensemble is first pre-aligned using Kabsch alignment to the first protein,
    then SE3 optimization is performed to find the best global pose.
    
    Args:
        coords: [B_ensembles, N, 3] ensemble protein coordinates
        lattice_coords_3d: [D,D,D,3] or [D^3,3] lattice coordinates  
        volume: [D,D,D] observed density map (fo)
        mask3d: [D^3] or [D,D,D] density mask
        elements: [B_ensembles, N] element types for each ensemble
        b_factors: [B_ensembles, N] b-factors for each ensemble
        
    Returns:
        Dict with best alignment results, including pre-alignment transformations
    """
    from src.utils.peng_model import ScatteringAttributes
    from pykeops.torch import LazyTensor
    from src.protenix.metrics.rmsd import self_aligned_rmsd
    from src.losses.em_loss_function import apply_bfactor_to_map
    
    device = volume.device
    dtype = volume.dtype
    
    # --------- Validate input shapes ----------
    assert coords.dim() == 3, f"coords should be [B_ensembles, N, 3], got {coords.shape}"
    B_ensembles, N_atoms, _ = coords.shape
    assert elements.shape == (B_ensembles, N_atoms), f"elements should be [{B_ensembles}, {N_atoms}], got {elements.shape}"
    assert b_factors.shape == (B_ensembles, N_atoms), f"b_factors should be [{B_ensembles}, {N_atoms}], got {b_factors.shape}"
    
    # --------- Pre-align ensemble to first protein using Kabsch (no gradients) ----------
    coords_ensemble = coords.to(device=device, dtype=dtype)
    R_prealign_list = []
    T_prealign_list = []
    
    # Compute pre-alignment transformations WITHOUT gradients
    with torch.no_grad():
        # Keep first protein as reference
        R_prealign_list.append(torch.eye(3, device=device, dtype=dtype))
        T_prealign_list.append(torch.zeros(3, device=device, dtype=dtype))  # [3] to match flattened output
        
        # Align all other proteins to the first one using Kabsch
        for i in range(1, B_ensembles):
            _, _, R_i, T_i = self_aligned_rmsd(
                coords_ensemble[i:i+1].detach(),  # [1, N, 3] - DETACHED for Kabsch
                coords_ensemble[0:1].detach(),    # [1, N, 3] - DETACHED reference
                atom_mask=torch.ones(N_atoms, device=device, dtype=torch.bool)
            )
            R_prealign_list.append(R_i[0])  # [3, 3]
            # T_i has shape [..., 3] where ... matches input batch dims [1]
            # We need to flatten it to [3] to match the reference tensor
            T_prealign_list.append(T_i.reshape(-1))  # Flatten to [3]
    
    R_prealign = torch.stack(R_prealign_list).detach()  # [B_ensembles, 3, 3] - DETACHED
    T_prealign = torch.stack(T_prealign_list).detach()  # [B_ensembles, 3] - DETACHED
    
    # Apply pre-alignment transformations to original coords WITH gradients
    # This preserves gradient flow through the original coordinates
    coords_prealigned = torch.zeros_like(coords_ensemble)
    coords_prealigned[0] = coords_ensemble[0]  # First protein unchanged
    
    for i in range(1, B_ensembles):
        # Apply pre-alignment transformation: coords_aligned = coords @ R.T + T
        coords_prealigned[i] = coords_ensemble[i] @ R_prealign[i].T + T_prealign[i]
    
    print(f"Pre-aligned ensemble: {B_ensembles} proteins to first reference")
    
    # --------- Prepare map, mask, and density points (same logic as blob function) ----------
    vol = volume.to(device=device, dtype=dtype).contiguous().detach()

    # Apply same mask smoothing as blob function
    # Ensure mask is float and on correct device
    mask3d_float = mask3d.to(device=device, dtype=vol.dtype)
    
    # More robust shape handling
    if mask3d_float.dim() == 3:
        mv = mask3d_float.unsqueeze(0).unsqueeze(0)  # [1,1,D,D,D]
    elif mask3d_float.dim() == 1:
        # If flattened, reshape to 3D first
        mask3d_float = mask3d_float.reshape(D, D, D)
        mv = mask3d_float.unsqueeze(0).unsqueeze(0)  # [1,1,D,D,D]
    else:
        mv = mask3d_float.reshape(1,1,D,D,D)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mask_vol = mv.squeeze(0).squeeze(0).contiguous()
    mx = mask_vol.max()
    if mx > 0: mask_vol = mask_vol / mx
    mask_vol = mask_vol.detach()                                      # [D,D,D]

    # Lattice coordinates (same as blob function)
    if lattice_coords_3d.dim() == 4:
        lattice_flat = lattice_coords_3d.reshape(-1, 3).to(device=device, dtype=dtype)
    else:
        lattice_flat = lattice_coords_3d.to(device=device, dtype=dtype).reshape(-1, 3)
    
    mask_flat = (mask_vol.reshape(-1) > 0.5)
    density_pts = lattice_flat[mask_flat].reshape(1, -1, 3)  # [1,M,3]
    
    # --------- PCA frame and initial rotations ----------
    # Use the centroid of the pre-aligned ensemble for PCA
    ensemble_centroid = coords_prealigned.mean(dim=(0,1), keepdim=True)  # [1, 1, 3]
    protein_1xNx3 = coords_prealigned.reshape(1, -1, 3)  # [1, 1, 3]
    
    R_pca = _pca_rotation_for_protein_vs_density(protein_1xNx3, density_pts)
    U_ref = _pca_axes_from_points(density_pts)
    Rx, Ry, Rz = rot_pi_about_pca_axes(U_ref)
    
    R_rand = random_rotation_matrices_haar(n_random, device=device, dtype=dtype, seed=seed)
    R_inits = torch.cat([
        R_pca.unsqueeze(0),
        torch.stack([R_pca @ Rx, R_pca @ Ry, R_pca @ Rz,
                    Rx @ R_pca, Ry @ R_pca, Rz @ R_pca], dim=0),
        R_rand @ R_pca,
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    ], dim=0)
    B = R_inits.shape[0]
    
    # --------- Pre-centering for optimization ----------
    # Density-weighted centroid (weighted by fo values, not just mask)
    masked_lattice = lattice_flat[mask_flat]  # [M, 3]
    masked_fo = vol.reshape(-1)[mask_flat]    # [M] - density values at masked points
    
    # Shift density values to be non-negative for proper weighting
    masked_fo_positive = masked_fo - masked_fo.min()
    blob_centroid = (masked_lattice * masked_fo_positive.unsqueeze(-1)).sum(dim=0) / masked_fo_positive.sum()
    
    ensemble_centroid_flat = coords_prealigned.mean(dim=(0,1))  # [3]
    T_init = (blob_centroid - ensemble_centroid_flat).detach()
    
    # Center the pre-aligned ensemble
    coords_centered = coords_prealigned - ensemble_centroid_flat.unsqueeze(0).unsqueeze(0)  # [B_ensembles, N, 3]
    
    # --------- Prepare observed density ----------
    # B-factor smoothing for gradient stability (first steps-5 iterations)
    # Bfactor decreases with time (annealing)
    Bfac_annealed = bfactor_minimum + (1.0 - time) * (Bfac - bfactor_minimum)
    
    # Create smoothed version of the volume for initial optimization
    vol_smoothed = apply_bfactor_to_map(vol, voxel_size, Bfac_annealed, device)
    
    # Extract masked observed density (original and smoothed versions)
    # Use same masking as mask_flat to ensure consistent shapes
    fo_masked_original = vol.reshape(-1)[mask_flat]  # [M] - original for final steps
    fo_masked_smoothed = vol_smoothed.reshape(-1)[mask_flat]  # [M] - smoothed for initial steps
    
    # Normalize observed densities (same as in guidance loss)
    fo_normalized_original = (fo_masked_original - fo_masked_original.mean()) / (fo_masked_original.std() + 1e-6)
    fo_normalized_smoothed = (fo_masked_smoothed - fo_masked_smoothed.mean()) / (fo_masked_smoothed.std() + 1e-6)
    
    # --------- Prepare elements and b-factors for batched calculation ----------
    elements_tensor = elements.to(device=device)  # [B_ensembles, N]
    b_factors_tensor = b_factors.to(device=device)  # [B_ensembles, N]
    
    # Masked lattice coordinates for density calculation
    lattice_masked = lattice_flat[mask_flat]  # [M, 3]
    
    def compute_elden_for_density_calculation_batched( 
        D, 
        lattice, # Shape (M_voxels, 3)
        atom_positions, # Shape (B_rotations, B_ensembles, N_atoms, 3) 
        atom_identities, # Shape (B_ensembles, N_atoms)
        b_factors, # Shape (B_ensembles, N_atoms)
        device
    ) -> torch.Tensor: # Shape (B_rotations, M_voxels)
        
        B_rotations = atom_positions.shape[0]
        B_ensembles = atom_positions.shape[1]
        N_atoms = atom_positions.shape[2]

        # Repeating b-factors and atom identities over rotations 
        b_factors_repeated = b_factors.repeat(B_rotations, 1, 1) # Shape (B_rotations, B_ensembles, N_atoms)
        atom_identities_repeated = atom_identities.repeat(B_rotations, 1, 1) # Shape (B_rotations, B_ensembles, N_atoms)

        # Squeezing the ensemble locations etc. into one stack
        atoms_squeezed = atom_positions.reshape(B_rotations, -1, 3) # Shape (B_rotations, N_atoms * B_ensembles, 3)
        bfactors_squeezed = b_factors_repeated.reshape(B_rotations, -1) # Shape (B_rotations, N_atoms * B_ensembles)
        elements_squeezed = atom_identities_repeated.reshape(B_rotations, -1) # Shape (B_rotations, N_atoms * B_ensembles)
        
        # Lazy pairwise distance calculation
        lattice_i = LazyTensor(lattice[:, None, :]) # Shape (M_voxels, 1, 3)
        atom_positions_j = LazyTensor(atoms_squeezed[:, None, :, :]) # Shape (B_rotations, N_atoms * B_ensembles, 1, 3)
        D_ij = ((lattice_i - atom_positions_j) ** 2).sum(dim=3) # Shape (M_voxels, B_rotations, N_atoms * B_ensembles)
        
        # Preparing scattering attributes
        scattering_attributes = ScatteringAttributes(device)
        gaussian_amplitudes, gaussian_widths = scattering_attributes(elements_squeezed)
        gaussian_widths = 1 / (gaussian_widths + bfactors_squeezed.unsqueeze(-1))
        a_jk = LazyTensor(gaussian_amplitudes[:, None, :, :]) # Shape (B_rotations, N_atoms * B_ensembles, 1, Kparam)
        b_jk = LazyTensor(gaussian_widths[:, None, :, :]) # Shape (B_rotations, N_atoms * B_ensembles, 1, Kparam)
        
        vol = (
            a_jk * (4 * torch.pi)**(3/2) * b_jk**(3/2) * \
            (-4 * torch.pi**2 * D_ij * b_jk).exp() # The shape is (M_voxels, B_rotations, N_atoms * B_ensembles, Kparam)
        ).sum(dim=-1).sum(2).squeeze(-1)
        
        return vol / B_ensembles # averaging out the forward model over ensemble...!
        # Shape (B_rotations, M_voxels)
    
    def objective_batched(ensemble_coords_batch: torch.Tensor, use_smoothed_density: bool = True, protein_bfactor_compensation: float = 0.0) -> torch.Tensor:
        """
        ensemble_coords_batch: [B_rotations, B_ensembles, N, 3] ensemble coordinates for each rotation
        use_smoothed_density: whether to use smoothed or original observed density
        protein_bfactor_compensation: additional b-factor to add to protein atoms when volume is smoothed
        returns: [B_rotations] scores to MAXIMIZE (negative L1 loss)
        """
        # Apply b-factor compensation to protein atoms if needed
        if protein_bfactor_compensation > 0.0:
            b_factors_compensated = b_factors_tensor + protein_bfactor_compensation
        else:
            b_factors_compensated = b_factors_tensor
            
        # Calculate densities using your optimized batched function
        fc_predicted_batch = compute_elden_for_density_calculation_batched(
            D=D,
            lattice=lattice_masked,  # [M, 3]
            atom_positions=ensemble_coords_batch,  # [B_rotations, B_ensembles, N, 3]
            atom_identities=elements_tensor,  # [B_ensembles, N]
            b_factors=b_factors_compensated,  # [B_ensembles, N] - with optional compensation
            device=device
        )  # [B_rotations, M]
        
        # Choose which observed density to use
        fo_target = fo_normalized_smoothed if use_smoothed_density else fo_normalized_original
        
        # Normalize each predicted density and compute L1 loss
        batch_scores = []
        for b in range(fc_predicted_batch.shape[0]):
            fc_predicted = fc_predicted_batch[b]  # [M]
            
            # Normalize predicted density (same as in guidance loss)
            fc_normalized = (fc_predicted - fc_predicted.mean()) / (fc_predicted.std() + 1e-6)
            
            # L1 loss (same as guidance loss) - minimize, so negate for maximization
            l1_loss = (0.5 * (fo_target - fc_normalized).abs()).mean()
            score = -l1_loss  # Negate to maximize
            batch_scores.append(score)
        
        return torch.stack(batch_scores)  # [B_rotations]
    
    # --------- Learnable batch params ---------
    if t_init_box_edge_voxels is None:
        t_init_box_edge_voxels = 1.0
    
    t_inits = (torch.rand(B, 3, device=device, dtype=dtype) - 0.5) * 2.0 * t_init_box_edge_voxels
    t = torch.nn.Parameter(t_inits + T_init)
    w = torch.nn.Parameter(torch.zeros(B, 3, device=device, dtype=dtype))
    opt = torch.optim.SGD(
        [{"params": [t], "lr": lr_t_A},
         {"params": [w], "lr": math.radians(lr_r_deg)}],
        momentum=0.0
    )
    
    print(f"[DEBUG] Batched density align start: B={B} starts | steps={steps}, lr_t_A={lr_t_A:.4f}Å, lr_r_deg={lr_r_deg:.2f}°")
    
    traj = [] if return_all else None
    
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        
        # Compose current rotations with init rotations
        R_delta = rodrigues_batch(w)  # [B,3,3]
        R_curr = torch.matmul(R_delta, R_inits)  # [B,3,3]
        
        # Transform ensemble: apply same rotation and translation to all proteins in ensemble
        # coords_centered: [B_ensembles, N, 3]
        # We need to expand to [B_rotations, B_ensembles, N, 3]
        coords_batch = coords_centered.unsqueeze(0).expand(B, -1, -1, -1)  # [B_rotations, B_ensembles, N, 3]
        
        # Apply rotation to each ensemble member: X' = X @ R.T + T
        ensemble_transformed = torch.einsum('bijk,bkl->bijl', coords_batch, R_curr.transpose(-1, -2))  # [B_rotations, B_ensembles, N, 3]
        ensemble_transformed = ensemble_transformed + t[:, None, None, :]  # Add translation
        
        # Add back the ensemble centroid
        ensemble_transformed = ensemble_transformed + ensemble_centroid_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Two-stage optimization: smoothed for first (steps-5) iterations, original for last 5
        use_smoothed = it <= (steps - 5)
        protein_bfactor_compensation = Bfac_annealed if use_smoothed else 0.0
        
        # Print transition message when switching from smoothed to original density
        if it == (steps - 4) and steps > 5:  # First iteration using original density
            print(f"[DEBUG] *** Switching to original density for final {5} iterations ***")
        
        # Objective and loss
        scores = objective_batched(ensemble_transformed, use_smoothed_density=use_smoothed, protein_bfactor_compensation=protein_bfactor_compensation)  # [B_rotations], maximize
        loss = -scores.sum()  # minimize negative total
        loss.backward()
        
        # Translation increment cap
        t_prev = t.detach().clone()
        opt.step()
        if per_step_t_cap_voxels is not None:
            with torch.no_grad():
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                delta = t - t_prev
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scale = torch.clamp(cap / norms, max=1.0)
                t.copy_(t_prev + delta * scale)
        
        if it == 1 or it % print_every == 0 or it == steps:
            gT = t.grad.norm().item() if t.grad is not None else 0.0
            gW = w.grad.norm().item() if w.grad is not None else 0.0
            s_max = scores.max().item()
            s_mean = scores.mean().item()
            print(f"[DEBUG] Iter {it:02d}/{steps} best={s_max:+.6f} mean={s_mean:+.6f} "
                  f"||grad_T||={gT:.6f} ||grad_w||={gW:.6f}")
        
        if return_all:
            traj.append({
                "it": it,
                "scores": scores.detach().clone(),
                "t": t.detach().clone(),
                "w": w.detach().clone(),
            })
    
    # --------- Finalize and pick best ---------
    # Compute final transformations (these can be detached since they're just for selection)
    with torch.no_grad():
        R_delta = rodrigues_batch(w)  # [B,3,3]
        R_best_all = torch.matmul(R_delta, R_inits)  # [B,3,3]
        
        # Transform final ensemble coordinates for scoring only
        coords_batch_final = coords_centered.unsqueeze(0).expand(B, -1, -1, -1)  # [B_rotations, B_ensembles, N, 3]
        ensemble_final = torch.einsum('bijk,bkl->bijl', coords_batch_final, R_best_all.transpose(-1, -2))  # [B_rotations, B_ensembles, N, 3]
        ensemble_final = ensemble_final + t[:, None, None, :]  # Add translation
        ensemble_final = ensemble_final + ensemble_centroid_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add back centroid
        
        # Final scoring uses original (unsmoothed) density and no b-factor compensation
        final_scores = objective_batched(ensemble_final, use_smoothed_density=False, protein_bfactor_compensation=0.0)  # [B_rotations]
        best_idx = torch.argmax(final_scores)
        
        # Best transformation parameters (detached for composition)
        R_optimal = R_best_all[best_idx].detach()  # [3,3]
        T_optimal = (T_init + t[best_idx]).detach()  # [3]
    
    # Apply best transformation to original coordinates WITH gradients
    # This ensures gradient flow through x_0_hat is preserved
    R_delta_best = rodrigues_batch(w[best_idx:best_idx+1])  # [1,3,3]
    R_best = torch.matmul(R_delta_best, R_inits[best_idx:best_idx+1])  # [1,3,3]
    
    # Transform ensemble coordinates WITH gradients
    coords_batch_best = coords_centered.unsqueeze(0)  # [1, B_ensembles, N, 3]
    best_ensemble_coords = torch.einsum('bijk,bkl->bijl', coords_batch_best, R_best.transpose(-1, -2))  # [1, B_ensembles, N, 3]
    best_ensemble_coords = best_ensemble_coords + t[best_idx:best_idx+1, None, None, :]  # Add translation
    best_ensemble_coords = best_ensemble_coords + ensemble_centroid_flat.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # Add back centroid
    best_ensemble_coords = best_ensemble_coords[0]  # [B_ensembles, N, 3]
    
    # For compatibility, return the first ensemble member's coordinates as "best_aligned_coords"
    aligned_coords_best = best_ensemble_coords[0]  # [N, 3] - first ensemble member
    
    # Compose transformations for reporting (detached)
    with torch.no_grad():
    # R_total = R_opt @ R_pre
        R_composed_all = torch.einsum('ij,bjk->bik', R_optimal, R_prealign)  # [B_ensembles, 3, 3]

        c = ensemble_centroid_flat                     # [3]
        T_shift = (T_init + t[best_idx]).detach()      # [3]

        # T_total = (T_pre - c) @ R_opt.T + c + T_shift
        T_composed_all = (T_prealign - c.unsqueeze(0)) @ R_optimal.transpose(0, 1) \
                        + c.unsqueeze(0) + T_shift.unsqueeze(0) 

        out = {
            "best_aligned_coords": aligned_coords_best,  # [N,3] - first ensemble member for compatibility
            "best_ensemble_coords": best_ensemble_coords,  # [B_ensembles, N, 3] - full ensemble
            "best_R": R_optimal,  # [3,3] - optimal rotation
            "best_T": T_optimal,  # [3] - optimal translation
            "best_score": final_scores[best_idx].item(),
            "best_batch_index": int(best_idx.item()),
            "all_scores": final_scores.detach(),  # [B_rotations]
            "all_R": R_best_all.detach(),  # [B_rotations, 3, 3]
            "all_T": (T_init.unsqueeze(0) + t).detach(),  # [B_rotations, 3]
            "T_global_best": T_optimal.detach(),  # [3] - for compatibility
            # Pre-alignment information
            "R_prealign": R_prealign.detach(),  # [B_ensembles, 3, 3]
            "T_prealign": T_prealign.detach(),  # [B_ensembles, 3]
            "R_composed": R_composed_all.detach(),  # [B_ensembles, 3, 3] - full composed transformations
            "T_composed": T_composed_all.detach(),  # [B_ensembles, 3] - full composed transformations
        }
        if return_all:
            out["trajectory"] = traj
        return out


def blob_se3_align_adam_multi_start_new_not_needed(
    coords: torch.Tensor,
    lattice_coords_3d: torch.Tensor,
    volume: torch.Tensor,
    mask3d: torch.Tensor,
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,
    lr_r_deg: float = 1.0,
    reduction: str = "mean",
    sampler_fn: Callable = None,
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0,
    Bfac: float = 300.0,
    n_random: int = 16,
    seed: int | None = None,
    return_all: bool = False,
):

    assert sampler_fn is not None, "Provide sampler_fn."

    device = volume.device
    dtype = volume.dtype

    # Preprocess map and density mask
    vol = volume.to(device=device, dtype=dtype).contiguous().detach()
    if mask3d.dim() == 3:
        mv = mask3d.to(device=device, dtype=vol.dtype).unsqueeze(0).unsqueeze(0)
    else:
        mv = mask3d.to(device=device, dtype=vol.dtype).reshape(1, 1, D, D, D)
    mv = torch.nn.functional.avg_pool3d(mv, 3, stride=1, padding=1)
    mv = torch.nn.functional.avg_pool3d(mv, 3, stride=1, padding=1)
    mask_vol = mv.squeeze(0).squeeze(0).contiguous()
    mx = mask_vol.max()
    if mx > 0:
        mask_vol = mask_vol / mx
    mask_vol = mask_vol.detach()

    # Flatten lattice + mask
    lattice_flat = lattice_coords_3d.reshape(-1, 3) if lattice_coords_3d.dim() == 4 else lattice_coords_3d.reshape(-1, 3)
    lattice_flat = lattice_flat.to(device=device, dtype=dtype)
    mask_flat = (mask_vol.reshape(-1) > 0.5)
    density_pts = lattice_flat[mask_flat].reshape(1, -1, 3)

    # Rotation initializations
    protein_1xNx3 = coords.to(device=device, dtype=dtype).reshape(1, -1, 3)
    R_pca = _pca_rotation_for_protein_vs_density(protein_1xNx3, density_pts)
    U_ref = _pca_axes_from_points(density_pts)
    Rx, Ry, Rz = rot_pi_about_pca_axes(U_ref)
    R_rand = random_rotation_matrices_haar(n_random, device=device, dtype=dtype, seed=seed)

    R_inits = torch.cat([
        R_pca.unsqueeze(0),
        torch.stack([R_pca @ Rx, R_pca @ Ry, R_pca @ Rz,
                     Rx @ R_pca, Ry @ R_pca, Rz @ R_pca]),
        R_rand  # << NOT @ R_pca anymore (pure randoms)
    ], dim=0)
    B = R_inits.shape[0]

    # Center coords
    X = coords.to(device=device, dtype=dtype).detach()
    coords_centroid = X.mean(dim=0)
    blob_centroid = lattice_flat[mask_flat].mean(dim=0)
    T_init = (blob_centroid - coords_centroid).detach()

    X_shifted = (X + T_init).detach()
    m0 = X_shifted.mean(0).detach()
    Xc = (X_shifted - m0).contiguous()  # [N,3]

    masked_map = (vol * mask_vol).contiguous().detach()
    masked_map = apply_bfactor_to_map(masked_map, voxel_size, Bfac, device)

    # Objective
    def objective_batched(qp: torch.Tensor) -> torch.Tensor:
        vals = sampler_fn(
            lattice_coords_3d=lattice_coords_3d.detach(),
            voxel_size=float(voxel_size), D=int(D),
            volume=masked_map, query_points=qp
        )
        return vals.mean(dim=-1) if reduction == "mean" else vals.sum(dim=-1)

    # --- Manual optimizer
    t = torch.zeros(B, 3, device=device, dtype=dtype)
    w = torch.zeros(B, 3, device=device, dtype=dtype)

    print(f"[DEBUG] Align start (batched): B={B} | steps={steps} | lr_t={lr_t_A:.4f}Å | lr_r={lr_r_deg:.2f}°")

    for it in range(1, steps + 1):
        t.requires_grad_(True)
        w.requires_grad_(True)

        R_delta = rodrigues_batch(w)                        # [B,3,3]
        R_curr = torch.matmul(R_delta, R_inits)             # [B,3,3]
        Xb = Xc.unsqueeze(0).expand(B, -1, -1)              # [B,N,3]
        qp = torch.matmul(Xb, R_curr.transpose(1, 2)) + t[:, None, :] + m0.unsqueeze(0).unsqueeze(0)

        scores = objective_batched(qp)                      # [B]
        loss = -scores.sum()
        grads = torch.autograd.grad(loss, [t, w])

        with torch.no_grad():
            t = t - lr_t_A * grads[0]
            w = w - math.radians(lr_r_deg) * grads[1]

            # Optional per-step cap
            if per_step_t_cap_voxels is not None:
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                delta = lr_t_A * grads[0]
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scale = torch.clamp(cap / norms, max=1.0)
                t = t + delta * (scale - 1.0)

        if it == 1 or it % print_every == 0 or it == steps:
            print(f"[DEBUG] Iter {it:02d}/{steps} best={scores.max():+.6f} mean={scores.mean():+.6f} "
                  f"||grad_T||={grads[0].norm():.6f} ||grad_w||={grads[1].norm():.6f} "
                  f"|T|max={t.norm(dim=-1).max():.6f}Å |w|max={(w.norm(dim=-1).max().item() * 180.0 / math.pi):.6f}°")

    # Final eval
    with torch.no_grad():
        R_delta = rodrigues_batch(w)
        R_best_all = torch.matmul(R_delta, R_inits)         # [B,3,3]
        Xb = Xc.unsqueeze(0).expand(B, -1, -1)
        qp_final = torch.matmul(Xb, R_best_all.transpose(1,2)) + t[:,None,:] + m0.unsqueeze(0).unsqueeze(0)

        final_scores = objective_batched(qp_final)
        best_idx = torch.argmax(final_scores)

        aligned_coords_best = qp_final[best_idx]
        R_final_best = R_best_all[best_idx]
        T_full_all = T_init + t
        T_full_best = T_full_all[best_idx]
        T_global_all = torch.einsum('i,bij->bj', (T_init - m0), R_best_all.transpose(1,2)) + m0 + t
        T_global_best = T_global_all[best_idx]

        out = {
            "best_aligned_coords": aligned_coords_best,
            "best_R": R_final_best,
            "best_T": T_full_best,
            "best_score": final_scores[best_idx].item(),
            "best_batch_index": int(best_idx.item()),
            "all_scores": final_scores.detach(),
            "all_R": R_best_all.detach(),
            "all_T": T_full_all.detach(),
            "T_global_best": T_global_best.detach(),
        }
        if return_all:
            out["trajectory"] = None  # optional: track per-step grads
        return out


def _masked_mean(tensor: torch.Tensor, mask: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    """
    Calculates the mean over a given dimension, correctly ignoring masked-out elements.
    The mask ensures padded atoms contribute nothing to the mean calculation.
    """
    masked_tensor = tensor * mask.unsqueeze(-1)
    sum_val = torch.sum(masked_tensor, dim=dim, keepdim=keepdim)
    # CORRECTED: Set keepdim=True to maintain rank for broadcasting during division.
    # This prevents the shape mismatch error (e.g., [C, 3] / [C]).
    count = torch.sum(mask, dim=dim, keepdim=True).clamp(min=1e-9)
    return sum_val / count

# --- Core Function 2: The Batched Gradient Descent ---

# --- Core Function 2: The Batched Gradient Descent (with Memory Fix) ---
def run_batched_alignment(
    coords_list: List[torch.Tensor],
    density_masks_list: List[torch.Tensor],
    lattice_coords_3d: torch.Tensor,
    volume: torch.Tensor,
    voxel_size: float,
    D: int,
    sampler_fn: Callable,
    steps: int = 30,
    lr_t_A: float = 100.0,
    lr_r_deg: float = 1000.0,
    reduction: str = "mean",
    per_step_t_cap_voxels: float = 3.0,
    Bfac: float = 150.0,
    n_random: int = 5_000,
    seed: int | None = None,
    start_batch_size: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Internal function that performs the multi-start SE(3) alignment on a batch of chains.
    This version processes random starts in mini-batches to prevent memory explosion.
    """
    device = volume.device
    dtype = volume.dtype
    C = len(coords_list)

    # 1. PAD coordinates and STACK masks into batched tensors
    coords_b = torch.nn.utils.rnn.pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    atom_mask_b = torch.arange(coords_b.shape[1], device=device)[None, :] < torch.tensor([len(c) for c in coords_list], device=device)[:, None]
    mask3d_b = torch.stack(density_masks_list, dim=0)
    
    # 2. BATCHED Pre-Centering (your original logic, but vectorized)
    coords_centroid_b = _masked_mean(coords_b, atom_mask_b, dim=1)
    lattice_flat = lattice_coords_3d.reshape(-1, 3)
    mask_flat_b = (mask3d_b.reshape(C, -1) > 0.5)
    blob_centroid_b = _masked_mean(lattice_flat.unsqueeze(0).expand(C, -1, -1), mask_flat_b, dim=1)
    T_init_b = (blob_centroid_b - coords_centroid_b).detach()
    X_shifted_b = coords_b + T_init_b.unsqueeze(1)
    m0_b = _masked_mean(X_shifted_b, atom_mask_b, dim=1)
    Xc_b = (X_shifted_b - m0_b.unsqueeze(1)) * atom_mask_b.unsqueeze(-1)

    # 3. BATCHED Map Preparation (Done once)
    mv = mask3d_b.to(dtype=volume.dtype).unsqueeze(1)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mask_vol_b = mv.squeeze(1)
    mx = mask_vol_b.view(C, -1).max(dim=-1, keepdim=True)[0].view(C, 1, 1, 1)
    mask_vol_b = (mask_vol_b / mx.clamp(min=1e-9)).detach()
    masked_map_b = (volume.unsqueeze(0) * mask_vol_b).detach()
    masked_map_b = apply_bfactor_to_map(masked_map_b, voxel_size, Bfac, device)

    # 4. Initialize tensors to store the best results found across all mini-batches
    overall_best_scores = torch.full((C,), -float('inf'), device=device, dtype=dtype)
    overall_best_R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(C, -1, -1).clone()
    overall_best_final_t = torch.zeros(C, 3, device=device, dtype=dtype)

    # 5. Process random starts in manageable mini-batches to avoid OOM
    num_batches = (n_random + start_batch_size - 1) // start_batch_size

    for i in range(num_batches):
        current_batch_size = min(start_batch_size, n_random - i * start_batch_size)
        
        B_starts = current_batch_size
        B = C * B_starts

        # Expand tensors for the current mini-batch of starts
        R_inits = random_rotation_matrices_haar(B, device=device, dtype=dtype)
        Xc_B = Xc_b.unsqueeze(1).expand(-1, B_starts, -1, -1).reshape(B, -1, 3)
        atom_mask_B = atom_mask_b.unsqueeze(1).expand(-1, B_starts, -1).reshape(B, -1)
        m0_B = m0_b.unsqueeze(1).expand(-1, B_starts, -1).reshape(B, 3)
        
        # CORRECTED: Use repeat_interleave to create a memory-safe batch of maps for the current chunk.
        # This replaces the unsafe .expand() operation.
        masked_map_B = masked_map_b.repeat_interleave(B_starts, dim=0)

        # Define objective for this mini-batch
        def objective_batched(qp: torch.Tensor, atom_mask: torch.Tensor) -> torch.Tensor:
            vals = sampler_fn(lattice_coords_3d=lattice_coords_3d, voxel_size=voxel_size, D=D, volume=masked_map_B, query_points=qp)
            # Note: _masked_mean now requires keepdim=True for the count
            if reduction == "mean":
                vals_masked = vals * atom_mask
                sum_vals = vals_masked.sum(dim=-1)
                count = atom_mask.sum(dim=-1).clamp(min=1e-9)
                return sum_vals / count
            else:
                return (vals * atom_mask).sum(dim=-1)

        # Run optimization for the mini-batch
        t = torch.nn.Parameter(torch.zeros(B, 3, device=device, dtype=dtype))
        w = torch.nn.Parameter(torch.zeros(B, 3, device=device, dtype=dtype))
        opt = torch.optim.SGD([{"params":[t], "lr": lr_t_A}, {"params":[w], "lr": math.radians(lr_r_deg)}])

        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            R_delta = rodrigues_batch(w)
            R_curr = torch.bmm(R_delta, R_inits)
            qp = torch.bmm(Xc_B, R_curr.transpose(1, 2)) + t.unsqueeze(1) + m0_B.unsqueeze(1)
            scores = objective_batched(qp, atom_mask_B)
            loss = -scores.sum()
            loss.backward()
            opt.step()

        # Finalize and check for improvements from this mini-batch
        with torch.no_grad():
            R_delta = rodrigues_batch(w)
            R_final_all = torch.bmm(R_delta, R_inits)
            qp_final = torch.bmm(Xc_B, R_final_all.transpose(1, 2)) + t.unsqueeze(1) + m0_B.unsqueeze(1)
            final_scores = objective_batched(qp_final, atom_mask_B)
            final_scores_per_chain = final_scores.view(C, B_starts)
            
            best_scores_in_batch, best_indices_in_batch = final_scores_per_chain.max(dim=1)
            improvement_mask = best_scores_in_batch > overall_best_scores
            
            if improvement_mask.any():
                overall_best_scores[improvement_mask] = best_scores_in_batch[improvement_mask]
                flat_indices = torch.arange(C, device=device) * B_starts + best_indices_in_batch
                overall_best_R[improvement_mask] = R_final_all[flat_indices][improvement_mask]
                overall_best_final_t[improvement_mask] = t[flat_indices][improvement_mask]

    # 6. Calculate final global translation using the best results found across all batches
    with torch.no_grad():
        T_global_best = torch.einsum('ci,cij->cj', (T_init_b - m0_b), overall_best_R.transpose(1,2)) + m0_b + overall_best_final_t

    return {"best_R_b": overall_best_R, "T_global_best_b": T_global_best}





def batched_density_se3_align_adam_multi_start_new(
    coords: torch.Tensor,                     # [B_ensembles, N, 3] ensemble coordinates
    lattice_coords_3d: torch.Tensor,          # [D,D,D,3] or [D^3,3] lattice coordinates
    volume: torch.Tensor,                     # [D,D,D] observed density (fo)
    mask3d: torch.Tensor,                     # [D^3] or [D,D,D] density mask
    elements: torch.Tensor,                   # [B_ensembles, N] element types for each ensemble
    b_factors: torch.Tensor,                  # [B_ensembles, N] b-factors for each ensemble
    voxel_size: float,
    D: int,
    steps: int = 50,
    lr_t_A: float = 0.1,
    lr_r_deg: float = 1.0,
    reduction: str = "mean",
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0,
    Bfac: float = 300.0,
    n_random: int = 16,
    seed: int | None = None,
    return_all: bool = False,
    time: float = 0.0,
    bfactor_minimum: float = 100.0,
    t_init_box_edge_voxels: float | None = None,
    use_ncc: bool = False,
):
    """
    Batched density-based SE3 alignment using your optimized compute_elden_for_density_calculation_batched approach.
    
    Two-stage optimization for gradient stability:
    - First (steps-5) iterations: Use smoothed volume and compensated protein b-factors  
    - Last 5 iterations: Use original volume and original protein b-factors
    
    This function aligns the ENTIRE ENSEMBLE of proteins to create the full density volume Fc that matches
    the observed density. The ensemble is first pre-aligned using Kabsch alignment to the first protein,
    then SE3 optimization is performed to find the best global pose.

    IMPORTANT: Poses are optimized and returned per-ENSEMBLE member; the loss is computed on the FULL ensemble Fc.
    
    Args:
        coords: [B_ensembles, N, 3] ensemble protein coordinates
        lattice_coords_3d: [D,D,D,3] or [D^3,3] lattice coordinates  
        volume: [D,D,D] observed density map (fo)
        mask3d: [D^3] or [D,D,D] density mask
        elements: [B_ensembles, N] element types for each ensemble
        b_factors: [B_ensembles, N] b-factors for each ensemble
        
    Returns:
        Dict with best alignment results for the WHOLE ENSEMBLE, plus composed transforms from RAW coords.
    """
    import math
    from src.utils.peng_model import ScatteringAttributes
    from pykeops.torch import LazyTensor
    from src.protenix.metrics.rmsd import self_aligned_rmsd
    from src.losses.em_loss_function import apply_bfactor_to_map
    
    device = volume.device
    dtype = volume.dtype
    
    # --------- Validate input shapes ----------
    assert coords.dim() == 3, f"coords should be [B_ensembles, N, 3], got {coords.shape}"
    B_ensembles, N_atoms, _ = coords.shape
    assert elements.shape == (B_ensembles, N_atoms), f"elements should be [{B_ensembles}, {N_atoms}], got {elements.shape}"
    assert b_factors.shape == (B_ensembles, N_atoms), f"b_factors should be [{B_ensembles}, {N_atoms}], got {b_factors.shape}"
    
    # --------- Pre-align ensemble to first protein using Kabsch (no gradients) ----------
    coords_ensemble = coords.to(device=device, dtype=dtype)
    R_prealign_list = []
    T_prealign_list = []
    
    with torch.no_grad():
        R_prealign_list.append(torch.eye(3, device=device, dtype=dtype))
        T_prealign_list.append(torch.zeros(3, device=device, dtype=dtype))
        for i in range(1, B_ensembles):
            _, _, R_i, T_i = self_aligned_rmsd(
                coords_ensemble[i:i+1].detach(),   # [1,N,3]
                coords_ensemble[0:1].detach(),     # [1,N,3]
                atom_mask=torch.ones(N_atoms, device=device, dtype=torch.bool)
            )
            R_prealign_list.append(R_i[0])         # [3,3]
            T_prealign_list.append(T_i.reshape(-1))# [3]
    
    R_prealign = torch.stack(R_prealign_list).detach()  # [E,3,3]
    T_prealign = torch.stack(T_prealign_list).detach()  # [E,3]
    
    # Apply pre-alignment to coords WITH gradients (only as an initialization frame)
    coords_prealigned = torch.zeros_like(coords_ensemble)
    coords_prealigned[0] = coords_ensemble[0]
    for i in range(1, B_ensembles):
        coords_prealigned[i] = coords_ensemble[i] @ R_prealign[i].T + T_prealign[i]
    print(f"Pre-aligned ensemble: {B_ensembles} proteins to first reference")
    
    # --------- Prepare map, mask, and density points ----------
    vol = volume.to(device=device, dtype=dtype).contiguous().detach()

    mask3d_float = mask3d.to(device=device, dtype=vol.dtype)
    if mask3d_float.dim() == 3:
        mv = mask3d_float.unsqueeze(0).unsqueeze(0)  # [1,1,D,D,D]
    elif mask3d_float.dim() == 1:
        mask3d_float = mask3d_float.reshape(D, D, D)
        mv = mask3d_float.unsqueeze(0).unsqueeze(0)
    else:
        mv = mask3d_float.reshape(1,1,D,D,D)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mv = torch.nn.functional.avg_pool3d(mv, kernel_size=3, stride=1, padding=1)
    mask_vol = mv.squeeze(0).squeeze(0).contiguous()
    mx = mask_vol.max()
    if mx > 0: mask_vol = mask_vol / mx
    mask_vol = mask_vol.detach()  # [D,D,D]

    if lattice_coords_3d.dim() == 4:
        lattice_flat = lattice_coords_3d.reshape(-1, 3).to(device=device, dtype=dtype)
    else:
        lattice_flat = lattice_coords_3d.to(device=device, dtype=dtype).reshape(-1, 3)
    
    mask_flat = (mask_vol.reshape(-1) > 0.5)
    density_pts = lattice_flat[mask_flat].reshape(1, -1, 3)  # [1,M,3]
    
    # --------- PCA frame and initial rotations ----------
    ensemble_centroid = coords_prealigned.mean(dim=(0,1), keepdim=True)  # [1,1,3]
    protein_1xNx3 = coords_prealigned.reshape(1, -1, 3)                  # [1,E*N,3]
    
    R_pca = _pca_rotation_for_protein_vs_density(protein_1xNx3, density_pts)
    U_ref = _pca_axes_from_points(density_pts)
    Rx, Ry, Rz = rot_pi_about_pca_axes(U_ref)
    
    R_rand = random_rotation_matrices_haar(n_random, device=device, dtype=dtype, seed=seed)
    R_inits = torch.cat([
        R_pca.unsqueeze(0),
        torch.stack([R_pca @ Rx, R_pca @ Ry, R_pca @ Rz,
                     Rx @ R_pca, Ry @ R_pca, Rz @ R_pca], dim=0),
        R_rand @ R_pca,
        torch.eye(3, device=device, dtype=dtype).unsqueeze(0)
    ], dim=0)  # [B,3,3]
    B = R_inits.shape[0]
    
    # --------- Pre-centering for optimization ----------
    masked_lattice = lattice_flat[mask_flat]                      # [M,3]
    masked_fo = vol.reshape(-1)[mask_flat]                        # [M]
    masked_fo_positive = masked_fo - masked_fo.min()
    blob_centroid = (masked_lattice * masked_fo_positive.unsqueeze(-1)).sum(dim=0) / masked_fo_positive.sum()
    
    ensemble_centroid_flat = coords_prealigned.mean(dim=(0,1))    # [3]
    T_init = (blob_centroid - ensemble_centroid_flat).detach()    # [3]
    
    coords_centered = coords_prealigned - ensemble_centroid_flat.unsqueeze(0).unsqueeze(0)  # [E,N,3]
    
    # --------- Prepare observed density ----------
    Bfac_annealed = bfactor_minimum + (1.0 - time) * (Bfac - bfactor_minimum)
    vol_smoothed = apply_bfactor_to_map(vol, voxel_size, Bfac_annealed, device)
    
    fo_masked_original = vol.reshape(-1)[mask_flat]       # [M]
    fo_masked_smoothed = vol_smoothed.reshape(-1)[mask_flat]
    
    fo_normalized_original = (fo_masked_original - fo_masked_original.mean()) / (fo_masked_original.std() + 1e-6)
    fo_normalized_smoothed = (fo_masked_smoothed - fo_masked_smoothed.mean()) / (fo_masked_smoothed.std() + 1e-6)
    
    # --------- Prepare elements and b-factors ----------
    elements_tensor = elements.to(device=device)      # [E,N]
    b_factors_tensor = b_factors.to(device=device)    # [E,N]
    
    lattice_masked = lattice_flat[mask_flat]          # [M,3]
    
    def compute_elden_for_density_calculation_batched( 
        D, 
        lattice,                 # (M,3)
        atom_positions,          # (B,E,N,3) 
        atom_identities,         # (E,N)
        b_factors,               # (E,N)
        device
    ) -> torch.Tensor:           # (B,M)
        
        B_rotations = atom_positions.shape[0]
        B_ensembles = atom_positions.shape[1]
        N_atoms = atom_positions.shape[2]

        b_factors_repeated = b_factors.repeat(B_rotations, 1, 1)               # (B,E,N)
        atom_identities_repeated = atom_identities.repeat(B_rotations, 1, 1)   # (B,E,N)

        atoms_squeezed = atom_positions.reshape(B_rotations, -1, 3)            # (B, E*N, 3)
        bfactors_squeezed = b_factors_repeated.reshape(B_rotations, -1)        # (B, E*N)
        elements_squeezed = atom_identities_repeated.reshape(B_rotations, -1)  # (B, E*N)
        
        lattice_i = LazyTensor(lattice[:, None, :])                        # (M,1,3)
        atom_positions_j = LazyTensor(atoms_squeezed[:, None, :, :])       # (B, E*N,1,3)
        D_ij = ((lattice_i - atom_positions_j) ** 2).sum(dim=3)            # (M,B,E*N)
        
        scattering_attributes = ScatteringAttributes(device)
        gaussian_amplitudes, gaussian_widths = scattering_attributes(elements_squeezed)  # (B,E*N,K), (B,E*N,K)
        gaussian_widths = 1 / (gaussian_widths + bfactors_squeezed.unsqueeze(-1))
        a_jk = LazyTensor(gaussian_amplitudes[:, None, :, :])              # (B, E*N,1,K)
        b_jk = LazyTensor(gaussian_widths[:, None, :, :])                  # (B, E*N,1,K)
        
        vol = (
            a_jk * (4 * torch.pi)**(3/2) * b_jk**(3/2) *
            (-4 * torch.pi**2 * D_ij * b_jk).exp()
        ).sum(dim=-1).sum(2).squeeze(-1)                                   # (B,M)
        
        return vol / B_ensembles
    
    def objective_batched(ensemble_coords_batch: torch.Tensor, use_smoothed_density: bool = True, protein_bfactor_compensation: float = 0.0) -> torch.Tensor:
        """
        ensemble_coords_batch: [B_rotations, B_ensembles, N, 3]
        returns: [B_rotations] scores to MAXIMIZE (negative L1 loss)
        """
        if protein_bfactor_compensation > 0.0:
            b_factors_compensated = b_factors_tensor + protein_bfactor_compensation
        else:
            b_factors_compensated = b_factors_tensor
            
        fc_predicted_batch = compute_elden_for_density_calculation_batched(
            D=D,
            lattice=lattice_masked,                 # [M,3]
            atom_positions=ensemble_coords_batch,   # [B,E,N,3]
            atom_identities=elements_tensor,        # [E,N]
            b_factors=b_factors_compensated,        # [E,N]
            device=device
        )  # [B,M]
        
        fo_target = fo_normalized_smoothed if use_smoothed_density else fo_normalized_original
        

        fc_z = fc_predicted_batch - fc_predicted_batch.mean(dim=1, keepdim=True)
        fc_z = fc_z / (fc_z.std(dim=1, keepdim=True) + 1e-6)
        #batch_scores = []
        #for b in range(fc_predicted_batch.shape[0]):
        #    fc_predicted = fc_predicted_batch[b]  # [M]
        #    fc_normalized = (fc_predicted - fc_predicted.mean()) / (fc_predicted.std() + 1e-6)
        #    l1_loss = (0.5 * (fo_target - fc_normalized).abs()).mean()
        #    score = -l1_loss
        #    batch_scores.append(score)
        #
        #return torch.stack(batch_scores)  # [B]
        if use_ncc:
            # Pearson/NCC: maximize mean(z * fo)
            scores = (fc_z * fo_target.unsqueeze(0)).mean(dim=1)                                 # [B]
            return scores
        else:
            # L1 on z-scores (your current objective): maximize negative L1
            l1 = (0.5 * (fo_target.unsqueeze(0) - fc_z).abs()).mean(dim=1)                       # [B]
            return -l1
    
    # --------- Learnable per-ENSEMBLE params (NO global pose) ---------
    if t_init_box_edge_voxels is None:
        t_init_box_edge_voxels = 1.0
    
    # seed-aware rand
    if seed is not None:
        g = torch.Generator(device=device); g.manual_seed(int(seed))
        rand = lambda *s: torch.rand(*s, generator=g, device=device, dtype=dtype)
    else:
        rand = lambda *s: torch.rand(*s, device=device, dtype=dtype)
    
    # per-start, per-ensemble absolute translations in Å, initialized near T_init
    t_jitter = (rand(B, B_ensembles, 3) - 0.5) * 2.0 * t_init_box_edge_voxels * float(voxel_size)
    t_ens = torch.nn.Parameter(T_init.view(1,1,3).expand(B, B_ensembles, 3) + t_jitter)    # [B,E,3]
    
    # per-start, per-ensemble rotation deltas (axis-angle, rad), init to 0
    w_ens = torch.nn.Parameter(torch.zeros(B, B_ensembles, 3, device=device, dtype=dtype)) # [B,E,3]
    
    opt = torch.optim.Adam(
        [{"params": [t_ens], "lr": lr_t_A},
         {"params": [w_ens], "lr": math.radians(lr_r_deg)}],
        betas=(0.9, 0.999)
    )
    
    print(f"[DEBUG] Batched density align start: B={B} starts | steps={steps}, lr_t_A={lr_t_A:.4f}Å, lr_r_deg={lr_r_deg:.2f}°")
    
    traj = [] if return_all else None
    
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        
        # Compose per-ensemble rotations with init rotations (no global pose)
        R_delta_e = rodrigues_batch(w_ens.reshape(-1,3)).reshape(B, B_ensembles, 3, 3)     # [B,E,3,3]
        R_inits_be = R_inits.view(B,1,3,3).expand(B, B_ensembles, 3, 3)                     # [B,E,3,3]
        R_curr_be = torch.einsum('beij,bejk->beik', R_delta_e, R_inits_be)                  # [B,E,3,3]
        
        # Expand coords to [B,E,N,3]
        coords_batch = coords_centered.unsqueeze(0).expand(B, -1, -1, -1)                  # [B,E,N,3]
        
        # Apply per-ensemble rotation & translation: X' = X @ R^T + t_e + centroid
        ensemble_transformed = torch.einsum('benc,bedc->bend', coords_batch, R_curr_be.transpose(-1, -2))  # [B,E,N,3]
        ensemble_transformed = ensemble_transformed + t_ens[:, :, None, :]                                   # [B,E,N,3]
        ensemble_transformed = ensemble_transformed + ensemble_centroid_flat.view(1,1,1,3)
        
        # Two-stage optimization
        use_smoothed = it <= (steps - 15)
        protein_bfactor_compensation = Bfac_annealed if use_smoothed else 0.0
        
        scores = objective_batched(ensemble_transformed, use_smoothed_density=use_smoothed, protein_bfactor_compensation=protein_bfactor_compensation)  # [B]
        loss = -scores.sum()
        loss.backward()
        
        # Translation increment cap (per-ensemble)
        t_prev = t_ens.detach().clone()
        opt.step()
        if per_step_t_cap_voxels is not None:
            with torch.no_grad():
                cap = float(per_step_t_cap_voxels) * float(voxel_size)
                delta = t_ens - t_prev                                # [B,E,3]
                norms = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                scale = torch.clamp(cap / norms, max=1.0)
                t_ens.copy_(t_prev + delta * scale)
        
        if it == 1 or it % print_every == 0 or it == steps:
            gT = t_ens.grad.norm().item() if t_ens.grad is not None else 0.0
            gW = w_ens.grad.norm().item() if w_ens.grad is not None else 0.0
            s_max = scores.max().item()
            s_mean = scores.mean().item()
            print(f"[DEBUG] Iter {it:02d}/{steps} best={s_max:+.6f} mean={s_mean:+.6f} "
                  f"||grad_T||={gT:.6f} ||grad_w||={gW:.6f}")
        
        if return_all:
            traj.append({
                "it": it,
                "scores": scores.detach().clone(),
                "t_ens": t_ens.detach().clone(),
                "w_ens": w_ens.detach().clone(),
            })
    
    # --------- Finalize and pick best ---------
    with torch.no_grad():
        # Per-ensemble total rotations for all starts: R_total = R_delta_e @ R_inits
        R_delta_e = rodrigues_batch(w_ens.reshape(-1,3)).reshape(B, B_ensembles, 3, 3)    # [B,E,3,3]
        R_inits_be = R_inits.view(B,1,3,3).expand(B, B_ensembles, 3, 3)                    # [B,E,3,3]
        R_total_all = torch.einsum('beij,bejk->beik', R_delta_e, R_inits_be)               # [B,E,3,3]

        # Build transformed coords for scoring
        coords_batch_final = coords_centered.unsqueeze(0).expand(B, -1, -1, -1)           # [B,E,N,3]
        ensemble_final = torch.einsum('benc,bedc->bend', coords_batch_final, R_total_all.transpose(-1, -2))
        ensemble_final = ensemble_final + t_ens[:, :, None, :] + ensemble_centroid_flat.view(1,1,1,3)

        final_scores = objective_batched(ensemble_final, use_smoothed_density=False, protein_bfactor_compensation=0.0)  # [B]
        best_idx = torch.argmax(final_scores)

        # Best per-ensemble transforms (TOTAL): THESE ARE THE ENSEMBLE OUTPUTS
        R_optimal_be = R_total_all[best_idx].detach()   # [E,3,3]
        T_optimal_be = t_ens[best_idx].detach()         # [E,3]
    
    # Apply best transformation to original coordinates WITH gradients (per-ensemble SE(3))
    R_best = R_total_all[best_idx:best_idx+1]               # [1,E,3,3]  (keeps grads through w_ens)
    coords_batch_best = coords_centered.unsqueeze(0)        # [1,E,N,3]
    best_ensemble_coords = torch.einsum('benc,bedc->bend', coords_batch_best, R_best.transpose(-1, -2))
    best_ensemble_coords = best_ensemble_coords + t_ens[best_idx:best_idx+1, :, None, :] + ensemble_centroid_flat.view(1,1,1,3)
    best_ensemble_coords = best_ensemble_coords[0]          # [E,N,3]
    
    # Default aligned coords (first member) kept for compatibility
    aligned_coords_best = best_ensemble_coords[0]           # [N,3]
    
    # Compose transforms from RAW input coords to final pose (per-ensemble)
    with torch.no_grad():
        # R_composed[e] = R_total[e] @ R_pre[e]
        R_composed_all = torch.einsum('eij,ejk->eik', R_optimal_be, R_prealign)  # [E,3,3]
        c = ensemble_centroid_flat                                              # [3]
        # T_total[e] = (T_pre[e] - c) @ R_total[e]^T + c + T_optimal_be[e]
        T_composed_all = torch.einsum('ei,eji->ej', (T_prealign - c.unsqueeze(0)), R_optimal_be) \
                         + c.unsqueeze(0) + T_optimal_be                        # [E,3]
        
        out = {
            # ENSEMBLE-FIRST outputs
            "best_ensemble_coords": best_ensemble_coords,               # [E,N,3]
            "best_score": final_scores[best_idx].item(),
            "best_batch_index": int(best_idx.item()),
            "all_scores": final_scores.detach(),                        # [B]
            "all_R": R_total_all.detach(),                              # [B,E,3,3]  (total per-ensemble rotations)
            "all_T": t_ens.detach(),                                    # [B,E,3]    (per-ensemble translations)
            # Composed transforms mapping RAW coords -> final pose (per member)
            "R_composed": R_composed_all.detach(),                      # [E,3,3]
            "T_composed": T_composed_all.detach(),                      # [E,3]
            # Pre-alignment info (reference only)
            "R_prealign": R_prealign.detach(),                          # [E,3,3]
            "T_prealign": T_prealign.detach(),                          # [E,3]
            # Optional compatibility
            "best_aligned_coords": aligned_coords_best,                 # [N,3]
        }
        if return_all:
            out["trajectory"] = traj
        return out
