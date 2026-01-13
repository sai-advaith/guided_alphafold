"""
ESP-based SE3 alignment function for ensemble structures.

This module provides functionality to align ensemble structures to target ESP volumes
using rotation and translation optimization.
"""

import math
import time
import warnings
import numpy as np
import torch
from torch.amp import autocast, GradScaler

from cryoforward.atom_stack import AtomStack
from cryoforward.lattice import Lattice
from cryoforward.utils.rigid_transform import rodrigues_batch
from cryoforward.cryoesp_calculator import setup_fast_esp_solver
from cryoforward.utils.fft_utils import fft_downsample_3d
from cryoforward.utils.so3_grid import grid_SO3
from cryoforward.utils.lie_tools import quaternions_to_SO3

# Import alignment utilities from protenix
from src.protenix.metrics.rmsd import (
    _pca_axes_from_points,
    rot_pi_about_pca_axes,
    self_aligned_rmsd,
    random_rotation_matrices_haar,
)

# SO(3) grid sizes: 72 * 8^resol
# resol=0 -> 72, resol=1 -> 576, resol=2 -> 4608, resol=3 -> 36864
SO3_GRID_SIZES = {0: 72, 1: 576, 2: 4608, 3: 36864, 4: 294912}


def _check_resolution_warning(reduced_resolution: float, volume_resolution_A: float | None) -> None:
    """
    Check if reduced resolution is worse than volume resolution and issue warning if needed.
    
    Args:
        reduced_resolution: Nyquist resolution of the reduced volume (in Angstroms)
        volume_resolution_A: Resolution of the original volume (in Angstroms)
    """
    if volume_resolution_A is not None:
        if reduced_resolution > volume_resolution_A:
            warnings.warn(
                f"Reduced Nyquist resolution ({reduced_resolution:.2f} Å) is WORSE (coarser) than "
                f"volume resolution ({volume_resolution_A:.2f} Å). This may result in loss of signal information. "
                f"Consider using a larger D_reduced to preserve more resolution.",
                UserWarning,
                stacklevel=3
            )


def _estimate_sublattice_radius(lattice: Lattice, voxel_size_original: float) -> float:
    """
    Get sublattice radius in Angstroms from the original lattice.
    
    Args:
        lattice: Original lattice object
        voxel_size_original: Original voxel size in Angstroms (unused, kept for compatibility)
        
    Returns:
        Sublattice radius in Angstroms (minimum 5.0 Å)
    """
    # First try to get the stored radius
    if hasattr(lattice, 'sublattice_radius_in_A'):
        return max(lattice.sublattice_radius_in_A, 5.0)
    
    # Fallback: estimate from dimensions
    if hasattr(lattice, 'sublattice_dimensions') and lattice.sublattice_dimensions is not None:
        estimated_radius = (lattice.sublattice_dimensions[0].item() / 2.0) * voxel_size_original
        return max(estimated_radius, 5.0)
    
    # Final fallback
    return 10.0


def _downsample_volumes_and_lattice(
    target_esp: torch.Tensor,
    mask3d: torch.Tensor,
    lattice: Lattice,
    D_original: int,
    D_reduced: int,
    device: torch.device,
    dtype: torch.dtype,
    volume_resolution_A: float | None = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, Lattice, int]:
    """
    Downsample target ESP, mask, and create a reduced lattice.
    
    Args:
        target_esp: Original target ESP volume [D_original, D_original, D_original]
        mask3d: Original mask [D_original, D_original, D_original] or flattened
        lattice: Original lattice object
        D_original: Original dimension
        D_reduced: Target reduced dimension
        device: Device for tensors
        dtype: Data type for tensors
        volume_resolution_A: Optional volume resolution for checking (in Angstroms)
        
    Returns:
        Tuple of (downsampled_target_esp, downsampled_mask3d, reduced_lattice, D_reduced)
    """
    if verbose:
        print(f"[ESP-Align] Downsampling from {D_original}^3 to {D_reduced}^3")
    
    # Store original voxel size for calculations
    voxel_size_original = lattice.voxel_sizes_in_A[0].item()
    
    # Downsample target ESP
    target_esp_downsampled = fft_downsample_3d(target_esp, (D_reduced, D_reduced, D_reduced))
    
    # Downsample mask (using FFT downsampling, then threshold)
    mask3d_reshaped = mask3d.to(device=device, dtype=dtype).reshape(D_original, D_original, D_original)
    mask3d_downsampled = fft_downsample_3d(mask3d_reshaped, (D_reduced, D_reduced, D_reduced))
    # Threshold to maintain binary-like mask (values > 0.5 become 1, else 0)
    mask3d_downsampled = (mask3d_downsampled > 0.5).to(dtype)
    
    # Verify downsampled shapes match D_reduced
    assert target_esp_downsampled.shape == (D_reduced, D_reduced, D_reduced), \
        f"Downsampled target_esp shape {target_esp_downsampled.shape} != ({D_reduced}, {D_reduced}, {D_reduced})"
    assert mask3d_downsampled.shape == (D_reduced, D_reduced, D_reduced), \
        f"Downsampled mask3d shape {mask3d_downsampled.shape} != ({D_reduced}, {D_reduced}, {D_reduced})"
    
    # Calculate new voxel sizes (scale up proportionally to maintain same physical extent)
    scale_factor = D_original / D_reduced
    voxel_sizes_reduced = lattice.voxel_sizes_in_A * scale_factor
    
    # Calculate reduced resolution (Nyquist resolution = 2 * voxel_size)
    reduced_resolution = 2.0 * voxel_sizes_reduced[0].item()
    
    # Check resolution and warn if needed
    _check_resolution_warning(reduced_resolution, volume_resolution_A)
    
    # Estimate sublattice radius (in Angstroms)
    sublattice_radius_in_A = _estimate_sublattice_radius(lattice, voxel_size_original)
    if verbose:
        print(f"[ESP-Align] Retrieved sublattice radius from original lattice: {sublattice_radius_in_A:.2f} Å")
    
    # Create reduced lattice with EXACTLY D_reduced dimensions to match downsampled volumes
    # Calculate right_upper_point to maintain same physical extent
    left_bottom = lattice.left_bottom_point.cpu().numpy()
    # grid_side_lengths = (D - 1) * voxel_size, so right_upper = left_bottom + (D_reduced - 1) * voxel_size
    right_upper = left_bottom + (D_reduced - 1) * voxel_sizes_reduced.cpu().numpy()
    
    reduced_lattice = Lattice.from_grid_dimensions_and_voxel_sizes(
        grid_dimensions=(D_reduced, D_reduced, D_reduced),
        voxel_sizes_in_A=tuple(voxel_sizes_reduced.cpu().numpy()),
        left_bottom_point_in_A=tuple(left_bottom),
        right_upper_point_in_A=tuple(right_upper),
        sublattice_radius_in_A=sublattice_radius_in_A,
        dtype=dtype,
        device=device
    )
    
    # Initialize lattice coordinates for the reduced lattice (needed for masked_lat calculation)
    reduced_lattice._initialize_lattice_coordinates()
    
    # Verify lattice dimensions match
    lattice_dims = tuple(reduced_lattice.grid_dimensions.cpu().numpy())
    assert lattice_dims == (D_reduced, D_reduced, D_reduced), \
        f"Lattice grid dimensions {lattice_dims} != ({D_reduced}, {D_reduced}, {D_reduced})"
    assert reduced_lattice.lattice_coordinates.shape[0] == D_reduced ** 3, \
        f"Lattice coordinates shape {reduced_lattice.lattice_coordinates.shape[0]} != {D_reduced ** 3}"
    
    if verbose:
        print(f"[ESP-Align] Reduced voxel size: {voxel_sizes_reduced[0].item():.4f} Å (from {voxel_size_original:.4f} Å)")
        print(f"[ESP-Align] Reduced Nyquist resolution: {reduced_resolution:.2f} Å")
    
    return target_esp_downsampled, mask3d_downsampled, reduced_lattice, D_reduced


def esp_se3_align_ensemble(
    atom_stack: AtomStack,
    lattice: Lattice,
    target_esp: torch.Tensor,
    mask3d: torch.Tensor,
    steps: int = 50,
    lr_t_A: float = 0.1,
    lr_r_deg: float = 1.0,
    print_every: int = 10,
    per_step_t_cap_voxels: float | None = 1.0,
    n_random: int = 16,
    seed: int | None = None,
    return_all: bool = False,
    t_init_box_edge_voxels: float | None = 1.0,
    max_volumes_per_batch: int = 4,
    use_checkpointing: bool = False,  # Parameter kept for API compatibility but ignored/forced False internally
    n_keep_after_pruning: int = 3,
    pruning_iteration: int = 4,
    second_pruning_iteration: int | None = None,  # If set, do a second prune at this iteration. First prune keeps 3x n_keep_after_pruning.
    min_cc_for_convergence: float = 0.5,  # If CC is below this after first pruning, algorithm may be stuck in local minimum
    target_atom_stack: AtomStack | None = None,  # Optional target structure for RMSD benchmarking
    D_reduced: int | None = None,  # If provided, downsample to this dimension
    volume_resolution_A: float | None = None,  # Resolution of the volume in Angstroms (for checking against reduced resolution)
    use_autocast: bool = False,  # If True, use mixed precision (FP16) for volume computation. Faster on modern GPUs (1.5-2x speedup).
    min_cc_threshold: float = 0.15,  # Poses with CC below this are reinitialized
    max_reinit_attempts: int = 3,  # Max attempts to reinitialize bad poses
    overshoot_recovery_drop: float = 0.02,  # Restore to best-ever if CC drops by more than this
    adaptive_reinit: bool = True,  # If True, periodically reinitialize poor-performing hypotheses during optimization
    adaptive_reinit_iterations: list[int] | None = None,  # Iterations at which to check and reinitialize (default: [5, 10, 15])
    adaptive_reinit_fraction: float = 0.1,  # Fraction of worst hypotheses to reinitialize (0.1 = bottom 10%)
    adaptive_reinit_cc_threshold: float | None = None,  # Reinitialize if CC is below this relative to best (None = auto: best_cc * 0.5)
    rmsd_regularization_weight: float = 0.0,  # If > 0, add RMSD-to-ensemble-0 regularization to the loss (normalized to CC magnitude)
    use_so3_grid: bool = True,  # If True, use structured SO(3) Hopf grid instead of random Haar rotations for better coverage
    so3_grid_resolution: int | None = None,  # SO(3) grid resolution: 0=72, 1=576, 2=4608, 3=36864. If None, auto-select based on n_random.
    use_pca_init: bool = False,  # If True, add 6 PCA-axis rotations to init (legacy behavior). False recommended with SO(3) grid.
    optimizer: str = "adam",  # Optimizer: "adam", "sgd", "lbfgs". LBFGS is best for smooth objectives.
    adam_betas: tuple[float, float] = (0.9, 0.999),  # Adam momentum parameters. Lower beta1 (e.g., 0.5) reduces oscillation.
    use_ema: bool = False,  # If True, use Exponential Moving Average of parameters (smooths oscillations)
    ema_decay: float = 0.95,  # EMA decay factor (higher = smoother, slower adaptation)
    use_lr_decay: bool = True,  # If False, disable automatic LR decay on plateau
    lr_decay_factor: float = 0.75,  # Factor to multiply LR by when plateau is detected (0.75 = reduce by 25%)
    lr_plateau_threshold: int = 8,  # Number of iterations without improvement before triggering LR decay
    lr_decay_warmup_steps: int = 0,  # Disable LR decay for first N steps (allows aggressive exploration before fine-tuning)
    lr_decay_cc_threshold: float | None = 0.5,  # Start LR decay when CC reaches this threshold (None = disable, use only plateau)
    lr_decay_cc_cooldown: int = 8,  # Minimum iterations between CC-based decays (prevents over-aggressive decay during convergence)
    lr_plateau_min_cc: float = 0.3,  # Only trigger plateau-based decay when CC >= this (prevents aggressive decay when stuck in bad local minima)
    lr_plateau_threshold_high_cc: int | None = None,  # Plateau threshold when CC >= 0.65 (fine-tuning phase). None = use lr_plateau_threshold
    verbose: bool = False,  # If True, print progress and debug information. Default False to reduce output noise.
    integrate_gaussians_over_voxel: bool = True,  # If True, integrate Gaussian contributions over entire voxel volume. If False, evaluate Gaussians only at voxel center point.
):
    """
    SE3 alignment of ensemble to target ESP using fast ESP computation.
    
    Optimizes rotation and translation for the entire ensemble such that
    the ensemble-computed ESP best matches the target ESP in the masked region.
    
    Args:
        atom_stack: Full ensemble AtomStack (contains all ensemble members with coords, b-factors, etc.)
                   Will be modified in-place during optimization
        lattice: Lattice object for ESP computation
        target_esp: [D, D, D] target ESP volume
        mask3d: [D, D, D] or [D^3] density mask (boolean or 0/1)
        steps: Number of optimization steps
        lr_t_A: Learning rate for translation (in Angstroms)
        lr_r_deg: Learning rate for rotation (in degrees)
        print_every: Print progress every N steps
        per_step_t_cap_voxels: Cap translation increment per step (in voxels)
        n_random: Number of random rotation starts
        seed: Random seed
        return_all: Whether to return trajectory
        t_init_box_edge_voxels: Initial translation box size (in voxels)
        max_volumes_per_batch: Maximum number of volumes to compute in parallel per batch
        use_checkpointing: Whether to use gradient checkpointing (ignored, forced to False)
        n_keep_after_pruning: Number of rotation starts to keep after pruning
        pruning_iteration: Iteration at which to perform pruning
        target_atom_stack: Optional target AtomStack for RMSD benchmarking. If None, RMSD is computed
                          against the original input structure. If provided, RMSD is computed against
                          this target structure (useful when aligning a rotated/misaligned structure
                          back to the original).
        D_reduced: If provided, downsample target_esp, mask, and lattice to this dimension for faster computation.
                   The physical extent remains the same, but voxel size increases proportionally.
        volume_resolution_A: Resolution of the volume in Angstroms. If provided and D_reduced is used, a warning
                            will be issued if the reduced resolution (Nyquist) is below this value.
        use_autocast: If True, enables mixed precision (FP16) computation for volume generation. Provides 1.5-2x
                     speedup on modern GPUs (A100, H100, RTX 30/40 series) and reduces memory usage by ~50%.
                     Default is False. Only works on CUDA devices.
        min_cc_threshold: Minimum CC threshold for initial poses. Poses with CC below this are reinitialized
                         with new random rotations. Set to 0 to disable. Default is 0.15.
        max_reinit_attempts: Maximum number of attempts to reinitialize poses with CC < min_cc_threshold.
                            After this many attempts, optimization continues with whatever poses we have.
        overshoot_recovery_drop: If current best CC drops by more than this amount from the best-ever CC,
                                restore the best hypothesis to its best-ever state and continue optimization
                                from there. Prevents losing good solutions due to optimizer overshooting.
                                Default is 0.02.
        use_so3_grid: If True (default), use structured SO(3) Hopf fibration grid for rotation initialization
                     instead of purely random Haar rotations. This provides more uniform coverage of rotation
                     space. Grid sizes: resol=0->72, resol=1->576. The largest grid fitting n_random is used,
                     with remaining slots filled by random rotations.
        use_pca_init: If True, add 6 PCA-axis based rotations to initialization (legacy behavior).
                     Default is False since SO(3) grid already provides uniform coverage.
    
    Returns:
        Dict with best alignment results containing:
        - best_ensemble_coords: Best aligned coordinates [B_ensembles, N, 3]
        - best_score: Best cross-correlation score
        - R_composed: Composed rotation matrices [B_ensembles, 3, 3]
        - T_composed: Composed translation vectors [B_ensembles, 3]
        - scores_history: History of scores for each iteration
    """
    device, dtype = target_esp.device, target_esp.dtype
    D_original = target_esp.shape[0]
    
    _t_start = time.perf_counter()
    _timings = {}
    
    if D_reduced is not None and D_reduced < D_original:
        target_esp, mask3d, lattice, D = _downsample_volumes_and_lattice(
            target_esp=target_esp,
            mask3d=mask3d,
            lattice=lattice,
            D_original=D_original,
            D_reduced=D_reduced,
            device=device,
            dtype=dtype,
            volume_resolution_A=volume_resolution_A,
            verbose=verbose,
        )
    else:
        D = D_original
        if not hasattr(lattice, 'lattice_coordinates') or lattice.lattice_coordinates is None:
            lattice._initialize_lattice_coordinates()
    
    _timings['downsample'] = time.perf_counter() - _t_start
    _t_step = time.perf_counter()

    if mask3d.dim() == 3:
        mask_vol = mask3d.to(device=device, dtype=dtype)
    else:
        mask_vol = mask3d.to(device=device, dtype=dtype).reshape(D, D, D)
    mv = torch.nn.functional.avg_pool3d(mask_vol[None, None], 3, 1, 1)[0, 0]
    mv = torch.nn.functional.avg_pool3d(mv[None, None], 3, 1, 1)[0, 0]
    mask_vol = mv / mv.max() if mv.max() > 0 else mv
    mask_flat_bool = (mask_vol.reshape(-1) > 0.5)

    target_masked = target_esp.reshape(-1)[mask_flat_bool]
    target_centered = (target_masked - target_masked.mean()).detach()
    target_var = (target_centered**2).mean().detach()

    grid_size = int(torch.prod(lattice.grid_dimensions).item())
    mask_size = mask_flat_bool.sum().item()
    sublattice_size = int(torch.prod(lattice.sublattice_dimensions).item()) if hasattr(lattice, 'sublattice_dimensions') and lattice.sublattice_dimensions is not None else 0
    sublattice_dims = tuple(lattice.sublattice_dimensions.cpu().numpy()) if hasattr(lattice, 'sublattice_dimensions') and lattice.sublattice_dimensions is not None else None
    sublattice_coords_shape = lattice.sublattice_coordinates.shape if hasattr(lattice, 'sublattice_coordinates') and lattice.sublattice_coordinates is not None else None
    voxel_size = lattice.voxel_sizes_in_A[0].item()
    sublattice_radius = getattr(lattice, 'sublattice_radius_in_A', None)
    if verbose:
        print(f"[ESP-Align] Using lattice: {tuple(lattice.grid_dimensions.cpu().numpy())} (grid_size={grid_size:,}, masked_voxels={mask_size:,}, sublattice_size={sublattice_size:,})")
        if sublattice_dims is not None:
            print(f"[ESP-Align] Sublattice dimensions: {sublattice_dims}")
            if sublattice_radius is not None:
                radius_voxels_calc = (sublattice_radius / voxel_size) - 0.5
                radius_voxels = int(np.ceil(radius_voxels_calc))
                print(f"[ESP-Align] Sublattice radius: {sublattice_radius:.2f} Å, voxel_size: {voxel_size:.4f} Å → radius_voxels: {radius_voxels} (calc: {radius_voxels_calc:.3f}) → extent: {2*radius_voxels+1}")
        if sublattice_coords_shape is not None:
            print(f"[ESP-Align] Sublattice coordinates shape: {sublattice_coords_shape}")
    _timings['mask_prep'] = time.perf_counter() - _t_step
    _t_step = time.perf_counter()
    
    # NOTE: use_checkpointing=False for fused multi-volume ops (in-place volume update breaks autograd with checkpointing)
    print(f"[DEBUG esp_se3_align_ensemble] per_voxel_averaging = {integrate_gaussians_over_voxel}")
    compute_batch_volumes, compute_batch_from_coords = setup_fast_esp_solver(
        atom_stack,
        lattice,
        per_voxel_averaging=integrate_gaussians_over_voxel,
        use_checkpointing=False,  # Never use checkpointing for vectorized path
        use_autocast=use_autocast
    )
    
    _timings['setup_esp_solver'] = time.perf_counter() - _t_step
    _t_step = time.perf_counter()

    coords_original = atom_stack.atom_coordinates
    B_ensembles, N_atoms, _ = coords_original.shape
    
    if target_atom_stack is not None:
        coords_target_ref = target_atom_stack.atom_coordinates.to(device, dtype).detach().clone()
        if coords_target_ref.shape[0] == 1 and B_ensembles > 1:
            coords_target_ref = coords_target_ref.repeat(B_ensembles, 1, 1)
        compute_rmsd = True
    else:
        coords_target_ref = None
        compute_rmsd = False

    coords_ens = coords_original.to(device, dtype).detach()
    
    # Pre-alignment: align all members to common frame
    with torch.no_grad():
        # Compute each member's centroid
        member_centroids = coords_ens.mean(1)  # [B_ens, 3]
        
        # Use member 0's centroid as the target (symmetric reference point)
        target_centroid = member_centroids[0].clone()  # [3]
        
        # Translate all members to align centroids
        translations = target_centroid[None, :] - member_centroids  # [B_ens, 3]
        
        # Apply translations
        coords_translated = coords_ens + translations[:, None, :]  # [B_ens, N, 3]
        
        # Rotate all members to align with member 0's orientation
        coords_centered = coords_translated - target_centroid[None, None, :]  # [B_ens, N, 3]
        ref_centered = coords_centered[0]  # [N, 3] - member 0 as reference
        
        R_pre_init, T_pre_init = [], []
        for i in range(B_ensembles):
            # All members go through Kabsch for symmetry
            curr_centered = coords_centered[i]  # [N, 3]
            
            # Align centered current to centered reference
            _, _, R_i, T_i = self_aligned_rmsd(
                curr_centered.unsqueeze(0),
                ref_centered.unsqueeze(0),
                torch.ones(N_atoms, device=device, dtype=torch.bool)
            )
            R_pre_init.append(R_i[0])
            T_pre_init.append(T_i.reshape(-1))
        
        R_pre_init = torch.stack(R_pre_init)
        T_pre_init = torch.stack(T_pre_init)
    
    _timings['pre_align_ensemble'] = time.perf_counter() - _t_step
    _t_step = time.perf_counter()
    
    # Apply pre-alignment transformations
    coords_pre = torch.zeros_like(coords_ens)
    for i in range(B_ensembles):
        # Transform: (coords_centered) @ R.T + target_centroid + T
        coords_pre[i] = coords_centered[i] @ R_pre_init[i].T + target_centroid + T_pre_init[i]
        # Force exact centering: ensure member is exactly at target_centroid
        member_centroid = coords_pre[i].mean(0)
        coords_pre[i] = coords_pre[i] - member_centroid + target_centroid
    coords_pre = coords_pre.detach()
    
    # Store target_centroid for use as centroid during optimization
    ens_centroid = target_centroid

    # Check pre-alignment quality
    if verbose and B_ensembles > 1:
        pre_align_rmsds = []
        for i in range(1, B_ensembles):
            diff = coords_pre[i] - coords_pre[0]
            rmsd = torch.sqrt((diff ** 2).sum(dim=-1).mean()).item()
            pre_align_rmsds.append(rmsd)
        print(f" [Pre-align] RMSD between members after pre-alignment: {[f'{r:.6f}' for r in pre_align_rmsds]}")

    # Use ens_centroid directly (not average of coords_pre) for symmetry
    centroid = ens_centroid.detach()
    
    # Generate random rotations: either from SO(3) grid (structured, uniform coverage) or Haar random
    if use_so3_grid:
        # Determine grid resolution: explicit or auto-select based on n_random
        if so3_grid_resolution is not None:
            # Use explicitly specified resolution
            best_resol = so3_grid_resolution
            if best_resol not in SO3_GRID_SIZES:
                raise ValueError(f"Invalid so3_grid_resolution={best_resol}. Valid values: {list(SO3_GRID_SIZES.keys())}")
        else:
            # Auto-select: find largest grid resolution that fits within n_random
            best_resol = 0
            for resol in range(5):
                if SO3_GRID_SIZES.get(resol, float('inf')) <= n_random:
                    best_resol = resol
                else:
                    break
        
        # Get grid quaternions and convert to rotation matrices
        quats = grid_SO3(best_resol)  # [N_grid, 4] numpy array
        R_grid = quaternions_to_SO3(torch.from_numpy(quats).to(device=device, dtype=dtype))  # [N_grid, 3, 3]
        n_grid = R_grid.shape[0]
        
        # Fill remainder with truly random rotations if needed (only if n_random > grid size)
        n_remaining = n_random - n_grid
        if n_remaining > 0:
            R_rand_extra = random_rotation_matrices_haar(n_remaining, device=device, seed=seed)
            R_rand = torch.cat([R_grid, R_rand_extra], dim=0)
        else:
            # If grid is larger than requested, subsample (though unlikely with resol=0->72)
            R_rand = R_grid[:n_random] if n_grid > n_random else R_grid
        
        if verbose:
            print(f"[ESP-Align] Using SO(3) grid (resol={best_resol}, {n_grid} points) + {max(0, n_remaining)} random = {R_rand.shape[0]} total rotations")
    else:
        R_rand = random_rotation_matrices_haar(n_random, device=device, seed=seed)
    
    _timings['rotation_init'] = time.perf_counter() - _t_step
    _t_step = time.perf_counter()
    
    # Build rotation initialization list
    if use_pca_init:
        # Legacy behavior: include PCA-based rotations (6 extra)
        U_prot = _pca_axes_from_points(coords_pre.reshape(1, -1, 3))
        Rx, Ry, Rz = rot_pi_about_pca_axes(U_prot)
        R_pca = torch.eye(3, device=device)
        R_inits = torch.cat([
            R_pca[None],  # identity
            torch.stack([R_pca @ Rx, R_pca @ Ry, R_pca @ Rz, Rx @ R_pca, Ry @ R_pca, Rz @ R_pca]),  # 6 PCA rotations
            R_rand,
            torch.eye(3, device=device)[None]
        ])
    else:
        # Simplified: just identity + grid/random rotations
        R_inits = torch.cat([
            torch.eye(3, device=device)[None],  # identity
            R_rand,
        ])
    B_rot = R_inits.shape[0]

    # Translation initialization
    masked_lat = lattice.lattice_coordinates.reshape(-1, 3)[mask_flat_bool]
    w_t = target_masked.abs()
    w_sum = w_t.sum()
    tgt_cent = (masked_lat * w_t[:, None]).sum(0) / w_sum if w_sum > 1e-10 else masked_lat.mean(0)
    T_init_base = (tgt_cent - centroid).detach()
    
    gen = torch.Generator(device).manual_seed(seed) if seed else None
    t_jitter = (torch.rand(B_rot, 3, device=device, generator=gen) - 0.5) * 2.0 * (t_init_box_edge_voxels or 1.0) * lattice.voxel_sizes_in_A[0]
    t_jitter[0] = 0.0
    
    # Initialize: all members get same rotation/translation per hypothesis
    
    R_inits_expanded = R_inits.unsqueeze(1).repeat(1, B_ensembles, 1, 1)  # [B_rot, B_ens, 3, 3]
    
    # T_member0 for all hypotheses: [B_rot, 3]
    T_member0_all = T_init_base + t_jitter  # [B_rot, 3]
    
    # All members get the same translation (since they're pre-aligned)
    # Use repeat() instead of expand() to allocate separate memory
    T_inits_expanded = T_member0_all.unsqueeze(1).repeat(1, B_ensembles, 1)  # [B_rot, B_ens, 3]
    
    _timings['hypothesis_init_loop'] = time.perf_counter() - _t_step
    _t_step = time.perf_counter()
    
    from cryoforward.utils.rigid_transform import _matrix_to_axis_angle
    w_param_init = _matrix_to_axis_angle(R_inits_expanded)
    
    # Clone and make contiguous to ensure each ensemble member has completely independent memory
    w_param = torch.nn.Parameter(w_param_init.clone().contiguous())
    t = torch.nn.Parameter(T_inits_expanded.clone().contiguous())
    
    # Create optimizer based on choice
    lr_r = math.radians(lr_r_deg)
    if optimizer.lower() == "adam":
        opt = torch.optim.Adam([
            {'params': t, 'lr': lr_t_A},
            {'params': w_param, 'lr': lr_r}
        ], betas=adam_betas, eps=1e-8)
        opt_name = f"Adam(β={adam_betas})"
    elif optimizer.lower() == "sgd":
        opt = torch.optim.SGD([
            {'params': t, 'lr': lr_t_A},
            {'params': w_param, 'lr': lr_r}
        ])
        opt_name = "SGD"
    elif optimizer.lower() == "lbfgs":
        # LBFGS uses single LR, we'll use the larger one
        opt = torch.optim.LBFGS([t, w_param], lr=1.0, max_iter=5, line_search_fn="strong_wolfe")
        opt_name = "LBFGS"
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}. Use 'adam', 'sgd', or 'lbfgs'.")
    
    # Initialize EMA if requested
    ema_w = None
    ema_t = None
    if use_ema:
        ema_w = w_param.detach().clone()
        ema_t = t.detach().clone()

    active_idx = torch.arange(B_rot, device=device)
    history = []
    
    # Initialize mixed precision scaler if autocast is enabled
    scaler = GradScaler('cuda') if use_autocast and device.type == 'cuda' else None
    if use_autocast and verbose:
        print(f"[ESP-Align] Mixed precision (FP16) enabled for faster computation")
        print(f"[ESP-Align] NOTE: Compiled kernels may have limited FP16 benefit. For maximum speedup, ")
        print(f"[ESP-Align]      consider modifying ESP solver to use FP16 natively.")
    
    _timings['optimizer_init'] = time.perf_counter() - _t_step
    _timings['total_setup'] = time.perf_counter() - _t_start
    
    if verbose:
        ema_str = f" | EMA(decay={ema_decay})" if use_ema else ""
        print(f"[ESP-Align] Hypotheses: {B_rot} | Ensembles: {B_ensembles} | MaxVol/Batch: {max_volumes_per_batch} | Opt: {opt_name}{ema_str}")
        print(f"[ESP-Align] Setup timings: " + " | ".join(f"{k}={v:.3f}s" for k, v in _timings.items()))

    bfactors_ready = atom_stack.bfactors.to(device).contiguous() if atom_stack.bfactors is not None else None
    occupancies_ready = atom_stack.occupancies.to(device).contiguous()
    
    best_ever_cc = -float('inf')
    best_ever_w = None
    best_ever_t = None
    best_ever_global_idx = None
    best_ever_rmsd = None
    best_ever_rmsd_min = None
    best_ever_rmsd_max = None
    best_ever_iter = None
    
    plateau_counter = 0
    plateau_threshold = lr_plateau_threshold
    min_lr_t = lr_t_A * 0.1
    min_lr_r = math.radians(lr_r_deg) * 0.1
    last_cc_decay_iter = -1  # Track last iteration when CC-based decay happened
    cc_decay_cooldown = lr_decay_cc_cooldown  # Minimum iterations between CC-based decays
    
    def _compute_all_cc():
        all_scores = []
        with torch.no_grad():
            for b_start in range(0, len(active_idx), max_volumes_per_batch):
                b_end = min(b_start + max_volumes_per_batch, len(active_idx))
                b_idxs = active_idx[b_start:b_end]
                k = len(b_idxs)
                
                # Vectorized: compute all rotations at once
                w_batch = w_param[b_idxs].reshape(-1, 3)
                R_batch = rodrigues_batch(w_batch).reshape(k, B_ensembles, 3, 3)
                t_batch = t[b_idxs]
                
                # Vectorized: transform all coords at once
                coords_batch = AtomStack.batch_transform_coordinates(
                    coords_pre, R_batch, t_batch, centroid
                )
                
                # Vectorized volume computation
                vols = compute_batch_from_coords(
                    coords_batch, bfactors_ready, atom_stack.atomic_numbers, occupancies_ready
                )
                vols = vols.reshape(k, -1)
                v_masked = vols[:, mask_flat_bool]
                
                v_mean = v_masked.mean(1, keepdim=True)
                v_cent = v_masked - v_mean
                num = (v_cent * target_centered[None]).mean(1)
                den = torch.sqrt(target_var * (v_cent**2).mean(1)) + 1e-10
                cc = num / den
                all_scores.extend(cc.cpu().tolist())
        return all_scores
    
    # Skip reinit when using SO(3) grid - the structured grid provides guaranteed coverage,
    # so even if initial CC is low, optimization will find the right basin. Reinit would
    # throw away the structured grid and replace with random rotations, defeating the purpose.
    if min_cc_threshold > 0 and not use_so3_grid:
        for reinit_attempt in range(max_reinit_attempts):
            init_scores = _compute_all_cc()
            init_scores_t = torch.tensor(init_scores, device=device)
            best_init_cc = init_scores_t.max().item()
            
            if best_init_cc >= min_cc_threshold:
                n_good = (init_scores_t >= min_cc_threshold).sum().item()
                if verbose:
                    print(f"[Reinit] {n_good}/{len(active_idx)} poses have CC >= {min_cc_threshold:.2f}, best={best_init_cc:.4f} (attempt {reinit_attempt + 1})")
                break
            
            if verbose:
                print(f"[Reinit] Attempt {reinit_attempt + 1}/{max_reinit_attempts}: Best CC={best_init_cc:.4f} < {min_cc_threshold:.2f}, reinitializing ALL poses...")
            
            with torch.no_grad():
                n_poses = len(active_idx)
                new_rots = random_rotation_matrices_haar(n_poses, device=device)
                from cryoforward.utils.rigid_transform import _matrix_to_axis_angle
                for idx, hyp_idx in enumerate(active_idx):
                    R_new_member0 = new_rots[idx]  # no need for @ R_pca since R_pca was identity
                    T_new_member0 = T_init_base + (torch.rand(3, device=device) - 0.5) * 2.0 * (t_init_box_edge_voxels or 1.0) * lattice.voxel_sizes_in_A[0]
                    coords_member0_rotated_new = (coords_pre[0:1] - centroid) @ R_new_member0.T + centroid + T_new_member0
                    
                    w_param.data[hyp_idx, 0] = _matrix_to_axis_angle(R_new_member0.unsqueeze(0))[0]
                    t.data[hyp_idx, 0] = T_new_member0
                    
                    for ens_idx in range(1, B_ensembles):
                        _, _, R_kabsch, T_kabsch = self_aligned_rmsd(
                            coords_pre[ens_idx:ens_idx+1],
                            coords_member0_rotated_new,
                            torch.ones(N_atoms, device=device, dtype=torch.bool)
                        )
                        
                        coords_test = coords_pre[ens_idx:ens_idx+1]
                        coords_kabsch_result = coords_test @ R_kabsch[0].T + T_kabsch.reshape(-1)
                        verify_rmsd = torch.sqrt(((coords_kabsch_result - coords_member0_rotated_new) ** 2).sum(dim=-1).mean()).item()
                        
                        if verify_rmsd > 0.1:
                            coords_kabsch_result2 = coords_test @ R_kabsch[0] + T_kabsch.reshape(-1)
                            verify_rmsd2 = torch.sqrt(((coords_kabsch_result2 - coords_member0_rotated_new) ** 2).sum(dim=-1).mean()).item()
                            if verify_rmsd2 < verify_rmsd:
                                R_use = R_kabsch[0].T
                            else:
                                R_use = R_kabsch[0]
                        else:
                            R_use = R_kabsch[0]
                        
                        coords_transformed = (coords_test - centroid) @ R_use.T + centroid
                        T_computed = (coords_member0_rotated_new - coords_transformed).mean(dim=1).squeeze(0)
                        
                        w_param.data[hyp_idx, ens_idx] = _matrix_to_axis_angle(R_use.unsqueeze(0))[0]
                        t.data[hyp_idx, ens_idx] = T_computed
        else:
            final_scores = _compute_all_cc()
            best_init_cc = max(final_scores)
            if verbose:
                print(f"[Reinit] Max attempts reached. Best initial CC: {best_init_cc:.4f}")
    
    for it in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        
        step_scores = []
        idx_list = active_idx.tolist()
        n_active = len(idx_list)
        
        for b_start in range(0, n_active, max_volumes_per_batch):
            b_end = min(b_start + max_volumes_per_batch, n_active)
            b_idxs = active_idx[b_start:b_end]
            k = len(b_idxs)
            
            # Vectorized: compute all rotations at once
            # w_param[b_idxs] is [k, B_ens, 3] → reshape to [k*B_ens, 3] for rodrigues_batch
            w_batch = w_param[b_idxs].reshape(-1, 3)  # [k*B_ens, 3]
            R_batch = rodrigues_batch(w_batch).reshape(k, B_ensembles, 3, 3)  # [k, B_ens, 3, 3]
            t_batch = t[b_idxs]  # [k, B_ens, 3]
            
            # Vectorized: transform all coords at once
            # coords_pre is [B_ens, N, 3], output is [k, B_ens, N, 3]
            coords_batch = AtomStack.batch_transform_coordinates(
                coords_pre, R_batch, t_batch, centroid
            )
            
            # Vectorized: compute RMSD regularization if requested
            rmsd_vec = None
            if rmsd_regularization_weight > 0.0 and B_ensembles > 1:
                # coords_batch: [k, B_ens, N, 3]
                # RMSD of members 1-N to member 0
                ref = coords_batch[:, 0:1, :, :]  # [k, 1, N, 3]
                diffs = coords_batch[:, 1:, :, :] - ref  # [k, B_ens-1, N, 3]
                msd_per_member = (diffs ** 2).sum(dim=-1).mean(dim=2)  # [k, B_ens-1]
                rmsd_per_member = torch.sqrt(msd_per_member + 1e-10)  # [k, B_ens-1]
                rmsd_vec = rmsd_per_member.mean(dim=1)  # [k]
            
            if use_autocast and device.type == 'cuda':
                with autocast('cuda'):
                    # Vectorized volume computation from pre-transformed coords
                    vols = compute_batch_from_coords(
                        coords_batch, bfactors_ready, atom_stack.atomic_numbers, occupancies_ready
                    )
                    vols = vols.reshape(k, -1)
                    v_masked = vols[:, mask_flat_bool]
                    v_mean = v_masked.mean(1, keepdim=True)
                    v_cent = v_masked - v_mean
                    num = (v_cent * target_centered[None]).mean(1)
                    den = torch.sqrt(target_var * (v_cent**2).mean(1)) + 1e-10
                    cc = num / den
                    cc_loss_vec = -cc  # shape [k]
                    if rmsd_regularization_weight > 0.0 and rmsd_vec is not None:
                        # Normalize RMSD term to have the same magnitude (L2-norm) as the CC loss,
                        # then scale by the user-specified weight.
                        cc_norm = cc_loss_vec.detach().norm()
                        rmsd_norm = rmsd_vec.detach().norm()
                        if cc_norm > 0 and rmsd_norm > 0:
                            rmsd_scaled = rmsd_vec * (cc_norm / (rmsd_norm + 1e-12))
                        else:
                            rmsd_scaled = rmsd_vec
                        loss_vec = cc_loss_vec + rmsd_regularization_weight * rmsd_scaled
                        loss = loss_vec.sum()
                    else:
                        loss = cc_loss_vec.sum()
            else:
                # Vectorized volume computation from pre-transformed coords
                vols = compute_batch_from_coords(
                    coords_batch, bfactors_ready, atom_stack.atomic_numbers, occupancies_ready
                )
                vols = vols.reshape(k, -1)
                v_masked = vols[:, mask_flat_bool]
                v_mean = v_masked.mean(1, keepdim=True)
                v_cent = v_masked - v_mean
                num = (v_cent * target_centered[None]).mean(1)
                den = torch.sqrt(target_var * (v_cent**2).mean(1)) + 1e-10
                cc = num / den
                cc_loss_vec = -cc  # shape [k]
                if rmsd_regularization_weight > 0.0 and rmsd_vec is not None:
                    cc_norm = cc_loss_vec.detach().norm()
                    rmsd_norm = rmsd_vec.detach().norm()
                    if cc_norm > 0 and rmsd_norm > 0:
                        rmsd_scaled = rmsd_vec * (cc_norm / (rmsd_norm + 1e-12))
                    else:
                        rmsd_scaled = rmsd_vec
                    loss_vec = cc_loss_vec + rmsd_regularization_weight * rmsd_scaled
                    loss = loss_vec.sum()
                else:
                    loss = cc_loss_vec.sum()
            
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            step_scores.extend(cc.detach().cpu().tolist())
            del vols, coords_batch

        skip_optimizer_step = False
        with torch.no_grad():
            current_best_cc = max(step_scores)
            improvement_threshold = 0.002
            
            if current_best_cc > best_ever_cc:
                is_significant_improvement = (current_best_cc - best_ever_cc) >= improvement_threshold
                best_ever_cc = current_best_cc
                best_ever_iter = it
                if is_significant_improvement:
                    plateau_counter = 0
                else:
                    plateau_counter += 1
                
                scores_t = torch.tensor(step_scores, device=device)
                best_local = torch.argmax(scores_t).item()
                best_global = active_idx[best_local]
                best_ever_w = w_param[best_global].detach().clone()
                best_ever_t = t[best_global].detach().clone()
                best_ever_global_idx = best_global
                
                if compute_rmsd:
                    R_best_tmp = rodrigues_batch(best_ever_w)
                    T_best_tmp = best_ever_t
                    coords_tmp = torch.zeros_like(coords_original)
                    for ens_idx in range(B_ensembles):
                        coords_tmp[ens_idx] = (coords_pre[ens_idx] - centroid) @ R_best_tmp[ens_idx].T + centroid + T_best_tmp[ens_idx]
                    
                    rmsds_tmp = []
                    for ens_idx in range(B_ensembles):
                        coords_aligned_tmp = coords_tmp[ens_idx].contiguous()
                        coords_target_tmp = coords_target_ref[ens_idx].to(device=device, dtype=dtype).contiguous()
                        if coords_aligned_tmp.shape[0] != coords_target_tmp.shape[0]:
                            min_atoms = min(coords_aligned_tmp.shape[0], coords_target_tmp.shape[0])
                            coords_aligned_tmp = coords_aligned_tmp[:min_atoms]
                            coords_target_tmp = coords_target_tmp[:min_atoms]
                        diff_tmp = coords_aligned_tmp - coords_target_tmp
                        msd_tmp = (diff_tmp ** 2).sum(dim=-1).mean()
                        rmsd_member = torch.sqrt(msd_tmp).item() if msd_tmp > 1e-10 else 0.0
                        rmsds_tmp.append(rmsd_member)
                    best_ever_rmsd = sum(rmsds_tmp) / len(rmsds_tmp) if rmsds_tmp else None
                    best_ever_rmsd_min = min(rmsds_tmp) if rmsds_tmp else None
                    best_ever_rmsd_max = max(rmsds_tmp) if rmsds_tmp else None
            else:
                plateau_counter += 1
                    
            recovery_threshold = overshoot_recovery_drop if it < 15 else overshoot_recovery_drop * 0.5
            if best_ever_w is not None and (best_ever_cc - current_best_cc) > recovery_threshold and it > 5:
                w_param.data[best_ever_global_idx] = best_ever_w.clone()
                t.data[best_ever_global_idx] = best_ever_t.clone()
                
                if w_param in opt.state:
                    state = opt.state[w_param]
                    if 'exp_avg' in state:
                        state['exp_avg'][best_ever_global_idx].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'][best_ever_global_idx].zero_()
                if t in opt.state:
                    state = opt.state[t]
                    if 'exp_avg' in state:
                        state['exp_avg'][best_ever_global_idx].zero_()
                    if 'exp_avg_sq' in state:
                        state['exp_avg_sq'][best_ever_global_idx].zero_()
                
                opt.zero_grad(set_to_none=True)
                skip_optimizer_step = True
                if verbose:
                    print(f" [Recovery] Iter {it}: CC dropped {best_ever_cc:.4f} -> {current_best_cc:.4f}, restoring best pose (optimizer state reset)")

        # DEBUG: Check parameter divergence (same as local_tests) - only print if verbose
        if verbose and B_ensembles > 1 and it in [1, 5, 10, 20]:
            best_idx = active_idx[torch.argmax(torch.tensor(step_scores, device=device))]
            w_diffs = [(w_param[best_idx, i] - w_param[best_idx, 0]).norm().item() for i in range(B_ensembles)]
            t_diffs = [(t[best_idx, i] - t[best_idx, 0]).norm().item() for i in range(B_ensembles)]
            print(f" [DEBUG] Iter {it}: w_div={[f'{d:.6f}' for d in w_diffs]}, t_div={[f'{d:.6f}' for d in t_diffs]}")
        
        best_rmsd = None
        best_rmsd_min = None
        best_rmsd_max = None
        if (it % print_every == 0 or it == steps) and compute_rmsd:
            with torch.no_grad():
                scores_t = torch.tensor(step_scores, device=device)
                best_local_idx = torch.argmax(scores_t).item()
                best_global_idx = active_idx[best_local_idx]
                
                R_best_curr = rodrigues_batch(w_param[best_global_idx])
                T_best_curr = t[best_global_idx]
                best_coords_curr = torch.zeros_like(coords_original)
                for ens_idx in range(B_ensembles):
                    best_coords_curr[ens_idx] = (coords_pre[ens_idx] - centroid) @ R_best_curr[ens_idx].T + centroid + T_best_curr[ens_idx]
                
                rmsds_per_member = []
                for ens_idx in range(B_ensembles):
                    coords_aligned = best_coords_curr[ens_idx].detach().contiguous()
                    coords_target = coords_target_ref[ens_idx].detach().to(device=device, dtype=dtype).contiguous()
                    
                    if coords_aligned.shape != coords_target.shape:
                        min_atoms = min(coords_aligned.shape[0], coords_target.shape[0])
                        coords_aligned = coords_aligned[:min_atoms]
                        coords_target = coords_target[:min_atoms]
                    
                    diff = coords_aligned - coords_target
                    squared_diffs = (diff ** 2).sum(dim=-1)
                    mean_squared_diff = squared_diffs.mean()
                    rmsd_member = torch.sqrt(mean_squared_diff).item() if mean_squared_diff > 1e-10 else 0.0
                    rmsds_per_member.append(rmsd_member)
                
                best_rmsd = sum(rmsds_per_member) / len(rmsds_per_member) if rmsds_per_member else None
                best_rmsd_min = min(rmsds_per_member) if rmsds_per_member else None
                best_rmsd_max = max(rmsds_per_member) if rmsds_per_member else None
                
                if max(step_scores) < 0.3 and best_rmsd is not None and best_rmsd < 0.01:
                    diff = best_coords_curr[0] - coords_target_ref[0]
                    max_diff = diff.abs().max().item()
                    if max_diff < 0.1:
                        best_rmsd = float('nan')

        # Multi-stage pruning for better stability
        if it == pruning_iteration and len(active_idx) > n_keep_after_pruning:
            scores_t = torch.tensor(step_scores, device=device)
            best_cc_at_prune = scores_t.max().item()
            
            # If second_pruning_iteration is set, first prune keeps 3x n_keep_after_pruning
            if second_pruning_iteration is not None and second_pruning_iteration > pruning_iteration:
                n_keep_first = min(3 * n_keep_after_pruning, len(active_idx))
                if len(active_idx) > n_keep_first:
                    keep_local = torch.topk(scores_t, n_keep_first).indices
                    active_idx = active_idx[keep_local]
                    if verbose:
                        print(f" [Pruning 1/2] Iter {it}: Kept {n_keep_first}. Best CC: {best_cc_at_prune:.4f}")
            else:
                # Single pruning (original behavior)
                keep_local = torch.topk(scores_t, n_keep_after_pruning).indices
                active_idx = active_idx[keep_local]
                if verbose:
                    print(f" [Pruning] Iter {it}: Kept {n_keep_after_pruning}. Best CC: {best_cc_at_prune:.4f}")
            
            # Warn if CC is low - might be stuck in local minimum
            if best_cc_at_prune < min_cc_for_convergence and verbose:
                print(f" [Warning] Best CC ({best_cc_at_prune:.4f}) < {min_cc_for_convergence:.2f} after pruning. May be stuck in local minimum.")
        
        # Second pruning stage (if enabled)
        if second_pruning_iteration is not None and it == second_pruning_iteration and len(active_idx) > n_keep_after_pruning:
            scores_t = torch.tensor(step_scores, device=device)
            keep_local = torch.topk(scores_t, n_keep_after_pruning).indices
            active_idx = active_idx[keep_local]
            if verbose:
                print(f" [Pruning 2/2] Iter {it}: Kept {n_keep_after_pruning}. Best CC: {scores_t.max():.4f}")

        # Adaptive reinitialization: replace poor-performing hypotheses with new random rotations
        if adaptive_reinit:
            if adaptive_reinit_iterations is None:
                # Default: check at iterations 5, 10, 15 (before aggressive pruning)
                check_iters = [5, 10, 15]
            else:
                check_iters = adaptive_reinit_iterations
            
            if it in check_iters and len(active_idx) > 1:
                scores_t = torch.tensor(step_scores, device=device)
                best_cc = scores_t.max().item()
                
                # Determine threshold: either relative to best or absolute
                if adaptive_reinit_cc_threshold is not None:
                    threshold = adaptive_reinit_cc_threshold
                else:
                    # Auto: reinitialize if CC < 50% of best (but at least 0.1)
                    threshold = max(best_cc * 0.5, 0.1)
                
                # Find poor performers
                poor_mask = scores_t < threshold
                n_poor = poor_mask.sum().item()
                
                # Reinitialize bottom fraction (or those below threshold)
                n_to_reinit = max(1, int(len(active_idx) * adaptive_reinit_fraction))
                if n_poor > 0:
                    # Reinitialize worst performers
                    _, worst_indices = torch.topk(scores_t, n_to_reinit, largest=False)
                else:
                    # If all are above threshold, reinitialize bottom fraction anyway
                    _, worst_indices = torch.topk(scores_t, n_to_reinit, largest=False)
                
                if len(worst_indices) > 0:
                    with torch.no_grad():
                        from cryoforward.utils.rigid_transform import _matrix_to_axis_angle
                        
                        n_reinit = len(worst_indices)
                        new_rots = random_rotation_matrices_haar(n_reinit, device=device)
                        
                        # Reinitialize rotation parameters
                        for idx, hyp_idx in enumerate(active_idx[worst_indices]):
                            w_param.data[hyp_idx] = _matrix_to_axis_angle(new_rots[idx:idx+1])[0]
                            # Keep translation near current best (small jitter)
                            t.data[hyp_idx] = t.data[active_idx[torch.argmax(scores_t)]].clone()
                            if t_init_box_edge_voxels:
                                t_jitter = (torch.rand(3, device=device) - 0.5) * 2.0 * t_init_box_edge_voxels * lattice.voxel_sizes_in_A[0]
                                t.data[hyp_idx] += t_jitter
                        
                        if verbose:
                            worst_cc = scores_t[worst_indices].min().item()
                            print(f" [Adaptive Reinit] Iter {it}: Reinitialized {n_reinit} poor hypotheses (worst CC={worst_cc:.4f}, best={best_cc:.4f})")

        # LR decay: trigger on plateau OR when CC reaches threshold (convergence phase)
        final_pruning_iter = second_pruning_iteration if second_pruning_iteration is not None else pruning_iteration
        warmup_complete = it > lr_decay_warmup_steps
        current_best_cc = max(step_scores)
        
        # Check if we should decay: plateau detected OR CC reached threshold (convergence phase)
        # Only trigger plateau decay if CC is above minimum threshold (prevents aggressive decay when stuck)
        # Use higher threshold during fine-tuning (CC >= 0.65) to allow more iterations for fine convergence
        effective_plateau_threshold = plateau_threshold
        if lr_plateau_threshold_high_cc is not None and current_best_cc >= 0.65:
            effective_plateau_threshold = lr_plateau_threshold_high_cc
        
        should_decay_plateau = (plateau_counter >= effective_plateau_threshold and 
                               current_best_cc >= lr_plateau_min_cc)
        # CC-based decay: only if threshold reached AND cooldown period has passed
        # Use longer cooldown during fine-tuning (CC >= 0.65) to allow faster convergence to 0.1 Å
        effective_cc_cooldown = cc_decay_cooldown
        if current_best_cc >= 0.65:
            effective_cc_cooldown = int(cc_decay_cooldown * 1.5)  # 50% longer cooldown during fine-tuning
        
        cc_decay_cooldown_passed = (it - last_cc_decay_iter) >= effective_cc_cooldown
        should_decay_cc = (lr_decay_cc_threshold is not None and 
                          current_best_cc >= lr_decay_cc_threshold and
                          cc_decay_cooldown_passed and
                          it > max(final_pruning_iter + 5, 10))  # At least 10 iters after pruning
        
        if use_lr_decay and warmup_complete and (should_decay_plateau or should_decay_cc) and it > max(final_pruning_iter + 5, 15):
            lr_decayed = False
            for param_group in opt.param_groups:
                old_lr = param_group['lr']
                new_lr = old_lr * lr_decay_factor
                if 'original_lr' in param_group:
                    min_lr = param_group['original_lr'] * 0.1
                else:
                    param_group['original_lr'] = old_lr
                    min_lr = old_lr * 0.1
                
                if new_lr >= min_lr:
                    param_group['lr'] = new_lr
                    lr_decayed = True
                else:
                    param_group['lr'] = min_lr
                    lr_decayed = True
            
            if lr_decayed:
                # Track if this was a CC-based decay
                if should_decay_cc:
                    last_cc_decay_iter = it
                
                current_lrs = [pg['lr'] for pg in opt.param_groups]
                if verbose:
                    if should_decay_cc and not should_decay_plateau:
                        reason = f"CC threshold reached (CC={current_best_cc:.4f} >= {lr_decay_cc_threshold:.2f})"
                    elif should_decay_plateau:
                        reason = f"Plateau detected ({plateau_counter} iters)"
                    else:
                        reason = "Both plateau and CC threshold"
                    print(f" [LR Decay] Iter {it}: {reason}, reducing LR by {lr_decay_factor}x (LRs: {[f'{lr:.4f}' for lr in current_lrs]})")
            plateau_counter = 0
        
        if not skip_optimizer_step:
            t_prev = t.detach().clone()
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()
            
            if per_step_t_cap_voxels:
                with torch.no_grad():
                    delta = t - t_prev
                    scale = torch.clamp(
                        (per_step_t_cap_voxels * lattice.voxel_sizes_in_A[0]) / (delta.norm(dim=-1, keepdim=True) + 1e-12),
                        max=1.0
                    )
                    t.copy_(t_prev + delta * scale)
            
            # Update EMA if enabled
            if use_ema and ema_w is not None:
                with torch.no_grad():
                    ema_w.mul_(ema_decay).add_(w_param.data, alpha=1 - ema_decay)
                    ema_t.mul_(ema_decay).add_(t.data, alpha=1 - ema_decay)

        if (it % print_every == 0 or it == steps) and verbose:
            if best_rmsd is not None:
                if best_rmsd_min is not None and best_rmsd_max is not None:
                    print(f" Iter {it}/{steps}: Best CC={max(step_scores):.4f} | Active={len(active_idx)} | RMSD={best_rmsd:.4f} Å [{best_rmsd_min:.4f}, {best_rmsd_max:.4f}]")
                else:
                    print(f" Iter {it}/{steps}: Best CC={max(step_scores):.4f} | Active={len(active_idx)} | RMSD={best_rmsd:.4f} Å")
            else:
                print(f" Iter {it}/{steps}: Best CC={max(step_scores):.4f} | Active={len(active_idx)} | RMSD=N/A (no target provided)")
        history.append(step_scores)

    with torch.no_grad():
        final = torch.tensor(history[-1], device=device)
        final_best_cc = final.max().item()
        
        # Always use best-ever if it's better than final (EMA only helps during optimization, not for final selection)
        if best_ever_w is not None and best_ever_cc >= final_best_cc:
            R_best = rodrigues_batch(best_ever_w)
            T_best = best_ever_t
            best_score = best_ever_cc
            if best_ever_rmsd is not None:
                if best_ever_rmsd_min is not None and best_ever_rmsd_max is not None and B_ensembles > 1:
                    rmsd_str = f", RMSD={best_ever_rmsd:.4f} Å [{best_ever_rmsd_min:.4f}, {best_ever_rmsd_max:.4f}]"
                else:
                    rmsd_str = f", RMSD={best_ever_rmsd:.4f} Å"
            else:
                rmsd_str = ""
            if verbose:
                print(f" [Best-Ever] Using iter {best_ever_iter} (CC={best_ever_cc:.4f}{rmsd_str}) instead of final iter (CC={final_best_cc:.4f})")
        else:
            best_loc = torch.argmax(final)
            best_glob = active_idx[best_loc]
            R_best = rodrigues_batch(w_param[best_glob])
            T_best = t[best_glob]
            best_score = final_best_cc
        
        best_coords = torch.zeros_like(coords_original)
        for ens_idx in range(B_ensembles):
            best_coords[ens_idx] = (coords_pre[ens_idx] - centroid) @ R_best[ens_idx].T + centroid + T_best[ens_idx]
        atom_stack.atom_coordinates = best_coords
        
        # Compute composed transformation: best_coords = coords_original @ R_comp.T + T_comp
        R_comp = torch.zeros(B_ensembles, 3, 3, device=device, dtype=dtype)
        T_comp = torch.zeros(B_ensembles, 3, device=device, dtype=dtype)
        
        for ens_idx in range(B_ensembles):
            orig_centroid = coords_original[ens_idx].mean(0)
            R_comp[ens_idx] = R_best[ens_idx] @ R_pre_init[ens_idx]
            T_comp[ens_idx] = centroid + T_best[ens_idx] - orig_centroid @ R_comp[ens_idx].T
        
        final_rmsd = None
        if compute_rmsd:
            rmsds_per_member = []
            for ens_idx in range(B_ensembles):
                coords_aligned = best_coords[ens_idx].detach().contiguous()
                coords_target = coords_target_ref[ens_idx].detach().to(device=device, dtype=dtype).contiguous()
                if coords_aligned.shape[0] != coords_target.shape[0]:
                    min_atoms = min(coords_aligned.shape[0], coords_target.shape[0])
                    coords_aligned = coords_aligned[:min_atoms]
                    coords_target = coords_target[:min_atoms]
                diff = coords_aligned - coords_target
                mean_squared_diff = (diff ** 2).sum(dim=-1).mean()
                rmsd_member = torch.sqrt(mean_squared_diff).item() if mean_squared_diff > 1e-10 else 0.0
                rmsds_per_member.append(rmsd_member)
            
            final_rmsd = sum(rmsds_per_member) / len(rmsds_per_member) if rmsds_per_member else None
            final_rmsd_min = min(rmsds_per_member) if rmsds_per_member else None
            final_rmsd_max = max(rmsds_per_member) if rmsds_per_member else None
            if B_ensembles > 1:
                if final_rmsd_min is not None and final_rmsd_max is not None:
                    rmsd_str = f"{final_rmsd:.4f} Å [{final_rmsd_min:.4f}, {final_rmsd_max:.4f}] (avg over {B_ensembles} members: {[f'{r:.4f}' for r in rmsds_per_member]})"
                else:
                    rmsd_str = f"{final_rmsd:.4f} Å (avg over {B_ensembles} members: {[f'{r:.4f}' for r in rmsds_per_member]})"
            else:
                rmsd_str = f"{final_rmsd:.4f} Å"
            print(f" [Final] Best CC={best_score:.4f} | RMSD={rmsd_str}")

    return {
        "best_ensemble_coords": best_coords,
        "best_score": best_score,
        "best_rmsd": final_rmsd,
        "R_composed": R_comp,
        "T_composed": T_comp,
        "scores_history": history
    }

