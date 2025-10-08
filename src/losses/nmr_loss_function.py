from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm
from biotite.structure.io import load_structure

from .s_2_loss_function import S2LossFunction
from ..protenix.metrics.rmsd import self_aligned_rmsd
from ..utils.io import load_pdb_atom_locations
from ..utils.hydrogen_addition import (
    FragmentLibrary,
    AtomNameLibrary,
    get_hydrogen_names,
)
from .abstract_loss_funciton import AbstractLossFunction  # (typo kept to match project)

# ----------------------------- small utilities ----------------------------- #

def methyl_group_names(nmr_name: str, residue_name: str) -> List[str]:
    if residue_name == "LEU" and nmr_name == "MD1":
        return ["HD11", "HD12", "HD13"]
    if residue_name == "LEU" and nmr_name == "MD2":
        return ["HD21", "HD22", "HD23"]
    if residue_name == "ALA" and nmr_name == "MB":
        return ["HB1", "HB2", "HB3"]
    if residue_name == "MET" and nmr_name == "ME":
        return ["HE1", "HE2", "HE3"]
    if residue_name == "ILE" and nmr_name == "MG":
        return ["HG21", "HG22", "HG23"]
    if residue_name == "ILE" and nmr_name == "MD":
        return ["HD11", "HD12", "HD13"]
    if residue_name == "VAL" and nmr_name == "MG1":
        return ["HG11", "HG12", "HG13"]
    if residue_name == "VAL" and nmr_name == "MG2":
        return ["HG21", "HG22", "HG23"]
    if residue_name == "THR" and nmr_name == "MG":
        return ["HG21", "HG22", "HG23"]
    raise ValueError("Unknown methyl hydrogen to pdb conversion")


def q_group_names(nmr_name: str, residue_name: str) -> List[str]:
    if residue_name == "TYR" and nmr_name == "QD":
        return ["HD1"]
    if residue_name == "TYR" and nmr_name == "QE":
        return ["HE1"]
    if residue_name == "PHE" and nmr_name == "QD":
        return ["HD1", "HD2"]
    if residue_name == "PHE" and nmr_name == "QE":
        return ["HE1", "HE2"]
    if residue_name == "TRP" and nmr_name == "QD":
        return ["HD1"]
    if residue_name == "TRP" and nmr_name == "QE":
        return ["HE1"]
    if residue_name == "TRP" and nmr_name == "QZ":
        return ["HZ2"]
    if residue_name == "LYS" and nmr_name == "QZ":
        return ["HZ1", "HZ2", "HZ3"]
    raise ValueError("Unknown q hydrogen to pdb conversion")


# ------------------------------- data holders ------------------------------ #
@dataclass
class ChainCtx:
    """Cached, per-chain context for a forward pass."""
    x0: torch.Tensor
    atom_arr: Any
    H: torch.Tensor
    Hnames: List[Tuple[int, str, str]]


# ------------------------------- main loss -------------------------------- #
class NMRLossFunction(AbstractLossFunction):
    """NOE loss with within-chain and explicit multi-chain (e.g., AB, BC) support."""

    # ------------------------------ init/setup ----------------------------- #
    def __init__(
        self,
        restraint_file,
        pdb_file,
        atom_array=None,
        device: str = "cpu",
        iid_loss: bool = False,
        methyl_relax_file=None,
        methyl_relax_scale: float = 0.0,
        methyl_rdc_file=None,
        methyl_rdc_scale: float = 0.0,
        amide_rdc_file=None,
        amide_rdc_scale: float = 0.0,
        amide_relax_file=None,
        amide_relax_scale: float = 0.0,
        noe_scale: float = 1.0,
    ):
        # restraints
        self.nmr_data = pd.read_csv(restraint_file)
        self.nmr_data = self.nmr_data[self.nmr_data["type"] == "NOE"].reset_index(drop=True)

        # bounds 
        self.nmr_data["lower_bound"] = self.nmr_data["lower_bound"].apply(lambda x: 0 if x == "." else x)
        self.lower_bound = torch.tensor(self.nmr_data["lower_bound"], dtype=torch.float32, device=device)
        self.upper_bound = torch.tensor(self.nmr_data["upper_bound"], dtype=torch.float32, device=device)

        # optional chain columns
        if "chain1" in self.nmr_data.columns:
            self.chain1_col = np.array(self.nmr_data["chain1"])
            self.chain2_col = np.array(self.nmr_data["chain2"])
            self.single_chain_mask_np = (self.nmr_data["chain1"] == self.nmr_data["chain2"]).values
            self.multi_chain_mask_np = (self.nmr_data["chain1"] != self.nmr_data["chain2"]).values
        else:
            self.chain1_col = None
            self.chain2_col = None
            self.single_chain_mask_np = np.ones(len(self.nmr_data), dtype=bool)
            self.multi_chain_mask_np = np.zeros(len(self.nmr_data), dtype=bool)

        self.device = device
        self.atom_array = atom_array
        self.iid_loss = iid_loss
        self.noe_scale = noe_scale

        self.fragment_library = FragmentLibrary.standard_library()
        self.name_library = AtomNameLibrary.standard_library()
        self.reference_atom_locations = load_pdb_atom_locations(pdb_file).to(device)

        # OR grouping
        or_ids = torch.tensor(self.nmr_data["constrain_id"], dtype=torch.float32, device=device)
        self.within_chain_unique_or, self.within_chain_inverse_or_indices = torch.unique(or_ids[self.single_chain_mask_np], return_inverse=True)
        self.multi_chain_unique_or, self.multi_chain_inverse_or_indices = torch.unique(or_ids[self.multi_chain_mask_np], return_inverse=True)

        # restraint tuples used by gather
        self.hydrogen_guidance_params = self._build_guidance_params()

        # chain segmentation (assumes equal-length segments in atom_array per chain)
        self.chain_ids = np.unique(atom_array.chain_id)
        self.chain_ids = np.array([c[0] for c in self.chain_ids])
        self.n_chains = self.chain_ids.size
        self.chain_indices = self._compute_chain_segments(atom_array)
        self.num_constraints = len(self.within_chain_unique_or)*self.n_chains+ len(self.multi_chain_unique_or)


        # mapping of explicit cross-chain pairs -> constraint indices
        self.pair_to_indices: Dict[Tuple[str, str], List[int]] = self._build_pair_index_map()

        # order parameter losses (constructed with the first chain's topology)
        chain_atom_array0 = atom_array[self.chain_indices[0][0] : self.chain_indices[0][1]]
        self.methyl_relax_scale = methyl_relax_scale
        self.methyl_rdc_scale = methyl_rdc_scale
        self.amide_rdc_scale = amide_rdc_scale
        self.amide_relax_scale = amide_relax_scale
        self.methyl_relax_loss = S2LossFunction(chain_atom_array0, methyl_relax_file, device, type="methyl_relax") if methyl_relax_file else None
        self.methyl_rdc_loss = S2LossFunction(chain_atom_array0, methyl_rdc_file, device, type="methyl_rdc") if methyl_rdc_file else None
        self.amide_rdc_loss = S2LossFunction(chain_atom_array0, amide_rdc_file, device, type="amide_rdc") if amide_rdc_file else None
        self.amide_relax_loss = S2LossFunction(chain_atom_array0, amide_relax_file, device, type="amide_relax") if amide_relax_file else None

        # logging state
        self._reset_logs()

    # ---------------------------- light helpers ---------------------------- #
    def _reset_logs(self):
        self.last_loss = None
        self.methyl_rdc_loss_val = None
        self.methyl_relax_loss_val = None
        self.amide_rdc_loss_val = None
        self.amide_relax_loss_val = None
        self.lb_loss_val = None
        self.ub_loss_val = None
        self.constraints_satisfied_ub = None
        self.constraints_satisfied_lb = None
        self.noe_loss_within_chain_val = None
        self.noe_loss_multi_chain_val = None

    def _build_guidance_params(self) -> List[Tuple[int, str, str, int, str, str]]:
        """(res1_num, res1_id, atom1, res2_num, res2_id, atom2) per restraint row."""
        out = []
        for _, row in tqdm(self.nmr_data.iterrows(), total=len(self.nmr_data), desc="generating comparison indices"):
            out.append(
                (
                    row["residue1_num"],
                    row["residue1_id"],
                    row["atom1"],
                    row["residue2_num"],
                    row["residue2_id"],
                    row["atom2"],
                )
            )
        return out

    def _compute_chain_segments(self, atom_array) -> List[Tuple[int, int]]:
        """Compute [start, end) slices per chain (assumes equal-length chains)."""
        n_total = len(atom_array)
        n_per_chain = n_total // self.n_chains
        starts = np.arange(0, n_total, n_per_chain)
        stops = starts + n_per_chain
        return list(zip(starts, stops))

    def _build_pair_index_map(self) -> Dict[Tuple[str, str], List[int]]:
        """Map explicit (chain1, chain2) to row indices for cross-chain constraints."""
        mapping: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        if self.chain1_col is None:
            return mapping
        for k, (c1, c2) in enumerate(zip(self.chain1_col, self.chain2_col)):
            if c1 != c2:
                mapping[(c1, c2)].append(k)
        return mapping

    # --------------------------- hydrogen utilities ------------------------ #
    def _make_hname_index(self, hydrogen_names: Sequence[Tuple[int, str, str]]) -> Dict[Tuple[int, str, str], int]:
        return {tuple(identifier): idx for idx, identifier in enumerate(hydrogen_names)}

    def _compute_group_coord(
        self,
        hname_to_idx: Dict[Tuple[int, str, str], int],
        H_batch: torch.Tensor,
        cond: Tuple[int, str, str],
        kind: str,
    ) -> torch.Tensor:
        """Average coordinates of methyl (M) or Q groups."""
        resnum, resid, atom = cond
        names = methyl_group_names(atom, resid) if kind == "M" else q_group_names(atom, resid)
        idxs = [hname_to_idx[(resnum, resid, nm)] for nm in names]
        return H_batch[:, idxs].mean(dim=1)

    def _resolve_heavy_atom(self, cond: Tuple[int, str, str], x0: torch.Tensor, atom_arr) -> Optional[torch.Tensor]:
        """Fallback coordinate when hydrogens are not defined for an atom selection."""
        arr_names = np.array([a.atom_name for a in atom_arr])
        arr_resid = np.array([a.res_id for a in atom_arr])
        try:
            idx = np.where((arr_names == cond[2]) & (arr_resid == cond[0]))[0][0]
            return x0[:, idx]
        except IndexError:
            # we'll mark the corresponding mask entry False upstream
            print(f"hydrogen was not added {cond}")
            return None

    def _get_atom_coord_any(
        self,
        cond: Tuple[int, str, str],
        hname_to_idx: Dict[Tuple[int, str, str], int],
        H_batch: torch.Tensor,
        x0: torch.Tensor,
        atom_arr,
    ) -> Optional[torch.Tensor]:
        """Return atom coord from hydrogens if possible, else heavy-atom fallback."""
        try:
            if "M" in cond[2]:
                return self._compute_group_coord(hname_to_idx, H_batch, cond, "M")
            if "Q" in cond[2]:
                return self._compute_group_coord(hname_to_idx, H_batch, cond, "Q")
            return H_batch[:, hname_to_idx[cond]]
        except KeyError:
            if  "#" in cond[2]:
                # average cond[2][:-1]1 and cond[2][:-1]2
                positions = []
                for i in [1, 2]:
                    temp_cond = (cond[0], cond[1], cond[2][:-1] + f"{i}")
                    pos = self._resolve_heavy_atom(temp_cond, x0, atom_arr)
                    positions += [pos]
                return torch.stack(positions, dim=1).mean(dim=1)
                
            return self._resolve_heavy_atom(cond, x0, atom_arr)

    # -------------------------- core gather/evaluate ------------------------ #
    def _gather_coords_indices(
        self,
        idxs: Sequence[int],
        left: ChainCtx,
        right: ChainCtx,
        mask: torch.Tensor,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """Gather (B, K, 3) coord tensors for a set of constraint indices, toggling mask for unresolvable rows."""
        if not idxs:
            return None

        L = self._make_hname_index(left.Hnames)
        R = self._make_hname_index(right.Hnames)

        atoms1, atoms2 = [], []

        for i in idxs:
            r1_num, r1_id, a1, r2_num, r2_id, a2 = self.hydrogen_guidance_params[i]
            c1 = (r1_num, r1_id, a1)
            c2 = (r2_num, r2_id, a2)

            p1 = self._get_atom_coord_any(c1, L, left.H, left.x0, left.atom_arr)
            if p1 is None:
                mask[i] = False
                continue

            p2 = self._get_atom_coord_any(c2, R, right.H, right.x0, right.atom_arr)
            if p2 is None:
                mask[i] = False
                continue

            atoms1.append(p1)
            atoms2.append(p2)

        if not atoms1:
            return None

        A = torch.stack(atoms1, dim=0).permute(1, 0, 2)  # (B, K, 3)
        B = torch.stack(atoms2, dim=0).permute(1, 0, 2)  # (B, K, 3)
        return A, B

    def _integrate_or_conditions(self, curr_loss: torch.Tensor, mask: torch.Tensor, within_chain: bool) -> torch.Tensor:
        """Reduce per-constraint losses into per-OR-group minima."""
        # curr_loss: (B, K_selected)
        unique_or = self.within_chain_unique_or if within_chain else self.multi_chain_unique_or
        min_vals = torch.zeros(
            (curr_loss.shape[0], len(unique_or)),
            dtype=curr_loss.dtype,
            device=curr_loss.device,
        )
        inverse_or_indices = self.within_chain_inverse_or_indices if within_chain else self.multi_chain_inverse_or_indices
        inv_idx = inverse_or_indices[None].repeat(curr_loss.shape[0], 1)
        dim = 1
        reduced_mask = mask[self.single_chain_mask_np] if within_chain else mask[self.multi_chain_mask_np]
        return torch.scatter_reduce(min_vals, dim, inv_idx[..., reduced_mask], curr_loss, reduce="amin", include_self=False)

    def _compute_bound_losses(self, dist: torch.Tensor, mask: torch.Tensor, within_chain: bool) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute lb/ub hinge losses and apply OR-group reduction."""
        # dist: (B, K_selected) or (B, K_all) with mask True at selected positions
        ub = torch.relu(dist - self.upper_bound[None][:, mask])
        lb = torch.relu(self.lower_bound[None][:, mask] - dist)

        ub_or = self._integrate_or_conditions(ub, mask, within_chain)
        lb_or = self._integrate_or_conditions(lb, mask, within_chain)

        return ub_or.mean(), lb_or.mean(), ub_or, lb_or

    def _evaluate_index_block(
        self,
        idxs: Sequence[int],
        left: ChainCtx,
        right: ChainCtx,
        within_chain: bool = True,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Single call that gathers coords and returns (ub_val, lb_val, ub_or, lb_or)."""
        if not idxs:
            return None

        mask = torch.zeros_like(self.upper_bound, dtype=torch.bool, device=self.device)
        mask[torch.tensor(list(idxs), dtype=torch.long, device=self.device)] = True

        gathered = self._gather_coords_indices(idxs, left, right, mask)
        if gathered is None:
            return None

        A, B = gathered
        # NEW: Use modern torch.linalg.vector_norm for deterministic behavior
        dist = torch.linalg.vector_norm(A - B, dim=-1)  # (B, K)
        # OLD: Deprecated torch.norm (can be non-deterministic)
        # dist = (A - B).norm(dim=-1)  # (B, K)

        dist = dist if self.iid_loss else dist.mean(dim=0)[None]

        return self._compute_bound_losses(dist, mask, within_chain)

    # --------------------------- order-parameter S2 ------------------------- #
    def _apply_s2_losses(self, ctx: ChainCtx, time: torch.Tensor, structures=None, i=None, step=None) -> torch.Tensor:
        """Apply available S2 losses, log plots + optional bootstraps, return total."""
        total = 0.0
        for key in ["methyl_relax", "methyl_rdc", "amide_relax", "amide_rdc"]:
            s2_func: Optional[S2LossFunction] = getattr(self, f"{key}_loss")
            if not s2_func:
                continue

            val, pred = s2_func(ctx.x0, time, ctx.H, ctx.Hnames, structures=structures, i=i, step=step)
            setattr(self, f"{key}_loss_val", float(val.item()))
            total = total + val * getattr(self, f"{key}_scale", 0.0)

            fig = s2_func.plot_s2_values(pred, val)
            wandb.log({f"{key}": wandb.Image(fig)})
            plt.close(fig)
        return total
    # ----------------------------- main forward ---------------------------- #
    def __call__(self, x_0_hat: torch.Tensor, time: torch.Tensor, structures=None, i=None, step=None,):
        x_0_hat = x_0_hat.to(self.device)

        # Build per-chain contexts (also used for cross-chain)
        chain_cache = self._prepare_chain_cache(x_0_hat)

        total_loss = torch.tensor(0.0, device=self.device)
        ub_or_total = torch.tensor(0.0, device=self.device)
        lb_or_total = torch.tensor(0.0, device=self.device)

        noe_within = torch.tensor(0.0, device=self.device)
        noe_multi = torch.tensor(0.0, device=self.device)

        # ---- WITHIN-CHAIN: apply same within-chain constraints to every chain ---- #
        sc_indices = np.nonzero(self.single_chain_mask_np)[0].tolist()
        for cidx in range(self.n_chains):
            ctx = chain_cache[cidx]
            res = self._evaluate_index_block(sc_indices, ctx, ctx, within_chain=True)
            if res is not None:
                ub_val, lb_val, ub_or, lb_or = res
                self.ub_loss_val, self.lb_loss_val = ub_val, lb_val
                # accumulate scaled NOE components
                noe_within = noe_within + (ub_val+lb_val) * self.noe_scale
                ub_or_total = ub_or_total + (ub_or != 0).sum().item()
                lb_or_total = lb_or_total + (lb_or != 0).sum().item()

            # order-parameter signals for guidance
            total_loss = total_loss + self._apply_s2_losses(ctx, time, structures=structures, i=i, step=step)
        total_loss = total_loss + noe_within

        # ---- MULTI-CHAIN: only the explicit pairs in the restraints (e.g., A-B, B-C) ---- #
        pair_count = 0
        if np.any(self.multi_chain_mask_np):
            label_to_index = {lbl: i for i, lbl in enumerate(self.chain_ids)}
            for (c1, c2), idxs in self.pair_to_indices.items():
                if c1 not in label_to_index or c2 not in label_to_index:
                    continue
                i, j = label_to_index[c1], label_to_index[c2]
                res = self._evaluate_index_block(idxs, chain_cache[i], chain_cache[j], within_chain=False)
                if res is None:
                    continue
                ub_val, lb_val, ub_or, lb_or = res
                # accumulate scaled NOE components
                noe_multi = noe_multi + (ub_val+lb_val) * self.noe_scale
                ub_or_total = ub_or_total + (ub_or != 0).sum().item()
                lb_or_total = lb_or_total + (lb_or != 0).sum().item()
                pair_count += 1
                            # add to total loss
            total_loss = total_loss + noe_multi

        self.constraints_satisfied_ub = 1 - ub_or_total / self.num_constraints
        self.constraints_satisfied_lb = 1 - lb_or_total / self.num_constraints

        # finalize scalars for logging
        self.noe_loss_within_chain_val = float(noe_within.item())
        self.noe_loss_multi_chain_val = float(noe_multi.item())
        self.last_loss = float(total_loss.item())
        return total_loss, None

    # ----------------------------- cache builder --------------------------- #
    def _prepare_chain_cache(self, x_0_hat: torch.Tensor) -> Dict[int, ChainCtx]:
        """Add hydrogens once per chain and cache the results for reuse."""
        cache: Dict[int, ChainCtx] = {}
        for i, (start, stop) in enumerate(self.chain_indices):
            x0_c = x_0_hat[:, start:stop]
            arr_c = self.atom_array[start:stop]

            H_batch, naming_sample = self.fragment_library.calculate_hydrogen_coord_batch(
                x0_c, arr_c.bonds, arr_c.atom_name, arr_c.element, arr_c.res_name, self.device
            )
            Hnames = get_hydrogen_names(arr_c, naming_sample, self.name_library)

            cache[i] = ChainCtx(x0=x0_c, atom_arr=arr_c, H=H_batch, Hnames=Hnames)
        return cache

    # ------------------------------- logging api --------------------------- #
    def wandb_log(self, _x_0_hat: torch.Tensor) -> Dict[str, Any]:
        """Return a dict of last computed scalars; call after __call__."""
        return {
            "loss": self.last_loss,
            "methyl_rdc_loss": self.methyl_rdc_loss_val,
            "methyl_relax_loss": self.methyl_relax_loss_val,
            "amide_rdc_loss": self.amide_rdc_loss_val,
            "amide_relax_loss": self.amide_relax_loss_val,
            "lb_loss": self.lb_loss_val,
            "ub_loss": self.ub_loss_val,
            "constraints_satisfied_ub": self.constraints_satisfied_ub,
            "constraints_satisfied_lb": self.constraints_satisfied_lb,
            "noe_loss_within_chain": self.noe_loss_within_chain_val,
            "noe_loss_multi_chain": self.noe_loss_multi_chain_val,
        }
