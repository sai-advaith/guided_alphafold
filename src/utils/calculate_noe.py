import json
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from .io import chain_segments_from_atom_array


# ============================== Group helpers ============================== #
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


# ============================== I/O utilities ============================= #
def _equal_segments(atom_array) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Infer chain IDs and per-chain [start, stop) segments.

    Delegates to the shared helper, which derives boundaries from contiguous chain_id
    runs so hetero-complexes are segmented correctly. The name is kept for callers; the
    "equal partitions" assumption it used to make is gone.
    """
    return chain_segments_from_atom_array(atom_array)


def _prefilter_restraints(
    nmr_data: pd.DataFrame,
    atom_array,
    chain_indices: List[Tuple[int, int]],
) -> pd.DataFrame:
    """
    Keep rows where at least the base heavy atoms seem resolvable in topology.
    (Conservative filter so we don't drop 'M/Q/#' rows that need expansion.)
    """
    present = set()
    names = np.array(atom_array.atom_name)
    resids = np.array(atom_array.res_id)
    for (a, b) in chain_indices:
        seg_names = names[a:b]
        seg_res = resids[a:b]
        for nm, rs in zip(seg_names, seg_res):
            present.add((rs, nm))

    def ok(row) -> bool:
        def allow(resnum: int, aname: str) -> bool:
            if (("M" in aname) or ("Q" in aname) or ("#" in aname)):
                return True
            if (resnum, aname) not in present:
                print(f"Filtering out NOE for missing atom: {row}")
            return (resnum, aname) in present

        return allow(row["residue1_num"], row["atom1"]) and allow(row["residue2_num"], row["atom2"])

    return nmr_data[nmr_data.apply(ok, axis=1)].reset_index(drop=True)


# ============================== Core evaluator ============================ #
class CalculateNOE:
    """
    Multi-chain evaluator with **per-chain/per-pair** accounting (no cross-pass merging).

    - WITHIN (chain1==chain2) rows are applied to each chain segment.
    - MULTI (chain1!=chain2) rows are applied to the listed chain pair only.
    - OR groups are reduced per-category (WITHIN vs MULTI), following project file.  # UPDATED
    - Totals are additive sums of nonzero (UB ∪ LB) over chains/pairs.
    """

    def __init__(self, restraint_file: str, atom_array, device: torch.device, start_residue_from: int = 1):
        # Load restraints
        nmr = pd.read_csv(restraint_file)
        nmr = nmr[nmr["type"] == "NOE"].reset_index(drop=True)

        # No residue-number shift here, deliberately. The structures being scored are read
        # back from written PDBs, which save_structure_full() numbers in the author
        # convention (start_residue_from..), the same convention the restraint file uses.
        # Only the guidance loss needs a shift, because there the topology comes from the
        # Protenix dataloader and is always indexed 1..n. start_residue_from is accepted so
        # callers can pass it uniformly, and to assert that assumption below.
        self.start_residue_from = int(start_residue_from or 1)

        # Topology & chains
        self.atom_array = atom_array
        self.device = device
        self.chain_ids, self.chain_indices = _equal_segments(atom_array)

        if self.start_residue_from != 1 and len(atom_array):
            observed_min = int(np.min(np.asarray(atom_array.res_id)))
            if observed_min < self.start_residue_from:
                print(
                    f"WARNING: structures are numbered from residue {observed_min} but the "
                    f"restraints assume numbering from {self.start_residue_from}. Restraints "
                    f"will not match. This usually means the structures were written before "
                    f"--start_residue_from was set; regenerate them."
                )

        # Filter rows conservatively
        self.nmr_data = _prefilter_restraints(nmr, atom_array, self.chain_indices)

        # Guard against a silently meaningless evaluation. The atom arrays of every
        # structure under comparison are intersected before this point, so a reference
        # without hydrogens (e.g. a "fixed"/cleaned PDB rather than the deposited NMR
        # ensemble) strips protons from all of them, and nearly every NOE row is dropped.
        # The surviving handful then tends to report zero violations, which reads as a
        # perfect fit rather than as an evaluation that never happened.
        n_before, n_after = len(nmr), len(self.nmr_data)
        if n_before and n_after / n_before < 0.5:
            n_hydrogens = int(sum(1 for name in np.array(atom_array.atom_name) if str(name).startswith("H")))
            print(
                f"WARNING: only {n_after}/{n_before} NOE restraints are resolvable against the "
                f"structures being compared ({len(atom_array)} atoms, {n_hydrogens} of them hydrogens). "
                f"Reported violation numbers cover only those {n_after} restraints and are not "
                f"comparable to a full evaluation."
            )
            if n_hydrogens == 0:
                print(
                    "WARNING: the compared atom set contains no hydrogens at all. If a reference "
                    "structure was supplied, use one that retains hydrogens (the deposited NMR "
                    "ensemble), not a hydrogen-stripped model."
                )

        # Bounds
        self.nmr_data["lower_bound"] = self.nmr_data["lower_bound"].apply(lambda x: 0 if x == "." else x)
        self.lower_bound = torch.tensor(self.nmr_data["lower_bound"], dtype=torch.float32, device=device)
        self.upper_bound = torch.tensor(self.nmr_data["upper_bound"], dtype=torch.float32, device=device)

        # ====== UPDATED: robust chain columns + per-category masks ======
        if {"chain1", "chain2"}.issubset(self.nmr_data.columns):
            self.single_chain_mask_np = (self.nmr_data["chain1"] == self.nmr_data["chain2"]).values
            self.multi_chain_mask_np  = (self.nmr_data["chain1"] != self.nmr_data["chain2"]).values
            self.chain1_col = self.nmr_data["chain1"].values
            self.chain2_col = self.nmr_data["chain2"].values
        else:
            # No multichain info -> treat everything as WITHIN, nothing as MULTI
            self.single_chain_mask_np = np.ones(len(self.nmr_data), dtype=bool)
            self.multi_chain_mask_np  = np.zeros(len(self.nmr_data), dtype=bool)
            self.chain1_col = None
            self.chain2_col = None

        # ====== UPDATED: per-category OR grouping ======
        or_ids_all = torch.tensor(self.nmr_data["constrain_id"].values, dtype=torch.long, device=device)
        self.within_unique_or, self.within_inverse_or = torch.unique(or_ids_all[self.single_chain_mask_np], return_inverse=True)
        self.multi_unique_or,  self.multi_inverse_or  = torch.unique(or_ids_all[self.multi_chain_mask_np],  return_inverse=True)

        # Keep easy splits as convenience (optional)
        self.inter_data = self.nmr_data[self.single_chain_mask_np]
        self.intra_data = self.nmr_data[self.multi_chain_mask_np]

        # Which within-chain rows belong to which chain (see _build_passes). None means
        # "no chain columns", i.e. apply every within-chain row to every chain.
        self.within_indices_by_label = None
        if self.chain1_col is not None:
            by_label = defaultdict(list)
            for k in np.nonzero(self.single_chain_mask_np)[0]:
                by_label[str(self.chain1_col[k])].append(int(k))
            self.within_indices_by_label = dict(by_label)

        # Denominator must match what is actually evaluated: with chain columns, the
        # per-chain OR-group counts summed over chains; without them, within_or * n_chains.
        if self.within_indices_by_label is None:
            within_total = len(self.within_unique_or) * len(self.chain_indices)
        else:
            or_ids_np = np.asarray(self.nmr_data["constrain_id"])
            within_total = sum(
                len(set(or_ids_np[self.within_indices_by_label.get(str(label), [])]))
                for label in self.chain_ids
            )
        self.total_constraints = int(within_total + len(self.multi_unique_or))

        # Pair index map (only if multi present)
        self.pair_to_indices: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        if np.any(self.multi_chain_mask_np):
            for k, (c1, c2) in enumerate(zip(self.chain1_col, self.chain2_col)):
                if c1 != c2:
                    self.pair_to_indices[(str(c1), str(c2))].append(k)

        # Cache per-chain name/res arrays
        self._names_resids: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        for i, (a, b) in enumerate(self.chain_indices):
            self._names_resids[i] = (
                np.array(self.atom_array[a:b].atom_name),
                np.array(self.atom_array[a:b].res_id),
            )

        # Label -> chain index
        self._label_to_index = {str(lbl): i for i, lbl in enumerate(self.chain_ids)}

    # ------------------------- coordinate collection ------------------------ #
    def _coords_single(
        self,
        structures: torch.Tensor,
        chain_idx: int,
        residue_num: int,
        atom_name: str,
    ) -> Optional[torch.Tensor]:
        """Return (S,3) or None if missing. Supports '#' -> average base1/base2."""
        a, b = self.chain_indices[chain_idx]
        names, resids = self._names_resids[chain_idx]
        seg = structures[:, a:b]

        if "#" in atom_name:
            base = atom_name[:-1]
            parts: List[torch.Tensor] = []
            for i in (1, 2):
                nm = f"{base}{i}"
                hits = np.where((names == nm) & (resids == residue_num))[0]
                if hits.size == 0:
                    return None
                parts.append(seg[:, hits[0], :])
            return torch.stack(parts, dim=1).mean(dim=1)

        hits = np.where((names == atom_name) & (resids == residue_num))[0]
        if hits.size == 0:
            return None
        return seg[:, hits[0]]

    def _coords_group(
        self,
        structures: torch.Tensor,
        chain_idx: int,
        residue_num: int,
        residue_id: str,
        atom_name: str,
        kind: str,
    ) -> Optional[torch.Tensor]:
        """Average coords for methyl (M) or Q groups; returns (S,3) or None if any member missing."""
        a, b = self.chain_indices[chain_idx]
        names, resids = self._names_resids[chain_idx]
        seg = structures[:, a:b, :]

        members = methyl_group_names(atom_name, residue_id) if kind == "M" else q_group_names(atom_name, residue_id)
        coords: List[torch.Tensor] = []
        for nm in members:
            hits = np.where((names == nm) & (resids == residue_num))[0]
            if hits.size == 0:
                return None
            coords.append(seg[:, hits[0], :])
        return torch.stack(coords, dim=1).mean(dim=1)

    def _coords_for_side(self, row, side: int, structures: torch.Tensor, chain_idx: int) -> Optional[torch.Tensor]:
        res_id = row[f"residue{side}_id"]
        atom = row[f"atom{side}"]
        res_num = row[f"residue{side}_num"]
        if "M" in atom:
            return self._coords_group(structures, chain_idx, res_num, res_id, atom, "M")
        if "Q" in atom:
            return self._coords_group(structures, chain_idx, res_num, res_id, atom, "Q")
        return self._coords_single(structures, chain_idx, res_num, atom)

    # ---------------------------- pass construction ------------------------ #
    def _gather_pass(
        self,
        structures: torch.Tensor,
        idxs: Sequence[int],
        left_chain_idx: int,
        right_chain_idx: int,
    ) -> Tuple[Optional[torch.Tensor], np.ndarray]:
        """For given row indices and chain indices, return D (S,K) and mask over all rows."""
        if not idxs:
            return None, np.zeros(len(self.nmr_data), dtype=bool)

        mask = np.zeros(len(self.nmr_data), dtype=bool)
        dists: List[torch.Tensor] = []

        for i in idxs:
            row = self.nmr_data.iloc[i]
            p1 = self._coords_for_side(row, 1, structures, left_chain_idx)
            p2 = self._coords_for_side(row, 2, structures, right_chain_idx)
            if (p1 is None) or (p2 is None):
                continue
            mask[i] = True
            dists.append(torch.linalg.vector_norm(p1 - p2, dim=-1))  # (S,)

        if not dists:
            return None, mask

        return torch.stack(dists, dim=1), mask  # (S,K)

    def _build_passes(self, structures: torch.Tensor) -> Tuple[
        List[Tuple[torch.Tensor, np.ndarray]], List[str], List[Tuple[str, str]]
    ]:
        """Create all WITHIN and MULTI passes with their kind and (left,right) chain labels."""
        per_pass: List[Tuple[torch.Tensor, np.ndarray]] = []
        kinds: List[str] = []
        pass_chain_meta: List[Tuple[str, str]] = []

        # WITHIN: with chain columns, each chain is evaluated only against the restraints
        # naming it. Without them, every chain gets every within-chain row, which is right
        # for a monomer or homo-oligomer but would otherwise score one chain's restraints
        # against another chain's coordinates in a hetero-complex.
        sc_indices_all = np.nonzero(self.single_chain_mask_np)[0].tolist()
        for ci, lbl in enumerate(self.chain_ids):
            if self.within_indices_by_label is None:
                sc_indices = sc_indices_all
            else:
                sc_indices = self.within_indices_by_label.get(str(lbl), [])
            D, m = self._gather_pass(structures, sc_indices, ci, ci)
            per_pass.append((D, m))
            kinds.append("within")
            pass_chain_meta.append((str(lbl), str(lbl)))

        # MULTI: explicit chain pairs (c1,c2)
        for (c1, c2), idxs in self.pair_to_indices.items():
            if (c1 in self._label_to_index) and (c2 in self._label_to_index):
                i, j = self._label_to_index[c1], self._label_to_index[c2]
                D, m = self._gather_pass(structures, idxs, i, j)
                per_pass.append((D, m))
                kinds.append("multi")
                pass_chain_meta.append((str(c1), str(c2)))

        return per_pass, kinds, pass_chain_meta

    # ------------------------------ OR reduction --------------------------- #
    def _or_reduce(
        self,
        lb: torch.Tensor,              # (K_sel, E)
        ub: torch.Tensor,              # (K_sel, E)
        inv: torch.Tensor,             # (K_sel,) long indices of OR group for each selected row
        or_count: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reduce losses into per-OR-group minima by choosing member with min(lb+ub)."""
        inv = inv.long()

        ub_g = torch.zeros(or_count, dtype=lb.dtype, device=lb.device)
        lb_g = torch.zeros(or_count, dtype=lb.dtype, device=lb.device)
        argmin_idx = torch.full((or_count,), -1, dtype=torch.long, device=lb.device)

        for g in range(or_count):
            sel = torch.nonzero(inv == g, as_tuple=False).flatten()
            if sel.numel() == 0:
                continue
            sums = ub.index_select(0, sel) + lb.index_select(0, sel)
            k = torch.argmin(sums)
            argmin_idx[g] = sel[k]
            ub_g[g] = ub.index_select(0, sel)[k]
            lb_g[g] = lb.index_select(0, sel)[k]

        return (
            ub_g,  
            lb_g,  
            argmin_idx,
        )

    def _or_losses_from_dist(
        self,
        D: torch.Tensor,       # (S,K_sel)
        mask: np.ndarray,      # (K_total,) True at rows actually present in D
        within: bool,          # UPDATED: choose WITHIN vs MULTI grouping
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mean over structures -> hinge -> OR-reduce (WITHIN or MULTI)."""
        D_mean = D.mean(dim=0)

        # Select bounds only for kept rows
        kept = torch.tensor(mask, device=self.device, dtype=torch.bool)
        ub = torch.relu(D_mean - self.upper_bound[kept])
        lb = torch.relu(self.lower_bound[kept] - D_mean)

        # Build the OR-index vector aligned to these kept rows
        if within:
            # indices in the WITHIN-subset space
            kept = kept[self.single_chain_mask_np]
            kept_within_idx = torch.nonzero(kept, as_tuple=False).squeeze(-1)
            inv_pool = self.within_inverse_or
            # We need the inverse indices only for rows in kept_within_idx, preserving order of 'kept'
            inv = inv_pool[kept_within_idx]
            or_count = len(self.within_unique_or)
        else:
            kept = kept[self.multi_chain_mask_np]
            kept_multi_idx = torch.nonzero(kept, as_tuple=False).squeeze(-1)
            inv_pool = self.multi_inverse_or
            inv = inv_pool[kept_multi_idx]
            or_count = len(self.multi_unique_or)

        return self._or_reduce(lb, ub, inv, or_count)

    # -------------------------- Violation accounting ------------------------ #
    def _accumulate(
        self,
        per_pass: List[Tuple[torch.Tensor, np.ndarray]],
        kinds: List[str],
        pass_chain_meta: List[Tuple[str, str]],
        subset: Sequence[int],
    ):
        """
        Do per-pass OR reduction and store results **without combining** passes:
        - WITHIN: per-chain dicts (ub/ub_values/lb/lb_values/argmin + counts)
        - MULTI: per-pair dicts (ub/ub_values/lb/lb_values/argmin + counts)
        """
        len_or_within = len(self.within_unique_or)
        len_or_multi  = len(self.multi_unique_or)
        chains = [str(c) for c in self.chain_ids]

        # WITHIN per-chain
        within_vals_by_chain_ub: Dict[str, List[List[float]]] = {}
        within_vals_by_chain_lb: Dict[str, List[List[float]]] = {}
        within_bool_by_chain_ub: Dict[str, List[List[bool]]] = {}
        within_bool_by_chain_lb: Dict[str, List[List[bool]]] = {}
        within_viols_by_chain: Dict[str, int] = {c: 0 for c in chains}
        within_argmin_by_chain: Dict[str, List[int]] = {c: [-1] * len_or_within for c in chains}

        # MULTI per-pair
        multi_details_by_pair: Dict[str, Dict[str, object]] = {}
        multi_argmin_by_pair: Dict[str, List[int]] = {}
        multi_viols_by_chain: Dict[str, int] = {c: 0 for c in chains}

        for (D, mask), kind, (lc, rc) in zip(per_pass, kinds, pass_chain_meta):
            if D is None or not mask.any():
                continue
            ub_or, lb_or, argmin = self._or_losses_from_dist(D, mask, within=(kind == "within"))

            ub_np = ub_or.detach().cpu().numpy()
            lb_np = lb_or.detach().cpu().numpy()
            ub_b = (ub_np > 0)
            lb_b = (lb_np > 0)

            # Map argmin (pass-local selection indices) back to global row indices
            mask_idx = np.nonzero(mask)[0]
            argmin_np = argmin.detach().cpu().numpy().astype(int)
            # global_arg = [-1] * (len_or_within if kind == "within" else len_or_multi)
            # for g in range(len(global_arg)):
            #     k = argmin_np[g, 0] if argmin_np.ndim == 2 else argmin_np[g]
            #     if 0 <= k < len(mask_idx):
            #         global_arg[g] = int(mask_idx[k])

            if kind == "within":
                within_vals_by_chain_ub[lc] = ub_np.tolist()
                within_vals_by_chain_lb[lc] = lb_np.tolist()
                within_bool_by_chain_ub[lc] = ub_b.tolist()
                within_bool_by_chain_lb[lc] = lb_b.tolist()
                within_viols_by_chain[lc]  = int((ub_b | lb_b).sum())
                within_argmin_by_chain[lc] = argmin_np.tolist()
            else:
                pair_key = f"{lc}-{rc}"
                multi_details_by_pair[pair_key] = {
                    "ub_values": ub_np.tolist(),
                    "lb_values": lb_np.tolist(),
                    "ub": ub_b.tolist(),
                    "lb": lb_b.tolist(),
                }
                multi_argmin_by_pair[pair_key] = argmin_np.tolist()
                viols = int((ub_b | lb_b).sum())
                multi_viols_by_chain[lc] += viols
                multi_viols_by_chain[rc] += viols

        return dict(
            within_vals_by_chain_ub=within_vals_by_chain_ub,
            within_vals_by_chain_lb=within_vals_by_chain_lb,
            within_bool_by_chain_ub=within_bool_by_chain_ub,
            within_bool_by_chain_lb=within_bool_by_chain_lb,
            within_viols_by_chain=within_viols_by_chain,
            within_argmin_by_chain=within_argmin_by_chain,
            multi_details_by_pair=multi_details_by_pair,
            multi_argmin_by_pair=multi_argmin_by_pair,
            multi_viols_by_chain=multi_viols_by_chain,
        )

    # ================================ Public API =========================== #
    def calculate_violations(
        self,
        per_pass: List[Tuple[torch.Tensor, np.ndarray]],
        kinds: List[str],
        pass_chain_meta: List[Tuple[str, str]],
        subset: Sequence[int],
    ):
        """
        Returns (in order):
          total_additive,
          within_total, multi_total,
          within_ub_values_by_chain, within_lb_values_by_chain, within_ub_bool_by_chain, within_lb_bool_by_chain,
          multi_ub_values_by_pair, multi_lb_values_by_pair, multi_ub_bool_by_pair, multi_lb_bool_by_pair,
          within_or_atom_indices_by_chain, multi_or_atom_indices_by_pair
        """
        bucket = self._accumulate(per_pass, kinds, pass_chain_meta, subset)

        # Totals
        within_total = int(sum(bucket["within_viols_by_chain"].values()))
        multi_total  = int(sum((np.array(v["ub"], dtype=bool) | np.array(v["lb"], dtype=bool)).sum()
                               for v in bucket["multi_details_by_pair"].values()))
        total_additive = int(within_total + multi_total)

        within_ub_values_by_chain = bucket["within_vals_by_chain_ub"]
        within_lb_values_by_chain = bucket["within_vals_by_chain_lb"]
        within_ub_bool_by_chain   = bucket["within_bool_by_chain_ub"]
        within_lb_bool_by_chain   = bucket["within_bool_by_chain_lb"]

        multi_ub_values_by_pair = {k: v["ub_values"] for k, v in bucket["multi_details_by_pair"].items()}
        multi_lb_values_by_pair = {k: v["lb_values"] for k, v in bucket["multi_details_by_pair"].items()}
        multi_ub_bool_by_pair   = {k: v["ub"]        for k, v in bucket["multi_details_by_pair"].items()}
        multi_lb_bool_by_pair   = {k: v["lb"]        for k, v in bucket["multi_details_by_pair"].items()}

        within_or_atom_indices_by_chain = bucket["within_argmin_by_chain"]
        multi_or_atom_indices_by_pair   = bucket["multi_argmin_by_pair"]

        return (
            total_additive,
            within_total, multi_total,
            within_ub_values_by_chain, within_lb_values_by_chain, within_ub_bool_by_chain, within_lb_bool_by_chain,
            multi_ub_values_by_pair, multi_lb_values_by_pair, multi_ub_bool_by_pair, multi_lb_bool_by_pair,
            within_or_atom_indices_by_chain, multi_or_atom_indices_by_pair,
        )

    def run(self, structures: torch.Tensor) -> Dict[str, object]:
        """
        structures: (S, N_atoms, 3) covering all chains concatenated to match atom_array order.
        Returns a dict with totals, WITHIN dicts (by chain), MULTI dicts (by pair), and argmins.
        """
        per_pass, kinds, pass_chain_meta = self._build_passes(structures)

        (
            ensemble_score,
            ensemble_within_score, ensemble_multi_score,
            ensemble_within_ub_values_by_chain, ensemble_within_lb_values_by_chain, ensemble_within_ub_by_chain, ensemble_within_lb_by_chain,
            ensemble_multi_ub_values_by_pair, ensemble_multi_lb_values_by_pair, ensemble_multi_ub_by_pair, ensemble_multi_lb_by_pair,
            ensemble_within_or_atom_indices, ensemble_multi_or_atom_indices,
        ) = self.calculate_violations(per_pass, kinds, pass_chain_meta, list(range(structures.shape[0])))

        viol_median = {}
        viol_mean = {}
        for k,v in ensemble_within_ub_values_by_chain.items():
            v = np.array(v) + np.array(ensemble_within_lb_values_by_chain[k])
            viol_median[k] = np.median(v[v.nonzero()[0]])
            viol_mean[k] = np.mean(v[v.nonzero()[0]])

        if self.total_constraints == 0:
            # Nothing was evaluable, so a percentage is meaningless. Fail with the cause
            # rather than a bare ZeroDivisionError from the line below.
            raise ValueError(
                "No restraints could be evaluated against the structures, so no metrics can "
                "be computed. The usual cause is a residue-numbering mismatch between the "
                "restraint file and the structures (check --start_residue_from), or a "
                "reference structure whose atom set does not overlap the restraints. See the "
                "'restraints are resolvable' warning above."
            )
        viol_percent = ensemble_score / self.total_constraints
        

        return {
            "viol_percent":viol_percent,
            "total_constraints": self.total_constraints,
            "viol_median": viol_median,
            "viol_mean": viol_mean,

            # category totals
            # "ensemble_within_score": ensemble_within_score,
            # "ensemble_intra_score":  ensemble_multi_score,  # kept old key for backward compat

            # # WITHIN (DICT BY CHAIN)
            # "ensemble_within_ub_values": json.dumps(ensemble_within_ub_values_by_chain),
            # "ensemble_within_lb_values": json.dumps(ensemble_within_lb_values_by_chain),
            # "ensemble_within_ub":        json.dumps(ensemble_within_ub_by_chain),
            # "ensemble_within_lb":        json.dumps(ensemble_within_lb_by_chain),

            # # MULTI (DICT BY PAIR)
            # "ensemble_intra_ub_values": json.dumps(ensemble_multi_ub_values_by_pair),
            # "ensemble_intra_lb_values": json.dumps(ensemble_multi_lb_values_by_pair),
            # "ensemble_intra_ub":        json.dumps(ensemble_multi_ub_by_pair),
            # "ensemble_intra_lb":        json.dumps(ensemble_multi_lb_by_pair),


            # # --- ARGMIN (within: by chain, intra: by pair) ---
            # "ensemble_within_or_atom_indices":     json.dumps(ensemble_within_or_atom_indices),
            # "ensemble_intra_or_atom_indices":      json.dumps(ensemble_multi_or_atom_indices),
        }
