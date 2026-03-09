import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader


def _install_dependency_stubs(monkeypatch):
    gemmi_mod = types.ModuleType("gemmi")
    gemmi_mod.read_ccp4_map = lambda path: None
    monkeypatch.setitem(sys.modules, "gemmi", gemmi_mod)

    cryo_pkg = types.ModuleType("cryoforward")
    cryo_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "cryoforward", cryo_pkg)

    class DummyLattice:
        def __init__(
            self,
            grid_dimensions,
            voxel_sizes_in_A,
            left_bottom_point_in_A,
            right_upper_point_in_A,
            sublattice_radius_in_A,
            dtype,
            device,
        ):
            self.grid_dimensions = torch.tensor(grid_dimensions, dtype=torch.long, device=device)
            self.voxel_sizes_in_A = torch.tensor(voxel_sizes_in_A, dtype=dtype, device=device)
            self.left_bottom_point_in_A = left_bottom_point_in_A
            self.right_upper_point_in_A = right_upper_point_in_A
            self.sublattice_radius_in_A = sublattice_radius_in_A

        @classmethod
        def from_grid_dimensions_and_voxel_sizes(cls, **kwargs):
            return cls(**kwargs)

    lattice_mod = types.ModuleType("cryoforward.lattice")
    lattice_mod.Lattice = DummyLattice
    monkeypatch.setitem(sys.modules, "cryoforward.lattice", lattice_mod)

    atom_stack_mod = types.ModuleType("cryoforward.atom_stack")

    class DummyAtomStack:
        def __init__(self, atom_coordinates, atomic_numbers, device=None):
            self.atom_coordinates = atom_coordinates
            self.atomic_numbers = atomic_numbers
            self.device = device
            self.bfactors = None
            self.occupancies = None

        @classmethod
        def from_coords_and_atomic_numbers(cls, atom_coordinates, atomic_numbers, device=None):
            return cls(atom_coordinates, atomic_numbers, device=device)

        def apply_rigid_transform(self, rotation, translation=None, center=None, copy=False):
            return self

    atom_stack_mod.AtomStack = DummyAtomStack
    monkeypatch.setitem(sys.modules, "cryoforward.atom_stack", atom_stack_mod)

    cryoesp_mod = types.ModuleType("cryoforward.cryoesp_calculator")

    def setup_fast_esp_solver(atom_stack, lattice, per_voxel_averaging=True, use_checkpointing=False, use_autocast=False):
        def compute_batch_from_coords(coords_batch, bfactors, atomic_numbers, occupancies):
            shape = coords_batch.shape[:2] + tuple(int(x) for x in lattice.grid_dimensions.tolist())
            return torch.zeros(shape, dtype=coords_batch.dtype, device=coords_batch.device)

        return None, compute_batch_from_coords

    cryoesp_mod.setup_fast_esp_solver = setup_fast_esp_solver
    monkeypatch.setitem(sys.modules, "cryoforward.cryoesp_calculator", cryoesp_mod)

    rmsd_mod = types.ModuleType("src.protenix.metrics.rmsd")

    def fake_self_aligned_rmsd(pred, true, mask):
        return torch.zeros(pred.shape[0], device=pred.device), pred, None, None

    rmsd_mod.self_aligned_rmsd = fake_self_aligned_rmsd
    monkeypatch.setitem(sys.modules, "src.protenix.metrics.rmsd", rmsd_mod)

    utils_io_mod = types.ModuleType("src.utils.io")

    def alignment_mask_by_chain(full_sequences, chains_to_align, sequence_types):
        return torch.ones(3, dtype=torch.bool)

    def load_pdb_atom_locations_full(
        pdb_file,
        full_sequences_dict,
        chains_to_read=None,
        return_elements=True,
        return_bfacs=True,
        return_mask=True,
        return_starting_indices=True,
    ):
        coords = torch.zeros((3, 3), dtype=torch.float32)
        mask = torch.ones(3, dtype=torch.bool)
        bfactors = torch.zeros(3, dtype=torch.float32)
        elements = torch.ones(3, dtype=torch.int64)
        starting_residue_indices = [0]
        return coords, mask, bfactors, elements, starting_residue_indices

    utils_io_mod.alignment_mask_by_chain = alignment_mask_by_chain
    utils_io_mod.load_pdb_atom_locations_full = load_pdb_atom_locations_full
    monkeypatch.setitem(sys.modules, "src.utils.io", utils_io_mod)


def _import_modules(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(repo_root))
    _install_dependency_stubs(monkeypatch)

    losses_pkg = types.ModuleType("src.losses")
    losses_pkg.__path__ = [str(repo_root / "src" / "losses")]
    monkeypatch.setitem(sys.modules, "src.losses", losses_pkg)

    for module_name in (
        "src.utils.cryoimage_renderer",
        "src.losses.em_images_loss_function",
    ):
        sys.modules.pop(module_name, None)

    renderer_mod = importlib.import_module("src.utils.cryoimage_renderer")
    loss_mod = importlib.import_module("src.losses.em_images_loss_function")
    return renderer_mod, loss_mod


def _write_projection_source(tmp_path: Path, name: str, projections: torch.Tensor) -> tuple[str, str]:
    pt_path = tmp_path / f"{name}.pt"
    json_path = tmp_path / f"{name}.json"
    rotations = torch.stack([torch.eye(3) for _ in range(int(projections.shape[0]))], dim=0)
    torch.save({"projections": projections, "rotations": rotations}, pt_path)
    meta = {
        "projection_axis": "z",
        "lattice": {
            "grid_dimensions": [2, 2, 1],
            "voxel_sizes": [1.0, 1.0, 1.0],
            "left_bottom": [0.0, 0.0, 0.0],
            "right_upper": [2.0, 2.0, 1.0],
            "sublattice_radius": 8.0,
            "projection_axis": 2,
            "collapse_projection_axis": True,
        },
    }
    json_path.write_text(json.dumps(meta))
    return str(pt_path), str(json_path)


def _make_loss_fn(tmp_path, monkeypatch, *, projections_by_conf, supervised_assignment_by_index):
    _, loss_mod = _import_modules(monkeypatch)

    def fake_setup_fast_solver(self):
        self._fast_renderers = [
            [object() for _ in self.reference_structures]
            for _ in range(self.dataset.num_conformations)
        ]

    def fake_align(self, x_0_hat, reference):
        return x_0_hat

    def fake_render(self, *, conformation_index, structure_index, aligned_structure, rotations):
        return torch.full((rotations.shape[0], 2, 2), float(structure_index), device=rotations.device)

    monkeypatch.setattr(
        loss_mod.CryoEM_Images_GuidanceLossFunction,
        "_setup_fast_solver",
        fake_setup_fast_solver,
    )
    monkeypatch.setattr(
        loss_mod.CryoEM_Images_GuidanceLossFunction,
        "_align_to_reference",
        fake_align,
    )
    monkeypatch.setattr(
        loss_mod.CryoEM_Images_GuidanceLossFunction,
        "_render_projection_batch",
        fake_render,
    )

    pt_paths = []
    json_paths = []
    for idx, projections in enumerate(projections_by_conf):
        pt_path, json_path = _write_projection_source(tmp_path, f"conf_{idx}", projections)
        pt_paths.append(pt_path)
        json_paths.append(json_path)

    mask = torch.ones(3, dtype=torch.bool)
    sequences = [{"sequence": "AAA", "count": 1}]

    return loss_mod.CryoEM_Images_GuidanceLossFunction(
        image_pt_files=pt_paths,
        image_json_files=json_paths,
        reference_pdbs=["ref_a.pdb", "ref_b.pdb"],
        mask=mask,
        sequences_dictionary=sequences,
        device="cpu",
        log_projection_every=0,
        log_projection_pairs=0,
        supervised_assignment_by_index=supervised_assignment_by_index,
        projection_batch_size=2,
        shuffle_projection_samples=False,
    )


def test_mixed_dataset_exposes_projection_level_samples(tmp_path, monkeypatch):
    renderer_mod, _ = _import_modules(monkeypatch)

    pt_a, json_a = _write_projection_source(
        tmp_path,
        "a",
        torch.stack([torch.zeros((2, 2)), torch.full((2, 2), 2.0)], dim=0),
    )
    pt_b, json_b = _write_projection_source(
        tmp_path,
        "b",
        torch.ones((1, 2, 2)),
    )

    dataset = renderer_mod.CryoImageDataset.from_paths(
        pt_paths=[pt_a, pt_b],
        json_paths=[json_a, json_b],
        device="cpu",
    )

    assert len(dataset) == 3
    assert dataset.num_conformations == 2

    first = dataset[0]
    last = dataset[2]
    assert first["conformation_index"] == 0
    assert first["rotation_index"] == 0
    assert last["conformation_index"] == 1
    assert last["rotation_index"] == 0

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    assert batch["projection"].shape == (2, 2, 2)
    assert torch.equal(batch["conformation_index"], torch.tensor([0, 0]))
    assert torch.equal(batch["rotation_index"], torch.tensor([0, 1]))


def test_dynamic_assignment_uses_lowest_loss_and_reports_metrics(tmp_path, monkeypatch):
    loss_fn = _make_loss_fn(
        tmp_path,
        monkeypatch,
        projections_by_conf=[
            torch.zeros((2, 2, 2)),
            torch.ones((2, 2, 2)),
        ],
        supervised_assignment_by_index=False,
    )

    x_0_hat = torch.zeros((2, 3, 3), dtype=torch.float32)
    loss, _, _ = loss_fn(x_0_hat, time=None)

    assert torch.isclose(loss, torch.tensor(0.0))
    assert loss_fn.last_loss_value == pytest.approx(0.0)
    assert loss_fn.last_assignment_accuracy == pytest.approx(1.0)
    assert loss_fn.last_assignment_macro_precision == pytest.approx(1.0)
    assert loss_fn.last_assignment_macro_recall == pytest.approx(1.0)
    assert loss_fn.last_assignment_macro_f1 == pytest.approx(1.0)
    assert loss_fn.last_assignment_margin_mean == pytest.approx(1.0)
    assert loss_fn.last_assignment_counts == {0: 2, 1: 2}
    assert torch.equal(
        loss_fn.last_assignment_confusion_matrix,
        torch.tensor([[2, 0], [0, 2]]),
    )

    wandb_log = loss_fn.wandb_log(x_0_hat)
    assert wandb_log["cryoimage/assignment_accuracy"] == pytest.approx(1.0)
    assert wandb_log["cryoimage/assigned_count_conf_0"] == pytest.approx(2.0)
    assert wandb_log["cryoimage/confusion_true_1_pred_1"] == pytest.approx(2.0)


def test_supervised_assignment_keeps_ground_truth_debug_path(tmp_path, monkeypatch):
    loss_fn = _make_loss_fn(
        tmp_path,
        monkeypatch,
        projections_by_conf=[
            torch.ones((2, 2, 2)),
            torch.zeros((2, 2, 2)),
        ],
        supervised_assignment_by_index=True,
    )

    x_0_hat = torch.zeros((2, 3, 3), dtype=torch.float32)
    loss, _, _ = loss_fn(x_0_hat, time=None)

    assert torch.isclose(loss, torch.tensor(1.0))
    assert loss_fn.last_assignment_mode == "supervised"
    assert loss_fn.last_assignment_accuracy == pytest.approx(1.0)
    assert loss_fn.last_assignment_counts == {0: 2, 1: 2}
    assert torch.equal(
        loss_fn.last_assignment_confusion_matrix,
        torch.tensor([[2, 0], [0, 2]]),
    )
