import importlib
import json
import sys
import types
from pathlib import Path

import pytest
import torch


def _install_dependency_stubs(monkeypatch, *, lattice_cls=None):
    """Install lightweight stubs so the module can import without heavy deps."""
    # losses.abstract_loss_funciton
    abstract_mod = types.ModuleType("losses.abstract_loss_funciton")

    class AbstractLossFunction:
        pass

    abstract_mod.AbstractLossFunction = AbstractLossFunction
    monkeypatch.setitem(sys.modules, "losses.abstract_loss_funciton", abstract_mod)

    # utils.io
    utils_pkg = types.ModuleType("utils")
    utils_pkg.__path__ = []
    utils_io = types.ModuleType("utils.io")

    def alignment_mask_by_chain(full_sequences, chains_to_align, sequence_types):
        return torch.ones(3, dtype=torch.bool)

    def load_pdb_atom_locations_full(
        pdb_file,
        full_sequences_dict,
        chains_to_read=None,
        return_elements=True,
        return_bfacs=True,
        return_mask=True,
    ):
        coords = torch.zeros((3, 3))
        mask = torch.ones(3, dtype=torch.bool)
        bfactors = torch.zeros(3)
        elements = torch.ones(3, dtype=torch.int64)
        return coords, mask, bfactors, elements

    utils_io.alignment_mask_by_chain = alignment_mask_by_chain
    utils_io.load_pdb_atom_locations_full = load_pdb_atom_locations_full
    monkeypatch.setitem(sys.modules, "utils", utils_pkg)
    monkeypatch.setitem(sys.modules, "utils.io", utils_io)

    # cryoforward namespace + stubs
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
            self.grid_dimensions = grid_dimensions
            self.voxel_sizes_in_A = torch.tensor(
                voxel_sizes_in_A, dtype=dtype, device=device
            )
            self.left_bottom_point_in_A = left_bottom_point_in_A
            self.right_upper_point_in_A = right_upper_point_in_A
            self.sublattice_radius_in_A = sublattice_radius_in_A

        @classmethod
        def from_grid_dimensions_and_voxel_sizes(cls, **kwargs):
            return cls(**kwargs)

    if lattice_cls is None:
        lattice_cls = DummyLattice

    lattice_mod = types.ModuleType("cryoforward.lattice")
    lattice_mod.Lattice = lattice_cls
    monkeypatch.setitem(sys.modules, "cryoforward.lattice", lattice_mod)

    atom_stack_mod = types.ModuleType("cryoforward.atom_stack")

    class DummyAtomStack:
        def __init__(self, atom_coordinates, atomic_numbers, device=None):
            self.atom_coordinates = atom_coordinates
            self.atomic_numbers = atomic_numbers
            self.device = device
            self.bfactors = None

        @classmethod
        def from_coords_and_atomic_numbers(
            cls, atom_coordinates, atomic_numbers, device=None
        ):
            return cls(atom_coordinates, atomic_numbers, device=device)

        def rotate(self, rotation, center=None, copy=True):
            return self

    atom_stack_mod.AtomStack = DummyAtomStack
    monkeypatch.setitem(sys.modules, "cryoforward.atom_stack", atom_stack_mod)

    cryoesp_mod = types.ModuleType("cryoforward.cryoesp_calculator")

    def compute_volume_over_insertable_matrices(
        atom_stack,
        lattice,
        B,
        per_voxel_averaging=True,
        subvolume_mask_in_indices=None,
        use_checkpointing=False,
        verbose=False,
    ):
        return torch.zeros(lattice.grid_dimensions, dtype=torch.float32)

    cryoesp_mod.compute_volume_over_insertable_matrices = (
        compute_volume_over_insertable_matrices
    )
    monkeypatch.setitem(sys.modules, "cryoforward.cryoesp_calculator", cryoesp_mod)

    cryo_utils_pkg = types.ModuleType("cryoforward.utils")
    cryo_utils_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "cryoforward.utils", cryo_utils_pkg)

    rigid_mod = types.ModuleType("cryoforward.utils.rigid_transform")

    class DummyRotation:
        def __init__(self, matrix):
            self.matrix = matrix

        @classmethod
        def from_matrix(cls, matrix):
            return cls(matrix)

    rigid_mod.Rotation = DummyRotation
    monkeypatch.setitem(sys.modules, "cryoforward.utils.rigid_transform", rigid_mod)



def _import_em_module(monkeypatch):
    src_path = Path(__file__).resolve().parents[1] / "src"
    monkeypatch.syspath_prepend(str(src_path))
    _install_dependency_stubs(monkeypatch)
    module = importlib.import_module("losses.em_images_loss_function")
    monkeypatch.setitem(sys.modules, "losses.em_images_loss_function", module)
    return module


def test_init_validation_errors(monkeypatch):
    em_module = _import_em_module(monkeypatch)
    mask = torch.ones(3, dtype=torch.bool)
    sequences = [{"sequence": "AAA", "count": 1}]

    with pytest.raises(ValueError, match="image_pt_files"):
        em_module.CryoEM_Images_GuidanceLossFunction(
            [], None, ["ref.pdb"], mask, sequences
        )

    with pytest.raises(ValueError, match="reference_pdbs"):
        em_module.CryoEM_Images_GuidanceLossFunction(
            ["data.pt"], None, [], mask, sequences
        )

    with pytest.raises(ValueError, match="same length"):
        em_module.CryoEM_Images_GuidanceLossFunction(
            ["data.pt"], None, ["a.pdb", "b.pdb"], mask, sequences
        )

    with pytest.raises(ValueError, match="image_json_files"):
        em_module.CryoEM_Images_GuidanceLossFunction(
            ["data.pt"], ["a.json", "b.json"], ["a.pdb"], mask, sequences
        )


def test_load_dataset_reads_json_meta_and_projection_axis(tmp_path, monkeypatch):
    em_module = _import_em_module(monkeypatch)

    instance = em_module.CryoEM_Images_GuidanceLossFunction.__new__(
        em_module.CryoEM_Images_GuidanceLossFunction
    )
    instance.device = torch.device("cpu")

    projections = torch.zeros((2, 2, 2), dtype=torch.float32)
    rotations = torch.stack([torch.eye(3), torch.eye(3)], dim=0)
    pt_path = tmp_path / "sample.pt"
    torch.save({"projections": projections, "rotations": rotations}, pt_path)

    meta = {
        "projection_axis": "y",
        "lattice": {
            "grid_dimensions": [2, 2, 2],
            "voxel_sizes": [1.0, 2.0, 3.0],
            "left_bottom": [0, 0, 0],
            "right_upper": [2, 2, 2],
            "sublattice_radius": 8.0,
        },
    }
    json_path = tmp_path / "sample.json"
    json_path.write_text(json.dumps(meta))

    dataset = instance._load_dataset(str(pt_path), None)

    assert dataset["projection_axis"] == 1
    assert dataset["collapse_projection_axis"] is True
    assert torch.equal(dataset["projections"], projections)
    assert torch.equal(dataset["rotations"], rotations)
    assert dataset["lattice"].voxel_sizes_in_A.tolist() == [1.0, 2.0, 3.0]


def test_load_dataset_missing_lattice_meta_raises(tmp_path, monkeypatch):
    em_module = _import_em_module(monkeypatch)

    instance = em_module.CryoEM_Images_GuidanceLossFunction.__new__(
        em_module.CryoEM_Images_GuidanceLossFunction
    )
    instance.device = torch.device("cpu")

    pt_path = tmp_path / "bad.pt"
    torch.save(
        {
            "projections": torch.zeros((1, 1, 1)),
            "rotations": torch.eye(3).unsqueeze(0),
            "meta": {"lattice": {"grid_dimensions": [1, 1, 1]}},
        },
        pt_path,
    )

    with pytest.raises(ValueError, match="Missing lattice metadata"):
        instance._load_dataset(str(pt_path), None)


def test_align_to_reference_combines_masks(monkeypatch):
    em_module = _import_em_module(monkeypatch)

    instance = em_module.CryoEM_Images_GuidanceLossFunction.__new__(
        em_module.CryoEM_Images_GuidanceLossFunction
    )
    instance.AF3_to_pdb_mask = torch.tensor([True, False, True])
    instance.align_to_chain_mask = torch.tensor([True, True, False])

    captured = {}

    def fake_self_aligned_rmsd(pred, true, mask):
        captured["mask"] = mask
        return None, pred + 1.0, None, None

    monkeypatch.setattr(em_module, "self_aligned_rmsd", fake_self_aligned_rmsd)

    x_0_hat = torch.zeros((1, 3, 3))
    reference = {"coords": torch.zeros((3, 3))}

    aligned = instance._align_to_reference(x_0_hat, reference)

    expected_mask = torch.tensor([True, False, False])
    assert torch.equal(captured["mask"], expected_mask)
    assert torch.equal(aligned, x_0_hat + 1.0)


def test_call_averages_losses_and_sets_last_value(monkeypatch):
    em_module = _import_em_module(monkeypatch)

    instance = em_module.CryoEM_Images_GuidanceLossFunction.__new__(
        em_module.CryoEM_Images_GuidanceLossFunction
    )
    instance.loss_reduction = "mean"
    instance.last_loss_value = None

    dataset_one = {
        "projections": torch.zeros((2, 2)),
        "rendered": torch.ones((2, 2)),
    }
    dataset_two = {
        "projections": torch.ones((2, 2)),
        "rendered": torch.ones((2, 2)),
    }
    instance.datasets = [dataset_one, dataset_two]
    instance.reference_structures = [
        {
            "coords": torch.zeros((3, 3)),
            "bfactors": torch.zeros(3),
            "atomic_numbers": torch.ones(3, dtype=torch.int64),
        },
        {
            "coords": torch.zeros((3, 3)),
            "bfactors": torch.zeros(3),
            "atomic_numbers": torch.ones(3, dtype=torch.int64),
        },
    ]

    def fake_align(self, x_0_hat, reference):
        return x_0_hat

    def fake_render(self, coords, atomic_numbers, bfactors, dataset):
        return dataset["rendered"]

    monkeypatch.setattr(
        em_module.CryoEM_Images_GuidanceLossFunction, "_align_to_reference", fake_align
    )
    monkeypatch.setattr(
        em_module.CryoEM_Images_GuidanceLossFunction,
        "_render_all_rotations",
        fake_render,
    )

    loss = instance(torch.zeros((1, 3, 3)), time=None)

    assert torch.isclose(loss, torch.tensor(0.5))
    assert instance.last_loss_value == pytest.approx(0.5)


def test_render_all_rotations_expands_bfactors(monkeypatch):
    em_module = _import_em_module(monkeypatch)

    class CaptureAtomStack:
        last_instance = None

        def __init__(self, atom_coordinates, atomic_numbers, device=None):
            self.atom_coordinates = atom_coordinates
            self.atomic_numbers = atomic_numbers
            self.device = device
            self.bfactors = None
            CaptureAtomStack.last_instance = self

        @classmethod
        def from_coords_and_atomic_numbers(
            cls, atom_coordinates, atomic_numbers, device=None
        ):
            return cls(atom_coordinates, atomic_numbers, device=device)

    monkeypatch.setattr(em_module, "AtomStack", CaptureAtomStack)

    def fake_render_single(self, atom_stack, lattice, rotation_matrix, projection_axis, collapse_projection_axis):
        return torch.zeros((2, 2))

    monkeypatch.setattr(
        em_module.CryoEM_Images_GuidanceLossFunction,
        "_render_single_projection",
        fake_render_single,
    )

    instance = em_module.CryoEM_Images_GuidanceLossFunction.__new__(
        em_module.CryoEM_Images_GuidanceLossFunction
    )
    instance.device = torch.device("cpu")

    coords = torch.zeros((2, 3, 3))
    atomic_numbers = torch.ones(3, dtype=torch.int64)
    bfactors = torch.zeros((1, 3))
    dataset = {
        "rotations": torch.stack([torch.eye(3), torch.eye(3)], dim=0),
        "lattice": object(),
        "projection_axis": 0,
        "collapse_projection_axis": True,
    }

    instance._render_all_rotations(coords, atomic_numbers, bfactors, dataset)

    assert CaptureAtomStack.last_instance is not None
    assert CaptureAtomStack.last_instance.bfactors.shape == (2, 3, 1)
