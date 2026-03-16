import pytest
import torch

from src.losses.assignment_strategies import (
    AssignmentStrategy,
    HardAssignment,
    SoftmaxAssignment,
    SinkhornAssignment,
    StickyAssignment,
    build_assignment_strategy,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_build_known_strategies():
    assert isinstance(build_assignment_strategy("hard"), HardAssignment)
    assert isinstance(build_assignment_strategy("softmax"), SoftmaxAssignment)
    assert isinstance(build_assignment_strategy("sinkhorn"), SinkhornAssignment)
    assert isinstance(build_assignment_strategy("sticky"), StickyAssignment)


def test_build_unknown_strategy_raises():
    with pytest.raises(ValueError, match="Unknown assignment strategy"):
        build_assignment_strategy("nonexistent")


def test_build_passes_kwargs():
    s = build_assignment_strategy("softmax", temperature_init=2.0, temperature_min=0.5)
    assert s.temperature_init == 2.0
    assert s.temperature_min == 0.5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_clear_loss_matrix():
    """Loss matrix where structure 0 is clearly best for samples 0-1,
    and structure 1 is clearly best for samples 2-3."""
    return torch.tensor([
        [0.1, 0.9],
        [0.2, 0.8],
        [0.8, 0.1],
        [0.9, 0.2],
    ])


def _make_ambiguous_loss_matrix():
    """Loss matrix where all losses are nearly identical."""
    return torch.tensor([
        [0.50, 0.51],
        [0.51, 0.50],
        [0.50, 0.51],
        [0.51, 0.50],
    ])


# ---------------------------------------------------------------------------
# HardAssignment
# ---------------------------------------------------------------------------


class TestHardAssignment:
    def test_assigns_to_minimum(self):
        loss_matrix = _make_clear_loss_matrix()
        strategy = HardAssignment()
        selected_losses, assignments = strategy.assign(loss_matrix)

        assert assignments.tolist() == [0, 0, 1, 1]
        assert torch.allclose(selected_losses, torch.tensor([0.1, 0.2, 0.1, 0.2]))

    def test_gradient_flows(self):
        loss_matrix = _make_clear_loss_matrix().requires_grad_(True)
        strategy = HardAssignment()
        selected_losses, _ = strategy.assign(loss_matrix)
        selected_losses.sum().backward()
        assert loss_matrix.grad is not None

    def test_extra_wandb_log_empty(self):
        assert HardAssignment().extra_wandb_log() == {}


# ---------------------------------------------------------------------------
# SoftmaxAssignment
# ---------------------------------------------------------------------------


class TestSoftmaxAssignment:
    def test_high_temperature_uniform_weights(self):
        loss_matrix = _make_clear_loss_matrix()
        strategy = SoftmaxAssignment(temperature_init=100.0)
        selected_losses, assignments = strategy.assign(loss_matrix)

        # With very high temperature, selected loss ≈ row mean
        row_means = loss_matrix.mean(dim=1)
        assert torch.allclose(selected_losses, row_means, atol=0.01)
        # Hard assignments should still pick the minimum
        assert assignments.tolist() == [0, 0, 1, 1]

    def test_low_temperature_approaches_hard(self):
        loss_matrix = _make_clear_loss_matrix()
        strategy = SoftmaxAssignment(temperature_init=0.001)
        selected_losses, assignments = strategy.assign(loss_matrix)

        # With very low temperature, should approach hard assignment
        hard = HardAssignment()
        hard_losses, _ = hard.assign(loss_matrix)
        assert torch.allclose(selected_losses, hard_losses, atol=0.01)

    def test_temperature_annealing(self):
        strategy = SoftmaxAssignment(
            temperature_init=1.0, temperature_decay=0.9, temperature_min=0.1,
        )
        loss_matrix = _make_clear_loss_matrix()

        strategy.assign(loss_matrix, step=0)
        assert strategy._last_temperature == pytest.approx(1.0)

        strategy.assign(loss_matrix, step=10)
        assert strategy._last_temperature == pytest.approx(1.0 * 0.9**10)

        # Should not go below min
        strategy.assign(loss_matrix, step=10000)
        assert strategy._last_temperature == pytest.approx(0.1)

    def test_gradient_flows_through_soft_weights(self):
        loss_matrix = _make_clear_loss_matrix().requires_grad_(True)
        strategy = SoftmaxAssignment(temperature_init=1.0)
        selected_losses, _ = strategy.assign(loss_matrix)
        selected_losses.sum().backward()
        # All entries should have non-zero gradient (soft weighting)
        assert (loss_matrix.grad.abs() > 0).all()

    def test_extra_wandb_log_reports_temperature(self):
        strategy = SoftmaxAssignment(temperature_init=0.5)
        loss_matrix = _make_clear_loss_matrix()
        strategy.assign(loss_matrix, step=0)
        log = strategy.extra_wandb_log()
        assert log["cryoimage/assignment_temperature"] == pytest.approx(0.5)

    def test_validation_rejects_bad_params(self):
        with pytest.raises(ValueError):
            SoftmaxAssignment(temperature_init=-1)
        with pytest.raises(ValueError):
            SoftmaxAssignment(temperature_min=0)
        with pytest.raises(ValueError):
            SoftmaxAssignment(temperature_decay=1.5)


# ---------------------------------------------------------------------------
# SinkhornAssignment
# ---------------------------------------------------------------------------


class TestSinkhornAssignment:
    def test_balanced_assignment_with_clear_separation(self):
        loss_matrix = _make_clear_loss_matrix()
        strategy = SinkhornAssignment(temperature_init=0.1, num_iters=20)
        selected_losses, assignments = strategy.assign(loss_matrix)

        # With clear separation and balanced marginals, assignments should be 0,0,1,1
        assert assignments.tolist() == [0, 0, 1, 1]

    def test_balancing_effect(self):
        """When one structure is slightly better for all samples, Sinkhorn
        should still split them roughly evenly (unlike hard which would assign all to one)."""
        # Structure 0 is slightly better for everyone
        loss_matrix = torch.tensor([
            [0.40, 0.41],
            [0.40, 0.41],
            [0.40, 0.41],
            [0.40, 0.41],
        ])
        strategy = SinkhornAssignment(temperature_init=1.0, num_iters=50)
        selected_losses, _ = strategy.assign(loss_matrix)

        # The OT weights should distribute mass roughly equally
        # so selected_loss per sample should be between 0.40 and 0.41 (mix of both)
        assert (selected_losses >= 0.39).all()
        assert (selected_losses <= 0.42).all()

    def test_gradient_flows(self):
        loss_matrix = _make_clear_loss_matrix().requires_grad_(True)
        strategy = SinkhornAssignment(temperature_init=1.0, num_iters=5)
        selected_losses, _ = strategy.assign(loss_matrix)
        selected_losses.sum().backward()
        assert loss_matrix.grad is not None

    def test_temperature_annealing(self):
        strategy = SinkhornAssignment(
            temperature_init=2.0, temperature_decay=0.5, temperature_min=0.1,
        )
        loss_matrix = _make_clear_loss_matrix()

        strategy.assign(loss_matrix, step=1)
        assert strategy._last_temperature == pytest.approx(1.0)

        strategy.assign(loss_matrix, step=100)
        assert strategy._last_temperature == pytest.approx(0.1)  # clamped at min

    def test_extra_wandb_log_reports_temperature(self):
        strategy = SinkhornAssignment(temperature_init=0.5)
        strategy.assign(_make_clear_loss_matrix(), step=0)
        log = strategy.extra_wandb_log()
        assert "cryoimage/assignment_temperature" in log

    def test_validation_rejects_bad_params(self):
        with pytest.raises(ValueError):
            SinkhornAssignment(num_iters=0)


# ---------------------------------------------------------------------------
# StickyAssignment
# ---------------------------------------------------------------------------


class TestStickyAssignment:
    def test_first_call_behaves_like_hard(self):
        loss_matrix = _make_clear_loss_matrix()
        strategy = StickyAssignment(stickiness=0.5)
        selected_losses, assignments = strategy.assign(loss_matrix)
        assert assignments.tolist() == [0, 0, 1, 1]

    def test_stickiness_prevents_switching(self):
        strategy = StickyAssignment(stickiness=0.5)

        # First call: clear assignment
        loss_matrix_1 = _make_clear_loss_matrix()
        _, assignments_1 = strategy.assign(loss_matrix_1)
        assert assignments_1.tolist() == [0, 0, 1, 1]

        # Second call: slightly favour the OTHER structure, but within stickiness margin
        loss_matrix_2 = torch.tensor([
            [0.5, 0.4],   # structure 1 is better by 0.1, but stickiness=0.5 keeps it at 0
            [0.5, 0.4],
            [0.4, 0.5],   # structure 0 is better by 0.1, but stickiness=0.5 keeps it at 1
            [0.4, 0.5],
        ])
        _, assignments_2 = strategy.assign(loss_matrix_2)
        # Should stick to previous assignments because margin < stickiness
        assert assignments_2.tolist() == [0, 0, 1, 1]

    def test_large_improvement_overrides_stickiness(self):
        strategy = StickyAssignment(stickiness=0.1)

        loss_matrix_1 = _make_clear_loss_matrix()
        _, assignments_1 = strategy.assign(loss_matrix_1)
        assert assignments_1.tolist() == [0, 0, 1, 1]

        # Now flip so that the other structure is dramatically better
        loss_matrix_2 = torch.tensor([
            [0.9, 0.1],   # structure 1 now much better
            [0.9, 0.1],
            [0.1, 0.9],   # structure 0 now much better
            [0.1, 0.9],
        ])
        _, assignments_2 = strategy.assign(loss_matrix_2)
        assert assignments_2.tolist() == [1, 1, 0, 0]

    def test_stickiness_decay(self):
        strategy = StickyAssignment(stickiness=1.0, stickiness_decay=0.5, stickiness_min=0.01)
        loss_matrix = _make_clear_loss_matrix()

        strategy.assign(loss_matrix, step=0)
        assert strategy._last_stickiness == pytest.approx(1.0)

        strategy.assign(loss_matrix, step=2)
        assert strategy._last_stickiness == pytest.approx(0.25)

        strategy.assign(loss_matrix, step=100)
        assert strategy._last_stickiness == pytest.approx(0.01)  # clamped

    def test_gradient_flows(self):
        loss_matrix = _make_clear_loss_matrix().requires_grad_(True)
        strategy = StickyAssignment(stickiness=0.1)
        selected_losses, _ = strategy.assign(loss_matrix)
        selected_losses.sum().backward()
        assert loss_matrix.grad is not None

    def test_extra_wandb_log_reports_stickiness(self):
        strategy = StickyAssignment(stickiness=0.3)
        strategy.assign(_make_clear_loss_matrix(), step=0)
        log = strategy.extra_wandb_log()
        assert log["cryoimage/assignment_stickiness"] == pytest.approx(0.3)

    def test_validation_rejects_bad_params(self):
        with pytest.raises(ValueError):
            StickyAssignment(stickiness=-1)
        with pytest.raises(ValueError):
            StickyAssignment(stickiness_decay=0)


# ---------------------------------------------------------------------------
# Integration: strategies work inside the loss function
# ---------------------------------------------------------------------------


class TestIntegrationWithLossFunction:
    """Test that strategies plug into _select_assignments correctly."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch

    def _make_loss_fn(self, strategy):
        """Build a loss function with dependency stubs and a given assignment strategy."""
        import importlib
        import json
        import sys
        import types
        from pathlib import Path

        # -- inline dependency stubs (same as test_em_images_loss_function) --
        repo_root = Path(__file__).resolve().parents[1]
        self.monkeypatch.syspath_prepend(str(repo_root))

        # Stub external deps
        gemmi_mod = types.ModuleType("gemmi")
        gemmi_mod.read_ccp4_map = lambda path: None
        self.monkeypatch.setitem(sys.modules, "gemmi", gemmi_mod)

        cryo_pkg = types.ModuleType("cryoforward")
        cryo_pkg.__path__ = []
        self.monkeypatch.setitem(sys.modules, "cryoforward", cryo_pkg)

        class DummyLattice:
            def __init__(self, grid_dimensions, voxel_sizes_in_A, left_bottom_point_in_A,
                         right_upper_point_in_A, sublattice_radius_in_A, dtype, device):
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
        self.monkeypatch.setitem(sys.modules, "cryoforward.lattice", lattice_mod)

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

        atom_stack_mod = types.ModuleType("cryoforward.atom_stack")
        atom_stack_mod.AtomStack = DummyAtomStack
        self.monkeypatch.setitem(sys.modules, "cryoforward.atom_stack", atom_stack_mod)

        cryoesp_mod = types.ModuleType("cryoforward.cryoesp_calculator")
        def setup_fast_esp_solver(atom_stack, lattice, per_voxel_averaging=True, use_checkpointing=False, use_autocast=False):
            def compute_batch_from_coords(coords_batch, bfactors, atomic_numbers, occupancies):
                shape = coords_batch.shape[:2] + tuple(int(x) for x in lattice.grid_dimensions.tolist())
                return torch.zeros(shape, dtype=coords_batch.dtype, device=coords_batch.device)
            return None, compute_batch_from_coords
        cryoesp_mod.setup_fast_esp_solver = setup_fast_esp_solver
        self.monkeypatch.setitem(sys.modules, "cryoforward.cryoesp_calculator", cryoesp_mod)

        rmsd_mod = types.ModuleType("src.protenix.metrics.rmsd")
        def fake_self_aligned_rmsd(pred, true, mask):
            return torch.zeros(pred.shape[0], device=pred.device), pred, None, None
        rmsd_mod.self_aligned_rmsd = fake_self_aligned_rmsd
        self.monkeypatch.setitem(sys.modules, "src.protenix.metrics.rmsd", rmsd_mod)

        utils_io_mod = types.ModuleType("src.utils.io")
        def alignment_mask_by_chain(full_sequences, chains_to_align, sequence_types):
            return torch.ones(3, dtype=torch.bool)
        def load_pdb_atom_locations_full(pdb_file, full_sequences_dict, chains_to_read=None,
                                          return_elements=True, return_bfacs=True,
                                          return_mask=True, return_starting_indices=True):
            coords = torch.zeros((3, 3), dtype=torch.float32)
            mask_t = torch.ones(3, dtype=torch.bool)
            bfactors = torch.zeros(3, dtype=torch.float32)
            elements = torch.ones(3, dtype=torch.int64)
            return coords, mask_t, bfactors, elements, [0]
        utils_io_mod.alignment_mask_by_chain = alignment_mask_by_chain
        utils_io_mod.load_pdb_atom_locations_full = load_pdb_atom_locations_full
        self.monkeypatch.setitem(sys.modules, "src.utils.io", utils_io_mod)

        losses_pkg = types.ModuleType("src.losses")
        losses_pkg.__path__ = [str(repo_root / "src" / "losses")]
        self.monkeypatch.setitem(sys.modules, "src.losses", losses_pkg)

        for module_name in (
            "src.utils.cryoimage_renderer",
            "src.losses.assignment_strategies",
            "src.losses.em_images_loss_function",
        ):
            sys.modules.pop(module_name, None)

        loss_mod = importlib.import_module("src.losses.em_images_loss_function")

        def fake_setup_fast_solver(self_inner):
            self_inner._fast_renderers = [
                [object() for _ in self_inner.reference_structures]
                for _ in range(self_inner.dataset.num_conformations)
            ]

        def fake_align(self_inner, x_0_hat, reference):
            return x_0_hat

        def fake_render(self_inner, *, conformation_index, structure_index, aligned_structure, rotations):
            return torch.full((rotations.shape[0], 2, 2), float(structure_index), device=rotations.device)

        self.monkeypatch.setattr(loss_mod.CryoEM_Images_GuidanceLossFunction, "_setup_fast_solver", fake_setup_fast_solver)
        self.monkeypatch.setattr(loss_mod.CryoEM_Images_GuidanceLossFunction, "_align_to_reference", fake_align)
        self.monkeypatch.setattr(loss_mod.CryoEM_Images_GuidanceLossFunction, "_render_projection_batch", fake_render)

        # Write projection data
        def _write_source(name, projections):
            pt_path = self.tmp_path / f"{name}.pt"
            json_path = self.tmp_path / f"{name}.json"
            rotations = torch.stack([torch.eye(3)] * int(projections.shape[0]), dim=0)
            torch.save({"projections": projections, "rotations": rotations}, pt_path)
            meta = {
                "projection_axis": "z",
                "lattice": {
                    "grid_dimensions": [2, 2, 1], "voxel_sizes": [1.0, 1.0, 1.0],
                    "left_bottom": [0.0, 0.0, 0.0], "right_upper": [2.0, 2.0, 1.0],
                    "sublattice_radius": 8.0, "projection_axis": 2,
                    "collapse_projection_axis": True,
                },
            }
            json_path.write_text(json.dumps(meta))
            return str(pt_path), str(json_path)

        pt_paths, json_paths = [], []
        for idx, projections in enumerate([torch.zeros((2, 2, 2)), torch.ones((2, 2, 2))]):
            pt_path, json_path = _write_source(f"conf_{idx}", projections)
            pt_paths.append(pt_path)
            json_paths.append(json_path)

        return loss_mod.CryoEM_Images_GuidanceLossFunction(
            image_pt_files=pt_paths,
            image_json_files=json_paths,
            reference_pdbs=["ref_a.pdb", "ref_b.pdb"],
            mask=torch.ones(3, dtype=torch.bool),
            sequences_dictionary=[{"sequence": "AAA", "count": 1}],
            device="cpu",
            log_projection_every=0,
            log_projection_pairs=0,
            supervised_assignment_by_index=False,
            projection_batch_size=None,
            shuffle_projection_samples=False,
            assignment_strategy=strategy,
        )

    def test_hard_strategy_integration(self):
        loss_fn = self._make_loss_fn(HardAssignment())
        x_0_hat = torch.zeros((2, 3, 3), dtype=torch.float32)
        loss, _, _ = loss_fn(x_0_hat, time=None)
        assert loss_fn.last_assignment_counts == {0: 2, 1: 2}

    def test_softmax_strategy_integration(self):
        loss_fn = self._make_loss_fn(SoftmaxAssignment(temperature_init=0.01))
        x_0_hat = torch.zeros((2, 3, 3), dtype=torch.float32)
        loss, _, _ = loss_fn(x_0_hat, time=None)
        assert loss_fn.last_assignment_mode == "SoftmaxAssignment"
        wandb_log = loss_fn.wandb_log(x_0_hat)
        assert "cryoimage/assignment_temperature" in wandb_log

    def test_sinkhorn_strategy_integration(self):
        loss_fn = self._make_loss_fn(SinkhornAssignment(temperature_init=0.1, num_iters=10))
        x_0_hat = torch.zeros((2, 3, 3), dtype=torch.float32)
        loss, _, _ = loss_fn(x_0_hat, time=None)
        assert loss_fn.last_assignment_counts == {0: 2, 1: 2}

    def test_sticky_strategy_integration(self):
        loss_fn = self._make_loss_fn(StickyAssignment(stickiness=0.1))
        x_0_hat = torch.zeros((2, 3, 3), dtype=torch.float32)
        loss, _, _ = loss_fn(x_0_hat, time=None)
        assert loss_fn.last_assignment_counts == {0: 2, 1: 2}
