from __future__ import annotations

import argparse
import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
pytest.importorskip("cryoforward", reason="cryoforward not installed")
gemmi = pytest.importorskip("gemmi", reason="gemmi not installed")

MODULE_PATH = Path(__file__).resolve().parents[1] / "dataset-gen.py"
spec = importlib.util.spec_from_file_location("dataset_gen", MODULE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Failed to load module from {MODULE_PATH}")

dataset_gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_gen)


def _make_ccp4_map(path: Path, D: int = 20, pixel_size: float = 3.0) -> Path:
    """Write a minimal synthetic CCP4 map for testing."""
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = gemmi.FloatGrid(D, D, D)
    maxsize = D * pixel_size
    ccp4.grid.unit_cell.set(maxsize, maxsize, maxsize, 90, 90, 90)
    ccp4.grid.spacegroup = gemmi.SpaceGroup("P1")
    map_path = path / "test.ccp4"
    ccp4.write_ccp4_map(str(map_path))
    return map_path


def test_parse_args_requires_input(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["dataset-gen.py"])
    with pytest.raises(SystemExit):
        dataset_gen._parse_args()


def test_parse_args_accepts_files_and_ids(monkeypatch, tmp_path):
    pdb_path = tmp_path / "example.pdb"
    pdb_path.write_text("HEADER\n")
    map_path = _make_ccp4_map(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        ["dataset-gen.py", "--pdb-file", str(pdb_path), "--pdb-id", "1abc", "--density-map", str(map_path)],
    )
    args = dataset_gen._parse_args()
    assert args.pdb_files == [str(pdb_path)]
    assert args.pdb_ids == ["1abc"]


def test_download_pdb_uses_cache(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / "1ABC.pdb"
    cached_path.write_text("CACHED\n")

    called = {"value": False}

    def fake_urlopen(_url):
        called["value"] = True
        raise AssertionError("urlopen should not be called for cached files.")

    monkeypatch.setattr(dataset_gen, "urlopen", fake_urlopen)
    result = dataset_gen._download_pdb_if_not_cached("1abc", cache_dir)
    assert result == cached_path
    assert cached_path.read_text() == "CACHED\n"
    assert called["value"] is False


def test_download_pdb_writes_file(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_urlopen(_url):
        return FakeResponse(b"HEADER\n")

    monkeypatch.setattr(dataset_gen, "urlopen", fake_urlopen)
    result = dataset_gen._download_pdb_if_not_cached("2xyz", cache_dir)
    assert result.exists()
    assert result.read_bytes() == b"HEADER\n"


def test_main_dispatches_to_generate(monkeypatch, tmp_path):
    calls = []

    def fake_generate(pdb_path, _args, _device, **_kwargs):
        calls.append(pdb_path)

    def fake_download(pdb_id, cache_dir):
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{pdb_id.upper()}.pdb"
        path.write_text("HEADER\n")
        return path

    pdb_file = tmp_path / "local.pdb"
    pdb_file.write_text("HEADER\n")

    args = argparse.Namespace(
        pdb_files=[str(pdb_file)],
        pdb_ids=["1abc", "2def"],
        pdb_cache_dir=None,
        out_dir=str(tmp_path / "out"),
        device="cpu",
        seed=0,
        config=None,
    )

    monkeypatch.setattr(dataset_gen, "_parse_args", lambda: args)
    monkeypatch.setattr(dataset_gen, "_generate_dataset_for_pdb", fake_generate)
    monkeypatch.setattr(dataset_gen, "_download_pdb_if_not_cached", fake_download)

    dataset_gen.main()

    expected = [
        pdb_file,
        Path(args.out_dir) / "pdbs" / "1ABC.pdb",
        Path(args.out_dir) / "pdbs" / "2DEF.pdb",
    ]
    assert calls == expected


def test_loads_test_pdb_file():
    pdb_path = Path(__file__).parent / "pdbs" / "7t54.pdb"
    atom_stack = dataset_gen.AtomStack.from_pdb_file(str(pdb_path), device="cpu")
    assert atom_stack.atom_coordinates.shape[1] > 0
    assert len(atom_stack.atom_names) == atom_stack.atom_coordinates.shape[1]
    assert atom_stack.bfactors is not None
    assert atom_stack.bfactors.shape[1] == atom_stack.atom_coordinates.shape[1]


def test_generate_for_pdb_renders_and_saves(monkeypatch, tmp_path):
    pdb_path = Path(__file__).parent / "pdbs" / "7t54.pdb"
    original_from_pdb = dataset_gen.AtomStack.from_pdb_file

    def limited_from_pdb_file(
        cls, file_path, model_index=0, chains_to_include=None, device="cpu"
    ):
        stack = original_from_pdb(
            file_path,
            model_index=model_index,
            chains_to_include=chains_to_include,
            device=device,
        )
        n_atoms = min(40, stack.atom_coordinates.shape[1])
        return dataset_gen.AtomStack(
            atom_coordinates=stack.atom_coordinates[:, :n_atoms, :],
            atom_names=stack.atom_names[:n_atoms],
            bfactors=stack.bfactors[:, :n_atoms, :] if stack.bfactors is not None else None,
            device=device,
            occupancies=stack.occupancies,
        )

    monkeypatch.setattr(
        dataset_gen.AtomStack,
        "from_pdb_file",
        classmethod(limited_from_pdb_file),
    )

    map_path = _make_ccp4_map(tmp_path, D=10, pixel_size=10.0)
    args = argparse.Namespace(
        density_map=str(map_path),
        num_rotations=1,
        sublattice_radius=2.0,
        projection_axis="z",
        collapse_projection_axis=True,
        bfactor=None,
        atomic_radius=0.5,
        output_format="pt",
        out_dir=str(tmp_path),
        clear_cache_every=0,
        progress=False,
    )

    dataset_gen._generate_dataset_for_pdb(pdb_path, args, torch.device("cpu"))

    output_path = Path(args.out_dir) / "7T54_esp_projections_1.pt"
    assert output_path.exists()

    data = torch.load(output_path)
    assert data["projections"].shape[0] == args.num_rotations
    assert data["rotations"].shape == (args.num_rotations, 3, 3)
    assert torch.isfinite(data["projections"]).all()
    assert data["projections"].abs().sum() > 0

    meta_path = output_path.with_suffix(".json")
    assert meta_path.exists()


def test_canonical_output_pdb_id_is_uppercase():
    assert dataset_gen._canonical_output_pdb_id(Path("7t54.pdb")) == "7T54"
    assert dataset_gen._canonical_output_pdb_id(Path("7T55.pdb")) == "7T55"


def test_prepare_lattice_collapses_axis(tmp_path):
    map_path = _make_ccp4_map(tmp_path, D=20, pixel_size=3.0)
    lattice, meta = dataset_gen.prepare_lattice_from_density_map(
        density_map_path=map_path,
        sublattice_radius=2.0,
        projection_axis=2,
        collapse_projection_axis=True,
        device=torch.device("cpu"),
    )

    assert lattice.grid_dimensions.tolist()[2] == 1
    assert meta["grid_dimensions"][2] == 1
    assert meta["voxel_sizes"][2] == pytest.approx(meta["projection_depth"])


def test_collapsed_projection_matches_full_sum(tmp_path):
    pdb_path = Path(__file__).parent / "pdbs" /  "7t54.pdb"
    atom_stack = dataset_gen.AtomStack.from_pdb_file(str(pdb_path), device="cpu")
    n_atoms = min(50, atom_stack.atom_coordinates.shape[1])
    atom_stack = dataset_gen.AtomStack(
        atom_coordinates=atom_stack.atom_coordinates[:, :n_atoms, :],
        atom_names=atom_stack.atom_names[:n_atoms],
        bfactors=atom_stack.bfactors[:, :n_atoms, :] if atom_stack.bfactors is not None else None,
        device="cpu",
        occupancies=atom_stack.occupancies,
    )

    sublattice_radius = 10.0
    axis = 2

    map_path = _make_ccp4_map(tmp_path, D=50, pixel_size=3.0)
    lattice_full, _ = dataset_gen.prepare_lattice_from_density_map(
        density_map_path=map_path,
        sublattice_radius=sublattice_radius,
        projection_axis=axis,
        collapse_projection_axis=False,
        device=torch.device("cpu"),
    )
    lattice_collapsed, _ = dataset_gen.prepare_lattice_from_density_map(
        density_map_path=map_path,
        sublattice_radius=sublattice_radius,
        projection_axis=axis,
        collapse_projection_axis=True,
        device=torch.device("cpu"),
    )

    rotation = dataset_gen.Rotation.identity(batch=1, device="cpu")
    projection_full, _ = dataset_gen._render_projection(
        atom_stack=atom_stack,
        lattice=lattice_full,
        projection_axis=axis,
        collapse_projection_axis=False,
        rotation=rotation,
    )
    projection_collapsed, _ = dataset_gen._render_projection(
        atom_stack=atom_stack,
        lattice=lattice_collapsed,
        projection_axis=axis,
        collapse_projection_axis=True,
        rotation=rotation,
    )

    assert projection_full.shape == projection_collapsed.shape
    assert torch.isfinite(projection_full).all()
    assert torch.isfinite(projection_collapsed).all()
    assert torch.allclose(
        projection_full,
        projection_collapsed,
        rtol=1e-2,
        atol=1e-2,
    )


def test_save_outputs_npz_roundtrip(tmp_path):
    projections = torch.ones((2, 3, 4), dtype=torch.float32)
    rotations = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(2, 1, 1)
    meta = {"pdb_id": "test"}

    output_path = tmp_path / "sample.npz"
    dataset_gen._save_outputs(projections, rotations, meta, output_path, "npz")

    assert output_path.exists()
    data = np.load(output_path)
    assert data["projections"].shape == (2, 3, 4)
    assert data["rotations"].shape == (2, 3, 3)
    assert (data["projections"] > 0).all()

    meta_path = output_path.with_suffix(".json")
    assert meta_path.exists()
