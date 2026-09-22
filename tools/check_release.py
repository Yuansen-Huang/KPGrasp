"""Check the distributable source without CUDA, checkpoints or datasets."""
from pathlib import Path
import ast
import json
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/dexgrasp"))
import _bootstrap
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

EXCLUDED_ROOTS = {
    ".git", "work", "output", "provenance", "release", "checkpoints",
    ".venv", "venv", "build", "dist", "__pycache__",
}
PRIVATE_DOCS = {"CLEANUP_DIFF.md", "CODE_DIFF.patch", "IK_FROM_DEXGRASPBENCH.patch"}
DATA_SUFFIXES = {".pth", ".pt", ".ckpt", ".h5", ".hdf5", ".npy", ".npz"}

def excluded(path):
    return (
        bool(set(path.parts) & EXCLUDED_ROOTS)
        or path.parts[:2] == ("assets", "dexonomy")
        or path.name in PRIVATE_DOCS
        or path.suffix in DATA_SUFFIXES
        or path.name == ".env"
        or path.name.startswith(".env.")
        or path.suffix in {".log", ".pyc"}
    )

def source_files():
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "-z"], cwd=ROOT,
            check=True, capture_output=True, text=True,
        )
        paths = [Path(p) for p in result.stdout.split("\0") if p]
        if paths:
            return paths
    return [
        p.relative_to(ROOT) for p in ROOT.rglob("*")
        if p.is_file() and not excluded(p.relative_to(ROOT))
    ]

def check():
    paths = sorted(source_files())
    assert paths, "No source files found."
    distributed_paths = set(paths)
    machine_path = re.compile(r"/(?:mnt|home|Users)/[A-Za-z0-9]")
    credentials = re.compile(
        r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}"
        r"|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)"
    )
    py_count = 0
    for rel in paths:
        p = ROOT / rel
        assert not excluded(rel), f"Local-only file tracked: {rel}"
        assert not p.is_symlink(), f"Symlink in source distribution: {rel}"
        assert p.resolve().is_relative_to(ROOT), f"Path escapes checkout: {rel}"
        assert p.stat().st_size < 5 * 1024**2, f"Unexpected large source file: {rel}"
        content = p.read_bytes()
        if rel.suffix == ".py":
            ast.parse(content, filename=str(rel))
            py_count += 1
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError:
            continue
        assert not machine_path.search(text), f"Machine-specific path in {rel}"
        assert not credentials.search(text), f"Credential-like content in {rel}"
        if rel.suffix == ".md":
            for link in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if "://" in link or link.startswith(("#", "mailto:")):
                    continue
                target = link.split("#", 1)[0]
                resolved = (p.parent / target).resolve()
                assert resolved.exists(), f"Broken link in {rel}: {target}"
                assert resolved.is_relative_to(ROOT), f"External file link in {rel}: {target}"
                if resolved.is_file():
                    assert resolved.relative_to(ROOT) in distributed_paths, (
                        f"Linked file excluded from distribution: {rel} -> {target}"
                    )

    os.environ["KPGRASP_ROOT"] = str(ROOT)
    previous_data = os.environ.pop("DEXONOMY_DATA", None)
    try:
        sample_paths = sorted((ROOT / "configs/experiments").glob("*.yaml"))
        assert len(sample_paths) == 6
        for path in sample_paths:
            cfg = OmegaConf.load(path)
            OmegaConf.resolve(cfg)
            assert cfg.data_folder == str(ROOT / "assets/dexonomy")
            assert cfg.hand.n_kp == 28 and cfg.data.n_frame == 3
            assert cfg.data_name == "kdex"
            assert Path(cfg.ckpt).is_relative_to(ROOT / "checkpoints")
            assert 0 < cfg.algo.test_topk <= cfg.algo.test_grasp_num
            if path.stem == "Final_base":
                assert cfg.algo.model.flow.sample_step == 5
                assert cfg.algo.test_grasp_num == 100 and cfg.algo.test_topk == 10

        # Resolve every training/scaling configuration, with no data on disk.
        training = sorted((ROOT / "src/dexgrasp/config/experiment").glob("*.yaml"))
        with initialize_config_dir(
            config_dir=str(ROOT / "src/dexgrasp/config"), version_base="1.2"
        ):
            for path in training:
                cfg = compose(config_name="base", overrides=[f"+experiment={path.stem}"])
                OmegaConf.resolve(cfg)
                assert cfg.hand.n_kp == 28 and cfg.data.n_frame == 3
                assert cfg.data_folder == str(ROOT / "assets/dexonomy")
        # An explicit data root must override the portable default.
        alternate = str(ROOT / "work/check-data")
        os.environ["DEXONOMY_DATA"] = alternate
        cfg = OmegaConf.load(ROOT / "configs/experiments/Final_base.yaml")
        OmegaConf.resolve(cfg)
        assert cfg.data_folder == alternate
    finally:
        if previous_data is None:
            os.environ.pop("DEXONOMY_DATA", None)
        else:
            os.environ["DEXONOMY_DATA"] = previous_data

    xml_path = ROOT / "assets/hand/shadow/right_hand_w_kp.xml"
    xml = ET.parse(xml_path).getroot()
    mesh_dir = xml_path.parent / xml.find("compiler").attrib.get("meshdir", "")
    meshes = xml.findall("./asset/mesh")
    for mesh in meshes:
        assert (mesh_dir / mesh.attrib["file"]).is_file(), mesh.attrib["file"]
    manifest = json.loads((ROOT / "configs/checkpoints.json").read_text())
    assert len(manifest) == 6
    for record in manifest:
        assert len(record["sha256"]) == 64 and record["step"] == 200000
        assert (ROOT / "configs/experiments" / (record["experiment"] + ".yaml")).exists()
    assert "Attribution-NonCommercial 4.0" in (ROOT / "LICENSE").read_text()
    print(json.dumps({
        "status": "passed", "source_files": len(paths), "python_files": py_count,
        "sampling_configs": len(sample_paths), "training_configs": len(training),
        "local_meshes": len(meshes), "data_or_weights_required": False,
    }, indent=2))

if __name__ == "__main__":
    check()
