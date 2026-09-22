"""Validate the installed runtime without installing or downloading anything."""
from pathlib import Path
import argparse
import importlib
from importlib import metadata
import json
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src/dexgrasp"))
import _bootstrap

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-cpu", action="store_true", help="Allow IK-only/CPU inspection.")
    args = parser.parse_args()
    required = {}
    for path in (ROOT / "requirements.txt", ROOT / "requirements/constraints.txt"):
        for line in path.read_text().splitlines():
            if "==" in line and not line.startswith("#"):
                name, expected = line.strip().split("==")
                required[name] = expected
    installed, errors = {}, []
    for name, expected in sorted(required.items()):
        try:
            actual = metadata.version(name)
            installed[name] = actual
            if actual.split("+")[0] != expected:
                errors.append(f"{name}: expected {expected}, found {actual}")
        except metadata.PackageNotFoundError:
            errors.append(f"{name}: not installed")
    if errors:
        print(json.dumps({"status": "failed", "errors": errors}, indent=2))
        return 1
    for name in ("torch", "torchvision", "MinkowskiEngine", "pytorch3d.transforms",
                 "mujoco", "mink", "h5py", "quadprog"):
        try:
            importlib.import_module(name)
        except Exception as error:
            errors.append(f"{name}: {type(error).__name__}: {error}")
    import torch
    cuda = torch.cuda.is_available()
    if not cuda and not args.allow_cpu:
        errors.append("CUDA is unavailable; model training/sampling needs an NVIDIA GPU.")
    print(json.dumps({
        "status": "failed" if errors else "passed",
        "python": sys.version.split()[0], "cuda_available": cuda,
        "torch_cuda": torch.version.cuda, "versions": installed, "errors": errors,
    }, indent=2))
    return 1 if errors else 0

if __name__ == "__main__":
    raise SystemExit(main())
