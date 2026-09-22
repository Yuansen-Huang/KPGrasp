"""Keep generated files within this checkout."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
os.environ.setdefault("KPGRASP_ROOT", str(ROOT))
os.environ.setdefault("GEOMSTATS_BACKEND", "pytorch")
for variable, folder in {
    "XDG_CACHE_HOME": "work/cache", "TMPDIR": "work/tmp",
    "MPLCONFIGDIR": "work/mpl", "CUDA_CACHE_PATH": "work/cuda",
    "TORCH_EXTENSIONS_DIR": "work/torch_extensions",
}.items():
    location = ROOT / folder
    location.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(variable, str(location))
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
