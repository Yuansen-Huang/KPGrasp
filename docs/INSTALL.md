# Installation

## Validated environment

The implementation was exercised in an existing Linux environment with
Python 3.10, PyTorch 2.1.0, torchvision 0.16.0, CUDA 11.8,
MinkowskiEngine 0.5.4 and PyTorch3D 0.7.8. GPU checks used an RTX 4090.
This records a working environment, not a claim that every command below has
been tested in a fresh environment.

MinkowskiEngine (sparse neural networks) and `mink` (inverse kinematics) are
different dependencies. Both are required for the complete pipeline.

## Installation order

1. Create and activate a Python 3.10 environment.
2. Install a matching PyTorch and torchvision build.
3. Build/install MinkowskiEngine and PyTorch3D against that PyTorch/CUDA pair.
4. Install the remaining Python requirements.

Example environment and PyTorch setup:

```bash
conda create -n kpgrasp python=3.10
conda activate kpgrasp
python -m pip install torch==2.1.0 torchvision==0.16.0 \
  --index-url https://download.pytorch.org/whl/cu118
```

This pairing follows the [official PyTorch version instructions](https://pytorch.org/get-started/previous-versions/).
A source build also needs a compatible C++ compiler, CUDA toolkit including
`nvcc`, and BLAS development headers. The CUDA version used by `nvcc` should
match the PyTorch build.

Use the upstream [MinkowskiEngine installation instructions](https://github.com/NVIDIA/MinkowskiEngine#installation)
to build version **0.5.4**, and the
[PyTorch3D installation instructions](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
for version **0.7.8**. These native packages are deliberately kept out of the
ordinary pip requirements file: they need platform-specific build settings.
Do not assume that a similarly named package or a wheel for another CUDA
version is interchangeable.

Then, from the KPGrasp checkout:

```bash
python -m pip install -r requirements.txt
python -B tools/check_environment.py
```

`requirements/constraints.txt` prevents pip from silently selecting a different
PyTorch/torchvision pair. The checker reports missing or mismatched direct
dependencies and verifies the CUDA extension and IK imports. It does not
download weights or datasets.

## Checks without a GPU

For documentation/configuration work, only the lightweight check dependencies
are needed:

```bash
python -m pip install -r requirements/check.txt
python -B tools/check_release.py
python -B src/dexgrasp/sample.py -e Final_base --dry-run
```

The GitHub Actions workflow runs these checks without weights, datasets or CUDA.

## Runtime files

Run the documented commands from the repository root. Entry points configure
cache and temporary directories under `work/`; training and sampling results
go under `output/`. Pre-existing environment overrides for cache locations
are respected.

`DEXONOMY_DATA` selects the input dataset root. It defaults to
`assets/dexonomy/` inside this checkout. See [DATA.md](DATA.md) for the layout.
Model weights are separate inputs; see [CHECKPOINTS.md](CHECKPOINTS.md).

The SE(3) ablation uses the PyTorch backend of geomstats. The entry points set
`GEOMSTATS_BACKEND=pytorch` when it is not already configured.
