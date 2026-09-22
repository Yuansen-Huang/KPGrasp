# KPGrasp

**Scalable Keypoint Flow Matching for Dexterous Grasp Generation**

[Paper](https://arxiv.org/abs/2606.09314) · [Installation](docs/INSTALL.md) ·
[Data](docs/DATA.md) · [Checkpoints](docs/CHECKPOINTS.md) ·
[Experiments](docs/EXPERIMENTS.md)

KPGrasp generates dexterous grasps from object point clouds using a sparse
point-cloud encoder and a Transformer flow model. This release retains the
**28-keypoint** implementation for Shadow Hand, with **three phases**
(pregrasp, grasp, squeeze). The main configuration uses **5 Euler steps**,
generates **100 candidates**, and selects **10** by the retained ranking score.

The main `Final_base` checkpoint is associated with a historical Dexonomy
evaluation of **76.3% grasp success**. This is a recorded benchmark result,
not a new evaluation of this release. See [experiment notes](docs/EXPERIMENTS.md)
for the checkpoint mapping and implementation differences.

## Included

- Keypoint flow-matching model with MinkUNet and DiT.
- Pose/joint, SE(3), single-token, FlowMap, and no-ranking ablations.
- Dexonomy training, sampling, and HDF5 packing utilities.
- Keypoint-to-Shadow-Hand IK with wrist registration and joint optimization.

The source archive includes the hand model, but **does not include dataset files
or pretrained checkpoints**. Weight filenames and checksums are documented;
public weight download URLs have not been assigned yet.

## Quick start

Use Linux, Python 3.10 and an NVIDIA GPU for model training/sampling.
Follow [INSTALL.md](docs/INSTALL.md) to install the matching PyTorch/CUDA,
MinkowskiEngine, PyTorch3D and Python dependencies.

From the repository root, organize the [Dexonomy data](docs/DATA.md) under
`assets/dexonomy/`, or point to an existing data directory:

```bash
export DEXONOMY_DATA=/path/to/dexonomy/assets
```

Place the [main checkpoint](docs/CHECKPOINTS.md) at
`checkpoints/Final_base/step_200000.pth`. Configuration checks need neither
data nor weights:

```bash
python -B tools/check_release.py
python -B src/dexgrasp/sample.py -e Final_base --dry-run
```

Sample and reconstruct joint poses:

```bash
python -B src/dexgrasp/sample.py -e Final_base \
  --output-dir output/samples/main_5step

python -B ik.py \
  --input output/samples/main_5step/step_200000 \
  --output output/ik/main_5step --workers 8
```

For a small pipeline check, append
`--limit 1 algo.batch_size=1 data.num_workers=0 data.mini_test=true`
to the sampling command. This evaluates one batch, not benchmark success.

IK uses the bundled 28-point hand model. Output `qpos` contains translation,
a **wxyz** quaternion, and 22 joint angles. Keypoints and scene metadata are
preserved. `ik_stats.json` reports fitting errors; IK fitting success is not
physical grasp success. Outputs must stay within this checkout.

## Training and ablations

Single GPU, reduced batch size:

```bash
python -B src/dexgrasp/train.py +experiment=Final_base algo.batch_size=8
```

Main training configuration: 8 GPUs, per-GPU batch size 512, 200,000 updates,
Muon, and the flow-matching objective:

```bash
python -B -m torch.distributed.run --nproc_per_node=8 \
  src/dexgrasp/train_ddp.py +experiment=Final_base
```

Replace `Final_base` with `Final_RTJ`, `Final_RTJ34token`, `Final_SE32`,
`Final_base_1token`, or `Final_lsd3` for model ablations. The single-step
FlowMap variant has its own configuration. W&B is disabled by default.
Scaling configurations and historical reproducibility limits are described in
[EXPERIMENTS.md](docs/EXPERIMENTS.md).

Sampling without ranking uses the main checkpoint:

```bash
python -B src/dexgrasp/sample.py -e Final_base --no-ranking \
  --output-dir output/samples/no_ranking
```

Multi-GPU sampling:

```bash
python -B -m torch.distributed.run --nproc_per_node=2 \
  src/dexgrasp/sample_ddp.py -e Final_base \
  --output-dir output/samples/main_ddp
```

For physical simulation evaluation, use
[DexGraspBench](https://github.com/JYChen18/DexGraspBench) with the corresponding
experiment settings. The full benchmark is not bundled here.

## Layout

```text
src/dexgrasp/          Models, data loading, training and sampling
configs/experiments/   Standalone sampling configurations
assets/hand/shadow/    Shadow Hand XML and meshes for IK
ik.py                 Batch keypoint-to-joint reconstruction
data_proc.py          Keypoint-enriched grasp arrays to HDF5
pc_proc.py            Complete object point clouds to HDF5
docs/                 Installation, formats, experiments and validation
licenses/             Third-party license texts
tools/check_release.py  Configuration and source checks without a GPU
```

`work/` is created automatically for temporary files, library caches and local
validation runs. `output/` stores generated results. Both are ignored by Git;
`work/` can be removed when no process is using it.

## Validation

The prepared implementation has been checked for strict loading of all six
checkpoints, finite model outputs and training gradients, numerical agreement
of the main model with the retained source, and a small data → sampling → IK run.
See [VALIDATION.md](docs/VALIDATION.md) for scope and limits. A fresh installation,
full retraining and full physical benchmark have not been run as part of this
release preparation.

## Citation

```bibtex
@article{huang2026kpgrasp,
  title={KPGrasp: Scalable Keypoint Flow Matching for Dexterous Grasp Generation},
  author={Huang, Yuansen and Chen, Jiayi and Liu, Haoran and Ke, Yubin and
          Han, Bing and Lyu, Jiangran and Yan, Mi and Yi, Li and Wang, He},
  journal={arXiv preprint arXiv:2606.09314},
  year={2026}
}
```

## License

KPGrasp contributions are provided under [CC BY-NC 4.0](LICENSE) for
noncommercial use. Included DiT-derived code carries the same restriction.
Other third-party components retain their own licenses; see [NOTICE.md](NOTICE.md).
