# Experiments and reproducibility

The released main configuration is **28 keypoints, three phases, five Euler
steps, 100 candidates and top-10 selection**. All six retained checkpoints
are from update 200,000.

| Experiment | Representation / variant | Historical logged GSR |
|---|---|---:|
| `Final_base` | Keypoint flow matching | 76.309802% |
| `Final_RTJ` | Euclidean pose and joint angles | 62.544024% |
| `Final_RTJ34token` | Pose and joints, 34 tokens | 66.157332% |
| `Final_SE32` | SE(3) pose and joints | 55.004675% |
| `Final_base_1token` | Keypoints in one token | 55.434783% |
| `Final_lsd3` | Single-step FlowMap, 16 layers | 52.678043% |
| `Final_base --no-ranking` | Main model with random candidate selection | 66.988468% |

These values are copied from historical evaluation logs. They are not fresh
measurements of the cleaned release, and some historical counts/statistics
differ from paper tables. See [CHECKPOINTS.md](CHECKPOINTS.md) for file hashes.

## Main result

The `Final_base_shadow_wdensity` evaluation recorded 97,936 successful grasps
out of 128,340, giving 0.7630980208820322 GSR. It also recorded object success
0.9727286894187315, penetration 0.00242423246170322 m, and contact consistency
0.0055630853133908 m. Its conversion record points to
`shadow_kdex_Final_base/tests/step_200000`.

The experiment's mutable training-directory configuration had subsequently
been overwritten by a sampling run. This release uses a standalone, explicit
five-step configuration. The historical log establishes the result/checkpoint
association; the released five-step path has not been re-evaluated over the
full benchmark.

## Implementation notes

- **Ranking:** the retained solver accumulates negative divergence from zero.
  It omits the initial Gaussian log density. The resulting ranking score is
  preserved for checkpoint behavior; it should not be described as a complete,
  normalized log likelihood.
- **Single token:** the checkpoint requires a `[1,1,512]` position embedding,
  `[512,252]` input projection and `[1,1,252]` normalization state. The explicit
  single-token option restores this shape contract; it does not independently
  establish historical numerical equivalence.
- **RTJ:** inference now projects the predicted rotation to a proper rotation
  with SVD, consistent with RTJ34. This changes the current source behavior;
  its effect on physical GSR requires re-evaluation. Later FK auxiliary-loss
  branches are omitted.
- **SE(3):** `Final_SE32` maps to the `Final_SE3_2` checkpoint, not the early
  `Final_SE3` run. It retains 20 internal pose updates and 20 joint updates.
  These are separate from the main model's five-step setting.
- **FlowMap:** `Final_lsd3` retains the 16-layer, single-step configuration and
  consistency objective. Later debugging variants are omitted.

## Scaling

Training accepts `+experiment=scaling_layers_2`, `scaling_layers_4`,
`scaling_layers_16`, `scaling_batch_1`, `scaling_batch_8`,
`scaling_batch_64`, `scaling_data_0.001`, `scaling_data_0.01` and
`scaling_data_0.1`. Batch size is per GPU; multiply by the number of training
processes for the global batch.

Data fractions are selected deterministically per object using
`data.grasp_fraction` and `data.subset_seed`. The exact historical subset
indices were not preserved, so these settings do not reconstruct the original
subsets exactly. Scaling checkpoints are not bundled.

## IK

IK retains wrist registration from seven palm sites followed by position-only
Mink tasks, using quadprog, time step 1, regularization 1e-5, and at most five
iterations. The original early-exit criterion `vel.max() * dt < 0.1` is
preserved. The default fitting-statistic threshold is 5 mm; it does not alter
the optimization.

The code is extracted from the experiment's DexGraspBench IK implementation.
Import-time model loading, absolute paths, unrelated visualization and repeated
benchmark scanning have been removed. SVD rejects reflected wrist transforms.
The hand XML and its nine meshes are included locally.

## Evaluation scope

This repository provides sampling and IK. Full physical evaluation requires
[DexGraspBench](https://github.com/JYChen18/DexGraspBench), object assets and
the matching benchmark configuration. A successful IK fit alone does not
establish a successful grasp.
