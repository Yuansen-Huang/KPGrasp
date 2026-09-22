# Checkpoints

Model weights are distributed separately from the Git source tree. Public download
URLs have not been assigned yet. The source archive contains no weights.
Place the files below at the indicated paths before sampling.

The retained files are original training checkpoints containing model parameters,
optimizer state and the update number; they support strict loading by the released
models. They have not been rewritten or stripped.

| Experiment | File | Size (MiB) |
|---|---|---:|
| Final_base | `checkpoints/Final_base/step_200000.pth` | 256.47 |
| Final_RTJ | `checkpoints/Final_RTJ/step_200000.pth` | 256.78 |
| Final_RTJ34token | `checkpoints/Final_RTJ34token/step_200000.pth` | 256.45 |
| Final_SE32 | `checkpoints/Final_SE32/step_200000.pth` | 472.84 |
| Final_base_1token | `checkpoints/Final_base_1token/step_200000.pth` | 257.36 |
| Final_lsd3 | `checkpoints/Final_lsd3/step_200000.pth` | 457.35 |

The machine-readable [manifest](../configs/checkpoints.json) records SHA-256, exact
byte sizes and update numbers. To verify files from the repository root:

```bash
sha256sum -c configs/checkpoints.sha256
```

Only the main checkpoint is needed for `Final_base` and the no-ranking ablation.
The SE(3) experiment name `Final_SE32` corresponds to the historical `Final_SE3_2`
checkpoint. These files are larger than ordinary source files and are excluded
from Git; use a model-hosting service or separate release assets when distributing
them.
