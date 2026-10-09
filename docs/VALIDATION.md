# Validation scope

The cleanup was checked in the existing GPU environment described in
[INSTALL.md](INSTALL.md).

| Check | Observed result |
|---|---|
| Six checkpoint state dictionaries | Strict loading passed for all six |
| Six model sampling paths | Finite outputs with expected shapes |
| Main model comparison with retained source, same input and seed | Maximum keypoint difference 0; maximum score difference 0 |
| Main model and five model ablations | Finite losses and gradients in forward/backward checks |
| Eight known-pose IK/FK round trips | Maximum keypoint error 0.242741 mm |
| One real Dexonomy sampling batch | Ten selected grasps written |
| Three-phase IK on those ten grasps | All files processed |
| Grasp-phase IK fit on the small batch | Mean point error 0.572079 mm; mean maximum error 1.331430 mm |

These are development checks, not full retraining or physical simulation.
The 76.3% success result in the README is a historical benchmark result.
Ten successful IK fits do not measure grasp success.

`tools/check_release.py` performs repeatable checks without CUDA, data or weights:
Python syntax, sampling/training configuration resolution, main settings,
relative mesh references, documentation links, and source-distribution exclusions.
The GitHub Actions workflow runs that same check.

`tools/check_environment.py` checks installed direct dependency versions,
native-library imports and GPU availability. It does not install packages or
change the environment.

The release preparation changes paths, documentation and attribution; the model
computations and checkpoint bytes remain unchanged from the validated cleanup.
A clean installation and the full benchmark remain untested.

## GitHub preparation checks

The release preparation rechecked all 6 sampling configurations and 15 training
configurations without loading data or weights. The installed dependencies and
CUDA imports passed. A fresh main-model sampling run wrote 10 grasps, and all
10 completed three-phase IK with no file failures. Missing-weight and missing-data
CLI errors were also checked. The source archive is checked after extraction into
a separate directory without datasets or checkpoints.
