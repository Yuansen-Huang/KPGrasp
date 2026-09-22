# Licensing and third-party notices

KPGrasp contributions are distributed under **CC BY-NC 4.0** for noncommercial use.
See [LICENSE](LICENSE). Third-party materials retain their original licenses.
This is a noncommercial research code release; it does not grant unrestricted
commercial-use rights.

| Included material | Upstream project and attribution | Applicable license |
|---|---|---|
| DiT blocks in `src/dexgrasp/network/policy/transformer.py` and timestep embedding in `network/flow_method.py` | [DiT](https://github.com/facebookresearch/DiT), Copyright Meta Platforms, Inc. and affiliates | [CC BY-NC 4.0](licenses/DiT-CC-BY-NC-4.0.txt) |
| Sinusoidal timestep embedding referenced by DiT | [GLIDE](https://github.com/openai/glide-text2im), Copyright OpenAI | [MIT](licenses/GLIDE-MIT.txt) |
| Sparse ResNet/MinkUNet definitions in `network/backbones/mink_unet.py` | [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine), NVIDIA Corporation and Chris Choy | [MIT](licenses/MinkowskiEngine-MIT.txt) |
| Farthest-point sampling in `network/backbones/pointnet2_util/pointnet2_utils.py` | [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch), Copyright 2019 benny | [MIT](licenses/PointNet2-MIT.txt) |
| Newton–Schulz/Muon optimizer in `network/muon.py` | [Muon](https://github.com/KellerJordan/Muon), Copyright 2024 Keller Jordan; additional contributors are credited in the code | [MIT](licenses/Muon-MIT.txt) |
| Shadow Hand XML and nine mesh files in `assets/hand/shadow/` | [MuJoCo Menagerie / Shadow Hand](https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand), Copyright 2022 Shadow Robot Company Ltd | [Apache 2.0](licenses/ShadowHand-Apache-2.0.txt) |

The DiT adaptation adds point-cloud conditioning and three-phase hand outputs.
The MinkUNet adaptation exposes point-cloud tokens and global features.
The FPS helper preserves the input tensor dtype.
The Muon adaptation retains the local precision and optimizer integration.
The Shadow Hand XML adds the KPGrasp keypoint sites and uses a relative mesh directory;
the mesh files are copied without geometry edits.

IK is adapted from the KPGrasp experiment's DexGraspBench implementation.
Its retained solver settings and changes are described in [EXPERIMENTS.md](docs/EXPERIMENTS.md).
Dexonomy datasets and model checkpoints are not part of the source archive.
Dataset use and redistribution remain subject to their original terms.
Installed dependencies are separate projects under their own licenses.
