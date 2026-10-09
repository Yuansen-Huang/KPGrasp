# Dexonomy data

Obtain data and object assets from the
[Dexonomy project](https://github.com/JYChen18/Dexonomy) and its
[dataset page](https://huggingface.co/datasets/JiayiChenPKU/Dexonomy).
Use the original dataset terms. No dataset files are included in this repository.

Set `DEXONOMY_DATA` to the asset root, or use the default
`assets/dexonomy/` location:

```text
assets/dexonomy/
  dex_kp_h5/
    index_obj_grasp.json
    shard_000.h5
    ...
  object/
    DGN_5k/
      valid_split/train.json
      valid_split/test.json
      scene_cfg/<object-id>/floating/scale*.npy
      vision_data/complete_point_cloud/<object-id>/pc.npy
      vision_data/compl_pc.h5
    objaverse_5k/
      valid_split/train.json
      valid_split/test.json
      scene_cfg/<object-id>/floating/scale*.npy
      vision_data/complete_point_cloud/<object-id>/pc.npy
      vision_data/compl_pc.h5
```

`DGN_5k` and `objaverse_5k` are the two object collections used by the
Dexonomy experiments. Both belong to the retained data pipeline.

## Grasp representation

Training consumes **keypoint-enriched** Dexonomy records, not unmodified raw
joint arrays. Each grasp has `pregrasp_kp`, `grasp_kp` and `squeeze_kp`
with shape `(28, 3)`, plus the corresponding `*_qpos`, scale and grasp-type
metadata. Use the site names/order in
`src/dexgrasp/kinematics/ik.py` and the bundled hand XML when constructing these
keypoints. The 28-point convention must match the checkpoint.

The HDF5 packer expects existing keypoint-enriched arrays at:

```text
<keypoint-records>/<grasp-type>/<object-id>/floating/scale005.npy
<keypoint-records>/<grasp-type>/<object-id>/floating/scale008.npy
...
```

Supported scale names are 005, 008, 011, 014, 017 and 020. The record keys consumed
by the packer are listed in `KEYS_TO_SAVE` in `data_proc.py`.
This packer does not generate keypoints from raw Dexonomy grasps.

```bash
python -B data_proc.py --source /path/to/keypoint-records \
  --output assets/dexonomy/dex_kp_h5 --workers 8
```

The packer writes HDF5 shards, `index_obj_grasp.json` and a scale-bin index.
Each object-index entry is `[start_offset, grasp_count, shard_filename]`.
The output directory must be empty.

## Point-cloud packing

Training uses `vision_data/compl_pc.h5` in each object collection. Sampling
reads the scene configurations and complete point clouds. To build the
training point-cloud HDF5 file:

```bash
python -B pc_proc.py \
  --source assets/dexonomy/object/DGN_5k/vision_data/complete_point_cloud \
  --output assets/dexonomy/object/DGN_5k/vision_data/compl_pc.h5
```

Repeat for `objaverse_5k`. The HDF5 file contains `data` and `filenames`.
All object point clouds supplied to a packing run must have the same shape.
Packing outputs must be inside this checkout and must not overwrite an existing
file. External dataset roots are read as inputs.

## Dataset availability

The repository does not host the processed keypoint HDF5 shards or claim that
the upstream raw dataset already contains them. A training run requires the
format above. Sampling also loads the grasp index, even though it does not read
training grasp shards in test mode.
