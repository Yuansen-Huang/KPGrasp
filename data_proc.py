"""Pack Dexonomy keypoint grasps into indexed HDF5 shards."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src/dexgrasp"))
import _bootstrap
import argparse
from dexgrasp.utils.paths import output_path
import os
import h5py
import numpy as np
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# --- CONFIGURATION ---
SOURCE_DIR = "assets/dex_kp"
OUTPUT_DIR = "assets/dex_h5_merged"
TARGET_SHARD_SIZE = 100000
INDEX_FILES = {"obj": "index_obj_grasp.json", "bin": "index_bin_obj_grasp.json"}
KEYS_TO_SAVE = [
    "grasp_qpos",
    "pregrasp_qpos",
    "squeeze_qpos",
    "pregrasp_kp",
    "grasp_kp",
    "squeeze_kp",
    "scene_scale",
    "grasp_type_id",
]


# --- HELPERS ---
def get_scale_bin(scale_float):
    if scale_float < 0.08:
        return "0.05-0.08"
    if scale_float < 0.11:
        return "0.08-0.11"
    if scale_float < 0.14:
        return "0.11-0.14"
    if scale_float < 0.17:
        return "0.14-0.17"
    return "0.17-0.20"


def save_shard(shard_id, buffers):
    filename = os.path.join(OUTPUT_DIR, f"shard_{shard_id:03d}.h5")
    with h5py.File(filename, "w") as f:
        for key, data_list in buffers.items():
            if not data_list:
                continue
            arr = np.concatenate(data_list, axis=0)
            dtype = "int32" if "id" in key else "float32"
            f.create_dataset(key, data=arr.astype(dtype), compression="lzf")
    return filename


scale_lst = [
    "scale005.npy",
    "scale008.npy",
    "scale011.npy",
    "scale014.npy",
    "scale017.npy",
    "scale020.npy",
]


# --- MERGED WORKER ---
def process_object_full(args):
    """
    Finds ALL files for a given Object ID across ALL Grasp Types,
    loads them, merges them, and returns the sorted data.
    """
    obj_id, all_gtypes, source_dir = args

    obj_data = {k: [] for k in KEYS_TO_SAVE}
    obj_bin_labels = []

    # 1. Iterate over all possible Grasp Types for this Object
    for gtype in all_gtypes:
        for scale in scale_lst:
            path = os.path.join(source_dir, gtype, obj_id, "floating", scale)

            if not os.path.exists(path):
                continue

            raw_scale_str = scale.replace("scale", "").replace(".npy", "")
            raw_factor = float(raw_scale_str) / 100.0

            # Load Data
            d = np.load(path, allow_pickle=True).item()
            n_items = len(d["grasp_qpos"])

            # Type ID
            # Assuming gtype folder starts with ID (e.g. "6_Prismatic...")
            t_int = int(gtype.split("_")[0])

            # Process Data
            real_scales = np.array(d["scene_scale"], dtype="float32") * raw_factor
            type_ids = np.full(n_items, t_int, dtype="int32")

            for k in KEYS_TO_SAVE:
                if k == "scene_scale":
                    obj_data[k].append(real_scales)
                elif k == "grasp_type_id":
                    obj_data[k].append(type_ids)
                else:
                    obj_data[k].append(d[k])

            obj_bin_labels.extend([get_scale_bin(s) for s in real_scales])

    # 2. If no data found for this object, return None
    if not obj_bin_labels:
        return None

    # 3. Flatten and Sort by Bin
    flat_data = {k: np.concatenate(obj_data[k], axis=0) for k in KEYS_TO_SAVE}
    flat_bins = np.array(obj_bin_labels)

    # Sort indices based on bin string
    sort_indices = np.argsort(flat_bins)

    sorted_bins = flat_bins[sort_indices]
    for k in KEYS_TO_SAVE:
        flat_data[k] = flat_data[k][sort_indices]

    return obj_id, flat_data, sorted_bins


# --- MAIN ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="assets/dexonomy/dex_kp_h5")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    SOURCE_DIR = args.source
    OUTPUT_DIR = str(output_path(args.output))
    if Path(OUTPUT_DIR).exists() and any(Path(OUTPUT_DIR).iterdir()):
        parser.error("Output must be empty.")
    if args.workers < 1:
        parser.error("--workers must be positive.")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_cpus = args.workers

    all_gtypes = sorted(p.name for p in Path(SOURCE_DIR).iterdir() if p.is_dir())

    all_objects = sorted({
        p.name for gtype in all_gtypes for p in (Path(SOURCE_DIR) / gtype).iterdir()
        if p.is_dir()
    })

    print(
        f"Processing {len(all_objects)} objects across {len(all_gtypes)} grasp types with {num_cpus} CPUs..."
    )

    # Prepare arguments for workers
    worker_tasks = [(obj, all_gtypes, SOURCE_DIR) for obj in all_objects]

    # 2. PARALLEL PROCESSING LOOP
    buffers = {k: [] for k in KEYS_TO_SAVE}
    items_in_shard = 0
    current_shard = 0
    idx_obj = {}
    idx_bin = {}

    with Pool(num_cpus) as pool:
        # imap_unordered is fastest as we don't care which object finishes first
        for result in tqdm(
            pool.imap_unordered(process_object_full, worker_tasks),
            total=len(worker_tasks),
        ):
            if result is None:
                continue

            o_id, flat_data, sorted_bins = result
            total_obj_count = len(sorted_bins)
            shard_name = f"shard_{current_shard:03d}.h5"

            # --- Indexing Logic ---
            idx_obj[o_id] = [items_in_shard, total_obj_count, shard_name]
            unique_bins, start_indices = np.unique(sorted_bins, return_index=True)

            for i, bin_name in enumerate(unique_bins):
                rel_start = start_indices[i]
                count = (
                    start_indices[i + 1]
                    if i < len(unique_bins) - 1
                    else total_obj_count
                ) - rel_start
                if bin_name not in idx_bin:
                    idx_bin[bin_name] = {}
                idx_bin[bin_name][o_id] = [
                    int(items_in_shard + rel_start),
                    int(count),
                    shard_name,
                ]

            # --- Buffer Logic ---
            for k in KEYS_TO_SAVE:
                buffers[k].append(flat_data[k])

            items_in_shard += total_obj_count

            # --- Flush Logic ---
            if items_in_shard >= TARGET_SHARD_SIZE:
                save_shard(current_shard, buffers)
                current_shard += 1
                items_in_shard = 0
                buffers = {k: [] for k in KEYS_TO_SAVE}

    # Final Flush
    if items_in_shard > 0:
        save_shard(current_shard, buffers)

    # 3. SAVE INDICES
    print("Saving Indices...")
    with open(os.path.join(OUTPUT_DIR, INDEX_FILES["obj"]), "w") as f:
        json.dump(idx_obj, f, indent=4)
    with open(os.path.join(OUTPUT_DIR, INDEX_FILES["bin"]), "w") as f:
        json.dump(idx_bin, f, indent=4)
    print("Done!")
