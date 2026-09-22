"""Pack complete Dexonomy object point clouds into HDF5."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src/dexgrasp"))
import _bootstrap
import argparse
from dexgrasp.utils.paths import output_path
import os
import glob
import numpy as np
import h5py
from tqdm import tqdm


def convert_npy_to_h5(source_dir, output_file, key="data"):
    """
    Reads all .npy files from source_dir and saves them into a single H5 file.

    Args:
        source_dir (str): Path to folder containing .npy files.
        output_file (str): Path for the output .h5 file.
        key (str): The group name inside the h5 file (e.g., 'data', 'points').
    """
    # 1. Gather all npy file paths
    # sorting ensures deterministic ordering in the dataset
    output_file = output_path(output_file)
    if output_file.exists():
        raise FileExistsError(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    npy_files = sorted(glob.glob(os.path.join(source_dir, "*/pc.npy")))

    num_files = len(npy_files)
    if num_files == 0:
        print("No .npy files found in the directory.")
        return

    print(f"Found {num_files} files. Inspecting first file for shape...")

    # 2. Load the first file to get shape and dtype
    sample_data = np.load(npy_files[0])
    sample_shape = sample_data.shape
    sample_dtype = sample_data.dtype

    # Final shape will be (N, points, channels), e.g., (10000, 2048, 3)
    h5_shape = (num_files,) + sample_shape

    print(f"Data Shape: {sample_shape}")
    print(f"Creating H5 file with total shape: {h5_shape}")

    # 3. Create H5 file and write data
    with h5py.File(output_file, "w") as f:
        # Create a dataset for the point clouds
        # chunks=True allows h5py to optimize storage chunks automatically
        dset = f.create_dataset(key, shape=h5_shape, dtype=sample_dtype, chunks=True)

        # Optional: Create a dataset to store filenames (useful for mapping back to IDs)
        # S256 limits filename string length to 256 bytes
        name_dset = f.create_dataset("filenames", shape=(num_files,), dtype="S256")

        print("Writing data to H5...")

        # Iterate and write one by one (or you could do batches)
        for i, filepath in tqdm(enumerate(npy_files), total=num_files):
            try:
                data = np.load(filepath)

                # Verify shape consistency
                if data.shape != sample_shape:
                    print(
                        f"Skipping {filepath}: Shape mismatch {data.shape} vs {sample_shape}"
                    )
                    continue

                dset[i] = data

                # Store filename (encode to bytes for HDF5 compatibility)
                filename = os.path.basename(os.path.dirname(filepath))
                name_dset[i] = filename.encode("utf-8")

            except Exception as e:
                print(f"Error loading {filepath}: {e}")

    print(f"Successfully saved {num_files} point clouds to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert_npy_to_h5(args.source, args.output)
