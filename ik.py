"""Convert KPGrasp keypoints to pose/joints and report IK reconstruction error."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src/dexgrasp"))
import _bootstrap
import argparse
import json
import multiprocessing as mp
import time
import numpy as np
from dexgrasp.kinematics.ik import ShadowHandIK, IK_PHASES
from dexgrasp.utils.paths import ROOT, output_path

_solver = None


def initialize_worker(xml_path):
    global _solver
    _solver = ShadowHandIK(xml_path)


def convert_file(job):
    source, destination = job
    try:
        record = np.load(source, allow_pickle=True).item()
        stats = {}
        for phase in IK_PHASES:
            candidates = [record.get(f"{phase}_kp"), record.get(f"{phase}_qpos")]
            keypoints = next((np.asarray(x) for x in candidates
                              if x is not None and np.asarray(x).shape[-2:] == (28, 3)), None)
            if keypoints is None:
                raise ValueError(f"Missing {phase} keypoints with shape (..., 28, 3)")
            shape = keypoints.shape[:-2]
            results, phase_stats = [], []
            for points in keypoints.reshape(-1, 28, 3):
                result, info = _solver.solve(points, return_stats=True)
                results.append(result)
                phase_stats.append(info)
            record[f"{phase}_kp"] = keypoints
            record[f"{phase}_qpos"] = np.stack(results).reshape(*shape, -1)
            stats[phase] = phase_stats
        destination = output_path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, record)
        return stats, None
    except Exception as exc:
        return {}, f"{source}: {type(exc).__name__}: {exc}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="One .npy file or directory (read only).")
    parser.add_argument("--output", required=True, help="New output directory inside KPGrasp.")
    parser.add_argument("--xml", default=str(ROOT / "assets/hand/shadow/right_hand_w_kp.xml"))
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive.")
    source = Path(args.input).expanduser().resolve()
    destination = output_path(args.output)
    if not source.exists():
        parser.error(f"Input does not exist: {source}")
    if destination == source or (source.is_dir() and destination.is_relative_to(source)):
        parser.error("Output must be separate from the input.")
    if destination.exists() and any(destination.iterdir()):
        parser.error("Output directory must be empty, to avoid overwriting prior results.")
    paths = sorted(source.rglob("*.npy")) if source.is_dir() else [source]
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        parser.error("No .npy input files found.")
    jobs = [(str(path), str(destination / (path.relative_to(source) if source.is_dir() else path.name)))
            for path in paths]
    totals = {phase: {"attempts": 0, "fit_success": 0, "mean_l2_sum": 0.0, "max_l2_sum": 0.0}
              for phase in IK_PHASES}
    errors, error_count = [], 0
    started = time.perf_counter()
    if args.workers == 1:
        initialize_worker(args.xml)
        iterator = map(convert_file, jobs)
        pool = None
    else:
        pool = mp.Pool(min(args.workers, len(jobs)), initializer=initialize_worker, initargs=(args.xml,))
        iterator = pool.imap_unordered(convert_file, jobs)
    try:
        for phases, error in iterator:
            if error:
                error_count += 1
                if len(errors) < 10:
                    errors.append(error)
            for phase, stats in phases.items():
                for info in stats:
                    total = totals[phase]
                    total["attempts"] += 1
                    total["fit_success"] += int(info["fit_success"])
                    total["mean_l2_sum"] += info["kp_mean_l2"]
                    total["max_l2_sum"] += info["kp_max_l2"]
    finally:
        if pool:
            pool.close()
            pool.join()
    elapsed = time.perf_counter() - started
    report = {"files": len(paths), "failed_files": error_count, "errors": errors,
              "threshold_m": 0.005, "seconds": elapsed, "workers": min(args.workers, len(jobs)),
              "phases": {}}
    for phase, total in totals.items():
        n = total["attempts"]
        report["phases"][phase] = {
            "attempts": n, "fit_success": total["fit_success"],
            "fit_success_rate": total["fit_success"] / n if n else None,
            "mean_keypoint_error_mm": 1000 * total["mean_l2_sum"] / n if n else None,
            "mean_max_keypoint_error_mm": 1000 * total["max_l2_sum"] / n if n else None,
        }
    report["seconds_per_grasp_pipeline"] = elapsed / totals["grasp"]["attempts"] if totals["grasp"]["attempts"] else None
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "ik_stats.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if error_count:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
