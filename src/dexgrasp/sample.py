"""Sample saved experiments without changing training configurations."""
import _bootstrap
import argparse
from pathlib import Path
import json
import os

from omegaconf import OmegaConf
from dexgrasp.utils.paths import ROOT, output_path


def load_config(experiment, config_path=None, overrides=()):
    name = experiment.removeprefix("shadow_kdex_")
    path = Path(config_path) if config_path else ROOT / "configs/experiments" / f"{name}.yaml"
    cfg = OmegaConf.load(path)
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    OmegaConf.resolve(cfg)
    return cfg


def save_samples(save_root, data, step, suffixes, rank):
    import numpy as np
    import torch
    for index, suffix in enumerate(suffixes):
        base = output_path(Path(save_root) / f"step_{step:06d}" / f"rank_{rank}" / suffix)
        base.parent.mkdir(parents=True, exist_ok=True)
        for sample in range(data["grasp_qpos"].shape[1]):
            record = {}
            for key, value in data.items():
                if key == "scene_path":
                    record[key] = value[index]
                elif key == "grasp_scale":
                    item = value[index]
                    record[key] = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else item
                else:
                    record[key] = value[index, sample].detach().cpu().numpy()
            np.save(base.with_name(f"{base.stem}_{sample}.npy"), record)


def main(distributed=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-e", "--experiment", "--exp_name", default="Final_base")
    parser.add_argument("--config")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--limit", type=int, help="Maximum batches per rank (smoke test).")
    parser.add_argument("--no-ranking", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args, overrides = parser.parse_known_args()
    if any("=" not in value for value in overrides):
        parser.error("Overrides must be key=value.")
    cfg = load_config(args.experiment, args.config, overrides)
    if args.checkpoint:
        cfg.ckpt = str(Path(args.checkpoint).resolve())
    if cfg.data_name != "kdex":
        parser.error("Only Dexonomy (kdex) is included.")
    if not 0 < cfg.algo.test_topk <= cfg.algo.test_grasp_num:
        parser.error("Require 0 < test_topk <= test_grasp_num.")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    label = cfg.exp_name + ("_no_ranking" if args.no_ranking else "")
    save_root = output_path(args.output_dir or ROOT / "output/samples" / label)
    if args.dry_run:
        print(OmegaConf.to_yaml(cfg))
        print(f"Sample output: {save_root}")
        return

    if not cfg.ckpt or not Path(cfg.ckpt).is_file():
        parser.error("Checkpoint is missing. See docs/CHECKPOINTS.md or set --checkpoint.")
    required_data = [Path(cfg.data.index_path)]
    required_data.extend(
        Path(folder) / cfg.data.split_path / f"{cfg.data.test_split}.json"
        for folder in cfg.data.object_path
    )
    missing = [str(path) for path in required_data if not path.is_file()]
    if missing:
        parser.error(
            "Dexonomy inputs are missing. Set DEXONOMY_DATA; see docs/DATA.md. "
            + "Missing: " + ", ".join(missing)
        )

    import torch
    import torch.distributed as dist
    from tqdm import tqdm
    from dexgrasp.dataset import create_test_ddploader
    from dexgrasp.network.model import create_model
    from dexgrasp.utils.util import set_seed

    rank = int(os.environ.get("RANK", 0)) if distributed else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if distributed else 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0)) if distributed else 0
    device = torch.device(f"cuda:{local_rank}" if distributed else cfg.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        torch.set_default_dtype(torch.float32)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        set_seed(cfg.seed + rank)
        model = create_model(cfg.algo.model)
        checkpoint = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        step = int(checkpoint["iter"])
        del checkpoint
        model.to(device).eval()
        loader = create_test_ddploader(cfg, rank, world_size)
        save_root.mkdir(parents=True, exist_ok=True)
        # This record is separate from every copied .hydra training snapshot.
        (save_root / f"sampling_config_rank{rank}.yaml").write_text(OmegaConf.to_yaml(cfg))
        (save_root / f"sampling_options_rank{rank}.json").write_text(
            json.dumps({"ranking": not args.no_ranking, "limit": args.limit,
                        "rank": rank, "world_size": world_size}, indent=2)
        )
        with torch.no_grad():
            for batch_number, batch in enumerate(tqdm(loader, disable=rank != 0)):
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = (value.float() if value.is_floating_point() else value).to(device)
                poses, scores, scenes = model.sample(
                    batch, cfg.algo.test_grasp_num, cfg.algo.test_topk
                )
                if args.no_ranking:
                    indices = torch.stack([
                        torch.randperm(poses.shape[1], device=device)[:cfg.algo.test_topk]
                        for _ in range(poses.shape[0])
                    ])
                else:
                    indices = scores.topk(cfg.algo.test_topk, dim=1).indices
                batch_indices = torch.arange(poses.shape[0], device=device)[:, None]
                poses, scores = poses[batch_indices, indices], scores[batch_indices, indices]
                if poses.shape[2] != 3:
                    raise ValueError("The saved Dexonomy pipeline expects three grasp phases.")
                record = {
                    "pregrasp_qpos": poses[:, :, 0], "grasp_qpos": poses[:, :, 1],
                    "squeeze_qpos": poses[:, :, 2], "grasp_error": -scores,
                    "scene_path": scenes,
                }
                if "grasp_scale" in batch:
                    record["grasp_scale"] = batch["grasp_scale"]
                save_samples(save_root, record, step, batch["save_path"], rank)
                if args.limit is not None and batch_number + 1 >= args.limit:
                    break
    finally:
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
