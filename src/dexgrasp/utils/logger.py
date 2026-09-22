"""Training logs and checkpoints; sampling has a separate output path."""
from pathlib import Path
import torch
from .paths import output_path


class Logger:
    def __init__(self, cfg):
        self.save_ckpt_dir = output_path(
            Path(cfg.output_folder) / cfg.exp_folder / cfg.wandb.id / "ckpts"
        )
        self.save_ckpt_dir.mkdir(parents=True, exist_ok=True)
        if cfg.resume and cfg.ckpt is None:
            checkpoints = sorted(self.save_ckpt_dir.glob("step_*.pth"))
            if checkpoints:
                cfg.ckpt = str(checkpoints[-1])
        self.run = None
        if cfg.wandb.mode != "disabled":
            import wandb
            folder = output_path(cfg.wandb.folder)
            folder.mkdir(parents=True, exist_ok=True)
            self.run = wandb.init(
                dir=str(folder), project=cfg.wandb.project,
                group=cfg.wandb.group, id=cfg.wandb.id,
                mode=cfg.wandb.mode, resume="allow" if cfg.resume else None,
            )

    def log(self, values, mode, step):
        payload = {f"{mode}/{key}": value for key, value in values.items()}
        if self.run is not None:
            self.run.log(payload, step=step)
        else:
            print(f"step={step} {payload}", flush=True)

    def save(self, values, step):
        torch.save(values, self.save_ckpt_dir / f"step_{step:06d}.pth")
