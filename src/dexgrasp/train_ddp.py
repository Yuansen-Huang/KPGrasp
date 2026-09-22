import _bootstrap
import sys
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
)
import hydra
from omegaconf import DictConfig
from tqdm import trange

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dexgrasp.utils.logger import Logger
from dexgrasp.utils.util import set_seed
from dexgrasp.dataset import create_train_ddploader
from dexgrasp.network.model import create_model
from dexgrasp.network.muon import MuonWithAuxAdamW

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def setup(rank, world_size, local_rank):
    """初始化 DDP"""
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)


def cleanup():
    """清理 DDP"""
    dist.destroy_process_group()


def reduce_tensor(tensor, world_size):
    """将所有 GPU 上的 tensor 进行求和"""
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt


@hydra.main(config_path="config", config_name="base", version_base=None)
def main(config: DictConfig) -> None:
    # DDP setup
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    setup(rank, world_size, local_rank)

    torch.set_default_dtype(torch.float32)

    set_seed(config.seed + rank)  # 为不同进程设置不同种子
    device = f"cuda:{local_rank}"

    # Logger 只在 rank 0 进程上创建
    logger = Logger(config) if rank == 0 else None
    ckpt_holder = [config.ckpt if rank == 0 else None]
    dist.broadcast_object_list(ckpt_holder, src=0)
    config.ckpt = ckpt_holder[0]

    train_loader, val_loader = create_train_ddploader(
        config, rank, world_size, local_rank=local_rank
    )

    model = create_model(config.algo.model).to(device)
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True,
    )

    if config.algo.opt == "adam":
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=config.algo.lr
        )
    elif config.algo.opt == "muon":
        hidden_weights = []
        nonhidden_params = []
        skip_params = ["norm", "adaln", "final_layer", "embed", "rms", "bn"]

        for name, param in model.named_parameters():
            if param.ndim >= 2 and all(
                [skip_key not in name.lower() for skip_key in skip_params]
            ):
                hidden_weights.append(param)
            else:
                nonhidden_params.append(param)
        param_groups = [
            dict(
                params=hidden_weights,
                use_muon=True,
                lr=config.algo.lr,
                weight_decay=0.01,
            ),
            dict(
                params=nonhidden_params,
                use_muon=False,
                lr=config.algo.lr,
                betas=(0.9, 0.95),
                weight_decay=0.01,
            ),
        ]
        optimizer = MuonWithAuxAdamW(
            param_groups,
            {
                "lr": config.algo.lr,
                "weight_decay": 0.01,
                "momentum": 0.95,
                "rank": rank,
                "world_size": world_size,
            },
            {"lr": config.algo.lr},
        )
    else:
        raise NotImplementedError

    # Scheduler
    if config.algo.warmup:
        warmup_step = 5000
        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_step
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=config.algo.max_iter - warmup_step,
            eta_min=config.algo.lr_min,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_step],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer, config.algo.max_iter, eta_min=config.algo.lr_min
        )

    # load ckpt if exists
    if config.ckpt is not None:
        ckpt = torch.load(config.ckpt, map_location=device, weights_only=False)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        cur_iter = int(ckpt["iter"])
        for _ in range(cur_iter):
            scheduler.step()
        if rank == 0:
            print(f"loaded ckpt from {config.ckpt}")
    else:
        cur_iter = 0
    cur_iter_tensor = torch.tensor([cur_iter], dtype=torch.long, device=device)
    dist.broadcast(cur_iter_tensor, src=0)
    cur_iter = int(cur_iter_tensor.item())

    # training
    model.train()
    for it in trange(cur_iter, config.algo.max_iter, disable=(rank != 0)):
        #torch.cuda.empty_cache()
        optimizer.zero_grad()

        # Dataloader for DDP will handle getting the next batch
        data = train_loader.get()

        # Move data to current device
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device)

        result_dict = model(data)
        loss = 0
        for k, v in result_dict.items():
            if k in config.algo.loss_weight:
                loss += config.algo.loss_weight[k] * v
            elif "loss" in k and rank == 0:
                print(f"Warning: {k} is not used in loss!")

        loss.backward()

        # 检查梯度 NaN
        grad_nan_tensor = torch.tensor([0.0], device=device)
        for p in model.parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                grad_nan_tensor[0] = 1.0
                break

        # 在所有进程中同步 NaN 状态
        dist.all_reduce(grad_nan_tensor, op=dist.ReduceOp.SUM)

        if grad_nan_tensor[0] > 0:
            optimizer.zero_grad()  # 清除坏梯度
            if rank == 0:
                print(
                    f"Iteration {it}: Found NaN in gradients on at least one GPU. Skipping update."
                )
            continue  # 跳过这个 batch
        grad_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad) for p in model.parameters() if p.grad is not None]
            ),
            p=2,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.algo.grad_clip)

        if it > 0:  # Skip first step for potential warmup issues
            optimizer.step()

        scheduler.step()

        # Log training stats from rank 0
        if (it + 1) % config.algo.log_every == 0:
            log_data = {}
            for k, v in result_dict.items():
                # Reduce tensors from all GPUs
                reduced_v = reduce_tensor(v.detach(), world_size)
                log_data[k] = reduced_v.mean().item()
            if rank == 0:
                log_data["lr"] = scheduler.get_last_lr()[0]
                log_data["grad_norm"] = grad_norm.item()
                logger.log(log_data, "train", it)

        # Save checkpoint from rank 0
        if rank == 0 and (it + 1) % config.algo.save_every == 0:
            logger.save(
                dict(
                    model=model.module.state_dict(),  # Save module state dict
                    optimizer=optimizer.state_dict(),
                    iter=it + 1,
                ),
                it + 1,
            )
        if (it + 1) % config.algo.val_every == 0:
            with torch.no_grad():
                model.eval()
                result_dicts = []
                for _ in range(config.algo.val_num):
                    data = val_loader.get()
                    result_dict = model(data)
                    result_dicts.append(result_dict)
                eval_data = {
                    k: reduce_tensor(
                        torch.stack([dic[k].detach().mean() for dic in result_dicts]).mean(),
                        world_size,
                    ).item()
                    for k in result_dicts[0].keys()
                }
                if rank == 0:
                    logger.log(eval_data, "eval", it)
                model.train()
    cleanup()


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method('spawn')
    main()
