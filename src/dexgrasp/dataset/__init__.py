from torch.utils.data import DataLoader, Subset
from omegaconf import DictConfig, ListConfig
from copy import deepcopy
from importlib import import_module
import torch
import numpy as np
import random


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id * random.randint(1, 10000))


DATASET_REGISTRY = {"kdex": ("dexgrasp.dataset.kdex", "KDexDataset")}


def get_dataset_cls(data_name):
    try:
        module_name, class_name = DATASET_REGISTRY[data_name]
    except KeyError as exc:
        raise ValueError(f"Unknown data_name: {data_name}") from exc
    return getattr(import_module(module_name), class_name)


def create_dataset(config, mode, rank=0, world_size=1):
    dataset_cls = get_dataset_cls(config.data_name)

    object_path = getattr(config.data, "object_path", None)
    if isinstance(object_path, ListConfig):
        dataset_lst = []
        for p in object_path:
            new_data_config = deepcopy(config.data)
            new_data_config.object_path = p
            dataset_lst.append(
                dataset_cls(new_data_config, mode, rank=rank, world_size=world_size)
            )
        dataset = torch.utils.data.ConcatDataset(dataset_lst)
    else:
        dataset = dataset_cls(config.data, mode, rank=rank, world_size=world_size)
    return dataset


def create_train_dataloader(config: DictConfig):
    train_dataset = create_dataset(config, mode="train")
    val_dataset = create_dataset(config, mode="eval")

    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=config.data.num_workers,
            shuffle=True,
            worker_init_fn=worker_init_fn,
        ),
        config.device,
    )
    val_loader = InfLoader(
        DataLoader(
            val_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=config.data.num_workers,
            shuffle=False,
            worker_init_fn=worker_init_fn,
        ),
        config.device,
    )
    return train_loader, val_loader


def create_test_dataloader(config: DictConfig, mode="test"):
    test_dataset = create_dataset(config, mode=mode)
    shuffle = False
    if config.data.mini_test:
        shuffle = True
    test_loader = FiniteLoader(
        DataLoader(
            test_dataset,
            batch_size=config.algo.batch_size,
            drop_last=False,
            num_workers=config.data.num_workers,
            shuffle=shuffle,
        ),
        config.device,
    )
    return test_loader


def create_test_ddploader(
    config: DictConfig, rank: int, world_size: int, mode: str = "test"
):
    """Create a distributed test loader where each rank processes a disjoint subset."""
    test_dataset = create_dataset(config, mode=mode, rank=rank, world_size=world_size)
    dataset_len = len(test_dataset)

    if getattr(config.data, "mini_test", False):
        rng = np.random.default_rng(config.seed)
        order = rng.permutation(dataset_len)
    else:
        order = np.arange(dataset_len)

    indices = order[rank::world_size]
    subset = Subset(test_dataset, indices.tolist())

    loader = DataLoader(
        subset,
        batch_size=config.algo.batch_size,
        drop_last=False,
        num_workers=config.data.num_workers,
        shuffle=False,
        # collate_fn=minkowski_collate_fn,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )

    return loader


class InfLoader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.device = device
        self.batch_idx = 0

    def get(self):
        self.batch_idx += 1
        try:
            data = next(self.iter_loader)
        except StopIteration:
            self.iter_loader = iter(self.loader)
            data = next(self.iter_loader)

        for k, v in data.items():
            if type(v).__module__ == "torch":
                if (
                    "Int" not in v.type()
                    and "Long" not in v.type()
                    and "Short" not in v.type()
                ):
                    v = v.float()
                data[k] = v.to(self.device)
        return data


class FiniteLoader:
    # a simple wrapper for DataLoader which can get data infinitely
    def __init__(self, loader: DataLoader, device: str):
        self.loader = loader
        self.iter_loader = iter(self.loader)
        self.device = device

    def __len__(self):
        return len(self.iter_loader)

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.iter_loader)
        for k, v in data.items():
            if type(v).__module__ == "torch":
                if (
                    "Int" not in v.type()
                    and "Long" not in v.type()
                    and "Short" not in v.type()
                ):
                    v = v.float()
                data[k] = v.to(self.device)
        return data




def create_train_ddploader(config, rank, world_size, local_rank=None):
    """为 DDP 创建 dataloader"""
    train_dataset = create_dataset(
        config, mode="train", rank=rank, world_size=world_size
    )
    val_dataset = create_dataset(config, mode="eval", rank=rank, world_size=world_size)

    if local_rank is None:
        local_rank = rank
    device = f"cuda:{local_rank}"
    train_loader = InfLoader(
        DataLoader(
            train_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=config.data.num_workers,
            shuffle=True,
            prefetch_factor=4 if config.data.num_workers else None,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        ),
        device,
    )
    val_loader = InfLoader(
        DataLoader(
            val_dataset,
            batch_size=config.algo.batch_size,
            drop_last=True,
            num_workers=config.data.num_workers,
            prefetch_factor=4 if config.data.num_workers else None,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
        ),
        device,
    )

    return train_loader, val_loader
