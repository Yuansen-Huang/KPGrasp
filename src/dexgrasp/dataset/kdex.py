import os
from os.path import join as pjoin
from glob import glob
import random
import numpy as np
from torch.utils.data import Dataset

from dexgrasp.utils.rot import numpy_quaternion_to_matrix
from dexgrasp.utils.util import load_json, load_scene_cfg


def import_h5py():
    import h5py

    return h5py


class KDexDataset(Dataset):
    def __init__(
        self,
        config: dict,
        mode: str,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.config = config
        self.mode = mode
        self.rank = rank
        self.world_size = world_size

        self.index_json = load_json(self.config.index_path)
        self.object_pc_folder = pjoin(self.config.object_path, self.config.pc_path)

        if mode == "train" or mode == "eval":
            self.init_train_eval(mode)
        elif mode == "test":
            self.init_test()
        return

    def init_train_eval(self, mode):
        h5py = import_h5py()
        split_name = "test" if mode == "eval" else "train"
        self.all_npy_lst = {}
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )


        p = len(self.obj_id_lst)
        start_idx = p * self.rank // self.world_size
        end_idx = p * (self.rank + 1) // self.world_size
        self.obj_id_lst = list(self.obj_id_lst)[start_idx:end_idx]

        fraction = float(getattr(self.config, "grasp_fraction", 1.0))
        if not 0 < fraction <= 1:
            raise ValueError("grasp_fraction must be in (0, 1]")
        self.grasp_subset = {}
        subset_rng = np.random.default_rng(int(getattr(self.config, "subset_seed", 0)))
        if fraction < 1:
            for obj in sorted(self.obj_id_lst):
                if obj in self.index_json:
                    start, count, _ = self.index_json[obj]
                    self.grasp_subset[obj] = start + subset_rng.choice(
                        count, max(1, round(count * fraction)), replace=False
                    )
        self.data_num = 0
        h5_file_lst = []
        for obj in list(self.index_json.keys()):
            if obj not in self.obj_id_lst:
                self.index_json.pop(obj)
            else:
                _, grasp_num, shard_name = self.index_json[obj]
                self.data_num += grasp_num
                if shard_name not in h5_file_lst:
                    h5_file_lst.append(shard_name)
        self.open_shared = {}
        if self.config.pre_load:
            for h5_file in h5_file_lst:
                self.open_shared[h5_file] = {}
                with h5py.File(os.path.join(self.config.grasp_path, h5_file), "r") as f:
                    for k in f.keys():
                        self.open_shared[h5_file][k] = f[k][:]

        with h5py.File(os.path.join(self.object_pc_folder, "../compl_pc.h5"), "r") as f:
            self.obj_pc_h5 = f["data"][:]
            all_obj_id = f["filenames"][:]
        self.obj_id2num = {}
        for obj_num, obj_id in enumerate(all_obj_id):
            self.obj_id2num[obj_id.decode("utf-8")] = obj_num
        print(f"mode: {mode}, grasp data num: {self.data_num}")
        return

    def init_test(self):
        split_name = self.config.test_split
        self.obj_id_lst = []
        self.test_cfg_lst = []
        self.obj_id_lst = load_json(
            pjoin(self.config.object_path, self.config.split_path, f"{split_name}.json")
        )

        if self.config.mini_test:
            self.obj_id_lst = self.obj_id_lst[:100]
        for o in self.obj_id_lst:
            self.test_cfg_lst.extend(
                glob(
                    pjoin(
                        self.config.object_path,
                        "scene_cfg",
                        o,
                        self.config.test_scene_cfg,
                    )
                )
            )
        self.data_num = len(self.test_cfg_lst)
        print(f"Test split: {split_name}, object cfg num: {len(self.test_cfg_lst)}")
        return

    def __len__(self):
        return self.data_num

    def __getitem__(self, id: int):
        ret_dict = {}

        if self.mode == "train" or self.mode == "eval":
            h5py = import_h5py()
            rand_obj_id = random.choice(list(self.index_json.keys()))
            start_id, grasp_num, shard_name = self.index_json[rand_obj_id]
            rand_grasp_id = (int(random.choice(self.grasp_subset[rand_obj_id]))
                             if rand_obj_id in self.grasp_subset else
                             random.randint(start_id, start_id + grasp_num - 1))

            if shard_name not in self.open_shared.keys():
                h5_path = os.path.join(self.config.grasp_path, shard_name)
                self.open_shared[shard_name] = {}
                with h5py.File(h5_path, "r") as f:
                    for k in f.keys():
                        self.open_shared[shard_name][k] = f[k][:]

            data = self.open_shared[shard_name]
            pregrasp_qpos = data["pregrasp_qpos"][rand_grasp_id]
            grasp_qpos = data["grasp_qpos"][rand_grasp_id]
            squeeze_qpos = data["squeeze_qpos"][rand_grasp_id]
            pregrasp_kp = data["pregrasp_kp"][rand_grasp_id]
            grasp_kp = data["grasp_kp"][rand_grasp_id]
            squeeze_kp = data["squeeze_kp"][rand_grasp_id]
            scene_scale = data["scene_scale"][rand_grasp_id]

            robot_pose = np.stack(
                [pregrasp_qpos, grasp_qpos, squeeze_qpos],
                axis=-2,
            )[None]
            pc = scene_scale * self.obj_pc_h5[self.obj_id2num[rand_obj_id]]

            idx = np.random.choice(pc.shape[0], self.config.num_points, replace=True)
            pc = pc[idx]

            ret_dict["grasp_scale"] = scene_scale
            ret_dict["hand_trans"] = robot_pose[:, :, :3]  # (K, n, 3)
            ret_dict["hand_rot"] = numpy_quaternion_to_matrix(
                robot_pose[:, :, 3:7]
            )  # (K, n, 3, 3)
            ret_dict["hand_joint"] = robot_pose[:, :, 7:]  # (K, n, Q)
            ret_dict["pregrasp_qpos"] = np.concatenate([ret_dict["hand_trans"][:,0], ret_dict["hand_rot"][:,0].reshape(-1, 9), ret_dict["hand_joint"][:,0]], axis=-1)  # (K, D)
            ret_dict["grasp_qpos"] = np.concatenate([ret_dict["hand_trans"][:,1], ret_dict["hand_rot"][:,1].reshape(-1, 9), ret_dict["hand_joint"][:,1]], axis=-1)  # (K, D)
            ret_dict["squeeze_qpos"] = np.concatenate([ret_dict["hand_trans"][:,2], ret_dict["hand_rot"][:,2].reshape(-1, 9), ret_dict["hand_joint"][:,2]], axis=-1)  # (K, D)
            ret_dict["pregrasp_kp"] = pregrasp_kp[None]  # (K, n, J, 3)
            ret_dict["grasp_kp"] = grasp_kp[None]  # (K, n, J, 3)
            ret_dict["squeeze_kp"] = squeeze_kp[None]  # (K, n, J, 3)

        elif self.mode == "test":

            scene_path = self.test_cfg_lst[id % len(self.test_cfg_lst)]

            scene_cfg = load_scene_cfg(scene_path)
            ret_dict["scene_id"] = scene_cfg["scene_id"]

            # read point cloud
            pc_path = (
                scene_path.replace(
                    "scene_cfg", "vision_data/complete_point_cloud"
                ).rsplit("/", 2)[0]
                + "/pc.npy"
            )
            pc = np.load(pc_path)
            pc *= scene_cfg["scene"][scene_cfg["task"]["obj_name"]]["scale"]

            idx = np.random.choice(pc.shape[0], self.config.num_points, replace=True)
            pc = pc[idx]


            ret_dict["grasp_scale"] = scene_cfg["scene"][scene_cfg["task"]["obj_name"]][
                "scale"
            ]

            ret_dict["save_path"] = pjoin(
                scene_cfg["scene_id"], os.path.basename(pc_path)
            )
            ret_dict["scene_path"] = scene_path

        ret_dict["point_clouds"] = pc  # (N, 3)
        return ret_dict
