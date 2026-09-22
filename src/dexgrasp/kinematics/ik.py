"""SVD registration + position-only Mink IK extracted from DexGraspBench/ik.py.

The solver equations, seven palm sites, five-iteration limit and stopping
condition retain the source implementation. No object/contact optimization.
"""
from copy import deepcopy
from pathlib import Path
import numpy as np
import mujoco
import mink
import transforms3d.quaternions as tq

IK_SUCCESS_ERROR_THRESHOLD = 0.005
IK_PHASES = ("pregrasp", "grasp", "squeeze")
kp_name_lst = ['palm_1', 'palm_2', 'palm_3', 'palm_4', 'palm_5', 'palm_6', 'palm_7', 'ff_p', 'ff_m', 'ff_d', 'ff_d2', 'mf_p', 'mf_m', 'mf_d', 'mf_d2', 'rf_p', 'rf_m', 'rf_d', 'rf_d2', 'lf_b', 'lf_p', 'lf_m', 'lf_d', 'lf_d2', 'th_p', 'th_m', 'th_d', 'th_d2']
palm_name_lst = kp_name_lst[:7]


def _kp_group_indices():
    palm_indices = np.array([kp_name_lst.index(name) for name in palm_name_lst])
    palm_index_set = set(palm_indices.tolist())
    finger_indices = np.array(
        [idx for idx in range(len(kp_name_lst)) if idx not in palm_index_set]
    )
    return {"palm": palm_indices, "finger": finger_indices}


def compute_transformation(P, Q):
    '''
    Compute the rotation R and translation t that transform P to Q.

    Parameters:
        P (np.ndarray): Source point set, shape (n, 3).
        Q (np.ndarray): Target point set, shape (n, 3).

    Returns:
        R (np.ndarray): Rotation matrix, shape (3, 3).
        t (np.ndarray): Translation vector, shape (3,).
    '''
    # Center the point sets
    mu_P = np.mean(P, axis=0)
    mu_Q = np.mean(Q, axis=0)
    P_centered = P - mu_P
    Q_centered = Q - mu_Q

    # Compute the covariance matrix
    H = np.dot(P_centered.T, Q_centered)

    # Perform SVD
    U, _, Vt = np.linalg.svd(H)

    # Compute the rotation matrix
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    t = mu_Q - np.dot(R, mu_P)

    return R, t


def fk_sph(mj_model, kp_name_lst, root_pose, qpos):
    configuration = mink.Configuration(mj_model)
    configuration.update(qpos)
    kp_trans_lst = []
    for kp_name in kp_name_lst:
        kp_pose = configuration.get_transform_frame_to_world(
            frame_name=kp_name, frame_type="site"
        )
        kp_trans = deepcopy(kp_pose.parameters()[4:])
        kp_trans_lst.append(kp_trans)
    kp_trans_lst = np.stack(kp_trans_lst)
    root_rot = tq.quat2mat(root_pose[3:])
    root_trans = root_pose[:3]
    posed_kp_trans = kp_trans_lst @ root_rot.T + root_trans
    return posed_kp_trans


def ik_sph(
    mj_model,
    kp_name_lst,
    kp_trans_lst,
    canon_palm_kp,
    return_stats=False,
    success_error_threshold=IK_SUCCESS_ERROR_THRESHOLD,
):
    # fit global root pose
    # palm_kp_lst = kp_trans_lst[: canon_palm_kp.shape[0]]
    palm_idx_lst = [kp_name_lst.index(name) for name in palm_name_lst]
    palm_kp_lst = kp_trans_lst[palm_idx_lst]
    rot, trans = compute_transformation(canon_palm_kp, palm_kp_lst)
    canon_kp_trans_lst = (kp_trans_lst - trans[None]) @ rot
    result_pose = np.concatenate([trans, tq.mat2quat(rot)])

    # ik to solve joint angles
    configuration = mink.Configuration(mj_model)
    initial_q = deepcopy(configuration.q)
    kp_tasks = []
    for kp_name in kp_name_lst:
        task = mink.FrameTask(
            frame_name=kp_name,
            frame_type="site",
            position_cost=1.0,
            orientation_cost=0.0,
            lm_damping=0.0,
        )
        kp_tasks.append(task)

    for kp_name, kp_pose, task in zip(kp_name_lst, canon_kp_trans_lst, kp_tasks):
        new_se3_pose = mink.lie.se3.SE3.identity()
        new_se3_pose.parameters()[-3:] = kp_pose
        task.set_target(new_se3_pose)

    dt = 1.0
    for i in range(5):
        vel = mink.solve_ik(configuration, kp_tasks, dt, "quadprog", 1e-5)
        configuration.integrate_inplace(vel, dt)
        if vel.max() * dt < 0.1:
            break
    result_q = deepcopy(configuration.q)
    result = np.concatenate([result_pose, result_q])
    if not return_stats:
        return result

    solved_kp_trans_lst = fk_sph(mj_model, kp_name_lst, result_pose, result_q)
    kp_l2 = np.linalg.norm(solved_kp_trans_lst - kp_trans_lst, axis=1)
    group_stats = {}
    for group, indices in KP_GROUP_INDICES.items():
        group_l2 = kp_l2[indices]
        group_stats[group] = {
            "fit_success": bool(np.max(group_l2) <= success_error_threshold),
            "kp_mean_l2": float(np.mean(group_l2)),
            "kp_max_l2": float(np.max(group_l2)),
        }
    stats = {
        "solve_success": True,
        "fit_success": bool(np.max(kp_l2) <= success_error_threshold),
        "solution_initial_q_l2": float(np.linalg.norm(result_q - initial_q)),
        "kp_mean_l2": float(np.mean(kp_l2)),
        "kp_max_l2": float(np.max(kp_l2)),
        "kp_groups": group_stats,
    }
    return result, stats


KP_GROUP_INDICES = _kp_group_indices()

class ShadowHandIK:
    """Lazy hand-model initialization; importing this module does not load assets."""
    def __init__(self, xml_path=None):
        if xml_path is None:
            xml_path = Path(__file__).resolve().parents[3] / "assets/hand/shadow/right_hand_w_kp.xml"
        self.model = mujoco.MjSpec.from_file(str(xml_path)).compile()
        self.canonical_palm = fk_sph(
            self.model, palm_name_lst, np.array([0., 0., 0., 1., 0., 0., 0.]),
            np.zeros(self.model.nq),
        )

    def solve(self, keypoints, return_stats=False):
        keypoints = np.asarray(keypoints, dtype=np.float64)
        if keypoints.shape != (len(kp_name_lst), 3) or not np.isfinite(keypoints).all():
            raise ValueError("Expected finite Shadow Hand keypoints with shape (28, 3).")
        return ik_sph(self.model, kp_name_lst, keypoints, self.canonical_palm, return_stats)

    def forward(self, pose_and_joints):
        values = np.asarray(pose_and_joints)
        return fk_sph(self.model, kp_name_lst, values[:7], values[7:])
