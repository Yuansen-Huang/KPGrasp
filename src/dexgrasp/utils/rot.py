import numpy as np
import torch


def numpy_normalize(vec):
    return vec / (np.linalg.norm(vec, axis=-1, keepdims=True) + 1e-6)


def numpy_quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = np.split(quaternions, 4, -1)

    two_s = 2.0 / (quaternions * quaternions).sum(-1, keepdims=True)

    o = np.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )

    return o.reshape(quaternions.shape[:-1] + (3, 3))


def proper_svd(rot: torch.Tensor):
    """
    compute proper svd of rotation matrix
    rot: (B, 3, 3)
    return rotation matrix (B, 3, 3) with det = 1
    """
    u, s, v = torch.svd(rot.double())
    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum("bij,bkj->bik", u, v)))
        diag = torch.stack(
            [torch.ones_like(s[:, 0]), torch.ones_like(s[:, 1]), sign], dim=-1
        )
        diag = torch.diag_embed(diag)
    return torch.einsum("bij,bjk,blk->bil", u, diag, v).to(rot.dtype)

from scipy.spatial.transform import Rotation as R

def add_noise_rot(grasp_rot, max_deg=5.0):
    """
    grasp_rot: [B, 3, 3] batch of rotation matrices
    max_deg:   最大扰动角度（单位: 度）
    return:    [B, 3, 3]
    """
    B = grasp_rot.shape[0]

    # 随机旋转轴 (B,3)，并归一化
    axis = torch.randn(B, 3)
    axis = axis / axis.norm(dim=1, keepdim=True)

    # 随机角度 [-max_deg, max_deg] (单位: 度)
    angles = (torch.rand(B) * 2 - 1) * max_deg

    # 轴角向量，注意 as_rotvec 默认是弧度，所以要转一下
    axis_angle = axis * np.deg2rad(angles).unsqueeze(1)

    # 用 scipy 生成扰动旋转矩阵 (B,3,3)
    R_perturb = torch.tensor(R.from_rotvec(axis_angle.numpy()).as_matrix(), dtype=grasp_rot.dtype, device=grasp_rot.device)

    return R_perturb @ grasp_rot
