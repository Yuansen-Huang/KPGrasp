import math
from typing import Tuple

import torch
from pytorch3d.transforms import so3_exp_map, so3_log_map

DEFAULT_ACOS_BOUND: float = 1.0 - 1e-4

basis = torch.tensor(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ]
)


def acos_linear_extrapolation(
    x: torch.Tensor,
    bounds: Tuple[float, float] = (-DEFAULT_ACOS_BOUND, DEFAULT_ACOS_BOUND),
) -> torch.Tensor:
    lower_bound, upper_bound = bounds
    if lower_bound > upper_bound:
        raise ValueError("lower bound has to be smaller or equal to upper bound.")
    if lower_bound <= -1.0 or upper_bound >= 1.0:
        raise ValueError("Both lower bound and upper bound have to be within (-1, 1).")

    acos_extrap = torch.empty_like(x)
    x_upper = x >= upper_bound
    x_lower = x <= lower_bound
    x_mid = (~x_upper) & (~x_lower)

    acos_extrap[x_mid] = torch.acos(x[x_mid])
    acos_extrap[x_upper] = _acos_linear_approximation(x[x_upper], upper_bound)
    acos_extrap[x_lower] = _acos_linear_approximation(x[x_lower], lower_bound)
    return acos_extrap


def _acos_linear_approximation(x: torch.Tensor, x0: float) -> torch.Tensor:
    return (x - x0) * _dacos_dx(x0) + math.acos(x0)


def _dacos_dx(x: float) -> float:
    return (-1.0) / math.sqrt(1.0 - x * x)


def my_hat(v: torch.Tensor) -> torch.Tensor:
    return torch.einsum("...i,ijk->...jk", v, basis.to(v))


def hat_inv(h: torch.Tensor) -> torch.Tensor:
    if h.shape[-2:] != (3, 3):
        raise ValueError("Input has to be a batch of 3x3 Tensors.")

    x = h[..., 2, 1]
    y = h[..., 0, 2]
    z = h[..., 1, 0]
    return torch.stack((x, y, z), dim=-1)


def log_SO3(R1: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
    R_rel = torch.matmul(R0.transpose(-1, -2), R1)
    logvec = so3_log_map(R_rel)
    logmat = my_hat(logvec)
    return torch.matmul(R0, logmat)


def exp_SO3(v: torch.Tensor, R0: torch.Tensor) -> torch.Tensor:
    v_id = torch.matmul(R0.transpose(-1, -2), v)
    vec = hat_inv(v_id)
    R_id = so3_exp_map(vec)
    return torch.matmul(R0, R_id)


def expmap(R0: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
    v_id = torch.matmul(R0.transpose(-1, -2), tangent)
    return torch.matmul(R0, torch.linalg.matrix_exp(v_id))


def pt_to_identity(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return torch.transpose(R, dim0=-2, dim1=-1) @ v


def norm_SO3(R: torch.Tensor, T_R: torch.Tensor) -> torch.Tensor:
    r = pt_to_identity(R, T_R)
    return -torch.diagonal(r @ r, dim1=-2, dim2=-1).sum(dim=-1) / 2


def tangent_space_proj(R: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
    skew_symmetric_part = 0.5 * (M - M.transpose(-1, -2))
    return R @ skew_symmetric_part
