# Adapted from KellerJordan/Muon. Copyright (c) 2024 Keller Jordan.
# MIT license; see licenses/Muon-MIT.txt and inline contributor credits.
# KPGrasp retains the local precision and optimizer integration.
import math
import torch
from torch.optim.optimizer import Optimizer, required

from collections import defaultdict
import torch.nn as nn
from typing import Tuple, Optional, Callable
from copy import deepcopy
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#from torch.distributed.tensor import DTensor, Replicate
## normed optimizers

# class
def exists(val):
    return val is not None


import os
import torch
import torch.distributed as dist
from torch import Tensor

def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    # X = G.bfloat16()
    X = torch.clone(G)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    https://kellerjordan.github.io/posts/muon/

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer should not be used for the embedding layer, the final fully connected layer,
    or any {0,1}-D parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=5, rank=None, world_size=None):
        if (rank is None) or (world_size is None):
            raise Exception("world_size and rank params required, if you want to use this optimizer on a single GPU, pass rank=0 and world_size=1.")
        self.rank = rank
        self.world_size = world_size
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[Tensor] = [*params]
        param_groups = []
        for size in {p.numel() for p in params}:
            b = torch.empty(world_size, size, dtype=torch.bfloat16, device="cuda")
            group = dict(params=[p for p in params if p.numel() == size],
                         update_buffer=b, update_buffer_views=[b[i] for i in range(world_size)])
            param_groups.append(group)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            update_buffer: Tensor = group["update_buffer"]
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[Tensor] = group["params"]
            handle = None
            params_world = None
            def update_prev(): # optimized Muon implementation contributed by @YouJiacheng
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    p_world.add_(g_world.view_as(p_world),
                                 alpha=-group["lr"] * 0.2 * max(p_world.size(-2), p_world.size(-1))**0.5)
                    # p_world.add_(g_world.view_as(p_world),
                    #              alpha=-group["lr"] * max(1, p_world.size(-2) / p_world.size(-1))**0.5)
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: Tensor = state["momentum_buffer"]
                    buf.lerp_(g, 1 - group["momentum"])
                    g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                    if g.ndim == 4: # for the case of conv filters
                        g = g.view(len(g), -1)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"]).flatten()
                else:
                    g = update_buffer_views[self.rank]
                if base_i > 0:
                    update_prev() # async all_gather instead of sync all_reduce by @YouJiacheng
                if g.dtype != update_buffer.dtype:
                    g = g.to(update_buffer.dtype)
                handle = dist.all_gather_into_tensor(update_buffer, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()

class MuonWithAuxAdamW(torch.optim.Optimizer):
    """
    MuonWithAuxAdam - Muon optimizer with an auxiliary Adam optimizer for non-2D parameters

    This optimizer combines the Muon optimizer for 2D parameters with a standard Adam optimizer
    for non-2D parameters (e.g., embeddings, final layers). This allows for effective optimization
    of models with a mix of parameter types.

    Arguments:
        param_groups: A list of parameter groups, each containing:
            - params: The parameters to optimize.
            - use_muon: A boolean indicating whether to use Muon (True) or Adam (False) for this group.
            - lr: The learning rate for this group.
            - weight_decay: The weight decay for this group.
            - betas: The beta coefficients for Adam (only used if use_muon is False).
        muon_kwargs: Additional keyword arguments to pass to the Muon optimizer.
        adam_kwargs: Additional keyword arguments to pass to the Adam optimizer.
    """
    def __init__(self, param_groups, muon_kwargs=None, adam_kwargs=None):
        muon_params = []
        adam_params = []
        for group in param_groups:
            if group.pop("use_muon", True):
                muon_params = group["params"]
            else:
                adam_params = group["params"]
        self.muon_optimizer = Muon(muon_params, **(muon_kwargs or {}))
        self.adam_optimizer = torch.optim.AdamW(adam_params, **(adam_kwargs or {}))
        super().__init__(param_groups, {})

    @torch.no_grad()
    def step(self):
        self.muon_optimizer.step()
        self.adam_optimizer.step()
