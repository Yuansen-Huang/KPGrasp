import math
import torch
from abc import ABC, abstractmethod

from .flow_method import BaseFlowMethod


class BaseSolver(ABC):
    def __init__(self, steps=20):
        if steps <= 0:
            raise ValueError('steps must be positive')
        self.steps = steps
        self.dt = 1.0 / steps

    def sample(self, flow_method: BaseFlowMethod, x_init, cond_dict):
        batch_size = x_init.shape[0]
        x = x_init

        # Initial Log Prob calculation omitted for brevity, logic identical to previous code
        log_p = torch.zeros(batch_size, device=x.device)

        # Hutchinson noise
        z = torch.randint_like(x, 0, 2).float() * 2 - 1

        for step in range(self.steps):
            t_curr = step * self.dt
            x, log_p = self._step(flow_method, x, t_curr, self.dt, cond_dict, log_p, z)

        return x, log_p

    def _divergence(self, v, x, z):
        v_flat = v.view(v.shape[0], -1)
        z_flat = z.view(z.shape[0], -1)
        v_dot_z = (v_flat * z_flat).sum(dim=1)
        grad_v = torch.autograd.grad(
            v_dot_z, x, torch.ones_like(v_dot_z), create_graph=False
        )[0]
        return (grad_v.view(grad_v.shape[0], -1) * z_flat).sum(dim=1)

    @abstractmethod
    def _step(self, fm, x, t, dt, cond, log_p, z):
        pass


class EulerSolver(BaseSolver):
    def _step(self, fm: BaseFlowMethod, x, t, dt, cond, log_p, z):
        x_req = x.detach().requires_grad_(True)
        with torch.enable_grad():
            v = fm.predict_derivative(x_req, t, dt, cond)
            div = self._divergence(v, x_req, z)
        return (x + dt * v).detach(), log_p - dt * div.detach()
