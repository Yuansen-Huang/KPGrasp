# Timestep embedding adapted from Meta DiT and OpenAI GLIDE.
# Copyright (c) Meta Platforms, Inc. and affiliates; Copyright (c) OpenAI.
# See licenses/DiT-CC-BY-NC-4.0.txt and licenses/GLIDE-MIT.txt.
# KPGrasp adds flow-matching and flow-map conditioning.
import torch
import math
from torch import nn
from abc import ABC, abstractmethod

from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256, flowmap=False):
        super().__init__()
        self.flowmap = flowmap
        self.frequency_embedding_size = frequency_embedding_size

        if flowmap:
            mlp_input_size = 2 * frequency_embedding_size
        else:
            mlp_input_size = frequency_embedding_size

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """Standard sinusoidal embedding from Glide/DDPM."""
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t, r=None):
        if self.flowmap:
            t_input = torch.cat(
                [
                    self.timestep_embedding(t, self.frequency_embedding_size),
                    self.timestep_embedding(r - t, self.frequency_embedding_size),
                ],
                dim=-1,
            )
        else:
            t_input = self.timestep_embedding(t, self.frequency_embedding_size)

        t_emb = self.mlp(t_input)
        return t_emb.unsqueeze(1)


class BaseFlowMethod(ABC):
    def __init__(self, model, time_embedder, pred_mode="v"):
        self.model = model
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.time_embedder = time_embedder
        self.pred_mode = pred_mode
        self.loss_fn = nn.MSELoss()

    @abstractmethod
    def calculate_loss(self, x_1, context):
        pass

    @abstractmethod
    def predict_derivative(self, x, t, dt, context):
        pass


class FlowMatch(BaseFlowMethod):
    def calculate_loss(self, x_1, context):
        batch_size = x_1.shape[0]
        device = x_1.device

        # 1. Sample Time & Path
        t = torch.rand(batch_size, device=device)
        x_0 = torch.randn_like(x_1)
        path = self.path.sample(t=t, x_0=x_0, x_1=x_1)

        # 2. Process Time
        t_emb = self.time_embedder(path.t)

        # 3. Model Prediction (Unified Interface)
        pred = self.model(x=path.x_t, t_emb=t_emb, context=context)

        # 4. Loss
        target = x_1 if self.pred_mode == "x" else path.dx_t
        loss = {"loss_fm": self.loss_fn(pred, target)}
        return loss

    def predict_derivative(self, x, t, dt, context):
        # Broadcast t to batch if scalar
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t, device=x.device)

        t_emb = self.time_embedder(t)
        pred = self.model(x=x, t_emb=t_emb, context=context)

        if self.pred_mode == "x":
            denom = (1 - t).clamp(min=1e-5).view(-1, 1, 1)
            return (pred - x) / denom
        return pred


class FlowMap(BaseFlowMethod):
    def _sample_logit_normal(self, batch_size, device):
        normal = torch.distributions.Normal(0.4, 1.0)
        return torch.sigmoid(normal.rsample((batch_size, 1)).to(device)).reshape(-1)

    def calculate_loss(self, x_1, context):
        batch_size, device = x_1.shape[0], x_1.device

        # --- Part A: Standard Loss (Matching straight line, r=t) ---
        x_0 = torch.randn_like(x_1)
        t_std = self._sample_logit_normal(batch_size, device)
        path = self.path.sample(t=t_std, x_0=x_0, x_1=x_1)

        # Embed time with r=t
        t_emb_std = self.time_embedder(t=path.t, r=path.t)

        pred_std = self.model(x=path.x_t, t_emb=t_emb_std, context=context)
        target_std = x_1 if self.pred_mode == "x" else path.dx_t
        loss_std = self.loss_fn(pred_std, target_std)

        # --- Part B: FlowMap Consistency (JVP) ---
        # Sample t and r
        t_samp = self._sample_logit_normal(batch_size, device)
        r_samp = self._sample_logit_normal(batch_size, device)
        t_samp = t_samp.clamp(max=0.999)
        r_samp = r_samp.clamp(max=0.999)

        # t = min, r = max
        tr = torch.cat([t_samp.unsqueeze(1), r_samp.unsqueeze(1)], dim=1)
        t_val = torch.min(tr, dim=1).values
        r_val = torch.max(tr, dim=1).values
        # r_t = torch.rand(batch_size, device=device)
        # t_val = torch.rand(batch_size, device=device) * (1 - r_t)
        # r_val = t_val + r_t

        # Sample x_t
        x_t = self.path.sample(t=t_val, x_0=x_0, x_1=x_1).x_t
        view_shape = (-1,) + (1,) * (x_t.ndim - 1)

        def xt_of_tr(t_in, r_in):
            t_emb_in = self.time_embedder(t=t_in, r=r_in)
            out = self.model(x=x_t, t_emb=t_emb_in, context=context)
            if self.pred_mode == "x":
                return (out * (r_in - t_in).view(view_shape) + x_t * (1 - r_in).view(view_shape)) / (1 - t_in).view(view_shape)
            else:
                return x_t + out * (r_in - t_in).view(view_shape)

        # Compute JVP
        x_r, dxdr = torch.func.jvp(
            xt_of_tr, (t_val, r_val), (torch.zeros_like(t_val), torch.ones_like(r_val))
        )

        # Prediction at future step r (r=r)
        t_emb_r = self.time_embedder(t=r_val, r=r_val)
        out = self.model(x=x_r, t_emb=t_emb_r, context=context)
        if self.pred_mode == "x":
            pred_r = (out - x_r) / (1 - r_val).view(view_shape)
        else:
            pred_r = out

        loss_flow = self.loss_fn(pred_r, dxdr)

        return {"loss_fm": loss_std, "loss_consis": loss_flow}


    def predict_derivative(self, x, t, dt, context):
        # Broadcast t
        if isinstance(t, float) or t.ndim == 0:
            t = torch.full((x.shape[0],), t, device=x.device)

        # FlowMap Inference: Look ahead to r = t + dt
        r = (t + dt).clamp(max=1.0)

        t_emb = self.time_embedder(t=t, r=r)
        pred = self.model(x=x, t_emb=t_emb, context=context)
        if self.pred_mode == "x":
            denom = (1 - t).clamp(min=1e-5).view(-1, 1, 1)
            return (pred - x) / denom
        return pred
