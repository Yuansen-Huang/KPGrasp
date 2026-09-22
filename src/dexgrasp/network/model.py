"""KPGrasp and the parameterization/tokenization ablations.

State-dict names are retained for the original Final_* checkpoints.
"""
import torch
import torch.nn as nn
from einops import rearrange, repeat
from scipy.spatial.transform import Rotation
from pytorch3d.transforms import matrix_to_axis_angle, matrix_to_quaternion, random_rotations
from torch.func import vmap

from .backbones import WrappedMinkUNet_coord
from .policy import DiT
from .flow_method import FlowMatch, FlowMap, TimestepEmbedder
from .ode_solver import EulerSolver
from .fm_util.util import expmap, tangent_space_proj, norm_SO3
from dexgrasp.utils.RMS import Normalization




def proper_svd(rot: torch.Tensor):
    """
    Compute proper SVD of rotation matrix.

    Args:
        rot: Rotation matrices with shape (B, 3, 3).

    Returns:
        Rotation matrices with shape (B, 3, 3) and determinant 1.
    """
    u, s, v = torch.svd(rot.double())
    with torch.no_grad():
        sign = torch.sign(torch.det(torch.einsum("bij,bkj->bik", u, v)))
        diag = torch.stack(
            [torch.ones_like(s[:, 0]), torch.ones_like(s[:, 1]), sign], dim=-1
        )
        diag = torch.diag_embed(diag)
    return torch.einsum("bij,bjk,blk->bil", u, diag, v).to(rot.dtype)


class KeyPointGraspModel(nn.Module):
    """Keypoint flow; single_token=True restores the single-token ablation."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = WrappedMinkUNet_coord(**cfg.backbone)
        self.n_kp = cfg.policy.n_kp
        self.in_channels = cfg.policy.n_frame * 3
        self.single_token = bool(getattr(cfg, "single_token", False))
        self.n_tokens = 1 if self.single_token else self.n_kp
        self.token_channels = self.n_kp * self.in_channels if self.single_token else self.in_channels
        self.policy = DiT(input_size=self.n_tokens, in_channels=self.token_channels, **cfg.policy)
        self.time_embedder = TimestepEmbedder(
            cfg.policy.hidden_size, flowmap=cfg.flow.name == "FlowMap"
        )
        self.RMS = Normalization((self.n_tokens, self.token_channels))
        methods = {"FlowMatch": FlowMatch, "FlowMap": FlowMap}
        self.method = methods[cfg.flow.name](self.policy, self.time_embedder, cfg.flow.pred_mode)
        self.solver = EulerSolver(cfg.flow.sample_step)

    def forward(self, data):
        sample_num = data["grasp_kp"].shape[1]
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        cond_dict = {"global": g_cond.unsqueeze(1), "local": l_cond}
        if self.cfg.policy.n_frame == 1:
            gt_kp = data["grasp_kp"].reshape(-1, self.n_kp, self.in_channels)
        else:
            gt_kp = torch.cat(
                [
                    data["pregrasp_kp"].reshape(-1, self.n_kp, 3),
                    data["grasp_kp"].reshape(-1, self.n_kp, 3),
                    data["squeeze_kp"].reshape(-1, self.n_kp, 3),
                ],
                dim=-1,
            )
        gt_kp = gt_kp.reshape(-1, self.n_tokens, self.token_channels)
        if self.RMS:
            gt_kp = self.RMS(gt_kp)
        loss = self.method.calculate_loss(gt_kp, cond_dict)
        return loss

    def sample(self, data, sample_num, topk_num):
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        cond_dict = {"global": g_cond.unsqueeze(1), "local": l_cond}

        batch_size = global_feature.shape[0]

        # Init Noise
        x_init = torch.randn(
            (batch_size * sample_num, self.n_tokens, self.token_channels),
            device=global_feature.device,
        )

        # Solve ODE
        x_final, log_prob = self.solver.sample(self.method, x_init, cond_dict)

        if self.RMS:
            x_final = self.RMS.inv(x_final)

        # Reshape & Select TopK
        x_reshaped = rearrange(x_final, "(b n) k c -> b n k c", b=batch_size).reshape(batch_size, sample_num, self.n_kp, self.in_channels)
        log_prob = rearrange(log_prob, "(b n) -> b n", b=batch_size)  # (B, N)

        if self.in_channels == 3:
            robot_pose = x_reshaped.unsqueeze(2)
        else:
            robot_pose = torch.stack(
                [x_reshaped[..., 0:3], x_reshaped[..., 3:6], x_reshaped[..., 6:9]], dim=2
            )

        return robot_pose, log_prob, data["scene_path"]


class RTJGraspModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = WrappedMinkUNet_coord(**cfg.backbone)
        self.n_kp = 1
        self.in_channels = (3 + 9 + 22) * 3
        self.policy = DiT(
            input_size=self.n_kp, in_channels=self.in_channels, **cfg.policy
        )

        use_flowmap = cfg.flow.name == "FlowMap"
        self.time_embedder = TimestepEmbedder(
            hidden_size=cfg.policy.hidden_size, flowmap=use_flowmap
        )

        self.RMS = Normalization((self.n_kp, self.in_channels))

        if use_flowmap:
            if cfg.flow.name == "FlowMap":
                self.method = FlowMap(self.policy, self.time_embedder, cfg.flow.pred_mode)
        else:
            assert cfg.flow.name == "FlowMatch"
            self.method = FlowMatch(self.policy, self.time_embedder, cfg.flow.pred_mode)

        self.solver = EulerSolver(cfg.flow.sample_step)

    def forward(self, data):
        sample_num = data["pregrasp_kp"].shape[1]
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        cond_dict = {"global": g_cond.unsqueeze(1), "local": l_cond}

        gt_rtj = torch.cat(
            [
                data["pregrasp_qpos"].reshape(-1, self.n_kp, self.in_channels//3),
                data["grasp_qpos"].reshape(-1, self.n_kp, self.in_channels//3),
                data["squeeze_qpos"].reshape(-1, self.n_kp, self.in_channels//3),
            ],
            dim=-1,
        )
        if self.RMS:
            gt_rtj = self.RMS(gt_rtj)
        loss = self.method.calculate_loss(gt_rtj, cond_dict)
        return loss

    def sample(self, data, sample_num, topk_num):
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        cond_dict = {"global": g_cond.unsqueeze(1), "local": l_cond}

        batch_size = global_feature.shape[0]

        # Init Noise
        x_init = torch.randn(
            (batch_size * sample_num, self.n_kp, self.in_channels),
            device=global_feature.device,
        )

        # Solve ODE
        x_final, log_prob = self.solver.sample(self.method, x_init, cond_dict)

        if self.RMS:
            x_final = self.RMS.inv(x_final)

        # Reshape & Select TopK
        x_reshaped = rearrange(x_final.squeeze(1), "(b n) c -> b n c", b=batch_size)
        log_prob = rearrange(log_prob, "(b n) -> b n", b=batch_size)  # (B, N)

        pose_chunks = [x_reshaped[..., 0:34], x_reshaped[..., 34:68], x_reshaped[..., 68:102]]
        pose_quat_chunks = []
        for pose in pose_chunks:
            trans = pose[..., 0:3]
            rot_mat = pose[..., 3:12].reshape(*pose.shape[:-1], 3, 3)
            rot_mat = proper_svd(rot_mat.reshape(-1, 3, 3)).reshape_as(rot_mat)
            quat = matrix_to_quaternion(rot_mat)
            joints = pose[..., 12:34]
            pose_quat_chunks.append(torch.cat([trans, quat, joints], dim=-1))

        robot_pose = torch.stack(pose_quat_chunks, dim=2)

        return robot_pose, log_prob, data["scene_path"]


class RTJGrasp34TokenModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = WrappedMinkUNet_coord(**cfg.backbone)
        self.state_dim = 3 + 9 + 22
        self.n_stage = 3
        self.policy = self._build_policy(cfg)

        use_flowmap = cfg.flow.name == "FlowMap"
        self.time_embedder = TimestepEmbedder(
            hidden_size=cfg.policy.hidden_size, flowmap=use_flowmap
        )

        self.RMS = Normalization((self.state_dim, self.n_stage))

        if use_flowmap:
            if cfg.flow.name == "FlowMap":
                self.method = FlowMap(self.policy, self.time_embedder, cfg.flow.pred_mode)
        else:
            assert cfg.flow.name == "FlowMatch"
            self.method = FlowMatch(self.policy, self.time_embedder, cfg.flow.pred_mode)

        self.solver = EulerSolver(cfg.flow.sample_step)

    def _build_policy(self, cfg):
        return DiT(
            input_size=self.state_dim, in_channels=self.n_stage, **cfg.policy
        )

    def _build_context(self, data, sample_num):
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        return {"global": g_cond.unsqueeze(1), "local": l_cond}, global_feature

    def _pack_rtj_sequence(self, data):
        return torch.stack(
            [
                data["pregrasp_qpos"].reshape(-1, self.state_dim),
                data["grasp_qpos"].reshape(-1, self.state_dim),
                data["squeeze_qpos"].reshape(-1, self.state_dim),
            ],
            dim=-1,
        )

    def _sequence_to_robot_pose(self, x_reshaped):
        pose_quat_chunks = []
        for stage_idx in range(self.n_stage):
            pose = x_reshaped[..., stage_idx]
            trans = pose[..., 0:3]
            rot_mat = pose[..., 3:12].reshape(*pose.shape[:-1], 3, 3)
            rot_mat = proper_svd(rot_mat.reshape(-1, 3, 3)).reshape_as(rot_mat)
            quat = matrix_to_quaternion(rot_mat)
            joints = pose[..., 12:34]
            pose_quat_chunks.append(torch.cat([trans, quat, joints], dim=-1))
        return torch.stack(pose_quat_chunks, dim=2)

    def forward(self, data):
        sample_num = data["pregrasp_qpos"].shape[1]
        cond_dict, _ = self._build_context(data, sample_num)

        gt_rtj = self._pack_rtj_sequence(data)
        if self.RMS:
            gt_rtj = self.RMS(gt_rtj)
        loss = self.method.calculate_loss(gt_rtj, cond_dict)
        return loss

    def sample(self, data, sample_num, topk_num):
        cond_dict, global_feature = self._build_context(data, sample_num)

        batch_size = global_feature.shape[0]

        x_init = torch.randn(
            (batch_size * sample_num, self.state_dim, self.n_stage),
            device=global_feature.device,
            dtype=global_feature.dtype,
        )

        x_final, log_prob = self.solver.sample(self.method, x_init, cond_dict)

        if self.RMS:
            x_final = self.RMS.inv(x_final)

        x_reshaped = rearrange(
            x_final, "(b n) d s -> b n d s", b=batch_size, n=sample_num
        )
        log_prob = rearrange(log_prob, "(b n) -> b n", b=batch_size)

        robot_pose = self._sequence_to_robot_pose(x_reshaped)
        return robot_pose, log_prob, data["scene_path"]


class SO3ConditionalFlowMatcher:
    def __init__(self):
        # geomstats changes Torch's global default dtype on its first import.
        previous_dtype = torch.get_default_dtype()
        try:
            from geomstats.geometry.special_orthogonal import SpecialOrthogonal
            self.vec_manifold = SpecialOrthogonal(n=3, point_type="vector")
        finally:
            torch.set_default_dtype(previous_dtype)
        self.sigma = None

    @staticmethod
    def rotmat_to_rotvec(matrix):
        return matrix_to_axis_angle(matrix)

    def vec_log_map(self, x0, x1):
        rot_x0 = self.rotmat_to_rotvec(x0)
        rot_x1 = self.rotmat_to_rotvec(x1)

        torch.set_default_dtype(torch.float64)
        log_x1 = self.vec_manifold.log_not_from_identity(rot_x1, rot_x0)
        torch.set_default_dtype(torch.float32)
        return log_x1, rot_x0

    def sample_xt(self, x0, x1, t):
        log_x1, rot_x0 = self.vec_log_map(x0.double(), x1.double())
        torch.set_default_dtype(torch.float64)
        xt = self.vec_manifold.exp_not_from_identity(t.reshape(-1, 1) * log_x1, rot_x0)
        xt = self.vec_manifold.matrix_from_rotation_vector(xt)
        torch.set_default_dtype(torch.float32)
        return xt

    def compute_conditional_flow_simple(self, t, xt):
        xt = rearrange(xt, "b c d -> b (c d)", c=3, d=3)

        def index_time_der(i):
            return torch.autograd.grad(xt, t, i, create_graph=True, retain_graph=True)[0]

        xt_dot = vmap(index_time_der, in_dims=1)(
            torch.eye(9, device=xt.device, dtype=xt.dtype).repeat(xt.shape[0], 1, 1)
        )
        return rearrange(xt_dot, "(c d) b -> b c d", c=3, d=3)

    def sample_location_and_conditional_flow_simple(self, x0, x1, t=None):
        device = x0.device
        x_0 = x0.cpu()
        x_1 = x1.cpu()
        if t is None:
            t = torch.rand(x_0.shape[0]).type_as(x_0).to(x_0.device)
        else:
            t = t.to(x_0.device).type_as(x_0)
        with torch.enable_grad():
            t.requires_grad_(True)
            xt = self.sample_xt(x_0, x_1, t)
            ut = self.compute_conditional_flow_simple(t, xt)
        xt = xt.to(device).float()
        ut = ut.to(device).float()
        t = t.to(device).float()
        return t, xt, ut


class SE3Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.backbone = WrappedMinkUNet_coord(**cfg.backbone)
        self.rt_sample_steps = int(getattr(cfg.flow, 'rt_sample_steps', 20))
        self.joint_sample_steps = int(getattr(cfg.flow, 'joint_sample_steps', 20))
        self.n_kp = 1
        self.stage_dim = 3 + 9
        self.n_stage = 3
        self.in_channels = self.stage_dim * self.n_stage
        self.joint_dim = 22
        self.joint_channels = self.joint_dim * self.n_stage
        self.hidden_size = cfg.policy.hidden_size

        self.policy = DiT(
            input_size=self.n_kp, in_channels=self.in_channels, **cfg.policy
        )
        self.joint_policy = DiT(
            input_size=self.n_kp, in_channels=self.joint_channels, **cfg.policy
        )

        self.time_embedder = TimestepEmbedder(
            hidden_size=cfg.policy.hidden_size, flowmap=False
        )
        self.joint_time_embedder = TimestepEmbedder(
            hidden_size=cfg.policy.hidden_size, flowmap=False
        )
        self.so3_cfm = SO3ConditionalFlowMatcher()
        self.pos_loss = nn.MSELoss()
        self.p_base_dist = torch.distributions.normal.Normal(
            torch.tensor(0.0), torch.tensor(1.0)
        )

        # keep SO(3) coordinates on manifold; do not normalize RT states.
        self.RMS = None

        # fuse global cond (H) + full-frame RT(n_frame*12) -> H
        self.rt_global_proj = nn.Sequential(
            nn.Linear(self.hidden_size + self.in_channels, self.hidden_size * 2),
            nn.SiLU(),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
        )

        assert (
            cfg.flow.name == "SE3FlowMatch" or cfg.flow.name == "FlowMatch"
        ), "SE3Model only supports SE3FlowMatch/FlowMatch configs."

        self.joint_method = FlowMatch(
            self.joint_policy, self.joint_time_embedder, cfg.flow.pred_mode
        )

    @staticmethod
    def _project_to_so3(rot):
        U, _, Vh = torch.linalg.svd(rot)
        R = U @ Vh
        det = torch.det(R)
        neg = det < 0
        if neg.any():
            U_fix = U.clone()
            U_fix[neg, :, -1] *= -1
            R = U_fix @ Vh
        return R

    def _build_joint_context(self, cond_dict, rt_state):
        # rt_state: (BN, 1, n_frame*12), keep BN unchanged.
        rt_flat = rt_state.squeeze(1)
        g_base = cond_dict["global"].squeeze(1)
        g_joint = self.rt_global_proj(torch.cat([g_base, rt_flat], dim=-1)).unsqueeze(1)
        return {"global": g_joint, "local": cond_dict["local"]}

    def _split_rt_state(self, x_rt):
        # x_rt: (BN, 1, 36)
        bn = x_rt.shape[0]
        z = x_rt.squeeze(1).reshape(bn, self.n_stage, self.stage_dim)
        p = z[..., :3]
        r = z[..., 3:].reshape(bn, self.n_stage, 3, 3)
        return p, r

    def _split_rt_velocity(self, v_rt):
        # v_rt: (BN, 1, 36) -> p velocity (BN,9), rot velocity matrices (BN,S,3,3)
        p_v, r_v = self._split_rt_state(v_rt)
        return p_v.reshape(v_rt.shape[0], self.n_stage * 3), r_v

    def _merge_rt_state(self, p, r):
        # p: (BN,S,3), r: (BN,S,3,3) -> (BN,1,36)
        bn = p.shape[0]
        return torch.cat([p, r.reshape(bn, self.n_stage, 9)], dim=-1).reshape(
            bn, 1, self.in_channels
        )

    def _predict_rt_velocity(self, p_t, r_t, t, cond_dict):
        # p_t: (BN,9), r_t: (BN,S,3,3), t: (BN,)
        x_state = self._merge_rt_state(p_t.view(-1, self.n_stage, 3), r_t)
        t_emb = self.time_embedder(t)
        v_pred = self.policy(x=x_state, t_emb=t_emb, context=cond_dict)
        v_pos_pred, v_rot_pred = self._split_rt_velocity(v_pred)
        v_rot_hat = tangent_space_proj(
            r_t.reshape(-1, 3, 3), v_rot_pred.reshape(-1, 3, 3)
        ).view_as(v_rot_pred)
        return v_pos_pred, v_rot_hat

    def _get_initial_rotation_log_prob(self, r_t_initial: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            r_t_initial.shape[0],
            device=r_t_initial.device,
            dtype=r_t_initial.dtype,
        )

    def _get_p_pos_pred_velocity_func(self, r_t_detached, cond_dict, current_time):
        def func(p_t_input):
            p_pred, _ = self._predict_rt_velocity(
                p_t_input,
                r_t_detached,
                current_time,
                cond_dict,
            )
            return p_pred

        return func

    def _get_r_rot_hat_manifold_vec_func(self, p_t_detached, cond_dict, current_time):
        def func(r_t_flat_input):
            r_t_input = r_t_flat_input.view(-1, self.n_stage, 3, 3)
            _, r_rot_hat = self._predict_rt_velocity(
                p_t_detached,
                r_t_input,
                current_time,
                cond_dict,
            )
            return r_rot_hat.view(r_rot_hat.shape[0], -1)

        return func

    def _sample_rt_with_logprob(self, x_init, cond_dict, steps=10):
        # Strict SE3CondFlowMatch2-style Euler update and split trace estimation.
        bn = x_init.shape[0]
        dt = 1.0 / steps

        p_t, r_t = self._split_rt_state(x_init)
        p_t = p_t.reshape(bn, self.n_stage * 3)

        log_p_p_init = self.p_base_dist.log_prob(p_t).sum(dim=-1)
        log_p_r_init = self._get_initial_rotation_log_prob(r_t)
        log_p_total = log_p_p_init + log_p_r_init

        time_grid = torch.linspace(0, 1, steps+1, device=x_init.device, dtype=x_init.dtype)[:-1]

        for t_val in time_grid:
            current_time = torch.full((bn,), t_val.item(), device=x_init.device, dtype=x_init.dtype)

            r_t_flat = r_t.reshape(bn, -1).contiguous().requires_grad_(True)
            p_t_with_grad = p_t.contiguous().requires_grad_(True)

            r_t_for_v = r_t_flat.reshape(bn, self.n_stage, 3, 3)
            p_pos_pred, r_rot_hat = self._predict_rt_velocity(
                p_t_with_grad,
                r_t_for_v,
                current_time,
                cond_dict,
            )

            eps_p = torch.randn_like(p_t_with_grad)
            p_vel_func = self._get_p_pos_pred_velocity_func(
                r_t_for_v.detach(), cond_dict, current_time
            )
            _, p_pos_pred_jvp = torch.autograd.functional.jvp(
                p_vel_func, p_t_with_grad, eps_p, create_graph=True
            )
            trace_p_euclidean = (eps_p * p_pos_pred_jvp).sum(dim=-1)

            eps_r = torch.randn_like(r_t_flat)
            r_vel_func = self._get_r_rot_hat_manifold_vec_func(
                p_t_with_grad.detach(), cond_dict, current_time
            )
            _, r_rot_hat_jvp = torch.autograd.functional.jvp(
                r_vel_func, r_t_flat, eps_r, create_graph=True
            )
            trace_r_manifold = (eps_r * r_rot_hat_jvp).sum(dim=-1)

            log_p_total = log_p_total - (trace_r_manifold * dt + trace_p_euclidean * dt)

            r_t = r_t.detach()
            p_t = p_t_with_grad.detach()

            r_t = expmap(r_t, r_rot_hat * dt)
            p_t = p_t + p_pos_pred * dt

        x_final = self._merge_rt_state(p_t.reshape(bn, self.n_stage, 3), r_t)
        return x_final, log_p_total

    def _sample_joint_without_density(self, x_init, cond_dict, steps=10):
        # Euler ODE solve for joint flow without density.
        x = x_init
        dt = 1.0 / steps
        for step in range(steps):
            t_curr = step * dt
            v = self.joint_method.predict_derivative(x, t_curr, dt, cond_dict)
            x = (x + dt * v).detach()
        return x

    def forward(self, data):
        sample_num = data["pregrasp_kp"].shape[1]
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        cond_dict = {"global": g_cond.unsqueeze(1), "local": l_cond}

        gt_se3 = torch.cat(
            [
                data["pregrasp_qpos"][..., : self.stage_dim].reshape(-1, self.n_kp, self.stage_dim),
                data["grasp_qpos"][..., : self.stage_dim].reshape(-1, self.n_kp, self.stage_dim),
                data["squeeze_qpos"][..., : self.stage_dim].reshape(-1, self.n_kp, self.stage_dim),
            ],
            dim=-1,
        )

        # flow-1: strict SE3CondFlowMatch2-style RT loss
        bsz = gt_se3.shape[0]
        r_1 = gt_se3.squeeze(1).view(bsz, self.n_stage, self.stage_dim)[..., 3:].view(
            bsz, self.n_stage, 3, 3
        )
        p_1 = gt_se3.squeeze(1).view(bsz, self.n_stage, self.stage_dim)[..., :3].reshape(
            bsz, self.n_stage * 3
        )

        r_0 = torch.tensor(
            Rotation.random(bsz * self.n_stage).as_matrix(),
            dtype=gt_se3.dtype,
            device=gt_se3.device,
        ).view(bsz, self.n_stage, 3, 3)
        p_0 = torch.randn_like(p_1)

        t_shared = torch.rand(bsz, device=gt_se3.device, dtype=gt_se3.dtype)
        t_rep = t_shared.repeat_interleave(self.n_stage)
        _, r_t_flat, u_rot_flat = self.so3_cfm.sample_location_and_conditional_flow_simple(
            r_0.view(-1, 3, 3),
            r_1.view(-1, 3, 3),
            t=t_rep,
        )
        r_t = r_t_flat.view(bsz, self.n_stage, 3, 3)
        u_rot = u_rot_flat.view(bsz, self.n_stage, 3, 3)

        t_ = t_shared.view(-1, 1)
        p_t = ((1 - t_) * p_0 + t_ * p_1).float()
        u_pos = (p_1 - p_0).float()

        v_pos_pred, v_rot_hat = self._predict_rt_velocity(p_t, r_t, t_shared, cond_dict)

        res_rot = v_rot_hat - u_rot
        rot_norms = norm_SO3(r_t.view(-1, 3, 3), res_rot.view(-1, 3, 3))
        loss_rot = torch.mean(rot_norms)
        loss_pos = self.pos_loss(v_pos_pred, u_pos)
        loss_rt = loss_rot + loss_pos

        # flow-2: joint on Euclidean space, conditioned by (global + gt RT)
        joint_context = self._build_joint_context(cond_dict, gt_se3)
        gt_joint = torch.cat(
            [
                data["pregrasp_qpos"][..., 12:34].reshape(-1, self.n_kp, self.joint_dim),
                data["grasp_qpos"][..., 12:34].reshape(-1, self.n_kp, self.joint_dim),
                data["squeeze_qpos"][..., 12:34].reshape(-1, self.n_kp, self.joint_dim),
            ],
            dim=-1,
        )
        loss_joint = self.joint_method.calculate_loss(gt_joint, joint_context)

        return {"loss_fm": loss_rt, "loss_joint": loss_joint["loss_fm"]}

    def sample(self, data, sample_num, topk_num):
        global_feature, local_feature = self.backbone(data)
        g_cond = repeat(global_feature, "b c -> (b n) c", n=sample_num)
        l_cond = repeat(local_feature, "b ... -> (b n) ...", n=sample_num)
        cond_dict = {"global": g_cond.unsqueeze(1), "local": l_cond}

        batch_size = global_feature.shape[0]
        bn = batch_size * sample_num
        device = global_feature.device
        dtype = global_feature.dtype

        p_init = torch.randn((bn, self.n_stage, 3), device=device, dtype=dtype)
        r_init = random_rotations(
            bn * self.n_stage, device=device, dtype=dtype
        ).view(bn, self.n_stage, 3, 3)
        x_init = torch.cat([p_init, r_init.view(bn, self.n_stage, 9)], dim=-1).reshape(
            bn, self.n_kp, self.in_channels
        )

        # flow-1: sample RT with density (custom Euler, fixed 10 steps)
        x_final, log_prob = self._sample_rt_with_logprob(x_init, cond_dict, steps=self.rt_sample_steps)

        x_reshaped = rearrange(x_final.squeeze(1), "(b n) c -> b n c", b=batch_size)
        log_prob = rearrange(log_prob, "(b n) -> b n", b=batch_size)

        # flow-2: sample joints without density, conditioned on sampled RT
        joint_context = self._build_joint_context(cond_dict, x_final)
        joint_init = torch.randn(
            (bn, self.n_kp, self.joint_channels), device=device, dtype=dtype
        )
        joint_final = self._sample_joint_without_density(joint_init, joint_context, steps=self.joint_sample_steps) # computing density does not help here.
        joint_reshaped = rearrange(
            joint_final.squeeze(1),
            "(b n) (s j) -> b n s j",
            b=batch_size,
            n=sample_num,
            s=self.n_stage,
            j=self.joint_dim,
        )

        pose_chunks = [
            x_reshaped[..., 0:12],
            x_reshaped[..., 12:24],
            x_reshaped[..., 24:36],
        ]
        pose_quat_chunks = []
        for stage_idx, pose in enumerate(pose_chunks):
            trans = pose[..., 0:3]
            rot_mat = pose[..., 3:12].reshape(*pose.shape[:-1], 3, 3)
            rot_mat = self._project_to_so3(rot_mat.reshape(-1, 3, 3)).reshape(*pose.shape[:-1], 3, 3)
            quat = matrix_to_quaternion(rot_mat)
            joints = joint_reshaped[..., stage_idx, :]
            pose_quat_chunks.append(torch.cat([trans, quat, joints], dim=-1))

        robot_pose = torch.stack(pose_quat_chunks, dim=2)
        return robot_pose, log_prob, data["scene_path"]


MODELS = {
    "KeyPointGraspModel": KeyPointGraspModel,
    "RTJGraspModel": RTJGraspModel,
    "RTJGrasp34TokenModel": RTJGrasp34TokenModel,
    "SE3Model": SE3Model,
}


def create_model(cfg):
    if cfg.backbone.name != "WrappedMinkUNet_coord" or cfg.policy.name != "DiT":
        raise ValueError("This release retains the MinkUNet + DiT architecture only.")
    try:
        model_cls = MODELS[cfg.name]
    except KeyError as exc:
        raise ValueError(f"Unsupported model: {cfg.name}") from exc
    return model_cls(cfg)
