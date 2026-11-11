#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Riemannian Flow Matching on SE(3)^N = SO(3)^N × R^{3N}.

Represents each CDR residue as a rigid frame (R_i, t_i) where R_i is a
rotation in SO(3) and t_i is a translation in R^3 (the CA position).

Flow matching:
  - Translations follow Euclidean OT (conditional) paths:
      t(s) = (1 - s) * t_0 + s * t_1
      velocity_t = t_1 - t_0
  - Rotations follow geodesics on SO(3):
      R(s) = R_0 * exp(s * log(R_0^T R_1))
      velocity_R = log(R_0^T R_1)   (tangent vector in body frame)

The network predicts the velocity field (omega, v) at each time s, and the
loss is:
  L = ||omega_pred - omega_target||^2 + ||v_pred - v_target||^2

During generation, we integrate the learned velocity field from s=0 to s=1
using Euler or midpoint steps.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from .so3_utils import (
    so3_exp, so3_log, so3_geodesic, so3_tangent_vector,
    random_so3, so3_noise, hat
)


class FrameFlowMatchingLoss(nn.Module):
    """Computes the Riemannian flow matching loss for SE(3) frames.

    Training procedure:
      1. Sample time s ~ logit-normal (concentrated around s=0.5)
      2. Compute noisy frames by interpolating between noise (s=0) and
         ground truth (s=1) along geodesics (rotations) and linear paths
         (translations)
      3. Network predicts velocity (omega_pred, v_pred) at the noisy frame
      4. Loss = importance-weighted SmoothL1 between predicted and target velocity

    The logit-normal schedule concentrates samples in s ∈ [0.3, 0.7] where
    the signal-to-noise ratio is most informative. Importance weights w(s) =
    1/p(s) ensure rare timesteps (near 0 and 1) are properly upweighted.
    """

    def __init__(self, rot_weight=1.0, trans_weight=1.0, sigma_min=1e-4):
        super().__init__()
        self.rot_weight = rot_weight
        self.trans_weight = trans_weight
        self.sigma_min = sigma_min

    def forward(self, omega_pred, v_pred, R_target, t_target,
                R_noise, t_noise, s, mask=None, importance_weights=None):
        """
        Args:
            omega_pred: [N, 3] predicted rotation velocity (body-frame)
            v_pred:     [N, 3] predicted translation velocity
            R_target:   [N, 3, 3] ground-truth rotation matrices
            t_target:   [N, 3] ground-truth translations (CA positions)
            R_noise:    [N, 3, 3] noise rotation matrices (at s=0)
            t_noise:    [N, 3] noise translations (at s=0)
            s:          [N] or scalar, interpolation time
            mask:       [N] bool, True for CDR residues to supervise
            importance_weights: [N] importance sampling weights w(s) = 1/p(s)
        Returns:
            loss: scalar
            loss_rot: scalar
            loss_trans: scalar
        """
        # Target rotation velocity: tangent vector from current to target
        # At flow time s the rotation velocity that would reach R_target
        # in the remaining (1-s) time is: log(R_s^T R_target) / (1 - s)
        R_s = so3_geodesic(R_noise, R_target, s)
        omega_target = so3_tangent_vector(R_s, R_target)
        s_clamp = s.clamp(min=self.sigma_min)
        omega_target = omega_target / (1.0 - s_clamp).unsqueeze(-1).clamp(min=1e-3)

        # Target translation velocity: v = (t_target - t_s) / (1 - s)
        t_s = (1 - s.unsqueeze(-1)) * t_noise + s.unsqueeze(-1) * t_target
        v_target = (t_target - t_s) / (1.0 - s_clamp).unsqueeze(-1).clamp(min=1e-3)

        if mask is not None:
            omega_pred = omega_pred[mask]
            v_pred = v_pred[mask]
            omega_target = omega_target[mask]
            v_target = v_target[mask]
            if importance_weights is not None:
                importance_weights = importance_weights[mask]

        if importance_weights is not None:
            # Per-residue weighted loss: w(s) * ||pred - target||^2
            # Using element-wise SmoothL1 then weighting per residue
            w = importance_weights.unsqueeze(-1)  # [M, 1]
            loss_rot = (w * F.smooth_l1_loss(
                omega_pred, omega_target, reduction='none')).mean()
            loss_trans = (w * F.smooth_l1_loss(
                v_pred, v_target, reduction='none')).mean()
        else:
            loss_rot = F.smooth_l1_loss(omega_pred, omega_target)
            loss_trans = F.smooth_l1_loss(v_pred, v_target)

        loss = self.rot_weight * loss_rot + self.trans_weight * loss_trans
        return loss, loss_rot, loss_trans


def _logit_normal_sample(shape, mu=0.0, sigma=1.0, device=None, dtype=None):
    """Sample from a logit-normal distribution concentrated around 0.5.

    s = sigmoid(mu + sigma * z),  z ~ N(0, 1)

    The logit-normal with mu=0, sigma=1 concentrates mass in [0.3, 0.7],
    reducing wasted capacity at extreme timesteps where uniform sampling
    spends equal budget but the loss landscape is degenerate.

    Args:
        shape: output shape
        mu: location parameter (0 centers at s=0.5)
        sigma: scale parameter (1.0 gives moderate concentration)
        device, dtype: torch specs
    Returns:
        s: tensor of shape `shape` in (0, 1)
    """
    z = torch.randn(shape, device=device, dtype=dtype)
    return torch.sigmoid(mu + sigma * z)


def _logit_normal_log_prob(s, mu=0.0, sigma=1.0):
    """Log probability of the logit-normal distribution.

    p(s) = (1 / (sigma * sqrt(2*pi))) * (1 / (s * (1-s)))
           * exp(-(logit(s) - mu)^2 / (2*sigma^2))

    Args:
        s: [N] samples in (0, 1)
        mu, sigma: distribution parameters
    Returns:
        log_p: [N] log probabilities
    """
    logit_s = torch.log(s / (1.0 - s))  # logit(s)
    log_p = (-0.5 * ((logit_s - mu) / sigma) ** 2
             - torch.log(s * (1.0 - s))
             - torch.log(torch.tensor(sigma, device=s.device, dtype=s.dtype))
             - 0.5 * torch.log(torch.tensor(2.0 * 3.141592653589793,
                                             device=s.device, dtype=s.dtype)))
    return log_p


class FrameNoiser(nn.Module):
    """Sample noisy frames for training by interpolating between
    a starting point (s=0) and ground truth (s=1).

    Supports two modes:
      - 'noise': s=0 is random noise (original behavior)
      - 'pred':  s=0 is the EGNN prediction (refinement mode)

    Uses a logit-normal time schedule s = sigmoid(mu + sigma * z) that
    concentrates samples around s ∈ [0.3, 0.7] where the signal-to-noise
    ratio is most informative, plus importance weights w(s) = 1/p(s)
    for unbiased loss estimation.

    In 'noise' mode, noise is centered around ground truth for translations
    (local perturbation with scale trans_noise_scale) to keep the velocity
    targets at a reasonable magnitude.
    """

    def __init__(self, sigma_min=1e-4, trans_noise_scale=2.0,
                 time_mu=0.0, time_sigma=1.0):
        super().__init__()
        self.sigma_min = sigma_min
        self.trans_noise_scale = trans_noise_scale
        self.time_mu = time_mu
        self.time_sigma = time_sigma

    @torch.no_grad()
    def forward(self, R_gt, t_gt, s=None, batch_id=None,
                R_pred=None, t_pred=None):
        """
        Args:
            R_gt:     [N, 3, 3] ground-truth local frames
            t_gt:     [N, 3] ground-truth CA positions
            s:        [N] or None (sampled via logit-normal if None)
            batch_id: [N] for per-graph time sampling
            R_pred:   [N, 3, 3] EGNN-predicted frames (if None, use random noise)
            t_pred:   [N, 3] EGNN-predicted CA positions (if None, use random noise)
        Returns:
            R_s:      [N, 3, 3] interpolated rotations at time s
            t_s:      [N, 3] interpolated translations at time s
            R_start:  [N, 3, 3] start rotations (s=0 endpoint)
            t_start:  [N, 3] start translations (s=0 endpoint)
            s:        [N] time values used
            importance_weights: [N] w(s) = 1/p(s), normalized
        """
        N = R_gt.shape[0]
        device, dtype = R_gt.device, R_gt.dtype

        if s is None:
            if batch_id is not None:
                bs = batch_id.max().item() + 1
                s_per_graph = _logit_normal_sample(
                    (bs,), mu=self.time_mu, sigma=self.time_sigma,
                    device=device, dtype=dtype)
                s = s_per_graph[batch_id]
            else:
                s = _logit_normal_sample(
                    (N,), mu=self.time_mu, sigma=self.time_sigma,
                    device=device, dtype=dtype)

        s = s.clamp(self.sigma_min, 1.0 - self.sigma_min)

        # Compute importance weights: w(s) = 1 / p(s), normalized
        log_p = _logit_normal_log_prob(s, mu=self.time_mu, sigma=self.time_sigma)
        raw_weights = torch.exp(-log_p)  # 1/p(s)
        # Normalize so E[w] = 1 (self-normalized importance sampling)
        importance_weights = raw_weights / raw_weights.mean().clamp(min=1e-8)

        if R_pred is not None and t_pred is not None:
            # Refinement mode: interpolate EGNN prediction -> GT
            R_start = R_pred
            t_start = t_pred
        else:
            # Original noise mode: interpolate random noise -> GT
            R_start = random_so3(N, device, dtype)
            t_start = t_gt + torch.randn_like(t_gt) * self.trans_noise_scale

        R_s = so3_geodesic(R_start, R_gt, s)
        t_s = (1 - s.unsqueeze(-1)) * t_start + s.unsqueeze(-1) * t_gt

        return R_s, t_s, R_start, t_start, s, importance_weights


class FrameVelocityHead(nn.Module):
    """Predicts (omega, v) velocity from hidden features, current frame,
    and flow time embedding.

    Inputs:
        H:         [N, hidden_dim] node hidden states
        R_current: [N, 3, 3] current rotation frames
        t_current: [N, 3] current CA positions
        s:         [N] flow time

    Outputs:
        omega: [N, 3] rotation velocity (body frame tangent vector)
        v:     [N, 3] translation velocity
    """

    def __init__(self, hidden_dim, time_embed_dim=32, dropout=0.1):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        feat_dim = hidden_dim + time_embed_dim + 9 + 3

        self.trunk = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.omega_head = nn.Linear(hidden_dim, 3)
        self.v_head = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.omega_head.weight)
        nn.init.zeros_(self.omega_head.bias)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)

    def forward(self, H, R_current, t_current, s):
        """
        Args:
            H:         [N, hidden_dim]
            R_current: [N, 3, 3]
            t_current: [N, 3]
            s:         [N]
        Returns:
            omega: [N, 3]
            v:     [N, 3]
        """
        s_embed = self.time_mlp(s.unsqueeze(-1))
        R_flat = R_current.reshape(-1, 9)
        features = torch.cat([H, s_embed, R_flat, t_current], dim=-1)

        trunk_out = self.trunk(features)
        omega = self.omega_head(trunk_out)
        v = self.v_head(trunk_out)
        return omega, v


class VelocityMPLayer(nn.Module):
    """Lightweight message passing layer for the GNN velocity head.

    At each integration step, residues exchange information based on their
    current spatial arrangement so the velocity prediction is geometry-aware.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h, edges, t_current):
        """
        Args:
            h:         [N, hidden_dim] node features
            edges:     [2, E] edge indices
            t_current: [N, 3] current translations (for distance features)
        Returns:
            h_out: [N, hidden_dim] updated node features (with residual)
        """
        row, col = edges
        if edges.shape[1] == 0:
            return h
        dist = torch.norm(t_current[row] - t_current[col], dim=-1, keepdim=True)
        edge_feat = self.edge_mlp(torch.cat([h[row], h[col], dist], dim=-1))
        agg = scatter_mean(edge_feat, row, dim=0, dim_size=h.shape[0])
        h_out = self.node_mlp(torch.cat([h, agg], dim=-1))
        return h + h_out


class GNNVelocityHead(nn.Module):
    """Predicts (omega, v) velocity using message passing at each step.

    Replaces the MLP-only FrameVelocityHead with a lightweight GNN that
    captures spatial relationships between residues at each integration step.
    At every call (each Euler/midpoint step), kNN edges are rebuilt from the
    current translations so the message passing reflects the evolving geometry.

    Inputs:
        H:         [N, hidden_dim] node hidden states from EGNN
        R_current: [N, 3, 3] current rotation frames
        t_current: [N, 3] current CA positions
        s:         [N] flow time
        batch_id:  [N] optional graph assignment for batched training

    Outputs:
        omega: [N, 3] rotation velocity (body frame tangent vector)
        v:     [N, 3] translation velocity
    """

    def __init__(self, hidden_dim, time_embed_dim=32, n_mp_layers=2,
                 k_neighbors=9, dropout=0.1):
        super().__init__()
        self.k_neighbors = k_neighbors
        self.time_embed_dim = time_embed_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Input: H + time_embed + R_flat(9) + t(3)
        feat_dim = hidden_dim + time_embed_dim + 9 + 3
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.mp_layers = nn.ModuleList([
            VelocityMPLayer(hidden_dim, dropout=dropout)
            for _ in range(n_mp_layers)
        ])

        self.omega_head = nn.Linear(hidden_dim, 3)
        self.v_head = nn.Linear(hidden_dim, 3)

        # Zero-init outputs for stable training start
        nn.init.zeros_(self.omega_head.weight)
        nn.init.zeros_(self.omega_head.bias)
        nn.init.zeros_(self.v_head.weight)
        nn.init.zeros_(self.v_head.bias)

    def _build_knn_edges(self, t_current, batch_id=None):
        """Build kNN edges from current translations.

        Args:
            t_current: [N, 3]
            batch_id:  [N] optional, prevents cross-graph edges in batches
        Returns:
            edges: [2, N*k]
        """
        N = t_current.shape[0]
        k = min(self.k_neighbors, N - 1)
        if k <= 0:
            return torch.zeros(2, 0, dtype=torch.long, device=t_current.device)

        dist = torch.cdist(t_current.unsqueeze(0),
                           t_current.unsqueeze(0)).squeeze(0)  # [N, N]

        if batch_id is not None:
            cross_graph = batch_id.unsqueeze(0) != batch_id.unsqueeze(1)
            dist = dist + cross_graph.float() * 1e10

        # topk smallest (including self at dist=0), then remove self
        _, idx = dist.topk(k + 1, dim=-1, largest=False)  # [N, k+1]
        idx = idx[:, 1:]  # [N, k]

        row = torch.arange(N, device=t_current.device
                           ).unsqueeze(1).expand(-1, k)
        return torch.stack([row.reshape(-1), idx.reshape(-1)], dim=0)

    def forward(self, H, R_current, t_current, s, batch_id=None):
        """
        Args:
            H:         [N, hidden_dim]
            R_current: [N, 3, 3]
            t_current: [N, 3]
            s:         [N]
            batch_id:  [N] optional
        Returns:
            omega: [N, 3]
            v:     [N, 3]
        """
        s_embed = self.time_mlp(s.unsqueeze(-1))
        R_flat = R_current.reshape(-1, 9)
        features = torch.cat([H, s_embed, R_flat, t_current], dim=-1)

        h = self.input_proj(features)

        edges = self._build_knn_edges(t_current, batch_id=batch_id)

        for layer in self.mp_layers:
            h = layer(h, edges, t_current)

        omega = self.omega_head(h)
        v = self.v_head(h)
        return omega, v


class FrameFlowIntegrator(nn.Module):
    """Integrates the learned velocity field to generate frames.

    Uses Euler or midpoint integration from s=0 (noise) to s=1 (data).
    """

    def __init__(self, n_steps=20, method='euler'):
        super().__init__()
        self.n_steps = n_steps
        assert method in ('euler', 'midpoint')
        self.method = method

    @torch.no_grad()
    def forward(self, velocity_fn, R_init, t_init, **kwargs):
        """
        Args:
            velocity_fn: callable(R, t, s, **kwargs) -> (omega, v)
            R_init:      [N, 3, 3] initial noise rotations
            t_init:      [N, 3] initial noise translations
        Returns:
            R_final: [N, 3, 3] generated rotations
            t_final: [N, 3] generated translations
            trajectory: list of (R, t) at each step (optional)
        """
        dt = 1.0 / self.n_steps
        R, t = R_init, t_init
        N = R.shape[0]
        device, dtype = R.device, R.dtype

        trajectory = [(R.clone(), t.clone())]

        for step in range(self.n_steps):
            s_val = step * dt
            s = torch.full((N,), s_val, device=device, dtype=dtype)

            if self.method == 'euler':
                omega, v = velocity_fn(R, t, s, **kwargs)
                R = R @ so3_exp(dt * omega)
                t = t + dt * v

            elif self.method == 'midpoint':
                omega1, v1 = velocity_fn(R, t, s, **kwargs)
                R_mid = R @ so3_exp(0.5 * dt * omega1)
                t_mid = t + 0.5 * dt * v1
                s_mid = torch.full((N,), s_val + 0.5 * dt, device=device, dtype=dtype)

                omega2, v2 = velocity_fn(R_mid, t_mid, s_mid, **kwargs)
                R = R @ so3_exp(dt * omega2)
                t = t + dt * v2

            trajectory.append((R.clone(), t.clone()))

        return R, t, trajectory


class FrameFlowModule(nn.Module):
    """Complete Riemannian flow matching module for CDR design.

    Wraps together:
      - FrameNoiser: creates training interpolants
      - FrameVelocityHead: predicts velocity from EGNN hidden states
      - FrameFlowMatchingLoss: computes training loss
      - FrameFlowIntegrator: generates new structures at inference
    """

    def __init__(self, hidden_dim, rot_weight=1.0, trans_weight=1.0,
                 n_integration_steps=20, integration_method='euler',
                 time_embed_dim=32, dropout=0.1, sigma_min=1e-4,
                 time_mu=0.0, time_sigma=1.0):
        super().__init__()
        self.noiser = FrameNoiser(sigma_min=sigma_min,
                                  time_mu=time_mu, time_sigma=time_sigma)
        self.velocity_head = GNNVelocityHead(
            hidden_dim, time_embed_dim=time_embed_dim, dropout=dropout)
        self.loss_fn = FrameFlowMatchingLoss(
            rot_weight=rot_weight, trans_weight=trans_weight,
            sigma_min=sigma_min)
        self.integrator = FrameFlowIntegrator(
            n_steps=n_integration_steps, method=integration_method)

    def training_step(self, H, R_gt, t_gt, mask=None, batch_id=None,
                       R_pred=None, t_pred=None):
        """Compute flow matching loss during training.

        Args:
            H:        [N, hidden_dim] hidden states from EGNN
            R_gt:     [N, 3, 3] ground-truth local frames
            t_gt:     [N, 3] ground-truth CA positions
            mask:     [N] bool, CDR residues
            batch_id: [N] graph assignment
            R_pred:   [N, 3, 3] EGNN-predicted frames (if None, use random noise)
            t_pred:   [N, 3] EGNN-predicted CA positions (if None, use random noise)
        Returns:
            loss, loss_rot, loss_trans
        """
        R_s, t_s, R_start, t_start, s, importance_weights = self.noiser(
            R_gt, t_gt, batch_id=batch_id,
            R_pred=R_pred, t_pred=t_pred)

        omega_pred, v_pred = self.velocity_head(H, R_s, t_s, s, batch_id=batch_id)

        loss, loss_rot, loss_trans = self.loss_fn(
            omega_pred, v_pred, R_gt, t_gt, R_start, t_start, s,
            mask=mask, importance_weights=importance_weights)

        return loss, loss_rot, loss_trans

    @torch.no_grad()
    def sample(self, H_fn, N, device, dtype, **kwargs):
        """Generate new frames by integrating from noise.

        Args:
            H_fn:   callable(**kwargs) -> [N, hidden_dim] that produces
                    hidden states for the current geometry
            N:      number of residues
            device, dtype: torch specs
        Returns:
            R_gen: [N, 3, 3]
            t_gen: [N, 3]
        """
        R_init = random_so3(N, device, dtype)
        t_init = torch.randn(N, 3, device=device, dtype=dtype)

        def velocity_fn(R, t, s, **kw):
            H = H_fn(**kw)
            return self.velocity_head(H, R, t, s)

        R_gen, t_gen, _ = self.integrator(velocity_fn, R_init, t_init,
                                          **kwargs)
        return R_gen, t_gen


class FrameAwareLoss(nn.Module):
    """Combined loss that adds Riemannian flow matching to the existing
    PRiME losses. This wraps the flow module's training_step and adds
    frame-consistency regularization using the pairwise frame alignment
    tr(R_i^T R_j) from the equivariant adapter.
    """

    def __init__(self, hidden_dim, flow_weight=0.5, consistency_weight=0.1,
                 rot_weight=1.0, trans_weight=1.0, n_integration_steps=20,
                 sigma_min=1e-4, dropout=0.1, time_mu=0.0, time_sigma=1.0):
        super().__init__()
        self.flow_weight = flow_weight
        self.consistency_weight = consistency_weight

        self.flow_module = FrameFlowModule(
            hidden_dim=hidden_dim,
            rot_weight=rot_weight,
            trans_weight=trans_weight,
            n_integration_steps=n_integration_steps,
            sigma_min=sigma_min,
            time_mu=time_mu,
            time_sigma=time_sigma,
            dropout=dropout,
        )

    def frame_consistency_loss(self, R_pred, R_gt, ctx_edges, mask):
        """Penalize misalignment of predicted pairwise frame orientations.

        Uses tr(R_i^T R_j) as the alignment measure (same as the geometric
        attention bias in equivariant_adapter.py).

        Args:
            R_pred: [N, 3, 3] predicted frames
            R_gt:   [N, 3, 3] ground-truth frames
            ctx_edges: [2, E] edge indices
            mask:   [N] bool, CDR residues
        Returns:
            loss: scalar
        """
        row, col = ctx_edges
        edge_mask = mask[row] & mask[col]
        if edge_mask.sum() == 0:
            return torch.tensor(0.0, device=R_pred.device)

        row_m = row[edge_mask]
        col_m = col[edge_mask]

        R_rel_pred = torch.bmm(
            R_pred[row_m].transpose(-1, -2), R_pred[col_m])
        R_rel_gt = torch.bmm(
            R_gt[row_m].transpose(-1, -2), R_gt[col_m])

        trace_pred = R_rel_pred.diagonal(dim1=-2, dim2=-1).sum(-1)
        trace_gt = R_rel_gt.diagonal(dim1=-2, dim2=-1).sum(-1)

        return F.smooth_l1_loss(trace_pred / 3.0, trace_gt / 3.0)

    def forward(self, H, R_gt, t_gt, R_pred, ctx_edges,
                mask=None, batch_id=None, t_pred=None,
                R_flow_start=None, t_flow_start=None):
        """
        Args:
            H:             [N, hidden_dim] EGNN hidden states
            R_gt:          [N, 3, 3] ground-truth frames
            t_gt:          [N, 3] ground-truth CA positions
            R_pred:        [N, 3, 3] predicted frames (for consistency loss)
            ctx_edges:     [2, E] context edges
            mask:          [N] bool, CDR residues
            batch_id:      [N] graph assignment
            t_pred:        [N, 3] predicted CA positions (for refinement-mode flow)
            R_flow_start:  [N, 3, 3] optional override start rotations for flow training
            t_flow_start:  [N, 3] optional override start translations for flow training
        Returns:
            total_loss, flow_loss, consistency_loss, loss_rot, loss_trans
        """
        # Use overridden start frames for flow if provided, else fall back to R_pred/t_pred
        R_for_flow = R_flow_start if R_flow_start is not None else R_pred
        t_for_flow = t_flow_start if t_flow_start is not None else t_pred

        flow_loss, loss_rot, loss_trans = self.flow_module.training_step(
            H, R_gt, t_gt, mask=mask, batch_id=batch_id,
            R_pred=R_for_flow, t_pred=t_for_flow)

        # Consistency loss always uses actual EGNN predictions (not noised/mixed)
        consistency_loss = self.frame_consistency_loss(
            R_pred, R_gt, ctx_edges, mask if mask is not None
            else torch.ones(H.shape[0], dtype=torch.bool, device=H.device))

        total_loss = (self.flow_weight * flow_loss
                      + self.consistency_weight * consistency_loss)

        return total_loss, flow_loss, consistency_loss, loss_rot, loss_trans
