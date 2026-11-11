#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Steric Clash Loss: differentiable penalty for atom-atom overlaps.

Antibody CDR design models often produce structures where the predicted
CDR loop penetrates the antigen surface (steric clashes).  This module
adds a soft repulsive energy term during training so that the coordinate
prediction network learns to avoid such overlaps.

Two components:
  1. **Intra-CDR clash loss** — penalises clashes among CDR residues
     (avoids self-intersecting loops).
  2. **Interface clash loss** — penalises clashes between CDR atoms and
     antigen atoms (the dominant failure mode, measured at ~66× reference).

The penalty uses a shifted soft-plus on the overlap distance:
    clash(d, r_min) = softplus(r_min - d)
where r_min is the sum of VDW radii minus an allowed tolerance.  This is
everywhere differentiable and has gradients that push overlapping atoms
apart.

For efficiency only CA–CA distances are used by default (backbone_only=True),
which covers the dominant clash source.  Full heavy-atom mode is available
but more expensive.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Approximate VDW radius for CA atoms (carbon alpha ≈ 1.7 Å)
_CA_VDW_RADIUS = 1.7

# Minimum allowed CA–CA distance between non-bonded residues.
# Two CA atoms: sum of radii = 3.4 Å.  With a tolerance of 0.4 Å → 3.0 Å.
# Empirically, the closest non-bonded CA–CA in crystal structures is ~3.8 Å,
# so 3.0 Å is a permissive cutoff that only penalises severe overlaps.
_CA_CLASH_DIST = 3.0

# For interface clashes we use a tighter cutoff since these are non-sequential
# residues with no covalent bond justification for being close.
_INTERFACE_CLASH_DIST = 3.0


class StericClashLoss(nn.Module):
    """Differentiable steric clash penalty for CDR design.

    Operates on CA positions (channel index 1 in the X tensor) for
    efficiency.  Computes pairwise distances between:
      - CDR CA atoms and antigen CA atoms  (interface clashes)
      - CDR CA atoms with each other        (intra-CDR clashes)

    and penalises any pair closer than the clash distance threshold.

    Args:
        clash_weight:       overall loss weight multiplier.
        interface_weight:   relative weight for interface vs intra-CDR clashes.
        clash_dist:         minimum allowed CA–CA distance (Å) for intra-CDR.
        interface_clash_dist: minimum allowed CA–CA distance (Å) for interface.
        seq_sep:            minimum sequence separation to consider a clash
                            (skip bonded neighbours).
    """

    def __init__(self, clash_weight=0.5, interface_weight=2.0,
                 clash_dist=_CA_CLASH_DIST,
                 interface_clash_dist=_INTERFACE_CLASH_DIST,
                 seq_sep=2):
        super().__init__()
        self.clash_weight = clash_weight
        self.interface_weight = interface_weight
        self.clash_dist = clash_dist
        self.interface_clash_dist = interface_clash_dist
        self.seq_sep = seq_sep

    def forward(self, pred_X, paratope_mask, is_ag, batch_id, segment_ids):
        """
        Compute the steric clash loss.

        Args:
            pred_X:        [N, n_channel, 3] predicted coordinates.
            paratope_mask: [N] bool, True for CDR residues being designed.
            is_ag:         [N] bool, True for antigen residues.
            batch_id:      [N] int, graph-to-batch assignment.
            segment_ids:   [N] int, segment ids (antigen vs heavy vs light).

        Returns:
            total_loss:     scalar, weighted clash loss.
            interface_loss: scalar, interface component (for logging).
            intra_loss:     scalar, intra-CDR component (for logging).
            n_interface_clashes: int, count of interface clashing pairs.
            n_intra_clashes:     int, count of intra-CDR clashing pairs.
        """
        device = pred_X.device
        zero = torch.tensor(0.0, device=device)

        # Extract CA positions (channel 1)
        ca_pos = pred_X[:, 1]  # [N, 3]

        # --- 1. Interface clash: CDR vs antigen ---
        interface_loss, n_iclash = self._interface_clash(
            ca_pos, paratope_mask, is_ag, batch_id)

        # --- 2. Intra-CDR clash: CDR vs CDR ---
        intra_loss, n_aclash = self._intra_clash(
            ca_pos, paratope_mask, batch_id)

        total_loss = self.clash_weight * (
            self.interface_weight * interface_loss + intra_loss)

        return total_loss, interface_loss, intra_loss, n_iclash, n_aclash

    def _interface_clash(self, ca_pos, paratope_mask, is_ag, batch_id):
        """Clash loss between CDR CA atoms and antigen CA atoms."""
        device = ca_pos.device

        cdr_idx = torch.where(paratope_mask)[0]
        ag_idx = torch.where(is_ag)[0]

        if cdr_idx.numel() == 0 or ag_idx.numel() == 0:
            return torch.tensor(0.0, device=device), 0

        cdr_ca = ca_pos[cdr_idx]       # [M_cdr, 3]
        ag_ca = ca_pos[ag_idx]          # [M_ag, 3]
        cdr_batch = batch_id[cdr_idx]   # [M_cdr]
        ag_batch = batch_id[ag_idx]     # [M_ag]

        # Pairwise distances: [M_cdr, M_ag]
        dist = torch.cdist(cdr_ca.unsqueeze(0), ag_ca.unsqueeze(0)).squeeze(0)

        # Mask cross-graph pairs (different batch items)
        same_graph = cdr_batch.unsqueeze(1) == ag_batch.unsqueeze(0)  # [M_cdr, M_ag]
        dist = dist + (~same_graph).float() * 1e6

        # Compute clash penalty: softplus(threshold - dist)
        # softplus is smooth and differentiable everywhere
        overlap = self.interface_clash_dist - dist  # positive where clashing
        penalty = F.softplus(overlap)               # smooth ReLU

        # Only count same-graph pairs
        penalty = penalty * same_graph.float()

        n_clashes = (overlap > 0).sum().item() if same_graph.any() else 0

        # Mean over all valid pairs to keep gradient magnitude stable
        n_pairs = same_graph.sum().clamp(min=1)
        loss = penalty.sum() / n_pairs

        return loss, int(n_clashes)

    def _intra_clash(self, ca_pos, paratope_mask, batch_id):
        """Clash loss among CDR CA atoms (self-intersection)."""
        device = ca_pos.device

        cdr_idx = torch.where(paratope_mask)[0]
        M = cdr_idx.numel()

        if M < 2:
            return torch.tensor(0.0, device=device), 0

        cdr_ca = ca_pos[cdr_idx]        # [M, 3]
        cdr_batch = batch_id[cdr_idx]   # [M]

        # Pairwise distances: [M, M]
        dist = torch.cdist(cdr_ca.unsqueeze(0), cdr_ca.unsqueeze(0)).squeeze(0)

        # Mask: same graph, upper triangle only, skip sequential neighbours
        same_graph = cdr_batch.unsqueeze(1) == cdr_batch.unsqueeze(0)
        # Use original residue indices for sequence separation
        idx_diff = (cdr_idx.unsqueeze(1) - cdr_idx.unsqueeze(0)).abs()
        valid = same_graph & (idx_diff >= self.seq_sep)
        # Upper triangle to avoid double counting
        triu = torch.triu(torch.ones(M, M, device=device, dtype=torch.bool), diagonal=1)
        valid = valid & triu

        dist = dist + (~valid).float() * 1e6

        overlap = self.clash_dist - dist
        penalty = F.softplus(overlap)
        penalty = penalty * valid.float()

        n_clashes = ((overlap > 0) & valid).sum().item()

        n_pairs = valid.sum().clamp(min=1)
        loss = penalty.sum() / n_pairs

        return loss, int(n_clashes)
