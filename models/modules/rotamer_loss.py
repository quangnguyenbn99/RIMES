#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Rotamer-Aware Joint Loss: couples sequence and structure predictions
through physics-based rotamer constraints from the Dunbrack library.

The key idea: after predicting both sequence logits and atom coordinates,
check whether the predicted sidechain geometry (chi1 dihedral angle) is
consistent with the predicted amino acid's preferred rotamers. This
explicitly couples the sequence and structure networks through a
physics-based constraint that neither loss alone captures.

Gradient flow:
  - Structure network: gradients through chi1 = dihedral(X[:,0], X[:,1], X[:,4], X[:,5])
    push predicted sidechain geometry toward preferred rotamer basins.
  - Sequence network: gradients through softmax(logits) push predicted amino acid
    probabilities toward types whose preferred rotamers match the predicted geometry.
  - Both networks share the hidden state H, so the rotamer loss jointly refines
    the shared representation for sequence-structure consistency.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Simplified Dunbrack backbone-independent rotamer library (chi1 only)
# Source: Shapovalov & Dunbrack (2011), backbone-independent rotamer library.
# Format: amino_acid_1letter -> list of (chi1_degrees, probability)
#
# Chi1 is the N-CA-CB-XG dihedral, where XG is the gamma atom.
# Three canonical basins: gauche- (g-, ~-60°), trans (t, ~180°), gauche+ (g+, ~60°)
# ---------------------------------------------------------------------------
_CHI1_ROTAMERS = {
    'G': [],   # glycine: no sidechain
    'A': [],   # alanine: only CB, no chi1
    'V': [(-60, 0.15), (175, 0.60), (65, 0.25)],
    'L': [(-65, 0.60), (180, 0.30), (65, 0.10)],
    'I': [(-60, 0.55), (60, 0.15), (180, 0.30)],
    'F': [(-65, 0.45), (180, 0.40), (65, 0.15)],
    'W': [(-65, 0.45), (180, 0.40), (65, 0.15)],
    'Y': [(-65, 0.45), (180, 0.40), (65, 0.15)],
    'D': [(-70, 0.45), (180, 0.20), (65, 0.35)],
    'H': [(-65, 0.45), (180, 0.35), (65, 0.20)],
    'N': [(-65, 0.45), (180, 0.25), (65, 0.30)],
    'E': [(-65, 0.55), (180, 0.30), (65, 0.15)],
    'K': [(-65, 0.50), (180, 0.35), (65, 0.15)],
    'Q': [(-65, 0.55), (180, 0.30), (65, 0.15)],
    'M': [(-65, 0.55), (180, 0.30), (65, 0.15)],
    'R': [(-65, 0.50), (180, 0.35), (65, 0.15)],
    'S': [(62, 0.45), (180, 0.30), (-65, 0.25)],
    'T': [(62, 0.55), (180, 0.15), (-65, 0.30)],
    'C': [(-65, 0.50), (180, 0.30), (65, 0.20)],
    'P': [(-30, 0.55), (30, 0.45)],
}

# Must match the ordering in data/pdb_utils.py AminoAcidVocab
_AA_ORDER = [
    'G', 'A', 'V', 'L', 'I', 'F', 'W', 'Y',
    'D', 'H', 'N', 'E', 'K', 'Q', 'M', 'R',
    'S', 'T', 'C', 'P',
]


def _compute_dihedral(x0, x1, x2, x3):
    """
    Compute dihedral angle defined by four atom positions.

    Uses the atan2-based formula for a signed angle in [-pi, pi],
    which is differentiable and numerically stable.

    Args:
        x0, x1, x2, x3: [N, 3] atom coordinates defining the torsion.

    Returns:
        angle: [N] dihedral angles in radians, range [-pi, pi].
    """
    eps = 1e-7
    u0 = x1 - x0   # [N, 3]
    u1 = x2 - x1   # [N, 3]
    u2 = x3 - x2   # [N, 3]

    # normals to the two planes
    n1 = torch.cross(u0, u1, dim=-1)   # [N, 3]
    n2 = torch.cross(u1, u2, dim=-1)   # [N, 3]

    n1 = F.normalize(n1, dim=-1, eps=eps)
    n2 = F.normalize(n2, dim=-1, eps=eps)
    u1_hat = F.normalize(u1, dim=-1, eps=eps)

    cos_angle = (n1 * n2).sum(-1).clamp(-1 + eps, 1 - eps)
    sin_angle = (torch.cross(n1, n2, dim=-1) * u1_hat).sum(-1)

    return torch.atan2(sin_angle, cos_angle)


class RotamerConsistencyLoss(nn.Module):
    """
    Rotamer-aware joint loss coupling sequence and structure predictions.

    For each designed position, this loss:
      1. Computes the chi1 dihedral angle from predicted atom coordinates
         using channels [0=N, 1=CA, 4=CB, 5=gamma_atom].
      2. For each of the 20 amino acid types, evaluates compatibility of
         the predicted chi1 with that type's preferred rotamers using a
         von Mises-like kernel.
      3. Weights the per-type rotamer penalties by the softmax sequence
         probabilities, creating explicit gradient coupling between the
         sequence logits and the structure coordinates.

    The loss is zero when the predicted sidechain geometry matches the
    preferred rotamers of the predicted amino acid type.
    """

    def __init__(self, num_aa_types=20, weight=0.1, sigma_deg=30.0):
        """
        Args:
            num_aa_types: number of standard amino acid types (20).
            weight: loss weight multiplier.
            sigma_deg: angular tolerance in degrees for the von Mises kernel.
                       Smaller = sharper peaks at preferred rotamers.
        """
        super().__init__()
        self.weight = weight
        self.sigma = sigma_deg * math.pi / 180.0

        # Build rotamer lookup tables as registered buffers
        max_rotamers = max(len(v) for v in _CHI1_ROTAMERS.values())
        max_rotamers = max(max_rotamers, 1)

        chi1_angles = torch.zeros(num_aa_types, max_rotamers)
        chi1_probs = torch.zeros(num_aa_types, max_rotamers)
        chi1_mask = torch.zeros(num_aa_types, max_rotamers, dtype=torch.bool)
        has_chi1 = torch.zeros(num_aa_types, dtype=torch.bool)

        for aa_idx, aa in enumerate(_AA_ORDER):
            rotamers = _CHI1_ROTAMERS[aa]
            if len(rotamers) > 0:
                has_chi1[aa_idx] = True
                for r_idx, (angle_deg, prob) in enumerate(rotamers):
                    chi1_angles[aa_idx, r_idx] = angle_deg * math.pi / 180.0
                    chi1_probs[aa_idx, r_idx] = prob
                    chi1_mask[aa_idx, r_idx] = True

        # Normalize probabilities per amino acid
        prob_sum = chi1_probs.sum(-1, keepdim=True).clamp(min=1e-8)
        chi1_probs = chi1_probs / prob_sum

        self.register_buffer('chi1_angles', chi1_angles)    # [20, max_rot]
        self.register_buffer('chi1_probs', chi1_probs)      # [20, max_rot]
        self.register_buffer('chi1_mask', chi1_mask)         # [20, max_rot]
        self.register_buffer('has_chi1', has_chi1)           # [20]

    def forward(self, pred_logits, pred_X, smask):
        """
        Compute the rotamer consistency loss.

        Args:
            pred_logits: [N, num_classes] sequence logits (final iteration).
            pred_X: [N, n_channel, 3] predicted all-atom coordinates.
            smask: [N] boolean mask for designed (CDR) positions.

        Returns:
            loss: scalar, weighted rotamer consistency loss.
        """
        if smask.sum() == 0 or self.weight == 0:
            return torch.tensor(0.0, device=pred_X.device)

        # Extract designed positions
        logits = pred_logits[smask]     # [M, num_classes]
        X = pred_X[smask]               # [M, n_channel, 3]
        num_aa = self.has_chi1.shape[0]  # 20

        # ------------------------------------------------------------------
        # 1. Compute chi1 dihedral from atom coordinate channels
        #    Channel mapping: 0=N, 1=CA, 2=C, 3=O, 4=CB, 5=gamma_atom
        #    chi1 = dihedral(N, CA, CB, X_gamma)
        # ------------------------------------------------------------------
        chi1_pred = _compute_dihedral(
            X[:, 0], X[:, 1], X[:, 4], X[:, 5]
        )  # [M]

        # ------------------------------------------------------------------
        # 2. Compute compatibility of predicted chi1 with each AA type's
        #    preferred rotamers using a von Mises-like kernel:
        #      compat(chi, rot) = exp( (cos(chi - rot) - 1) / sigma^2 )
        #    This is 1.0 when chi matches the rotamer exactly, and decays
        #    smoothly to ~0 for large angular deviations.
        # ------------------------------------------------------------------
        # chi1_pred: [M] -> [M, 1, 1]
        # chi1_angles: [num_aa, max_rot] -> [1, num_aa, max_rot]
        angular_diff = chi1_pred[:, None, None] - self.chi1_angles[None, :, :]
        # [M, num_aa, max_rot]

        k = 1.0 / (self.sigma ** 2)
        rot_compat = torch.exp(k * (torch.cos(angular_diff) - 1.0))
        # [M, num_aa, max_rot], values in (0, 1]

        # Mask out invalid (padding) rotamer entries
        rot_compat = rot_compat * self.chi1_mask[None, :, :]

        # ------------------------------------------------------------------
        # 3. Probability-weighted compatibility per amino acid type:
        #    aa_compat[i, aa] = sum_r  prob[aa, r] * compat(chi1_pred[i], rot[aa, r])
        #    For GLY and ALA (no chi1): set compatibility = 1 (no penalty).
        # ------------------------------------------------------------------
        aa_compat = (rot_compat * self.chi1_probs[None, :, :]).sum(-1)
        # [M, num_aa]

        # AAs without chi1 are fully compatible with any geometry
        aa_compat = aa_compat + (~self.has_chi1[None, :]).float()

        # Penalty: 0 = perfect match, approaching 1 = poor match
        aa_penalty = 1.0 - aa_compat.clamp(max=1.0)  # [M, num_aa]

        # ------------------------------------------------------------------
        # 4. Couple sequence and structure: weight penalty by soft sequence
        #    probabilities. This creates gradient flow to both networks.
        # ------------------------------------------------------------------
        seq_probs = F.softmax(logits[:, :num_aa], dim=-1)  # [M, num_aa]

        expected_penalty = (seq_probs * aa_penalty).sum(-1)  # [M]

        loss = self.weight * expected_penalty.mean()
        return loss
