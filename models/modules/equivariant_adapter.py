#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Equivariant Adapters: Bridging Language Models and Geometric Deep Learning
for Protein Design.

Transforms SE(3)-invariant PLM embeddings into structure-aware features using
local coordinate frames built from backbone atoms (N-CA-C). All operations
provably maintain SE(3) invariance — only invariant quantities (distances,
angles, traces of relative rotations) are used as features or attention biases.

Components:
    LocalFrameBuilder  – constructs per-residue SO(3) frames from backbone
    EquivariantAdapter – full adapter: LoRA projection, geometric gating,
                         and geometry-biased spatial attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_softmax, scatter_sum


class LocalFrameBuilder(nn.Module):
    """Construct local SE(3)-equivariant coordinate frames from backbone
    N-CA-C atoms via Gram-Schmidt orthogonalization.

    Under a global rotation Q the frame transforms as R_i -> Q R_i,
    but all downstream features are derived from *invariant* operations
    on these frames (e.g. tr(R_i^T R_j)), preserving SE(3) invariance
    of the final output.
    """

    def forward(self, X):
        """
        Args:
            X: [N, n_channel, 3]  (channels 0=N, 1=CA, 2=C, 3=O, ...)
        Returns:
            R: [N, 3, 3]  local rotation matrices
            t: [N, 3]     CA positions (frame origins)
        """
        N_pos  = X[:, 0]  # [N, 3]
        CA_pos = X[:, 1]  # [N, 3]
        C_pos  = X[:, 2]  # [N, 3]

        # e1: CA -> C direction
        e1 = F.normalize(C_pos - CA_pos, dim=-1, eps=1e-6)

        # e2: component of (CA -> N) orthogonal to e1 (Gram-Schmidt)
        v2 = N_pos - CA_pos
        e2 = v2 - (v2 * e1).sum(-1, keepdim=True) * e1
        e2 = F.normalize(e2, dim=-1, eps=1e-6)

        # e3: right-hand rule
        e3 = torch.cross(e1, e2, dim=-1)

        R = torch.stack([e1, e2, e3], dim=-1)  # [N, 3, 3]
        return R, CA_pos


class EquivariantAdapter(nn.Module):
    """Transform SE(3)-invariant PLM embeddings into structure-aware
    invariant features using local coordinate frames.

    Architecture:
        1. LoRA-style low-rank PLM projection  (W_0 + A B)
        2. Invariant geometric feature extraction from backbone geometry
        3. Structure-conditioned gating  (PLM * σ(gate) + geom * (1-σ))
        4. Geometry-biased multi-head attention over kNN graph
        5. Feed-forward residual block

    SE(3) invariance proof sketch:
        - Per-node geometric features use only distances and angles (invariant).
        - Pairwise biases use ||CA_i - CA_j|| and tr(R_i^T R_j), both invariant
          because (QR_i)^T (QR_j) = R_i^T Q^T Q R_j = R_i^T R_j.
        - Edge topology from kNN on distances is invariant.
        - All other operations (linear, softmax, scatter) are pointwise or
          permutation operations that preserve invariance.
    """

    def __init__(self, plm_dim, embed_size, adapter_rank=16,
                 n_heads=4, dropout=0.1):
        super().__init__()
        self.plm_dim = plm_dim
        self.embed_size = embed_size
        self.n_heads = n_heads
        assert embed_size % n_heads == 0
        self.head_dim = embed_size // n_heads

        # --- 1. LoRA-style low-rank PLM projection ---
        self.base_proj = nn.Linear(plm_dim, embed_size)
        self.lora_down = nn.Linear(plm_dim, adapter_rank, bias=False)
        self.lora_up = nn.Linear(adapter_rank, embed_size, bias=False)
        nn.init.zeros_(self.lora_up.weight)

        # --- 2. Local frame builder ---
        self.frame_builder = LocalFrameBuilder()

        # --- 3. Invariant geometric feature encoder ---
        n_geom_feats = 6  # 4 backbone distances + 2 backbone angles
        self.geom_encoder = nn.Sequential(
            nn.Linear(n_geom_feats, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size),
        )

        # --- 4. Structure-conditioned gating ---
        self.gate_net = nn.Sequential(
            nn.Linear(embed_size * 2, embed_size),
            nn.SiLU(),
            nn.Linear(embed_size, embed_size),
            nn.Sigmoid(),
        )

        # --- 5. Geometry-biased multi-head attention ---
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)
        self.dist_bias_proj = nn.Linear(1, n_heads, bias=False)
        self.frame_bias_proj = nn.Linear(1, n_heads, bias=False)
        self.attn_out = nn.Linear(embed_size, embed_size)

        # --- 6. FFN ---
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_size * 2, embed_size),
        )

        # --- norms & dropout ---
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
        self.ln3 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _invariant_geometric_features(self, X, global_mask):
        """Per-residue SE(3)-invariant descriptors from backbone geometry.

        Features (all scalar / invariant):
            d(N, CA), d(CA, C), d(C, O), d(N, C),
            cos(N-CA-C angle), cos(CA-C-O angle)
        """
        N_pos, CA_pos, C_pos, O_pos = X[:, 0], X[:, 1], X[:, 2], X[:, 3]

        d_nca = torch.norm(N_pos  - CA_pos, dim=-1, keepdim=True)
        d_cac = torch.norm(CA_pos - C_pos,  dim=-1, keepdim=True)
        d_co  = torch.norm(C_pos  - O_pos,  dim=-1, keepdim=True)
        d_nc  = torch.norm(N_pos  - C_pos,  dim=-1, keepdim=True)

        e_ca_n = F.normalize(N_pos - CA_pos, dim=-1, eps=1e-6)
        e_ca_c = F.normalize(C_pos - CA_pos, dim=-1, eps=1e-6)
        cos_ncc = (e_ca_n * e_ca_c).sum(-1, keepdim=True)

        e_c_ca = F.normalize(CA_pos - C_pos, dim=-1, eps=1e-6)
        e_c_o  = F.normalize(O_pos  - C_pos, dim=-1, eps=1e-6)
        cos_aco = (e_c_ca * e_c_o).sum(-1, keepdim=True)

        feats = torch.cat([d_nca, d_cac, d_co, d_nc, cos_ncc, cos_aco],
                          dim=-1)  # [N, 6]
        feats[global_mask] = 0.0
        return feats

    # ------------------------------------------------------------------
    def _geometric_attention(self, h, X, R, ctx_edges):
        """Multi-head attention with SE(3)-invariant geometric biases.

        Biases:
            - Negative CA–CA distance  (closer residues attend more)
            - Trace of relative rotation  tr(R_i^T R_j) / 3
              (aligned frames attend more)
        """
        row, col = ctx_edges
        N = h.shape[0]

        Q = self.q_proj(h)[row].view(-1, self.n_heads, self.head_dim)
        K = self.k_proj(h)[col].view(-1, self.n_heads, self.head_dim)
        V = self.v_proj(h)[col].view(-1, self.n_heads, self.head_dim)

        attn = (Q * K).sum(-1) / (self.head_dim ** 0.5)   # [E, heads]

        # distance bias (invariant)
        ca_dist = torch.norm(X[row, 1] - X[col, 1], dim=-1,
                             keepdim=True)                 # [E, 1]
        attn = attn + self.dist_bias_proj(-ca_dist)        # closer → higher

        # frame-alignment bias (invariant: (QR_i)^T(QR_j) = R_i^T R_j)
        R_rel = torch.bmm(R[row].transpose(-1, -2), R[col])   # [E,3,3]
        frame_sim = R_rel.diagonal(dim1=-2, dim2=-1).sum(-1,
                                                          keepdim=True) / 3.0
        attn = attn + self.frame_bias_proj(frame_sim)

        attn = scatter_softmax(attn, row, dim=0)
        attn = self.dropout(attn)

        msg = (attn.unsqueeze(-1) * V).reshape(-1, self.embed_size)
        out = h.new_zeros(N, self.embed_size)
        out.scatter_add_(0, row.unsqueeze(-1).expand_as(msg), msg)
        return self.attn_out(out)

    # ------------------------------------------------------------------
    def forward(self, plm_embeddings, X, ctx_edges, batch_id, global_mask):
        """
        Args:
            plm_embeddings: [N, plm_dim]   precomputed PLM features
            X:              [N, C, 3]      atomic coordinates
            ctx_edges:      [2, E]         context kNN edges
            batch_id:       [N]            graph-to-batch assignment
            global_mask:    [N] bool       True for global (BOA/BOH/BOL) nodes

        Returns:
            H: [N, embed_size]  structure-aware invariant node features
        """
        # 1. LoRA projection
        h = self.base_proj(plm_embeddings) \
            + self.lora_up(self.lora_down(plm_embeddings))
        h = self.ln1(h)

        # 2. Local frames & invariant geometric features
        R, _ = self.frame_builder(X)
        geom_h = self.geom_encoder(
            self._invariant_geometric_features(X, global_mask))

        # 3. Structure-conditioned gating
        gate = self.gate_net(torch.cat([h, geom_h], dim=-1))
        h = h * gate + geom_h * (1.0 - gate)
        h = self.ln2(h)

        # 4. Geometry-biased spatial attention
        h = h + self.dropout(self._geometric_attention(h, X, R, ctx_edges))

        # 5. FFN with residual
        h = h + self.dropout(self.ffn(h))
        h = self.ln3(h)

        return h
