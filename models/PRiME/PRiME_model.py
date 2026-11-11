#!/usr/bin/python
# -*- coding:utf-8 -*-
import math
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean

from data.pdb_utils import VOCAB
from utils.nn_utils import SeparatedAminoAcidFeature, ProteinFeature
from utils.nn_utils import GMEdgeConstructor, SeperatedCoordNormalizer
from utils.nn_utils import _knn_edges
from evaluation.rmsd import kabsch_torch

from ..modules.am_enc import AMEncoder
from ..modules.am_egnn import AMEGNN


class PRiMEModel(nn.Module):
    def __init__(self, embed_size, hidden_size, n_channel, num_classes,
                 mask_id=VOCAB.get_mask_idx(), k_neighbors=9, bind_dist_cutoff=6,
                 n_layers=3, iter_round=3, dropout=0.1, struct_only=False,
                 backbone_only=False, fix_channel_weights=False, pred_edge_dist=True,
                 keep_memory=True, cdr_type='H3', paratope='H3', relative_position=False,
                 plm_dim=0,
                 use_equivariant_adapter=False, adapter_rank=16,
                 adapter_n_heads=4,
                 use_riemannian_flow=False, flow_weight=0.5,
                 flow_consistency_weight=0.1, flow_rot_weight=1.0,
                 flow_trans_weight=1.0, flow_n_steps=20,
                 flow_sigma_min=1e-4,
                 flow_time_mu=0.0, flow_time_sigma=1.0,
                 use_flow_iter=False, flow_iter_alpha=0.05,
                 rotamer_weight=0.1, rotamer_sigma_deg=30.0,
                 clash_weight=0.5, clash_interface_weight=2.0,
                 clash_dist=3.0, clash_interface_dist=3.0) -> None:
        super().__init__()
        self.mask_id = mask_id
        self.num_classes = num_classes
        self.bind_dist_cutoff = bind_dist_cutoff
        self.k_neighbors = k_neighbors
        self.round = iter_round
        self.struct_only = struct_only
        self.use_riemannian_flow = use_riemannian_flow
        self.use_flow_iter = use_flow_iter and use_riemannian_flow
        self.flow_iter_alpha = flow_iter_alpha

        # options
        self.backbone_only = backbone_only
        self.fix_channel_weights = fix_channel_weights
        self.pred_edge_dist = pred_edge_dist
        self.keep_memory = keep_memory
        if self.backbone_only:
            n_channel = 4
        self.cdr_type = cdr_type
        self.paratope = paratope

        atom_embed_size = embed_size // 4
        self.aa_feature = SeparatedAminoAcidFeature(
            embed_size, atom_embed_size,
            relative_position=relative_position,
            edge_constructor=GMEdgeConstructor,
            fix_atom_weights=fix_channel_weights,
            backbone_only=backbone_only,
            plm_dim=plm_dim,
            use_equivariant_adapter=use_equivariant_adapter,
            adapter_rank=adapter_rank,
            adapter_n_heads=adapter_n_heads,
        )
        self.protein_feature = ProteinFeature(backbone_only=backbone_only)
        if keep_memory:
            self.memory_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, embed_size)
            )
        if self.pred_edge_dist:  # use predicted dist for KNN-graph at the interface
            if self.keep_memory:  # this ffn acts on the memory
                self.edge_H_ffn = nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size),
                    nn.SiLU(),
                    nn.Linear(hidden_size, hidden_size)
                )
            self.edge_dist_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(2 * hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
            # this GNN encodes the initial hidden states for initial edge distance prediction
            self.init_gnn = AMEGNN(
                embed_size, hidden_size, hidden_size, n_channel,
                channel_nf=atom_embed_size, radial_nf=hidden_size,
                in_edge_nf=0, n_layers=n_layers, residual=True,
                dropout=dropout, dense=False)
        if not struct_only:
            self.ffn_residue = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, self.num_classes)
            )
        else:
            self.prmsd_ffn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 1)
            )
        self.gnn = AMEncoder(
            embed_size, hidden_size, hidden_size, n_channel,
            channel_nf=atom_embed_size, radial_nf=hidden_size,
            in_edge_nf=0, n_layers=n_layers, residual=True,
            dropout=dropout, dense=False)
        
        self.normalizer = SeperatedCoordNormalizer()

        # Riemannian flow matching on SE(3) frames
        if self.use_riemannian_flow:
            from ..modules.riemannian_flow import FrameAwareLoss
            from ..modules.equivariant_adapter import LocalFrameBuilder
            self.frame_builder = LocalFrameBuilder()
            self.frame_flow_loss = FrameAwareLoss(
                hidden_dim=hidden_size,
                flow_weight=flow_weight,
                consistency_weight=flow_consistency_weight,
                rot_weight=flow_rot_weight,
                trans_weight=flow_trans_weight,
                n_integration_steps=flow_n_steps,
                sigma_min=flow_sigma_min,
                dropout=dropout,
                time_mu=flow_time_mu,
                time_sigma=flow_time_sigma,
            )

        # Rotamer-aware joint loss (couples sequence and structure)
        self.use_rotamer_loss = (rotamer_weight > 0) and (not struct_only) and (not backbone_only)
        if self.use_rotamer_loss:
            from ..modules.rotamer_loss import RotamerConsistencyLoss
            self.rotamer_loss_fn = RotamerConsistencyLoss(
                num_aa_types=num_classes,
                weight=rotamer_weight,
                sigma_deg=rotamer_sigma_deg,
            )

        # Steric clash loss (penalises atom overlaps at CDR-antigen interface)
        self.use_clash_loss = clash_weight > 0
        if self.use_clash_loss:
            from ..modules.clash_loss import StericClashLoss
            self.clash_loss_fn = StericClashLoss(
                clash_weight=clash_weight,
                interface_weight=clash_interface_weight,
                clash_dist=clash_dist,
                interface_clash_dist=clash_interface_dist,
            )

        # training related cache
        self.batch_constants = {}

        # Ideal backbone bond vectors in local frame (Gram-Schmidt: e1=CA->C, e2=perp(CA->N), e3=cross)
        # These are average values from high-resolution protein structures (Angstroms).
        # N is at ~1.46A from CA, roughly in the -e1 + e2 direction
        # C is at ~1.52A from CA, along +e1
        # O is at ~2.40A from CA, roughly in the +e1 - e2 direction
        # Registered as buffers so they move with model.to(device)/model.cuda()
        self.register_buffer('_ideal_N_local', torch.tensor([[-0.522, 1.362, 0.0]]).T)  # [3, 1]
        self.register_buffer('_ideal_C_local', torch.tensor([[1.522, 0.0, 0.0]]).T)     # [3, 1]
        self.register_buffer('_ideal_O_local', torch.tensor([[2.094, -1.174, 0.0]]).T)  # [3, 1]

    def _ideal_N_offset(self, N, device, dtype):
        """[N, 3, 1] ideal N atom position in local frame."""
        return self._ideal_N_local.to(dtype=dtype).expand(N, -1, -1)

    def _ideal_C_offset(self, N, device, dtype):
        """[N, 3, 1] ideal C atom position in local frame."""
        return self._ideal_C_local.to(dtype=dtype).expand(N, -1, -1)

    def _ideal_O_offset(self, N, device, dtype):
        """[N, 3, 1] ideal O atom position in local frame."""
        return self._ideal_O_local.to(dtype=dtype).expand(N, -1, -1)

    def init_mask(self, X, S, cmask, smask, template):
        if not self.struct_only:
            S[smask] = self.mask_id
        X[cmask] = template
        return X, S

    def message_passing(self, X, S, residue_pos, interface_X, paratope_mask, batch_id, t, memory_H=None, smooth_prob=None, smooth_mask=None, plm_embeddings=None):
        # embeddings
        H_0, (ctx_edges, inter_edges), (atom_embeddings, atom_weights) = self.aa_feature(X, S, batch_id, self.k_neighbors, residue_pos, smooth_prob=smooth_prob, smooth_mask=smooth_mask, plm_embeddings=plm_embeddings)

        if not self.keep_memory:
            memory_H = None

        if memory_H is not None:
            H_0 = H_0 + self.memory_ffn(memory_H)

        if self.pred_edge_dist:
            if memory_H is not None:
                edge_H = self.edge_H_ffn(memory_H)
            else:
                # replace the MLP with gnn for initial edge distance prediction
                edge_H, dumb_X = self.init_gnn(H_0, X, ctx_edges,
                                       channel_attr=atom_embeddings,
                                       channel_weights=atom_weights)
                X = X + dumb_X * 0  # to cheat the autograd check

        # update coordination of the global node
        X = self.aa_feature.update_globel_coordinates(X, S)

        # prepare local complex
        local_mask = self.batch_constants['local_mask']
        local_is_ab = self.batch_constants['local_is_ab']
        local_batch_id = self.batch_constants['local_batch_id']
        local_X = X[local_mask].clone()
        # prepare local complex edges
        local_ctx_edges = self.batch_constants['local_ctx_edges']  # [2, Ec]
        local_inter_edges = self.batch_constants['local_inter_edges']  # [2, Ei]
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        offsets, max_n, gni2lni = self.batch_constants['local_edge_infos']
        # for context edges, use edges in the native paratope
        local_ctx_edges = _knn_edges(
            local_X, atom_pos, local_ctx_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni))
        # for interative edges, use edges derived from the predicted distance
        local_X[local_is_ab] = interface_X
        if self.pred_edge_dist:
            local_H = edge_H[local_mask]
            src_H, dst_H = local_H[local_inter_edges[0]], local_H[local_inter_edges[1]]
            p_edge_dist = self.edge_dist_ffn(torch.cat([src_H, dst_H], dim=-1)) +\
                          self.edge_dist_ffn(torch.cat([dst_H, src_H], dim=-1))  # perm-invariant
            p_edge_dist = p_edge_dist.squeeze()
        else:
            p_edge_dist = None
        local_inter_edges = _knn_edges(
            local_X, atom_pos, local_inter_edges.T,
            self.aa_feature.atom_pos_pad_idx, self.k_neighbors,
            (offsets, local_batch_id, max_n, gni2lni), given_dist=p_edge_dist)
        local_edges = torch.cat([local_ctx_edges, local_inter_edges], dim=1)

        # message passing
        H, pred_X, pred_local_X = self.gnn(H_0, X, ctx_edges,
                                           local_mask, local_X, local_edges,
                                           paratope_mask, local_is_ab,
                                           channel_attr=atom_embeddings,
                                           channel_weights=atom_weights)
        interface_X = pred_local_X[local_is_ab]
        pred_logits = None if self.struct_only else self.ffn_residue(H)

        return pred_logits, pred_X, interface_X, H, p_edge_dist  # [N, num_classes], [N, n_channel, 3], [Ncdr, n_channel, 3], [N, hidden_size]
    
    @torch.no_grad()
    def init_interface(self, X, S, paratope_mask, batch_id, init_noise=None):
        ag_centers = X[S == self.aa_feature.boa_idx][:, 0]  # [bs, 3]
        init_local_X = torch.zeros_like(X[paratope_mask])
        init_local_X = init_local_X + ag_centers[batch_id[paratope_mask]].unsqueeze(1)
        noise = torch.randn_like(init_local_X) if init_noise is None else init_noise
        ca_noise = noise[:, 1]
        noise = noise / 10  + ca_noise.unsqueeze(1) # scale other atoms
        noise[:, 1] = ca_noise
        init_local_X = init_local_X + noise
        return init_local_X

    @torch.no_grad()
    def _prepare_batch_constants(self, S, paratope_mask, lengths):
        # generate batch id
        batch_id = torch.zeros_like(S)  # [N]
        batch_id[torch.cumsum(lengths, dim=0)[:-1]] = 1
        batch_id.cumsum_(dim=0)  # [N], item idx in the batch
        self.batch_constants['batch_id'] = batch_id
        self.batch_constants['batch_size'] = torch.max(batch_id) + 1

        segment_ids = self.aa_feature._construct_segment_ids(S)
        self.batch_constants['segment_ids'] = segment_ids

        # interface relatd
        is_ag = segment_ids == self.aa_feature.ag_seg_id
        not_ag_global = S != self.aa_feature.boa_idx
        local_mask = torch.logical_or(
            paratope_mask, torch.logical_and(is_ag, not_ag_global)
        )
        local_segment_ids = segment_ids[local_mask]
        local_is_ab = local_segment_ids != self.aa_feature.ag_seg_id
        local_batch_id = batch_id[local_mask]
        self.batch_constants['is_ag'] = is_ag
        self.batch_constants['local_mask'] = local_mask
        self.batch_constants['local_is_ab'] = local_is_ab
        self.batch_constants['local_batch_id'] = local_batch_id
        self.batch_constants['local_segment_ids'] = local_segment_ids
        # interface local edges
        (row, col), (offsets, max_n, gni2lni) = self.aa_feature.edge_constructor.get_batch_edges(local_batch_id)
        row_segment_ids, col_segment_ids = local_segment_ids[row], local_segment_ids[col]
        is_ctx = row_segment_ids == col_segment_ids
        is_inter = torch.logical_not(is_ctx)

        self.batch_constants['local_ctx_edges'] = torch.stack([row[is_ctx], col[is_ctx]])  # [2, Ec]
        self.batch_constants['local_inter_edges'] = torch.stack([row[is_inter], col[is_inter]])  # [2, Ei]
        self.batch_constants['local_edge_infos'] = (offsets, max_n, gni2lni)

        interface_batch_id = batch_id[paratope_mask]
        self.batch_constants['interface_batch_id'] = interface_batch_id
    
    def _clean_batch_constants(self):
        self.batch_constants = {}

    @torch.no_grad()
    def _get_inter_edge_dist(self, X, S):
        local_mask = self.batch_constants['local_mask']
        atom_pos = self.aa_feature._construct_atom_pos(S[local_mask])
        src_dst = self.batch_constants['local_inter_edges'].T
        dist = X[local_mask][src_dst]  # [Ef, 2, n_channel, 3]
        dist = dist[:, 0].unsqueeze(2) - dist[:, 1].unsqueeze(1)  # [Ef, n_channel, n_channel, 3]
        dist = torch.norm(dist, dim=-1)  # [Ef, n_channel, n_channel]
        pos_pad = atom_pos[src_dst] == self.aa_feature.atom_pos_pad_idx # [Ef, 2, n_channel]
        pos_pad = torch.logical_or(pos_pad[:, 0].unsqueeze(2), pos_pad[:, 1].unsqueeze(1))  # [Ef, n_channel, n_channel]
        dist = dist + pos_pad * 1e10  # [Ef, n_channel, n_channel]
        dist = torch.min(dist.reshape(dist.shape[0], -1), dim=1)[0]  # [Ef]
        return dist
        is_binding = dist <= self.bind_dist_cutoff
        return is_binding

    def _forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, plm_embeddings=None, temperature=0.0):
        batch_id = self.batch_constants['batch_id']

        # mask sequence and initialize coordinates with template
        X, S = self.init_mask(X, S, cmask, smask, template)

        # zero out PLM embeddings at masked positions to prevent ground-truth leakage
        if plm_embeddings is not None and not self.struct_only:
            plm_embeddings = plm_embeddings.clone()
            plm_embeddings[smask] = 0

        # normalize
        X = self.normalizer.centering(X, S, batch_id, self.aa_feature)
        X = self.normalizer.normalize(X)

        # update center
        X = self.aa_feature.update_globel_coordinates(X, S)

        # prepare initial interface
        interface_X = self.init_interface(X, S, paratope_mask, batch_id, init_noise)

        # sequence and structure loss
        r_pred_S_logits, pred_S_dist, = [], None
        r_interface_X = [interface_X.clone()]  # init
        r_edge_dist = []
        memory_H = None
        # message passing
        for t in range(self.round):
            pred_S_logits, pred_X, interface_X, H, edge_dist = self.message_passing(X, S, residue_pos, interface_X, paratope_mask, batch_id, t, memory_H, pred_S_dist, smask, plm_embeddings=plm_embeddings)
            memory_H = H
            r_interface_X.append(interface_X.clone())
            r_pred_S_logits.append((pred_S_logits, smask))
            r_edge_dist.append(edge_dist)
            # 1. update X
            X = X.clone()
            X[cmask] = pred_X[cmask]
            X = self.aa_feature.update_globel_coordinates(X, S)

            # Flow-aware iterative refinement: nudge CA positions on
            # non-final rounds so the EGNN sees flow-corrected geometry.
            # s = round_index / num_rounds maps refinement progress to
            # flow time.  Skip the last round to preserve clean EGNN
            # output for CDR/docking.
            if getattr(self, 'use_flow_iter', False) and t < self.round - 1:
                cdr_X = X[cmask]                         # [N_cdr, n_channel, 3]
                R_cur, t_ca = self.frame_builder(cdr_X)  # frames from current geometry
                s_flow = torch.full(
                    (R_cur.shape[0],), t / self.round,
                    device=R_cur.device, dtype=R_cur.dtype)
                cdr_batch_id = batch_id[cmask]
                _, v = self.frame_flow_loss.flow_module.velocity_head(
                    H[cmask], R_cur, t_ca, s_flow, batch_id=cdr_batch_id)
                dt_flow = 1.0 / self.round
                X[cmask, 1] = X[cmask, 1] + self.flow_iter_alpha * dt_flow * v

            if not self.struct_only:
                # 2. update S
                S = S.clone()
                if t == self.round - 1:
                    if temperature > 0:
                        probs = torch.softmax(pred_S_logits[smask] / temperature, dim=-1)
                        S[smask] = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    else:
                        S[smask] = torch.argmax(pred_S_logits[smask], dim=-1)
                else:
                    pred_S_dist = torch.softmax(pred_S_logits[smask], dim=-1)

        interface_batch_id = self.batch_constants['interface_batch_id']

        if self.struct_only:
            # predicted rmsd
            prmsd = self.prmsd_ffn(H[cmask]).squeeze()  # [N_ab]
        else:
            prmsd = None

        # uncentering and unnormalize
        pred_X = self.normalizer.unnormalize(pred_X)
        pred_X = self.normalizer.uncentering(pred_X, batch_id)
        for i, interface_X in enumerate(r_interface_X):
            interface_X = self.normalizer.unnormalize(interface_X)
            interface_X = self.normalizer.uncentering(interface_X, interface_batch_id, _type=4)
            r_interface_X[i] = interface_X
        self.normalizer.clear_cache()

        return H, S, r_pred_S_logits, pred_X, r_interface_X,  r_edge_dist, prmsd

    def forward(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, xloss_mask, context_ratio=0, plm_embeddings=None):
        '''
        :param X: [N, n_channel, 3], Cartesian coordinates
        :param context_ratio: float, rate of context provided in masked sequence, should be [0, 1) and anneal to 0 in training
        :param plm_embeddings: [N, plm_dim] optional precomputed PLM embeddings
        '''
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
            xloss_mask = xloss_mask[:, :4]
        # clone ground truth coordinates, sequence
        true_X, true_S = X.clone(), S.clone()

        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)
        batch_id = self.batch_constants['batch_id']

        # provide some ground truth for annealing sequence training
        if context_ratio > 0:
            not_ctx_mask = torch.rand_like(smask, dtype=torch.float) >= context_ratio
            smask = torch.logical_and(smask, not_ctx_mask)

        # get results
        H, pred_S, r_pred_S_logits, pred_X, r_interface_X, r_edge_dist, prmsd = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, plm_embeddings=plm_embeddings)

        # sequence negtive log likelihood
        snll, total = 0, 0
        if not self.struct_only:
            for logits, mask in r_pred_S_logits:
                snll = snll + F.cross_entropy(logits[mask], true_S[mask], reduction='sum')
                total = total + mask.sum()
            snll = snll / total.clamp(min=1)

        # structure loss
        struct_loss, struct_loss_details, bb_rmsd, ops = self.protein_feature.structure_loss(pred_X, true_X, true_S, cmask, batch_id, xloss_mask, self.aa_feature)

        # docking loss
        gt_interface_X = true_X[paratope_mask]
        # 1. interface loss (shadow paratope)
        interface_atom_pos = self.aa_feature._construct_atom_pos(true_S[paratope_mask])
        interface_atom_mask = interface_atom_pos != self.aa_feature.atom_pos_pad_idx
        interface_loss = F.smooth_l1_loss(
            r_interface_X[-1][interface_atom_mask],
            gt_interface_X[interface_atom_mask])
        # 2. edge dist loss
        if self.pred_edge_dist:
            gt_edge_dist = self._get_inter_edge_dist(self.normalizer.normalize(true_X), true_S)
            ed_loss, r_ed_losses = 0, []
            for edge_dist in r_edge_dist:
                r_ed_loss = F.smooth_l1_loss(edge_dist, gt_edge_dist)
                ed_loss = ed_loss + r_ed_loss
                r_ed_losses.append(r_ed_loss)
        else:
            r_ed_losses = [0 for _ in range(self.round)]
            ed_loss = 0
        dock_loss = interface_loss + ed_loss

        if self.struct_only:
            # predicted rmsd
            prmsd_loss = F.smooth_l1_loss(prmsd, bb_rmsd)
            pdev_loss = prmsd_loss
        else:
            pdev_loss, prmsd_loss = None, None

        # Riemannian flow matching loss on SE(3) frames (pred -> GT)
        # Fix #1: Detach — no gradient competition between EGNN and flow
        # Fix #2: Perturb pred frames — prevents collapse to trivial velocity
        # Fix #3: Mix 50% pred-mode / 50% noise-mode — broader coverage
        flow_loss_total, flow_loss_rot, flow_loss_trans, flow_consistency = \
            torch.tensor(0.0, device=H.device), torch.tensor(0.0, device=H.device), \
            torch.tensor(0.0, device=H.device), torch.tensor(0.0, device=H.device)
        if self.use_riemannian_flow:
            from ..modules.so3_utils import so3_noise, random_so3
            R_gt, t_gt_ca = self.frame_builder(true_X)
            R_pred, t_pred_ca = self.frame_builder(pred_X)
            _, (ctx_edges, _), _ = self.aa_feature(
                pred_X, pred_S, batch_id, self.k_neighbors,
                residue_pos, plm_embeddings=plm_embeddings)

            # Fix #1: Detach pred frames from EGNN graph
            R_start = R_pred.detach()
            t_start = t_pred_ca.detach()

            # Fix #2: Perturb pred frames to avoid trivial velocity targets
            R_start = so3_noise(R_start, sigma=0.1)
            t_start = t_start + torch.randn_like(t_start) * 0.5

            # Fix #3: Per-graph 50/50 mix of pred-mode and noise-mode
            bs = batch_id.max().item() + 1
            use_noise = torch.rand(bs, device=H.device) < 0.5
            noise_mask = use_noise[batch_id]  # [N]
            if noise_mask.any():
                R_noise = random_so3(R_start.shape[0], R_start.device, R_start.dtype)
                t_noise = t_gt_ca + torch.randn_like(t_gt_ca) * \
                    self.frame_flow_loss.flow_module.noiser.trans_noise_scale
                R_start[noise_mask] = R_noise[noise_mask]
                t_start[noise_mask] = t_noise[noise_mask]

            flow_loss_total, flow_loss_detail, flow_consistency, flow_loss_rot, flow_loss_trans = \
                self.frame_flow_loss(
                    H, R_gt, t_gt_ca, R_pred, ctx_edges,
                    mask=cmask, batch_id=batch_id,
                    R_flow_start=R_start, t_flow_start=t_start)

        # Rotamer consistency loss (couples sequence and structure)
        rotamer_loss = torch.tensor(0.0, device=H.device)
        if self.use_rotamer_loss and not self.struct_only:
            # Use the final iteration's logits and predicted coordinates
            final_logits = r_pred_S_logits[-1][0]  # [N, num_classes]
            rotamer_loss = self.rotamer_loss_fn(final_logits, pred_X, smask)

        # Steric clash loss (CDR vs antigen + intra-CDR)
        clash_loss_total = torch.tensor(0.0, device=H.device)
        clash_interface = torch.tensor(0.0, device=H.device)
        clash_intra = torch.tensor(0.0, device=H.device)
        n_interface_clashes, n_intra_clashes = 0, 0
        if self.use_clash_loss:
            is_ag = self.batch_constants['is_ag']
            segment_ids = self.batch_constants['segment_ids']
            clash_loss_total, clash_interface, clash_intra, \
                n_interface_clashes, n_intra_clashes = \
                self.clash_loss_fn(
                    pred_X, paratope_mask, is_ag, batch_id, segment_ids)

        # comprehensive loss
        loss = snll + struct_loss + dock_loss \
            + (0 if pdev_loss is None else pdev_loss) \
            + flow_loss_total \
            + rotamer_loss \
            + clash_loss_total

        # NaN debugger — prints once per rank then disables itself
        if torch.isnan(loss) and not getattr(self, '_nan_reported', False):
            self._nan_reported = True
            _p = lambda k, v: f'{k}={v.item():.4f}' if isinstance(v, torch.Tensor) else f'{k}={v}'
            parts = dict(snll=snll, struct=struct_loss, dock=dock_loss,
                         iface=interface_loss, ed=ed_loss, flow=flow_loss_total,
                         flow_rot=flow_loss_rot, flow_trans=flow_loss_trans,
                         flow_consist=flow_consistency,
                         rotamer=rotamer_loss, clash=clash_loss_total)
            print(f'[NaN] ' + ' | '.join(_p(k,v) for k,v in parts.items()), flush=True)
            print(f'[NaN] pred_X nan={torch.isnan(pred_X).any()} H nan={torch.isnan(H).any()}', flush=True)

        self._clean_batch_constants()

        # AAR
        with torch.no_grad():
            aa_hit = pred_S[smask] == true_S[smask]
            aar = aa_hit.long().sum() / aa_hit.shape[0]

        return loss, (snll, aar), (struct_loss, *struct_loss_details), \
            (dock_loss, interface_loss, ed_loss, r_ed_losses), \
            (pdev_loss, prmsd_loss), \
            (flow_loss_total, flow_loss_rot, flow_loss_trans, flow_consistency), \
            (rotamer_loss,), \
            (clash_loss_total, clash_interface, clash_intra,
             n_interface_clashes, n_intra_clashes)

    def sample(self, X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise=None, return_hidden=False, plm_embeddings=None, flow_refine_steps=0, temperature=0.0):
        if self.backbone_only:
            X, template = X[:, :4], template[:, :4]  # backbone
        gen_X, gen_S = X.clone(), S.clone()
        
        # prepare constants
        self._prepare_batch_constants(S, paratope_mask, lengths)

        batch_id = self.batch_constants['batch_id']
        batch_size = self.batch_constants['batch_size']
        segment_ids = self.batch_constants['segment_ids']
        interface_batch_id = self.batch_constants['interface_batch_id']
        is_ab = segment_ids != self.aa_feature.ag_seg_id
        s_batch_id = batch_id[smask]

        best_metric = torch.ones(batch_size, dtype=torch.float, device=X.device) * 1e10
        interface_cmask = paratope_mask[cmask]

        n_tries = 10 if self.struct_only else 1
        for i in range(n_tries):
        
            # generate
            H, pred_S, r_pred_S_logits, pred_X, r_interface_X, _, prmsd = self._forward(X, S, cmask, smask, paratope_mask, residue_pos, template, lengths, init_noise, plm_embeddings=plm_embeddings, temperature=temperature)

            # PPL or PRMSD
            if not self.struct_only:
                S_logits = r_pred_S_logits[-1][0][smask]
                S_probs = torch.max(torch.softmax(S_logits, dim=-1), dim=-1)[0]
                nlls = -torch.log(S_probs)
                metric = scatter_mean(nlls, s_batch_id)  # [batch_size]
            else:
                metric = scatter_mean(prmsd[interface_cmask], interface_batch_id)  # [batch_size]

            update = metric < best_metric
            cupdate = cmask & update[batch_id]
            supdate = smask & update[batch_id]
            # update metric history
            best_metric[update] = metric[update]

            # 1. set generated part
            gen_X[cupdate] = pred_X[cupdate]
            if not self.struct_only:
                gen_S[supdate] = pred_S[supdate]
        
            interface_X = r_interface_X[-1]
            # 2. align by cdr
            for i in range(batch_size):
                if not update[i]:
                    continue
                # 1. align CDRH3
                is_cur_graph = batch_id == i
                cdrh3_cur_graph = torch.logical_and(is_cur_graph, paratope_mask)
                ori_cdr = gen_X[cdrh3_cur_graph][:, :4]  # backbone
                pred_cdr = interface_X[interface_batch_id == i][:, :4]
                _, R, t = kabsch_torch(ori_cdr.reshape(-1, 3), pred_cdr.reshape(-1, 3))

                # 2. tranform antibody
                is_cur_ab = is_cur_graph & is_ab
                ab_X = torch.matmul(gen_X[is_cur_ab], R.T) + t
                gen_X[is_cur_ab] = ab_X

        # Flow refinement: integrate from EGNN prediction toward GT.
        # Fix #1: Reconstruct full backbone (N, CA, C, O) from refined frames.
        # Fix #2: Recompute H at each integration step via single-round message passing.
        # Fix #3: Start from EGNN prediction (matching training distribution).
        if self.use_riemannian_flow and flow_refine_steps > 0:
            from ..modules.so3_utils import so3_exp

            cdr_mask = cmask
            cdr_X = gen_X[cdr_mask]
            R_init, t_init = self.frame_builder(cdr_X)  # EGNN prediction frames
            N_cdr = R_init.shape[0]
            device, dtype = R_init.device, R_init.dtype
            cdr_batch_id = batch_id[cdr_mask]

            n_steps = flow_refine_steps
            dt = 1.0 / n_steps
            R_cur, t_cur = R_init.clone(), t_init.clone()

            for step in range(n_steps):
                s_val = step * dt
                s = torch.full((N_cdr,), s_val, device=device, dtype=dtype)

                # Fix #2: Recompute H by running message passing on current geometry
                temp_X = gen_X.clone()
                temp_X[cdr_mask, 1] = t_cur
                temp_X[cdr_mask, 0] = t_cur + torch.bmm(R_cur, self._ideal_N_offset(N_cdr, device, dtype))[:, :, 0]
                temp_X[cdr_mask, 2] = t_cur + torch.bmm(R_cur, self._ideal_C_offset(N_cdr, device, dtype))[:, :, 0]
                temp_X = self.aa_feature.update_globel_coordinates(temp_X, gen_S)

                # Single message passing round to get fresh H (lightweight)
                interface_X_tmp = temp_X[paratope_mask]
                _, _, _, H_fresh, _ = self.message_passing(
                    temp_X, gen_S, residue_pos, interface_X_tmp,
                    paratope_mask, batch_id, 0, None, None, None,
                    plm_embeddings=plm_embeddings)

                # Predict velocity with fresh H
                omega, v = self.frame_flow_loss.flow_module.velocity_head(
                    H_fresh[cdr_mask], R_cur, t_cur, s, batch_id=cdr_batch_id)

                # Midpoint integration
                R_mid = R_cur @ so3_exp(0.5 * dt * omega)
                t_mid = t_cur + 0.5 * dt * v
                s_mid = torch.full((N_cdr,), s_val + 0.5 * dt, device=device, dtype=dtype)

                omega2, v2 = self.frame_flow_loss.flow_module.velocity_head(
                    H_fresh[cdr_mask], R_mid, t_mid, s_mid, batch_id=cdr_batch_id)

                R_cur = R_cur @ so3_exp(dt * omega2)
                t_cur = t_cur + dt * v2

            # Fix #1: Reconstruct full backbone from refined frames (R, t)
            gen_X[cdr_mask, 1] = t_cur                  # CA
            gen_X[cdr_mask, 0] = t_cur + torch.bmm(     # N
                R_cur, self._ideal_N_offset(N_cdr, device, dtype))[:, :, 0]
            gen_X[cdr_mask, 2] = t_cur + torch.bmm(     # C
                R_cur, self._ideal_C_offset(N_cdr, device, dtype))[:, :, 0]
            gen_X[cdr_mask, 3] = t_cur + torch.bmm(     # O
                R_cur, self._ideal_O_offset(N_cdr, device, dtype))[:, :, 0]

        self._clean_batch_constants()

        if return_hidden:
            return gen_X, gen_S, metric, H
        return gen_X, gen_S, metric

if __name__ == '__main__':
    torch.random.manual_seed(0)
    # equivariance test
    embed_size, hidden_size = 64, 128
    n_channel, d = 14, 3
    n_aa_type = 20
    scale = 10
    dtype = torch.float
    device = torch.device('cuda:0')
    model = PRiMEModel(embed_size, hidden_size, n_channel,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   k_neighbors=9, bind_dist_cutoff=6.6, n_layers=3)
    model.to(device)
    model.eval()

    ag_len, h_len, l_len = 48, 120, 110
    center_x = torch.randn(3, 1, n_channel, d, device=device, dtype=dtype) * scale
    ag_X = torch.randn(ag_len, n_channel, d, device=device, dtype=dtype) * scale
    h_X = torch.randn(h_len, n_channel, d, device=device, dtype=dtype) * scale
    l_X = torch.randn(l_len, n_channel, d, device=device, dtype=dtype) * scale

    X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
    S = torch.cat([torch.tensor([model.aa_feature.boa_idx], device=device),
                   torch.randint(low=0, high=20, size=(ag_len,), device=device),
                   torch.tensor([model.aa_feature.boh_idx], device=device),
                   torch.randint(low=0, high=20, size=(h_len,), device=device),
                   torch.tensor([model.aa_feature.bol_idx], device=device),
                   torch.randint(low=0, high=20, size=(l_len,), device=device)], dim=0)
    cmask = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + [1 for _ in range(h_len)] + [0] + [1 for _ in range(l_len)], device=device).bool()
    smask = torch.zeros_like(cmask)
    smask[ag_len+10:ag_len+20] = 1
    paratope_mask = smask
    residue_pos = torch.tensor([0] + [0 for _ in range(ag_len)] + [0] + list(range(1, h_len + 1)) + [0] + list(range(1, l_len + 1)), device=device)
    template = torch.randn_like(X[cmask]) * scale
    lengths = torch.tensor([ag_len + h_len + l_len + 3], device=device)
    noise = torch.randn_like(X[paratope_mask])

    torch.random.manual_seed(1)
    gen_X, _, _ = model.sample(X, S, cmask, smask, smask, residue_pos, template, lengths, init_noise=noise)
    # tmpx1 = model.tmpx

    # random rotaion matrix
    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q_ag, t_ag = U.mm(V), torch.randn(3, device=device)

    U, _, V = torch.linalg.svd(torch.randn(3, 3, device=device, dtype=torch.float))
    if torch.linalg.det(U) * torch.linalg.det(V) < 0:
        U[:, -1] = -U[:, -1]
    Q_ab, t_ab = U.mm(V), torch.randn(3, device=device)

    ag_X = torch.matmul(ag_X, Q_ag) + t_ag
    X = torch.cat([center_x[0], ag_X, center_x[1], h_X, center_x[2], l_X], dim=0)
    template = torch.matmul(template, Q_ab) + t_ab

    # this is f([Q1x1+t1, Q2x2+t2])
    torch.random.manual_seed(1)
    noise = torch.matmul(noise, Q_ag)
    gen_op_X, _, _ = model.sample(X, S, cmask, smask, smask, residue_pos, template, lengths, init_noise=noise)
    # tmpx2 = model.tmpx
    # gt_tmpx = torch.matmul(tmpx1, Q_ag) + t_ag
    # error = torch.abs(gt_tmpx[:, :4] - tmpx2[:, :4]).sum(-1).flatten().mean()
    # # error = torch.abs(tmpx2 - tmpx1).sum(-1).flatten().mean()
    # print(error.item())
    # assert error < 1e-3
    # print('independent equivariance check passed')


    gt_op_X = torch.matmul(gen_X, Q_ag) + t_ag

    error = torch.abs(gt_op_X[cmask][:, :4] - gen_op_X[cmask][:, :4]).sum(-1).flatten().mean()
    print(error.item())
    assert error < 1e-3
    print('independent equivariance check passed')


