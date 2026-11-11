#!/usr/bin/python
# -*- coding:utf-8 -*-
from math import cos, pi, log, exp
import torch
from .abs_trainer import Trainer


class PRiMETrainer(Trainer):

    ########## Override start ##########

    def __init__(self, model, train_loader, valid_loader, config):
        self.global_step = 0
        self.epoch = 0
        self.max_step = config.max_epoch * config.step_per_epoch
        self.log_alpha = log(config.final_lr / config.lr) / self.max_step
        super().__init__(model, train_loader, valid_loader, config)

    def get_optimizer(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        return optimizer

    def get_scheduler(self, optimizer):
        log_alpha = self.log_alpha
        lr_lambda = lambda step: exp(log_alpha * (step + 1))  # equal to alpha^{step}
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'scheduler': scheduler,
            'frequency': 'batch'
        }

    def train_step(self, batch, batch_idx):
        batch['context_ratio'] = self.get_context_ratio()
        return self.share_step(batch, batch_idx, val=False)

    def valid_step(self, batch, batch_idx):
        batch['context_ratio'] = 0
        return self.share_step(batch, batch_idx, val=True)

    ########## Override end ##########

    def get_context_ratio(self):
        step = self.global_step
        ratio = 0.5 * (cos(step / self.max_step * pi) + 1) * 0.9  # scale to [0, 0.9]
        return ratio

    def share_step(self, batch, batch_idx, val=False):
        results = self.model(**batch)
        if len(results) == 8:
            loss, seq_detail, structure_detail, dock_detail, pdev_detail, flow_detail, rotamer_detail, clash_detail = results
        elif len(results) == 7:
            loss, seq_detail, structure_detail, dock_detail, pdev_detail, flow_detail, rotamer_detail = results
            clash_detail = None
        elif len(results) == 6:
            loss, seq_detail, structure_detail, dock_detail, pdev_detail, flow_detail = results
            rotamer_detail = None
            clash_detail = None
        else:
            loss, seq_detail, structure_detail, dock_detail, pdev_detail = results
            flow_detail = None
            rotamer_detail = None
            clash_detail = None
        snll, aar = seq_detail
        struct_loss, xloss, bond_loss, sc_bond_loss = structure_detail
        dock_loss, interface_loss, ed_loss, r_ed_losses = dock_detail
        pdev_loss, prmsd_loss = pdev_detail

        log_type = 'Validation' if val else 'Train'

        self.log(f'Overall/Loss/{log_type}', loss, batch_idx, val)

        self.log(f'Seq/SNLL/{log_type}', snll, batch_idx, val)
        self.log(f'Seq/AAR/{log_type}', aar, batch_idx, val)

        self.log(f'Struct/StructLoss/{log_type}', struct_loss, batch_idx, val)
        self.log(f'Struct/XLoss/{log_type}', xloss, batch_idx, val)
        self.log(f'Struct/BondLoss/{log_type}', bond_loss, batch_idx, val)
        self.log(f'Struct/SidechainBondLoss/{log_type}', sc_bond_loss, batch_idx, val)

        self.log(f'Dock/DockLoss/{log_type}', dock_loss, batch_idx, val)
        self.log(f'Dock/SPLoss/{log_type}', interface_loss, batch_idx, val)
        self.log(f'Dock/EDLoss/{log_type}', ed_loss, batch_idx, val)
        for i, l in enumerate(r_ed_losses):
            self.log(f'Dock/edloss{i}/{log_type}', l, batch_idx, val)

        if pdev_loss is not None:
            self.log(f'PDev/PDevLoss/{log_type}', pdev_loss, batch_idx, val)
            self.log(f'PDev/PRMSDLoss/{log_type}', prmsd_loss, batch_idx, val)

        if flow_detail is not None:
            flow_total, flow_rot, flow_trans, flow_consist = flow_detail
            self.log(f'Flow/Total/{log_type}', flow_total, batch_idx, val)
            self.log(f'Flow/Rotation/{log_type}', flow_rot, batch_idx, val)
            self.log(f'Flow/Translation/{log_type}', flow_trans, batch_idx, val)
            self.log(f'Flow/Consistency/{log_type}', flow_consist, batch_idx, val)

        if rotamer_detail is not None:
            rotamer_loss, = rotamer_detail
            self.log(f'Rotamer/Loss/{log_type}', rotamer_loss, batch_idx, val)

        if clash_detail is not None:
            clash_total, clash_interface, clash_intra, n_iclash, n_aclash = clash_detail
            self.log(f'Clash/Total/{log_type}', clash_total, batch_idx, val)
            self.log(f'Clash/Interface/{log_type}', clash_interface, batch_idx, val)
            self.log(f'Clash/Intra/{log_type}', clash_intra, batch_idx, val)
            self.log(f'Clash/N_Interface/{log_type}', n_iclash, batch_idx, val)
            self.log(f'Clash/N_Intra/{log_type}', n_aclash, batch_idx, val)

        if not val:
            lr = self.config.lr if self.scheduler is None else self.scheduler.get_last_lr()
            lr = lr[0]
            self.log('lr', lr, batch_idx, val)
            self.log('context_ratio', batch['context_ratio'], batch_idx, val)
        return loss
