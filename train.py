#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import argparse
import torch
from torch.utils.data import DataLoader

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
setup_seed(SEED)

########### Import your packages below ##########
from data.dataset import E2EDataset, VOCAB
from trainer import TrainConfig


def parse():
    parser = argparse.ArgumentParser(description='training')
    # data
    parser.add_argument('--train_set', type=str, required=True, help='path to train set')
    parser.add_argument('--valid_set', type=str, required=True, help='path to valid set')
    parser.add_argument('--cdr', type=str, default=None, nargs='+', help='cdr to generate, L1/2/3, H1/2/3,(can be list, e.g., L3 H3) None for all including framework')
    parser.add_argument('--paratope', type=str, default='H3', nargs='+', help='cdrs to use as paratope')

    # training related
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--final_lr', type=float, default=1e-4, help='exponential decay from lr to final_lr')
    parser.add_argument('--warmup', type=int, default=0, help='linear learning rate warmup')
    parser.add_argument('--max_epoch', type=int, default=10, help='max training epoch')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='clip gradients with too big norm')
    parser.add_argument('--save_dir', type=str, required=True, help='directory to save model and logs')
    parser.add_argument('--batch_size', type=int, required=True, help='batch size')
    parser.add_argument('--patience', type=int, default=1000, help='patience before early stopping (set with a large number to turn off early stopping)')
    parser.add_argument('--save_topk', type=int, default=10, help='save topk checkpoint. -1 for saving all ckpt that has a better validation metric than its previous epoch')
    parser.add_argument('--shuffle', action='store_true', help='shuffle data')
    parser.add_argument('--num_workers', type=int, default=4)

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    
    # model
    parser.add_argument('--model_type', type=str, required=True, choices=['PRiME', 'PRiMEOpt'],
                        help='Type of model')
    parser.add_argument('--embed_dim', type=int, default=64, help='dimension of residue/atom embedding')
    parser.add_argument('--hidden_size', type=int, default=128, help='dimension of hidden states')
    parser.add_argument('--k_neighbors', type=int, default=9, help='Number of neighbors in KNN graph')
    parser.add_argument('--n_layers', type=int, default=3, help='Number of layers')
    parser.add_argument('--iter_round', type=int, default=3, help='Number of iterations for generation')

    # PRiMEOpt related
    parser.add_argument('--seq_warmup', type=int, default=0, help='Number of epochs before starting training sequence')

    # task setting
    parser.add_argument('--struct_only', action='store_true', help='Predict complex structure given the sequence')
    parser.add_argument('--bind_dist_cutoff', type=float, default=6.6, help='distance cutoff to decide the binding interface')

    # ablation
    parser.add_argument('--no_pred_edge_dist', action='store_true', help='Turn off edge distance prediction at the interface')
    parser.add_argument('--backbone_only', action='store_true', help='Model backbone only')
    parser.add_argument('--fix_channel_weights', action='store_true', help='Fix channel weights, may also for special use (e.g. antigen with modified AAs)')
    parser.add_argument('--no_memory', action='store_true', help='No memory passing')

    # PLM embeddings (ESM-2)
    parser.add_argument('--use_plm', action='store_true', help='Use precomputed PLM (ESM-2) residue embeddings')
    parser.add_argument('--plm_dim', type=int, default=1280, help='Dimension of PLM embeddings (1280 for ESM-2 650M)')
    parser.add_argument('--train_plm_embeddings', type=str, default=None, help='Path to precomputed PLM embeddings for training set')
    parser.add_argument('--valid_plm_embeddings', type=str, default=None, help='Path to precomputed PLM embeddings for validation set')

    # Equivariant adapter (structure-aware PLM adaptation)
    parser.add_argument('--use_equivariant_adapter', action='store_true', help='Use equivariant adapter for PLM projection (requires --use_plm)')
    parser.add_argument('--adapter_rank', type=int, default=16, help='LoRA rank for equivariant adapter')
    parser.add_argument('--adapter_n_heads', type=int, default=4, help='Number of attention heads in equivariant adapter')

    # Riemannian flow matching on SE(3) frames
    parser.add_argument('--use_riemannian_flow', action='store_true', help='Enable Riemannian flow matching on SO(3)^N x R^3N frames')
    parser.add_argument('--flow_weight', type=float, default=0.5, help='Weight for flow matching loss')
    parser.add_argument('--flow_consistency_weight', type=float, default=0.1, help='Weight for frame consistency regularization')
    parser.add_argument('--flow_rot_weight', type=float, default=1.0, help='Weight for rotation velocity loss')
    parser.add_argument('--flow_trans_weight', type=float, default=1.0, help='Weight for translation velocity loss')
    parser.add_argument('--flow_n_steps', type=int, default=20, help='Number of integration steps for flow generation')
    parser.add_argument('--flow_sigma_min', type=float, default=1e-4, help='Minimum sigma for flow noising')
    parser.add_argument('--flow_time_mu', type=float, default=0.0, help='Logit-normal time schedule location (0 = centered at s=0.5)')
    parser.add_argument('--flow_time_sigma', type=float, default=1.0, help='Logit-normal time schedule scale (1.0 = moderate concentration)')
    parser.add_argument('--use_flow_iter', action='store_true', help='Enable flow-aware iterative refinement (nudge CA in non-final rounds)')
    parser.add_argument('--flow_iter_alpha', type=float, default=0.05, help='Step size for flow-aware iterative nudge')

    # Rotamer-aware joint loss (sequence-structure coupling)
    parser.add_argument('--rotamer_weight', type=float, default=0.1, help='Weight for rotamer consistency loss (0 to disable)')
    parser.add_argument('--rotamer_sigma_deg', type=float, default=30.0, help='Angular tolerance in degrees for rotamer matching')

    # Steric clash loss (penalises CDR-antigen atom overlaps)
    parser.add_argument('--clash_weight', type=float, default=0.5, help='Weight for steric clash loss (0 to disable)')
    parser.add_argument('--clash_interface_weight', type=float, default=2.0, help='Relative weight for interface vs intra-CDR clashes')
    parser.add_argument('--clash_dist', type=float, default=3.0, help='Min CA-CA distance (A) for intra-CDR clash')
    parser.add_argument('--clash_interface_dist', type=float, default=3.0, help='Min CA-CA distance (A) for interface clash')

    return parser.parse_args()


def main(args):
    ########### load your train / valid set ###########
    if (len(args.gpus) > 1 and int(os.environ['LOCAL_RANK']) == 0) or len(args.gpus) == 1:
        print_log(args)
        print_log(f'CDR type: {args.cdr}')
        print_log(f'Paratope: {args.paratope}')
        print_log('structure only' if args.struct_only else 'sequence & structure codesign')

    plm_train = args.train_plm_embeddings if args.use_plm else None
    plm_valid = args.valid_plm_embeddings if args.use_plm else None
    train_set = E2EDataset(args.train_set, cdr=args.cdr, paratope=args.paratope,
                           plm_embeddings_path=plm_train)
    valid_set = E2EDataset(args.valid_set, cdr=args.cdr, paratope=args.paratope,
                           plm_embeddings_path=plm_valid)


    ########## set your collate_fn ##########
    collate_fn = train_set.collate_fn

    ########## define your model/trainer/trainconfig #########
    config = TrainConfig(**vars(args))

    effective_plm_dim = args.plm_dim if args.use_plm else 0

    if args.model_type == 'PRiME':
        from trainer import PRiMETrainer as Trainer
        from models import PRiMEModel
        model = PRiMEModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   iter_round=args.iter_round,
                   backbone_only=args.backbone_only,
                   fix_channel_weights=args.fix_channel_weights,
                   pred_edge_dist=not args.no_pred_edge_dist,
                   keep_memory=not args.no_memory,
                   cdr_type=args.cdr, paratope=args.paratope,
                   plm_dim=effective_plm_dim,
                   use_equivariant_adapter=args.use_equivariant_adapter,
                   adapter_rank=args.adapter_rank,
                   adapter_n_heads=args.adapter_n_heads,
                   use_riemannian_flow=args.use_riemannian_flow,
                   flow_weight=args.flow_weight,
                   flow_consistency_weight=args.flow_consistency_weight,
                   flow_rot_weight=args.flow_rot_weight,
                   flow_trans_weight=args.flow_trans_weight,
                   flow_n_steps=args.flow_n_steps,
                   flow_sigma_min=args.flow_sigma_min,
                   flow_time_mu=args.flow_time_mu,
                   flow_time_sigma=args.flow_time_sigma,
                   use_flow_iter=args.use_flow_iter,
                   flow_iter_alpha=args.flow_iter_alpha,
                   rotamer_weight=args.rotamer_weight,
                   rotamer_sigma_deg=args.rotamer_sigma_deg,
                   clash_weight=args.clash_weight,
                   clash_interface_weight=args.clash_interface_weight,
                   clash_dist=args.clash_dist,
                   clash_interface_dist=args.clash_interface_dist)
    elif args.model_type == 'PRiMEOpt':
        from trainer import PRiMEOptTrainer as Trainer
        from models import PRiMEOptModel
        model = PRiMEOptModel(args.embed_dim, args.hidden_size, VOCAB.MAX_ATOM_NUMBER,
                   VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
                   args.k_neighbors, bind_dist_cutoff=args.bind_dist_cutoff,
                   n_layers=args.n_layers, struct_only=args.struct_only,
                   fix_atom_weights=args.fix_channel_weights, cdr_type=args.cdr,
                   plm_dim=effective_plm_dim)
    else:
        raise NotImplemented(f'model {args.model_type} not implemented')

    step_per_epoch = (len(train_set) + args.batch_size - 1) // args.batch_size
    config.add_parameter(step_per_epoch=step_per_epoch)

    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=args.shuffle)
        args.batch_size = int(args.batch_size / len(args.gpus))
        if args.local_rank == 0:
            print_log(f'Batch size on a single GPU: {args.batch_size}')
    else:
        args.local_rank = -1
        train_sampler = None
    config.local_rank = args.local_rank

    if args.local_rank == 0 or args.local_rank == -1:
        print_log(f'step per epoch: {step_per_epoch}')

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=(args.shuffle and train_sampler is None),
                              sampler=train_sampler,
                              collate_fn=collate_fn)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              collate_fn=collate_fn)
    
    trainer = Trainer(model, train_loader, valid_loader, config)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args = parse()
    main(args)
