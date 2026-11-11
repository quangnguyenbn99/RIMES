#!/usr/bin/env python3
"""Prepare a per-target metrics CSV for the PLM-free (no_adapter) ablation.

The no_adapter ablation already has 60 targets * 40 samples of relaxed PDBs
under ``baseline_outputs/ablation_no_adapter/openmm_relaxed/<pdb>_s<N>_relaxed.pdb``,
but lacks the aggregated per-target metrics CSV that the leakage analysis
needs as input. This script:

  1. Reads chain assignments (H, L, A) from the PLM-augmented summary.json
     (one record per (pdb, sample_idx)).
  2. Builds a testset.jsonl pointing to the no_adapter relaxed PDBs and the
     same reference PDBs.
  3. Invokes cal_metrics_all.py with --skip_plm to compute AAR / RMSD / etc.
     and emit per_target_metrics.csv.

Usage:
    python -m scripts.prepare_no_adapter_per_target \
        --plm_aug_summary  checkpoints/esm2_3B_eqadapter_rflow_adaptive_time/version_13/checkpoint/results_best/summary.json \
        --no_adapter_dir   baseline_outputs/ablation_no_adapter/openmm_relaxed \
        --ref_pdb_dir      /home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/pdb \
        --out_dir          baseline_outputs/ablation_no_adapter/metrics
"""
import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plm_aug_summary', required=True,
                    help='summary.json from the PLM-augmented test run; supplies chain assignments.')
    ap.add_argument('--no_adapter_dir', required=True,
                    help='Folder with <pdb>_s<N>_relaxed.pdb files for the no_adapter ablation.')
    ap.add_argument('--ref_pdb_dir', required=True,
                    help='Folder with reference RAbD PDBs (one per pdb_id).')
    ap.add_argument('--out_dir', required=True,
                    help='Where to write testset.jsonl and metrics outputs.')
    ap.add_argument('--num_workers', type=int, default=32)
    ap.add_argument('--skip_metrics_run', action='store_true',
                    help='Build testset.jsonl only; do not call cal_metrics_all.py.')
    ap.add_argument('--cal_metrics_script', default='cal_metrics_all.py')
    ap.add_argument('--conda_env', default=None,
                    help='If set, run via "conda run -n <env> python ..."')
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build assignment dict from PLM-aug summary -----------------------
    assignments = {}
    with open(args.plm_aug_summary) as fh:
        for line in fh:
            rec = json.loads(line)
            pdb = rec['pdb'].lower()
            A = rec['A']
            if isinstance(A, list):
                A = ''.join(A)
            assignments[pdb] = (rec['H'], rec['L'], A, rec.get('cdr_type', ['H3']))

    print(f'Loaded chain assignments for {len(assignments)} targets.')

    # ---- Build testset.jsonl pointing to no_adapter PDBs ------------------
    no_dir = Path(args.no_adapter_dir)
    ref_dir = Path(args.ref_pdb_dir)
    testset_path = out_dir / 'no_adapter_testset.jsonl'

    pdb_to_samples = defaultdict(list)
    for sample_pdb in sorted(no_dir.glob('*_s*_relaxed.pdb')):
        bn = sample_pdb.name.replace('_relaxed.pdb', '')
        if '_original' in bn:
            continue
        try:
            pdb_id, sample_tag = bn.split('_s', 1)
            sample_idx = int(sample_tag)
        except ValueError:
            continue
        pdb_to_samples[pdb_id.lower()].append((sample_idx, sample_pdb))

    n_records, n_skipped = 0, 0
    with open(testset_path, 'w') as fh:
        for pdb_id, samples in sorted(pdb_to_samples.items()):
            if pdb_id not in assignments:
                n_skipped += len(samples)
                continue
            H, L, A, cdr_type = assignments[pdb_id]
            ref_pdb = ref_dir / f'{pdb_id}.pdb'
            if not ref_pdb.exists():
                print(f'  [skip] {pdb_id}: no reference at {ref_pdb}')
                n_skipped += len(samples)
                continue
            for sample_idx, sample_pdb in sorted(samples):
                rec = {
                    'mod_pdb': str(sample_pdb.resolve()),
                    'ref_pdb': str(ref_pdb.resolve()),
                    'H': H, 'L': L, 'A': A,
                    'cdr_type': cdr_type,
                    'pdb': pdb_id,
                    'pmetric': 0.0,
                    'sample_idx': sample_idx,
                }
                fh.write(json.dumps(rec) + '\n')
                n_records += 1

    print(f'Wrote {testset_path}  ({n_records} samples, {n_skipped} skipped)')

    if args.skip_metrics_run:
        return

    # ---- Run cal_metrics_all.py ------------------------------------------
    save_dir = out_dir / 'no_adapter_metrics'
    save_dir.mkdir(parents=True, exist_ok=True)

    cmd_prefix = []
    if args.conda_env:
        cmd_prefix = ['conda', 'run', '-n', args.conda_env]
    cmd = cmd_prefix + [
        'python', args.cal_metrics_script,
        '--test_set', str(testset_path),
        '--save_dir', str(save_dir),
        '--num_workers', str(args.num_workers),
        '--skip_plm',
        '--numbering', 'imgt',
    ]
    print('Running:', ' '.join(cmd))
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f'cal_metrics_all.py exited with code {rc}')
        sys.exit(rc)

    per_target = save_dir / 'metrics' / 'per_target_metrics.csv'
    if per_target.exists():
        print(f'\nDone. Per-target metrics at: {per_target}')
        print('Now run scripts/analyze_plm_leakage.py with this CSV as --plm_free_csv.')
    else:
        print(f'\nWARNING: expected output not found at {per_target}')


if __name__ == '__main__':
    main()
