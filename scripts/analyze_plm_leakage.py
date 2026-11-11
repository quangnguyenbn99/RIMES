#!/usr/bin/env python3
"""PLM-leakage analysis for the RAbD CDR-H3 benchmark.

Reviewer concern: ``the 72.7% AAR is partially memorisation by ESM-2 because
the test antibody heavy chains were almost certainly in UniRef50 (ESM-2's
pre-training corpus).''

This script answers it by computing ESM-2-3B per-residue masked perplexity
on each test heavy chain, then bucketing the per-target PLM lift
(PLM-augmented AAR - PLM-free AAR) by H3 perplexity quartile. If the lift
is invariant across buckets, the PLM signal is a genuine evolutionary prior;
if it concentrates in low-perplexity (i.e. ESM-2-familiar) buckets, the lift
is partially memorisation and we report it openly.

Outputs:
    rabd_h3_perplexity.csv             -- per-target raw perplexities
    rabd_plm_leakage_table.csv         -- bucketed AAR comparison
    rabd_plm_leakage_table.tex         -- LaTeX-ready table

Requires:
    - PLM-aug per-target metrics CSV       (already on disk for RiME)
    - PLM-free per-target metrics CSV      (run cal_metrics_all.py on the
                                            no_adapter PDBs once, see README)
    - REF_PDB_DIR with one reference PDB per test target.

Usage:
    python -m scripts.analyze_plm_leakage \
        --plm_aug_csv  checkpoints/.../per_target_metrics.csv \
        --plm_free_csv .../no_adapter_per_target_metrics.csv \
        --ref_pdb_dir  /home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/pdb \
        --output_dir   leakage_analysis
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pdb_utils import AgAbComplex, VOCAB

VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')


# --- 60 RAbD targets and their canonical chain assignments -----------------
RAbD_PDB = [
    '1a14', '1a2y', '1fe8', '1ic7', '1iqd', '1n8z', '1ncb', '1osp', '1uj3',
    '1w72', '2adf', '2b2x', '2cmr', '2dd8', '2ghw', '2vxt', '2xqy', '2xwt',
    '2ypv', '3bn9', '3cx5', '3ffd', '3h3b', '3hi6', '3k2u', '3l95', '3mxw',
    '3nid', '3o2d', '3rkd', '3s35', '3uzq', '3w9e', '4cmh', '4dtg', '4dvr',
    '4etq', '4ffv', '4fqj', '4g6j', '4g6m', '4h8w', '4ki5', '4lvn', '4ot1',
    '4qci', '4xnq', '4ydk', '5b8c', '5bv7', '5d93', '5d96', '5en2', '5f9o',
    '5ggs', '5hi4', '5j13', '5l6y', '5mes', '5nuz',
]


def _sanitize_seq(symbols):
    return ''.join(s if s in VALID_AA else 'X' for s in symbols)


def _load_chain_assignments(rabd_summary_jsonl):
    """Return {pdb: (H_chain, L_chain, A_chain_str)}.
    Falls back to scanning a summary.json from any RAbD test run."""
    assignments = {}
    if rabd_summary_jsonl and os.path.exists(rabd_summary_jsonl):
        with open(rabd_summary_jsonl) as f:
            for line in f:
                item = json.loads(line)
                pdb = item['pdb'].lower()
                A = item['A']
                if isinstance(A, list):
                    A = ''.join(A)
                assignments[pdb] = (item['H'], item['L'], A)
    return assignments


def _build_complex(pdb_id, ref_pdb_dir, assignments):
    pdb_path = os.path.join(ref_pdb_dir, f'{pdb_id}.pdb')
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f'Reference PDB not found: {pdb_path}')
    if pdb_id not in assignments:
        # Fallback: assume H/L/A common-case naming. RAbD typically uses H, L, then antigen.
        H, L, A = 'H', 'L', 'A'
    else:
        H, L, A = assignments[pdb_id]
    A_chains = list(A) if A else []
    cplx = AgAbComplex.from_pdb(pdb_path, H, L, A_chains)
    return cplx


def compute_h3_perplexity(model, alphabet, device, hc_seq, h3_start, h3_end):
    """Return (mean H3 NLL, mean framework NLL, mean overall NLL) under ESM-2."""
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("h", hc_seq)])
    tokens = tokens.to(device)

    with torch.no_grad():
        logits = model(tokens)["logits"][0, 1:-1]   # drop BOS/EOS
    log_probs = torch.log_softmax(logits, dim=-1)
    aa_idx = tokens[0, 1:-1]
    per_res_nll = -log_probs.gather(1, aa_idx.unsqueeze(1)).squeeze(1).cpu().numpy()

    h3_mask = np.zeros(len(per_res_nll), dtype=bool)
    h3_mask[h3_start:h3_end + 1] = True
    fw_mask = ~h3_mask

    return (
        float(per_res_nll[h3_mask].mean()) if h3_mask.any() else float('nan'),
        float(per_res_nll[fw_mask].mean()) if fw_mask.any() else float('nan'),
        float(per_res_nll.mean()),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plm_aug_csv', required=True,
                    help='Per-target metrics CSV from the PLM-augmented (RiME⋆) checkpoint.')
    ap.add_argument('--plm_free_csv', required=True,
                    help='Per-target metrics CSV from the PLM-free (no_adapter) checkpoint.')
    ap.add_argument('--ref_pdb_dir', required=True,
                    help='Directory with reference RAbD PDBs (one per pdb_id).')
    ap.add_argument('--rabd_summary_jsonl', default=None,
                    help='Optional summary.json{,l} from a RAbD test run, used to '
                         'recover (H, L, antigen) chain assignments per target.')
    ap.add_argument('--esm_model', default='esm2_t36_3B_UR50D')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--output_dir', default='leakage_analysis')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- 1. Load ESM-2-3B ------------------------------------------------
    print(f'Loading {args.esm_model} on {args.device} ...')
    import esm as esm_lib
    loader = getattr(esm_lib.pretrained, args.esm_model)
    model, alphabet = loader()
    model = model.eval().to(args.device)

    # ---- 2. Per-target perplexity ---------------------------------------
    assignments = _load_chain_assignments(args.rabd_summary_jsonl)
    rows = []
    for pdb_id in tqdm(RAbD_PDB, desc='ESM-2 perplexity'):
        try:
            cplx = _build_complex(pdb_id, args.ref_pdb_dir, assignments)
            hc = cplx.get_heavy_chain()
            hc_seq = _sanitize_seq([hc.get_residue(i).get_symbol() for i in range(len(hc))])
            h3_start, h3_end = cplx.get_cdr_pos('H3')
            ppl_h3, ppl_fw, ppl_full = compute_h3_perplexity(
                model, alphabet, args.device, hc_seq, h3_start, h3_end)
            rows.append({
                'pdb': pdb_id,
                'h3_len': h3_end - h3_start + 1,
                'hc_len': len(hc_seq),
                'nll_h3': ppl_h3,
                'nll_framework': ppl_fw,
                'nll_full': ppl_full,
                'ppl_h3': float(np.exp(ppl_h3)),
                'ppl_framework': float(np.exp(ppl_fw)),
                'ppl_full': float(np.exp(ppl_full)),
            })
        except Exception as e:
            print(f'  [skip] {pdb_id}: {e}')

    ppl_df = pd.DataFrame(rows)
    ppl_path = os.path.join(args.output_dir, 'rabd_h3_perplexity.csv')
    ppl_df.to_csv(ppl_path, index=False)
    print(f'Wrote {ppl_path}  (n={len(ppl_df)})')

    # ---- 3. Join with per-target AAR -----------------------------------
    plm_aug = pd.read_csv(args.plm_aug_csv)
    plm_free = pd.read_csv(args.plm_free_csv)

    # Auto-pick AAR-H3 column name across slightly-different schemas.
    def _aar_col(df):
        for c in ['AAR H3_mean', 'AAR_H3_mean', 'AAR_mean', 'AAR H3', 'aar_h3']:
            if c in df.columns:
                return c
        raise KeyError(f'No AAR column found in {df.columns.tolist()[:8]}...')

    aug_col = _aar_col(plm_aug)
    free_col = _aar_col(plm_free)
    plm_aug = plm_aug[['pdb', aug_col]].rename(columns={aug_col: 'aar_plm_aug'})
    plm_free = plm_free[['pdb', free_col]].rename(columns={free_col: 'aar_plm_free'})

    df = ppl_df.merge(plm_aug, on='pdb').merge(plm_free, on='pdb')
    df['plm_lift'] = df['aar_plm_aug'] - df['aar_plm_free']

    # ---- 4. Quartile bucketing on H3 perplexity -------------------------
    df['ppl_h3_bucket'] = pd.qcut(
        df['ppl_h3'], q=4,
        labels=['Q1: lowest ppl (most ESM-2-familiar)',
                'Q2', 'Q3',
                'Q4: highest ppl (least ESM-2-familiar)'])

    table = df.groupby('ppl_h3_bucket', observed=True).agg(
        n=('pdb', 'size'),
        ppl_h3_min=('ppl_h3', 'min'),
        ppl_h3_max=('ppl_h3', 'max'),
        aar_plm_aug=('aar_plm_aug', 'mean'),
        aar_plm_aug_std=('aar_plm_aug', 'std'),
        aar_plm_free=('aar_plm_free', 'mean'),
        aar_plm_free_std=('aar_plm_free', 'std'),
        plm_lift=('plm_lift', 'mean'),
        plm_lift_std=('plm_lift', 'std'),
    ).round(3)

    table_path = os.path.join(args.output_dir, 'rabd_plm_leakage_table.csv')
    table.to_csv(table_path)
    print('\n=== H3-perplexity bucketed PLM lift ===')
    print(table.to_string())
    print(f'\nWrote {table_path}')

    # ---- 5. LaTeX -------------------------------------------------------
    tex = (
        '\\begin{table}[h]\n'
        '\\centering\\small\n'
        '\\caption{RAbD CDR-H3 amino-acid recovery bucketed by '
        'ESM-2-3B masked perplexity on the native H3 sequence. Q1 contains '
        'targets ESM-2 finds most familiar (lowest perplexity); Q4 contains '
        'targets ESM-2 finds least familiar. A roughly constant PLM lift '
        '(PLM-aug $-$ PLM-free) across buckets indicates the PLM signal is a '
        'genuine evolutionary prior rather than memorisation.}\n'
        '\\label{tab:plm_leakage}\n'
    )
    tex += table.to_latex(float_format='%.3f', column_format='lcccccc')
    tex += '\n\\end{table}\n'
    tex_path = os.path.join(args.output_dir, 'rabd_plm_leakage_table.tex')
    with open(tex_path, 'w') as fh:
        fh.write(tex)
    print(f'Wrote {tex_path}')

    # ---- 6. Headline diagnostic ----------------------------------------
    overall_lift = df['plm_lift'].mean()
    q1_lift, q4_lift = table['plm_lift'].iloc[0], table['plm_lift'].iloc[-1]
    print(f'\nOverall mean PLM lift on RAbD-60: {overall_lift:+.3f}')
    print(f'  Q1 (most familiar) PLM lift:    {q1_lift:+.3f}')
    print(f'  Q4 (least familiar) PLM lift:   {q4_lift:+.3f}')
    print(f'  Q1 - Q4 differential:           {(q1_lift - q4_lift):+.3f}')
    if abs(q1_lift - q4_lift) < 0.03:
        print('  -> PLM lift is roughly invariant; defends architecture story.')
    elif q1_lift > q4_lift + 0.05:
        print('  -> PLM lift concentrates in familiar bucket; partial memorisation.')
    else:
        print('  -> Intermediate regime; report and discuss.')


if __name__ == '__main__':
    main()
