#!/usr/bin/env python3
"""Option-A ΔΔG re-ranking: penalise scoring failures as +∞ ΔΔG.

For each (method, experiment), reads the per-sample ddG CSVs, counts:
  - total samples attempted (60*40 = 2400 for RAbD/multi, 754*40 = 30160 for time)
  - total samples scored
  - %<0 over scored (the value typically reported)
  - %<0 over attempted  (Option A: missing samples count as +∞)
  - similar for %<-5
  - median ddG over scored

Two CSV schemas are handled:
  A. pdb_id, sample_file, original_file, interface, dG_sample_relaxed,
     dG_original, ddG, error
  B. dG_separated, initial_score, interface, pdb_file
     (no precomputed ddG: extract pdb_id from pdb_file, use the
     <pdb_id>_original_relaxed row as reference)

Output: a single CSV summarising completion rates and the corrected
percentages, plus a markdown table per experiment for easy paper
inclusion.
"""
from __future__ import annotations

import argparse
import os
import re
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = {
    'RAbD CDR-H3 (60x40)': {
        'attempted': 60 * 40,
        'methods': {
            'dyMEAN':           ['baseline_outputs/dyMEAN_singlecdr/ddG_results.csv'],
            'dyAb':             ['baseline_outputs/dyAb_singlecdr/ddG_results.csv'],
            'AbEgDiffuser':     ['baseline_outputs/abegdiffuser_singlecdr/ddG_results.csv'],
            'FlowDesign':       ['baseline_outputs/FlowDesign_singlecdr/ddG_results.csv'],
            'AbX':              ['baseline_outputs/AbX_singlecdr/ddG_results.csv'],
            'AbMEGD':           ['baseline_outputs/abmegd_singlecdr/ddG_results.csv'],
            'RiME (full)':      ['checkpoints/esm2_3B_eqadapter_rflow_adaptive_time/version_13/checkpoint/ddG_results.csv'],
            'RiME w/o adapter': ['baseline_outputs/ablation_no_adapter/ddG_results_part*.csv'],
        },
    },
    'Multi-CDR (60x40)': {
        'attempted': 60 * 40,
        'methods': {
            'dyMEAN':           ['baseline_outputs/dyMEAN_multicdr/ddG_results.csv'],
            'dyAb':             ['baseline_outputs/dyAb_multicdr/ddG_results.csv'],
            'AbEgDiffuser':     ['baseline_outputs/abegdiffuser_multicdr/ddG_results.csv'],
            'FlowDesign':       ['baseline_outputs/FlowDesign_multicdr/ddG_results.csv'],
            'AbX':              ['baseline_outputs/AbX_multicdr/ddG_results.csv'],
            'AbMEGD':           ['baseline_outputs/abmegd_multicdr/ddG_results.csv'],
            'RiME (full)':      ['checkpoints/esm2_3B_eqadapter_rflow_adaptive_time_multi/version_1/checkpoint/ddG_results.csv'],
            'RiME w/o adapter': ['baseline_outputs/multi_cdr_ablation_no_adapter/ddG_results_part*.csv'],
        },
    },
    'Time split (754x40)': {
        'attempted': 754 * 40,
        'methods': {
            'dyMEAN':           ['baseline_outputs/dyMEAN_time/ddG_results_part*.csv'],
            'dyAb':             ['baseline_outputs/dyAb_time/ddG_results_part*.csv'],
            'AbEgDiffuser':     ['baseline_outputs/AbEgDiffuser_time/ddG_results_part*.csv'],
            'FlowDesign':       ['baseline_outputs/FlowDesign_time/ddG_results_part*.csv'],
            'AbX':              ['baseline_outputs/AbX_time/ddG_results_part*.csv'],
            # AbMEGD time-split not generated
            'RiME (full)':      ['checkpoints/esm2_3B_eqadapter_rflow_adaptive_time_time/version_2/checkpoint/ddG_results_part*.csv'],
            # RiME w/o adapter time-split not yet scored
        },
    },
}

PDB_PREFIX_RE = re.compile(r'^(?P<pdb>[A-Za-z0-9]{4})_')
SAMPLE_IDX_RE = re.compile(r'_s(?P<idx>\d+)_relaxed')


def _parse_pdb_file(name):
    """Return (pdb_id_lower, kind) where kind is 'original', int(sample_idx) or None."""
    name = str(name)
    pm = PDB_PREFIX_RE.match(name)
    if not pm:
        return None, None
    pdb = pm.group('pdb').lower()
    if 'original' in name:
        return pdb, 'original'
    sm = SAMPLE_IDX_RE.search(name)
    if sm:
        return pdb, int(sm.group('idx'))
    return pdb, None


def _resolve(patterns):
    files = []
    for p in patterns:
        full = str(ROOT / p)
        if any(c in full for c in '*?['):
            files.extend(sorted(glob(full)))
        elif os.path.exists(full):
            files.append(full)
    return files


def _read_schema_a(paths):
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if 'ddG' not in d.columns:
            continue
        d = d.copy()
        d['_path'] = p
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    # Drop duplicate (pdb_id, sample_file) entries (parts may overlap with concat)
    if 'sample_file' in df.columns and 'pdb_id' in df.columns:
        df = df.drop_duplicates(subset=['pdb_id', 'sample_file'], keep='first')
    return df


def _read_schema_b(paths, external_ref_dg=None):
    """ablation_no_adapter / abx_multi style: derive ddG from raw dG_separated.

    external_ref_dg: optional {pdb_id_lower: dG_original} dict to use when
    the CSVs lack <pdb>_original rows (the reference is method-independent).
    """
    dfs = []
    for p in paths:
        try:
            d = pd.read_csv(p)
        except Exception:
            continue
        if 'pdb_file' not in d.columns or 'dG_separated' not in d.columns:
            continue
        dfs.append(d)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=['pdb_file'], keep='first')

    # Build reference: prefer in-CSV originals, else fall back to external map
    ref_dg = dict(external_ref_dg or {})
    for _, row in df.iterrows():
        pdb, kind = _parse_pdb_file(row['pdb_file'])
        if kind == 'original' and pd.notna(row['dG_separated']):
            ref_dg[pdb] = row['dG_separated']

    rows = []
    for _, row in df.iterrows():
        pdb, kind = _parse_pdb_file(row['pdb_file'])
        if kind == 'original' or kind is None:
            continue
        if pdb not in ref_dg:
            continue
        rows.append({
            'pdb_id': pdb,
            'sample_file': row['pdb_file'],
            'dG_sample_relaxed': row['dG_separated'],
            'dG_original': ref_dg[pdb],
            'ddG': row['dG_separated'] - ref_dg[pdb] if pd.notna(row['dG_separated']) else float('nan'),
            'error': '' if pd.notna(row['dG_separated']) else 'nan',
        })
    return pd.DataFrame(rows)


def build_external_ref_dg(schema_a_paths_per_method):
    """From any number of Schema-A CSVs (which carry dG_original per row),
    build a {pdb_id_lower: dG_original} reference map. Median across methods
    handles small re-relax differences."""
    pdb_to_dgs = {}
    for paths in schema_a_paths_per_method:
        files = _resolve(paths)
        for fp in files:
            try:
                d = pd.read_csv(fp)
            except Exception:
                continue
            if 'pdb_id' not in d.columns or 'dG_original' not in d.columns:
                continue
            for _, row in d.iterrows():
                if pd.isna(row.get('dG_original')):
                    continue
                pid = str(row['pdb_id']).lower()
                pdb_to_dgs.setdefault(pid, []).append(float(row['dG_original']))
    return {pid: float(np.median(v)) for pid, v in pdb_to_dgs.items()}


def load_method(paths, external_ref_dg=None):
    paths = _resolve(paths)
    if not paths:
        return pd.DataFrame()
    df = _read_schema_a(paths)
    if df.empty:
        df = _read_schema_b(paths, external_ref_dg=external_ref_dg)
    return df


def aggregate(df, attempted, samples_per_target=40):
    if df.empty:
        return None
    n_in_csv = len(df)
    valid = df[pd.to_numeric(df.get('ddG'), errors='coerce').notna()].copy()
    n_scored = len(valid)
    if n_scored == 0:
        return None
    valid['ddG'] = valid['ddG'].astype(float)
    ddg = valid['ddG']

    # Per-target proportions for Option-A reporting (mean +- std across targets).
    # For each pdb_id, count successes in {<0} and {<-5} divided by samples_per_target
    # (so unscored samples contribute 0 successes -- the +inf imputation).
    per_pdb_lt0, per_pdb_ltn5 = [], []
    if 'pdb_id' in valid.columns:
        grouped = valid.groupby('pdb_id')['ddG']
        for _, vals in grouped:
            denom = samples_per_target
            per_pdb_lt0.append((vals < 0).sum() / denom)
            per_pdb_ltn5.append((vals < -5).sum() / denom)
    n_targets_with_any_score = len(per_pdb_lt0)
    n_targets_total = attempted // samples_per_target
    # Targets that had zero scored samples contribute 0 to mean as well.
    pad = max(0, n_targets_total - n_targets_with_any_score)
    per_pdb_lt0_padded = per_pdb_lt0 + [0.0] * pad
    per_pdb_ltn5_padded = per_pdb_ltn5 + [0.0] * pad

    summary = {
        'n_attempted': attempted,
        'n_in_csv': n_in_csv,
        'n_scored': n_scored,
        'completion_pct': 100.0 * n_scored / attempted,
        'median_ddg_scored': float(np.median(ddg)),
        'mean_ddg_scored': float(np.mean(ddg)),
        'pct_below_0_scored':       100.0 * (ddg < 0).sum() / n_scored,
        'pct_below_neg5_scored':    100.0 * (ddg < -5).sum() / n_scored,
        # Option A: missing -> +inf, so they don't satisfy ddG<0 / ddG<-5
        'pct_below_0_attempted':    100.0 * (ddg < 0).sum() / attempted,
        'pct_below_neg5_attempted': 100.0 * (ddg < -5).sum() / attempted,
        # Per-target Option-A stats (mean +- std across targets)
        'pct_below_0_per_target_mean':    100.0 * float(np.mean(per_pdb_lt0_padded)) if per_pdb_lt0_padded else float('nan'),
        'pct_below_0_per_target_std':     100.0 * float(np.std(per_pdb_lt0_padded)) if per_pdb_lt0_padded else float('nan'),
        'pct_below_neg5_per_target_mean': 100.0 * float(np.mean(per_pdb_ltn5_padded)) if per_pdb_ltn5_padded else float('nan'),
        'pct_below_neg5_per_target_std':  100.0 * float(np.std(per_pdb_ltn5_padded)) if per_pdb_ltn5_padded else float('nan'),
    }
    return summary


def render_md(rows, title, attempted):
    print(f'\n### {title} (attempted = {attempted})\n')
    cols = ['Method', 'completion%', 'median ΔΔG (scored)',
            '%<0 honest (overall)',
            '%<0 honest (per-target mean ± std)',
            '%<-5 honest (per-target mean ± std)']
    print('| ' + ' | '.join(cols) + ' |')
    print('|' + '|'.join(['---'] * len(cols)) + '|')
    for r in rows:
        print(('| {method} | {comp:.1f}% | {med:+.2f} | {p0a:.1f}% | '
               '{p0t_m:.1f} ± {p0t_s:.1f} | {p5t_m:.1f} ± {p5t_s:.1f} |').format(
            method=r['method'],
            comp=r['completion_pct'],
            med=r['median_ddg_scored'],
            p0a=r['pct_below_0_attempted'],
            p0t_m=r['pct_below_0_per_target_mean'],
            p0t_s=r['pct_below_0_per_target_std'],
            p5t_m=r['pct_below_neg5_per_target_mean'],
            p5t_s=r['pct_below_neg5_per_target_std'],
        ))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_csv', default='ddg_completion_corrected.csv')
    args = ap.parse_args()

    all_rows = []
    for exp_name, exp_cfg in EXPERIMENTS.items():
        attempted = exp_cfg['attempted']
        # Build a per-experiment external reference map by harvesting
        # dG_original from any Schema-A CSV the experiment ships.
        external_ref = build_external_ref_dg(
            list(exp_cfg['methods'].values())
        )
        rows_for_table = []
        for method, paths in exp_cfg['methods'].items():
            df = load_method(paths, external_ref_dg=external_ref)
            agg = aggregate(df, attempted)
            if agg is None:
                print(f'  [warn] {exp_name} | {method}: no data')
                continue
            agg['experiment'] = exp_name
            agg['method'] = method
            all_rows.append(agg)
            rows_for_table.append(agg)
        render_md(rows_for_table, exp_name, attempted)

    if all_rows:
        out = pd.DataFrame(all_rows)[
            ['experiment', 'method', 'n_attempted', 'n_scored', 'completion_pct',
             'median_ddg_scored', 'mean_ddg_scored',
             'pct_below_0_scored', 'pct_below_0_attempted',
             'pct_below_neg5_scored', 'pct_below_neg5_attempted',
             'pct_below_0_per_target_mean', 'pct_below_0_per_target_std',
             'pct_below_neg5_per_target_mean', 'pct_below_neg5_per_target_std']
        ]
        out.to_csv(args.out_csv, index=False)
        print(f'\nWrote {args.out_csv}')


if __name__ == '__main__':
    main()
