#!/usr/bin/env python3
"""
Standalone script to evaluate a baseline output directory using the PRiME
metric pipeline.  Handles FlowDesign, AbEgDiffuser, AbMEGD and similar
formats (each target dir contains a H_CDR3/ subdir of numbered PDBs).

Usage:
    python /tmp/eval_baseline.py \
        --workspace  <path to PRiME project root> \
        --baseline   <path to baseline result folder, e.g. FlowDesign_singlecdr_RAbD> \
        --test_json  <path to RAbD test.json> \
        --save_dir   <output directory> \
        [--num_workers 8]
"""

import argparse
import json
import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import numpy as np


# ─────────────────────────────── helpers ────────────────────────────────────

def load_test_map(test_json_path):
    test_map = {}
    with open(test_json_path) as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                test_map[item["pdb"]] = item
    return test_map


def normalize_cdr(cdr_str):
    """H_CDR3 -> H3, L_CDR1 -> L1, etc."""
    return cdr_str.replace("_CDR", "").replace("CDR_", "")


def available_chains_in_pdb(pdb_path):
    chains = set()
    try:
        with open(pdb_path) as fh:
            for line in fh:
                if line.startswith(("ATOM", "HETATM")) and len(line) > 21:
                    chains.add(line[21].strip())
    except OSError:
        pass
    return chains


def build_entry_list(baseline_dir, test_map):
    """Walk baseline output folder and build list of metric inputs."""
    base = Path(baseline_dir)
    entries = []

    for metadata_path in sorted(base.rglob("metadata.json")):
        case_dir = metadata_path.parent
        with open(metadata_path) as f:
            meta = json.load(f)

        identifier = meta["identifier"]
        pdb_id = identifier.split("_")[0]
        if pdb_id not in test_map:
            continue
        ti = test_map[pdb_id]

        # prefer case-level reference.pdb, else fall back to ground-truth PDB
        ref_candidates = [
            case_dir / "reference.pdb",
            Path(ti["pdb_data_path"]),
        ]
        ref_pdb = next((str(c) for c in ref_candidates if c.exists()), None)
        if ref_pdb is None:
            print(f"[skip] no ref PDB for {pdb_id}")
            continue

        for sub in meta.get("items", []):
            tag = sub["tag"]                      # e.g. "H_CDR3"
            cdr_type = normalize_cdr(sub["cdr"])  # e.g. "H3"
            pred_dir = case_dir / tag
            if not pred_dir.exists():
                continue

            for mod_pdb in sorted(pred_dir.glob("[0-9][0-9][0-9][0-9].pdb")):
                entries.append({
                    "mod_pdb":  str(mod_pdb),
                    "ref_pdb":  ref_pdb,
                    "H":        ti["heavy_chain"],
                    "L":        ti["light_chain"],
                    "A":        ti["antigen_chains"],
                    "cdr_type": cdr_type,
                    "pdb":      pdb_id,
                    "pmetric":  0.0,
                })
    return entries


# ─────────────────────────────── workspace setup ────────────────────────────

_WORKSPACE = None
_TEST_MAP  = None


def _init_worker(workspace, test_json):
    global _WORKSPACE, _TEST_MAP
    _WORKSPACE = workspace
    sys.path.insert(0, workspace)
    _TEST_MAP = load_test_map(test_json)


# ─────────────────────────────── metric function ────────────────────────────

def make_cdr_pos_from_test_item(ti):
    cdr_pos = {}
    for key, field in [
        ("CDR-H1", "cdrh1_pos"), ("CDR-H2", "cdrh2_pos"), ("CDR-H3", "cdrh3_pos"),
        ("CDR-L1", "cdrl1_pos"), ("CDR-L2", "cdrl2_pos"), ("CDR-L3", "cdrl3_pos"),
    ]:
        v = ti.get(field)
        if v is not None:
            cdr_pos[key] = tuple(v)
    return cdr_pos


def _load_cplx(pdb_path, H, L, A, pdb_id, skip_epitope, need_validity):
    from data.pdb_utils import AgAbComplex
    global _TEST_MAP

    for attempt in range(3):
        try:
            if attempt < 2:
                return AgAbComplex.from_pdb(
                    pdb_path, H, L, A,
                    skip_epitope_cal=skip_epitope,
                    skip_validity_check=(attempt == 1),
                )
            else:
                # Filter antigen chains to those physically present
                present = available_chains_in_pdb(pdb_path)
                A_ok = [c for c in A if c in present] or A
                cplx = AgAbComplex.from_pdb(
                    pdb_path, H, L, A_ok,
                    skip_epitope_cal=True,
                    skip_validity_check=True,
                )
                ti = _TEST_MAP[pdb_id]
                cplx.cdr_pos = make_cdr_pos_from_test_item(ti)
                if not skip_epitope:
                    try:
                        cplx.epitope = cplx._cal_epitope()
                    except Exception:
                        cplx.epitope = []
                return cplx
        except Exception:
            if attempt == 2:
                raise


def cal_metrics_single(entry):
    """Compute all metrics for one (mod_pdb, ref_pdb) pair."""
    import numpy as np
    from evaluation.rmsd import compute_rmsd
    from evaluation.tm_score import tm_score
    from evaluation.lddt import lddt
    from evaluation.dockq import dockq
    from configs import CONTACT_DIST

    mod_pdb  = entry["mod_pdb"]
    ref_pdb  = entry["ref_pdb"]
    H, L, A  = entry["H"], entry["L"], entry["A"]
    cdr_type = entry["cdr_type"]
    pdb_id   = entry["pdb"]

    try:
        ref_cplx = _load_cplx(ref_pdb, H, L, A, pdb_id, skip_epitope=False, need_validity=True)
    except Exception as e:
        return None, None

    try:
        mod_cplx = _load_cplx(mod_pdb, H, L, A, pdb_id, skip_epitope=True, need_validity=False)
    except Exception as e:
        return None, None

    results = {}
    cdr_list = [cdr_type] if isinstance(cdr_type, str) else cdr_type

    # AAR / CAAR
    epitope = ref_cplx.get_epitope()
    gt_s, pred_s = "", ""
    is_contact = []
    for cdr in cdr_list:
        gt_cdr   = ref_cplx.get_cdr(cdr)
        pred_cdr = mod_cplx.get_cdr(cdr)
        if gt_cdr is None or pred_cdr is None:
            continue
        cur_gt   = gt_cdr.get_seq()
        cur_pred = pred_cdr.get_seq()
        gt_s   += cur_gt
        pred_s += cur_pred
        cur_contact = []
        for ab_res in gt_cdr:
            contact = any(ab_res.dist_to(ag_res) < CONTACT_DIST for ag_res, _, _ in epitope)
            cur_contact.append(int(contact))
        is_contact.extend(cur_contact)
        hit  = sum(a == b for a, b in zip(cur_gt, cur_pred))
        chit = sum(a == b for a, b, c in zip(cur_gt, cur_pred, cur_contact) if c)
        results[f"AAR {cdr}"]  = hit / max(len(cur_gt), 1)
        results[f"CAAR {cdr}"] = chit / (sum(cur_contact) + 1e-10)

    if not gt_s or len(gt_s) != len(pred_s):
        return None, None

    hit  = sum(a == b for a, b in zip(gt_s, pred_s))
    chit = sum(a == b for a, b, c in zip(gt_s, pred_s, is_contact) if c)
    results["AAR"]  = hit / len(gt_s)
    results["CAAR"] = chit / (sum(is_contact) + 1e-10)

    # RMSD(CA) – full antibody (Kabsch-aligned)
    gt_x, pred_x = [], []
    for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
        for chain in [c.get_heavy_chain(), c.get_light_chain()]:
            for i in range(len(chain)):
                xl.append(chain.get_ca_pos(i))
    if len(gt_x) == len(pred_x) and len(gt_x) > 0:
        gt_arr, pred_arr = np.array(gt_x), np.array(pred_x)
        results["RMSD(CA) aligned"] = compute_rmsd(gt_arr, pred_arr, aligned=False)

    # RMSD per CDR
    for cdr in cdr_list:
        gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
        if gt_cdr is None or pred_cdr is None:
            continue
        gt_x2   = np.array([gt_cdr.get_ca_pos(i)   for i in range(len(gt_cdr))])
        pred_x2 = np.array([pred_cdr.get_ca_pos(i)  for i in range(len(pred_cdr))])
        if len(gt_x2) == len(pred_x2) and len(gt_x2) > 0:
            results[f"RMSD(CA) CDR{cdr}"]         = compute_rmsd(gt_x2, pred_x2, aligned=True)
            results[f"RMSD(CA) CDR{cdr} aligned"] = compute_rmsd(gt_x2, pred_x2, aligned=False)

    # TMscore
    try:
        results["TMscore"] = tm_score(mod_cplx.antibody, ref_cplx.antibody)
    except Exception:
        results["TMscore"] = float("nan")

    # LDDT
    try:
        score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
        results["LDDT"] = score
    except Exception:
        results["LDDT"] = float("nan")

    # DockQ
    try:
        results["DockQ"] = dockq(mod_cplx, ref_cplx, cdrh3_only=False)
    except Exception:
        results["DockQ"] = float("nan")

    return results, pred_s


def _worker(entry):
    return cal_metrics_single(entry)


# ─────────────────────────────── aggregation ────────────────────────────────

def aggregate(entries, raw_results, save_dir):
    metrics_list = [r[0] for r in raw_results]
    pred_seqs    = [r[1] for r in raw_results]
    pdbs         = [e["pdb"] for e in entries]
    pmets        = [e["pmetric"] for e in entries]

    # Drop failed entries but keep structure for CSV
    first = next((m for m in metrics_list if m is not None and m), None)
    if first is None:
        raise RuntimeError("No metrics computed successfully.")

    pmets_np = np.array(pmets, dtype=float)
    summary  = {}
    for name in first:
        vals = [m[name] if m is not None else float("nan") for m in metrics_list]
        vals_np = np.array(vals, dtype=float)
        mean_val = float(np.nanmean(vals_np))
        print(f"{name}: {mean_val:.4f}")
        finite_idx = np.where(np.isfinite(vals_np))[0]
        low_i  = int(finite_idx[np.argmin(vals_np[finite_idx])]) if len(finite_idx) else 0
        high_i = int(finite_idx[np.argmax(vals_np[finite_idx])]) if len(finite_idx) else 0
        mask   = np.isfinite(pmets_np) & np.isfinite(vals_np)
        corr   = float(np.corrcoef(pmets_np[mask], vals_np[mask])[0][1]) if mask.sum() >= 2 else float("nan")
        summary[name] = {
            "mean": mean_val,
            "std":  float(np.nanstd(vals_np)),
            "lowest":  {"value": vals[low_i],  "pdb": pdbs[low_i]},
            "highest": {"value": vals[high_i], "pdb": pdbs[high_i]},
            "correlation": corr,
        }

    # per-target
    target_groups = defaultdict(list)
    for pdb, m in zip(pdbs, metrics_list):
        if m:
            target_groups[pdb].append(m)

    metric_names = list(first.keys())
    per_target   = {}
    for pdb in sorted(target_groups):
        per_target[pdb] = {}
        for name in metric_names:
            vals = np.array([m[name] for m in target_groups[pdb]], dtype=float)
            per_target[pdb][name] = {"mean": float(np.nanmean(vals)),
                                     "std":  float(np.nanstd(vals)),
                                     "n":    len(vals)}

    print("\n===== Per-target (mean ± std) =====")
    tgt_summary = {}
    for name in metric_names:
        tmeans = [per_target[p][name]["mean"] for p in per_target]
        tstds  = [per_target[p][name]["std"]  for p in per_target]
        mof_m  = float(np.nanmean(tmeans))
        mof_s  = float(np.nanmean(tstds))
        print(f"{name}: {mof_m:.4f} ± {mof_s:.4f}")
        tgt_summary[name] = {"mean_of_target_means": mof_m,
                             "mean_of_target_stds":  mof_s}

    summary["per_target_aggregation"] = tgt_summary
    summary["per_target_details"]     = per_target

    # ── mmseqs diversity (greedy fallback) ───────────────────────────────
    tgt_seqs = defaultdict(list)
    for pdb, seq in zip(pdbs, pred_seqs):
        if seq:
            tgt_seqs[pdb].append(seq)

    def seq_id(a, b):
        return sum(x == y for x, y in zip(a, b)) / max(len(a), len(b), 1)

    diversity_stats = {}
    for pdb, seqs in sorted(tgt_seqs.items()):
        clusters, assigned = [], [False] * len(seqs)
        for i, s in enumerate(seqs):
            if not assigned[i]:
                clusters.append(i)
                for j in range(i + 1, len(seqs)):
                    if not assigned[j] and seq_id(s, seqs[j]) >= 0.9:
                        assigned[j] = True
        n_clu  = len(clusters)
        n_uniq = len(set(seqs))
        ratio  = n_clu / len(seqs)
        diversity_stats[pdb] = {"n_clusters": n_clu, "n_unique": n_uniq,
                                "n_samples": len(seqs), "cluster_ratio": ratio}
    summary["diversity"] = {"per_target": diversity_stats}

    # ── save ─────────────────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    csv_path = os.path.join(save_dir, "per_sample_metrics.csv")
    with open(csv_path, "w") as f:
        f.write("pdb," + ",".join(metric_names) + ",pmetric\n")
        for pdb, m, pm in zip(pdbs, metrics_list, pmets):
            if m is None:
                row = pdb + "," + ",".join(["nan"] * len(metric_names)) + f",{pm}"
            else:
                row = pdb + "," + ",".join(str(m[n]) for n in metric_names) + f",{pm}"
            f.write(row + "\n")
    print(f"CSV  → {csv_path}")

    json_path = os.path.join(save_dir, "summary_metrics.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON → {json_path}")

    tgt_csv = os.path.join(save_dir, "per_target_metrics.csv")
    with open(tgt_csv, "w") as f:
        header = "pdb,n," + ",".join(f"{n}_mean,{n}_std" for n in metric_names)
        f.write(header + "\n")
        for pdb in sorted(per_target):
            n = per_target[pdb][metric_names[0]]["n"]
            row = f"{pdb},{n}"
            for nm in metric_names:
                s = per_target[pdb][nm]
                row += f",{s['mean']},{s['std']}"
            f.write(row + "\n")
    print(f"Per-target CSV → {tgt_csv}")


# ─────────────────────────────── main ───────────────────────────────────────

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--workspace",   required=True, help="PRiME project root")
    p.add_argument("--baseline",    required=True, help="Baseline result folder")
    p.add_argument("--test_json",   required=True, help="RAbD test.json path")
    p.add_argument("--save_dir",    required=True, help="Output directory")
    p.add_argument("--num_workers", type=int, default=8)
    return p.parse_args()


if __name__ == "__main__":
    import multiprocessing as mp
    args = parse()

    sys.path.insert(0, args.workspace)
    os.environ["PATH"] = f"/tmp:{os.environ['PATH']}"

    test_map = load_test_map(args.test_json)
    entries  = build_entry_list(args.baseline, test_map)
    print(f"Total entries to evaluate: {len(entries)}")

    if args.num_workers > 1:
        with mp.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.workspace, args.test_json),
        ) as pool:
            raw_results = list(tqdm(
                pool.imap(_worker, entries, chunksize=4),
                total=len(entries),
                desc="Evaluating",
            ))
    else:
        _init_worker(args.workspace, args.test_json)
        raw_results = [_worker(e) for e in tqdm(entries)]

    aggregate(entries, raw_results, args.save_dir)
    print("Done.")
