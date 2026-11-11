#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Comprehensive antibody CDR design metrics.

Combines all metric computation into a single script:
  1. Standard       -- AAR, CAAR, RMSD, TMscore, LDDT, DockQ
  2. Physics        -- steric clashes (interface + intra-CDR + all-atom),
                       bond geometry, Ramachandran validity, FoldX ddG
  3. PLM            -- ESM2 pseudo-perplexity, ESMFold pLDDT
  4. Designability  -- charge, hydrophobicity, PTM liabilities, humanness
  5. Sequence       -- BLOSUM62 score, Vendi diversity, AA distribution (KL)
  6. scRMSD         -- self-consistency RMSD via ESMFold
  7. Joint success  -- multi-threshold pass rates (AAR + RMSD + DockQ)

Usage:
  # Fast (skip PLM models):
  python cal_metrics_all.py --test_set path/to/test.jsonl --save_dir results/ --skip_plm

  # Full (with ESM2 + ESMFold):
  python cal_metrics_all.py --test_set path/to/test.jsonl --save_dir results/

  # With scRMSD:
  python cal_metrics_all.py --test_set path/to/test.jsonl --save_dir results/ --scrmsd

  # With FoldX binding energy (slow):
  python cal_metrics_all.py --test_set path/to/test.jsonl --save_dir results/ --foldx

  # Everything:
  python cal_metrics_all.py --test_set path/to/test.jsonl --save_dir results/ --scrmsd --foldx
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from time import time

import numpy as np
import torch
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from data.pdb_utils import AgAbComplex, Protein, Residue, Peptide
from evaluation.rmsd import compute_rmsd
from evaluation.tm_score import tm_score
from evaluation.lddt import lddt
from evaluation.dockq import dockq

from utils.logger import print_log
from configs import CONTACT_DIST, CACHE_DIR, FOLDX_BIN

# ============================================================================
# Constants
# ============================================================================

_VDW_RADII = {'C': 1.7, 'N': 1.55, 'O': 1.52, 'S': 1.8}
_CLASH_TOLERANCE = 0.4
_CA_CLASH_DIST = 2 * _VDW_RADII['C'] - _CLASH_TOLERANCE  # 3.0 A
_HEAVY_ATOM_CLASH_DIST = 2.0  # all-atom clash threshold

# Ideal backbone bond geometry (Engh & Huber, 1991)
_IDEAL_CA_C_DIST = 1.525
_IDEAL_C_N_DIST = 1.329
_IDEAL_N_CA_DIST = 1.458
_BOND_LENGTH_TOL = 0.05

# Amino acid properties
_HYDROPHOBICITY = {  # Kyte-Doolittle
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}
_CHARGE_PH7 = {'R': 1, 'K': 1, 'D': -1, 'E': -1, 'H': 0.1}

# PTM liability motifs
_PTM_PATTERNS = {
    'deamidation_NG': r'N[G]',
    'deamidation_NS': r'N[S]',
    'deamidation_NT': r'N[T]',
    'deamidation_NN': r'N[N]',
    'isomerization_DG': r'D[G]',
    'isomerization_DS': r'D[S]',
    'oxidation_M': r'M',
    'glycosylation_NxS': r'N[^P][ST]',
    'unpaired_cys': r'C',
}

# Human CDR-H3 AA frequencies (aggregated from OAS database)
_HUMAN_CDRH3_FREQ = {
    'A': 0.055, 'R': 0.065, 'N': 0.040, 'D': 0.075, 'C': 0.015,
    'Q': 0.025, 'E': 0.025, 'G': 0.120, 'H': 0.020, 'I': 0.030,
    'L': 0.045, 'K': 0.025, 'M': 0.015, 'F': 0.050, 'P': 0.035,
    'S': 0.065, 'T': 0.045, 'W': 0.035, 'Y': 0.130, 'V': 0.040,
}

# BLOSUM62 substitution matrix (upper-triangle, standard 20 AA)
# Row/col order: A R N D C Q E G H I L K M F P S T W Y V
_BLOSUM62_AA = 'ARNDCQEGHILKMFPSTWYV'
_BLOSUM62_RAW = [
    [ 4,-1,-2,-2, 0,-1,-1, 0,-2,-1,-1,-1,-1,-2,-1, 1, 0,-3,-2, 0],
    [-1, 5, 0,-2,-3, 1, 0,-2, 0,-3,-2, 2,-1,-3,-2,-1,-1,-3,-2,-3],
    [-2, 0, 6, 1,-3, 0, 0, 0, 1,-3,-3, 0,-2,-3,-2, 1, 0,-4,-2,-3],
    [-2,-2, 1, 6,-3, 0, 2,-1,-1,-3,-4,-1,-3,-3,-1, 0,-1,-4,-3,-3],
    [ 0,-3,-3,-3, 9,-3,-4,-3,-3,-1,-1,-3,-1,-2,-3,-1,-1,-2,-2,-1],
    [-1, 1, 0, 0,-3, 5, 2,-2, 0,-3,-2, 1, 0,-3,-1, 0,-1,-2,-1,-2],
    [-1, 0, 0, 2,-4, 2, 5,-2, 0,-3,-3, 1,-2,-3,-1, 0,-1,-3,-2,-2],
    [ 0,-2, 0,-1,-3,-2,-2, 6,-2,-4,-4,-2,-3,-3,-2, 0,-2,-2,-3,-3],
    [-2, 0, 1,-1,-3, 0, 0,-2, 8,-3,-3,-1,-2,-1,-2,-1,-2,-2, 2,-3],
    [-1,-3,-3,-3,-1,-3,-3,-4,-3, 4, 2,-3, 1, 0,-3,-2,-1,-3,-1, 3],
    [-1,-2,-3,-4,-1,-2,-3,-4,-3, 2, 4,-2, 2, 0,-3,-2,-1,-2,-1, 1],
    [-1, 2, 0,-1,-3, 1, 1,-2,-1,-3,-2, 5,-1,-3,-1, 0,-1,-3,-2,-2],
    [-1,-1,-2,-3,-1, 0,-2,-3,-2, 1, 2,-1, 5, 0,-2,-1,-1,-1,-1, 1],
    [-2,-3,-3,-3,-2,-3,-3,-3,-1, 0, 0,-3, 0, 6,-4,-2,-2, 1, 3,-1],
    [-1,-2,-2,-1,-3,-1,-1,-2,-2,-3,-3,-1,-2,-4, 7,-1,-1,-4,-3,-2],
    [ 1,-1, 1, 0,-1, 0, 0, 0,-1,-2,-2, 0,-1,-2,-1, 4, 1,-3,-2,-2],
    [ 0,-1, 0,-1,-1,-1,-1,-2,-2,-1,-1,-1,-1,-2,-1, 1, 5,-2,-2, 0],
    [-3,-3,-4,-4,-2,-2,-3,-2,-2,-3,-2,-3,-1, 1,-4,-3,-2,11, 2,-3],
    [-2,-2,-2,-3,-2,-1,-2,-3, 2,-1,-1,-2,-1, 3,-3,-2,-2, 2, 7,-1],
    [ 0,-3,-3,-3,-1,-2,-2,-3,-3, 3, 1,-2, 1,-1,-2,-2, 0,-3,-1, 4],
]
_BLOSUM62 = {}
for _i, _aa1 in enumerate(_BLOSUM62_AA):
    for _j, _aa2 in enumerate(_BLOSUM62_AA):
        _BLOSUM62[(_aa1, _aa2)] = _BLOSUM62_RAW[_i][_j]

# Natural CDR-H3 amino acid distribution (from OAS, for KL divergence)
_NATURAL_CDRH3_DIST = {
    'A': 0.055, 'R': 0.065, 'N': 0.040, 'D': 0.075, 'C': 0.015,
    'Q': 0.025, 'E': 0.025, 'G': 0.120, 'H': 0.020, 'I': 0.030,
    'L': 0.045, 'K': 0.025, 'M': 0.015, 'F': 0.050, 'P': 0.035,
    'S': 0.065, 'T': 0.045, 'W': 0.035, 'Y': 0.130, 'V': 0.040,
}
_AA_ORDER = sorted(_NATURAL_CDRH3_DIST.keys())


# ============================================================================
# Sequence metrics: BLOSUM62, Vendi diversity, AA distribution (KL)
# ============================================================================

def compute_blosum62_score(pred_seq, native_seq):
    """Per-position BLOSUM62 score normalized to [0, 1].

    More informative than binary AAR: scores biochemically conservative
    substitutions (e.g. Leu->Ile) higher than disruptive ones (e.g. Leu->Asp).
    """
    if len(pred_seq) != len(native_seq) or len(pred_seq) == 0:
        return float('nan')
    scores = []
    for p, n in zip(pred_seq, native_seq):
        if p not in _BLOSUM62_AA or n not in _BLOSUM62_AA:
            continue
        raw = _BLOSUM62[(p, n)]
        self_max = max(_BLOSUM62[(n, n)], _BLOSUM62[(p, p)])
        self_min = min(_BLOSUM62[(n, a)] for a in _BLOSUM62_AA)
        denom = self_max - self_min
        if denom < 1e-8:
            scores.append(1.0 if p == n else 0.0)
        else:
            scores.append((raw - self_min) / denom)
    return float(np.mean(scores)) if scores else float('nan')


def _pairwise_seq_similarity(seq1, seq2):
    """Normalized pairwise sequence similarity via Needleman-Wunsch alignment."""
    from Bio.Align import PairwiseAligner
    from Bio.Seq import Seq
    aligner = PairwiseAligner()
    s1, s2 = Seq(seq1), Seq(seq2)
    alignments = aligner.align(s1, s2)
    return alignments[0].score / max(len(s1), len(s2))


def _vendi_from_sim(mat):
    """Vendi score from upper-triangular similarity matrix.

    Returns effective number of distinct samples (1 = all identical, N = all unique).
    """
    mat = mat + mat.T
    np.fill_diagonal(mat, 1.0)
    eigvals, _ = np.linalg.eigh(mat / len(mat))
    eigvals = np.clip(eigvals, 0.0, None)
    # Avoid log(0)
    eigvals = eigvals[eigvals > 1e-12]
    return float(np.exp(np.nansum(-(eigvals * np.log(eigvals)))))


def compute_vendi_sequence_score(sequences, max_samples=500):
    """Vendi diversity score over a list of amino acid sequences.

    Uses pairwise alignment similarity as the kernel. O(N^2) so
    subsamples to max_samples if needed.
    """
    if len(sequences) <= 1:
        return float('nan')
    seqs = list(sequences)
    if len(seqs) > max_samples:
        import random as _rand
        seqs = _rand.sample(seqs, max_samples)
    N = len(seqs)
    sims = np.zeros((N, N), dtype=np.float32)
    for i in range(N):
        for j in range(i + 1, N):
            sims[i, j] = _pairwise_seq_similarity(seqs[i], seqs[j])
    return _vendi_from_sim(sims)


def compute_aa_kl_divergence(sequences):
    """KL divergence of generated AA distribution vs natural CDR-H3 distribution.

    Lower = closer to natural antibody repertoire. Uses KL(generated || natural).
    Also returns Jensen-Shannon divergence (symmetric, bounded [0, ln2]).
    """
    if not sequences:
        return float('nan'), float('nan')
    # Count AA frequencies in generated sequences
    counts = {aa: 0 for aa in _AA_ORDER}
    total = 0
    for seq in sequences:
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
                total += 1
    if total == 0:
        return float('nan'), float('nan')

    # Generated distribution (with Laplace smoothing)
    eps = 1e-8
    gen_dist = np.array([(counts[aa] + eps) / (total + eps * len(_AA_ORDER)) for aa in _AA_ORDER])
    nat_dist = np.array([_NATURAL_CDRH3_DIST[aa] for aa in _AA_ORDER])

    # KL(gen || nat)
    kl = float(np.sum(gen_dist * np.log(gen_dist / nat_dist)))
    # Jensen-Shannon
    m = 0.5 * (gen_dist + nat_dist)
    js = float(0.5 * np.sum(gen_dist * np.log(gen_dist / m)) +
               0.5 * np.sum(nat_dist * np.log(nat_dist / m)))
    return kl, js


# ============================================================================
# Diversity (from cal_metrics.py)
# ============================================================================

def cluster_seqs_mmseqs(seqs, identity_threshold=0.9):
    """Cluster sequences with MMseqs2 and return number of clusters."""
    if len(seqs) <= 1:
        return len(seqs), len(set(seqs))
    with tempfile.TemporaryDirectory() as tmpdir:
        fasta = os.path.join(tmpdir, 'seqs.fasta')
        with open(fasta, 'w') as f:
            for i, s in enumerate(seqs):
                f.write(f'>seq{i}\n{s}\n')
        db = os.path.join(tmpdir, 'db')
        clu = os.path.join(tmpdir, 'clu')
        tmp = os.path.join(tmpdir, 'tmp')
        subprocess.run(['mmseqs', 'createdb', fasta, db],
                       capture_output=True, check=True)
        subprocess.run(['mmseqs', 'cluster', db, clu, tmp,
                        '--min-seq-id', str(identity_threshold),
                        '-c', '0.8', '--cluster-mode', '0'],
                       capture_output=True, check=True)
        tsv = os.path.join(tmpdir, 'clu.tsv')
        subprocess.run(['mmseqs', 'createtsv', db, db, clu, tsv],
                       capture_output=True, check=True)
        clusters = set()
        with open(tsv) as f:
            for line in f:
                clusters.add(line.strip().split('\t')[0])
        return len(clusters), len(set(seqs))


# ============================================================================
# Physics metrics
# ============================================================================

def compute_interface_clashes_ca(mod_cplx, cdr_type):
    """Count CA-CA clashes between CDR and antigen (< 3.0 A)."""
    cdr_ca, ag_ca = [], []
    if cdr_type is not None:
        for cdr in cdr_type:
            mod_cdr = mod_cplx.get_cdr(cdr)
            for i in range(len(mod_cdr)):
                cdr_ca.append(mod_cdr.get_ca_pos(i))
    ag = mod_cplx.get_antigen()
    if ag is not None:
        for chain_name in ag.get_chain_names():
            chain = ag.get_chain(chain_name)
            for i in range(len(chain)):
                try:
                    ag_ca.append(chain.get_ca_pos(i))
                except KeyError:
                    continue
    if not cdr_ca or not ag_ca:
        return 0
    cdr_ca = np.array(cdr_ca)
    ag_ca = np.array(ag_ca)
    diff = cdr_ca[:, None, :] - ag_ca[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    return int((dists < _CA_CLASH_DIST).sum())


def compute_interface_clashes_allatom(mod_cplx, cdr_type):
    """Count all heavy-atom clashes between CDR and antigen (< 2.0 A)."""
    cdr_atoms, ag_atoms = [], []
    if cdr_type is not None:
        for cdr in cdr_type:
            mod_cdr = mod_cplx.get_cdr(cdr)
            for residue in mod_cdr:
                for atom_name, coord in residue:
                    if atom_name[0] != 'H':
                        cdr_atoms.append(coord)
    ag = mod_cplx.get_antigen()
    if ag is not None:
        for chain_name in ag.get_chain_names():
            chain = ag.get_chain(chain_name)
            for residue in chain:
                for atom_name, coord in residue:
                    if atom_name[0] != 'H':
                        ag_atoms.append(coord)
    if not cdr_atoms or not ag_atoms:
        return 0
    cdr_arr = np.array(cdr_atoms)
    ag_arr = np.array(ag_atoms)
    n_clashes = 0
    chunk_size = 500
    for i in range(0, len(cdr_arr), chunk_size):
        chunk = cdr_arr[i:i + chunk_size]
        diff = chunk[:, None, :] - ag_arr[None, :, :]
        dists = np.linalg.norm(diff, axis=-1)
        n_clashes += int((dists < _HEAVY_ATOM_CLASH_DIST).sum())
    return n_clashes


def compute_intra_cdr_clashes(mod_cplx, cdr_type):
    """Count CA-CA clashes within CDR (seq separation >= 2)."""
    cdr_ca = []
    if cdr_type is not None:
        for cdr in cdr_type:
            mod_cdr = mod_cplx.get_cdr(cdr)
            for i in range(len(mod_cdr)):
                cdr_ca.append(mod_cdr.get_ca_pos(i))
    if len(cdr_ca) < 2:
        return 0
    cdr_ca = np.array(cdr_ca)
    n = len(cdr_ca)
    diff = cdr_ca[:, None, :] - cdr_ca[None, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    mask = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 2, n):
            mask[i, j] = True
    return int((dists[mask] < _CA_CLASH_DIST).sum())


def compute_bond_geometry(mod_cplx, cdr_type):
    """Check backbone bond lengths in CDR. Returns (valid_frac, violations, total)."""
    violations = 0
    total_bonds = 0
    if cdr_type is None:
        return float('nan'), 0, 0

    for cdr in cdr_type:
        mod_cdr = mod_cplx.get_cdr(cdr)
        n_res = len(mod_cdr)
        for i in range(n_res):
            cm = mod_cdr.get_residue(i).get_coord_map()
            if 'N' in cm and 'CA' in cm:
                d = np.linalg.norm(np.array(cm['N']) - np.array(cm['CA']))
                total_bonds += 1
                if abs(d - _IDEAL_N_CA_DIST) > _BOND_LENGTH_TOL:
                    violations += 1
            if 'CA' in cm and 'C' in cm:
                d = np.linalg.norm(np.array(cm['CA']) - np.array(cm['C']))
                total_bonds += 1
                if abs(d - _IDEAL_CA_C_DIST) > _BOND_LENGTH_TOL:
                    violations += 1
            if i < n_res - 1:
                next_cm = mod_cdr.get_residue(i + 1).get_coord_map()
                if 'C' in cm and 'N' in next_cm:
                    d = np.linalg.norm(np.array(cm['C']) - np.array(next_cm['N']))
                    total_bonds += 1
                    if abs(d - _IDEAL_C_N_DIST) > _BOND_LENGTH_TOL:
                        violations += 1
    if total_bonds == 0:
        return float('nan'), 0, 0
    return 1.0 - violations / total_bonds, violations, total_bonds


def _dihedral_angle(p0, p1, p2, p3):
    """Compute dihedral angle (degrees) from 4 points."""
    b1, b2, b3 = p1 - p0, p2 - p1, p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    n1_norm, n2_norm = np.linalg.norm(n1), np.linalg.norm(n2)
    if n1_norm < 1e-8 or n2_norm < 1e-8:
        return float('nan')
    n1, n2 = n1 / n1_norm, n2 / n2_norm
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.degrees(np.arctan2(np.dot(m1, n2), np.dot(n1, n2)))


def compute_ramachandran(mod_cplx, cdr_type):
    """Compute Ramachandran validity for CDR residues.

    Returns (frac_allowed, n_outliers, n_total).
    """
    if cdr_type is None:
        return float('nan'), 0, 0
    outliers, total = 0, 0

    for cdr in cdr_type:
        mod_cdr = mod_cplx.get_cdr(cdr)
        n_res = len(mod_cdr)
        for i in range(n_res):
            res_i = mod_cdr.get_residue(i)
            cm_i = res_i.get_coord_map()
            symbol = res_i.get_symbol()
            phi, psi = float('nan'), float('nan')

            if i > 0:
                cm_prev = mod_cdr.get_residue(i - 1).get_coord_map()
                if 'C' in cm_prev and all(k in cm_i for k in ['N', 'CA', 'C']):
                    phi = _dihedral_angle(
                        np.array(cm_prev['C']), np.array(cm_i['N']),
                        np.array(cm_i['CA']), np.array(cm_i['C']))
            if i < n_res - 1:
                cm_next = mod_cdr.get_residue(i + 1).get_coord_map()
                if all(k in cm_i for k in ['N', 'CA', 'C']) and 'N' in cm_next:
                    psi = _dihedral_angle(
                        np.array(cm_i['N']), np.array(cm_i['CA']),
                        np.array(cm_i['C']), np.array(cm_next['N']))

            if math.isnan(phi) or math.isnan(psi):
                continue
            total += 1
            if symbol == 'G':
                continue  # glycine: almost everything allowed
            if symbol == 'P':
                if phi > 0 or phi < -100:
                    outliers += 1
                continue
            # General: disallow top-right except left-handed alpha (~60, ~40)
            if phi > 0:
                if 30 <= phi <= 90 and 10 <= psi <= 70:
                    continue
                outliers += 1

    if total == 0:
        return float('nan'), 0, 0
    return 1.0 - outliers / total, outliers, total


def compute_foldx_ddg(mod_cplx, ref_cplx):
    """FoldX binding energy: ddG = dG(model) - dG(reference). More negative = better."""
    if not os.path.exists(FOLDX_BIN):
        return float('nan'), float('nan'), float('nan')

    from utils.time_sign import get_time_sign
    prefix = get_time_sign(suffix='foldx_ddg')
    mod_pdb = os.path.join(CACHE_DIR, prefix + '_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, prefix + '_ref.pdb')

    try:
        mod_cplx.to_pdb(mod_pdb)
        ref_cplx.to_pdb(ref_pdb)

        ag_chains = ref_cplx.get_antigen().get_chain_names() if ref_cplx.get_antigen() else []
        H, L = ref_cplx.heavy_chain, ref_cplx.light_chain
        rec_str = ''.join(ag_chains)
        lig_str = ''.join([H, L])

        def _foldx_energy(pdb_path):
            fn = os.path.basename(pdb_path)
            p = os.popen(
                f'cd {CACHE_DIR}; {FOLDX_BIN} --command=AnalyseComplex '
                f'--pdb={fn} --analyseComplexChains={rec_str},{lig_str}')
            output = p.read()
            p.close()
            for line in reversed(output.strip().split('\n')):
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        return float(parts[5])
                    except (ValueError, IndexError):
                        continue
            return float('nan')

        dg_mod = _foldx_energy(mod_pdb)
        dg_ref = _foldx_energy(ref_pdb)
        return dg_mod - dg_ref, dg_mod, dg_ref
    except Exception as e:
        print_log(f'FoldX ddG error: {e}', level='ERROR')
        return float('nan'), float('nan'), float('nan')
    finally:
        for p in [mod_pdb, ref_pdb]:
            if os.path.exists(p):
                os.remove(p)


# ============================================================================
# Self-consistency RMSD (scRMSD) via ESMFold
# ============================================================================

def extract_ca_from_pdb_string(pdb_string):
    """Extract CA coordinates from PDB string (ESMFold output)."""
    cas = []
    for line in pdb_string.split('\n'):
        if line.startswith('ATOM') and line[12:16].strip() == 'CA':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            cas.append([x, y, z])
    return np.array(cas)


def compute_scrmsd_batch(items, device='cuda:0'):
    """Compute scRMSD for all items using ESMFold.

    For each designed antibody, fold the heavy chain with ESMFold and compute
    RMSD to the designed structure (CDR-only and full heavy chain).
    """
    import esm

    print("Loading ESMFold for scRMSD...")
    esmfold = esm.pretrained.esmfold_v1()
    esmfold = esmfold.to(device)
    esmfold.eval()

    results = []
    for item in tqdm(items, desc='scRMSD'):
        mod_pdb = item['mod_pdb']
        H, L, A = item['H'], item['L'], item['A']
        cdr_type = item['cdr_type']
        if isinstance(cdr_type, str):
            cdr_type = [cdr_type]

        try:
            mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, skip_epitope_cal=True)
            ref_cplx = AgAbComplex.from_pdb(item['ref_pdb'], H, L, A, skip_epitope_cal=True)

            h_seq = mod_cplx.get_heavy_chain().get_seq()
            h_chain = mod_cplx.get_heavy_chain()

            # AAR
            gt_cdr_seq, pred_cdr_seq = '', ''
            for cdr in cdr_type:
                gt_cdr_seq += ref_cplx.get_cdr(cdr).get_seq()
                pred_cdr_seq += mod_cplx.get_cdr(cdr).get_seq()
            aar = sum(a == b for a, b in zip(gt_cdr_seq, pred_cdr_seq)) / len(gt_cdr_seq)

            if len(h_seq) > 600:
                results.append({'scrmsd_cdr': float('nan'), 'scrmsd_h': float('nan'),
                               'aar': aar, 'pdb': item['pdb'], 'sample_idx': item.get('sample_idx', 0)})
                continue

            with torch.no_grad():
                pdb_out = esmfold.infer_pdb(h_seq)

            pred_ca = extract_ca_from_pdb_string(pdb_out)
            designed_h_ca = np.array([h_chain.get_ca_pos(i) for i in range(len(h_chain))])

            if len(pred_ca) != len(designed_h_ca):
                print(f"  {item['pdb']}_s{item.get('sample_idx', '?')}: length mismatch "
                      f"pred={len(pred_ca)} designed={len(designed_h_ca)}")
                results.append({'scrmsd_cdr': float('nan'), 'scrmsd_h': float('nan'),
                               'aar': aar, 'pdb': item['pdb'], 'sample_idx': item.get('sample_idx', 0)})
                continue

            scrmsd_h = compute_rmsd(pred_ca, designed_h_ca, aligned=False)

            # CDR-only scRMSD: find CDR residue indices in heavy chain
            cdr_indices = []
            for cdr in cdr_type:
                if cdr.startswith('H'):
                    mod_cdr = mod_cplx.get_cdr(cdr)
                    for i in range(len(h_chain)):
                        for j in range(len(mod_cdr)):
                            if (h_chain.get_ca_pos(i) is not None and
                                    mod_cdr.get_ca_pos(j) is not None and
                                    np.allclose(h_chain.get_ca_pos(i), mod_cdr.get_ca_pos(j), atol=0.01)):
                                cdr_indices.append(i)

            if cdr_indices:
                scrmsd_cdr = compute_rmsd(pred_ca[cdr_indices], designed_h_ca[cdr_indices], aligned=False)
            else:
                scrmsd_cdr = float('nan')

            results.append({
                'scrmsd_cdr': float(scrmsd_cdr),
                'scrmsd_h': float(scrmsd_h),
                'aar': float(aar),
                'pdb': item['pdb'],
                'sample_idx': item.get('sample_idx', 0),
            })

        except Exception as e:
            print(f"  Error {item.get('pdb', '?')}_s{item.get('sample_idx', '?')}: {e}")
            results.append({'scrmsd_cdr': float('nan'), 'scrmsd_h': float('nan'),
                           'aar': float('nan'), 'pdb': item.get('pdb', '?'),
                           'sample_idx': item.get('sample_idx', '?')})

    del esmfold
    torch.cuda.empty_cache()
    return results


# ============================================================================
# Joint success rates
# ============================================================================

_JOINT_THRESHOLDS = [
    ('AAR>30% & RMSD<3.0 & DockQ>0.23',
     lambda m: m.get('AAR', 0) > 0.3 and m.get('RMSD_CA', 99) < 3.0 and m.get('DockQ', 0) > 0.23),
    ('AAR>40% & RMSD<2.5 & DockQ>0.23',
     lambda m: m.get('AAR', 0) > 0.4 and m.get('RMSD_CA', 99) < 2.5 and m.get('DockQ', 0) > 0.23),
    ('AAR>50% & RMSD<2.5 & DockQ>0.23',
     lambda m: m.get('AAR', 0) > 0.5 and m.get('RMSD_CA', 99) < 2.5 and m.get('DockQ', 0) > 0.23),
    ('AAR>60% & RMSD<2.0 & DockQ>0.23',
     lambda m: m.get('AAR', 0) > 0.6 and m.get('RMSD_CA', 99) < 2.0 and m.get('DockQ', 0) > 0.23),
    ('AAR>50% & RMSD<2.0',
     lambda m: m.get('AAR', 0) > 0.5 and m.get('RMSD_CA', 99) < 2.0),
    ('AAR>60% & RMSD<2.0',
     lambda m: m.get('AAR', 0) > 0.6 and m.get('RMSD_CA', 99) < 2.0),
]


def compute_joint_success(metrics):
    """Compute joint success rates from per-sample metric dicts.

    Each metric dict should have 'AAR', 'RMSD(CA) aligned' or CDR-specific RMSD, and 'DockQ'.
    """
    if not metrics:
        return {}

    # Normalize keys: find best CDR RMSD key available
    normalized = []
    for m in metrics:
        nm = {
            'AAR': m.get('AAR', m.get('AAR H3', 0)),
            'DockQ': m.get('DockQ', 0),
        }
        # Pick the most specific CDR RMSD available
        for key in ['RMSD(CA) CDRH3 aligned', 'RMSD(CA) aligned']:
            if key in m:
                nm['RMSD_CA'] = m[key]
                break
        else:
            nm['RMSD_CA'] = 99
        normalized.append(nm)

    results = {}
    for name, fn in _JOINT_THRESHOLDS:
        passed = sum(1 for m in normalized if fn(m))
        rate = passed / len(normalized)
        results[name] = rate
    return results


# ============================================================================
# PLM metrics (batch, GPU)
# ============================================================================

def compute_esm2_pseudoperplexity_batch(sequences, device='cuda:0', batch_size=8):
    """ESM-2 pseudo-perplexity: exp(-1/L * sum_i log P(x_i | x_{\\i})).

    Lower = more natural/protein-like. Masks one position at a time.
    """
    import esm

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    results = []
    for seq in tqdm(sequences, desc='ESM2 PPL'):
        L = len(seq)
        if L == 0:
            results.append(float('nan'))
            continue

        data = [(f'pos_{i}', seq[:i] + '<mask>' + seq[i + 1:]) for i in range(L)]
        _, _, tokens = batch_converter(data)
        tokens = tokens.to(device)

        log_probs_sum = 0.0
        sub_batch = 32
        for sb_start in range(0, L, sub_batch):
            sb_tokens = tokens[sb_start:sb_start + sub_batch]
            with torch.no_grad():
                logits = model(sb_tokens)['logits']
            log_probs = torch.log_softmax(logits, dim=-1)
            for j, pos in enumerate(range(sb_start, min(sb_start + sub_batch, L))):
                target_idx = alphabet.get_idx(seq[pos])
                log_probs_sum += log_probs[j, pos + 1, target_idx].item()

        results.append(float(np.exp(-log_probs_sum / L)))

    del model
    torch.cuda.empty_cache()
    return results


def compute_esmfold_plddt_batch(full_sequences, device='cuda:0'):
    """ESMFold pLDDT: would the designed sequence fold confidently? (0-100)."""
    try:
        import esm
        esmfold = esm.pretrained.esmfold_v1()
        esmfold = esmfold.to(device)
        esmfold.eval()

        results = []
        for seq in tqdm(full_sequences, desc='ESMFold pLDDT'):
            if not seq or len(seq) > 800:
                results.append(float('nan'))
                continue
            try:
                with torch.no_grad():
                    output = esmfold.infer_pdb(seq)
                plddts = []
                for line in output.split('\n'):
                    if line.startswith('ATOM') and line[12:16].strip() == 'CA':
                        try:
                            plddts.append(float(line[60:66].strip()))
                        except (ValueError, IndexError):
                            continue
                results.append(float(np.mean(plddts)) if plddts else float('nan'))
            except Exception as e:
                print_log(f'ESMFold error (len={len(seq)}): {e}', level='ERROR')
                results.append(float('nan'))

        del esmfold
        torch.cuda.empty_cache()
        return results
    except Exception as e:
        print_log(f'ESMFold not available: {e}', level='ERROR')
        return [float('nan')] * len(full_sequences)


# ============================================================================
# Designability metrics
# ============================================================================

def compute_charge(seq):
    """Net charge of CDR at pH 7."""
    return sum(_CHARGE_PH7.get(aa, 0) for aa in seq)


def compute_hydrophobicity(seq):
    """Mean Kyte-Doolittle hydrophobicity."""
    if not seq:
        return float('nan')
    return float(np.mean([_HYDROPHOBICITY.get(aa, 0) for aa in seq]))


def compute_ptm_liabilities(seq):
    """Count PTM liability motifs. Returns (counts_dict, total)."""
    counts = {}
    total = 0
    for name, pattern in _PTM_PATTERNS.items():
        n = len(re.findall(pattern, seq))
        counts[name] = n
        total += n
    return counts, total


def compute_humanness(seq):
    """Humanness proxy: geometric mean of per-position human CDR-H3 AA frequency.

    Score 0-1, higher = more human-like. For publication use Hu-mAb or OASis.
    """
    if not seq:
        return float('nan')
    log_sum = sum(np.log(_HUMAN_CDRH3_FREQ.get(aa, 0.005)) for aa in seq)
    return float(np.exp(log_sum / len(seq)))


# ============================================================================
# Per-sample metric computation
# ============================================================================

def cal_metrics_all(inputs):
    """Compute all metrics for a single model-reference pair.

    Returns (results_dict, pred_cdr_seq, pred_full_ab_seq).
    """
    mod_pdb, ref_pdb, H, L, A, cdr_type = inputs[:6]
    numbering = inputs[6] if len(inputs) > 6 else 'imgt'

    try:
        mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, numbering=numbering, skip_epitope_cal=True)
        ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A, numbering=numbering, skip_epitope_cal=False)
    except (AssertionError, Exception) as e:
        print_log(f'Skipping {mod_pdb}: {e}', level='ERROR')
        return {}, None, None

    results = {}
    cdr_type = [cdr_type] if isinstance(cdr_type, str) else cdr_type

    # === 1. AAR & CAAR ===
    epitope = ref_cplx.get_epitope()
    is_contact = []
    if cdr_type is None:
        gt_s = ref_cplx.get_heavy_chain().get_seq() + ref_cplx.get_light_chain().get_seq()
        pred_s = mod_cplx.get_heavy_chain().get_seq() + mod_cplx.get_light_chain().get_seq()
        for chain in [ref_cplx.get_heavy_chain(), ref_cplx.get_light_chain()]:
            for ab_residue in chain:
                contact = any(ab_residue.dist_to(ag_r) < CONTACT_DIST for ag_r, _, _ in epitope)
                is_contact.append(int(contact))
    else:
        gt_s, pred_s = '', ''
        for cdr in cdr_type:
            gt_cdr = ref_cplx.get_cdr(cdr)
            cur_gt_s = gt_cdr.get_seq()
            cur_pred_s = mod_cplx.get_cdr(cdr).get_seq()
            gt_s += cur_gt_s
            pred_s += cur_pred_s
            cur_contact = []
            for ab_residue in gt_cdr:
                contact = any(ab_residue.dist_to(ag_r) < CONTACT_DIST for ag_r, _, _ in epitope)
                cur_contact.append(int(contact))
            is_contact.extend(cur_contact)

            hit = sum(1 for a, b in zip(cur_gt_s, cur_pred_s) if a == b)
            chit = sum(1 for a, b, c in zip(cur_gt_s, cur_pred_s, cur_contact) if a == b and c)
            results[f'AAR {cdr}'] = hit / len(cur_gt_s)
            results[f'CAAR {cdr}'] = chit / (sum(cur_contact) + 1e-10)

    if len(gt_s) != len(pred_s):
        print_log(f'Length conflict: {len(gt_s)} vs {len(pred_s)}', level='WARN')
    hit = sum(1 for a, b in zip(gt_s, pred_s) if a == b)
    chit = sum(1 for a, b, c in zip(gt_s, pred_s, is_contact) if a == b and c)
    results['AAR'] = hit / len(gt_s)
    results['CAAR'] = chit / (sum(is_contact) + 1e-10)

    # === 1b. BLOSUM62 score ===
    results['BLOSUM62'] = compute_blosum62_score(pred_s, gt_s)
    if cdr_type is not None:
        for cdr in cdr_type:
            gt_cdr_s = ref_cplx.get_cdr(cdr).get_seq()
            pred_cdr_s = mod_cplx.get_cdr(cdr).get_seq()
            results[f'BLOSUM62 {cdr}'] = compute_blosum62_score(pred_cdr_s, gt_cdr_s)

    # === 2. RMSD(CA) ===
    gt_x, pred_x = [], []
    for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
        for chain in [c.get_heavy_chain(), c.get_light_chain()]:
            for i in range(len(chain)):
                try:
                    xl.append(chain.get_ca_pos(i))
                except KeyError:
                    pass
    gt_x, pred_x = np.array(gt_x), np.array(pred_x)
    if len(gt_x) == len(pred_x):
        results['RMSD(CA) aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
    else:
        results['RMSD(CA) aligned'] = float('nan')
    if cdr_type is not None:
        for cdr in cdr_type:
            gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
            gx = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr))])
            px = np.array([pred_cdr.get_ca_pos(i) for i in range(len(pred_cdr))])
            if len(gx) == len(px):
                results[f'RMSD(CA) CDR{cdr}'] = compute_rmsd(gx, px, aligned=True)
                results[f'RMSD(CA) CDR{cdr} aligned'] = compute_rmsd(gx, px, aligned=False)
            else:
                results[f'RMSD(CA) CDR{cdr}'] = float('nan')
                results[f'RMSD(CA) CDR{cdr} aligned'] = float('nan')

    # === 3. TMscore ===
    try:
        results['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)
    except Exception as e:
        print_log(f'TMscore error: {e}', level='ERROR')
        results['TMscore'] = float('nan')

    # === 4. LDDT ===
    try:
        score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
        results['LDDT'] = score
    except Exception as e:
        print_log(f'LDDT error: {e}', level='ERROR')
        results['LDDT'] = float('nan')

    # === 5. DockQ ===
    try:
        results['DockQ'] = dockq(mod_cplx, ref_cplx, cdrh3_only=False)
    except Exception as e:
        print_log(f'DockQ error: {e}', level='ERROR')
        results['DockQ'] = float('nan')

    # === 6. Physics: steric clashes ===
    try:
        results['interface_clashes_CA'] = compute_interface_clashes_ca(mod_cplx, cdr_type)
        results['interface_clashes_allatom'] = compute_interface_clashes_allatom(mod_cplx, cdr_type)
        results['intra_cdr_clashes'] = compute_intra_cdr_clashes(mod_cplx, cdr_type)
    except Exception as e:
        print_log(f'Clash error: {e}', level='ERROR')
        results['interface_clashes_CA'] = -1
        results['interface_clashes_allatom'] = -1
        results['intra_cdr_clashes'] = -1

    # === 7. Physics: bond geometry ===
    try:
        bv, bviol, btot = compute_bond_geometry(mod_cplx, cdr_type)
        results['bond_validity'] = bv
        results['bond_violations'] = bviol
        results['bond_total'] = btot
    except Exception as e:
        print_log(f'Bond geometry error: {e}', level='ERROR')
        results['bond_validity'] = float('nan')
        results['bond_violations'] = -1
        results['bond_total'] = -1

    # === 8. Physics: Ramachandran ===
    try:
        ra, ro, rt = compute_ramachandran(mod_cplx, cdr_type)
        results['rama_allowed_frac'] = ra
        results['rama_outliers'] = ro
        results['rama_total'] = rt
    except Exception as e:
        print_log(f'Ramachandran error: {e}', level='ERROR')
        results['rama_allowed_frac'] = float('nan')
        results['rama_outliers'] = -1
        results['rama_total'] = -1

    # === 9. Designability ===
    results['cdr_charge'] = compute_charge(pred_s)
    results['cdr_hydrophobicity'] = compute_hydrophobicity(pred_s)
    ptm_counts, ptm_total = compute_ptm_liabilities(pred_s)
    results['ptm_liabilities_total'] = ptm_total
    for ptm_name, count in ptm_counts.items():
        results[f'ptm_{ptm_name}'] = count
    results['humanness_score'] = compute_humanness(pred_s)

    full_ab_seq = mod_cplx.get_heavy_chain().get_seq() + mod_cplx.get_light_chain().get_seq()

    print(f'{mod_cplx.get_id()}: AAR={results["AAR"]:.3f} BLOSUM62={results["BLOSUM62"]:.3f} '
          f'DockQ={results.get("DockQ", 0):.3f} '
          f'clash_CA={results["interface_clashes_CA"]} clash_aa={results["interface_clashes_allatom"]} '
          f'bond={results["bond_validity"]:.3f} rama={results["rama_allowed_frac"]:.3f}')

    return results, pred_s, full_ab_seq


# ============================================================================
# Main
# ============================================================================

def main(args):
    with open(args.test_set, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(line) for line in lines]
    metric_inputs = []
    pdbs = [item['pdb'] for item in items]
    pmets = []
    for item in items:
        keys = ['mod_pdb', 'ref_pdb', 'H', 'L', 'A', 'cdr_type']
        inp = [item[key] for key in keys]
        inp.append(args.numbering)
        metric_inputs.append(inp)
        pmets.append(item['pmetric'])

    # --- Phase 1: per-sample (standard + physics + designability) ---
    print('=' * 60)
    print('Phase 1: Per-sample metrics')
    print('=' * 60)
    if args.num_workers > 1:
        raw_results = process_map(cal_metrics_all, metric_inputs, max_workers=args.num_workers)
    else:
        raw_results = [cal_metrics_all(inp) for inp in tqdm(metric_inputs)]

    metrics_raw = [r[0] for r in raw_results]
    pred_seqs_raw = [r[1] for r in raw_results]
    full_ab_seqs_raw = [r[2] for r in raw_results]

    # Filter out failed samples
    valid_idx = [i for i, m in enumerate(metrics_raw) if m and len(m) > 0]
    n_failed = len(metrics_raw) - len(valid_idx)
    if n_failed > 0:
        print_log(f'WARNING: {n_failed}/{len(metrics_raw)} samples failed, skipping them.')
    metrics = [metrics_raw[i] for i in valid_idx]
    pred_seqs = [pred_seqs_raw[i] for i in valid_idx]
    full_ab_seqs = [full_ab_seqs_raw[i] for i in valid_idx]
    pdbs = [pdbs[i] for i in valid_idx]
    pmets = [pmets[i] for i in valid_idx]

    first = next((m for m in metrics if m and len(m) > 0), None)
    if first is None:
        raise RuntimeError('No metrics computed successfully.')

    # --- Phase 2: ESM-2 pseudo-perplexity ---
    if not args.skip_plm:
        print('\n' + '=' * 60)
        print('Phase 2: ESM-2 pseudo-perplexity')
        print('=' * 60)
        try:
            ppls = compute_esm2_pseudoperplexity_batch(
                pred_seqs, device=args.device, batch_size=args.esm_batch_size)
            for i, ppl in enumerate(ppls):
                metrics[i]['esm2_ppl'] = ppl
        except Exception as e:
            print_log(f'ESM-2 PPL failed: {e}', level='ERROR')
            for m in metrics:
                m['esm2_ppl'] = float('nan')

        # --- Phase 3: ESMFold pLDDT ---
        if not args.skip_esmfold:
            print('\n' + '=' * 60)
            print('Phase 3: ESMFold pLDDT')
            print('=' * 60)
            try:
                plddts = compute_esmfold_plddt_batch(full_ab_seqs, device=args.device)
                for i, plddt in enumerate(plddts):
                    metrics[i]['esmfold_plddt'] = plddt
            except Exception as e:
                print_log(f'ESMFold failed: {e}', level='ERROR')
                for m in metrics:
                    m['esmfold_plddt'] = float('nan')
        else:
            for m in metrics:
                m['esmfold_plddt'] = float('nan')
    else:
        for m in metrics:
            m['esm2_ppl'] = float('nan')
            m['esmfold_plddt'] = float('nan')

    # --- Phase 4: FoldX ddG (optional) ---
    if args.foldx:
        print('\n' + '=' * 60)
        print('Phase 4: FoldX binding energy')
        print('=' * 60)
        for i, inp in enumerate(tqdm(metric_inputs, desc='FoldX')):
            try:
                mod_cplx = AgAbComplex.from_pdb(inp[0], inp[2], inp[3], inp[4], skip_epitope_cal=True)
                ref_cplx = AgAbComplex.from_pdb(inp[1], inp[2], inp[3], inp[4], skip_epitope_cal=True)
                ddg, dg_mod, dg_ref = compute_foldx_ddg(mod_cplx, ref_cplx)
                metrics[i]['foldx_ddg'] = ddg
                metrics[i]['foldx_dg_model'] = dg_mod
                metrics[i]['foldx_dg_ref'] = dg_ref
            except Exception as e:
                print_log(f'FoldX error {pdbs[i]}: {e}', level='ERROR')
                metrics[i]['foldx_ddg'] = float('nan')
                metrics[i]['foldx_dg_model'] = float('nan')
                metrics[i]['foldx_dg_ref'] = float('nan')
    else:
        for m in metrics:
            m['foldx_ddg'] = float('nan')
            m['foldx_dg_model'] = float('nan')
            m['foldx_dg_ref'] = float('nan')

    # --- Aggregation ---
    first = metrics[0]
    metric_names = list(first.keys())
    pmets_np = np.asarray(pmets, dtype=float)
    summary = {}

    print('\n' + '=' * 60)
    print('Summary')
    print('=' * 60)

    for name in metric_names:
        vals = [m.get(name, float('nan')) for m in metrics]
        vals_np = np.asarray(vals, dtype=float)
        mean_val = float(np.nanmean(vals_np)) if not np.all(np.isnan(vals_np)) else float('nan')
        std_val = float(np.nanstd(vals_np)) if not np.all(np.isnan(vals_np)) else float('nan')

        finite_idx = np.where(np.isfinite(vals_np))[0]
        if len(finite_idx) > 0:
            lowest_i = int(finite_idx[np.argmin(vals_np[finite_idx])])
            highest_i = int(finite_idx[np.argmax(vals_np[finite_idx])])
        else:
            lowest_i, highest_i = 0, 0

        mask = np.isfinite(pmets_np) & np.isfinite(vals_np)
        if mask.sum() < 2 or np.std(vals_np[mask]) < 1e-12 or np.std(pmets_np[mask]) < 1e-12:
            corr = float('nan')
        else:
            corr = float(np.corrcoef(pmets_np[mask], vals_np[mask])[0][1])

        summary[name] = {
            'mean': mean_val, 'std': std_val,
            'lowest': {'value': vals[lowest_i], 'pdb': pdbs[lowest_i]} if len(finite_idx) > 0 else None,
            'highest': {'value': vals[highest_i], 'pdb': pdbs[highest_i]} if len(finite_idx) > 0 else None,
            'correlation': corr,
        }

    # Per-target aggregation
    target_groups = defaultdict(list)
    for pdb, m in zip(pdbs, metrics):
        if m and len(m) > 0:
            target_groups[pdb].append(m)

    per_target_stats = {}
    for pdb in sorted(target_groups.keys()):
        per_target_stats[pdb] = {}
        for name in metric_names:
            vals = np.array([m.get(name, float('nan')) for m in target_groups[pdb]], dtype=float)
            per_target_stats[pdb][name] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals)),
                'n': int(len(vals)),
            }

    target_summary = {}
    for name in metric_names:
        tmeans = np.array([per_target_stats[p][name]['mean'] for p in per_target_stats], dtype=float)
        tstds = np.array([per_target_stats[p][name]['std'] for p in per_target_stats], dtype=float)
        target_summary[name] = {
            'mean_of_target_means': float(np.nanmean(tmeans)),
            'mean_of_target_stds': float(np.nanmean(tstds)),
            'std_of_target_means': float(np.nanstd(tmeans)),
        }
    summary['per_target_aggregation'] = target_summary
    summary['per_target_details'] = per_target_stats

    # Diversity
    target_seqs = defaultdict(list)
    for pdb, seq in zip(pdbs, pred_seqs):
        if seq is not None:
            target_seqs[pdb].append(seq)

    print('\n===== Diversity (MMseqs2, 90% identity) =====')
    diversity_stats = {}
    cluster_ratios, n_clusters_list, n_unique_list = [], [], []
    for pdb in sorted(target_seqs.keys()):
        seqs = target_seqs[pdb]
        n_clu, n_uniq = cluster_seqs_mmseqs(seqs, identity_threshold=0.9)
        ratio = n_clu / len(seqs)
        cluster_ratios.append(ratio)
        n_clusters_list.append(n_clu)
        n_unique_list.append(n_uniq)
        diversity_stats[pdb] = {
            'n_clusters': n_clu, 'n_unique': n_uniq,
            'n_samples': len(seqs), 'cluster_ratio': ratio,
        }
        print(f'\t{pdb} (n={len(seqs)}): clusters={n_clu}, unique={n_uniq}, ratio={ratio:.3f}')

    summary['diversity'] = {
        'avg_clusters': float(np.mean(n_clusters_list)) if n_clusters_list else 0,
        'avg_unique_seqs': float(np.mean(n_unique_list)) if n_unique_list else 0,
        'avg_cluster_ratio': float(np.mean(cluster_ratios)) if cluster_ratios else 0,
        'targets_1_cluster': sum(1 for c in n_clusters_list if c == 1),
        'per_target': diversity_stats,
    }

    # --- Vendi sequence diversity ---
    print('\n===== Vendi Sequence Diversity =====')
    all_pred_seqs = [s for s in pred_seqs if s is not None]
    vendi_global = compute_vendi_sequence_score(all_pred_seqs)
    print(f'  Global Vendi score: {vendi_global:.4f}')
    vendi_per_target = {}
    for pdb in sorted(target_seqs.keys()):
        seqs = target_seqs[pdb]
        v = compute_vendi_sequence_score(seqs)
        vendi_per_target[pdb] = v
        print(f'\t{pdb} (n={len(seqs)}): Vendi={v:.4f}')
    summary['vendi_sequence'] = {
        'global': vendi_global,
        'per_target_mean': float(np.nanmean(list(vendi_per_target.values()))) if vendi_per_target else float('nan'),
        'per_target': vendi_per_target,
    }

    # --- AA distribution KL divergence ---
    print('\n===== AA Distribution (KL divergence vs natural CDR-H3) =====')
    kl_div, js_div = compute_aa_kl_divergence(all_pred_seqs)
    print(f'  KL(generated || natural): {kl_div:.4f}')
    print(f'  Jensen-Shannon:           {js_div:.4f}')
    summary['aa_distribution'] = {
        'kl_divergence': kl_div,
        'jensen_shannon': js_div,
    }

    # --- Joint success rates ---
    print('\n' + '=' * 60)
    print('Joint Success Rates')
    print('=' * 60)
    joint_results = compute_joint_success(metrics)
    for name, rate in joint_results.items():
        n_pass = int(rate * len(metrics))
        print(f'  {name}: {n_pass}/{len(metrics)} = {rate:.1%}')
    summary['joint_success'] = joint_results

    # --- scRMSD (ESMFold self-consistency) ---
    if args.scrmsd:
        print('\n' + '=' * 60)
        print('Self-Consistency RMSD (ESMFold)')
        print('=' * 60)
        scrmsd_results = compute_scrmsd_batch(items, device=args.device)

        valid_cdr = [r['scrmsd_cdr'] for r in scrmsd_results
                     if not math.isnan(r.get('scrmsd_cdr', float('nan')))]
        valid_h = [r['scrmsd_h'] for r in scrmsd_results
                   if not math.isnan(r.get('scrmsd_h', float('nan')))]

        scrmsd_summary = {}
        if valid_cdr:
            scrmsd_summary['scrmsd_cdr'] = {'mean': float(np.mean(valid_cdr)), 'std': float(np.std(valid_cdr))}
            print(f'  scRMSD (CDR):   {np.mean(valid_cdr):.4f} +/- {np.std(valid_cdr):.4f}  '
                  f'({len(valid_cdr)}/{len(scrmsd_results)} valid)')
        if valid_h:
            scrmsd_summary['scrmsd_heavy'] = {'mean': float(np.mean(valid_h)), 'std': float(np.std(valid_h))}
            print(f'  scRMSD (heavy): {np.mean(valid_h):.4f} +/- {np.std(valid_h):.4f}  '
                  f'({len(valid_h)}/{len(scrmsd_results)} valid)')
        summary['scrmsd'] = scrmsd_summary

        # Merge scRMSD into per-sample metrics
        for i, r in enumerate(scrmsd_results):
            if i < len(metrics):
                metrics[i]['scrmsd_cdr'] = r.get('scrmsd_cdr', float('nan'))
                metrics[i]['scrmsd_h'] = r.get('scrmsd_h', float('nan'))

    # --- Save ---
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)

        csv_path = os.path.join(args.save_dir, 'per_sample_metrics.csv')
        with open(csv_path, 'w') as f:
            f.write('pdb,' + ','.join(metric_names) + ',pmetric\n')
            for pdb, m, pm in zip(pdbs, metrics, pmets):
                f.write(pdb + ',' + ','.join(str(m.get(n, '')) for n in metric_names) + ',' + str(pm) + '\n')
        print_log(f'Per-sample CSV: {csv_path}')

        summary_path = os.path.join(args.save_dir, 'summary_metrics.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print_log(f'Summary JSON: {summary_path}')

        jsonl_path = os.path.join(args.save_dir, 'per_sample_metrics.jsonl')
        with open(jsonl_path, 'w') as f:
            for pdb, m, pm in zip(pdbs, metrics, pmets):
                entry = {'pdb': pdb, 'pmetric': pm}
                entry.update(m)
                f.write(json.dumps(entry) + '\n')
        print_log(f'Per-sample JSONL: {jsonl_path}')

        target_csv = os.path.join(args.save_dir, 'per_target_metrics.csv')
        with open(target_csv, 'w') as f:
            hdr = 'pdb,n,' + ','.join(f'{n}_mean,{n}_std' for n in metric_names)
            hdr += ',n_clusters,n_unique,cluster_ratio'
            f.write(hdr + '\n')
            for pdb in sorted(per_target_stats.keys()):
                ns = per_target_stats[pdb][metric_names[0]]['n']
                row = f'{pdb},{ns}'
                for n in metric_names:
                    s = per_target_stats[pdb][n]
                    row += f',{s["mean"]},{s["std"]}'
                ds = diversity_stats.get(pdb, {})
                row += f',{ds.get("n_clusters", 0)},{ds.get("n_unique", 0)},{ds.get("cluster_ratio", 0)}'
                f.write(row + '\n')
        print_log(f'Per-target CSV: {target_csv}')

        if joint_results:
            with open(os.path.join(args.save_dir, 'joint_success.json'), 'w') as f:
                json.dump(joint_results, f, indent=2)
            print_log(f'Joint success JSON: {os.path.join(args.save_dir, "joint_success.json")}')

        if args.scrmsd:
            with open(os.path.join(args.save_dir, 'scrmsd_per_sample.jsonl'), 'w') as f:
                for r in scrmsd_results:
                    f.write(json.dumps(r) + '\n')
            print_log(f'scRMSD per-sample: {os.path.join(args.save_dir, "scrmsd_per_sample.jsonl")}')

    # --- Final summary table ---
    print('\n' + '=' * 60)
    print('FINAL SUMMARY')
    print('=' * 60)
    categories = {
        'Standard': ['AAR', 'CAAR', 'BLOSUM62', 'RMSD(CA) aligned', 'TMscore', 'LDDT', 'DockQ'],
        'Physics': ['interface_clashes_CA', 'interface_clashes_allatom', 'intra_cdr_clashes',
                    'bond_validity', 'rama_allowed_frac'],
        'PLM': ['esm2_ppl', 'esmfold_plddt'],
        'Designability': ['cdr_charge', 'cdr_hydrophobicity', 'ptm_liabilities_total', 'humanness_score'],
        'Energy': ['foldx_ddg', 'foldx_dg_model', 'foldx_dg_ref'],
        'scRMSD': ['scrmsd_cdr', 'scrmsd_h'],
    }
    for cat, names in categories.items():
        printed_header = False
        for name in names:
            if name in summary:
                if not printed_header:
                    print(f'\n--- {cat} ---')
                    printed_header = True
                s = summary[name]
                print(f'  {name:35s}: {s["mean"]:10.4f} (+/- {s["std"]:.4f})')

    if joint_results:
        print(f'\n--- Joint Success ---')
        for name, rate in joint_results.items():
            print(f'  {name:35s}: {rate:10.1%}')

    # Vendi + KL
    if 'vendi_sequence' in summary:
        print(f'\n--- Sequence Diversity ---')
        vs = summary['vendi_sequence']
        print(f'  {"Vendi (global)":35s}: {vs["global"]:10.4f}')
        print(f'  {"Vendi (per-target mean)":35s}: {vs["per_target_mean"]:10.4f}')
    if 'aa_distribution' in summary:
        ad = summary['aa_distribution']
        print(f'  {"KL(gen || natural)":35s}: {ad["kl_divergence"]:10.4f}')
        print(f'  {"Jensen-Shannon":35s}: {ad["jensen_shannon"]:10.4f}')


def parse():
    parser = argparse.ArgumentParser(description='Comprehensive antibody CDR design metrics')
    parser.add_argument('--test_set', type=str, required=True, help='Path to test set JSONL')
    parser.add_argument('--num_workers', type=int, default=4, help='Parallel workers')
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0', help='GPU device for PLM')
    parser.add_argument('--esm_batch_size', type=int, default=8, help='ESM-2 batch size')
    parser.add_argument('--skip_plm', action='store_true', help='Skip ESM-2 + ESMFold (faster)')
    parser.add_argument('--skip_esmfold', action='store_true', help='Skip ESMFold only')
    parser.add_argument('--foldx', action='store_true', help='Enable FoldX ddG (slow)')
    parser.add_argument('--scrmsd', action='store_true', help='Compute scRMSD via ESMFold')
    parser.add_argument('--numbering', type=str, default='imgt', choices=['imgt', 'chothia'],
                        help='Antibody numbering scheme (default: imgt)')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
