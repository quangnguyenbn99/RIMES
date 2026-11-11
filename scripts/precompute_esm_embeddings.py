#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Precompute per-residue embeddings from ESM-2 or ESM C models.

Supports two model families (auto-detected by model name):
  - ESM-2  (fair-esm package, Python 3.8+): esm2_t30_150M, esm2_t33_650M, esm2_t36_3B
  - ESM C  (esm package, Python 3.10+):     esmc_300m, esmc_600m

Usage:
    python -m scripts.precompute_esm_embeddings \
        --dataset_path data/RAbD/train.json \
        --output_path data/RAbD/train_esmc300_embeddings.pt \
        --esm_model esmc_300m \
        --device cuda:0

The output .pt file maps complex keys to [n_residues, esm_dim] float16
tensors whose residue order matches E2EDataset.__getitem__.
"""
import os
import sys
import json
import argparse

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.pdb_utils import AgAbComplex, VOCAB


VALID_AA = set('ACDEFGHIKLMNPQRSTVWY')

ESM2_MODELS = {
    'esm2_t30_150M': ('esm2_t30_150M_UR50D', 30,  640),
    'esm2_t33_650M': ('esm2_t33_650M_UR50D', 33, 1280),
    'esm2_t36_3B':   ('esm2_t36_3B_UR50D',   36, 2560),
}

ESMC_MODELS = {
    'esmc_300m': ('esmc_300m',  960),
    'esmc_600m': ('esmc_600m', 1152),
}

ALL_MODELS = list(ESM2_MODELS.keys()) + list(ESMC_MODELS.keys())


def get_complex_key(cplx):
    return f"{cplx.pdb_id}_{cplx.heavy_chain}_{cplx.light_chain}"


MASK_TOKEN = '<mask>'


def sanitize_seq(symbols):
    """Replace non-standard residue symbols with 'X'."""
    return ''.join(s if s in VALID_AA else 'X' for s in symbols)


def _apply_cdr_mask(cplx, chain_seq, chain_kind, mask_cdrs):
    """Splice the ESM mask token into CDR positions of a chain sequence.

    Args:
        cplx:        AgAbComplex (provides .get_cdr_pos).
        chain_seq:   sanitized 1-char-per-residue sequence string.
        chain_kind:  'H' for heavy, 'L' for light.
        mask_cdrs:   iterable of CDR ids, e.g. ['H3'] or ['H1','H2','H3','L1','L2','L3'].

    Returns the spliced sequence (a list, one element per residue, where masked
    residues are MASK_TOKEN and the rest are single-char AAs). Residue count is
    preserved.
    """
    out = list(chain_seq)
    for cdr in mask_cdrs:
        cdr = cdr.upper()
        if not cdr.startswith(chain_kind):
            continue
        try:
            start, end = cplx.get_cdr_pos(cdr)
        except Exception:
            continue
        for i in range(start, end + 1):
            if 0 <= i < len(out):
                out[i] = MASK_TOKEN
    return out


def extract_chain_sequences(cplx, full_antigen=False, mask_cdrs=None):
    """Extract residue sequences matching E2EDataset.__getitem__ layout.

    If ``mask_cdrs`` is non-empty, CDR-position residues in the heavy/light
    chains are replaced with the ESM ``<mask>`` token to prevent ESM-2's
    bidirectional attention from leaking the native CDR identity into
    framework embeddings.
    """
    ag_symbols = []
    if full_antigen:
        ag = cplx.get_antigen()
        for chain_name in ag.get_chain_names():
            chain = ag.get_chain(chain_name)
            for i in range(len(chain)):
                ag_symbols.append(chain.get_residue(i).get_symbol())
    else:
        for residue, chain_name, i in cplx.get_epitope():
            ag_symbols.append(residue.get_symbol())

    hc = cplx.get_heavy_chain()
    hc_symbols = [hc.get_residue(i).get_symbol() for i in range(len(hc))]

    lc = cplx.get_light_chain()
    lc_symbols = [lc.get_residue(i).get_symbol() for i in range(len(lc))]

    ag_seq = sanitize_seq(ag_symbols)
    hc_seq = sanitize_seq(hc_symbols)
    lc_seq = sanitize_seq(lc_symbols)

    if mask_cdrs:
        hc_tokens = _apply_cdr_mask(cplx, hc_seq, 'H', mask_cdrs)
        lc_tokens = _apply_cdr_mask(cplx, lc_seq, 'L', mask_cdrs)
        hc_seq = ''.join(hc_tokens)
        lc_seq = ''.join(lc_tokens)

    return ag_seq, hc_seq, lc_seq


# ---------------------------------------------------------------------------
# ESM-2 backend (fair-esm)
# ---------------------------------------------------------------------------
class ESM2Backend:
    def __init__(self, model_name, device):
        import esm as esm_lib
        full_name, self.repr_layer, self.esm_dim = ESM2_MODELS[model_name]
        loader_fn = getattr(esm_lib.pretrained, full_name)
        model, alphabet = loader_fn()
        self.model = model.eval().to(device)
        self.batch_converter = alphabet.get_batch_converter()
        self.device = device

    def embed(self, sequences, residue_lengths):
        """Embed sequences with explicit per-sequence residue counts.

        Args:
            sequences:        list of (label, seq_str). seq_str may contain MASK_TOKEN.
            residue_lengths:  list of expected residue counts (one per sequence).
                              fair-esm tokenises ``<mask>`` as a single token, so the
                              representation length equals the residue count regardless
                              of how many character positions the mask substring spans.

        Returns a list of [L, D] half tensors of length len(sequences).
        """
        if not sequences:
            return []
        assert len(residue_lengths) == len(sequences)
        batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
        batch_tokens = batch_tokens.to(self.device)
        with torch.no_grad():
            results = self.model(batch_tokens, repr_layers=[self.repr_layer])
        reps = results["representations"][self.repr_layer]
        out = []
        for i, n_res in enumerate(residue_lengths):
            out.append(reps[i, 1:n_res + 1, :].cpu().half())
        return out


# ---------------------------------------------------------------------------
# ESM C backend (evolutionaryscale esm)
# ---------------------------------------------------------------------------
class ESMCBackend:
    def __init__(self, model_name, device):
        from esm.models.esmc import ESMC
        hf_name, self.esm_dim = ESMC_MODELS[model_name]
        self.client = ESMC.from_pretrained(hf_name).to(device)
        self.device = device

    def embed(self, sequences, residue_lengths):
        """Embed sequences. ESM-C tokenisation of ``<mask>`` is not validated
        in this script; ``--mask_cdr`` is therefore restricted to ESM-2 backends
        upstream in :func:`main`. residue_lengths is accepted for API parity."""
        from esm.sdk.api import ESMProtein, LogitsConfig
        out = []
        for (_, seq), n_res in zip(sequences, residue_lengths):
            protein = ESMProtein(sequence=seq)
            protein_tensor = self.client.encode(protein)
            with torch.no_grad():
                result = self.client.logits(
                    protein_tensor, LogitsConfig(return_embeddings=True))
            emb = result.embeddings[0, 1:-1, :]   # strip BOS/EOS
            assert emb.shape[0] == n_res, \
                f"Length mismatch: emb {emb.shape[0]} vs n_res {n_res}"
            out.append(emb.cpu().half())
        return out


def make_backend(model_name, device):
    if model_name in ESM2_MODELS:
        return ESM2Backend(model_name, device)
    elif model_name in ESMC_MODELS:
        return ESMCBackend(model_name, device)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from {ALL_MODELS}")


def _residue_count(seq):
    """Count residues in a sequence string, treating MASK_TOKEN as a single residue."""
    if MASK_TOKEN not in seq:
        return len(seq)
    # Each occurrence of MASK_TOKEN spans len(MASK_TOKEN) chars but counts as 1 residue.
    n_masks = seq.count(MASK_TOKEN)
    return len(seq) - n_masks * (len(MASK_TOKEN) - 1)


def main():
    parser = argparse.ArgumentParser(
        description='Precompute ESM-2 / ESM C embeddings')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to JSONL dataset')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Where to save the .pt embedding file')
    parser.add_argument('--esm_model', type=str, default='esmc_300m',
                        choices=ALL_MODELS,
                        help='Model to use for embeddings')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--full_antigen', action='store_true')
    parser.add_argument(
        '--mask_cdr', type=str, default=None,
        help=("Comma-separated CDR ids to mask before ESM extraction, "
              "e.g. 'H3' or 'H1,H2,H3,L1,L2,L3'. CDR-position residues are "
              "replaced with the ESM <mask> token so framework embeddings "
              "are not contaminated by the native CDR via bidirectional "
              "self-attention. Only supported with the ESM-2 backend."))
    args = parser.parse_args()

    mask_cdrs = []
    if args.mask_cdr:
        mask_cdrs = [c.strip().upper() for c in args.mask_cdr.split(',') if c.strip()]
        if args.esm_model not in ESM2_MODELS:
            raise ValueError(
                "--mask_cdr is only supported with ESM-2 backends "
                f"({list(ESM2_MODELS)}); got {args.esm_model}.")
        print(f"CDR masking enabled: {mask_cdrs}")

    print(f"Loading {args.esm_model}...")
    backend = make_backend(args.esm_model, args.device)
    esm_dim = backend.esm_dim
    print(f"  dim={esm_dim}")

    with open(args.dataset_path, 'r') as f:
        lines = f.read().strip().split('\n')

    all_embeddings = {}
    errors = []
    zero = torch.zeros(1, esm_dim, dtype=torch.half)

    for line in tqdm(lines, desc=f"Precomputing {args.esm_model}"):
        item = json.loads(line)
        try:
            cplx = AgAbComplex.from_pdb(
                item['pdb_data_path'], item['heavy_chain'],
                item['light_chain'], item['antigen_chains'])
        except Exception as e:
            errors.append((item.get('pdb', 'unknown'), str(e)))
            continue

        ag_seq, hc_seq, lc_seq = extract_chain_sequences(
            cplx, args.full_antigen, mask_cdrs=mask_cdrs)

        seqs_to_embed = []
        residue_lengths = []
        for label, seq in [('ag', ag_seq), ('hc', hc_seq), ('lc', lc_seq)]:
            if len(seq) > 0:
                seqs_to_embed.append((label, seq))
                residue_lengths.append(_residue_count(seq))

        embs = backend.embed(seqs_to_embed, residue_lengths)
        chain_emb = {label: e for (label, _), e in zip(seqs_to_embed, embs)}

        parts = []
        for chain_key in ['ag', 'hc', 'lc']:
            parts.append(zero)
            if chain_key in chain_emb:
                parts.append(chain_emb[chain_key])
        full_emb = torch.cat(parts, dim=0)

        key = get_complex_key(cplx)
        all_embeddings[key] = full_emb

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save({
        'embeddings': all_embeddings,
        'esm_model': args.esm_model,
        'esm_dim': esm_dim,
        'full_antigen': args.full_antigen,
        'mask_cdr': mask_cdrs,
    }, args.output_path)

    print(f"\nSaved {len(all_embeddings)} embeddings to {args.output_path}")
    if mask_cdrs:
        print(f"  CDRs masked before ESM extraction: {mask_cdrs}")
    if errors:
        print(f"Skipped {len(errors)} entries due to errors:")
        for pdb, err in errors[:10]:
            print(f"  {pdb}: {err}")


if __name__ == '__main__':
    main()
