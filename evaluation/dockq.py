#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import re
import sys
import subprocess

from data.pdb_utils import AgAbComplex, Protein
from utils.time_sign import get_time_sign
from configs import CACHE_DIR


def dockq(mod_cplx: AgAbComplex, ref_cplx: AgAbComplex, cdrh3_only=False):
    """
    Compute DockQ score using modern DockQ API.
    
    Args:
        mod_cplx: Model antibody-antigen complex
        ref_cplx: Reference (native) complex
        cdrh3_only: If True, only evaluate CDRH3 (not used with modern DockQ)
    
    Returns:
        float: Total DockQ score across all interfaces
    """
    H, L = ref_cplx.heavy_chain, ref_cplx.light_chain
    
    prefix = get_time_sign(suffix=ref_cplx.get_id().replace('(', '').replace(')', ''))
    mod_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_mod.pdb')
    ref_pdb = os.path.join(CACHE_DIR, prefix + '_dockq_ref.pdb')
    
    try:
        if cdrh3_only:
            # Only evaluate CDRH3-antigen interface
            mod_cdr, ref_cdr = mod_cplx.get_cdr(), ref_cplx.get_cdr()
            mod_peptides, ref_peptides = mod_cplx.antigen.peptides.copy(), ref_cplx.antigen.peptides.copy()
            mod_peptides[H], ref_peptides[H] = mod_cdr, ref_cdr
            pdb = mod_cplx.get_id()
            mod_cplx_temp, ref_cplx_temp = Protein(pdb, mod_peptides), Protein(pdb, ref_peptides)
            mod_cplx_temp.to_pdb(mod_pdb)
            ref_cplx_temp.to_pdb(ref_pdb)
        else:
            mod_cplx.to_pdb(mod_pdb)
            ref_cplx.to_pdb(ref_pdb)
        
        # Use modern DockQ command (python -m DockQ)
        cmd = [
            sys.executable, '-m', 'DockQ',
            mod_pdb, ref_pdb,
            '--allowed_mismatches', '50',  # Handle CDR mutations
            '--short'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        text = result.stdout + result.stderr
        
        if result.returncode != 0:
            # DockQ failed - return NaN instead of crashing
            return float('nan')
        
        # Parse "Total DockQ over N native interfaces: X.XXX"
        res = re.search(r'Total DockQ over \d+ native interfaces:\s+([0-9]+\.?[0-9]*)', text)
        if res is None:
            # Fallback: find first "DockQ X.XXX" line
            res = re.search(r'DockQ\s+([0-9]+\.?[0-9]*)', text)
        
        if res is None:
            # Cannot parse - return NaN
            return float('nan')
        
        score = float(res.group(1))
        return score
        
    finally:
        # Clean up temp files
        if os.path.exists(mod_pdb):
            os.remove(mod_pdb)
        if os.path.exists(ref_pdb):
            os.remove(ref_pdb)