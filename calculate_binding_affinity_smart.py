#!/usr/bin/env python3
"""
Calculate binding affinity for antibody-antigen complexes using PyRosetta.
Smart version that only considers chains with >100 atoms as antigen.
"""
import os
import sys
import argparse
from pathlib import Path
import csv
from typing import List, Set, Dict

# pyright: reportMissingImports=false
import pyrosetta
from pyrosetta.rosetta.protocols.analysis import InterfaceAnalyzerMover
from Bio.PDB import PDBParser


def init_pyrosetta():
    """Initialize PyRosetta with appropriate settings."""
    pyrosetta.init(' '.join([
        '-mute', 'all',
        '-use_input_sc',
        '-ignore_unrecognized_res',
        '-ignore_zero_occupancy', 'false',
        '-load_PDB_components', 'false',
        '-relax:default_repeats', '2',
        '-no_fconfig',
    ]))


def pyrosetta_interface_energy(pdb_path: str, interface: str, relax: bool = True) -> Dict[str, float]:
    """
    Calculate interface energy using PyRosetta.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    interface : str
        Interface definition (e.g., "AB_C" for chains AB vs C)
    relax : bool
        Whether to relax the structure before calculating energy (default: True)
    
    Returns
    -------
    Dict[str, float]
        Dictionary with dG values and optionally relaxed energy
    """
    try:
        from pyrosetta.rosetta.protocols.relax import FastRelax
        from pyrosetta import get_fa_scorefxn
        
        # Try to load pose with error checking
        pose = pyrosetta.pose_from_pdb(pdb_path)
        
        # Check if pose is valid
        if pose.size() == 0:
            raise ValueError("Loaded pose has no residues")
        
        results = {}
        
        # Get initial score
        scorefxn = get_fa_scorefxn()
        initial_score = scorefxn(pose)
        results['initial_score'] = initial_score
        
        if relax:
            print(f"    Relaxing structure (initial score: {initial_score:.2f})...", flush=True)
            # Create FastRelax mover with constrained backbone
            relax_mover = FastRelax()
            relax_mover.set_scorefxn(scorefxn)
            relax_mover.constrain_relax_to_start_coords(True)
            relax_mover.coord_constrain_sidechains(True)
            relax_mover.ramp_down_constraints(False)
            
            # Apply relaxation
            relax_mover.apply(pose)
            
            # Get relaxed score
            relaxed_score = scorefxn(pose)
            results['relaxed_score'] = relaxed_score
            print(f"    Relaxed (final score: {relaxed_score:.2f}, Δ={relaxed_score-initial_score:.2f})", flush=True)
        
        # Calculate interface energy
        print(f"    Calculating interface energy...", flush=True)
        mover = InterfaceAnalyzerMover(interface)
        mover.set_pack_separated(True)
        
        # Apply the mover
        mover.apply(pose)
        
        # Get the dG_separated score
        if 'dG_separated' in pose.scores:
            results['dG_separated'] = pose.scores['dG_separated']
        else:
            print(f"Warning: dG_separated not found in scores for {pdb_path}", file=sys.stderr)
            results['dG_separated'] = float('nan')
        
        return results
            
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"Error in interface energy calculation for {pdb_path}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return {'dG_separated': float('nan'), 'error': str(e)}


def get_chain_info_from_pdb(pdb_path: str) -> Dict[str, int]:
    """
    Get chain IDs and their atom counts from a PDB file.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    
    Returns
    -------
    Dict[str, int]
        Dictionary mapping chain ID to atom count
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    chain_counts = {}
    for model in structure:
        for chain in model:
            atom_count = len([atom for atom in chain.get_atoms()])
            chain_counts[chain.id] = atom_count
    return chain_counts


def detect_interface(pdb_path: str, antibody_chains: List[str] = None, min_antigen_atoms: int = 100) -> str:
    """
    Detect interface definition from PDB file.
    Only considers chains with more than min_antigen_atoms as antigen.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file
    antibody_chains : List[str], optional
        List of antibody chain IDs. If None, assumes first 2 chains are antibody.
    min_antigen_atoms : int
        Minimum number of atoms for a chain to be considered as antigen (default: 100)
    
    Returns
    -------
    str
        Interface definition (e.g., "AB_C")
    """
    chain_info = get_chain_info_from_pdb(pdb_path)
    all_chains = set(chain_info.keys())
    
    if antibody_chains is None:
        # Assume first 2 chains are antibody (heavy and light)
        sorted_chains = sorted(all_chains)
        if len(sorted_chains) >= 2:
            antibody_chains = sorted_chains[:2]
        else:
            raise ValueError(f"PDB file must have at least 2 chains, found: {sorted_chains}")
    
    # Only consider chains with sufficient atoms as antigen
    antigen_chains = [
        c for c in all_chains 
        if c not in antibody_chains and chain_info[c] >= min_antigen_atoms
    ]
    
    if not antigen_chains:
        # If no large chains found, use all non-antibody chains
        print(f"Warning: No chains with >{min_antigen_atoms} atoms. Using all non-antibody chains.", file=sys.stderr)
        antigen_chains = [c for c in all_chains if c not in antibody_chains]
    
    antibody_str = ''.join(sorted(antibody_chains))
    antigen_str = ''.join(sorted(antigen_chains))
    
    if not antigen_str:
        raise ValueError(f"No antigen chains found. All chains: {all_chains}, Antibody: {antibody_chains}")
    
    # Print chain info
    print(f"  Chain info:", flush=True)
    for chain_id in sorted(all_chains):
        chain_type = "Ab" if chain_id in antibody_chains else "Ag" if chain_id in antigen_chains else "Skip"
        print(f"    {chain_id}: {chain_info[chain_id]:4d} atoms [{chain_type}]", flush=True)
    
    return f"{antibody_str}_{antigen_str}"


def process_pdb_file(pdb_path: str, antibody_chains: List[str] = None, reference_path: str = None, 
                     min_antigen_atoms: int = 100, relax: bool = True) -> dict:
    """
    Process a single PDB file and calculate binding affinity.
    
    Parameters
    ----------
    pdb_path : str
        Path to PDB file to analyze
    antibody_chains : List[str], optional
        List of antibody chain IDs (e.g., ['A', 'B'])
    reference_path : str, optional
        Path to reference PDB for comparison
    min_antigen_atoms : int
        Minimum atoms for antigen chain (default: 100)
    relax : bool
        Whether to relax structures before calculating energy (default: True)
    
    Returns
    -------
    dict
        Dictionary with results
    """
    try:
        interface = detect_interface(pdb_path, antibody_chains, min_antigen_atoms)
        print(f"  Interface: {interface}", flush=True)
        
        energy_results = pyrosetta_interface_energy(pdb_path, interface, relax=relax)
        
        result = {
            'pdb_file': Path(pdb_path).name,
            'interface': interface,
        }
        result.update(energy_results)
        
        if reference_path:
            print(f"  Processing reference structure...", flush=True)
            ref_results = pyrosetta_interface_energy(reference_path, interface, relax=relax)
            result['dG_ref'] = ref_results.get('dG_separated', float('nan'))
            if 'dG_separated' in result and 'dG_separated' in ref_results:
                result['ddG'] = result['dG_separated'] - ref_results['dG_separated']
        
        if 'dG_separated' in result:
            print(f"  → dG_separated = {result['dG_separated']:.2f}", flush=True)
        
        return result
        
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  ✗ Error: {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        return {
            'pdb_file': Path(pdb_path).name,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(
        description='Calculate binding affinity for PDB files using PyRosetta (smart version)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all PDB files in a folder
  python calculate_binding_affinity_smart.py input_folder/ --antibody-chains A B
  
  # Compare against a reference structure
  python calculate_binding_affinity_smart.py input_folder/ --reference reference.pdb --antibody-chains A B
  
  # Save results to CSV
  python calculate_binding_affinity_smart.py input_folder/ --output results.csv --antibody-chains A B
  
  # Adjust minimum antigen size
  python calculate_binding_affinity_smart.py input_folder/ --antibody-chains A B --min-antigen-atoms 50
        """
    )
    parser.add_argument('input_folder', type=str,
                        help='Folder containing PDB files to analyze')
    parser.add_argument('--antibody-chains', nargs='+', default=None,
                        help='Antibody chain IDs (e.g., A B). If not specified, assumes first 2 chains.')
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference PDB file for ddG calculation')
    parser.add_argument('--output', type=str, default=None,
                        help='Output CSV file for results (default: print to stdout)')
    parser.add_argument('--pattern', type=str, default='*.pdb',
                        help='File pattern to match (default: *.pdb)')
    parser.add_argument('--min-antigen-atoms', type=int, default=100,
                        help='Minimum atoms for a chain to be considered antigen (default: 100)')
    parser.add_argument('--no-relax', action='store_true',
                        help='Skip relaxation step before calculating binding energy')
    
    args = parser.parse_args()
    
    # Initialize PyRosetta
    print("Initializing PyRosetta...")
    init_pyrosetta()
    print("PyRosetta initialized.\n")
    
    # Find all PDB files
    input_path = Path(args.input_folder)
    if not input_path.exists():
        print(f"Error: Input folder '{args.input_folder}' does not exist", file=sys.stderr)
        sys.exit(1)
    
    pdb_files = sorted(input_path.glob(args.pattern))
    if not pdb_files:
        print(f"Error: No files matching pattern '{args.pattern}' found in '{args.input_folder}'", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(pdb_files)} PDB file(s) to process\n")
    
    # Process each PDB file
    results = []
    relax = not args.no_relax
    if relax:
        print("Relaxation enabled (use --no-relax to skip)\n")
    else:
        print("Relaxation disabled\n")
    
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"[{i}/{len(pdb_files)}] {pdb_file.name}", flush=True)
        result = process_pdb_file(
            str(pdb_file),
            antibody_chains=args.antibody_chains,
            reference_path=args.reference,
            min_antigen_atoms=args.min_antigen_atoms,
            relax=relax
        )
        results.append(result)
        print()
    
    # Output results
    if results:
        # Determine all keys
        all_keys = set()
        for result in results:
            all_keys.update(result.keys())
        all_keys = sorted(all_keys)
        
        if args.output:
            # Write to CSV
            with open(args.output, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"✓ Results written to {args.output}")
            
            # Print summary
            successful = sum(1 for r in results if 'dG' in r and not ('error' in r))
            print(f"\nSummary: {successful}/{len(results)} files processed successfully")
        else:
            # Print to stdout
            print("\n" + "="*80)
            print("RESULTS SUMMARY")
            print("="*80)
            
            # Print as table
            col_widths = {key: max(len(key), max(len(str(r.get(key, ''))) for r in results)) 
                         for key in all_keys}
            
            # Header
            header = " | ".join(key.ljust(col_widths[key]) for key in all_keys)
            print(header)
            print("-" * len(header))
            
            # Rows
            for result in results:
                row = " | ".join(str(result.get(key, '')).ljust(col_widths[key]) for key in all_keys)
                print(row)
            
            successful = sum(1 for r in results if 'dG' in r and not ('error' in r))
            print(f"\nSummary: {successful}/{len(results)} files processed successfully")


if __name__ == '__main__':
    main()
