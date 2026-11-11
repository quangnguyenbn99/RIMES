#!/usr/bin/env python3
"""
Combined pipeline: OpenMM relaxation + PyRosetta binding affinity scoring.

Stages:
  1. relax  - OpenMM force-field minimization on all PDBs in input folder
  2. score  - PyRosetta InterfaceAnalyzerMover dG on PDBs (optionally with FastRelax)
  3. all    - run relax then score (score reads from the relaxed folder)

Usage:
    # Full pipeline, relaxed PDBs saved alongside input
    python relax_and_score.py input/ --run all --save-in-input \
        --output results.csv --antibody-chains A B

    # Relax only, write to a separate folder
    python relax_and_score.py input/ --run relax \
        --relaxed-folder input_relaxed/ --gpus 0,1 --parallel 8

    # Score only (PDBs already relaxed)
    python relax_and_score.py input_relaxed/ --run score \
        --output results.csv --antibody-chains A B
"""

import argparse
import sys
from pathlib import Path

from relax_structures_gpu import relax_folder_gpu


def run_score_stage(input_folder, output_csv, antibody_chains, reference, pattern,
                    min_antigen_atoms, pyrosetta_relax):
    """Run PyRosetta binding-affinity scoring on a folder of PDBs."""
    import csv
    from calculate_binding_affinity_smart import init_pyrosetta, process_pdb_file

    input_path = Path(input_folder)
    pdb_files = sorted(input_path.glob(pattern))
    if not pdb_files:
        print(f"Error: No files matching '{pattern}' in '{input_folder}'", file=sys.stderr)
        return 1

    print("Initializing PyRosetta...")
    init_pyrosetta()
    print(f"Found {len(pdb_files)} PDB file(s) to score")
    print(f"PyRosetta FastRelax: {'ON' if pyrosetta_relax else 'OFF'}\n")

    results = []
    for i, pdb_file in enumerate(pdb_files, 1):
        print(f"[{i}/{len(pdb_files)}] {pdb_file.name}", flush=True)
        result = process_pdb_file(
            str(pdb_file),
            antibody_chains=antibody_chains,
            reference_path=reference,
            min_antigen_atoms=min_antigen_atoms,
            relax=pyrosetta_relax,
        )
        results.append(result)
        print()

    if not results:
        return 1

    all_keys = sorted({k for r in results for k in r.keys()})

    if output_csv:
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            writer.writerows(results)
        print(f"Results written to {output_csv}")
    else:
        col_widths = {k: max(len(k), max(len(str(r.get(k, ''))) for r in results))
                      for k in all_keys}
        header = " | ".join(k.ljust(col_widths[k]) for k in all_keys)
        print("\n" + "=" * len(header))
        print(header)
        print("-" * len(header))
        for r in results:
            print(" | ".join(str(r.get(k, '')).ljust(col_widths[k]) for k in all_keys))

    successful = sum(1 for r in results if 'dG_separated' in r and 'error' not in r)
    print(f"\nSummary: {successful}/{len(results)} scored successfully")
    return 0 if successful == len(results) else 1


def main():
    parser = argparse.ArgumentParser(
        description='OpenMM relax + PyRosetta binding-affinity pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument('input_folder', type=str,
                        help='Input folder containing PDB files')

    parser.add_argument('--run', choices=['relax', 'score', 'all'], default='all',
                        help='Which stage to run (default: all)')

    # Output locations
    parser.add_argument('--relaxed-folder', type=str, default=None,
                        help='Output folder for relaxed PDBs '
                             '(default: <input_folder>_relaxed, or input folder if --save-in-input)')
    parser.add_argument('--save-in-input', action='store_true',
                        help='Write relaxed PDBs into the input folder (alongside originals)')
    parser.add_argument('--output', type=str, default=None,
                        help='CSV file for scoring results (default: stdout table)')

    # Relax stage knobs
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs (default: 0)')
    parser.add_argument('--parallel', type=int, default=10,
                        help='Parallel relax processes (default: 10)')
    parser.add_argument('--stiffness', type=float, default=10.0,
                        help='Position restraint kcal/mol/Å² (default: 10.0)')
    parser.add_argument('--max-iterations', type=int, default=0,
                        help='Max minimization iterations (0 = until convergence)')
    parser.add_argument('--keep-heterogens', action='store_true',
                        help='Keep glycans/ligands during relaxation')

    # Score stage knobs
    parser.add_argument('--antibody-chains', nargs='+', default=None,
                        help='Antibody chain IDs (e.g. A B). Default: first 2 chains')
    parser.add_argument('--reference', type=str, default=None,
                        help='Reference PDB for ddG calculation')
    parser.add_argument('--pattern', type=str, default='*.pdb',
                        help='Glob pattern for scoring input (default: *.pdb)')
    parser.add_argument('--min-antigen-atoms', type=int, default=100,
                        help='Min atoms for a chain to count as antigen (default: 100)')
    parser.add_argument('--pyrosetta-relax', action='store_true',
                        help='Also run PyRosetta FastRelax before scoring '
                             '(off by default since OpenMM already relaxed)')

    args = parser.parse_args()

    input_folder = Path(args.input_folder)
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory", file=sys.stderr)
        sys.exit(1)

    # Resolve relaxed folder
    if args.save_in_input:
        relaxed_folder = input_folder
    elif args.relaxed_folder:
        relaxed_folder = Path(args.relaxed_folder)
    else:
        relaxed_folder = input_folder.parent / f"{input_folder.name}_relaxed"

    # --- Stage 1: relax ---
    if args.run in ('relax', 'all'):
        print(f"=== Stage: RELAX ===")
        print(f"Input:   {input_folder}")
        print(f"Output:  {relaxed_folder}")
        gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
        summary = relax_folder_gpu(
            input_folder,
            relaxed_folder,
            gpu_ids=gpu_ids,
            num_processes=args.parallel,
            stiffness=args.stiffness,
            max_iterations=args.max_iterations,
            remove_heterogens=not args.keep_heterogens,
        )
        if summary['failed'] == summary['total']:
            print("All relaxations failed — aborting.", file=sys.stderr)
            sys.exit(1)

    # --- Stage 2: score ---
    if args.run in ('score', 'all'):
        print(f"\n=== Stage: SCORE ===")
        # For 'all' we score the relaxed folder; for 'score' alone we score the input folder.
        score_input = relaxed_folder if args.run == 'all' else input_folder
        # When saving in input, relaxed files end with _relaxed.pdb — score those.
        score_pattern = '*_relaxed.pdb' if (args.run == 'all' and args.save_in_input) else args.pattern
        print(f"Input:   {score_input}")
        print(f"Pattern: {score_pattern}")
        rc = run_score_stage(
            score_input,
            args.output,
            antibody_chains=args.antibody_chains,
            reference=args.reference,
            pattern=score_pattern,
            min_antigen_atoms=args.min_antigen_atoms,
            pyrosetta_relax=args.pyrosetta_relax,
        )
        sys.exit(rc)


if __name__ == '__main__':
    main()
