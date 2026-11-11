#!/usr/bin/env python3
"""
GPU-Accelerated Structural Relaxation using OpenMM
Much faster than PyRosetta - uses CUDA GPUs for force field minimization

Usage:
    python relax_structures_gpu.py input_folder output_folder [--gpus 0,1,2,3,4] [--parallel 10]
    
Example:
    python relax_structures_gpu.py designs/ designs_relaxed/ --gpus 0,1,2,3,4 --parallel 20
"""

import argparse
import sys
import os
import io
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
import pdbfixer
import openmm
from openmm import app as openmm_app
from openmm import unit

ENERGY = unit.kilocalories_per_mole
LENGTH = unit.angstroms


class ForceFieldMinimizer(object):

    def __init__(self, stiffness=10.0, max_iterations=0, tolerance=2.39*unit.kilojoule_per_mole/unit.nanometer, platform='CPU'):
        super().__init__()
        self.stiffness = stiffness
        self.max_iterations = max_iterations
        # Ensure tolerance has force units (kJ/mol/nm)
        force_unit = unit.kilojoule_per_mole/unit.nanometer
        if isinstance(tolerance, (int, float)):
            self.tolerance = tolerance * force_unit
        else:
            try:
                self.tolerance = tolerance.in_units_of(force_unit)
            except Exception:
                # Fallback to a sensible default (≈1 kcal/mol/Å)
                self.tolerance = 2.39 * force_unit
        assert platform in ('CUDA', 'CPU', 'OpenCL')
        self.platform = platform

    def _fix(self, pdb_str, remove_heterogens=True):
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(pdb_str))
        
        # Remove heterogens (ligands, glycans, etc.) that don't have force field parameters
        if remove_heterogens:
            fixer.removeHeterogens(keepWater=False)
        
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()

        fixer.findMissingResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms(seed=0)
        fixer.addMissingHydrogens()

        out_handle = io.StringIO()
        openmm_app.PDBFile.writeFile(fixer.topology, fixer.positions, out_handle, keepIds=True)
        return out_handle.getvalue()

    def _get_pdb_string(self, topology, positions):
        with io.StringIO() as f:
            openmm_app.PDBFile.writeFile(topology, positions, f, keepIds=True)
            return f.getvalue()
        
    def _minimize(self, pdb_str):
        pdb = openmm_app.PDBFile(io.StringIO(pdb_str))

        force_field = openmm_app.ForceField("amber14-all.xml")
        constraints = openmm_app.HBonds
        system = force_field.createSystem(pdb.topology, constraints=constraints)

        # Add constraints to non-generated regions
        force = openmm.CustomExternalForce("0.5 * k * ((x-x0)^2 + (y-y0)^2 + (z-z0)^2)")
        force.addGlobalParameter("k", self.stiffness)
        for p in ["x0", "y0", "z0"]:
            force.addPerParticleParameter(p)
        
        for i, a in enumerate(pdb.topology.atoms()):
            if a.element.name != 'hydrogen':
                force.addParticle(i, pdb.positions[i])
                
        system.addForce(force)

        # Set up the integrator and simulation
        integrator = openmm.LangevinIntegrator(0, 0.01, 0.0)
        platform = openmm.Platform.getPlatformByName(self.platform)
        simulation = openmm_app.Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)

        # Perform minimization
        ret = {}
        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["einit"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["posinit"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)

        simulation.minimizeEnergy(maxIterations=self.max_iterations, tolerance=self.tolerance)

        state = simulation.context.getState(getEnergy=True, getPositions=True)
        ret["efinal"] = state.getPotentialEnergy().value_in_unit(ENERGY)
        ret["pos"] = state.getPositions(asNumpy=True).value_in_unit(LENGTH)
        ret["min_pdb"] = self._get_pdb_string(simulation.topology, state.getPositions())

        return ret['min_pdb'], ret
    
    def _add_energy_remarks(self, pdb_str, ret):
        pdb_lines = pdb_str.splitlines()
        pdb_lines.insert(1, "REMARK   1  FINAL ENERGY:   {:.3f} KCAL/MOL".format(ret['efinal']))
        pdb_lines.insert(1, "REMARK   1  INITIAL ENERGY: {:.3f} KCAL/MOL".format(ret['einit']))
        return "\n".join(pdb_lines)

    def __call__(self, pdb_str, out_path, return_info=True, remove_heterogens=True):
        if '\n' not in pdb_str and pdb_str.lower().endswith(".pdb"):
            with open(pdb_str) as f:
                pdb_str = f.read()

        pdb_fixed = self._fix(pdb_str, remove_heterogens=remove_heterogens)
        pdb_min, ret = self._minimize(pdb_fixed)
        pdb_min = self._add_energy_remarks(pdb_min, ret)
        with open(out_path, 'w') as f:
            f.write(pdb_min)
        if return_info:
            return pdb_min, ret
        else:
            return pdb_min


def relax_single_structure_gpu(pdb_file, output_dir, gpu_id, stiffness=10.0, max_iterations=0, remove_heterogens=True):
    """
    Relax a single PDB structure using OpenMM on a specific GPU
    
    Parameters:
    -----------
    pdb_file : str or Path
        Path to input PDB file
    output_dir : str or Path
        Directory to save relaxed structure
    gpu_id : int
        GPU device ID to use
    stiffness : float
        Force constant for position restraints (kcal/mol/Å²)
    max_iterations : int
        Maximum minimization iterations (0 = until convergence)
    remove_heterogens : bool
        Remove heterogens (glycans, ligands, etc.) before relaxation
        
    Returns:
    --------
    tuple : (input_file, output_file, success, error_message, energy_info)
    """
    pdb_file = Path(pdb_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set GPU for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    output_file = output_dir / pdb_file.name.replace('.pdb', '_relaxed.pdb')
    if output_file.suffix != '.pdb':
        output_file = output_dir / f"{pdb_file.stem}_relaxed.pdb"
    
    platform = os.environ.get('OPENMM_PLATFORM', 'CUDA')
    try:
        minimizer = ForceFieldMinimizer(
            stiffness=stiffness,
            max_iterations=max_iterations,
            platform=platform,
        )
        
        pdb_min, ret = minimizer(str(pdb_file), str(output_file), return_info=True, remove_heterogens=remove_heterogens)
        
        energy_info = {
            'einit': ret['einit'],
            'efinal': ret['efinal'],
            'delta': ret['efinal'] - ret['einit']
        }
        
        return (str(pdb_file), str(output_file), True, None, energy_info)
        
    except Exception as e:
        error_msg = f"Error processing {pdb_file}: {str(e)}"
        return (str(pdb_file), None, False, error_msg, None)


def relax_folder_gpu(input_folder, output_folder, gpu_ids, num_processes=10, stiffness=10.0, max_iterations=0, remove_heterogens=True):
    """
    Relax all PDB files in a folder using multiple GPUs in parallel
    
    Parameters:
    -----------
    input_folder : str or Path
        Input directory containing PDB files
    output_folder : str or Path
        Output directory for relaxed structures
    gpu_ids : list of int
        List of GPU device IDs to use
    num_processes : int
        Number of parallel processes (distributed across GPUs)
    stiffness : float
        Force constant for position restraints
    max_iterations : int
        Maximum minimization iterations
    remove_heterogens : bool
        Remove heterogens (glycans, ligands) before relaxation
        
    Returns:
    --------
    dict : Summary statistics
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find all PDB files
    pdb_files = list(input_folder.glob('*.pdb'))
    
    if not pdb_files:
        print(f"No PDB files found in {input_folder}")
        return {"success": 0, "failed": 0, "total": 0}
    
    print(f"Found {len(pdb_files)} PDB files in {input_folder}")
    print(f"Using {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"Running {num_processes} parallel processes")
    if remove_heterogens:
        print("Removing heterogens (glycans, ligands, water) before relaxation")
    print(f"Relaxing structures with OpenMM (GPU-accelerated)...")
    
    # Assign GPU to each task in round-robin fashion
    tasks = []
    for i, pdb_file in enumerate(pdb_files):
        gpu_id = gpu_ids[i % len(gpu_ids)]
        tasks.append((pdb_file, output_folder, gpu_id, stiffness, max_iterations, remove_heterogens))
    
    # Process in parallel with progress bar
    with mp.Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.starmap(relax_single_structure_gpu, tasks),
            total=len(tasks),
            desc="Relaxing structures"
        ))
    
    # Print errors and energy summaries
    successful_results = []
    for result in results:
        if not result[2]:  # If failed
            print(f"  ❌ {result[3]}")
        else:
            successful_results.append(result)
    
    # Energy statistics
    if successful_results:
        energies_init = [r[4]['einit'] for r in successful_results]
        energies_final = [r[4]['efinal'] for r in successful_results]
        deltas = [r[4]['delta'] for r in successful_results]
        
        print(f"\nEnergy Statistics (kcal/mol):")
        print(f"  Initial Energy: {sum(energies_init)/len(energies_init):.1f} ± {max(energies_init)-min(energies_init):.1f}")
        print(f"  Final Energy:   {sum(energies_final)/len(energies_final):.1f} ± {max(energies_final)-min(energies_final):.1f}")
        print(f"  Average ΔE:     {sum(deltas)/len(deltas):.1f} kcal/mol")
    
    # Summary
    success_count = sum(1 for r in results if r[2])
    failed_count = len(results) - success_count
    
    print(f"\n{'='*60}")
    print(f"Relaxation Complete:")
    print(f"  ✅ Successfully relaxed: {success_count}/{len(results)}")
    print(f"  ❌ Failed: {failed_count}/{len(results)}")
    print(f"  📁 Output directory: {output_folder}")
    print(f"{'='*60}")
    
    return {
        "success": success_count,
        "failed": failed_count,
        "total": len(results),
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(
        description='GPU-accelerated batch relax PDB structures using OpenMM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Relax all PDBs using GPUs 0,1,2,3,4 with 20 parallel processes
  python relax_structures_gpu.py designs/ designs_relaxed/ --gpus 0,1,2,3,4 --parallel 20
  
  # Use single GPU with 4 processes
  python relax_structures_gpu.py designs/ designs_relaxed/ --gpus 0 --parallel 4
  
  # Adjust restraint stiffness (default 10 kcal/mol/Å²)
  python relax_structures_gpu.py designs/ designs_relaxed/ --stiffness 5.0
        """
    )
    
    parser.add_argument(
        'input_folder',
        type=str,
        help='Input folder containing PDB files'
    )
    
    parser.add_argument(
        'output_folder',
        type=str,
        help='Output folder for relaxed structures'
    )
    
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='Comma-separated GPU IDs to use (default: 0). Example: 0,1,2,3,4'
    )
    
    parser.add_argument(
        '--parallel',
        type=int,
        default=10,
        metavar='N',
        help='Number of parallel processes (default: 10)'
    )
    
    parser.add_argument(
        '--stiffness',
        type=float,
        default=10.0,
        help='Position restraint force constant in kcal/mol/Å² (default: 10.0)'
    )
    
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=0,
        help='Maximum minimization iterations (default: 0 = until convergence)'
    )
    
    parser.add_argument(
        '--keep-heterogens',
        action='store_true',
        help='Keep heterogens (glycans, ligands) - may cause errors if no force field params available'
    )
    
    args = parser.parse_args()
    
    # Validate input folder
    input_folder = Path(args.input_folder)
    if not input_folder.exists():
        print(f"Error: Input folder '{input_folder}' does not exist")
        sys.exit(1)
    
    if not input_folder.is_dir():
        print(f"Error: '{input_folder}' is not a directory")
        sys.exit(1)
    
    # Parse GPU IDs
    gpu_ids = [int(x.strip()) for x in args.gpus.split(',')]
    
    # Run relaxation
    summary = relax_folder_gpu(
        input_folder,
        args.output_folder,
        gpu_ids=gpu_ids,
        num_processes=args.parallel,
        stiffness=args.stiffness,
        max_iterations=args.max_iterations,
        remove_heterogens=not args.keep_heterogens
    )
    
    # Exit with error code if any failed
    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
