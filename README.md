# RiMES antibody CDR co-design

RiMES jointly predicts antibody CDR sequences and atomic coordinates given an
antibody framework, antigen context and fixed CDR lengths. Our design combines
iterative equivariant refinement with residue-frame Riemannian flow supervision,
rotamer-aware sequence–geometry regularization and steric penalties.

This repository retains the `dyMEANModel` class and `--model_type dyMEAN` option
from the underlying dyMEAN implementation. Those names also identify the extended
RiMES model in the training and generation entry points.

## Model overview

- A multi-channel equivariant graph network refines residue features and atomic
  coordinates over three rounds by default.
- A flow head learns rotational and translational velocities on residue frames.
  Intermediate translation feedback is enabled with `--use_flow_iter`.
- Rotamer-aware chi1 compatibility couples sequence probabilities to sidechain
  geometry; C-alpha clash penalties discourage steric overlap.
- Optional precomputed PLM features support parent-conditioned redesign, using
  either a linear projection or an equivariant adapter.

The evaluated refinement workflow uses flow supervision and intermediate
translation corrections. Optional post-EGNN ODE integration is controlled
separately by `--flow_refine_steps` and defaults to `0`.

## Setup

Run commands from the repository root. The existing runtime uses Python 3.10.12,
PyTorch `2.0.0+cu117`, torch-scatter `2.1.2+pt20cu117`, NumPy `1.25.2`, SciPy
`1.11.1` and Biopython `1.81`.

A Linux environment using the same core versions can be prepared with:

```bash
conda create -n rimes python=3.10 -y
conda activate rimes
python -m pip install numpy==1.25.2 scipy==1.11.1 biopython==1.81 tqdm requests tensorboard
python -m pip install torch==2.0.0+cu117 --index-url https://download.pytorch.org/whl/cu117
python -m pip install torch-scatter==2.1.2 --no-index -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

Match torch-scatter to the installed PyTorch/CUDA build. See the official
[PyTorch version instructions](https://pytorch.org/get-started/previous-versions/)
and [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/2.4.0/install/installation.html)
for other builds. This workspace also has an existing `flow` conda environment.

Check the main entry points:

```bash
python train.py --help
python generate.py --help
python cal_metrics_all.py --help
```

CUDA is supported for training and generation. Generation also supports CPU
execution with `--gpu -1`; training uses `--gpus -1` for CPU.

## Input data

Training, validation and test splits are newline-delimited JSON files, even when
their filenames end in `.json`. Each record identifies an antibody–antigen
complex and its chains:

```json
{"pdb":"example","pdb_data_path":"/path/to/complex.pdb","heavy_chain":"H","light_chain":"L","antigen_chains":["A"]}
```

The main model dataset loader expects IMGT-numbered PDB residues. It derives CDR
positions from the structure, constructs coordinate/sequence masks and creates
`*_processed` caches beside the input split. Keep `data/template.json`, which is
used for conserved-framework coordinate initialization.

Our current loader selects epitope context from the reference complex. The
workflow therefore assumes an input complex and fixed loop lengths; it does not
infer a complete binding pose from an antigen sequence alone. Distinguish this
conditioning from methods that retain native framework coordinates in their
generated outputs.

For preprocessing SAbDab structures and creating benchmark splits, see
[scripts/data_preprocess.sh](scripts/data_preprocess.sh),
[data/download.py](data/download.py) and [data/split.py](data/split.py).
The full preprocessing script also prepares additional benchmarks and requires
the relevant structures, summary files and clustering tools. Rebuild or relocate
dataset caches when moving preprocessed data between machines.

Set the location of an existing RAbD split directory:

```bash
export RIMES_DATA=/path/to/processed/RAbD
```

It should contain `train.json`, `valid.json` and `test.json`, with readable PDB
paths. Locally prepared study splits are also available under `rebuttal/data/`.

## Train a PLM-free model

This example trains CDR-H3 co-design on one GPU, with no PLM inputs:

```bash
python train.py \
  --train_set "$RIMES_DATA/train.json" \
  --valid_set "$RIMES_DATA/valid.json" \
  --save_dir checkpoints/rimes_plmfree_seed1 \
  --model_type dyMEAN \
  --cdr H3 --paratope H3 \
  --gpus 0 --seed 1 \
  --batch_size 32 --max_epoch 200 --num_workers 4 --shuffle \
  --lr 0.001 --final_lr 0.0001 --validation_objective base \
  --embed_dim 64 --hidden_size 128 --k_neighbors 9 --n_layers 3 --iter_round 3 \
  --plm_dim 0 \
  --use_riemannian_flow --equivariant_velocity_head \
  --flow_weight 0.1 --flow_consistency_weight 0.05 \
  --flow_rot_weight 1.0 --flow_trans_weight 1.0 \
  --use_flow_iter --flow_iter_alpha 0.05 \
  --rotamer_weight 0.1 --clash_weight 0.5 --clash_interface_weight 2.0
```

`--validation_objective base` selects checkpoints using shared base losses,
excluding auxiliary losses. `total` uses the combined validation objective.
Record this choice when comparing ablations.

For joint design of all six CDRs, replace `--cdr H3` with:

```text
--cdr H1 H2 H3 L1 L2 L3
```

`--paratope` separately controls the residues used for interface processing.
Use a setting consistent with the training task. CDR lengths follow the input
complex; this entry point does not sample new loop lengths.

Training writes:

```text
checkpoints/rimes_plmfree_seed1/version_*/
  namespace.json
  epoch_timing.jsonl
  checkpoint/
    epoch*_step*.ckpt
    topk_map.txt
```

Select a checkpoint from `topk_map.txt` using validation performance. Checkpoints
store serialized model objects, so generation requires the matching model code
and dependencies.

Existing experiment configurations are in
[scripts/train/configs/](scripts/train/configs/) and
[rebuttal/configs/](rebuttal/configs/). Update their dataset, embedding and output
paths before reuse. Prefer direct CLI commands for controls requiring explicit
zeros: the legacy `scripts/train/train.sh` launcher omits zero-valued JSON
settings when converting them to arguments.

## Optional PLM conditioning

PLM-free and adapter-disabled are different settings:

| Setting | PLM features | Equivariant adapter |
|---|---|---|
| PLM-free | Disabled; effective `plm_dim=0` | Disabled |
| PLM with linear projection | Enabled | Disabled |
| PLM with equivariant adaptation | Enabled | Enabled |

In our implementation, disabling the adapter leaves a linear PLM projection
when PLM features are enabled. A `noadapter` filename alone does not identify a
PLM-free run; inspect `namespace.json` and the embedding inputs.

For ESM-2 features, install the optional backend:

```bash
python -m pip install fair-esm==2.0.0
```

Precompute embeddings with the same chain and epitope conventions as the dataset:

```bash
for RIMES_SPLIT in train valid test; do
  python scripts/precompute_esm_embeddings.py \
    --dataset_path "$RIMES_DATA/$RIMES_SPLIT.json" \
    --output_path "$RIMES_DATA/${RIMES_SPLIT}_esm2_3B.pt" \
    --esm_model esm2_t36_3B --device cuda:0
done
```

ESM-2 3B features have dimension 2560. To train with them, replace `--plm_dim 0`
in the training command with the following options:

```text
--use_plm --plm_dim 2560 --use_equivariant_adapter
--train_plm_embeddings /path/to/train_esm2_3B.pt
--valid_plm_embeddings /path/to/valid_esm2_3B.pt
```

Omit `--use_equivariant_adapter` to use linear projection. Generation must receive
the corresponding test embeddings through `--test_plm_embeddings`.

Parent-conditioned hit-to-lead redesign deliberately uses parent-sequence
context. For a control that removes native CDR identity before PLM encoding,
add `--mask_cdr H3` during extraction; use `--mask_cdr H1,H2,H3,L1,L2,L3` for all
six loops. Prepare consistent masked train/validation/test features and retrain
the control. Masking only the test features is a different experiment.

## Generate designs

Choose a trained checkpoint and generate 40 samples per test target:

```bash
export RIMES_CKPT=/path/to/selected_checkpoint.ckpt
export RIMES_OUTPUT=outputs/rimes_h3

python generate.py \
  --ckpt "$RIMES_CKPT" \
  --test_set "$RIMES_DATA/test.json" \
  --save_dir "$RIMES_OUTPUT" \
  --gpu 0 --batch_size 8 --num_workers 4 \
  --n_samples 40 --temperature 0.0 --seed 1 \
  --flow_refine_steps 0
```

The checkpoint determines which CDRs are redesigned. For a PLM-conditioned
checkpoint, append `--test_plm_embeddings /path/to/test_esm2_3B.pt`. Use `--gpu -1`
for CPU generation. Temperature `0` uses argmax sequence decoding; positive
temperatures enable sampling. Repeated samples are not guaranteed to be unique.

| Option | Effect |
|---|---|
| `--flow_refine_steps 0` | Disables optional post-EGNN ODE integration; intermediate feedback can still be active. |
| `--disable_flow_iter` | Disables intermediate flow feedback for inference with the same checkpoint. |
| `--correction_alpha VALUE` | Overrides feedback strength; choose it using validation data. |
| `--use_ideal_framework` | Copies input framework coordinates into the final output after generation. Report this separately when comparing structural scores. |

Despite its name, `--use_ideal_framework` copies the input framework coordinates
in the current implementation. It is off by default.

Outputs include generated complex PDBs, reference PDBs and an evaluation manifest:

```text
outputs/rimes_h3/
  target_s0.pdb
  target_s1.pdb
  ...
  target_original.pdb
  summary.json
```

`summary.json` contains one JSON object per line, including target/sample IDs,
generated/reference paths, chain IDs, CDR selection and prediction score. For a
single sample, generated files use `target.pdb` without a sample suffix.

## Evaluate designs

Install the optional DockQ metric dependency:

```bash
python -m pip install DockQ==2.1.3
```

Run the coordinate evaluator on the generation manifest:

```bash
python cal_metrics_all.py \
  --test_set "$RIMES_OUTPUT/summary.json" \
  --save_dir "$RIMES_OUTPUT/metrics" \
  --skip_plm --num_workers 4 --numbering imgt
```

The script writes per-sample JSONL/CSV records, per-target CSV records,
`summary_metrics.json` and joint-success summaries. `--skip_plm` skips PLM-based
evaluation scores; it does not change the generator's input setting.

Important metric definitions:

| Metric | Definition in this evaluator |
|---|---|
| AAR | Recovery over all redesigned residues within each design; multi-CDR recovery uses residue pooling. |
| CAAR | Recovery on native antigen-contacting CDR residues, with a 6.6-Angstrom contact cutoff. |
| BLOSUM62 | Normalized substitution score on the designed sequence. |
| `RMSD(CA) CDRH3 aligned` | CDR-self-fitted C-alpha RMSD; measures local loop shape. |
| `RMSD(CA) CDRH3` | Unfitted CDR C-alpha RMSD in the supplied coordinate frame. |
| DockQ | The global wrapper can include H-L interfaces as well as antibody-antigen interfaces. |

Loop-self-fitted, framework-fitted and antigen-fitted RMSD answer different
questions. Additional alignment and physical audits are in
[scripts/rebuttal/](scripts/rebuttal/). The Ramachandran score is a broad angular
proxy; C-alpha clashes and all-heavy-atom overlap counts use different scopes.

The evaluator uses the supplied `evaluation/TMscore` executable. Its source can
be rebuilt with `g++ -O3 -o evaluation/TMscore evaluation/TMscore.cpp`.
lDDT requires an external `lddt` executable on PATH; unavailable optional metrics
are recorded as NaN. FoldX and PLM/refolding scores have additional dependencies
and are enabled separately.

### Aggregation and incomplete outputs

The evaluator's default means condition on successful evaluations.
`per_target_aggregation.AAR.mean_of_target_means` averages the valid per-target
means. It does not automatically insert missing targets or apply full-budget
failure penalties.

For a full-test AAR with documented failures assigned zero, retain the expected
targets and sample budget and aggregate explicitly. For 60 targets and 40
attempts per target:

```text
AAR_full = (1/60) * sum_t [(1/40) * sum_k AAR(t,k)]
```

Use zero only for documented failed/unreadable generations under the stated
protocol. Missing archive files alone leave the generation outcome unknown.
Valid-only, full-budget and shared-target analyses should identify their own
denominators and coordinate phase.

## Controlled experiments and reproducibility

The following configurations document the separate PLM-free control study:

| Configuration | Purpose |
|---|---|
| [full_rfm_seed1.json](rebuttal/configs/full_rfm_seed1.json) | Full flow-supervised model. |
| [matched_no_flow_seed1.json](rebuttal/configs/matched_no_flow_seed1.json) | Same head architecture, zero flow/consistency weights and disabled feedback. |
| [no_rotamer_seed1.json](rebuttal/configs/no_rotamer_seed1.json) | Removes rotamer loss. |
| [translation_flow_seed1.json](rebuttal/configs/translation_flow_seed1.json) | Removes rotational-velocity loss. |
| [residual_seed1.json](rebuttal/configs/residual_seed1.json) | Direct displacement-regression control. |

Full and strict no-flow have completed Seeds 1 and 2 in this study; other
component controls use Seed 1. See the
[two-seed report](rebuttal/seed2_pair/RESULTS.md) for the protocol and uncertainty.
These runs use different input/selection settings from saved ESM-enabled runs;
their checkpoint identities and results should remain separate.

For reproducibility, record the dataset manifest, CDR/paratope masks, PLM inputs
and masking, training and generation seeds, checkpoint selection rule, correction
strength, sample count, flow integration setting, framework-copying setting,
coordinate phase and aggregation denominator.

Independent sequence-only refolding is implemented in
[scripts/rebuttal/refold.py](scripts/rebuttal/refold.py). Use its `--help` output
for manifest and backend options. Refolding consistency measures computational
sequence/structure agreement; experimental binding is evaluated separately.

## Main source files

| Path | Role |
|---|---|
| [train.py](train.py) | Training entry point and model/loss flags. |
| [generate.py](generate.py) | Checkpoint loading, generation and PDB/manifest export. |
| [models/dyMEAN/dyMEAN_model.py](models/dyMEAN/dyMEAN_model.py) | Joint sequence/coordinate refinement and flow coupling. |
| [models/modules/riemannian_flow.py](models/modules/riemannian_flow.py) | Residue-frame flow supervision and integration. |
| [models/modules/rotamer_loss.py](models/modules/rotamer_loss.py) | Sequence-weighted chi1 compatibility. |
| [models/modules/clash_loss.py](models/modules/clash_loss.py) | Steric penalties. |
| [models/modules/equivariant_adapter.py](models/modules/equivariant_adapter.py) | Optional PLM adapter. |
| [data/dataset.py](data/dataset.py) | Complex loading, CDR/interface masks and PLM features. |
| [cal_metrics_all.py](cal_metrics_all.py) | Coordinate and sequence evaluation. |

## License

See [LICENSE](LICENSE) for the repository's MIT license and retained copyright
notice. Third-party tools retain their respective licenses and attribution.
