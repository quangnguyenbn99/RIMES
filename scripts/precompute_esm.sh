#!/bin/bash
# Precompute ESM embeddings for train / valid / test splits.
#
# Supported models (set ESM_MODEL env var):
#   ESM-2 (needs PRiME env):  esm2_t30_150M, esm2_t33_650M, esm2_t36_3B
#   ESM C (needs esmc env):    esmc_300m, esmc_600m
#
# Usage:
#   GPU=0 ESM_MODEL=esmc_300m bash scripts/precompute_esm.sh
#   GPU=0 ESM_MODEL=esmc_600m bash scripts/precompute_esm.sh

set -e

GPU="${GPU:-0}"
ESM_MODEL="${ESM_MODEL:-esmc_300m}"
DATA_DIR="/home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/RAbD"
CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$CODE_DIR"

# Pick the right conda env based on model family
if [[ "$ESM_MODEL" == esmc_* ]]; then
    CONDA_ENV="esmc"
else
    CONDA_ENV="PRiME"
fi

echo "============================================"
echo "  Precompute ESM embeddings"
echo "============================================"
echo "ESM model:  ${ESM_MODEL}"
echo "Conda env:  ${CONDA_ENV}"
echo "Data dir:   ${DATA_DIR}"
echo "GPU:        ${GPU}"
echo ""

for SPLIT in train valid test; do
    INPUT="${DATA_DIR}/${SPLIT}.json"
    OUTPUT="${DATA_DIR}/${SPLIT}_${ESM_MODEL}_embeddings.pt"

    if [ ! -f "$INPUT" ]; then
        echo "Skipping ${SPLIT}: ${INPUT} not found"
        continue
    fi

    if [ -f "$OUTPUT" ]; then
        echo "[${SPLIT}] Already exists: ${OUTPUT}  (delete to regenerate)"
        continue
    fi

    echo "[${SPLIT}] ${INPUT} -> ${OUTPUT}"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n "$CONDA_ENV" \
        python -m scripts.precompute_esm_embeddings \
            --dataset_path "$INPUT" \
            --output_path  "$OUTPUT" \
            --esm_model    "$ESM_MODEL" \
            --device       cuda:0
    echo ""
done

echo "Done. Embedding files:"
ls -lh "${DATA_DIR}"/*${ESM_MODEL}*.pt 2>/dev/null || echo "(none found)"
