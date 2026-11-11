#!/bin/bash
# Precompute ESM2-3B embeddings for SAbDab_time_split (train / valid / test)
# Runs all three splits in parallel on separate GPUs.
#
# Usage:
#   bash scripts/precompute_esm_embeddings_sabdab.sh
#
# Override GPUs if needed:
#   TRAIN_GPU=0 VALID_GPU=1 TEST_GPU=2 bash scripts/precompute_esm_embeddings_sabdab.sh

set -e

TRAIN_GPU="${TRAIN_GPU:-0}"
VALID_GPU="${VALID_GPU:-1}"
TEST_GPU="${TEST_GPU:-2}"

DATA="/home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/SAbDab_time_split"
ESM_MODEL="esm2_t36_3B"

CODE_DIR=$(realpath "$(dirname "$0")/..")
cd "$CODE_DIR"

echo "============================================"
echo "  Precompute ESM2-3B embeddings (SAbDab)"
echo "============================================"
echo "Data dir:  $DATA"
echo "Model:     $ESM_MODEL"
echo "GPUs:      train=$TRAIN_GPU  valid=$VALID_GPU  test=$TEST_GPU"
echo ""

mkdir -p "$DATA"

run_split() {
    local split="$1"
    local gpu="$2"
    local log="$DATA/${split}_esm2_t36_3B_embed.log"

    echo "[$split] Launching on GPU $gpu  (log: $log)"
    CUDA_VISIBLE_DEVICES="$gpu" python scripts/precompute_esm_embeddings.py \
        --dataset_path  "$DATA/${split}.json" \
        --output_path   "$DATA/${split}_esm2_t36_3B_embeddings.pt" \
        --esm_model     "$ESM_MODEL" \
        --device        "cuda:0" \
        > "$log" 2>&1
    echo "[$split] Done."
}

# Launch all three splits in parallel
run_split train "$TRAIN_GPU" &
PID_TRAIN=$!

run_split valid "$VALID_GPU" &
PID_VALID=$!

run_split test  "$TEST_GPU"  &
PID_TEST=$!

FAIL=0
for pid_var in PID_TRAIN PID_VALID PID_TEST; do
    pid="${!pid_var}"
    if wait "$pid"; then
        echo "${pid_var} completed (pid=$pid)"
    else
        echo "ERROR: ${pid_var} failed (pid=$pid)"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "One or more embedding jobs failed. Check logs in $DATA/"
    exit 1
fi

echo ""
echo "All embeddings saved to $DATA/"
ls -lh "$DATA/"*_esm2_t36_3B_embeddings.pt
echo "============================================"
