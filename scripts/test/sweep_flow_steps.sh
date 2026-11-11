#!/bin/bash
# Sweep flow_refine_steps to evaluate how integration steps affect generation quality.
# Runs each step count in parallel on separate GPUs (matching run_cdrH3.sh conventions).
#
# Usage:
#   GPUS=2,3,5,6,7 bash scripts/test/sweep_flow_steps.sh [checkpoint]
#
# Defaults to the best checkpoint from esm2_3B_eqadapter_rflow_adaptive_time.

set -e

CODE_DIR=$(realpath "$(dirname "$0")/../..")
cd "$CODE_DIR"

GPUS="${GPUS:-2,3,5,6,7}"
N_SAMPLES="${N_SAMPLES:-40}"
NUM_WORKERS="${NUM_WORKERS:-80}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"

CKPT_ROOT="${CKPT_ROOT:-./checkpoints/esm2_3B_eqadapter_rflow_adaptive_time}"
TEST_SET="${TEST_SET:-/home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/RAbD/test.json}"
TEST_PLM="${TEST_PLM:-/home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/RAbD/test_esm2_t36_3B_embeddings.pt}"

# Find best checkpoint
if [ -n "$1" ]; then
    CKPT=$(realpath "$1")
else
    TOPK_MAP=$(find "$CKPT_ROOT" -name "topk_map.txt" | sort -V | tail -1)
    if [ -z "$TOPK_MAP" ]; then
        echo "ERROR: No topk_map.txt found in ${CKPT_ROOT}. Pass checkpoint path as argument."
        exit 1
    fi
    CKPT=$(head -1 "$TOPK_MAP" | sed 's/^[^:]*: //')
    if [ ! -f "$CKPT" ]; then
        echo "ERROR: Best checkpoint not found at ${CKPT}"
        exit 1
    fi
fi

CKPT_DIR=$(dirname "$CKPT")
STEPS=(0 5 20 30 50 70)

IFS=',' read -ra GPU_ARR <<< "$GPUS"

echo "============================================"
echo "  Flow refinement steps sweep"
echo "============================================"
echo "Checkpoint:  $CKPT"
echo "Test set:    $TEST_SET"
echo "PLM embed:   $TEST_PLM"
echo "Steps:       ${STEPS[*]}"
echo "GPU pool:    ${GPUS}"
echo "N_SAMPLES:   $N_SAMPLES"
echo "NUM_WORKERS: $NUM_WORKERS"
echo "============================================"

# Build PLM flag
GEN_PLM_FLAG=""
if [ -f "$TEST_PLM" ]; then
    GEN_PLM_FLAG="--test_plm_embeddings $TEST_PLM"
fi

PIDS=()
TAGS=()
LOGS=()

for i in "${!STEPS[@]}"; do
    NSTEPS="${STEPS[$i]}"
    gpu_slot="${GPU_ARR[$(( i % ${#GPU_ARR[@]} ))]}"
    SAVE_DIR="${CKPT_DIR}/results_flow_steps_${NSTEPS}"
    LOG_FILE="${SAVE_DIR}/run.log"
    mkdir -p "$SAVE_DIR"

    nohup bash -c "
        set -e
        cd \"$CODE_DIR\"
        export CUDA_VISIBLE_DEVICES=\"$gpu_slot\"

        echo \"[steps=${NSTEPS}] start on GPU ${gpu_slot}\"
        echo \"[steps=${NSTEPS}] save_dir=${SAVE_DIR}\"

        python generate.py \\
            --ckpt \"$CKPT\" \\
            --test_set \"$TEST_SET\" \\
            --save_dir \"$SAVE_DIR\" \\
            --n_samples $N_SAMPLES \\
            --batch_size $EVAL_BATCH_SIZE \\
            --gpu 0 \\
            --flow_refine_steps $NSTEPS \\
            $GEN_PLM_FLAG

        echo \"[steps=${NSTEPS}] generation done\"

        OPENMM_CPU_THREADS=1 python cal_metrics.py \\
            --test_set \"$SAVE_DIR/summary.json\" \\
            --num_workers $NUM_WORKERS \\
            --save_dir \"$SAVE_DIR/metrics\"

        echo \"[steps=${NSTEPS}] all done\"
    " < /dev/null > "$LOG_FILE" 2>&1 &

    pid=$!
    PIDS+=("$pid")
    TAGS+=("steps=${NSTEPS}")
    LOGS+=("$LOG_FILE")
    echo "Launched steps=${NSTEPS} on GPU ${gpu_slot} (pid=${pid}, log=${LOG_FILE})"
done

echo ""
echo "All ${#PIDS[@]} jobs launched. Waiting for completion..."
echo "Monitor logs with:  tail -f ${CKPT_DIR}/results_flow_steps_*/run.log"
echo ""

FAIL=0
for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
        echo "Completed ${TAGS[$i]} (log: ${LOGS[$i]})"
    else
        echo "FAILED ${TAGS[$i]} (log: ${LOGS[$i]})"
        FAIL=1
    fi
done

if [ "$FAIL" -ne 0 ]; then
    echo "ERROR: one or more jobs failed. Check logs above."
    exit 1
fi

echo ""
echo "============================================"
echo "  All sweeps complete. Results in:"
for NSTEPS in "${STEPS[@]}"; do
    echo "  steps=${NSTEPS}: ${CKPT_DIR}/results_flow_steps_${NSTEPS}/"
done
echo "============================================"
