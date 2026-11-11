#!/bin/bash
########## Configuration ##########
GPU="${GPU:-0,1}"
ADDR="${ADDR:-localhost}"
PORT="${PORT:-9901}"
N_SAMPLES="${N_SAMPLES:-40}"
NUM_WORKERS="${NUM_WORKERS:-80}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
# Temperature ablation: comma-separated list, e.g. "0.1,0.3,0.5,1.0"
TEMPERATURE="${TEMPERATURE:-0.0}"
MODE="${1:-${MODE:-111}}" # 3 digits: train/gen/eval, e.g. 111, 011, 001
# Optional epoch ablation list (comma separated), e.g. "20,40,60,80"
EPOCH_ABLATION="${EPOCH_ABLATION:-}"
# Parallel ablation options
PARALLEL_ABLATION="${PARALLEL_ABLATION:-1}"  # 1 to run multiple checkpoints concurrently
ABLAT_GPUS="${ABLAT_GPUS:-$GPU}"             # GPU pool for parallel jobs, e.g. "0,1,2,3"
MAX_PARALLEL_JOBS="${MAX_PARALLEL_JOBS:-4}"  # max concurrent jobs

_SABDAB_TIME=/home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean/SAbDab_time_split

# Experiment name -- controls where checkpoints & results are saved.
# Override via:  EXP_NAME=my_ablation GPU=0,1 bash run_cdrH3_time.sh
EXP_NAME="${EXP_NAME:-esm2_3B_eqadapter_rflow_adaptive_time_sabdab}"

TRAIN_CONFIG="${TRAIN_CONFIG:-scripts/train/configs/single_cdr_design_esm2_3B_eqadapter_rflow_adaptive_time_sabdab.json}"
TEST_SET="${TEST_SET:-${_SABDAB_TIME}/test.json}"
# Auto-detect test PLM embeddings path from the train config's embedding path pattern
_PLM_PATTERN=$(python -c "
import json, sys, os
try:
    cfg = json.load(open('$TRAIN_CONFIG'))
    p = cfg.get('train_plm_embeddings', '')
    if p:
        print(p.replace('/train_', '/test_'))
except: pass
" 2>/dev/null)
TEST_PLM_EMBEDDINGS="${TEST_PLM_EMBEDDINGS:-${_PLM_PATTERN}}"
CKPT_ROOT="${CKPT_ROOT:-./checkpoints/${EXP_NAME}}"
CKPT_DIR="$CKPT_ROOT"
########## End of configuration ##########

# Script location
CODE_DIR=$(realpath "$(dirname "$0")")
cd "$CODE_DIR"

# Error handling
set -e
trap 'echo "An error occurred. Exiting..."' ERR

# Build --test_plm_embeddings flag if the file exists
GEN_PLM_FLAG=""
if [ -f "$TEST_PLM_EMBEDDINGS" ]; then
    GEN_PLM_FLAG="--test_plm_embeddings $TEST_PLM_EMBEDDINGS"
fi

echo "============================================"
echo "  CDR-H3 Design (PRiME + PLM) [SAbDab Time Split]"
echo "============================================"
echo "Experiment:  ${EXP_NAME}"
echo "GPUs:        ${GPU}"
echo "Config:      ${TRAIN_CONFIG}"
echo "Checkpoints: ${CKPT_ROOT}"
echo "Test set:    ${TEST_SET}"
echo "Test PLM:    ${TEST_PLM_EMBEDDINGS}"
echo "Mode (train/gen/eval): ${MODE}"
echo "Temperature:  ${TEMPERATURE}"
echo "Epoch ablation list: ${EPOCH_ABLATION:-<best-only>}"
echo "Parallel ablation: ${PARALLEL_ABLATION} (gpu_pool=${ABLAT_GPUS}, max_jobs=${MAX_PARALLEL_JOBS})"

if [[ ! "$MODE" =~ ^[01]{3}$ ]]; then
    echo "Error: MODE must be a 3-digit binary string, e.g. 111, 011, 001."
    exit 1
fi

DO_TRAIN=${MODE:0:1}
DO_GEN=${MODE:1:1}
DO_EVAL=${MODE:2:1}

# Parse config
if [ ! -f "$TRAIN_CONFIG" ]; then
    echo "Error: Config file $TRAIN_CONFIG not found!"
    exit 1
fi

# Override save_dir in config with our EXP_NAME-based path
CONFIG=$(cat "$TRAIN_CONFIG" | python -c "
import sys, json
config = json.load(sys.stdin)
config['save_dir'] = '$CKPT_ROOT'
print(' '.join([
    f'--{key} {config[key]}' if not isinstance(config[key], bool)
    else (f'--{key}' if config[key] else '')
    for key in config
]))
")

export CUDA_VISIBLE_DEVICES=$GPU
GPU_ARR=($(echo $GPU | tr ',' ' '))

if [ ${#GPU_ARR[@]} -gt 1 ]; then
    export OMP_NUM_THREADS=2
    PREFIX="torchrun --nproc_per_node=${#GPU_ARR[@]} --rdzv_backend=c10d --rdzv_endpoint=${ADDR}:${PORT} --nnodes=1"
else
    PREFIX="python"
fi

if [ "$DO_TRAIN" = "1" ]; then
    ${PREFIX} train.py --gpus "${!GPU_ARR[@]}" ${CONFIG}
    echo ""
    echo "Training complete. Checkpoints at: ${CKPT_ROOT}"
else
    echo "Skipping training (MODE=${MODE})."
fi

if [ "$DO_GEN" = "1" ] || [ "$DO_EVAL" = "1" ]; then
    # Find best checkpoint from latest version directory
    LATEST_TOPK_MAP=$(find "$CKPT_ROOT" -name "topk_map.txt" | sort -V | tail -1)
    if [ -n "$LATEST_TOPK_MAP" ]; then
        CKPT_DIR=$(dirname "$LATEST_TOPK_MAP")
    fi

    TOPK_MAP="$CKPT_DIR/topk_map.txt"
    if [ ! -f "$TOPK_MAP" ]; then
        echo "ERROR: No topk_map.txt found in latest CKPT_DIR=${CKPT_DIR}."
        exit 1
    fi

    CKPT_BEST=$(head -1 "$TOPK_MAP" | sed 's/^[^:]*: //')
    if [ ! -f "$CKPT_BEST" ]; then
        echo "ERROR: Best checkpoint not found at ${CKPT_BEST}"
        exit 1
    fi

    CKPT_LIST=()
    CKPT_TAGS=()
    if [ -n "$EPOCH_ABLATION" ]; then
        IFS=',' read -ra EPOCHS <<< "$EPOCH_ABLATION"
        for ep in "${EPOCHS[@]}"; do
            ep="$(echo "$ep" | xargs)"
            [ -z "$ep" ] && continue
            CANDIDATE=$(ls "$CKPT_DIR"/epoch"${ep}"_step*.ckpt 2>/dev/null | sort -V | tail -1 || true)
            if [ -n "$CANDIDATE" ] && [ -f "$CANDIDATE" ]; then
                CKPT_LIST+=("$CANDIDATE")
                CKPT_TAGS+=("epoch${ep}")
            else
                echo "Warning: no checkpoint matched epoch ${ep} in ${CKPT_DIR}"
            fi
        done
        if [ ${#CKPT_LIST[@]} -eq 0 ]; then
            echo "ERROR: no checkpoints found for EPOCH_ABLATION=${EPOCH_ABLATION}"
            exit 1
        fi
    else
        CKPT_LIST+=("$CKPT_BEST")
        CKPT_TAGS+=("best")
    fi

    echo ""
    echo "============================================"
    echo "  Evaluation  [${EXP_NAME}]"
    echo "============================================"
    echo "Checkpoint dir: ${CKPT_DIR}"
    echo "Test set: ${TEST_SET}"
    echo "Samples per target: ${N_SAMPLES}"
    echo "Generation batch size: ${EVAL_BATCH_SIZE}"
    echo "Planned checkpoints:"
    for i in "${!CKPT_LIST[@]}"; do
        echo "  - ${CKPT_TAGS[$i]} -> ${CKPT_LIST[$i]}"
    done
fi

run_one_ckpt() {
    local ckpt="$1"
    local tag="$2"
    local gpu_slot="$3"
    local temp="$4"

    local save_dir
    if [ "$temp" = "0.0" ] || [ "$temp" = "0" ]; then
        save_dir="$(dirname "$ckpt")/results_${tag}"
    else
        save_dir="$(dirname "$ckpt")/results_${tag}_temp${temp}"
    fi
    mkdir -p "$save_dir"

    echo ""
    echo "--------------------------------------------"
    echo "Checkpoint tag: ${tag}  temp: ${temp}"
    echo "Checkpoint:     ${ckpt}"
    echo "Save dir:       ${save_dir}"
    echo "GPU slot:       ${gpu_slot}"
    echo "--------------------------------------------"

    export CUDA_VISIBLE_DEVICES="$gpu_slot"

    if [ "$DO_GEN" = "1" ]; then
        if [ "$EVAL_BATCH_SIZE" -gt 1 ]; then
            echo "Warning: EVAL_BATCH_SIZE>1 may trigger CUDA index errors for this checkpoint/model. Consider EVAL_BATCH_SIZE=1."
        fi
        python generate.py \
            --ckpt "$ckpt" \
            --test_set "$TEST_SET" \
            --save_dir "$save_dir" \
            --n_samples $N_SAMPLES \
            --batch_size $EVAL_BATCH_SIZE \
            --gpu 0 \
            --temperature $temp \
            $GEN_PLM_FLAG
        echo "Done generation for ${tag} temp=${temp}."
    else
        echo "Skipping generation for ${tag} (MODE=${MODE})."
    fi

    if [ "$DO_EVAL" = "1" ]; then
        OPENMM_CPU_THREADS=1 python cal_metrics.py \
            --test_set "$save_dir/summary.json" \
            --num_workers $NUM_WORKERS \
            --save_dir "$save_dir/metrics"
        echo "Done metrics for ${tag} temp=${temp}."
    else
        echo "Skipping metrics for ${tag} (MODE=${MODE})."
    fi
}

if [ "$DO_GEN" = "1" ] || [ "$DO_EVAL" = "1" ]; then
    IFS=',' read -ra _TEMP_CHECK <<< "$TEMPERATURE"
    if [ "$PARALLEL_ABLATION" = "1" ] && { [ "${#CKPT_LIST[@]}" -gt 1 ] || [ "${#_TEMP_CHECK[@]}" -gt 1 ]; }; then
        IFS=',' read -ra ABLAT_GPU_ARR <<< "$ABLAT_GPUS"
        if [ ${#ABLAT_GPU_ARR[@]} -eq 0 ]; then
            echo "ERROR: ABLAT_GPUS is empty."
            exit 1
        fi
        if [ "$MAX_PARALLEL_JOBS" -lt 1 ]; then
            MAX_PARALLEL_JOBS=1
        fi

        IFS=',' read -ra TEMP_ARR <<< "$TEMPERATURE"
        PIDS=()
        TAGS=()
        LOGS=()
        job_idx=0
        for i in "${!CKPT_LIST[@]}"; do
            ckpt="${CKPT_LIST[$i]}"
            tag="${CKPT_TAGS[$i]}"
            for temp in "${TEMP_ARR[@]}"; do
                gpu_slot="${ABLAT_GPU_ARR[$(( job_idx % ${#ABLAT_GPU_ARR[@]} ))]}"
                if [ "$temp" = "0.0" ] || [ "$temp" = "0" ]; then
                    save_dir="$(dirname "$ckpt")/results_${tag}"
                    job_label="${tag}"
                else
                    save_dir="$(dirname "$ckpt")/results_${tag}_temp${temp}"
                    job_label="${tag}_temp${temp}"
                fi
                log_file="${save_dir}/ablation_run.log"
                mkdir -p "$save_dir"

                while [ "$(jobs -pr | wc -l)" -ge "$MAX_PARALLEL_JOBS" ]; do
                    sleep 2
                done

                nohup bash -c "
                    set -e
                    cd \"$CODE_DIR\"
                    export CUDA_VISIBLE_DEVICES=\"$gpu_slot\"
                    echo \"[${job_label}] start on GPU ${gpu_slot}\"
                    echo \"[${job_label}] ckpt=${ckpt}\"
                    echo \"[${job_label}] save_dir=${save_dir}\"

                    if [ \"$DO_GEN\" = \"1\" ]; then
                        python generate.py \\
                            --ckpt \"$ckpt\" \\
                            --test_set \"$TEST_SET\" \\
                            --save_dir \"$save_dir\" \\
                            --n_samples $N_SAMPLES \\
                            --batch_size $EVAL_BATCH_SIZE \\
                            --gpu 0 \\
                            --temperature $temp \\
                            $GEN_PLM_FLAG
                        echo \"[${job_label}] generation done\"
                    fi

                    if [ \"$DO_EVAL\" = \"1\" ]; then
                        OPENMM_CPU_THREADS=1 python cal_metrics.py \\
                            --test_set \"$save_dir/summary.json\" \\
                            --num_workers $NUM_WORKERS \\
                            --save_dir \"$save_dir/metrics\"
                        echo \"[${job_label}] metrics done\"
                    fi

                    echo \"[${job_label}] all done\"
                " < /dev/null > "$log_file" 2>&1 &

                pid=$!
                PIDS+=("$pid")
                TAGS+=("$job_label")
                LOGS+=("$log_file")
                echo "Launched ${job_label} on GPU ${gpu_slot} (pid=${pid}, log=${log_file})"
                job_idx=$((job_idx + 1))
            done
        done

        echo ""
        echo "All ${#PIDS[@]} jobs launched. Waiting for completion..."
        echo "Monitor logs with:  tail -f ${CKPT_DIR}/results_epoch*/ablation_run.log"
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
            echo "ERROR: one or more parallel ablation jobs failed."
            exit 1
        fi
    else
        eval_gpu=$(echo $GPU | cut -d',' -f1)
        IFS=',' read -ra TEMP_ARR <<< "$TEMPERATURE"
        for i in "${!CKPT_LIST[@]}"; do
            for temp in "${TEMP_ARR[@]}"; do
                run_one_ckpt "${CKPT_LIST[$i]}" "${CKPT_TAGS[$i]}" "$eval_gpu" "$temp"
            done
        done
    fi
else
    echo "Skipping generation/metrics (MODE=${MODE})."
fi

echo ""
echo "============================================"
if [ "$DO_GEN" = "1" ] || [ "$DO_EVAL" = "1" ]; then
    if [ "$DO_EVAL" = "1" ]; then
        echo "  [${EXP_NAME}] Results saved under:"
        echo "  ${CKPT_DIR}/results_*"
    else
        echo "  [${EXP_NAME}] Generation saved under:"
        echo "  ${CKPT_DIR}/results_*"
    fi
else
    echo "  [${EXP_NAME}] All done."
fi
echo "============================================"
