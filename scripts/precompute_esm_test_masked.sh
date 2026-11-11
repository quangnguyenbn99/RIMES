#!/bin/bash
# Precompute ESM-2-3B test-set embeddings with the native CDR(s) masked,
# for each of the three paper experiments. These are the leakage-corrected
# embeddings used to re-evaluate trained checkpoints without retraining.
#
# Three jobs, one per experiment:
#   (1) single_cdr_design          (RAbD, CDR-H3 design)
#         input : RAbD/test.json
#         mask  : H3
#         output: RAbD/test_esm2_t36_3B_maskH3_embeddings.pt
#
#   (2) multi_cdr_design           (RAbD, all 6 CDRs design)
#         input : RAbD/test.json
#         mask  : H1,H2,H3,L1,L2,L3
#         output: RAbD/test_esm2_t36_3B_maskAllCDR_embeddings.pt
#
#   (3) single_cdr_design_sabdab   (SAbDab time-split, CDR-H3 design)
#         input : SAbDab_time_split/test.json
#         mask  : H3
#         output: SAbDab_time_split/test_esm2_t36_3B_maskH3_embeddings.pt
#
# Usage:
#   bash scripts/precompute_esm_test_masked.sh
#
# Override GPUs (defaults: 0, 1, 2):
#   RABD_H3_GPU=0 RABD_ALL_GPU=1 SABDAB_H3_GPU=2 \
#     bash scripts/precompute_esm_test_masked.sh
#
# Run sequentially on a single GPU instead of in parallel:
#   SEQUENTIAL=1 GPU=0 bash scripts/precompute_esm_test_masked.sh

set -e

DATA_ROOT="/home/ubuntu/quang/antibody_design/dymean_data/processed_data_dymean"
RABD_DIR="${DATA_ROOT}/RAbD"
SABDAB_DIR="${DATA_ROOT}/SAbDab_time_split"
ESM_MODEL="esm2_t36_3B"

CODE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$CODE_DIR"

# ---- Job table -----------------------------------------------------------
# tag | input json | mask spec | output path
JOBS=(
    "rabd_h3|${RABD_DIR}/test.json|H3|${RABD_DIR}/test_${ESM_MODEL}_maskH3_embeddings.pt"
    "rabd_all|${RABD_DIR}/test.json|H1,H2,H3,L1,L2,L3|${RABD_DIR}/test_${ESM_MODEL}_maskAllCDR_embeddings.pt"
    "sabdab_h3|${SABDAB_DIR}/test.json|H3|${SABDAB_DIR}/test_${ESM_MODEL}_maskH3_embeddings.pt"
)

# ---- GPU assignment ------------------------------------------------------
RABD_H3_GPU="${RABD_H3_GPU:-0}"
RABD_ALL_GPU="${RABD_ALL_GPU:-1}"
SABDAB_H3_GPU="${SABDAB_H3_GPU:-2}"
SEQUENTIAL_GPU="${GPU:-0}"

declare -A GPU_OF=(
    ["rabd_h3"]="$RABD_H3_GPU"
    ["rabd_all"]="$RABD_ALL_GPU"
    ["sabdab_h3"]="$SABDAB_H3_GPU"
)

echo "============================================"
echo "  Precompute leakage-corrected test ESM embeddings"
echo "============================================"
echo "ESM model:  ${ESM_MODEL}"
echo "Mode:       $([ -n "${SEQUENTIAL:-}" ] && echo sequential || echo parallel)"
echo ""

# ---- Worker --------------------------------------------------------------
run_one() {
    local tag="$1"
    local input="$2"
    local mask="$3"
    local output="$4"
    local gpu="$5"
    local log="${output%.pt}.log"

    if [ ! -f "$input" ]; then
        echo "[$tag] SKIP: input not found: $input"
        return 0
    fi
    if [ -f "$output" ]; then
        echo "[$tag] SKIP: output already exists: $output  (delete to regenerate)"
        return 0
    fi

    echo "[$tag] GPU=$gpu  mask=$mask"
    echo "         in : $input"
    echo "         out: $output"
    echo "         log: $log"

    CUDA_VISIBLE_DEVICES="$gpu" python -m scripts.precompute_esm_embeddings \
        --dataset_path "$input" \
        --output_path  "$output" \
        --esm_model    "$ESM_MODEL" \
        --mask_cdr     "$mask" \
        --device       cuda:0 \
        > "$log" 2>&1
    echo "[$tag] done."
}

# ---- Dispatch ------------------------------------------------------------
PIDS=()
TAGS=()

for spec in "${JOBS[@]}"; do
    IFS='|' read -r tag input mask output <<< "$spec"

    if [ -n "${SEQUENTIAL:-}" ]; then
        run_one "$tag" "$input" "$mask" "$output" "$SEQUENTIAL_GPU"
    else
        gpu="${GPU_OF[$tag]}"
        run_one "$tag" "$input" "$mask" "$output" "$gpu" &
        PIDS+=("$!")
        TAGS+=("$tag")
    fi
done

# ---- Wait & report -------------------------------------------------------
FAIL=0
if [ -z "${SEQUENTIAL:-}" ]; then
    for i in "${!PIDS[@]}"; do
        pid="${PIDS[$i]}"
        tag="${TAGS[$i]}"
        if wait "$pid"; then
            echo "[$tag] OK (pid=$pid)"
        else
            echo "[$tag] FAILED (pid=$pid)"
            FAIL=1
        fi
    done
fi

echo ""
echo "============================================"
echo "Generated files:"
ls -lh \
    "${RABD_DIR}/test_${ESM_MODEL}_maskH3_embeddings.pt" \
    "${RABD_DIR}/test_${ESM_MODEL}_maskAllCDR_embeddings.pt" \
    "${SABDAB_DIR}/test_${ESM_MODEL}_maskH3_embeddings.pt" \
    2>/dev/null || true

if [ "$FAIL" -ne 0 ]; then
    echo ""
    echo "One or more jobs failed. Check the .log files above for details."
    exit 1
fi
