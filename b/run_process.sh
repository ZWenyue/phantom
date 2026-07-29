#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Phantom processing launcher (run b/run_convert.sh first to prepare data)
#
# Usage:
#   bash b/run_process.sh                          # run all steps
#   bash b/run_process.sh --step bbox              # run one step only
#   bash b/run_process.sh --step hand2d --gpus 4   # use 4 GPUs
#   bash b/run_process.sh --task pour              # different task
#   bash b/run_process.sh --step hand2d,action     # multiple steps
#   bash b/run_process.sh --data-root /path/to/raw # custom input dir
#   bash b/run_process.sh --dry-run                # print commands only
# =============================================================================

# ── defaults ──────────────────────────────────────────────────────────────────
TASK="basic_pick_place"
STEP="all"
NUM_GPUS=4
CPU_WORKERS=16
DATA_ROOT="/mnt/r/DATA/EgoDex/test_phantom"
PROCESSED_ROOT="/mnt/r/DATA/EgoDex/test_phantom_processed"
DRY_RUN=false

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2";                 shift 2 ;;
        --step)          STEP="$2";                 shift 2 ;;
        --gpus)          NUM_GPUS="$2";             shift 2 ;;
        --cpu-workers)   CPU_WORKERS="$2";          shift 2 ;;
        --data-root)     DATA_ROOT="$2";            shift 2 ;;
        --processed-root) PROCESSED_ROOT="$2";      shift 2 ;;
        --dry-run)       DRY_RUN=true;              shift   ;;
        -h|--help)
            sed -n '3,12p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

DEMO_NAME="egodex_${TASK}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/../phantom"
CONFIG_ARGS="--config-path=../b/configs --config-name=egodex"
DATA_ARGS="data_root_dir=${DATA_ROOT} processed_data_root_dir=${PROCESSED_ROOT}"

# ── helpers ───────────────────────────────────────────────────────────────────
run_cmd() {
    echo "▸ $*"
    if ! $DRY_RUN; then
        "$@"
    fi
}

count_episodes() {
    local demo_dir="${DATA_ROOT}/${DEMO_NAME}"
    if [[ -d "$demo_dir" ]]; then
        find "$demo_dir" -mindepth 1 -maxdepth 1 -type d | wc -l
    else
        echo 0
    fi
}

# ── step: CPU-parallel (bbox) ─────────────────────────────────────────────────
step_cpu() {
    local mode="$1"
    echo "═══ ${mode^^} (CPU×${CPU_WORKERS}) ═══"
    cd "$PHANTOM_DIR"
    run_cmd python process_data.py \
        ${CONFIG_ARGS} \
        ${DATA_ARGS} \
        demo_name="${DEMO_NAME}" \
        mode="${mode}" \
        n_processes="${CPU_WORKERS}" \
        skip_existing=true
}

# ── step: GPU-parallel (hand2d, arm_segmentation, hand_inpaint, robot_inpaint)
step_gpu() {
    local mode="$1"
    local n_episodes
    n_episodes=$(count_episodes)

    if [[ "$n_episodes" -eq 0 ]]; then
        echo "Error: no episodes found for ${DEMO_NAME}"
        return 1
    fi

    echo "═══ ${mode^^} (GPU×${NUM_GPUS}, ${n_episodes} episodes) ═══"

    local per_gpu=$(( (n_episodes + NUM_GPUS - 1) / NUM_GPUS ))
    local pids=()

    cd "$PHANTOM_DIR"
    for gpu_id in $(seq 0 $((NUM_GPUS - 1))); do
        local start=$((gpu_id * per_gpu))
        if [[ $start -ge $n_episodes ]]; then
            break
        fi
        local end=$(( (gpu_id + 1) * per_gpu - 1 ))
        if [[ $end -ge $n_episodes ]]; then
            end=$((n_episodes - 1))
        fi

        local log_file="/tmp/phantom_${DEMO_NAME}_${mode}_gpu${gpu_id}.log"

        (
            for demo_idx in $(seq "$start" "$end"); do
                CUDA_VISIBLE_DEVICES=${gpu_id} \
                python process_data.py \
                    ${CONFIG_ARGS} \
                    ${DATA_ARGS} \
                    demo_name="${DEMO_NAME}" \
                    mode="${mode}" \
                    demo_num="${demo_idx}" \
                    skip_existing=true \
                    n_processes=1 \
                    2>&1
            done
        ) > "$log_file" 2>&1 &
        pids+=($!)
        echo "  GPU ${gpu_id}: episodes ${start}-${end} (pid $!) → ${log_file}"
    done

    if ! $DRY_RUN; then
        echo "  Waiting for ${#pids[@]} workers..."
        # Poll progress every 15s until all workers finish
        while true; do
            local all_done=true
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    all_done=false
                    break
                fi
            done
            if $all_done; then break; fi

            # Print per-GPU progress: count completed episodes by "100%|" markers
            local status=""
            for gpu_id in $(seq 0 $((${#pids[@]} - 1))); do
                local lf="/tmp/phantom_${DEMO_NAME}_${mode}_gpu${gpu_id}.log"
                local done_ep=$(grep -c '100%|██████████|' "$lf" 2>/dev/null || echo 0)
                local gpu_start=$((gpu_id * per_gpu))
                local gpu_end=$(( (gpu_id + 1) * per_gpu - 1 ))
                if [[ $gpu_end -ge $n_episodes ]]; then gpu_end=$((n_episodes - 1)); fi
                local gpu_total=$(( gpu_end - gpu_start + 1 ))
                status+="GPU${gpu_id}:${done_ep}/${gpu_total} "
            done
            echo "  [$(date +%H:%M:%S)] ${status}"
            sleep 15
        done

        local failed=0
        for pid in "${pids[@]}"; do
            wait "$pid" || ((failed++)) || true
        done
        if [[ $failed -gt 0 ]]; then
            echo "  WARNING: ${failed} worker(s) had errors — check /tmp/phantom_${DEMO_NAME}_${mode}_gpu*.log"
        else
            echo "  Done: ${mode}"
        fi
    fi
}

# ── step: CPU-serial (action, smoothing — fast, no GPU needed) ────────────────
step_serial() {
    local mode="$1"
    echo "═══ ${mode^^} (serial) ═══"
    cd "$PHANTOM_DIR"
    run_cmd python process_data.py \
        ${CONFIG_ARGS} \
        ${DATA_ARGS} \
        demo_name="${DEMO_NAME}" \
        mode="${mode}" \
        skip_existing=true \
        n_processes=1
}

# ── dispatch ──────────────────────────────────────────────────────────────────
run_step() {
    case "$1" in
        bbox)             step_cpu bbox ;;
        hand2d)           step_gpu hand2d ;;
        arm_segmentation) step_gpu arm_segmentation ;;
        action)           step_serial action ;;
        smoothing)        step_serial smoothing ;;
        hand_inpaint)     step_gpu hand_inpaint ;;
        robot_inpaint)    step_gpu robot_inpaint ;;
        *)                echo "Unknown step: $1"; exit 1 ;;
    esac
}

# ── main ──────────────────────────────────────────────────────────────────────
STEPS_ORDER=(bbox hand2d arm_segmentation action smoothing hand_inpaint robot_inpaint)

echo "Task: ${TASK} | Demo: ${DEMO_NAME} | GPUs: ${NUM_GPUS}"
echo "Data: ${DATA_ROOT} | Processed: ${PROCESSED_ROOT}"
echo ""

if [[ "$STEP" == "all" ]]; then
    for s in "${STEPS_ORDER[@]}"; do
        run_step "$s"
        echo ""
    done
else
    IFS=',' read -ra SELECTED <<< "$STEP"
    for s in "${SELECTED[@]}"; do
        run_step "$s"
        echo ""
    done
fi

echo "═══ ALL DONE ═══"
