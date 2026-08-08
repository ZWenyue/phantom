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
#   bash b/run_process.sh --config egodex_r1pro_bimanual  # R1 Pro overlay
#   bash b/run_process.sh --config egodex_panda --step robot_inpaint  # Franka overlay
#   bash b/run_process.sh --demo-num 0_useful --no-skip   # one episode, force rerun
#   bash b/run_process.sh --dry-run                # print commands only
# =============================================================================

# ── defaults ──────────────────────────────────────────────────────────────────
TASK="basic_pick_place"
STEP="all"
NUM_GPUS=8
NUM_WORKERS="32"  # defaults to NUM_GPUS if not set
CPU_WORKERS=32
DATA_ROOT="/home/a26160/DATA/test_phantom"
PROCESSED_ROOT="/home/a26160/DATA/test_phantom_processed"
CONFIG_NAME="egodex"
DEMO_NUM=""       # if set, only process this episode folder (e.g. 0_useful)
SKIP_EXISTING="true"
DRY_RUN=false

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2";                 shift 2 ;;
        --step)          STEP="$2";                 shift 2 ;;
        --gpus)          NUM_GPUS="$2";             shift 2 ;;
        --workers)       NUM_WORKERS="$2";          shift 2 ;;
        --cpu-workers)   CPU_WORKERS="$2";          shift 2 ;;
        --data-root)     DATA_ROOT="$2";            shift 2 ;;
        --processed-root) PROCESSED_ROOT="$2";      shift 2 ;;
        --config)        CONFIG_NAME="$2";          shift 2 ;;
        --demo-num)      DEMO_NUM="$2";             shift   2 ;;
        --no-skip)       SKIP_EXISTING="false";     shift   ;;
        --dry-run)       DRY_RUN=true;              shift   ;;
        -h|--help)
            sed -n '3,14p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

DEMO_NAME="egodex_${TASK}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/../phantom"
CONFIG_ARGS="--config-path=../b/configs --config-name=${CONFIG_NAME}"
DATA_ARGS="data_root_dir=${DATA_ROOT} processed_data_root_dir=${PROCESSED_ROOT}"

# Activate phantom conda env
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom
export PYTHONUNBUFFERED=1

# MuJoCo/robosuite headless EGL rendering (required for robot_inpaint)
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="${HOME}/.local/share/glvnd/egl_vendor.d"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

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
# --workers controls parallelism (defaults to NUM_GPUS); GPUs assigned round-robin
default_workers() {
    case "$1" in
        robot_inpaint)    echo 32 ;;
        arm_segmentation) echo $(($NUM_GPUS * 2)) ;;
        hand_inpaint)     echo $(($NUM_GPUS * 1)) ;;
        *)                echo $(($NUM_GPUS * 8)) ;;
    esac
}

step_gpu() {
    local mode="$1"

    # Single explicit episode (e.g. 0_useful) — skip worker sharding
    if [[ -n "$DEMO_NUM" ]]; then
        echo "═══ ${mode^^} (demo_num=${DEMO_NUM}, config=${CONFIG_NAME}) ═══"
        cd "$PHANTOM_DIR"
        local log_file="/tmp/phantom_${DEMO_NAME}_${mode}_${DEMO_NUM}.log"
        echo "  → ${log_file}"
        if ! $DRY_RUN; then
            CUDA_VISIBLE_DEVICES=0 \
            python process_data.py \
                ${CONFIG_ARGS} \
                ${DATA_ARGS} \
                demo_name="${DEMO_NAME}" \
                mode="${mode}" \
                demo_num="${DEMO_NUM}" \
                skip_existing="${SKIP_EXISTING}" \
                n_processes=1 \
                2>&1 | tee "$log_file"
        else
            echo "▸ python process_data.py ... demo_num=${DEMO_NUM} skip_existing=${SKIP_EXISTING}"
        fi
        return 0
    fi

    local n_episodes
    n_episodes=$(count_episodes)
    local n_workers=${NUM_WORKERS:-$(default_workers "$mode")}

    if [[ "$n_episodes" -eq 0 ]]; then
        echo "Error: no episodes found for ${DEMO_NAME}"
        return 1
    fi

    echo "═══ ${mode^^} (${n_workers} workers × ${NUM_GPUS} GPUs, ${n_episodes} episodes, config=${CONFIG_NAME}) ═══"

    local per_worker=$(( (n_episodes + n_workers - 1) / n_workers ))
    pids=()
    _w_starts=()
    _w_ends=()

    cd "$PHANTOM_DIR"
    for wid in $(seq 0 $((n_workers - 1))); do
        local start=$((wid * per_worker))
        if [[ $start -ge $n_episodes ]]; then
            break
        fi
        local end=$(( (wid + 1) * per_worker - 1 ))
        if [[ $end -ge $n_episodes ]]; then
            end=$((n_episodes - 1))
        fi
        local gpu_id=$((wid % NUM_GPUS))

        local log_file="/tmp/phantom_${DEMO_NAME}_${mode}_w${wid}.log"

        (
            for demo_idx in $(seq "$start" "$end"); do
                CUDA_VISIBLE_DEVICES=${gpu_id} \
                python process_data.py \
                    ${CONFIG_ARGS} \
                    ${DATA_ARGS} \
                    demo_name="${DEMO_NAME}" \
                    mode="${mode}" \
                    demo_num="${demo_idx}" \
                    skip_existing="${SKIP_EXISTING}" \
                    n_processes=1 \
                    2>&1
            done
        ) > "$log_file" 2>&1 &
        pids+=($!)
        _w_starts+=($start)
        _w_ends+=($end)
        echo "  W${wid}(GPU${gpu_id}): episodes ${start}-${end} (pid $!) → ${log_file}"
    done

    if ! $DRY_RUN; then
        local actual_workers=${#pids[@]}
        echo "  Waiting for ${actual_workers} workers..."
        while true; do
            local all_done=true
            for pid in "${pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    all_done=false
                    break
                fi
            done
            if $all_done; then break; fi

            local status="" total_done=0 total_all=0
            for wid in $(seq 0 $((actual_workers - 1))); do
                local lf="/tmp/phantom_${DEMO_NAME}_${mode}_w${wid}.log"
                local done_ep=0
                local started=0
                started=$(grep -c 'PROCESSOR -' "$lf" 2>/dev/null) || true
                done_ep=$(( started > 0 ? started - 1 : 0 ))
                local w_total=$(( _w_ends[wid] - _w_starts[wid] + 1 ))
                # If process exited, all its episodes are done
                if ! kill -0 "${pids[wid]}" 2>/dev/null; then
                    done_ep=$w_total
                fi
                total_done=$((total_done + done_ep))
                total_all=$((total_all + w_total))
            done
            echo "  [$(date +%H:%M:%S)] ${total_done}/${total_all} episodes done"
            sleep 15
        done

        local failed=0
        for pid in "${pids[@]}"; do
            wait "$pid" || ((failed++)) || true
        done
        if [[ $failed -gt 0 ]]; then
            echo "  WARNING: ${failed} worker(s) had errors — check /tmp/phantom_${DEMO_NAME}_${mode}_w*.log"
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
        skip_existing="${SKIP_EXISTING}" \
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

echo "Task: ${TASK} | Demo: ${DEMO_NAME} | GPUs: ${NUM_GPUS} | Config: ${CONFIG_NAME}"
echo "Data: ${DATA_ROOT} | Processed: ${PROCESSED_ROOT}"
if [[ -n "$DEMO_NUM" ]]; then
    echo "Episode: ${DEMO_NUM} | skip_existing=${SKIP_EXISTING}"
fi
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
