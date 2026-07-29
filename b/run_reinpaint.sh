#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Re-run hand_inpaint + robot_inpaint after mask dilation tuning
#
# Deletes existing inpaint outputs so skip_existing won't short-circuit,
# then runs hand_inpaint followed by robot_inpaint.
#
# Usage:
#   bash b/run_reinpaint.sh                          # all episodes
#   bash b/run_reinpaint.sh --task pour              # different task
#   bash b/run_reinpaint.sh --gpus 4                 # use 4 GPUs
#   bash b/run_reinpaint.sh --episodes 0-5           # only episodes 0..5
#   bash b/run_reinpaint.sh --dry-run                # print only
#   bash b/run_reinpaint.sh --hand-only              # skip robot_inpaint
#
# Dilation params are in b/configs/egodex.yaml:
#   mask_dilate_kernel, mask_dilate_size, mask_dilate_iterations
# =============================================================================

TASK="basic_fold"
NUM_GPUS=4
DATA_ROOT="/mnt/r/DATA/EgoDex/test_phantom"
PROCESSED_ROOT="/mnt/r/DATA/EgoDex/test_phantom_processed"
DRY_RUN=false
HAND_ONLY=false
EPISODES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)            TASK="$2";            shift 2 ;;
        --gpus)            NUM_GPUS="$2";        shift 2 ;;
        --data-root)       DATA_ROOT="$2";       shift 2 ;;
        --processed-root)  PROCESSED_ROOT="$2";  shift 2 ;;
        --episodes)        EPISODES="$2";        shift 2 ;;
        --hand-only)       HAND_ONLY=true;       shift   ;;
        --dry-run)         DRY_RUN=true;         shift   ;;
        -h|--help)         sed -n '3,14p' "$0";  exit 0  ;;
        *)                 echo "Unknown option: $1"; exit 1 ;;
    esac
done

DEMO_NAME="egodex_${TASK}"
DEMO_DIR="${PROCESSED_ROOT}/${DEMO_NAME}"

# ── resolve episode range ────────────────────────────────────────────────────
if [[ -n "$EPISODES" ]]; then
    IFS='-' read -r EP_START EP_END <<< "$EPISODES"
    EP_END=${EP_END:-$EP_START}
else
    EP_START=0
    EP_END=$(( $(find "$DEMO_DIR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l) - 1 ))
fi

echo "Task: ${TASK} | Demo: ${DEMO_NAME} | GPUs: ${NUM_GPUS}"
echo "Episodes: ${EP_START}-${EP_END}"
echo "Processed: ${PROCESSED_ROOT}"
echo ""

# ── clean old inpaint outputs ────────────────────────────────────────────────
echo "═══ CLEANING OLD INPAINT OUTPUTS ═══"
for ep in $(seq "$EP_START" "$EP_END"); do
    ep_dir="${DEMO_DIR}/${ep}"
    if [[ ! -d "$ep_dir" ]]; then continue; fi

    for subdir in segmentation_processor inpaint_processor; do
        target="${ep_dir}/${subdir}"
        if [[ -d "$target" ]]; then
            echo "  rm -rf ${target}"
            if ! $DRY_RUN; then rm -rf "$target"; fi
        fi
    done
    # also remove robot overlay videos (glob for any robot/setup combo)
    for f in "${ep_dir}"/video_overlay_*.mkv; do
        if [[ -f "$f" ]]; then
            echo "  rm ${f}"
            if ! $DRY_RUN; then rm "$f"; fi
        fi
    done
done
echo ""

# ── re-run ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/../phantom"
CONFIG_ARGS="--config-path=../b/configs --config-name=egodex"
DATA_ARGS="data_root_dir=${DATA_ROOT} processed_data_root_dir=${PROCESSED_ROOT}"

# Activate phantom conda env
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom

STEPS=(arm_segmentation hand_inpaint)
if ! $HAND_ONLY; then
    STEPS+=(robot_inpaint)
fi

for mode in "${STEPS[@]}"; do
    echo "═══ RE-RUNNING: ${mode^^} (episodes ${EP_START}-${EP_END}, ${NUM_GPUS} GPUs) ═══"

    pids=()
    for ep in $(seq "$EP_START" "$EP_END"); do
        gpu_id=$((ep % NUM_GPUS))
        log_file="/tmp/phantom_reinpaint_${DEMO_NAME}_${mode}_ep${ep}.log"
        echo "  ep${ep}(GPU${gpu_id}) → ${log_file}"

        if ! $DRY_RUN; then
            (
                cd "$PHANTOM_DIR"
                CUDA_VISIBLE_DEVICES=${gpu_id} \
                python process_data.py \
                    ${CONFIG_ARGS} \
                    ${DATA_ARGS} \
                    demo_name="${DEMO_NAME}" \
                    mode="${mode}" \
                    demo_num="${ep}" \
                    skip_existing=false \
                    n_processes=1
            ) > "$log_file" 2>&1 &
            pids+=($!)
        fi
    done

    if ! $DRY_RUN && [[ ${#pids[@]} -gt 0 ]]; then
        echo "  Waiting for ${#pids[@]} episode(s)..."
        local_failed=0
        for pid in "${pids[@]}"; do
            wait "$pid" || ((local_failed++)) || true
        done
        if [[ $local_failed -gt 0 ]]; then
            echo "  WARNING: ${local_failed} episode(s) had errors — check /tmp/phantom_reinpaint_${DEMO_NAME}_${mode}_ep*.log"
        else
            echo "  Done: ${mode}"
        fi
    fi
    echo ""
done

echo "═══ ALL DONE ═══"
