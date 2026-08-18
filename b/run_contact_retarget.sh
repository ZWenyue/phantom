#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Contact-grounded retargeting for one EgoDex demo.
#
#   1) export   objects.json + GT hands + T_camera  (overwrite)
#   2) process  intent → stageb → overlay           (YOLO+SAM, no mask reuse)
#
# Needs depth.npy already in the processed demo dir.
#
# Usage (from repo root phantom/):
#   bash b/run_contact_retarget.sh --demo-num 1
#   bash b/run_contact_retarget.sh --demo-num 1 --step stageb,retarget_inpaint
#   bash b/run_contact_retarget.sh --demo-num 1 --reuse-masks
#   bash b/run_contact_retarget.sh --demo-num 1 --skip-export
#   bash b/run_contact_retarget.sh --dry-run
# =============================================================================

TASK="basic_pick_place"
STEP="all"
CONFIG_NAME="egodex_panda_intent"
DEMO_NUM="0"
EGODEX_ROOT="/home/a26160/DATA/test"
DATA_ROOT="/home/a26160/DATA/test_phantom"
PROCESSED_ROOT="/home/a26160/DATA/test_phantom_processed"
SKIP_EXISTING="false"
REUSE_MASKS="false"
SKIP_EXPORT=false
MAX_NFEV=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)            TASK="$2";            shift 2 ;;
        --step)            STEP="$2";            shift 2 ;;
        --config)          CONFIG_NAME="$2";     shift 2 ;;
        --demo-num|--demo) DEMO_NUM="$2";        shift 2 ;;
        --egodex-root)     EGODEX_ROOT="$2";     shift 2 ;;
        --data-root)       DATA_ROOT="$2";       shift 2 ;;
        --processed-root)  PROCESSED_ROOT="$2";  shift 2 ;;
        --skip)            SKIP_EXISTING="true"; shift   ;;
        --reuse-masks)     REUSE_MASKS="true";   shift   ;;
        --no-reuse-masks)  REUSE_MASKS="false";  shift   ;;
        --skip-export)     SKIP_EXPORT=true;     shift   ;;
        --max-nfev)        MAX_NFEV="$2";        shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift   ;;
        -h|--help)
            sed -n '3,18p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

DEMO_NAME="egodex_${TASK}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/../phantom"

hydra_mode_arg() {
    local step="$1"
    if [[ "$step" == "all" ]]; then
        echo 'mode=[intent,stageb,retarget_inpaint]'
        return
    fi
    IFS=',' read -ra parts <<< "$step"
    if [[ ${#parts[@]} -eq 1 ]]; then
        echo "mode=${parts[0]}"
        return
    fi
    local joined
    joined=$(IFS=,; echo "${parts[*]}")
    echo "mode=[${joined}]"
}

run() {
    echo "▸ $*"
    if ! $DRY_RUN; then
        "$@"
    fi
}

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="${HOME}/.local/share/glvnd/egl_vendor.d"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

echo "Task: ${TASK} | Demo: ${DEMO_NAME}/${DEMO_NUM} | Config: ${CONFIG_NAME}"
echo "Step: ${STEP} | reuse_masks=${REUSE_MASKS} | skip_export=${SKIP_EXPORT} | max_nfev=${MAX_NFEV:-config}"
echo "EgoDex: ${EGODEX_ROOT}/${TASK} | Processed: ${PROCESSED_ROOT}"
echo ""

if ! $SKIP_EXPORT; then
    echo "═══ EXPORT objects + GT hands ═══"
    run python "${SCRIPT_DIR}/export_egodex_objects.py" \
        --task "${TASK}" \
        --demo "${DEMO_NUM}" \
        --egodex-root "${EGODEX_ROOT}" \
        --processed-root "${PROCESSED_ROOT}" \
        --overwrite
    run python "${SCRIPT_DIR}/export_egodex_hand_gt.py" \
        --task "${TASK}" \
        --demo "${DEMO_NUM}" \
        --egodex-root "${EGODEX_ROOT}" \
        --processed-root "${PROCESSED_ROOT}" \
        --overwrite
    echo ""
fi

MODE_ARG="$(hydra_mode_arg "$STEP")"
CMD=(
    python process_data.py
    --config-path=../b/configs
    --config-name="${CONFIG_NAME}"
    "${MODE_ARG}"
    demo_num="${DEMO_NUM}"
    demo_name="${DEMO_NAME}"
    data_root_dir="${DATA_ROOT}"
    processed_data_root_dir="${PROCESSED_ROOT}"
    skip_existing="${SKIP_EXISTING}"
    intent_reuse_masks="${REUSE_MASKS}"
    n_processes=1
)
if [[ -n "${MAX_NFEV}" ]]; then
    CMD+=("stageb_max_nfev=${MAX_NFEV}")
fi

echo "═══ INTENT → STAGEB → RETARGET ═══"
echo "▸ (cd ${PHANTOM_DIR} && ${CMD[*]})"
echo ""

if ! $DRY_RUN; then
    cd "$PHANTOM_DIR"
    "${CMD[@]}"
fi

echo ""
echo "═══ DONE ═══"
echo "  objects:  ${PROCESSED_ROOT}/${DEMO_NAME}/${DEMO_NUM}/objects.json"
echo "  intent:   ${PROCESSED_ROOT}/${DEMO_NAME}/${DEMO_NUM}/intent_processor/"
echo "  stageb:   ${PROCESSED_ROOT}/${DEMO_NAME}/${DEMO_NUM}/stageb_processor/"
echo "  retarget: ${PROCESSED_ROOT}/${DEMO_NAME}/${DEMO_NUM}/retarget_processor/video_overlay.mkv"
