#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Convert contact-grounded retarget_processor demos to LeRobot v2.1.
#
# Only demos with retarget_processor/quality_report.npz accept=True are exported.
# Does not touch the legacy inpaint_processor path (see b/run_convert2lerobot.sh).
#
# Usage (from repo root phantom/):
#   bash b/run_convert2lerobot_retarget.sh
#   bash b/run_convert2lerobot_retarget.sh --task basic_pick_place
#   bash b/run_convert2lerobot_retarget.sh --dry-run
#   bash b/run_convert2lerobot_retarget.sh --require-grasp-release
# =============================================================================

TASK="basic_pick_place"
PROCESSED_ROOT="/home/a26160/DATA/test_phantom_processed"
DST_ROOT="/home/a26160/DATA/test_phantom_processed_lerobot"
SRC=""
DST=""
DRY_RUN=false
REQUIRE_GRASP_RELEASE=false
ALLOW_PARTIAL=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)                   TASK="$2";                 shift 2 ;;
        --processed-root)         PROCESSED_ROOT="$2";       shift 2 ;;
        --dst-root)               DST_ROOT="$2";             shift 2 ;;
        --src)                    SRC="$2";                  shift 2 ;;
        --dst)                    DST="$2";                  shift 2 ;;
        --dry-run)                DRY_RUN=true;              shift ;;
        --require-grasp-release)  REQUIRE_GRASP_RELEASE=true; shift ;;
        --allow-partial-frames)   ALLOW_PARTIAL=true;        shift ;;
        -h|--help)
            sed -n '3,15p' "$0"
            exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

DEMO_NAME="egodex_${TASK}"
if [[ -z "${SRC}" ]]; then
    SRC="${PROCESSED_ROOT}/${DEMO_NAME}"
fi
if [[ -z "${DST}" ]]; then
    DST="${DST_ROOT}/${DEMO_NAME}_retarget"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/.."

CMD=(
    python "${SCRIPT_DIR}/export_lerobot_retarget.py"
    --src "${SRC}"
    --dst "${DST}"
)
if $DRY_RUN; then
    CMD+=(--dry-run)
fi
if $REQUIRE_GRASP_RELEASE; then
    CMD+=(--require-grasp-release)
fi
if $ALLOW_PARTIAL; then
    CMD+=(--allow-partial-frames)
fi

echo "Task: ${TASK}"
echo "Src:  ${SRC}"
echo "Dst:  ${DST}"
echo "▸ ${CMD[*]}"
echo ""

cd "${PHANTOM_DIR}"
"${CMD[@]}"
