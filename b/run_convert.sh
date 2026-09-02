#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Convert EgoDex HDF5 → Phantom format converter
#
# Usage:
#   bash b/run_convert.sh                                    # default tasks, max 300
#   bash b/run_convert.sh --task pour                        # different task
#   bash b/run_convert.sh --tasks stack,vertical_pick_place  # batch (comma-separated)
#   bash b/run_convert.sh --task stack --task stack_unstack_plates
#   bash b/run_convert.sh --max-demos 300                    # default
#   bash b/run_convert.sh --max-demos all                    # every episode
#   bash b/run_convert.sh --max-episodes 10                  # alias of --max-demos
#   bash b/run_convert.sh --egodex-root /path/to/egodex      # custom source
#   bash b/run_convert.sh --data-root /path/to/output        # custom output
#   bash b/run_convert.sh --scale 0.5                        # half-res video + scaled K
#   bash b/run_convert.sh --height 720                       # 1280x720 + scaled K
# =============================================================================

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_TASKS=(
#   stack
#   vertical_pick_place
#   stack_unstack_plates
  stack_unstack_bowls
  stack_unstack_cups
  stack_unstack_tupperware
)
TASKS=()
# EGODEX_ROOT="/home/a26160/DATA/Ego-Dex/test"
DATA_ROOT="/tmp/zwy/DATA/test_phantom"
EGODEX_ROOT="/tmp/zwy/ego-dex/part5"
# DATA_ROOT="/home/a26160/DATA/tmp/test_phantom"
MAX_DEMOS="300"
OVERWRITE=false
SCALE=""
HEIGHT="720"

append_tasks() {
    local raw="$1"
    local part
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        if [[ -n "$part" ]]; then
            TASKS+=("$part")
        fi
    done
}

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task|--tasks)  append_tasks "$2"; shift 2 ;;
        --egodex-root)   EGODEX_ROOT="$2";  shift 2 ;;
        --data-root)     DATA_ROOT="$2";    shift 2 ;;
        --max-demos|--max-episodes) MAX_DEMOS="$2"; shift 2 ;;
        --overwrite)     OVERWRITE=true;    shift   ;;
        --scale)         SCALE="$2";        shift 2 ;;
        --height)        HEIGHT="$2";       shift 2 ;;
        -h|--help)
            sed -n '3,20p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=("${DEFAULT_TASKS[@]}")
fi
if [[ -n "$SCALE" && -n "$HEIGHT" ]]; then
    echo "pass only one of --scale or --height" >&2
    exit 1
fi

# --max-demos all|0|none → unlimited; default is 300.
if [[ "${MAX_DEMOS}" == "all" || "${MAX_DEMOS}" == "0" || "${MAX_DEMOS}" == "none" ]]; then
    MAX_DEMOS=""
elif [[ ! "${MAX_DEMOS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-demos must be a positive integer or all" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

COMMON_ARGS=("--egodex-root" "$EGODEX_ROOT" "--output-root" "$DATA_ROOT")
if [[ -n "$MAX_DEMOS" ]]; then
    COMMON_ARGS+=("--max-episodes" "$MAX_DEMOS")
fi
if $OVERWRITE; then
    COMMON_ARGS+=("--overwrite")
fi
if [[ -n "$SCALE" ]]; then
    COMMON_ARGS+=("--scale" "$SCALE")
fi
if [[ -n "$HEIGHT" ]]; then
    COMMON_ARGS+=("--height" "$HEIGHT")
fi

FAILED=()
for TASK in "${TASKS[@]}"; do
    echo "═══ CONVERT: ${TASK} ═══"
    echo "Source: ${EGODEX_ROOT}/${TASK}"
    echo "Output: ${DATA_ROOT}/egodex_${TASK}"
    echo "Max demos: ${MAX_DEMOS:-all}"
    if [[ -n "$SCALE" ]]; then
        echo "Scale:  ${SCALE}"
    elif [[ -n "$HEIGHT" ]]; then
        echo "Height: ${HEIGHT}"
    fi
    echo ""

    if python "${SCRIPT_DIR}/convert_egodex.py" --task "$TASK" "${COMMON_ARGS[@]}"; then
        echo ""
    else
        echo "FAILED: ${TASK}"
        echo ""
        FAILED+=("$TASK")
    fi
done

echo "═══ SUMMARY ═══"
echo "Tasks: ${#TASKS[@]}  ok: $((${#TASKS[@]} - ${#FAILED[@]}))  failed: ${#FAILED[@]}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "Failed: ${FAILED[*]}"
    exit 1
fi
