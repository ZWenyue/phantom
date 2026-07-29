#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# EgoDex HDF5 → Phantom format converter
#
# Usage:
#   bash b/run_convert.sh                                    # default task
#   bash b/run_convert.sh --task pour                        # different task
#   bash b/run_convert.sh --max-episodes 10                  # limit episodes
#   bash b/run_convert.sh --egodex-root /path/to/egodex      # custom source
#   bash b/run_convert.sh --data-root /path/to/output        # custom output
# =============================================================================

# ── defaults ──────────────────────────────────────────────────────────────────
TASK="make_sandwich"
EGODEX_ROOT="/mnt/r/DATA/EgoDex/test"
DATA_ROOT="/mnt/r/DATA/EgoDex/test_phantom"
MAX_EPISODES=""
OVERWRITE=false

# ── parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)          TASK="$2";         shift 2 ;;
        --egodex-root)   EGODEX_ROOT="$2";  shift 2 ;;
        --data-root)     DATA_ROOT="$2";    shift 2 ;;
        --max-episodes)  MAX_EPISODES="$2"; shift 2 ;;
        --overwrite)     OVERWRITE=true;    shift   ;;
        -h|--help)
            sed -n '3,10p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "═══ CONVERT: ${TASK} ═══"
echo "Source: ${EGODEX_ROOT}/${TASK}"
echo "Output: ${DATA_ROOT}/egodex_${TASK}"
echo ""

ARGS=("--task" "$TASK" "--egodex-root" "$EGODEX_ROOT" "--output-root" "$DATA_ROOT")
if [[ -n "$MAX_EPISODES" ]]; then
    ARGS+=("--max-episodes" "$MAX_EPISODES")
fi
if $OVERWRITE; then
    ARGS+=("--overwrite")
fi

python "${SCRIPT_DIR}/convert_egodex.py" "${ARGS[@]}"
