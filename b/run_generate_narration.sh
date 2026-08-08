#!/usr/bin/env bash
set -euo pipefail

# Generate narration.csv for LeRobot export from EgoDex HDF5 metadata.
#
# Usage:
#   bash b/run_generate_narration.sh
#   bash b/run_generate_narration.sh --task basic_pick_place --overwrite
#   bash b/run_generate_narration.sh --dry-run

TASK="basic_pick_place"
EGODEX_ROOT="/home/a26160/DATA/test"
PROCESSED_ROOT="/home/a26160/DATA/test_phantom_processed"
OVERWRITE=""
DRY_RUN=""
FIELD="llm_description"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)           TASK="$2";           shift 2 ;;
        --egodex-root)    EGODEX_ROOT="$2";    shift 2 ;;
        --processed-root) PROCESSED_ROOT="$2"; shift 2 ;;
        --field)          FIELD="$2";          shift 2 ;;
        --overwrite)      OVERWRITE="--overwrite"; shift ;;
        --dry-run)        DRY_RUN="--dry-run"; shift ;;
        -h|--help)
            sed -n '3,8p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom

python "${SCRIPT_DIR}/generate_narration_csv.py" \
    --task "$TASK" \
    --egodex-root "$EGODEX_ROOT" \
    --processed-root "$PROCESSED_ROOT" \
    --field "$FIELD" \
    $OVERWRITE $DRY_RUN
