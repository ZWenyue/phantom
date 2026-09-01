#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# One-shot EgoDex → contact retarget → LeRobot (human2robot).
#
# Per task:
#   1) optional DA3 depth  (skipped if depth.npy already exists)
#   2) intent → stageb → overlay   (needs depth.npy)
#   3) generate narration.csv from EgoDex HDF5 metadata
#   4) export accept=True demos to LeRobot v2.1
#
# Usage (from repo root phantom/):
#   bash b/run_human2robot_all.sh
#   bash b/run_human2robot_all.sh --task stack
#   bash b/run_human2robot_all.sh --tasks stack,vertical_pick_place
#   bash b/run_human2robot_all.sh --with-depth
#   bash b/run_human2robot_all.sh --stage retarget,lerobot
#   bash b/run_human2robot_all.sh --stage retarget,narration,lerobot
#   bash b/run_human2robot_all.sh --per-gpu 4
#   bash b/run_human2robot_all.sh --dry-run
# =============================================================================

DEFAULT_TASKS=(
  stack
  vertical_pick_place
  stack_unstack_plates
  stack_unstack_bowls
  stack_unstack_cups
  stack_unstack_tupperware
)
TASKS=()
EGODEX_ROOT="/home/a26160/DATA/Ego-Dex/test"
DATA_ROOT="/tmp/zwy/DATA/test_phantom"
PROCESSED_ROOT="/tmp/zwy/DATA/test_phantom_processed"
LEROBOT_ROOT="/home/a26160/DATA/test_phantom_processed_lerobot"
CONFIG_NAME="egodex_panda_intent"
GPU="0-7"
DA3_JOBS=8
RETARGET_JOBS=""
RETARGET_PER_GPU=4
STAGE="retarget,lerobot"
WITH_DEPTH=false
DRY_RUN=false
SKIP_EXISTING=false

append_tasks() {
    local raw="$1" part
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        [[ -n "${part}" ]] && TASKS+=("${part}")
    done
}

has_stage() {
    local want="$1"
    [[ ",${STAGE}," == *",${want},"* ]]
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task|--tasks)      append_tasks "$2";          shift 2 ;;
        --egodex-root)       EGODEX_ROOT="$2";           shift 2 ;;
        --data-root)         DATA_ROOT="$2";             shift 2 ;;
        --processed-root)    PROCESSED_ROOT="$2";        shift 2 ;;
        --lerobot-root)      LEROBOT_ROOT="$2";          shift 2 ;;
        --config)            CONFIG_NAME="$2";           shift 2 ;;
        --gpu|--gpus)        GPU="$2";                   shift 2 ;;
        --da3-jobs)          DA3_JOBS="$2";              shift 2 ;;
        --jobs)              RETARGET_JOBS="$2";         shift 2 ;;
        --per-gpu)           RETARGET_PER_GPU="$2";      shift 2 ;;
        --stage)             STAGE="$2";                 shift 2 ;;
        --with-depth)        WITH_DEPTH=true;            shift   ;;
        --skip)              SKIP_EXISTING=true;         shift   ;;
        --dry-run)           DRY_RUN=true;               shift   ;;
        -h|--help)
            sed -n '3,22p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=("${DEFAULT_TASKS[@]}")
fi

# --with-depth prepends DA3 unless the user already listed it.
if $WITH_DEPTH && ! has_stage depth; then
    STAGE="depth,${STAGE}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DA3_SH="/home/a26160/SRC/Depth-Anything-3/b/run_egodex_da3_depth.sh"
RETARGET_SH="${SCRIPT_DIR}/run_contact_retarget.sh"
NARRATION_PY="${SCRIPT_DIR}/generate_narration_csv.py"
LEROBOT_SH="${SCRIPT_DIR}/run_convert2lerobot_retarget.sh"

# The narration step runs python here rather than in a sub-script, so this needs
# its own env. The sub-scripts activate phantom themselves.
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# torch's pip wheel pulls in the system libstdc++, which is older than what the
# env's icu/sqlite need (CXXABI_1.3.15). Whichever loads first wins the soname,
# so force the conda copy.
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

run_or_echo() {
    echo "▸ $*"
    if ! $DRY_RUN; then
        "$@"
    fi
}

echo "Tasks:     ${TASKS[*]}"
echo "Stage:     ${STAGE}"
echo "EgoDex:    ${EGODEX_ROOT}"
echo "Phantom:   ${DATA_ROOT}"
echo "Processed: ${PROCESSED_ROOT}"
echo "LeRobot:   ${LEROBOT_ROOT}"
echo "GPU:       ${GPU}  da3_jobs=${DA3_JOBS}  retarget_per_gpu=${RETARGET_PER_GPU}  retarget_jobs=${RETARGET_JOBS:-auto}"
echo ""

FAILED=()
for TASK in "${TASKS[@]}"; do
    echo "════════════════════════════════════════"
    echo "═══ ${TASK}  $(date -Is)"
    echo "════════════════════════════════════════"

    if has_stage depth; then
        echo "── DA3 depth ──"
        set +e
        run_or_echo bash "${DA3_SH}" \
            --task "${TASK}" \
            --all-demos \
            --egodex-root "${EGODEX_ROOT}" \
            --processed-root "${PROCESSED_ROOT}" \
            --gpu "${GPU}" \
            --jobs "${DA3_JOBS}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            echo "[warn] DA3 ${TASK} rc=${rc} (continuing; demos without depth.npy are skipped)"
        fi
    fi

    if has_stage retarget; then
        echo "── contact retarget ──"
        RETARGET_ARGS=(
            --task "${TASK}"
            --all-demos
            --config "${CONFIG_NAME}"
            --egodex-root "${EGODEX_ROOT}"
            --data-root "${DATA_ROOT}"
            --processed-root "${PROCESSED_ROOT}"
            --gpu "${GPU}"
            --per-gpu "${RETARGET_PER_GPU}"
        )
        if [[ -n "${RETARGET_JOBS}" ]]; then
            RETARGET_ARGS+=(--jobs "${RETARGET_JOBS}")
        fi
        $SKIP_EXISTING && RETARGET_ARGS+=(--skip)
        $DRY_RUN && RETARGET_ARGS+=(--dry-run)
        set +e
        bash "${RETARGET_SH}" "${RETARGET_ARGS[@]}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            echo "[warn] retarget ${TASK} rc=${rc}"
            FAILED+=("retarget:${TASK}:${rc}")
        fi
    fi

    if has_stage narration || has_stage lerobot; then
        echo "── narration generation ──"
        NARRATION_ARGS=(
            python "${NARRATION_PY}"
            --task "${TASK}"
            --egodex-root "${EGODEX_ROOT}"
            --processed-root "${PROCESSED_ROOT}"
        )
        $DRY_RUN && NARRATION_ARGS+=(--dry-run)
        set +e
        run_or_echo "${NARRATION_ARGS[@]}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            echo "[warn] narration ${TASK} rc=${rc}"
            FAILED+=("narration:${TASK}:${rc}")
        fi
    fi

    if has_stage lerobot; then
        echo "── LeRobot export ──"
        LEROBOT_ARGS=(
            --task "${TASK}"
            --processed-root "${PROCESSED_ROOT}"
            --dst-root "${LEROBOT_ROOT}"
        )
        $DRY_RUN && LEROBOT_ARGS+=(--dry-run)
        set +e
        bash "${LEROBOT_SH}" "${LEROBOT_ARGS[@]}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            echo "[warn] lerobot ${TASK} rc=${rc}"
            FAILED+=("lerobot:${TASK}:${rc}")
        fi
    fi
done

echo ""
echo "═══ ALL TASKS DONE  $(date -Is) ═══"
echo "LeRobot root: ${LEROBOT_ROOT}"
if [[ ${#FAILED[@]} -eq 0 ]]; then
    echo "Failed stages: none"
    exit 0
fi
echo "Failed stages: ${FAILED[*]}"
exit 1
