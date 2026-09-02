#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# One-shot EgoDex → contact retarget → LeRobot (human2robot).
#
# Per task:
#   1) DA3 depth (default on; skipped if depth.npy already exists)
#   2) remove both human hands/arms with ProPainter
#   3) per-hand intent → matching-arm stageb → bimanual overlay
#   4) generate narration.csv from EgoDex HDF5 metadata
#   5) export accept=True demos to LeRobot v2.1
#
# Usage (from repo root phantom/):
#   bash b/run_human2robot_all.sh
#   bash b/run_human2robot_all.sh --task stack
#   bash b/run_human2robot_all.sh --tasks stack,vertical_pick_place
#   bash b/run_human2robot_all.sh --no-depth          # skip DA3 (default is on)
#   bash b/run_human2robot_all.sh --stage inpaint,retarget,lerobot
#   bash b/run_human2robot_all.sh --stage retarget,lerobot
#   bash b/run_human2robot_all.sh --stage retarget,narration,lerobot
#   bash b/run_human2robot_all.sh --per-gpu 4
#   bash b/run_human2robot_all.sh --max-demos 300   # default
#   bash b/run_human2robot_all.sh --max-demos all   # every demo
#   bash b/run_human2robot_all.sh --bg              # nohup + log, return immediately
#   bash b/run_human2robot_all.sh --bg --log /tmp/h2r.log
#   bash b/run_human2robot_all.sh --dry-run
#
# Stop a --bg job:
#   kill <Background_PID> alone is NOT enough — child stages (inpaint/retarget/
#   python/ProPainter) are orphaned and keep running (PPID→1).
#   Kill the whole tree, then verify:
#     kill <Background_PID>
#     pkill -f 'run_hand_inpaint|run_contact_retarget|run_human2robot_all|process_data.py|inference_propainter'
#     pgrep -af 'run_human2robot_all|run_hand_inpaint|run_contact_retarget|process_data.py|inference_propainter'
#   Or kill by process group of a still-living child (e.g. inpaint root):
#     kill -- -$(ps -o pgid= -p <child_pid> | tr -d ' ')
# =============================================================================

ORIG_ARGS=("$@")

DEFAULT_TASKS=(
#   stack
  vertical_pick_place
  stack_unstack_plates
#   stack_unstack_bowls
#   stack_unstack_cups
#   stack_unstack_tupperware
)
TASKS=()
EGODEX_ROOT="/home/a26160/DATA/Ego-Dex/test"
DATA_ROOT="/tmp/zwy/DATA/test_phantom"
PROCESSED_ROOT="/tmp/zwy/DATA/test_phantom_processed"
LEROBOT_ROOT="/home/a26160/DATA/test_phantom_processed_lerobot"
CONFIG_NAME="egodex_panda_intent"
GPU="0-8"
DA3_JOBS=8
RETARGET_JOBS=""
RETARGET_PER_GPU=2
STAGE="inpaint,retarget,lerobot"
WITH_DEPTH=true
DRY_RUN=false
SKIP_EXISTING=false
MAX_DEMOS="300"
BACKGROUND=false
LOG_FILE=""

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
        --max-demos)         MAX_DEMOS="$2";             shift 2 ;;
        --with-depth)        WITH_DEPTH=true;            shift   ;;
        --no-depth)          WITH_DEPTH=false;           shift   ;;
        --skip)              SKIP_EXISTING=true;         shift   ;;

        --dry-run)           DRY_RUN=true;               shift   ;;
        --bg|--background)   BACKGROUND=true;            shift   ;;
        --log)               LOG_FILE="$2";              shift 2 ;;
        -h|--help)
            sed -n '3,38p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=("${DEFAULT_TASKS[@]}")
fi

# --max-demos all|0|none → unlimited; default is 300.
if [[ "${MAX_DEMOS}" == "all" || "${MAX_DEMOS}" == "0" || "${MAX_DEMOS}" == "none" ]]; then
    MAX_DEMOS=""
elif [[ ! "${MAX_DEMOS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--max-demos must be a positive integer or all" >&2
    exit 1
fi

# --with-depth prepends DA3 unless the user already listed it.
if $WITH_DEPTH && ! has_stage depth; then
    STAGE="depth,${STAGE}"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Detach early (before conda activate) so the foreground shell returns immediately.
if $BACKGROUND; then
    if [[ -z "${LOG_FILE}" ]]; then
        LOG_FILE="${PROCESSED_ROOT}/logs/human2robot_$(date +%Y%m%d_%H%M%S).log"
    fi
    mkdir -p "$(dirname "${LOG_FILE}")"
    filtered=()
    skip_next=false
    for a in "${ORIG_ARGS[@]}"; do
        if $skip_next; then
            skip_next=false
            continue
        fi
        case "$a" in
            --bg|--background) continue ;;
            --log)             skip_next=true; continue ;;
            *)                 filtered+=("$a") ;;
        esac
    done
    nohup bash "${SCRIPT_DIR}/$(basename "$0")" "${filtered[@]}" \
        >"${LOG_FILE}" 2>&1 </dev/null &
    pid=$!
    disown "${pid}" 2>/dev/null || true
    echo "Background PID: ${pid}"
    echo "Log:            ${LOG_FILE}"
    echo "Tail:           tail -f ${LOG_FILE}"
    exit 0
fi

DA3_SH="/home/a26160/SRC/Depth-Anything-3/b/run_egodex_da3_depth.sh"
HAND_INPAINT_SH="${SCRIPT_DIR}/run_hand_inpaint.sh"
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
echo "Max demos: ${MAX_DEMOS:-all}  (per task, numeric id order)"
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
        DA3_ARGS=(
            --task "${TASK}"
            --all-demos
            --egodex-root "${EGODEX_ROOT}"
            --processed-root "${PROCESSED_ROOT}"
            --gpu "${GPU}"
            --jobs "${DA3_JOBS}"
        )
        [[ -n "${MAX_DEMOS}" ]] && DA3_ARGS+=(--max-demos "${MAX_DEMOS}")
        run_or_echo bash "${DA3_SH}" "${DA3_ARGS[@]}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            echo "[warn] DA3 ${TASK} rc=${rc} (continuing; demos without depth.npy are skipped)"
        fi
    fi

    if has_stage inpaint; then
        echo "── bimanual hand/arm removal ──"
        INPAINT_ARGS=(
            --task "${TASK}"
            --all-demos
            --config "${CONFIG_NAME}"
            --data-root "${DATA_ROOT}"
            --processed-root "${PROCESSED_ROOT}"
            --gpu "${GPU}"
            --per-gpu "${RETARGET_PER_GPU}"
        )
        if [[ -n "${RETARGET_JOBS}" ]]; then
            INPAINT_ARGS+=(--jobs "${RETARGET_JOBS}")
        fi
        [[ -n "${MAX_DEMOS}" ]] && INPAINT_ARGS+=(--max-demos "${MAX_DEMOS}")
        $SKIP_EXISTING && INPAINT_ARGS+=(--skip)
        $DRY_RUN && INPAINT_ARGS+=(--dry-run)
        set +e
        bash "${HAND_INPAINT_SH}" "${INPAINT_ARGS[@]}"
        rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            echo "[warn] inpaint ${TASK} rc=${rc}"
            FAILED+=("inpaint:${TASK}:${rc}")
            echo "[warn] skipping retarget/export for ${TASK}: clean bimanual background is required"
            continue
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
        [[ -n "${MAX_DEMOS}" ]] && RETARGET_ARGS+=(--max-demos "${MAX_DEMOS}")
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
        [[ -n "${MAX_DEMOS}" ]] && NARRATION_ARGS+=(--max-demos "${MAX_DEMOS}")
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
        [[ -n "${MAX_DEMOS}" ]] && LEROBOT_ARGS+=(--max-demos "${MAX_DEMOS}")
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
