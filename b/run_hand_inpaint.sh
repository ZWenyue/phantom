#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Arm mask + ProPainter hand_inpaint, batched over EgoDex tasks.
#
# Default: every egodex_* under DATA_ROOT, every numeric demo, GPUs 0-5.
# Concurrency: NWORKERS = n_gpus * per_gpu (H200 can pack several demos/GPU).
# Masks run in phantom conda; inpaint uses a separate propainter python.
#
# Usage (from repo root phantom/):
#   bash b/run_hand_inpaint.sh
#   bash b/run_hand_inpaint.sh --task stack
#   bash b/run_hand_inpaint.sh --tasks stack,vertical_pick_place --gpu 0-5 --per-gpu 6
#   bash b/run_hand_inpaint.sh --task stack --demo-num 0
#   bash b/run_hand_inpaint.sh --inpaint-only
#   bash b/run_hand_inpaint.sh --seg-only
# =============================================================================

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"

DEFAULT_TASKS=(
    stack
    vertical_pick_place
    stack_unstack_plates
    stack_unstack_bowls
    stack_unstack_cups
    stack_unstack_tupperware
)
TASKS=()
CONFIG_NAME="egodex_panda_intent"
DEMO_NUM=""
ALL_DEMOS=true
STEP="all"
# DATA_ROOT="/tmp/zwy/DATA/test_phantom"
DATA_ROOT="/home/a26160/DATA/tmp/test_phantom"
PROCESSED_ROOT="/tmp/zwy/DATA/test_phantom_processed"
SKIP_EXISTING="false"
DRY_RUN=false
GPU_SPEC="${CUDA_VISIBLE_DEVICES:-0-7}"
GPU_SPEC_SET=true
JOBS=""
PER_GPU=""
GPUS=()
NWORKERS=1
PROPAINTER_PYTHON="${PROPAINTER_PYTHON:-}"
PROPAINTER_ROOT="/home/a26160/SRC/ProPainter"
TASK=""
DEMO_NAME=""
RESOLUTION_HYDRA=()

DEFAULT_STEPS=(bbox hand2d arm_segmentation hand_inpaint)

append_tasks() {
    local raw="$1" part
    IFS=',' read -ra parts <<< "$raw"
    for part in "${parts[@]}"; do
        part="${part#"${part%%[![:space:]]*}"}"
        part="${part%"${part##*[![:space:]]}"}"
        if [[ "${part}" == egodex_* ]]; then
            part="${part#egodex_}"
        fi
        [[ -n "${part}" ]] && TASKS+=("${part}")
    done
}

discover_tasks() {
    local d base
    shopt -s nullglob
    for d in "${DATA_ROOT}"/egodex_*/; do
        base="$(basename "${d}")"
        echo "${base#egodex_}"
    done | sort -u
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task|--tasks)    append_tasks "$2";    shift 2 ;;
        --config)          CONFIG_NAME="$2";     shift 2 ;;
        --step)            STEP="$2";            shift 2 ;;
        --inpaint-only)    STEP="hand_inpaint";  shift   ;;
        --seg-only)        STEP="bbox,hand2d,arm_segmentation"; shift ;;
        --demo-num|--demo)
            DEMO_NUM="$2"
            ALL_DEMOS=false
            if [[ "$2" == "all" ]]; then
                ALL_DEMOS=true
                DEMO_NUM=""
            fi
            shift 2 ;;
        --all-demos)       ALL_DEMOS=true; DEMO_NUM=""; shift ;;
        --data-root)       DATA_ROOT="$2";       shift 2 ;;
        --processed-root)  PROCESSED_ROOT="$2";  shift 2 ;;
        --skip)            SKIP_EXISTING="true"; shift   ;;
        --gpu|--gpus)      GPU_SPEC="$2"; GPU_SPEC_SET=true; shift 2 ;;
        --jobs)            JOBS="$2";            shift 2 ;;
        --per-gpu)         PER_GPU="$2";         shift 2 ;;
        --propainter-python) PROPAINTER_PYTHON="$2"; shift 2 ;;
        --propainter-root)   PROPAINTER_ROOT="$2";   shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift   ;;
        -h|--help)
            sed -n '3,16p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/../phantom"

if [[ ${#TASKS[@]} -eq 0 ]]; then
    TASKS=("${DEFAULT_TASKS[@]}")
fi

has_step() {
    local want="$1"
    local s
    if [[ "${STEP}" == "all" ]]; then
        for s in "${DEFAULT_STEPS[@]}"; do
            [[ "${s}" == "${want}" ]] && return 0
        done
        return 1
    fi
    [[ ",${STEP}," == *",${want},"* ]]
}

hydra_mode_arg() {
    if [[ "${STEP}" == "all" ]]; then
        local joined
        joined=$(IFS=,; echo "${DEFAULT_STEPS[*]}")
        echo "mode=[${joined}]"
        return
    fi
    IFS=',' read -ra parts <<< "${STEP}"
    if [[ ${#parts[@]} -eq 1 ]]; then
        echo "mode=${parts[0]}"
        return
    fi
    echo "mode=[${STEP}]"
}

parse_gpu_spec() {
    local spec="$1"
    GPUS=()
    if [[ -z "${spec}" || "${spec}" == "all" ]]; then
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
        return
    fi
    local part a b i
    local -a parts
    IFS=',' read -ra parts <<< "${spec}"
    for part in "${parts[@]}"; do
        part="${part// /}"
        [[ -z "${part}" ]] && continue
        if [[ "${part}" =~ ^([0-9]+)-([0-9]+)$ ]]; then
            a="${BASH_REMATCH[1]}"
            b="${BASH_REMATCH[2]}"
            for ((i = a; i <= b; i++)); do
                GPUS+=("${i}")
            done
        elif [[ "${part}" =~ ^[0-9]+$ ]]; then
            GPUS+=("${part}")
        else
            echo "bad --gpu spec: ${part}" >&2
            exit 1
        fi
    done
}

if $GPU_SPEC_SET; then
    parse_gpu_spec "${GPU_SPEC}"
elif [[ -n "${GPU_SPEC}" ]]; then
    parse_gpu_spec "${GPU_SPEC}"
else
    GPUS=(0)
fi

NEED_PROPAINTER=false
has_step hand_inpaint && NEED_PROPAINTER=true

if [[ -z "${PER_GPU}" ]]; then
    if $NEED_PROPAINTER; then
        PER_GPU=4
    else
        PER_GPU=8
    fi
fi
if [[ ! "${PER_GPU}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--per-gpu must be a positive integer" >&2
    exit 1
fi

if [[ -n "${JOBS}" ]]; then
    if [[ ! "${JOBS}" =~ ^[1-9][0-9]*$ ]]; then
        echo "--jobs must be a positive integer" >&2
        exit 1
    fi
    NWORKERS="${JOBS}"
elif $ALL_DEMOS; then
    NWORKERS=$(( ${#GPUS[@]} * PER_GPU ))
else
    NWORKERS=1
fi

if $NEED_PROPAINTER; then
    if [[ -z "${PROPAINTER_PYTHON}" ]] && command -v conda >/dev/null 2>&1; then
        PROPAINTER_PYTHON="$(conda run -n propainter python -c 'import sys; print(sys.executable)' 2>/dev/null || true)"
    fi
    if [[ -z "${PROPAINTER_PYTHON}" ]]; then
        echo "Set --propainter-python or PROPAINTER_PYTHON to the propainter conda python." >&2
        echo "Do not use phantom's interpreter. Or pass --seg-only to skip inpaint." >&2
        exit 1
    fi
fi

link_if_missing() {
    local src="$1"
    local dst="$2"
    if [[ -e "$dst" || -L "$dst" ]]; then
        return 0
    fi
    if [[ ! -e "$src" ]]; then
        return 1
    fi
    mkdir -p "$(dirname "$dst")"
    ln -s "$(realpath "$src")" "$dst"
    echo "  linked $(basename "$dst") <- ${src}"
}

ensure_demo_inputs() {
    local demo="$1"
    local processed="${PROCESSED_ROOT}/${DEMO_NAME}/${demo}"
    local raw="${DATA_ROOT}/${DEMO_NAME}/${demo}"
    mkdir -p "$processed"
    # Processed dirs often already exist (DA3/intent), so copytree never
    # copies convert outputs. Epic bbox needs hand_det.pkl in processed/.
    if ! link_if_missing "${raw}/hand_det.pkl" "${processed}/hand_det.pkl"; then
        echo "missing hand_det.pkl for ${DEMO_NAME}/${demo} (run convert first: ${raw}/hand_det.pkl)" >&2
        return 1
    fi
    link_if_missing "${raw}/video_L.mp4" "${processed}/video_L.mp4" || true
}

list_demos() {
    local dir="${PROCESSED_ROOT}/${DEMO_NAME}"
    if [[ ! -d "$dir" ]]; then
        dir="${DATA_ROOT}/${DEMO_NAME}"
    fi
    if [[ ! -d "$dir" ]]; then
        echo "[warn] missing task dir: ${PROCESSED_ROOT}/${DEMO_NAME} and ${DATA_ROOT}/${DEMO_NAME}" >&2
        return 0
    fi
    local d base
    shopt -s nullglob
    for d in "${dir}"/*/; do
        base="$(basename "${d}")"
        [[ "${base}" =~ ^[0-9]+$ ]] || continue
        if has_step hand_inpaint && ! has_step arm_segmentation && [[ ! -f "${d}segmentation_processor/masks_arm.npy" ]]; then
            local alt="${PROCESSED_ROOT}/${DEMO_NAME}/${base}/segmentation_processor/masks_arm.npy"
            if [[ ! -f "${alt}" ]]; then
                echo "[warn] skip demo ${base}: no masks_arm.npy (run without --inpaint-only)" >&2
                continue
            fi
        fi
        echo "${base}"
    done | sort -n
}

load_convert_resolution_overrides() {
    RESOLUTION_HYDRA=()
    local meta="${DATA_ROOT}/${DEMO_NAME}/convert_meta.json"
    if [[ ! -f "${meta}" ]]; then
        return 0
    fi
    local parsed h cam
    parsed="$(python -c "import json; d=json.load(open('${meta}')); print(int(d['input_resolution']), d['camera_intrinsics'])")"
    h="${parsed%% *}"
    cam="${parsed#* }"
    RESOLUTION_HYDRA=(
        "input_resolution=${h}"
        "output_resolution=${h}"
        "camera_intrinsics=${cam}"
    )
    echo "Resolution: ${h}p  camera_intrinsics=${cam}  (from ${meta})"
}

run_one() {
    local demo="$1"
    local gpu="$2"
    local out_dir="${PROCESSED_ROOT}/${DEMO_NAME}/${demo}"
    local mode_arg
    mode_arg="$(hydra_mode_arg)"
    local CMD=(
        python process_data.py
        --config-path=../b/configs
        --config-name="${CONFIG_NAME}"
        "${mode_arg}"
        demo_num="${demo}"
        demo_name="${DEMO_NAME}"
        data_root_dir="${DATA_ROOT}"
        processed_data_root_dir="${PROCESSED_ROOT}"
        skip_existing="${SKIP_EXISTING}"
        n_processes=1
    )
    if [[ ${#RESOLUTION_HYDRA[@]} -gt 0 ]]; then
        CMD+=("${RESOLUTION_HYDRA[@]}")
    fi
    if $NEED_PROPAINTER; then
        CMD+=(
            inpaint_backend=propainter
            "propainter_root=${PROPAINTER_ROOT}"
            "propainter_python=${PROPAINTER_PYTHON}"
        )
    fi
    echo "Task: ${TASK} | Demo: ${DEMO_NAME}/${demo} | GPU: ${gpu} | ${mode_arg}"
    echo "▸ CUDA_VISIBLE_DEVICES=${gpu} (cd ${PHANTOM_DIR} && ${CMD[*]})"
    echo ""
    if $DRY_RUN; then
        echo "  mask: ${out_dir}/segmentation_processor/masks_arm.npy"
        echo "  mkv:  ${out_dir}/inpaint_processor/video_human_inpaint.mkv"
        return 0
    fi
    ensure_demo_inputs "${demo}" || return 1
    (
        cd "${PHANTOM_DIR}"
        CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}"
    )
}

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# torch's pip wheel pulls in the system libstdc++, which is older than what the
# env's icu/sqlite need (CXXABI_1.3.15). Whichever loads first wins the soname,
# so force the conda copy. Preloading one lib (not the whole lib dir) keeps the
# propainter subprocess on its own libraries.
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi

run_task_demos() {
    TASK_FAILED=0
    if ! $ALL_DEMOS; then
        set +e
        run_one "${DEMO_NUM}" "${GPUS[0]}"
        local rc=$?
        set -e
        if [[ "${rc}" -ne 0 ]]; then
            TASK_FAILED=1
        fi
        return 0
    fi

    local -a DEMOS=()
    mapfile -t DEMOS < <(list_demos)
    if [[ ${#DEMOS[@]} -eq 0 ]]; then
        echo "[warn] no demos for ${DEMO_NAME}" >&2
        return 0
    fi
    echo "Batch: ${DEMO_NAME}  ${#DEMOS[@]} demo(s)  GPUs: ${GPUS[*]}  jobs: ${NWORKERS}"

    local i=0 running=0 rc gpu
    if [[ "${NWORKERS}" -eq 1 ]] || $DRY_RUN; then
        for demo in "${DEMOS[@]}"; do
            gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
            i=$((i + 1))
            set +e
            run_one "${demo}" "${gpu}"
            rc=$?
            set -e
            if [[ "${rc}" -ne 0 ]]; then
                TASK_FAILED=$((TASK_FAILED + 1))
            fi
        done
        return 0
    fi
    for demo in "${DEMOS[@]}"; do
        gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
        i=$((i + 1))
        while (( running >= NWORKERS )); do
            set +e
            wait -n
            rc=$?
            set -e
            running=$((running - 1))
            if [[ "${rc}" -ne 0 ]]; then
                TASK_FAILED=$((TASK_FAILED + 1))
            fi
        done
        run_one "${demo}" "${gpu}" &
        running=$((running + 1))
    done
    while (( running > 0 )); do
        set +e
        wait -n
        rc=$?
        set -e
        running=$((running - 1))
        if [[ "${rc}" -ne 0 ]]; then
            TASK_FAILED=$((TASK_FAILED + 1))
        fi
    done
}

echo "Tasks:      ${TASKS[*]}"
echo "Step:       ${STEP}"
echo "Mode:       $(hydra_mode_arg)"
echo "Demos:      $(if $ALL_DEMOS; then echo all; else echo "${DEMO_NUM}"; fi)"
if $NEED_PROPAINTER; then
    echo "Python:     ${PROPAINTER_PYTHON}"
    echo "Root:       ${PROPAINTER_ROOT}"
fi
echo "Data:       ${DATA_ROOT}"
echo "Processed:  ${PROCESSED_ROOT}"
echo "GPU:        ${GPUS[*]}  jobs=${NWORKERS}  per_gpu=${PER_GPU}"
echo ""

FAILED_TASKS=()
TOTAL_FAILED=0
for TASK in "${TASKS[@]}"; do
    DEMO_NAME="egodex_${TASK}"
    load_convert_resolution_overrides
    echo "════════════════════════════════════════"
    echo "═══ ${DEMO_NAME}  $(date -Is)"
    echo "════════════════════════════════════════"
    run_task_demos
    if [[ "${TASK_FAILED}" -ne 0 ]]; then
        FAILED_TASKS+=("${TASK}:${TASK_FAILED}")
        TOTAL_FAILED=$((TOTAL_FAILED + TASK_FAILED))
    fi
done

echo ""
echo "═══ ALL TASKS DONE  $(date -Is) ═══"
echo "Then re-run retarget_inpaint so overlay uses the de-handed background."
if [[ ${#FAILED_TASKS[@]} -eq 0 ]]; then
    echo "Failed: none"
    exit 0
fi
echo "Failed: ${FAILED_TASKS[*]}  (demos=${TOTAL_FAILED})"
exit 1
