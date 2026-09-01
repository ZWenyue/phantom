#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Contact-grounded retargeting for EgoDex demos.
#
#   1) export   objects.json + GT hands + T_camera  (overwrite)
#   2) process  intent → stageb → overlay           (YOLO+SAM, no mask reuse)
#
# Needs depth.npy already in the processed demo dir.
#
# Usage (from repo root phantom/):
#   bash b/run_contact_retarget.sh --demo-num 1
#   bash b/run_contact_retarget.sh --task basic_pick_place --all-demos
#   bash b/run_contact_retarget.sh --all-demos --gpu 0-7 --jobs 8
#   bash b/run_contact_retarget.sh --all-demos --gpu 0-7 --per-gpu 4
#   bash b/run_contact_retarget.sh --demo-num 1 --step stageb,retarget_inpaint
#   bash b/run_contact_retarget.sh --demo-num 1 --reuse-masks
#   bash b/run_contact_retarget.sh --demo-num 1 --skip-export
#   bash b/run_contact_retarget.sh --dry-run
# =============================================================================

TASK="basic_pick_place"
STEP="all"
CONFIG_NAME="egodex_panda_intent"
DEMO_NUM="0"
ALL_DEMOS=false
EGODEX_ROOT="/home/a26160/DATA/test"
DATA_ROOT="/home/a26160/DATA/test_phantom"
PROCESSED_ROOT="/home/a26160/DATA/test_phantom_processed"
SKIP_EXISTING="false"
REUSE_MASKS="false"
SKIP_EXPORT=false
MAX_NFEV=""
DRY_RUN=false
GPU_SPEC="${CUDA_VISIBLE_DEVICES:-}"
GPU_SPEC_SET=false
JOBS=""
PER_GPU=4
GPUS=()
NWORKERS=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --task)            TASK="$2";            shift 2 ;;
        --step)            STEP="$2";            shift 2 ;;
        --config)          CONFIG_NAME="$2";     shift 2 ;;
        --demo-num|--demo)
            DEMO_NUM="$2"
            if [[ "$2" == "all" ]]; then
                ALL_DEMOS=true
            fi
            shift 2 ;;
        --all-demos)       ALL_DEMOS=true;       shift   ;;
        --egodex-root)     EGODEX_ROOT="$2";     shift 2 ;;
        --data-root)       DATA_ROOT="$2";       shift 2 ;;
        --processed-root)  PROCESSED_ROOT="$2";  shift 2 ;;
        --skip)            SKIP_EXISTING="true"; shift   ;;
        --reuse-masks)     REUSE_MASKS="true";   shift   ;;
        --no-reuse-masks)  REUSE_MASKS="false";  shift   ;;
        --skip-export)     SKIP_EXPORT=true;     shift   ;;
        --max-nfev)        MAX_NFEV="$2";        shift 2 ;;
        --gpu|--gpus)      GPU_SPEC="$2"; GPU_SPEC_SET=true; shift 2 ;;
        --all-gpus)        GPU_SPEC="all"; GPU_SPEC_SET=true; shift ;;
        --jobs)            JOBS="$2";            shift 2 ;;
        --per-gpu)         PER_GPU="$2";         shift 2 ;;
        --dry-run)         DRY_RUN=true;         shift   ;;
        -h|--help)
            sed -n '3,21p' "$0"; exit 0 ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

DEMO_NAME="egodex_${TASK}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PHANTOM_DIR="${SCRIPT_DIR}/../phantom"
RESOLUTION_HYDRA=()

parse_gpu_spec() {
    local spec="$1"
    GPUS=()
    if [[ -z "${spec}" || "${spec}" == "all" ]]; then
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi not found; pass --gpu 0,1,2,..." >&2
            exit 1
        fi
        mapfile -t GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
        if [[ ${#GPUS[@]} -eq 0 ]]; then
            echo "nvidia-smi returned no GPUs" >&2
            exit 1
        fi
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
            if (( b < a )); then
                echo "bad --gpu range: ${part}" >&2
                exit 1
            fi
            for ((i = a; i <= b; i++)); do
                GPUS+=("${i}")
            done
        elif [[ "${part}" =~ ^[0-9]+$ ]]; then
            GPUS+=("${part}")
        else
            echo "bad --gpu spec: ${part}  (use 0, 0,1,2, 0-7, or all)" >&2
            exit 1
        fi
    done
    if [[ ${#GPUS[@]} -eq 0 ]]; then
        echo "empty --gpu spec" >&2
        exit 1
    fi
}

if $GPU_SPEC_SET; then
    parse_gpu_spec "${GPU_SPEC}"
elif [[ -n "${GPU_SPEC}" ]]; then
    parse_gpu_spec "${GPU_SPEC}"
else
    GPUS=(0)
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

list_demos() {
    local dir="${PROCESSED_ROOT}/${DEMO_NAME}"
    if [[ ! -d "$dir" ]]; then
        echo "missing processed task dir: ${dir}" >&2
        echo "run DA3 depth first: bash /home/a26160/SRC/Depth-Anything-3/b/run_egodex_da3_depth.sh --task ${TASK} --all-demos" >&2
        exit 1
    fi
    local d base demos=()
    shopt -s nullglob
    for d in "${dir}"/*/; do
        base="$(basename "${d}")"
        if [[ ! "${base}" =~ ^[0-9]+$ ]]; then
            echo "[warn] skip non-integer demo dir: ${base}" >&2
            continue
        fi
        if [[ ! -f "${d}depth.npy" ]]; then
            echo "[warn] skip demo ${base}: no depth.npy" >&2
            continue
        fi
        demos+=("${base}")
    done
    if [[ ${#demos[@]} -eq 0 ]]; then
        echo "no demos with depth.npy under ${dir}" >&2
        exit 1
    fi
    printf '%s\n' "${demos[@]}" | sort -n
}

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
    local egodex_mp4="${EGODEX_ROOT}/${TASK}/${demo}.mp4"
    mkdir -p "$processed" "$raw"

    if ! link_if_missing "${raw}/video_L.mp4" "${processed}/video_L.mp4"; then
        if ! link_if_missing "$egodex_mp4" "${processed}/video_L.mp4"; then
            echo "missing video for demo ${demo}: ${processed}/video_L.mp4 and ${egodex_mp4}" >&2
            return 1
        fi
    fi
    if [[ ! -e "${raw}/video_L.mp4" && ! -L "${raw}/video_L.mp4" ]]; then
        link_if_missing "${processed}/video_L.mp4" "${raw}/video_L.mp4" || \
            link_if_missing "$egodex_mp4" "${raw}/video_L.mp4" || true
    fi
}

should_skip() {
    local demo="$1"
    local overlay="${PROCESSED_ROOT}/${DEMO_NAME}/${demo}/retarget_processor/video_overlay.mkv"
    [[ "${SKIP_EXISTING}" == "true" ]] && [[ -f "$overlay" ]]
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

build_process_cmd() {
    local demo="$1"
    local mode_arg
    mode_arg="$(hydra_mode_arg "$STEP")"
    CMD=(
        python process_data.py
        --config-path=../b/configs
        --config-name="${CONFIG_NAME}"
        "${mode_arg}"
        demo_num="${demo}"
        demo_name="${DEMO_NAME}"
        data_root_dir="${DATA_ROOT}"
        processed_data_root_dir="${PROCESSED_ROOT}"
        skip_existing="${SKIP_EXISTING}"
        intent_reuse_masks="${REUSE_MASKS}"
        n_processes=1
    )
    if [[ ${#RESOLUTION_HYDRA[@]} -gt 0 ]]; then
        CMD+=("${RESOLUTION_HYDRA[@]}")
    fi
    if [[ -n "${MAX_NFEV}" ]]; then
        CMD+=("stageb_max_nfev=${MAX_NFEV}")
    fi
}

run_one() {
    local demo="$1"
    local gpu="$2"
    local out_dir="${PROCESSED_ROOT}/${DEMO_NAME}/${demo}"

    echo "Task: ${TASK} | Demo: ${DEMO_NAME}/${demo} | GPU: ${gpu} | Config: ${CONFIG_NAME}"
    echo "Step: ${STEP} | reuse_masks=${REUSE_MASKS} | skip_export=${SKIP_EXPORT} | max_nfev=${MAX_NFEV:-config}"
    echo "EgoDex: ${EGODEX_ROOT}/${TASK} | Processed: ${out_dir}"
    echo ""

    if should_skip "${demo}"; then
        echo "skip demo ${demo}: video_overlay.mkv exists (pass without --skip to rerun)"
        return 2
    fi

    if ! $DRY_RUN; then
        ensure_demo_inputs "${demo}" || return 1
    fi

    if ! $SKIP_EXPORT; then
        echo "═══ EXPORT objects + GT hands + narration ═══"
        run python "${SCRIPT_DIR}/export_egodex_objects.py" \
            --task "${TASK}" \
            --demo "${demo}" \
            --egodex-root "${EGODEX_ROOT}" \
            --processed-root "${PROCESSED_ROOT}" \
            --overwrite
        run python "${SCRIPT_DIR}/export_egodex_hand_gt.py" \
            --task "${TASK}" \
            --demo "${demo}" \
            --egodex-root "${EGODEX_ROOT}" \
            --processed-root "${PROCESSED_ROOT}" \
            --overwrite
        run python "${SCRIPT_DIR}/generate_narration_csv.py" \
            --task "${TASK}" \
            --demo "${demo}" \
            --egodex-root "${EGODEX_ROOT}" \
            --processed-root "${PROCESSED_ROOT}" \
            --overwrite
        echo ""
    fi

    build_process_cmd "${demo}"
    echo "═══ INTENT → STAGEB → RETARGET ═══"
    echo "▸ CUDA_VISIBLE_DEVICES=${gpu} (cd ${PHANTOM_DIR} && ${CMD[*]})"
    echo ""

    if $DRY_RUN; then
        echo "  objects:  ${out_dir}/objects.json"
        echo "  retarget: ${out_dir}/retarget_processor/video_overlay.mkv"
        return 0
    fi

    (
        cd "$PHANTOM_DIR"
        CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}"
    )

    echo ""
    echo "═══ DONE demo ${demo} ═══"
    echo "  objects:  ${out_dir}/objects.json"
    echo "  intent:   ${out_dir}/intent_processor/"
    echo "  stageb:   ${out_dir}/stageb_processor/"
    echo "  retarget: ${out_dir}/retarget_processor/video_overlay.mkv"
    echo ""
}

require_int_gpu() {
    local gpu="$1"
    if [[ ! "${gpu}" =~ ^[0-9]+$ ]]; then
        echo "internal error: GPU id is not an integer: ${gpu@Q}" >&2
        exit 1
    fi
}

eval "$(conda shell.bash hook 2>/dev/null)"
conda activate phantom
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONUNBUFFERED=1
# torch's pip wheel pulls in the system libstdc++, which is older than what the
# env's icu/sqlite need (CXXABI_1.3.15). Whichever loads first wins the soname,
# so force the conda copy for every interpreter in this process tree.
if [[ -f "${CONDA_PREFIX}/lib/libstdc++.so.6" ]]; then
    export LD_PRELOAD="${CONDA_PREFIX}/lib/libstdc++.so.6${LD_PRELOAD:+:${LD_PRELOAD}}"
fi
export LD_LIBRARY_PATH="/usr/local/nvidia/lib64:${LD_LIBRARY_PATH:-}"
export __EGL_VENDOR_LIBRARY_DIRS="${HOME}/.local/share/glvnd/egl_vendor.d"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

load_convert_resolution_overrides

if ! $ALL_DEMOS; then
    run_one "${DEMO_NUM}" "${GPUS[0]}"
    exit 0
fi

mapfile -t DEMOS < <(list_demos)
echo "Batch: ${#DEMOS[@]} demo(s) with depth.npy under ${PROCESSED_ROOT}/${DEMO_NAME}"
echo "GPUs:  ${GPUS[*]}"
echo "Jobs:  ${NWORKERS} concurrent  (${PER_GPU}/GPU)"
echo "Demos: ${DEMOS[*]}"
echo ""

ok=()
skipped=()
failed=()
record_rc() {
    local demo="$1" rc="$2"
    case "${rc}" in
        0) ok+=("${demo}") ;;
        2) skipped+=("${demo}") ;;
        *) failed+=("${demo}") ;;
    esac
}

if $DRY_RUN || [[ "${NWORKERS}" -eq 1 ]]; then
    gpu_i=0
    for demo in "${DEMOS[@]}"; do
        gpu="${GPUS[$((gpu_i % ${#GPUS[@]}))]}"
        gpu_i=$((gpu_i + 1))
        if ! $DRY_RUN && should_skip "${demo}"; then
            echo "skip demo ${demo}: already has video_overlay.mkv"
            skipped+=("${demo}")
            continue
        fi
        set +e
        run_one "${demo}" "${gpu}"
        rc=$?
        set -e
        record_rc "${demo}" "${rc}"
    done
else
    declare -A PID_DEMO=()
    declare -A PID_GPU=()
    declare -A GPU_BUSY=()
    for g in "${GPUS[@]}"; do
        GPU_BUSY["${g}"]=0
    done

    next_free_gpu() {
        local g best="" best_n=""
        for g in "${GPUS[@]}"; do
            if (( GPU_BUSY[${g}] < PER_GPU )); then
                if [[ -z "${best}" ]] || (( GPU_BUSY[${g}] < best_n )); then
                    best="${g}"
                    best_n="${GPU_BUSY[${g}]}"
                fi
            fi
        done
        if [[ -n "${best}" ]]; then
            printf '%s\n' "${best}"
            return 0
        fi
        return 1
    }

    busy_count() {
        local g n=0
        for g in "${GPUS[@]}"; do
            n=$((n + GPU_BUSY[${g}]))
        done
        printf '%s\n' "${n}"
    }

    reap_finished() {
        local pid demo rc gpu
        local -a pids=("${!PID_DEMO[@]}")
        for pid in "${pids[@]}"; do
            if kill -0 "${pid}" 2>/dev/null; then
                continue
            fi
            demo="${PID_DEMO[${pid}]}"
            gpu="${PID_GPU[${pid}]}"
            set +e
            wait "${pid}"
            rc=$?
            set -e
            if (( GPU_BUSY[${gpu}] > 0 )); then
                GPU_BUSY["${gpu}"]=$((GPU_BUSY[${gpu}] - 1))
            fi
            record_rc "${demo}" "${rc}"
            if [[ "${rc}" -eq 0 ]]; then
                echo "✓ demo ${demo}  GPU ${gpu}" >&2
            elif [[ "${rc}" -eq 2 ]]; then
                echo "– skip demo ${demo}" >&2
            else
                echo "✗ demo ${demo}  GPU ${gpu}  rc=${rc}  log=${PROCESSED_ROOT}/${DEMO_NAME}/${demo}/retarget_run.log" >&2
            fi
            unset "PID_DEMO[${pid}]"
            unset "PID_GPU[${pid}]"
        done
    }

    wait_for_gpu() {
        WAIT_GPU=""
        local gpu=""
        while true; do
            reap_finished
            if (( $(busy_count) < NWORKERS )); then
                if gpu="$(next_free_gpu)"; then
                    WAIT_GPU="${gpu}"
                    return 0
                fi
            fi
            wait -n || true
        done
    }

    for demo in "${DEMOS[@]}"; do
        if should_skip "${demo}"; then
            echo "skip demo ${demo}: already has video_overlay.mkv"
            skipped+=("${demo}")
            continue
        fi
        wait_for_gpu
        gpu="${WAIT_GPU}"
        require_int_gpu "${gpu}"
        out_dir="${PROCESSED_ROOT}/${DEMO_NAME}/${demo}"
        mkdir -p "${out_dir}"
        log="${out_dir}/retarget_run.log"
        echo "▸ start demo ${demo} on GPU ${gpu}  log=${log}"
        (
            run_one "${demo}" "${gpu}"
        ) >"${log}" 2>&1 &
        PID_DEMO[$!]="${demo}"
        PID_GPU[$!]="${gpu}"
        GPU_BUSY["${gpu}"]=$((GPU_BUSY[${gpu}] + 1))
    done

    while [[ ${#PID_DEMO[@]} -gt 0 ]]; do
        wait -n || true
        reap_finished
    done
fi

echo ""
echo "═══ BATCH SUMMARY ═══"
echo "  ok:      ${#ok[@]}  ${ok[*]:-}"
echo "  skipped: ${#skipped[@]}  ${skipped[*]:-}"
echo "  failed:  ${#failed[@]}  ${failed[*]:-}"
if [[ ${#failed[@]} -gt 0 ]]; then
    exit 1
fi
exit 0
