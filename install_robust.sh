#!/usr/bin/env bash
# Alternative Phantom installer with pins/workarounds for common failures.
# Does NOT replace ./install.sh — use this if the original script breaks.
#
# Run from the repository root:
#   ./install_robust.sh
#
# Improvements over the original:
# - Pin torch 2.1.0 (cu121) before packages that would otherwise upgrade it
# - Install detectron2 / chumpy with --no-build-isolation (needs torch + setuptools)
# - Keep setuptools<81 so pkg_resources (mmcv / torch cpp_extension) works
# - Re-pin numpy==1.26.4 after any step that may pull numpy 2.x
# - Disable ~/.local site-packages in this conda env (PYTHONNOUSERSITE)
# - Download E2FGVI weights by file id (no gdown --fuzzy)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

ENV_NAME="${PHANTOM_ENV_NAME:-phantom}"
PYTHON_VERSION="${PHANTOM_PYTHON_VERSION:-3.10}"

pin_core_versions() {
  # Critical pins for this project (mmcv / ViTPose / transformers).
  pip install --upgrade \
    "setuptools>=65,<81" \
    "numpy==1.26.4" \
    "protobuf==3.20.0" \
    "PyOpenGL==3.1.4"
}

echo "=== [1/10] Create conda env: ${ENV_NAME} (python=${PYTHON_VERSION}) ==="
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "Conda env '${ENV_NAME}' already exists; reusing it."
else
  conda create -n "${ENV_NAME}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"
export PYTHONNOUSERSITE=1

# Persist PYTHONNOUSERSITE for future activations of this env.
mkdir -p "${CONDA_PREFIX}/etc/conda/activate.d" "${CONDA_PREFIX}/etc/conda/deactivate.d"
cat > "${CONDA_PREFIX}/etc/conda/activate.d/disable_usersite.sh" <<'EOF'
# Ignore ~/.local packages that can override conda pins (e.g. numpy 2.x).
export PYTHONNOUSERSITE=1
EOF
cat > "${CONDA_PREFIX}/etc/conda/deactivate.d/disable_usersite.sh" <<'EOF'
unset PYTHONNOUSERSITE
EOF

echo "=== [2/10] CUDA toolkit 12.1 (conda) ==="
conda install -y nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0

echo "=== [3/10] PyTorch 2.1.0 + cu121 (pin before SAM2 / HaMeR) ==="
pip install --upgrade pip
pin_core_versions
pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.1.0 torchvision==0.16.0
pin_core_versions

echo "=== [4/10] SAM 2 (editable, no-deps to keep torch pin) ==="
cd "${ROOT_DIR}/submodules/sam2"
# Official SAM2 wants torch>=2.5.1; Phantom intentionally stays on 2.1.0 for mmcv.
pip install -v -e ".[notebooks]" --no-deps
pip install matplotlib jupyter opencv-python "eva-decord>=0.6.1" tqdm hydra-core iopath pillow
pin_core_versions
cd "${ROOT_DIR}"

echo "=== [5/10] Detectron2 + HaMeR + ViTPose ==="
# detectron2 must see torch during metadata/build; disable build isolation.
pip install --no-build-isolation "git+https://github.com/facebookresearch/detectron2"
pip install --no-build-isolation "git+https://github.com/mattloper/chumpy"

cd "${ROOT_DIR}/submodules/phantom-hamer"
pip install -e ".[all]" --no-build-isolation
pip install -v -e third-party/ViTPose --no-build-isolation
# mmpose extras that sometimes get skipped
pip install json_tricks munkres

if [[ ! -d _DATA ]]; then
  echo "Downloading HaMeR demo data..."
  wget -c https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
  tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
fi
pin_core_versions
cd "${ROOT_DIR}"

echo "=== [6/10] mmcv / mmcv-full (torch 2.1 + cu121) ==="
# Install lite mmcv first (HaMeR pins 1.3.9), then full ops.
# Official prebuilt mmcv-full wheels lack sm_90 (H100/H200); build from source.
pip install "mmcv==1.3.9"
export CUDA_HOME="${CONDA_PREFIX}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"
export MMCV_WITH_OPS=1
export FORCE_CUDA=1
pip install --no-cache-dir --no-build-isolation "mmcv-full==1.7.2"
pin_core_versions

echo "=== [7/10] robosuite + robomimic ==="
# Phantom camera XML needs sensorsize/focalpixel (MuJoCo >=3.0).
# base_controller.py patches mj_fullM for both old and new MuJoCo 3.x APIs.
pip install "mujoco>=3.1.4,<3.11"
cd "${ROOT_DIR}/submodules/phantom-robosuite"
pip install -e .
cd "${ROOT_DIR}/submodules/phantom-robomimic"
pip install -e .
cd "${ROOT_DIR}"
pin_core_versions
# Re-assert mujoco pin in case a dependency pulled an incompatible version.
pip install "mujoco>=3.1.4,<3.11"

echo "=== [8/10] Extra Python packages ==="
pip install joblib mediapy open3d pandas
pip install "transformers==4.42.4"
pip install Rtree
pip install "git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git"
pip install "hydra-core==1.3.2" "omegaconf==2.3.0"
pip install cloudpickle tabulate yacs pycocotools termcolor fvcore
pin_core_versions

echo "=== [9/10] E2FGVI weights + package ==="
mkdir -p "${ROOT_DIR}/submodules/phantom-E2FGVI/E2FGVI/release_model"
cd "${ROOT_DIR}/submodules/phantom-E2FGVI/E2FGVI/release_model"
pip install gdown
if [[ ! -f E2FGVI-CVPR22.pth ]]; then
  # Use file id (works across gdown versions; --fuzzy is not always available).
  gdown 10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3 -O E2FGVI-CVPR22.pth || \
    gdown "https://drive.google.com/uc?id=10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3" -O E2FGVI-CVPR22.pth
fi
cd "${ROOT_DIR}/submodules/phantom-E2FGVI"
pip install -e .
cd "${ROOT_DIR}"

echo "=== [10/10] Install phantom + sample data ==="
pip install -e .
pin_core_versions

mkdir -p "${ROOT_DIR}/data/raw"
cd "${ROOT_DIR}/data/raw"
if [[ ! -d pick_and_place ]]; then
  wget -c https://download.cs.stanford.edu/juno/phantom/pick_and_place.zip
  unzip -o pick_and_place.zip
  rm -f pick_and_place.zip
fi
if [[ ! -d epic ]]; then
  wget -c https://download.cs.stanford.edu/juno/phantom/epic.zip
  unzip -o epic.zip
  rm -f epic.zip
fi
cd "${ROOT_DIR}"

echo
echo "================================================================"
echo "Install finished for conda env: ${ENV_NAME}"
echo
echo "Next steps:"
echo "  1) conda activate ${ENV_NAME}"
echo "     (PYTHONNOUSERSITE=1 is set automatically for this env)"
echo "  2) Download MANO_LEFT.pkl / MANO_RIGHT.pkl from https://mano.is.tue.mpg.de/"
echo "     and place them in:"
echo "     ${ROOT_DIR}/submodules/phantom-hamer/_DATA/data/mano/"
echo "  3) cd phantom && python process_data.py demo_name=pick_and_place \\"
echo "       data_root_dir=../data/raw processed_data_root_dir=../data/processed mode=all"
echo "================================================================"
