eval "$(conda shell.bash hook)"
# ######################## Phantom Env ###############################
conda create -n phantom python=3.10 -y
conda activate phantom
conda install nvidia/label/cuda-12.1.0::cuda-toolkit -c nvidia/label/cuda-12.1.0 -y

# Pin packaging stack: setuptools>=82 drops pkg_resources (breaks torch/mmcv),
# and old packages like chumpy/mmcv fail under PEP517 build isolation.
pip install 'setuptools<81' wheel pip

# Install PyTorch before packages that need it at build/import time
pip install --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0

# Pin torch/torchvision for the rest of the install: some submodules (e.g. SAM2)
# declare loose lower bounds like "torch>=2.5.1" and would otherwise cause pip to
# silently upgrade torch to whatever the latest PyPI release is (pulling in a
# non-cu121 build). PIP_CONSTRAINT makes pip fail loudly instead if that happens.
PIP_CONSTRAINT_FILE="$(pwd)/pip-constraints.txt"
printf 'torch==2.1.0\ntorchvision==0.16.0\n' > "$PIP_CONSTRAINT_FILE"
export PIP_CONSTRAINT="$PIP_CONSTRAINT_FILE"

# Install SAM2
cd submodules/sam2
pip install -v -e ".[notebooks]"
cd ../..

# Preinstall legacy deps that break under build isolation
pip install --no-build-isolation 'git+https://github.com/mattloper/chumpy.git'
# Prefer prebuilt mmcv-full for cu121/torch2.1 (mmcv==1.3.9 sdist fails on modern pip)
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
pip install numpy==1.26.4

# Install Hamer (mmcv/chumpy already installed; avoid pulling mmcv==1.3.9 sdist)
cd submodules/phantom-hamer
pip install --no-build-isolation --no-deps -e .
pip install gdown opencv-python pyrender pytorch-lightning scikit-image \
  'smplx==0.1.28' yacs timm einops xtcocotools pandas \
  hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils rich webdataset
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2'
pip install -v --no-build-isolation --no-deps -e third-party/ViTPose
pip install json_tricks matplotlib munkres
if [[ -d _DATA ]]; then
  echo "HaMeR demo data already present (_DATA), skipping download."
elif [[ -f hamer_demo_data.tar.gz ]]; then
  echo "Found hamer_demo_data.tar.gz, extracting..."
  tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
else
  wget -c https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
  tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
fi
cd ../..

# Install phantom-robosuite
cd submodules/phantom-robosuite
pip install -e .
cd ../..

# Install phantom-robomimic
cd submodules/phantom-robomimic
pip install -e .
cd ../..

# Install additional packages
pip install joblib mediapy open3d pandas
pip install transformers==4.42.4
pip install PyOpenGL==3.1.4
pip install Rtree
pip install git+https://github.com/epic-kitchens/epic-kitchens-100-hand-object-bboxes.git
pip install protobuf==3.20.0
pip install hydra-core==1.3.2
pip install omegaconf==2.3.0

# Download E2FGVI weights
cd submodules/phantom-E2FGVI/E2FGVI/release_model/
pip install gdown
if [[ -f E2FGVI-CVPR22.pth ]]; then
  echo "E2FGVI weights already present, skipping download."
else
  gdown --fuzzy https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing
fi
cd ../..

# Install phantom-E2FGVI
pip install -e .
cd ../..

# Install phantom 
pip install -e .

# Download sample data
cd data/raw
if [[ -d pick_and_place ]]; then
  echo "pick_and_place already present, skipping download."
elif [[ -f pick_and_place.zip ]]; then
  echo "Found pick_and_place.zip, extracting..."
  unzip -o pick_and_place.zip
  rm -f pick_and_place.zip
else
  wget -c https://download.cs.stanford.edu/juno/phantom/pick_and_place.zip
  unzip -o pick_and_place.zip
  rm -f pick_and_place.zip
fi
if [[ -d epic ]]; then
  echo "epic already present, skipping download."
elif [[ -f epic.zip ]]; then
  echo "Found epic.zip, extracting..."
  unzip -o epic.zip
  rm -f epic.zip
else
  wget -c https://download.cs.stanford.edu/juno/phantom/epic.zip
  unzip -o epic.zip
  rm -f epic.zip
fi
cd ../..
