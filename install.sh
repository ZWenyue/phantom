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
wget https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz
tar --warning=no-unknown-keyword --exclude=".*" -xvf hamer_demo_data.tar.gz
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
gdown --fuzzy https://drive.google.com/file/d/10wGdKSUOie0XmCr8SQ2A2FeDe-mfn5w3/view?usp=sharing
cd ../..

# Install phantom-E2FGVI
pip install -e .
cd ../..

# Install phantom 
pip install -e .

# Download sample data
cd data/raw
wget https://download.cs.stanford.edu/juno/phantom/pick_and_place.zip
unzip pick_and_place.zip
rm pick_and_place.zip
wget https://download.cs.stanford.edu/juno/phantom/epic.zip
unzip epic.zip
rm epic.zip
cd ../..
