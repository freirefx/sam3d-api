#!/bin/bash

# 1. Check if the script is being sourced
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    echo "#######################################################################"
    echo "WARNING: You are running this script as a subshell (./setup.sh)."
    echo "To remain inside the conda environment after the script finishes,"
    echo "please run: source setup.sh"
    echo "#######################################################################"
    sleep 2
fi

# Exit on error
set -e

echo "--- 1a. Handling SAM-3D-Objects Repository ---"
REPO_DIR="sam-3d-objects"
if [ -d "$REPO_DIR" ]; then
    echo "Directory '$REPO_DIR' exists. Updating..."
    cd "$REPO_DIR"
    git pull
    cd ..
else
    git clone https://github.com/facebookresearch/sam-3d-objects.git
fi

echo "--- 1b. Handling SAM3 Repository ---"
SAM3_REPO_DIR="sam3"
if [ -d "$SAM3_REPO_DIR" ]; then
    echo "Directory '$SAM3_REPO_DIR' exists. Updating..."
    cd "$SAM3_REPO_DIR"
    git pull
    cd ..
else
    echo "Cloning SAM3 repository..."
    git clone https://github.com/facebookresearch/sam3.git || echo "⚠ SAM3 repository not available yet. Skipping..."
fi

# Enter sam-3d-objects for the rest of the setup
cd "$REPO_DIR"

echo "--- 2. Checking Miniconda ---"
CONDA_ROOT="$HOME/miniconda3"
if [ -d "$CONDA_ROOT" ]; then
    echo "Miniconda already installed at $CONDA_ROOT."
else
    echo "Installing Miniconda..."
    curl -fsSL -o Miniconda3.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3.sh -b -p "$CONDA_ROOT"
    rm Miniconda3.sh
fi

# Initialize conda for the current shell session
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda init bash

echo "--- 3. Setting up Conda Environment ---"
ENV_NAME="sam3d-objects"
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' exists. Updating..."
    conda env update -f environments/default.yml --prune
else
    conda env create -f environments/default.yml
fi

conda activate "$ENV_NAME"

echo "--- 4. Installing PyTorch & Dependencies ---"
export PIP_EXTRA_INDEX_URL="https://pypi.ngc.nvidia.com https://download.pytorch.org/whl/cu121"
pip install -e '.[dev]'
pip install -e '.[p3d]'

echo "--- 5. Installing Inference & Patching ---"
export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
pip install -e '.[inference]'

if [ -f "./patching/hydra" ]; then
    chmod +x ./patching/hydra
    ./patching/hydra
fi

echo "--- 6. Downloading Model Checkpoints ---"
pip install 'huggingface-hub[cli]<1.0'

TAG=hf
if [ ! -d "checkpoints/${TAG}" ]; then
    mkdir -p checkpoints
    huggingface-cli download \
      --repo-type model \
      --local-dir checkpoints/${TAG}-download \
      --max-workers 1 \
      facebook/sam-3d-objects

    mv checkpoints/${TAG}-download/checkpoints checkpoints/${TAG}
    rm -rf checkpoints/${TAG}-download
else
    echo "Checkpoints already present. Skipping download."
fi

echo "--- 7. Final Requirements (Root) ---"
cd ..
if [ -f "requirements.txt" ]; then
    echo "Installing requirements.txt from $(pwd)..."

    pip install -r requirements.txt
    pip install git+https://github.com/NVlabs/nvdiffrast.git --no-build-isolation
else
    echo "No requirements.txt found in $(pwd)."
fi

echo "--- 8. Installing SAM3 (if available) ---"
if [ -d "sam3" ]; then
    echo "Installing SAM3 from local directory..."
    cd sam3
    pip install -e . || echo "⚠ SAM3 installation failed. Routes will use SAM2 fallback."
    cd ..
else
    echo "⚠ SAM3 directory not found. SAM3D routes will use SAM2 fallback."
fi

echo "--- 8b. Downloading SAM3 Checkpoints (Image + Video) ---"
if [ -d "sam3" ]; then
    mkdir -p sam3/checkpoints
    
    # Download image checkpoint if not present
    if [ ! -f "sam3/checkpoints/sam3_hiera_large.pt" ] && [ ! -f "sam3/checkpoints/sam3.pt" ]; then
        echo "Downloading SAM3 image checkpoint..."
        huggingface-cli download facebook/sam3 sam3_hiera_large.pt --local-dir sam3/checkpoints/ || \
        huggingface-cli download facebook/sam3 --local-dir sam3/checkpoints/ || \
        echo "⚠ Could not download SAM3 checkpoint. Please download manually or request access at https://huggingface.co/facebook/sam3"
    else
        echo "SAM3 image checkpoint already present."
    fi
    
    # Download video checkpoint if not present
    if [ ! -f "sam3/checkpoints/sam3_video.pt" ]; then
        echo "Downloading SAM3 video checkpoint..."
        huggingface-cli download facebook/sam3 sam3_video.pt --local-dir sam3/checkpoints/ || \
        echo "⚠ Could not download SAM3 video checkpoint. Video features may not be available."
    else
        echo "SAM3 video checkpoint already present."
    fi
else
    echo "⚠ SAM3 directory not found. Skipping checkpoint download."
fi

echo "--- 9. Installing Video Processing Dependencies ---"
pip install decord av imageio imageio-ffmpeg || echo "⚠ Some video dependencies failed to install."

echo "--- 10. Validating SAM3 Installation ---"
python -c "
try:
    import sam3
    print('✓ SAM3 module imported successfully')
    if hasattr(sam3, 'build_sam3_image_model'):
        print('  - build_sam3_image_model: available')
    if hasattr(sam3, 'build_sam3_video_predictor'):
        print('  - build_sam3_video_predictor: available')
except ImportError as e:
    print(f'⚠ SAM3 import failed: {e}')
" || echo "⚠ SAM3 validation skipped (not installed)"

echo "--- Setup Complete! ---"
echo ""
echo "Available endpoints:"
echo ""
echo "  SAM 2 (Image):"
echo "    - /segment          : Single point segmentation"
echo "    - /segment-binary   : Multiple points, masked image"
echo ""
echo "  SAM 3 (Image):"
echo "    - /segment-sam3d        : Single point segmentation"
echo "    - /segment-binary-sam3d : Multiple points, masked image"
echo "    - /segment-text-sam3d   : Text prompt segmentation"
echo "    - /segment-box-sam3d    : Bounding box segmentation"
echo "    - /segment-mask-sam3d   : Mask refinement"
echo "    - /segment-auto-sam3d   : Automatic mask generation"
echo ""
echo "  SAM 3 (Video):"
echo "    - /video/start-session  : Start video session"
echo "    - /video/add-prompt     : Add prompt to frame"
echo "    - /video/propagate      : Propagate masks"
echo "    - /video/get-frame      : Get frame with masks"
echo "    - /video/remove-object  : Remove tracked object"
echo "    - /video/end-session    : End session"
echo ""
echo "  3D Generation:"
echo "    - /generate-3d      : 3D generation from image + mask"
echo ""
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    echo "SUCCESS: You are now active in the '$ENV_NAME' environment."
else
    echo "To activate the environment now, run: conda activate $ENV_NAME"
fi