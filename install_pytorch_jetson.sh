#!/bin/bash
# Install CUDA-enabled PyTorch for Jetson
# This script installs the correct PyTorch wheel for Jetson devices

set -e

echo "======================================"
echo "PyTorch Installation for Jetson"
echo "======================================"
echo ""

# Check if running on Jetson
if [ ! -f /etc/nv_tegra_release ]; then
    echo "ERROR: This script is for Jetson devices only!"
    echo "For x86_64 systems, install PyTorch normally:"
    echo "  pip install torch torchvision"
    exit 1
fi

# Get JetPack version
JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep -oP '(?<=R)[0-9]+' | head -1)
echo "Detected JetPack: R${JETPACK_VERSION}.x"

# Get Python version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}{sys.version_info.minor}")')
echo "Python version: 3.${PYTHON_VERSION#3}"

# Uninstall existing PyTorch
echo ""
echo "Uninstalling any existing PyTorch installations..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true

# Install based on JetPack version
echo ""
echo "Installing NVIDIA PyTorch wheel for Jetson..."

if [ "$JETPACK_VERSION" = "35" ]; then
    # JetPack 5.x (L4T R35.x)
    echo "Installing for JetPack 5.x (R35.x)..."
    
    if [ "$PYTHON_VERSION" = "38" ]; then
        # Python 3.8 - PyTorch 2.0.0 for JetPack 5.x
        # Download from NVIDIA's PyTorch for Jetson repository
        TORCH_URL="https://nvidia.box.com/shared/static/i8pukc49h3lhak4kkn67tg9j4goqm0m7.whl"
        TORCH_WHL="torch-2.0.0a0+8aa34602.nv23.03-cp38-cp38-linux_aarch64.whl"
        TORCHVISION_VER="0.15.1"
    elif [ "$PYTHON_VERSION" = "310" ]; then
        # Python 3.10
        TORCH_URL="https://nvidia.box.com/shared/static/0h6tk4msrl9xz3evft9t0mpwwwkw7a32.whl"
        TORCH_WHL="torch-2.0.0a0+8aa34602.nv23.03-cp310-cp310-linux_aarch64.whl"
        TORCHVISION_VER="0.15.1"
    else
        echo "ERROR: Unsupported Python version for JetPack 5.x"
        exit 1
    fi
    
elif [ "$JETPACK_VERSION" = "36" ]; then
    # JetPack 6.x (L4T R36.x)
    echo "Installing for JetPack 6.x (R36.x)..."
    
    if [ "$PYTHON_VERSION" = "310" ]; then
        TORCH_URL="https://nvidia.box.com/shared/static/mp164asf3sceb570wvjsrezk1p4ftj8t.whl"
        TORCH_WHL="torch-2.3.0a0+40ec155e.nv24.3-cp310-cp310-linux_aarch64.whl"
        TORCHVISION_VER="0.18.0a0"
    else
        echo "ERROR: JetPack 6.x requires Python 3.10"
        exit 1
    fi
else
    echo "ERROR: Unsupported JetPack version: R${JETPACK_VERSION}"
    echo "Please install PyTorch manually from:"
    echo "  https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
    exit 1
fi

echo "Downloading and installing PyTorch..."
echo "Downloading from: $TORCH_URL"
wget -O /tmp/$TORCH_WHL "$TORCH_URL"
pip install --no-cache-dir /tmp/$TORCH_WHL
rm /tmp/$TORCH_WHL

echo ""
echo "Installing torchvision ${TORCHVISION_VER}..."
pip install --no-deps torchvision==$TORCHVISION_VER

# Verify installation
echo ""
echo "======================================"
echo "Verifying installation..."
echo "======================================"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    try:
        print(f'cuDNN version: {torch.backends.cudnn.version()}')
    except RuntimeError as e:
        print(f'cuDNN note: {str(e)[:100]}...')
        print('(This is usually fine - PyTorch bundles its own cuDNN)')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print('')
    print('✓ PyTorch with CUDA support installed successfully!')
else:
    print('✗ WARNING: CUDA not available!')
    exit(1)
"

echo ""
echo "======================================"
echo "Installation complete!"
echo ""
echo "IMPORTANT: If you see cuDNN version warnings,"
echo "you can safely ignore them. PyTorch bundles its own cuDNN."
echo "======================================"
