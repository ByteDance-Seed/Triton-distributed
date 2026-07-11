#!/bin/bash

# Control which packages to build (both enabled by default)
# Set BUILD_TRITON_DIST=0 to skip triton_dist, BUILD_FLASHCOMM=0 to skip FlashComm
BUILD_TRITON_DIST="${BUILD_TRITON_DIST:-1}"
BUILD_FLASHCOMM="${BUILD_FLASHCOMM:-1}"

if [ "$BUILD_TRITON_DIST" -eq 0 ] && [ "$BUILD_FLASHCOMM" -eq 0 ]; then
    echo "ERROR: Both BUILD_TRITON_DIST and BUILD_FLASHCOMM are disabled. Nothing to build."
    exit 1
fi

echo "BUILD_TRITON_DIST=$BUILD_TRITON_DIST, BUILD_FLASHCOMM=$BUILD_FLASHCOMM"

# set cuda env for scm
export PATH=/usr/local/cuda/bin:$PATH

ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  if [ -d /usr/local/cuda/targets/sbsa-linux ]; then
    CUDA_TARGET="sbsa-linux"
  else
    CUDA_TARGET="aarch64-linux"
  fi
else
  CUDA_TARGET="x86_64-linux"
fi
export LIBRARY_PATH="/usr/local/cuda/lib64/:/usr/local/cuda/targets/${CUDA_TARGET}/lib/stubs/:${LIBRARY_PATH}"

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTORCH_24="2.4.0"
PYTORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
CLEAN_VERSION=$(echo "$PYTORCH_VERSION" | cut -d'+' -f1)

git submodule init
git submodule update --recursive
pip install packaging
NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
echo $NVCC_VERSION
NVCC_MAJOR_VERSION=$(echo $NVCC_VERSION | cut -d'.' -f1)
NVCC_MINOR_VERSION=$(echo $NVCC_VERSION | cut -d'.' -f2)

if [ "$NVCC_MAJOR_VERSION" -ge 12 ] && [ "$NVCC_MINOR_VERSION" -ge 4 ]; then
    ARCHS="80;89;90"
    SM_CORES="108;92;78;132"
elif [ "$NVCC_MAJOR_VERSION" -ge 12 ]; then
    ARCHS="80;90"
    SM_CORES="108;78;132"
else
    ARCHS="80"
    SM_CORES="108"
fi
echo $ARCHS
echo $SM_CORES

# adapt to scm envs that must start with CUSTOM_
# allow CUSTOM_JOBS to replace JOBS

export $(env | grep '^CUSTOM_JOBS' | sed 's/^CUSTOM_JOBS/JOBS/g')
pip3 install cuda-python==12.4 
pip3 install setuptools==69.0.0 
curl -IivvL https://oaitriton.blob.core.windows.net/public/llvm-builds/
pip3 install ninja cmake wheel pybind11

mkdir -p output/python

# ============ Build triton_dist ============
if [ "$BUILD_TRITON_DIST" -eq 1 ]; then
    echo "========== Building triton_dist =========="
    cd $SCRIPT_DIR
    pip3 uninstall triton -y
    export USE_TRITON_DISTRIBUTED_AOT=0
    echo 'numpy<2' > /tmp/pip_install_constraint.txt
    MAX_JOBS=40 pip3 install -c /tmp/pip_install_constraint.txt -e python[build,tests,tutorials] --verbose --no-build-isolation --use-pep517
    # bash ./scripts/gen_aot_code.sh
    # export USE_TRITON_DISTRIBUTED_AOT=1
    # MAX_JOBS=40 pip3 install -e python --verbose --no-build-isolation --use-pep517
    cd python
    python3 setup.py bdist_wheel
    cd $SCRIPT_DIR
    unzip python/dist/*.whl -d output/python
    echo "========== triton_dist build done =========="
else
    echo "========== Skipping triton_dist (BUILD_TRITON_DIST=0) =========="
fi

# ============ Build FlashComm ============
if [ "$BUILD_FLASHCOMM" -eq 1 ]; then
    echo "========== Building FlashComm =========="
    cd $SCRIPT_DIR/FlashComm
    python3 setup.py bdist_wheel
    cd $SCRIPT_DIR
    unzip FlashComm/dist/*.whl -d output/python
    echo "========== FlashComm build done =========="
else
    echo "========== Skipping FlashComm (BUILD_FLASHCOMM=0) =========="
fi

echo "All done. Output packages are in output/python/"
