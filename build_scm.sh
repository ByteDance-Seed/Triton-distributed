#!/bin/bash
# Release / SCM packaging entry point (independently maintained; NOT used for a
# normal dev install -- that is just `pip install -e python`).
#
# Assembles redistributable wheels under output/python so a production env can
# `PYTHONPATH=output/python` import everything:
#   1. patched Triton  -> build_triton.sh (wheel mode), unzipped into output/python
#   2. triton_dist     -> built against that Triton (skip in-setup rebuild), pinned
#                         to its exact patched version, wheel unzipped into output/python
#   3. FlashComm       -> wheel unzipped into output/python
# Toggle parts via BUILD_TRITON_DIST / BUILD_FLASHCOMM (both default on).

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
    echo 'numpy<2' > /tmp/pip_install_constraint.txt

    # --- 1) Patched (EXT-enabled) Triton: build its wheel, extract it into
    # output/python, install it. triton_dist is an out-of-tree plugin that can
    # ONLY load into this patched Triton (default-visibility + source patches +
    # libstdc++ version-script); a stock PyPI triton cannot host it. Extracting it
    # next to triton_dist lets production use both via PYTHONPATH=output/python
    # (the original triton_dist convention), and pip consumers get bound to the
    # exact version below.
    echo "========== Building patched Triton (wheel) =========="
    pip3 uninstall triton -y || true
    TRITON_DIST_TRITON_INSTALL_MODE=wheel MAX_JOBS=40 bash scripts/build_triton.sh
    TRITON_WHL=$(ls -t "$SCRIPT_DIR"/3rdparty/triton/dist/triton-*.whl | head -1)
    echo "patched Triton wheel: $TRITON_WHL"
    pip3 install --no-cache-dir "$TRITON_WHL"
    TRITON_VER=$(python3 -c "import triton; print(triton.__version__)")
    echo "patched Triton version (triton_dist will pin to this): $TRITON_VER"
    # Extract into output/python (same convention as triton_dist below) so the
    # production env can import both via PYTHONPATH=output/python directly.
    unzip -o "$TRITON_WHL" -d output/python

    # --- 2) triton_dist: compile the plugin against the SAME source tree we just
    # built the wheel from (the installed wheel ships no C++ headers), skip the
    # in-setup Triton rebuild, and bind the dependency to the exact patched build.
    export USE_TRITON_DISTRIBUTED_AOT=0
    export TRITON_DIST_SKIP_TRITON_BUILD=1
    export TRITON_SOURCE_DIR="$SCRIPT_DIR/3rdparty/triton"
    export TRITON_DIST_TRITON_REQUIREMENT="triton==${TRITON_VER}"
    MAX_JOBS=40 pip3 install -c /tmp/pip_install_constraint.txt -e python[build,tests,tutorials] --verbose --no-build-isolation --use-pep517
    # Release wheels are intentionally JIT-only (no AOT csrc): AOT is optional and
    # the same kernels JIT-compile on first use. To ship an AOT-enabled release,
    # uncomment the three lines below (generate kernels, then rebuild with AOT on).
    # bash ./scripts/gen_aot_code.sh
    # export USE_TRITON_DISTRIBUTED_AOT=1
    # MAX_JOBS=40 pip3 install -e python --verbose --no-build-isolation --use-pep517
    cd python
    python3 setup.py bdist_wheel
    cd $SCRIPT_DIR
    unzip python/dist/*.whl -d output/python
    echo "========== triton_dist build done (bound to $TRITON_VER) =========="
else
    echo "========== Skipping triton_dist (BUILD_TRITON_DIST=0) =========="
fi

# ============ Build FlashComm ============
if [ "$BUILD_FLASHCOMM" -eq 1 ]; then
    echo "========== Building FlashComm =========="
    # FlashComm needs NCCL device API headers (nccl_device.h). Use the public
    # PyPI wheel unless NCCL_HOME / CUSTOM_NCCL_HOME already points at a tree.
    NCCL_VERSION="${NCCL_VERSION:-${CUSTOM_NCCL_VERSION:-2.30.7}}"
    NCCL_HOME="${NCCL_HOME:-${CUSTOM_NCCL_HOME:-}}"
    if [ -z "${NCCL_HOME}" ]; then
        NCCL_PIP_PACKAGE="nvidia-nccl-cu${NVCC_MAJOR_VERSION}"
        echo "Using NCCL pip package: ${NCCL_PIP_PACKAGE}==${NCCL_VERSION}"
        pip3 install "${NCCL_PIP_PACKAGE}==${NCCL_VERSION}"
    else
        echo "Using NCCL from NCCL_HOME=${NCCL_HOME}"
        export LIBRARY_PATH="${NCCL_HOME}/lib:${LIBRARY_PATH}"
        export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
        export CUSTOM_NCCL_HOME="${NCCL_HOME}"
    fi
    cd $SCRIPT_DIR/FlashComm
    python3 setup.py bdist_wheel
    cd $SCRIPT_DIR
    unzip FlashComm/dist/*.whl -d output/python
    echo "========== FlashComm build done =========="
else
    echo "========== Skipping FlashComm (BUILD_FLASHCOMM=0) =========="
fi

echo "All done. Output packages are in output/python/"
