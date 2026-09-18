#!/bin/bash

# Packaging helper. FlashComm needs NCCL's device API (nccl_device.h).
# Install the public PyPI wheel nvidia-nccl-cu{12,13}==NCCL_VERSION (default
# 2.30.7), or point NCCL_HOME / CUSTOM_NCCL_HOME at a local tree.

# set cuda env for scm
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export PATH="${CUDA_HOME}/bin:$PATH"

apt-get install -y --no-install-recommends \
        zip=3.0-13 \
        unzip=6.0-28

# Make -lcuda resolvable (stub libcuda.so) for both x86_64 and aarch64.
ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  # CUDA on ARM servers often uses targets/sbsa-linux
  if [ -d "${CUDA_HOME}/targets/sbsa-linux" ]; then
    CUDA_TARGET="sbsa-linux"
  else
    CUDA_TARGET="aarch64-linux"
  fi
else
  CUDA_TARGET="x86_64-linux"
fi
export LIBRARY_PATH="${CUDA_HOME}/lib64/:${CUDA_HOME}/targets/${CUDA_TARGET}/lib/stubs/:${LIBRARY_PATH}"

NCCL_VERSION="${NCCL_VERSION:-${CUSTOM_NCCL_VERSION:-2.30.7}}"
NCCL_HOME="${NCCL_HOME:-${CUSTOM_NCCL_HOME:-}}"
CUDA_MAJOR="$("${CUDA_HOME}/bin/nvcc" --version | sed -n 's/.*release \([0-9][0-9]*\)\..*/\1/p' | head -n 1)"
if [ -z "${CUDA_MAJOR}" ]; then
  echo "Failed to parse CUDA major version from ${CUDA_HOME}/bin/nvcc --version" >&2
  exit 1
fi
case "${CUDA_MAJOR}" in
  12|13) ;;
  *)
    echo "Unsupported CUDA major version for NCCL pip package: ${CUDA_MAJOR}" >&2
    exit 1
    ;;
esac
NCCL_PIP_PACKAGE="nvidia-nccl-cu${CUDA_MAJOR}"

if [ -n "${NCCL_HOME}" ]; then
  if [ ! -f "${NCCL_HOME}/include/nccl_device.h" ] || [ ! -f "${NCCL_HOME}/include/nccl.h" ]; then
    echo "NCCL_HOME=${NCCL_HOME} is missing NCCL headers" >&2
    exit 1
  fi
  if [ ! -f "${NCCL_HOME}/lib/libnccl.so" ] && [ ! -f "${NCCL_HOME}/lib/libnccl.so.2" ]; then
    echo "NCCL_HOME=${NCCL_HOME} is missing lib/libnccl.so*" >&2
    exit 1
  fi
  echo "Using NCCL from NCCL_HOME=${NCCL_HOME}"
  export LIBRARY_PATH="${NCCL_HOME}/lib:${LIBRARY_PATH}"
  export LD_LIBRARY_PATH="${NCCL_HOME}/lib:${LD_LIBRARY_PATH:-}"
  export CUSTOM_NCCL_HOME="${NCCL_HOME}"
else
  echo "Using NCCL pip package: ${NCCL_PIP_PACKAGE}==${NCCL_VERSION}"
fi

SCRIPT_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "SCRIPT_DIR: $SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "PROJECT_ROOT: $PROJECT_ROOT"

cd $PROJECT_ROOT
pip3 install setuptools==69.0.0
pip3 install ninja cmake wheel pybind11
if [ -z "${NCCL_HOME}" ]; then
  pip3 install "${NCCL_PIP_PACKAGE}==${NCCL_VERSION}"
fi

python3 setup.py bdist_wheel

mkdir -p ../output/python
unzip dist/*.whl -d ../output/python

if [ -n "${FLASH_COMM_CUTEDSL_AOT_MANIFEST:-}" ]; then
  : "${FLASH_COMM_CUTEDSL_AOT_DIR:=$PROJECT_ROOT/dist/cutedsl_aot}"
  : "${FLASH_COMM_CUTEDSL_AOT_NPROC:=1}"
  : "${FLASH_COMM_CUTEDSL_AOT_TIMEOUT:=1800s}"

  mkdir -p "$FLASH_COMM_CUTEDSL_AOT_DIR"
  PYTHONPATH="$PROJECT_ROOT/python:${PYTHONPATH:-}" \
    timeout "$FLASH_COMM_CUTEDSL_AOT_TIMEOUT" \
    torchrun --standalone --nproc_per_node "$FLASH_COMM_CUTEDSL_AOT_NPROC" \
    -m flash_comm.tools.cutedsl_aot prebuild-ep-overlap \
    --manifest "$FLASH_COMM_CUTEDSL_AOT_MANIFEST" \
    --aot-dir "$FLASH_COMM_CUTEDSL_AOT_DIR"

  mkdir -p ../output/cutedsl_aot
  cp -a "$FLASH_COMM_CUTEDSL_AOT_DIR"/. ../output/cutedsl_aot/
fi
