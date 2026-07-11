#!/bin/bash

# set cuda env for scm
export PATH=/usr/local/cuda/bin:$PATH

apt-get install -y --no-install-recommends \
        zip=3.0-13 \
        unzip=6.0-28

# Make -lcuda resolvable (stub libcuda.so) for both x86_64 and aarch64.
ARCH="$(uname -m)"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
  # CUDA on ARM servers often uses targets/sbsa-linux
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
echo "SCRIPT_DIR: $SCRIPT_DIR"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
echo "PROJECT_ROOT: $PROJECT_ROOT"

cd $PROJECT_ROOT
pip3 install setuptools==69.0.0
pip3 install ninja cmake wheel pybind11

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
