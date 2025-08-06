#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export NVSHMEM_SRC=${NVSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/nvshmem}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}

export BITCODE_LIB_ARCH=sm_89
export PATH=$PATH:~/.triton/llvm/llvm-ubuntu-x64/bin/

mkdir -p ${SCRIPT_DIR}/llvm_lib

pushd ${SCRIPT_DIR}/llvm_lib

clang-19 -c -emit-llvm -O1 -std=c++11 -x cuda --cuda-path=${CUDA_HOME} --cuda-device-only \
  --cuda-gpu-arch=${BITCODE_LIB_ARCH} -I ${NVSHMEM_SRC}/src/include \
  -I ${SCRIPT_DIR}/llvm_lib \
  -D__clang_llvm_bitcode_lib__ ${SCRIPT_DIR}/transfer_device.cu -o libnvshmemi_device.bc.unoptimized

opt --passes='internalize,inline,globaldce' -internalize-public-api-list='nvshmemi_*' \
  libnvshmemi_device.bc.unoptimized -o libnvshmemi_device.bc

llvm-dis libnvshmemi_device.bc

${NVSHMEM_SRC}/scripts/bitcode_lib_cleanup.sh ${SCRIPT_DIR}/llvm_lib/libnvshmemi_device.ll ${SCRIPT_DIR}/llvm_lib/libnvshmemi_device.ll.new

llvm-as libnvshmemi_device.ll.new -o ${SCRIPT_DIR}/libnvshmemi_device.bc

rm libnvshmemi_device.bc.unoptimized libnvshmemi_device.ll libnvshmemi_device.ll.new libnvshmemi_device.bc

popd
