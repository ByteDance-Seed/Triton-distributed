#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/../rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/rocshmem}
export ROCM_PATH=/opt/rocm-6.3.0
export OMPI_DIR=/opt/ompi_build/install/ompi
echo "*** ROCSHMEM_INSTALL_DIR: ${ROCSHMEM_INSTALL_DIR}"
pushd ${ROCSHMEM_INSTALL_DIR}/lib

${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only \
 -std=c++17  -emit-llvm  --offload-arch=gfx942              \
 -I${ROCSHMEM_INSTALL_DIR}/include                          \
 -I${OMPI_DIR}/include                                      \
 -c ${SCRIPT_DIR}/../runtime/rocshmem_wrapper.cc            \
 -o rocshmem_wrapper.bc
 
${ROCM_PATH}/lib/llvm/bin/llvm-link                         \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_gpu.bc                \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_backend_ipc.bc        \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_context_device.bc     \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_context_ipc_device_coll.bc \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_ipc_policy.bc         \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_team.bc               \
 ${ROCSHMEM_INSTALL_DIR}/lib/rocshmem_abql_block_mutex.bc   \
 rocshmem_wrapper.bc -o librocshmem_device.bc 

# ${ROCM_PATH}/lib/llvm/bin/llvm-dis librocshmem_device.bc -o librocshmem_device.ll

# llc -march=amdgcn -mcpu=gfx942 -filetype=obj librocshmem_device.bc -o librocshmem_device.o
# ld.lld -shared librocshmem_device.o -o rocshmem_device.hsaco
popd
