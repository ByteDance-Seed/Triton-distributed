#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/../rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/rocshmem}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export OMPI_DIR=/opt/ompi_build/install/ompi

pushd ${ROCSHMEM_INSTALL_DIR}/lib

# TODO: arch is hardcoded
${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -DENABLE_IPC_BITCODE \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/rocshmem_gpu.cpp \
 -o rocshmem_gpu.bc

${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc/context_ipc_device.cpp \
 -o rocshmem_context_device.bc

${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc/backend_ipc.cpp \
 -o rocshmem_backend_ipc.bc


${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc/context_ipc_device_coll.cpp \
 -o rocshmem_context_ipc_device_coll.bc

${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc_policy.cpp \
 -o rocshmem_ipc_policy.bc

${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/team.cpp \
 -o rocshmem_team.bc

 ${ROCM_PATH}/lib/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/sync/abql_block_mutex.cpp \
 -o rocshmem_abql_block_mutex.bc

 
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

popd
