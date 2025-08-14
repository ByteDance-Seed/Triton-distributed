#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/../rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/rocshmem}
export ROCM_PATH=/opt/rocm-6.3.0/lib
export OMPI_DIR=/opt/ompi_build/install/ompi

pushd ${ROCSHMEM_INSTALL_DIR}/lib

# TODO: arch is hardcoded
${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -DENABLE_IPC_BITCODE \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/rocshmem_gpu.cpp \
 -o rocshmem_gpu.bc

${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc/context_ipc_device.cpp \
 -o rocshmem_context_device.bc

${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc/backend_ipc.cpp \
 -o rocshmem_backend_ipc.bc


${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc/context_ipc_device_coll.cpp \
 -o rocshmem_context_ipc_device_coll.bc

${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/ipc_policy.cpp \
 -o rocshmem_ipc_policy.bc

${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/team.cpp \
 -o rocshmem_team.bc

 ${ROCM_PATH}/llvm/bin/clang++ -x hip --cuda-device-only -std=c++17  -emit-llvm  --offload-arch=gfx942 \
 -I${ROCSHMEM_INSTALL_DIR}/include \
 -I${ROCSHMEM_INSTALL_DIR}/../ \
 -I${OMPI_DIR}/include \
 -c ${ROCSHMEM_SRC}/src/sync/abql_block_mutex.cpp \
 -o rocshmem_abql_block_mutex.bc


# ${ROCM_PATH}/llvm/bin/llvm-dis rocshmem_gpu.bc -o rocshmem_gpu.ll

# ${ROCM_PATH}/llvm/bin/llvm-dis rocshmem_context_device.bc -o rocshmem_context_device.ll

# ${ROCM_PATH}/llvm/bin/llvm-dis rocshmem_backend_ipc.bc -o rocshmem_backend_ipc.ll

# ${ROCM_PATH}/llvm/bin/llvm-dis rocshmem_context_ipc_device_coll.bc -o rocshmem_context_ipc_device_coll.ll

# ${ROCM_PATH}/llvm/bin/llvm-dis rocshmem_ipc_policy.bc -o rocshmem_ipc_policy.ll

# ${ROCM_PATH}/llvm/bin/llvm-dis rocshmem_team.bc -o rocshmem_team.ll

#${ROCM_PATH}/llvm/bin/llvm-link rocshmem_gpu.bc rocshmem_backend_ipc.bc rocshmem_context_device.bc rocshmem_context_ipc_device_coll.bc rocshmem_ipc_policy.bc rocshmem_team.bc rocshmem_abql_block_mutex.bc -o librocshmem_device.bc

# rm rocshmem_gpu.bc rocshmem_backend_ipc.bc rocshmem_context_device.bc rocshmem_context_ipc_device_coll.bc rocshmem_ipc_policy.bc rocshmem_team.bc

popd
