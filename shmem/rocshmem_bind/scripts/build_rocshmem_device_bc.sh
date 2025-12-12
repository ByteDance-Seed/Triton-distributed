#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/../rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/rocshmem}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export OMPI_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"

pushd ${ROCSHMEM_INSTALL_DIR}/lib

# TODO: backend is hardcoded in the bitcode
DEFINES="-DENABLE_IBGDA_BITCODE"
# TODO: arch is hardcoded
ARCH=gfx942
# Note that there may be issues if the ROCM llvm version doesn't match triton.
ROCM_CXX=${ROCM_CXX:-${ROCM_PATH}/lib/llvm/bin/clang++}
ROCM_LD=${ROCM_LD:-${ROCM_PATH}/lib/llvm/bin/llvm-link}
ALL_BC=()

compile () {
    local cpp_file="$1"
    local filename=$(basename "$cpp_file")
    local base=${filename%.*}
    local base=${base#rocshmem_}
    local bc_file="rocshmem_${base}.bc"
    ALL_BC+=("${bc_file}")
    ${ROCM_CXX} -x hip --cuda-device-only -std=c++20  -emit-llvm  --offload-arch=${ARCH} -ggdb \
        ${DEFINES} \
        -I${ROCSHMEM_INSTALL_DIR}/include/rocshmem \
        -I${ROCSHMEM_INSTALL_DIR}/include \
        -I${ROCSHMEM_INSTALL_DIR}/../ \
        -I${ROCSHMEM_SRC}/src \
        -I${OMPI_DIR}/include \
        -I${ROCSHMEM_SRC}/src \
        -c "${cpp_file}" \
        -o "${bc_file}"
}

compile ${ROCSHMEM_SRC}/src/rocshmem_gpu.cpp
compile ${ROCSHMEM_SRC}/src/ipc/backend_ipc.cpp
compile ${ROCSHMEM_SRC}/src/ipc/context_ipc_device.cpp
compile ${ROCSHMEM_SRC}/src/ipc/context_ipc_device_coll.cpp
compile ${ROCSHMEM_SRC}/src/ipc_policy.cpp
compile ${ROCSHMEM_SRC}/src/gda/context_gda_device.cpp
compile ${ROCSHMEM_SRC}/src/gda/backend_gda.cpp
compile ${ROCSHMEM_SRC}/src/gda/queue_pair.cpp
compile ${ROCSHMEM_SRC}/src/gda/ionic/queue_pair_ionic.cpp
compile ${ROCSHMEM_SRC}/src/gda/mlx5/queue_pair_mlx5.cpp
compile ${ROCSHMEM_SRC}/src/gda/mlx5/segment_builder.cpp
compile ${ROCSHMEM_SRC}/src/gda/endian.cpp
compile ${ROCSHMEM_SRC}/src/team.cpp
compile ${ROCSHMEM_SRC}/src/sync/abql_block_mutex.cpp
compile ${ROCSHMEM_SRC}/src/util.cpp
compile ${SCRIPT_DIR}/../runtime/rocshmem_wrapper.cc

${ROCM_LD} "${ALL_BC[@]}" -o librocshmem_device.bc 

popd
