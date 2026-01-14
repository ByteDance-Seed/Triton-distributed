#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/../rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/rocshmem}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export OMPI_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"

pushd ${ROCSHMEM_INSTALL_DIR}/lib

export BITCODE_LIB_ARCH=gfx942
CLANG="${ROCM_CXX:-${ROCM_PATH}/lib/llvm/bin/clang++}"
CLANG_FLAGS=(
    -x hip
    --cuda-device-only
    -std=c++20
    -emit-llvm
    --offload-arch=${BITCODE_LIB_ARCH}
    -I${ROCSHMEM_INSTALL_DIR}/include/rocshmem
    -I${ROCSHMEM_INSTALL_DIR}/include
    -I${ROCSHMEM_INSTALL_DIR}/../
    -I${ROCSHMEM_SRC}/src
    -I${OMPI_DIR}/include
)

LINKER="${ROCM_LD:-${ROCM_PATH}/lib/llvm/bin/llvm-link}"
OUTPUT_DIR="${ROCSHMEM_INSTALL_DIR}/lib"

declare -A SOURCE_MAP
SOURCE_MAP=(
    ["${ROCSHMEM_SRC}/src/rocshmem_gpu.cpp"]="rocshmem_gpu.bc"
    ["${ROCSHMEM_SRC}/src/reverse_offload/backend_ro.cpp"]="rocshmem_backend_ro.bc"
    ["${ROCSHMEM_SRC}/src/reverse_offload/context_ro_device.cpp"]="rocshmem_context_ro_device.bc"
    ["${ROCSHMEM_SRC}/src/ipc/backend_ipc.cpp"]="rocshmem_backend_ipc.bc"
    ["${ROCSHMEM_SRC}/src/ipc/context_ipc_device.cpp"]="rocshmem_context_ipc_device.bc"
    ["${ROCSHMEM_SRC}/src/ipc/context_ipc_device_coll.cpp"]="rocshmem_context_ipc_device_coll.bc"
    ["${ROCSHMEM_SRC}/src/ipc_policy.cpp"]="rocshmem_ipc_policy.bc"
    ["${ROCSHMEM_SRC}/src/gda/context_gda_device.cpp"]="rocshmem_context_gda_device.bc"
    ["${ROCSHMEM_SRC}/src/gda/backend_gda.cpp"]="rocshmem_backend_gda.bc"
    ["${ROCSHMEM_SRC}/src/gda/queue_pair.cpp"]="rocshmem_queue_pair.bc"
    ["${ROCSHMEM_SRC}/src/gda/ionic/queue_pair_ionic.cpp"]="rocshmem_queue_pair_ionic.bc"
    ["${ROCSHMEM_SRC}/src/gda/mlx5/queue_pair_mlx5.cpp"]="rocshmem_queue_pair_mlx5.bc"
    ["${ROCSHMEM_SRC}/src/gda/mlx5/segment_builder.cpp"]="rocshmem_segment_builder.bc"
    ["${ROCSHMEM_SRC}/src/gda/endian.cpp"]="rocshmem_endian.bc"
    ["${ROCSHMEM_SRC}/src/team.cpp"]="rocshmem_team.bc"
    ["${ROCSHMEM_SRC}/src/sync/abql_block_mutex.cpp"]="rocshmem_abql_block_mutex.bc"
    ["${ROCSHMEM_SRC}/src/util.cpp"]="rocshmem_util.bc"
    ["${SCRIPT_DIR}/../runtime/rocshmem_wrapper.cc"]="rocshmem_wrapper.bc"
)

declare -A SOURCE_MAP_FLAGS
SOURCE_MAP_FLAGS=(
)

BITCODE_FILES=()
# Compiling each source file into bitcode
for src_file in "${!SOURCE_MAP[@]}"; do
    output_basename="${SOURCE_MAP[$src_file]}"
    output_file="${OUTPUT_DIR}/${output_basename}"

    extra_flags=${SOURCE_MAP_FLAGS[$src_file]:-}
    "${CLANG}" "${CLANG_FLAGS[@]}" ${extra_flags} -c "${src_file}" -o "${output_file}"

    BITCODE_FILES+=("${output_file}")
done

# Linking all bitcode files into librocshmem_device.bc
"${LINKER}" "${BITCODE_FILES[@]}" -o "${OUTPUT_DIR}/librocshmem_device.bc"

popd
