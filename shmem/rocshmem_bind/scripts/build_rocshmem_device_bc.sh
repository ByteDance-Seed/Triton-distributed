#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_INSTALL_DIR:-${SCRIPT_DIR}/../rocshmem_build/install}
export ROCSHMEM_SRC=${ROCSHMEM_SRC:-${SCRIPT_DIR}/../../../3rdparty/rocshmem}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
export OMPI_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"

pushd ${ROCSHMEM_INSTALL_DIR}/lib

# Device bitcode is offload-arch specific, so it must match the GPU it will run
# on. Default to the local GPU's arch (auto-detected) so the build is correct on
# both gfx942 and gfx950; overridable via BITCODE_LIB_ARCH, and falls back to
# gfx942 if detection fails.
if [ -z "${BITCODE_LIB_ARCH}" ]; then
    BITCODE_LIB_ARCH="$("${ROCM_PATH}/bin/rocm_agent_enumerator" 2>/dev/null | grep -v gfx000 | grep -m1 gfx || true)"
    [ -z "${BITCODE_LIB_ARCH}" ] && BITCODE_LIB_ARCH=gfx942
fi
export BITCODE_LIB_ARCH
echo "[build_rocshmem_device_bc] BITCODE_LIB_ARCH=${BITCODE_LIB_ARCH}"
CLANG="${ROCM_CXX:-${ROCM_PATH}/lib/llvm/bin/clang++}"
CLANG_FLAGS=(
    -x hip
    --cuda-device-only
    -std=c++20
    -emit-llvm
    # Do NOT let each TU bundle+internalize the ROCm device libs. clang's per-TU
    # device-lib linking makes ockl/ocml helpers `internal`, so llvm-linking the
    # TUs yields many internal copies (e.g. __ockl_get_local_size, .6, .11, ...).
    # When this bitcode is linked into a Triton kernel (whose own __ockl_* refs are
    # external) and run through Triton's O3, those internal copies collide with the
    # external refs and degrade into an invalid `declare internal` that aborts LLVM
    # IR parsing on newer LLVM. Instead keep ockl/ocml external here and link the
    # device libs exactly once below, for a single external copy that dedups cleanly.
    -nogpulib
    -mcode-object-version=5
    --offload-arch=${BITCODE_LIB_ARCH}
    -I${ROCSHMEM_INSTALL_DIR}/include/rocshmem
    -I${ROCSHMEM_INSTALL_DIR}/include
    -I${ROCSHMEM_INSTALL_DIR}/../
    -I${ROCSHMEM_SRC}/src
    -I${OMPI_DIR}/include
    -DOMPI_SKIP_MPICXX
)
# Distro Open MPI on Ubuntu may leave mpi.h only in the multiarch tree.
# Keep those as extra -I so device bitcode still compiles if staging missed them.
for _mpi_inc in \
    /usr/lib/x86_64-linux-gnu/openmpi/include \
    /usr/lib/aarch64-linux-gnu/openmpi/include \
    /usr/include; do
    if [ -f "${_mpi_inc}/mpi.h" ]; then
        CLANG_FLAGS+=("-I${_mpi_inc}")
    fi
done
if [ ! -f "${OMPI_DIR}/include/mpi.h" ] && [ ! -f /usr/lib/x86_64-linux-gnu/openmpi/include/mpi.h ] && [ ! -f /usr/include/mpi.h ]; then
    echo "error: mpi.h not found under ${OMPI_DIR}/include or distro Open MPI paths" >&2
    exit 1
fi

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
    ["${ROCSHMEM_SRC}/src/gda/context_gda_device_coll.cpp"]="rocshmem_context_gda_device_coll.bc"
    ["${ROCSHMEM_SRC}/src/gda/backend_gda.cpp"]="rocshmem_backend_gda.bc"
    ["${ROCSHMEM_SRC}/src/gda/queue_pair.cpp"]="rocshmem_queue_pair.bc"
    ["${ROCSHMEM_SRC}/src/gda/ionic/queue_pair_ionic.cpp"]="rocshmem_queue_pair_ionic.bc"
    ["${ROCSHMEM_SRC}/src/gda/mlx5/queue_pair_mlx5.cpp"]="rocshmem_queue_pair_mlx5.bc"
    ["${ROCSHMEM_SRC}/src/team.cpp"]="rocshmem_team.bc"
    ["${ROCSHMEM_SRC}/src/sync/abql_block_mutex.cpp"]="rocshmem_abql_block_mutex.bc"
    ["${ROCSHMEM_SRC}/src/util.cpp"]="rocshmem_util.bc"
    ["${ROCSHMEM_SRC}/src/context_device.cpp"]="rocshmem_context_device.bc"
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

# Linking all rocshmem TU bitcode into librocshmem_device.bc (ockl/ocml still
# external at this point).
"${LINKER}" "${BITCODE_FILES[@]}" -o "${OUTPUT_DIR}/librocshmem_device.bc"

# Link the ROCm device libraries exactly once so the rocshmem device code's
# __ockl_*/__ocml_* references (get_local_id/size, the assert->fprintf machinery,
# etc.) resolve to a single external copy. This bitcode is consumed both by
# kernels that pull in ockl themselves (which dedups against this copy) and by
# kernels that don't (which rely on this copy), so it must be self-contained. The
# oclc control constants must match the build flags (code object v5 => abi 500;
# CDNA gfx9xx => wavefrontsize64) and the target arch.
DEVICE_LIB_DIR="${ROCM_DEVICE_LIB_DIR:-${ROCM_PATH}/amdgcn/bitcode}"
_isa="${BITCODE_LIB_ARCH#gfx}"
DEVICE_LIBS=(
    "${DEVICE_LIB_DIR}/ockl.bc"
    "${DEVICE_LIB_DIR}/ocml.bc"
    "${DEVICE_LIB_DIR}/oclc_isa_version_${_isa}.bc"
    "${DEVICE_LIB_DIR}/oclc_abi_version_500.bc"
    "${DEVICE_LIB_DIR}/oclc_wavefrontsize64_on.bc"
    "${DEVICE_LIB_DIR}/oclc_daz_opt_off.bc"
    "${DEVICE_LIB_DIR}/oclc_finite_only_off.bc"
    "${DEVICE_LIB_DIR}/oclc_correctly_rounded_sqrt_on.bc"
    "${DEVICE_LIB_DIR}/oclc_unsafe_math_off.bc"
)
"${LINKER}" "${OUTPUT_DIR}/librocshmem_device.bc" "${DEVICE_LIBS[@]}" \
    -o "${OUTPUT_DIR}/librocshmem_device.bc"

# Drop the now-unused device-lib functions (keeps the bitcode small) without
# internalizing the externally-referenced rocshmem API or the ockl symbols the
# consuming kernel still needs.
OPT="${ROCM_OPT:-${ROCM_PATH}/lib/llvm/bin/opt}"
"${OPT}" -passes='globaldce' "${OUTPUT_DIR}/librocshmem_device.bc" \
    -o "${OUTPUT_DIR}/librocshmem_device.bc"

popd
