#!/bin/bash
set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})

ARCH=""

# Iterate over the command-line arguments
while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
    --arch)
        # Process the arch argument
        ARCH="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    --jobs)
        # Process the jobs argument
        JOBS="$2"
        shift # Skip the argument value
        shift # Skip the argument key
        ;;
    *)
        # Unknown argument
        echo "Unknown argument: $1"
        shift # Skip the argument
        ;;
    esac
done

if [[ -n $ARCH ]]; then
    export CMAKE_CUDA_ARCHITECTURES=${ARCH}
    CUDAARCH_ARGS="-DCMAKE_CUDA_ARCHITECTURES=${ARCH}"
fi

if [[ -z $JOBS ]]; then
    JOBS=$(nproc --ignore 2)
fi

export NVSHMEM_IBGDA_SUPPORT=${NVSHMEM_IBGDA_SUPPORT:-1}
export NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=0
export NVSHMEM_IBDEVX_SUPPORT=0
export NVSHMEM_IBRC_SUPPORT=1
export NVSHMEM_LIBFABRIC_SUPPORT=0
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_TORCH_SUPPORT=1
export NVSHMEM_ENABLE_ALL_DEVICE_INLINING=${NVSHMEM_ENABLE_ALL_DEVICE_INLINING:-1}

export NVSHMEM_BUILD_BITCODE_LIBRARY=${NVSHMEM_BUILD_BITCODE_LIBRARY:-0}


# download souce code
NVSHMEM_SRC_DIR=${NVSHMEM_SRC_DIR:-$HOME/.cache/nvshmem_src}
mkdir -p "${NVSHMEM_SRC_DIR}"

nvshmem_src=https://github.com/NVIDIA/nvshmem/releases/download/v3.4.5-0/nvshmem_src_cuda-all-all-3.4.5.tar.gz
wget -O "${NVSHMEM_SRC_DIR}/nvshmem_src.tar.gz" ${nvshmem_src}
tar -xzf "${NVSHMEM_SRC_DIR}/nvshmem_src.tar.gz" -C "${NVSHMEM_SRC_DIR}" --strip-components=1

pushd "${NVSHMEM_SRC_DIR}"
mkdir -p build
cd build
CMAKE=${CMAKE:-cmake}
if [ ! -f CMakeCache.txt ] || [ -z ${FLUX_BUILD_SKIP_CMAKE} ]; then
    ${CMAKE} .. \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        ${CUDAARCH_ARGS} \
        -DNVSHMEM_BUILD_TESTS=OFF \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_PACKAGES=OFF \
        -DNVSHMEM_BUILD_BITCODE_LIBRARY=${NVSHMEM_BUILD_BITCODE_LIBRARY} \
        -DNVSHMEM_ENABLE_ALL_DEVICE_INLINING=${NVSHMEM_ENABLE_ALL_DEVICE_INLINING} \
        -DNVSHMEM_IBGDA_SUPPORT=${NVSHMEM_IBGDA_SUPPORT}
fi

make VERBOSE=1 -j${JOBS}
popd

NVSHMEM_SRC_HOME=${NVSHMEM_SRC_DIR}/build/src
echo "NVSHMEM_SRC_HOME=${NVSHMEM_SRC_HOME}"
