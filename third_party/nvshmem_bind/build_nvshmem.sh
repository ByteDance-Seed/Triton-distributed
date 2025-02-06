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

export NVSHMEM_IBGDA_SUPPORT=0
export NVSHMEM_IBGDA_SUPPORT_GPUMEM_ONLY=0
export NVSHMEM_IBDEVX_SUPPORT=0
export NVSHMEM_IBRC_SUPPORT=1
export NVSHMEM_LIBFABRIC_SUPPORT=0
export NVSHMEM_MPI_SUPPORT=1
export NVSHMEM_USE_GDRCOPY=0
export NVSHMEM_ENABLE_ALL_DEVICE_INLINING=0
export NVSHMEM_BUILD_BITCODE_LIBRARY=1

export NVSHMEM_SRC=${PROJECT_ROOT}/../nvshmem

pushd ${NVSHMEM_SRC}
mkdir -p build
cd build
if [ ${NVSHMEM_BUILD_BITCODE_LIBRARY} -eq "1" ]; then
  echo "libclang-19-dev or higher is required."
  mkdir -p src/llvm_lib
fi
CMAKE=${CMAKE:-cmake} # default cmake version maybe <= 3.19
if [ ! -f CMakeCache.txt ]; then
    ${CMAKE} .. \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
        ${CUDAARCH_ARGS} \
        -DNVSHMEM_BUILD_TESTS=OFF \
        -DNVSHMEM_BUILD_EXAMPLES=OFF \
        -DNVSHMEM_BUILD_PACKAGES=OFF \
        -DNVSHMEM_BUILD_BITCODE_LIBRARY=${NVSHMEM_BUILD_BITCODE_LIBRARY} \
        -DCMAKE_INSTALL_PREFIX=${NVSHMEM_SRC}/build/install
fi
make VERBOSE=1 -j${JOBS}
make install
popd
