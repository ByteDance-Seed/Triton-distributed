#!/bin/bash

set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})

function apt_install_deps() {
    apt update -y
    apt-get install -y miopen-hip
}

function build_pyrocshmem_cmake() {
  pushd ${PROJECT_ROOT}/pyrocshmem
  mkdir -p build
  pushd build
  cmake .. \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DROCSHMEM_DIR=${ROCSHMEM_DIR}/lib/cmake/rocshmem \
    -DOMPI_DIR=${OPENMPI_UCX_INSTALL_DIR} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  make -j VERBOSE=1
  popd
  popd
}

function build_pyrocshmem_setup() {
  export CXX="hipcc"
  pushd ${PROJECT_ROOT}/pyrocshmem
  TORCH_DONT_CHECK_COMPILER_ABI=1 python3 setup.py install
  popd
}

function copy_rocshmem_bitcode() {
    if [ -n "$ROCSHMEM_HOME" ]; then
        local dst_path="$ROCSHMEM_HOME/lib"
    else
        local dst_path="${PROJECT_ROOT}/../../python/triton_dist/tools/compile"
    fi
    rocshmem_dir=${ROCSHMEM_DIR:-${PROJECT_ROOT}/rocshmem_build/install}
    lib_file=$rocshmem_dir/lib/librocshmem_device.bc
    if ! cp -f $lib_file $dst_path; then
      echo "Rocshmem bitcode move failed." >&2
      rm -rf "$tmp_dir"
      return 1
    fi

    echo "Rocshmem bitcode move done."
}

export ROCSHMEM_DIR=${PROJECT_ROOT}/rocshmem_build/install
export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_DIR}
export ROCSHMEM_HEADER=${ROCSHMEM_INSTALL_DIR}/include/rocshmem
export OPENMPI_UCX_INSTALL_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}/install/ompi"
export ROCM_INSTALL_DIR="/opt/rocm"

export PATH="${OPENMPI_UCX_INSTALL_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${OPENMPI_UCX_INSTALL_DIR}/lib:$LD_LIBRARY_PATH"

apt_install_deps

# build rocshmem
bash -x ${PROJECT_ROOT}/build_rocshmem.sh
# build rocshmem bitcode
bash -x ${PROJECT_ROOT}/scripts/build_rocshmem_device_bc.sh
# move bitcode
copy_rocshmem_bitcode
# build pyrocshmem
build_pyrocshmem_setup

echo "done"
