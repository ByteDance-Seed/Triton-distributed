#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})

function apt_install_deps() {
    apt-get install -y miopen-hip
}

function build_rocshmem_hsaco() {
  pushd ${PROJECT_ROOT}/runtime
  hipcc -c -fgpu-rdc -x hip rocshmem_wrapper.cc -I${ROCM_INSTALL_DIR}/include -I${ROCSHMEM_HEADER}  -I${OPENMPI_UCX_INSTALL_DIR}/include -o rocshmem_wrapper.o
  #hipcc -fgpu-rdc --hip-link rocshmem_wrapper.o -o rocshmem_wrapper $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so -L${ROCM_INSTALL_DIR}/lib -lamdhip64 -lhsa-runtime64
  popd
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

function download_and_copy() {
    local dst_path=${PROJECT_ROOT}/../amd/backend/lib
}

# build rocshmem
export ROCSHMEM_DIR=${PROJECT_ROOT}/rocshmem_build/install
export ROCSHMEM_INSTALL_DIR=${ROCSHMEM_DIR}
export ROCSHMEM_HEADER=${ROCSHMEM_INSTALL_DIR}/include/rocshmem
export OPENMPI_UCX_INSTALL_DIR=${PROJECT_ROOT}/ompi_build/install/ompi
export ROCM_INSTALL_DIR="/opt/rocm"

export PATH="${OPENMPI_UCX_INSTALL_DIR}/bin:$PATH"
export LD_LIBRARY_PATH="${OPENMPI_UCX_INSTALL_DIR}/lib:$LD_LIBRARY_PATH"

apt_install_deps

bash -x ${PROJECT_ROOT}/build_rocshmem.sh

# build pyrocshmem
build_pyrocshmem_setup
build_rocshmem_hsaco
download_and_copy

echo "done"
