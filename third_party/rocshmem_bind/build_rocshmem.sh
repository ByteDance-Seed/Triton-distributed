#!/bin/bash

set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})
ROCSHMEM_SRC_DIR=${PROJECT_ROOT}/../rocshmem


# prepare for building ompi, ucx
apt-get install autoconf libtool flex -y
pushd ${ROCSHMEM_SRC_DIR}

ROCSHMEM_BUILD_DIR=${PROJECT_ROOT}/rocshmem_build
ROCSHMEM_INSTALL_DIR=${ROCSHMEM_BUILD_DIR}/install
OMPI_INSTALL_DIR=${SCRIPT_DIR}/ompi_build

# build ompi, ucx
if [ ! -e "$OMPI_INSTALL_DIR" ]; then
    BUILD_DIR=${OMPI_INSTALL_DIR} bash ${ROCSHMEM_SRC_DIR}/scripts/install_dependencies.sh
else
    echo "ompi exists, skip building ompi and ucx"
fi
export PATH="${OMPI_INSTALL_DIR}/install/ompi/bin:$PATH"
export LD_LIBRARY_PATH="${OMPI_INSTALL_DIR}/install/ompi/lib:$LD_LIBRARY_PATH"

# build rocSHMEM
mkdir -p ${ROCSHMEM_BUILD_DIR} && cd ${ROCSHMEM_BUILD_DIR}
bash ../scripts/build_rshm_ipc_single.sh ${ROCSHMEM_INSTALL_DIR}

popd
