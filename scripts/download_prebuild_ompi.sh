#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../)
ROCSHMEM_BIND_DIR="${PROJECT_ROOT}/shmem/rocshmem_bind/"
TEMP_DIR="${TMPDIR:-/tmp}"


pushd ${ROCSHMEM_BIND_DIR}

CLONED_AMD_TOOLS_DIR="${TEMP_DIR}/triton-distributed-amd-tools"
if [[ -d ${CLONED_AMD_TOOLS_DIR} ]]; then
  rm -r ${CLONED_AMD_TOOLS_DIR}
fi

git clone git@code.byted.org:mlsys/triton-distributed-amd-tools.git ${CLONED_AMD_TOOLS_DIR}

ROCSHMEM_VERSION="2025_7_9_87179b1"
PREBUILD_OMPI_PKG="ompi_build_${ROCSHMEM_VERSION}.tar.gz"
PREBUILD_OMPI_PKG_DIR=${CLONED_AMD_TOOLS_DIR}/${PREBUILD_OMPI_PKG}
OMPI_INSTALL_DIR="/opt/"

if [[ ! -f "${PREBUILD_OMPI_PKG_DIR}" ]]; then
  echo "err: download prebuild ompi package failed"
  exit -1
fi

tar zxf ${PREBUILD_OMPI_PKG_DIR} -C ${OMPI_INSTALL_DIR}

if [[ ! -d "${OMPI_INSTALL_DIR}/ompi_build" ]]; then
  echo "extract ompi package failed"
  exit -1
else
  echo "download prebuild ompi to ${OMPI_INSTALL_DIR}/ompi_build"
fi

popd
