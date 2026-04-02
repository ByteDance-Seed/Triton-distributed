#!/bin/bash

set -x
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})
ROCSHMEM_SRC_DIR=${PROJECT_ROOT}/../../3rdparty/rocshmem

sys_path="${PROJECT_ROOT}/../../3rdparty/rocm-systems"

if [ -d "${ROCSHMEM_SRC_DIR}" ]; then
  pushd "${PROJECT_ROOT}/../.."
  active=$(git config submodule.3rdparty/rocshmem.active || echo "nil")
  if [ "${active}" = "true" ]; then
    echo "Error: Rocshmem submodule still active, please delete it"
    quit=1
  fi
  popd
  pushd "${ROCSHMEM_SRC_DIR}"
  url=$(git remote get-url origin || echo "nil")
  if [ "${url}" = "https://github.com/ROCm/rocSHMEM.git" ]; then
    echo "Error: Old rocshmem checkout found, please delete it"
    quit=1
  fi
  popd
  if ! [ -z "${quit}" ]; then
    exit $quit
  fi

  if ! [ "$(ls -A "${ROCSHMEM_SRC_DIR}")" ]; then
    rmdir "${ROCSHMEM_SRC_DIR}"
  fi
fi

rocm_systems_tag=hip-version_7.12.60610

if ! [ -d "${ROCSHMEM_SRC_DIR}" ]; then
  echo "Creating sparse checkout"
  pushd "${PROJECT_ROOT}/../.."
  git clone "https://github.com/ROCm/rocm-systems.git" -b "${rocm_systems_tag}" --depth 1 --sparse "${sys_path}"
  popd
  pushd "${sys_path}"
  git config core.sparseCheckoutCone true
  git sparse-checkout set projects/rocshmem
  ln -s rocm-systems/projects/rocshmem ../rocshmem
  popd
fi

pushd "${sys_path}"
git checkout "${rocm_systems_tag}"
popd

pushd ${ROCSHMEM_SRC_DIR}

ROCSHMEM_BUILD_DIR=${PROJECT_ROOT}/rocshmem_build
ROCSHMEM_INSTALL_DIR=${ROCSHMEM_BUILD_DIR}/install
OMPI_INSTALL_DIR="${OMPI_INSTALL_DIR:-/opt/ompi_build}"

# build ompi, ucx
if [ ! -e "${OMPI_INSTALL_DIR}" ]; then
    # prepare for building ompi, ucx
    BUILD_DIR=${OMPI_INSTALL_DIR} bash ${ROCSHMEM_SRC_DIR}/scripts/install_dependencies.sh
else
    echo "ompi exists, skip building ompi and ucx"
fi

if [ ! -e "$OMPI_INSTALL_DIR" ]; then
  echo "error: build ompi failed"
  exit -1
fi

export PATH="${OMPI_INSTALL_DIR}/install/ompi/bin:$PATH"
export LD_LIBRARY_PATH="${OMPI_INSTALL_DIR}/install/ompi/lib:$LD_LIBRARY_PATH"


# build rocSHMEM
mkdir -p ${ROCSHMEM_BUILD_DIR} && cd ${ROCSHMEM_BUILD_DIR}
bash ../scripts/build_rshm_ipc_single.sh ${ROCSHMEM_INSTALL_DIR}

if [ ! -e "$ROCSHMEM_INSTALL_DIR" ]; then
  echo "error: build rocshmem failed"
  exit -1
fi

popd
