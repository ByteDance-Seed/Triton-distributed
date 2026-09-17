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

# GitHub runners already install distro Open MPI. rocSHMEM's
# install_dependencies.sh otherwise clones Open MPI from source and hits
# submodule SHAs that are no longer on any advertised ref.
# Resolve a distro MPI compiler without using the staging prefix. build.sh
# prepends ${prefix}/bin to PATH, so `command -v mpicc` would otherwise return
# the staging symlink and `ln -sfn` would rewrite it into a self-loop.
# Keep the wrapper path (e.g. /usr/bin/mpicc). `readlink -f` lands on
# opal_wrapper, which ignores --showme unless argv[0] is mpicc/mpicxx.
resolve_distro_mpi_bin() {
    local name="$1"
    local prefix="$2"
    local saved="${PATH}"
    local cand
    PATH="$(printf '%s' "${saved}" | tr ':' '\n' | grep -vx "${prefix}/bin" | paste -sd: -)"
    export PATH
    cand="$(command -v "${name}" || true)"
    PATH="${saved}"
    export PATH
    [ -n "${cand}" ] && [ -x "${cand}" ] || return 1
    printf '%s\n' "${cand}"
}

stage_ompi_includes() {
    local prefix="$1"
    local d
    shift
    for d in "$@"; do
        [ -n "${d}" ] || continue
        if [ -f "${d}/mpi.h" ]; then
            ln -sfn "${d}/mpi.h" "${prefix}/include/mpi.h"
        fi
        ln -sfn "${d}"/*.h "${prefix}/include/" 2>/dev/null || true
        if [ -d "${d}/openmpi" ]; then
            ln -sfn "${d}/openmpi" "${prefix}/include/openmpi"
        fi
    done
}

stage_system_ompi() {
    local prefix="${OMPI_INSTALL_DIR}/install/ompi"
    local libdir d mpicc_bin mpicxx_bin
    mkdir -p "${prefix}/bin" "${prefix}/include" "${prefix}/lib"
    mpicc_bin="$(resolve_distro_mpi_bin mpicc "${prefix}")" || return 1
    ln -sfn "${mpicc_bin}" "${prefix}/bin/mpicc"
    if mpicxx_bin="$(resolve_distro_mpi_bin mpicxx "${prefix}")"; then
        ln -sfn "${mpicxx_bin}" "${prefix}/bin/mpicxx"
    elif mpicxx_bin="$(resolve_distro_mpi_bin mpic++ "${prefix}")"; then
        ln -sfn "${mpicxx_bin}" "${prefix}/bin/mpicxx"
    fi
    # mpicc --showme:incdirs is a space-separated list. Ubuntu's mpi.h pulls
    # openmpi/ompi/mpi/cxx/mpicxx.h from a nested include dir, so stage every
    # reported prefix (and the nested openmpi/ tree), not just mpi.h.
    while read -r d; do
        stage_ompi_includes "${prefix}" "${d}"
    done < <("${mpicc_bin}" --showme:incdirs 2>/dev/null | tr ' ' '\n')
    # Debian/Ubuntu libopenmpi-dev keeps mpi.h under the multiarch tree, not
    # /usr/include. Probe those paths when --showme is empty (opal_wrapper).
    stage_ompi_includes "${prefix}" \
        /usr/include \
        /usr/lib/x86_64-linux-gnu/openmpi/include \
        /usr/lib/aarch64-linux-gnu/openmpi/include
    if [ ! -e "${prefix}/include/openmpi" ]; then
        if [ -d /usr/include/openmpi ]; then
            ln -sfn /usr/include/openmpi "${prefix}/include/openmpi"
        elif [ -d /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi ]; then
            ln -sfn /usr/lib/x86_64-linux-gnu/openmpi/include/openmpi "${prefix}/include/openmpi"
        elif [ -d /usr/lib/aarch64-linux-gnu/openmpi/include/openmpi ]; then
            ln -sfn /usr/lib/aarch64-linux-gnu/openmpi/include/openmpi "${prefix}/include/openmpi"
        fi
    fi
    libdir="$("${mpicc_bin}" --showme:libdirs 2>/dev/null | awk '{print $1}')"
    if [ -n "${libdir}" ]; then
        ln -sfn "${libdir}"/libmpi.so* "${prefix}/lib/" 2>/dev/null || true
        ln -sfn "${libdir}"/libopen-rte.so* "${prefix}/lib/" 2>/dev/null || true
        ln -sfn "${libdir}"/libopen-pal.so* "${prefix}/lib/" 2>/dev/null || true
    fi
    [ -e "${prefix}/include/mpi.h" ] && [ -x "${prefix}/bin/mpicc" ]
}

# Prefer restaging distro Open MPI whenever mpicc is present so a previous
# incomplete prefix (mpi.h without cxx headers) is repaired. Fall back to
# rocSHMEM's from-source Open MPI only when no usable distro prefix exists.
if command -v mpicc >/dev/null 2>&1 && stage_system_ompi; then
    echo "Using distro Open MPI at ${OMPI_INSTALL_DIR}/install/ompi"
elif [ -e "${OMPI_INSTALL_DIR}/install/ompi/include/mpi.h" ] && [ -x "${OMPI_INSTALL_DIR}/install/ompi/bin/mpicc" ]; then
    echo "ompi exists, skip building ompi and ucx"
else
    BUILD_DIR=${OMPI_INSTALL_DIR} bash ${ROCSHMEM_SRC_DIR}/scripts/install_dependencies.sh
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
