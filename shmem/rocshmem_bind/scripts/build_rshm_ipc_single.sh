#!/bin/bash

set -e

if [ -z $1 ]
then
  install_path=~/rocshmem
else
  install_path=$1
fi

root_path="$(dirname "$(realpath $0)")/../../.."
src_path="${root_path}/3rdparty/rocshmem"
sys_path="${root_path}/3rdparty/rocm-systems"

if [ -d "${src_path}" ]; then
  pushd "${root_path}"
  active=$(git config submodule.3rdparty/rocshmem.active || echo "nil")
  if [ "${active}" = "true" ]; then
    echo "Error: Rocshmem submodule still active, please delete it"
    quit=1
  fi
  popd
  pushd "${src_path}"
  url=$(git remote get-url origin || echo "nil")
  if [ "${url}" = "https://github.com/ROCm/rocSHMEM.git" ]; then
    echo "Error: Old rocshmem checkout found, please delete it"
    quit=1
  fi
  popd
  if ! [ -z "${quit}" ]; then
    exit $quit
  fi

  if ! [ "$(ls -A "${src_path}")" ]; then
    rmdir "${src_path}"
  fi
fi

if ! [ -d "${src_path}" ]; then
  echo "Creating sparse checkout"
  tag=hip-version_7.12.60610
  pushd "${root_path}"
  git clone "https://github.com/ROCm/rocm-systems.git" -b "${tag}" --depth 1 --sparse "${sys_path}"
  popd
  pushd "${sys_path}"
  git config core.sparseCheckoutCone true
  git sparse-checkout set projects/rocshmem
  ln -s rocm-systems/projects/rocshmem ../rocshmem
  popd
fi

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=$install_path \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_VERBOSE_MAKEFILE=OFF \
    -DDEBUG=OFF \
    -DPROFILE=OFF \
    -DUSE_GDA=ON \
    -DUSE_RO=ON \
    -DUSE_DC=OFF \
    -DUSE_IPC=ON \
    -DGDA_IONIC=ON \
    -DGDA_MLX5=ON \
    -DUSE_COHERENT_HEAP=ON \
    -DUSE_THREADS=OFF \
    -DUSE_WF_COAL=OFF \
    -DUSE_SINGLE_NODE=ON \
    -DUSE_HOST_SIDE_HDP_FLUSH=OFF \
    -DBUILD_LOCAL_GPU_TARGET_ONLY=ON \
    $src_path
cmake --build . --parallel
cmake --install .
