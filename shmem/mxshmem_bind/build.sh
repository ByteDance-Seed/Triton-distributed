#!/bin/bash
set -x
CUR_DIR="$(cd $(dirname $0);pwd)/"
# export MXSHMEM_DIR=
rm -rf ${CUR_DIR}/pymxshmem/build/

#set LLVM and MACA path
LLVM_SYSPATH=${LLVM_SYSPATH:-$CUR_DIR/../../../llvm_release/}
MACA_PATH=${MACA_PATH:-/opt/maca/}
#set MACA's env
export MACA_CLANG_PATH=${MACA_PATH}/mxgpu_llvm/bin
export PATH=${MACA_PATH}/bin:${MACA_CLANG_PATH}:${PATH}
export LD_LIBRARY_PATH=${MACA_PATH}/mxgpu_llvm/lib:${MACA_PATH}/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH}
#set cu-bridge's env
export CUDA_PATH=${MACA_PATH}/tools/cu-bridge
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export PATH=${CUCC_PATH}/tools:$PATH
#set llvm's env
export LD_LIBRARY_PATH=${LLVM_SYSPATH}/lib:$LD_LIBRARY_PATH
export PATH=${LLVM_SYSPATH}/bin:$PATH

if [ -n "$MACA_PATH" ]; then
    USE_MACA=ON
else
    USE_MACA=OFF
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT=$(realpath ${SCRIPT_DIR})
ARCH=""

while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
  --arch)
    ARCH="$2"
    shift
    shift
    ;;
  *)
    echo "Unknown argument: $1"
    shift
    ;;
  esac
done

if [[ -n $ARCH ]]; then
  build_args=" --arch ${ARCH}"
fi

function build_pymxshmem() {
  pushd ${PROJECT_ROOT}/pymxshmem
  MXSHMEM_HOME=${MXSHMEM_BUILD_DIR} pip3 install . -v --no-build-isolation
  popd
}

function set_arch() {
  if [[ -z $ARCH ]]; then
    export ARCH=$(python3 -c 'import torch; print("".join([str(x) for x in torch.cuda.get_device_capability()]))')
    echo "using CUDA arch: ${ARCH}"
  fi
}

function set_gencode() {
  GENCODE="" # default none
  arch_list=()
  IFS=";" read -ra arch_list <<<"$ARCH"
  for _arch in "${arch_list[@]}"; do
    GENCODE="-gencode=arch=compute_${_arch},code=sm_${_arch} ${GENCODE}"
  done
}

function move_libmxshmem_device_bc() {
  local dst_path=${PROJECT_ROOT}/../../../metax/backend/lib/
  if [ ! -d "$dst_path" ]; then
    mkdir -p "$dst_path"
    echo "Create $dst_path"
  fi
  lib_file=${MXSHMEM_DIR}/build/src/libmxshmem_device.bc
  if ! cp -f $lib_file $dst_path; then
    echo "File move failed" >&2
    rm -rf "$tmp_dir"
    return 1
  fi
}

# set_arch
set_gencode

export MXSHMEM_BUILD_DIR=${MXSHMEM_DIR}/build/src
bash -x ${PROJECT_ROOT}/build_mxshmem.sh ${build_args}
build_pymxshmem

move_libmxshmem_device_bc
ret=$?
if [ $ret == 1 ]; then
  echo "mxshmem build failed!"
else
  echo "mxshmem build success!"
fi
exit $ret
