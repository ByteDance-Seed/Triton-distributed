#!/bin/bash
set -x
cur_dir=$(dirname $(realpath $0))

export USE_TRITON_DISTRIBUTED_AOT=0
export TRITON_USE_MACA=ON
export TRITON_OFFLINE_BUILD=1
export TRITON_BUILD_PROTON=OFF

if [ $# -lt 1 ]; then
    echo "Usage: $0 <LLVM_SYSPATH>"
    echo "Please set LLVM_SYSPATH first"
    exit 1
fi

export LLVM_SYSPATH=$1
export LLVM_LIBRARY_DIR=${LLVM_SYSPATH}/lib
export LLVM_INCLUDE_DIRS=${LLVM_SYSPATH}/include
mlir_opt_path=${LLVM_SYSPATH}/bin/mlir-opt
if [ ! -e "$mlir_opt_path" ]; then
    echo "mlir-opt does not exist: $mlir_opt_path"
    echo "Please set and check LLVM_SYSPATH first."
    exit 1
fi

export MACA_PATH=/opt/maca
export LD_LIBRARY_PATH=/opt/maca/lib:/opt/maca/mxshmem/lib:/opt/maca/ompi/lib:/opt/maca/ucx/lib:/opt/mxdriver/lib
export CUDA_PATH=${MACA_PATH}/tools/cu-bridge
export CUCC_PATH=${MACA_PATH}/tools/cu-bridge
export CUCC_CMAKE_ENTRY=2
export PATH=${CUDA_PATH}/bin:${CUCC_PATH}/tools:${PATH}

# fetch metax triton to submodule 3rdparty/triton
METAX_TRITON_BRANCH=3.6
METAX_TRITON_COMMIT=70eb31eb1899ec64f1a301b3b065e7cf2c9f5802
if [ ! -d "${cur_dir}/../3rdparty/triton" ] || [ ! -d "${cur_dir}/../3rdparty/triton/third_party/metax" ]; then
    bash "${cur_dir}/pull_metax_triton.sh" "${METAX_TRITON_BRANCH}" "${METAX_TRITON_COMMIT}"
fi

backend_path=${cur_dir}/../3rdparty/triton/third_party/metax/backend/bin/
if [ ! -d $backend_path ]; then
    mkdir -p $backend_path
    cp "$mlir_opt_path" "$backend_path"
fi

# build
cd ${cur_dir}/..
pip3 install -r 3rdparty/triton/python/requirements.txt
echo 'numpy<2' > /tmp/pip_install_constraint.txt
pip3 install -c /tmp/pip_install_constraint.txt -e python[build,tests,tutorials] --verbose --use-pep517
