export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

SCRIPT_DIR="$(pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})
TRITON_NVSHMEM_DIR=${SCRIPT_DIR}/third_party/nvshmem_bind/python
PYNVSHMEM_DIR=${SCRIPT_DIR}/third_party/nvshmem_bind/pynvshmem
NVSHMEM_ROOT=${SCRIPT_DIR}/third_party/nvshmem/build/install

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_ROOT}/lib
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID

export PYTHONPATH=$PYTHONPATH:${PYNVSHMEM_DIR}/build:$TRITON_NVSHMEM_DIR
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

export TRITON_CACHE_DIR=${SCRIPT_DIR}/third_party/nvshmem_bind/python/triton_cache
export NVSHMEM_HOME=${NVSHMEM_ROOT}
mkdir -p ${SCRIPT_DIR}/triton_cache

