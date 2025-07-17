export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

SCRIPT_DIR="$(pwd)"
SCRIPT_DIR=$(realpath ${SCRIPT_DIR})

# 1. Check if NVSHMEM_HOME environment variable is set
if [ -n "$NVSHMEM_HOME" ]; then
  echo "Found NVSHMEM_HOME from environment variable: $NVSHMEM_HOME"
else
  # 2. Try to find from Python command
  NVSHMEM_HOME=$(python -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))" 2>/dev/null)

  if [ -n "$NVSHMEM_HOME" ]; then
    echo "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: $NVSHMEM_HOME"
  else
    # 3. Fallback to ldconfig
    NVSHMEM_HOME=$(ldconfig -p | grep 'libnvshmem_host' | awk '{print $NF}' | xargs dirname | head -n 1)

    if [ -n "$NVSHMEM_HOME" ]; then
      echo "Found NVSHMEM_HOME from ldconfig: $NVSHMEM_HOME"
    else
      echo "warning: NVSHMEM_HOME could not be determined."
    fi
  fi
fi


OMPI_BUILD=${SCRIPT_DIR}/shmem/rocshmem_bind/ompi_build/install/ompi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_HOME}/lib:${OMPI_BUILD}/lib
export NVSHMEM_DISABLE_CUDA_VMM=1 # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID

export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

export TRITON_CACHE_DIR=${SCRIPT_DIR}/triton_cache

export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/python
mkdir -p ${SCRIPT_DIR}/triton_cache
