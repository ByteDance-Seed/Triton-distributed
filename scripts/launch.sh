#!/bin/bash

user_nproc_per_node=""
final_args=()

for arg in "$@"; do
  case $arg in
    --nproc_per_node=*)
      user_nproc_per_node="${arg#*=}"
      ;;
    *)
      final_args+=("$arg")
      ;;
  esac
done
set -- "${final_args[@]}"

function list_interfaces() {
  # Try /sys first (most reliable)
  if [ -d "/sys/class/net" ]; then
    ls /sys/class/net/
    return 0
  fi

  # Try /proc
  if [ -f "/proc/net/dev" ]; then
    tail -n +3 /proc/net/dev | cut -d: -f1 | tr -d ' '
    return 0
  fi

  # Try ip command
  if command -v ip >/dev/null 2>&1; then
    ip link show | grep -E "^[0-9]+:" | cut -d: -f2 | tr -d ' '
    return 0
  fi

  # Try ifconfig
  if command -v ifconfig >/dev/null 2>&1; then
    ifconfig -a | grep -E "^[a-zA-Z]" | cut -d: -f1
    return 0
  fi

  echo "No method available to list interfaces"
  return 1
}

check_if_interface_exists() {
  # Usage: check_if_interface_exists <interface_name>
  # Returns: 0=exists, 1=doesn't exist, 2=error
  local iface="$1"
  local interfaces

  # Get list of interfaces or fail
  if ! interfaces=$(list_interfaces 2>/dev/null); then
    echo "Error: Could not list network interfaces" >&2
    return 2
  fi

  # Check for exact match
  echo "$interfaces" | grep -qxF "$iface"
  case $? in
  0) return 0 ;; # Found
  1) return 1 ;; # Not found
  *) return 2 ;; # Error
  esac
}

function check_nvshmem_bootstrap_uid_sock() {
  local YELLOW='\033[0;33m'
  local BOLD='\033[1m'
  local RESET='\033[0m'
  local WARN_ICON='⚠️'

  local HAS_IPV4=0

  # check NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME: better matches the NCCL
  if [ -n "${NCCL_SOCKET_IFNAME}" ]; then
    echo "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME}"
    _NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME#=} # remove leading '='. refer to NCCL syntax: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html#nccl-socket-ifname
    if [ "${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}" != "${_NCCL_SOCKET_IFNAME}" ]; then
      echo -e "${YELLOW}${BOLD}${WARN_ICON} WARNING: ${RESET}${BOLD}${RESET} NVSHMEM and NCCL use the different socket interface. force set NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME(${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}) to NCCL_SOCKET_IFNAME(${_NCCL_SOCKET_IFNAME}) instead..."
      export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${_NCCL_SOCKET_IFNAME}
    fi
  fi

  if [ -n "${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}" ]; then
    if ! check_if_interface_exists ${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}; then
      echo -e "${YELLOW}${BOLD}${WARN_ICON} WARNING: ${RESET}${BOLD}${RESET} NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME} does not exist..."
    fi
  fi

  if [ -n "${NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY}" ]; then
    if command -v ip >/dev/null 2>&1; then
      ip -4 addr show dev "${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME}" 2>/dev/null | grep -q 'inet' && HAS_IPV4=1
    fi
  fi

  if [ ${HAS_IPV4} -eq 0 ]; then
    echo -e "${YELLOW}${BOLD}${WARN_ICON} WARNING: ${RESET}${BOLD}${RESET} NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=${NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY} does not support IPv4, force set NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY to AF_INET6..."
    export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=AF_INET6
  fi
}

function set_nvshmem_home() {
  # 1. Check if NVSHMEM_HOME environment variable is set
  if [ -n "$NVSHMEM_HOME" ]; then
    echo "Found NVSHMEM_HOME from environment variable: $NVSHMEM_HOME"
  else
    # 2. Try to find from Python command
    export NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))" 2>/dev/null)

    if [ -n "$NVSHMEM_HOME" ]; then
      echo "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: $NVSHMEM_HOME"
    else
      # 3. Fallback to ldconfig
      export NVSHMEM_HOME=$(ldconfig -p | grep 'libnvshmem_host' | awk '{print $NF}' | xargs dirname | head -n 1)

      if [ -n "$NVSHMEM_HOME" ]; then
        echo "Found NVSHMEM_HOME from ldconfig: $NVSHMEM_HOME"
      else
        echo "warning: NVSHMEM_HOME could not be determined."
      fi
    fi
  fi
}

set_nvshmem_home
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

export NVSHMEM_SYMMETRIC_SIZE=${NVSHMEM_SYMMETRIC_SIZE:-1000000000}
NVSHMEM_DIR=${NVSHMEM_DIR:-$NVSHMEM_HOME}
export LD_LIBRARY_PATH=${NVSHMEM_DIR}/lib:${LD_LIBRARY_PATH}
export NVSHMEM_DISABLE_CUDA_VMM=${NVSHMEM_DISABLE_CUDA_VMM:-1} # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-eth0}
export NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY=${NVSHMEM_BOOTSTRAP_UID_SOCK_FAMILY:-AF_INET}

check_nvshmem_bootstrap_uid_sock

if [ -n "$user_nproc_per_node" ]; then
  nproc_per_node=${user_nproc_per_node}
else
  nproc_per_node=${ARNOLD_WORKER_GPU:=$(nvidia-smi --list-gpus | wc -l)}
fi
nnodes=${ARNOLD_WORKER_NUM:=1}
node_rank=${ARNOLD_ID:=0}

master_addr=${ARNOLD_WORKER_0_HOST:="127.0.0.1"}
if [ -z ${ARNOLD_WORKER_0_PORT} ]; then
  master_port="23456"
else
  master_port=$(echo "$ARNOLD_WORKER_0_PORT" | cut -d "," -f 1)
fi

additional_args="--rdzv_endpoint=${master_addr}:${master_port}"

# If you want to use compute-sanitizer, please set TORCHRUN="/usr/local/cuda/bin/compute-sanitizer --tool memcheck torchrun"
# export PYTORCH_NO_CUDA_MEMORY_CACHING=1
# TORCHRUN="/usr/local/cuda/bin/compute-sanitizer --tool memcheck torchrun"
TORCHRUN=torchrun
CMD="${TORCHRUN} \
  --node_rank=${node_rank} \
  --nproc_per_node=${nproc_per_node} \
  --nnodes=${nnodes} \
  ${additional_args} \
  ${DIST_TRITON_EXTRA_TORCHRUN_ARGS} \
  $@"

echo ${CMD}
${CMD}

ret=$?
exit $ret
