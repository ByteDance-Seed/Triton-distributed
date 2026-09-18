export CUDA_LAUNCH_BLOCKING=0
export TORCH_CPP_LOG_LEVEL=1
export NCCL_DEBUG=ERROR

# Project root (this script lives in <root>/scripts/). Derive it from BASH_SOURCE
# instead of $(pwd) so the script behaves the same regardless of the caller's cwd
# (tutorials source it from the repo root; launch.sh / CI harnesses source it via
# an absolute path).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 1. Check if NVSHMEM_HOME environment variable is set
if [ -n "$NVSHMEM_HOME" ]; then
  echo "Found NVSHMEM_HOME from environment variable: $NVSHMEM_HOME"
else
  # 2. Try to find from Python command
  NVSHMEM_HOME=$(python3 -c "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))" 2>/dev/null)

  if [ -n "$NVSHMEM_HOME" ]; then
    echo "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: $NVSHMEM_HOME"
  else
    # 3. Fallback to ldconfig
    NVSHMEM_HOME=$(ldconfig -p | grep 'libnvshmem_host' | awk '{print $NF}' | xargs -r dirname | head -n 1)
    if [ -n "$NVSHMEM_HOME" ]; then
      echo "Found NVSHMEM_HOME from ldconfig: $NVSHMEM_HOME"
    else
      echo "warning: NVSHMEM_HOME could not be determined."
    fi
  fi
fi


OMPI_BUILD=${SCRIPT_DIR}/shmem/rocshmem_bind/ompi_build/install/ompi

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${NVSHMEM_HOME}/lib:${OMPI_BUILD}/lib

NVSHMEM_SRC_DIR=${NVSHMEM_SRC_DIR:-$HOME/.cache/nvshmem_src}
if [ -d "${NVSHMEM_SRC_DIR}/build/src/lib" ]; then
  NVSHMEM_SRC_HOME="${NVSHMEM_SRC_DIR}/build/src"
  export LD_LIBRARY_PATH=${NVSHMEM_SRC_HOME}/lib:$LD_LIBRARY_PATH
  export NVSHMEM_SRC_HOME=${NVSHMEM_SRC_HOME}
  echo "NVSHMEM_SRC_HOME=${NVSHMEM_SRC_HOME}"
fi

export NVSHMEM_DISABLE_CUDA_VMM=${NVSHMEM_DISABLE_CUDA_VMM:-1} # moving from cpp to shell
export NVSHMEM_BOOTSTRAP=UID
export NVSHMEM_HOME=${NVSHMEM_HOME}
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=eth0

export TRITON_CACHE_DIR=${SCRIPT_DIR}/triton_cache

export PYTHONPATH=$PYTHONPATH:${SCRIPT_DIR}/python
mkdir -p ${SCRIPT_DIR}/triton_cache

# --- Out-of-tree Triton-distributed plugin path -----------------------------
# The distributed dialects/passes/ops live in an out-of-tree Triton plugin that
# Triton discovers via TRITON_PLUGIN_PATHS. Triton reads that env var exactly
# once, the first time it loads plugins -- which happens during `import triton`.
# Sourcing this file exports the path *before* any child python process imports
# triton, and the value is inherited by direct `python3 ...` invocations and by
# `torchrun` children launched via launch.sh. Defined as a function (and invoked
# below) so harnesses can also re-run it explicitly after tweaking env.
function set_triton_dist_plugin() {
  if [ -n "${TRITON_DIST_PLUGIN_PATH}" ]; then
    _dist_plugin="${TRITON_DIST_PLUGIN_PATH}"
  else
    # Resolve via the canonical discovery logic in triton_dist._plugin (handles
    # both the installed layout and the in-tree developer build). Importing
    # triton_dist._plugin does NOT import triton, so it is safe here.
    _dist_plugin="$(python3 -c 'import triton_dist._plugin as p; print(p.find_plugin() or "")' 2>/dev/null)"
  fi
  if [ -n "${_dist_plugin}" ] && [ -f "${_dist_plugin}" ]; then
    case ":${TRITON_PLUGIN_PATHS}:" in
      *":${_dist_plugin}:"*) : ;; # already present
      *) export TRITON_PLUGIN_PATHS="${_dist_plugin}${TRITON_PLUGIN_PATHS:+:${TRITON_PLUGIN_PATHS}}" ;;
    esac
    echo "TRITON_PLUGIN_PATHS=${TRITON_PLUGIN_PATHS}"
  else
    echo "WARNING: libtriton_dist.so not found; build it via 'pip install ./python'"
  fi
}

set_triton_dist_plugin
