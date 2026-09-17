#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MORI_DIR=${MORI_DIR:-${SCRIPT_DIR}/../3rdparty/mori}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}
PYTHON=${PYTHON:-python3}

echo "=========================================="
echo "Step 1: Build and install amd_mori"
echo "=========================================="

# The distribution was renamed mori -> amd_mori; both own the same `mori/`
# import path, so drop a stale `mori` install before reinstalling. --force-reinstall
# is required because amd_mori's version is always 0.0.0: pip would otherwise treat
# a changed source tree as "already satisfied" and skip it.
"${PYTHON}" -m pip uninstall -y mori >/dev/null 2>&1 || true
cd "${MORI_DIR}"
"${PYTHON}" -m pip install . --no-build-isolation --no-deps --force-reinstall --verbose

echo "=========================================="
echo "Step 2: Resolve and stage device bitcode"
echo "=========================================="

# MORI exposes its Triton-compatible (code object v5) device bitcode via find_bitcode.
# On a cold cache find_bitcode triggers a JIT compile that prints "[mori-jit] ..."
# progress to stdout, so write the resolved path to a temp file instead of capturing
# stdout (which would mix the log lines into the path).
MORI_BC_PATH_FILE=$(mktemp)
trap 'rm -f "${MORI_BC_PATH_FILE}"' EXIT
MORI_BC_PATH_FILE="${MORI_BC_PATH_FILE}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
from mori.ir import find_bitcode

Path(os.environ["MORI_BC_PATH_FILE"]).write_text(str(find_bitcode(cov=5)), encoding="utf-8")
PY
MORI_BC=$(<"${MORI_BC_PATH_FILE}")
if [ ! -f "${MORI_BC}" ]; then
    echo "Error: MORI bitcode not found at resolved path: ${MORI_BC}" >&2
    exit 1
fi

if [ -n "${MORI_HOME:-}" ]; then
    DST_PATH="${MORI_HOME}/lib"
else
    DST_PATH="${SCRIPT_DIR}/../python/triton_dist/tools/compile"
fi
mkdir -p "${DST_PATH}"
STAGED_BC="${DST_PATH}/libmori_shmem_device.bc"
cp -f "${MORI_BC}" "${STAGED_BC}"

# Fail the build here if the staged bitcode is ABI-incompatible with the AMD
# fused-MoE path, so the breakage surfaces now instead of at kernel-compile time.
# Two representative symbols are checked:
#
#   mori_shmem_putmem_nbi_signal_block
#     A cooperative block-scope put-with-signal. It stands in for the whole family
#     of warp/block SHMEM primitives the fused-MoE kernels link against. Older MORI
#     only exported thread-scope wrappers, so this symbol was absent -- that was the
#     original build breakage this update fixes.
#
#   _ZN4mori5shmem15globalGpuStatesE  (== mori::shmem::globalGpuStates)
#     A device global holding this GPU's SHMEM runtime state (PE id, peer pointer
#     table, symmetric-heap base, transport info); every mori_shmem_* device fn
#     reads it. Triton compiles each kernel into its own HIP module, so each module
#     that links this bitcode gets its OWN copy of the symbol, zero-initialized.
#     jit.py's post-compile hook copies the host-initialized GpuStates into it via
#     shmem_module_init(). If the symbol is missing that init has nowhere to land;
#     it must also be a strong definition (not weak/declared) or the host state
#     lands on the wrong instance and device reads see garbage peer pointers.
# ROCm 7.1 images ship llvm-dis under llvm/bin, not lib/llvm/bin. Walk the
# common prefixes (and PATH) so a missing default path does not fail the job.
resolve_llvm_dis() {
    local cand
    for cand in \
        "${LLVM_DIS:-}" \
        "${ROCM_PATH}/lib/llvm/bin/llvm-dis" \
        "${ROCM_PATH}/llvm/bin/llvm-dis" \
        /opt/rocm/lib/llvm/bin/llvm-dis \
        /opt/rocm/llvm/bin/llvm-dis \
        "$(command -v llvm-dis 2>/dev/null || true)"
    do
        if [ -n "${cand}" ] && [ -x "${cand}" ]; then
            printf '%s\n' "${cand}"
            return 0
        fi
    done
    return 1
}
LLVM_DIS="$(resolve_llvm_dis)" || {
    echo "Error: llvm-dis not found under ${ROCM_PATH} or PATH" >&2
    exit 1
}
echo "Using llvm-dis: ${LLVM_DIS}"
STAGED_LL=$(mktemp)
trap 'rm -f "${STAGED_LL}"' EXIT
"${LLVM_DIS}" "${STAGED_BC}" -o "${STAGED_LL}"
grep -q '@mori_shmem_putmem_nbi_signal_block' "${STAGED_LL}" \
    || { echo "Error: MORI bitcode missing cooperative APIs (mori_shmem_putmem_nbi_signal_block)" >&2; exit 1; }
grep -q '@_ZN4mori5shmem15globalGpuStatesE' "${STAGED_LL}" \
    || { echo "Error: MORI bitcode missing globalGpuStates" >&2; exit 1; }

echo "✓ MORI bitcode staged at: ${STAGED_BC}"
