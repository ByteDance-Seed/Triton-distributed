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
# ROCm 7.1 images may omit llvm-dis next to clang++, and apt `llvm` is LLVM 18
# which cannot read the LLVM 20 bitcode HIP produces. Prefer ROCm copies, then
# versioned llvm-dis-2x, and only keep a candidate that can actually disassemble.
# If none can, fall back to strings on the .bc (symbol names are still present).
collect_llvm_dis_candidates() {
    local cand hip_bin hip_dir
    [ -n "${LLVM_DIS:-}" ] && printf '%s\n' "${LLVM_DIS}"
    for cand in \
        "${ROCM_PATH}/lib/llvm/bin/llvm-dis" \
        "${ROCM_PATH}/llvm/bin/llvm-dis" \
        /opt/rocm/lib/llvm/bin/llvm-dis \
        /opt/rocm/llvm/bin/llvm-dis
    do
        printf '%s\n' "${cand}"
    done
    if hip_bin="$(command -v hipcc 2>/dev/null)"; then
        hip_dir="$(cd "$(dirname "${hip_bin}")/.." && pwd)"
        printf '%s\n' "${hip_dir}/lib/llvm/bin/llvm-dis" "${hip_dir}/llvm/bin/llvm-dis"
    fi
    if [ -d "${ROCM_PATH}" ]; then
        find "${ROCM_PATH}" /opt/rocm -name 'llvm-dis' -type f 2>/dev/null || true
    fi
    for cand in llvm-dis-21 llvm-dis-20 llvm-dis; do
        command -v "${cand}" 2>/dev/null || true
    done
}

STAGED_LL=$(mktemp)
trap 'rm -f "${STAGED_LL}"' EXIT
LLVM_DIS=""
while read -r cand; do
    [ -n "${cand}" ] && [ -x "${cand}" ] || continue
    if "${cand}" "${STAGED_BC}" -o "${STAGED_LL}" >/dev/null 2>&1; then
        LLVM_DIS="${cand}"
        break
    fi
done < <(collect_llvm_dis_candidates | awk 'NF && !seen[$0]++')

if [ -n "${LLVM_DIS}" ]; then
    echo "Using llvm-dis: ${LLVM_DIS}"
    grep -q '@mori_shmem_putmem_nbi_signal_block' "${STAGED_LL}" \
        || { echo "Error: MORI bitcode missing cooperative APIs (mori_shmem_putmem_nbi_signal_block)" >&2; exit 1; }
    grep -q '@_ZN4mori5shmem15globalGpuStatesE' "${STAGED_LL}" \
        || { echo "Error: MORI bitcode missing globalGpuStates" >&2; exit 1; }
else
    echo "Warning: no llvm-dis could read ${STAGED_BC}; checking symbols via strings"
    strings "${STAGED_BC}" | grep -q 'mori_shmem_putmem_nbi_signal_block' \
        || { echo "Error: MORI bitcode missing cooperative APIs (mori_shmem_putmem_nbi_signal_block)" >&2; exit 1; }
    strings "${STAGED_BC}" | grep -q '_ZN4mori5shmem15globalGpuStatesE' \
        || { echo "Error: MORI bitcode missing globalGpuStates" >&2; exit 1; }
fi

echo "✓ MORI bitcode staged at: ${STAGED_BC}"
