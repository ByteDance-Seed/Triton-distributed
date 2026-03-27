#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MORI_DIR=${MORI_DIR:-${SCRIPT_DIR}/../3rdparty/mori}

echo ""
echo "=========================================="
echo "Step 1: Build and install mori"
echo "=========================================="

echo "Installing mori..."
cd "${MORI_DIR}"
if pip3 install . --no-build-isolation --verbose; then
    echo "pip3 install completed successfully."
else
    echo "Error: pip3 install failed. Aborting build." >&2
    exit 1
fi

echo "Mori build and installation complete"

echo ""
echo "=========================================="
echo "Step 2: Precompile device bitcode via JIT"
echo "=========================================="

if [ -n "$MORI_HOME" ]; then
    DST_PATH="$MORI_HOME/lib"
else
    DST_PATH="${SCRIPT_DIR}/../python/triton_dist/tools/compile"
fi
mkdir -p "$DST_PATH"

BC_PATH=$(python3 -c "from mori.ir.bitcode import find_bitcode; print(find_bitcode())" 2>&1 | tail -1)
if [ -z "$BC_PATH" ] || [ ! -f "$BC_PATH" ]; then
    echo "Error: JIT bitcode compilation failed." >&2
    exit 1
fi

cp -f "$BC_PATH" "$DST_PATH/libmori_shmem_device.bc"

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo "  Mori Python module: installed via pip"
echo "  Device BC (JIT): $BC_PATH"
echo "  Device BC (copy): $DST_PATH/libmori_shmem_device.bc"
echo "=========================================="
