#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MORI_DIR=${MORI_DIR:-${SCRIPT_DIR}/../3rdparty/mori}
export MORI_BUILD_DIR=${MORI_BUILD_DIR:-${MORI_DIR}/build}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}

echo "=========================================="
echo "Step 1: Build and install mori (including Python interface)"
echo "=========================================="

# Install mori Python interface (this will also build and install C++ library via CMake)
echo "Installing mori..."
cd "${MORI_DIR}"
pip3 install . --no-build-isolation

echo "Mori build and installation complete"

echo ""
echo "=========================================="
echo "Step 2: Build device BC file"
echo "=========================================="

# Source BC files from mori build (all components that make up mori_shmem)
WRAPPER_BC="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir/shmem_device_api_wrapper-hip-amdgcn-amd-amdhsa-gfx942.bc"
INIT_BC="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir/init-hip-amdgcn-amd-amdhsa-gfx942.bc"
MEMORY_BC="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir/memory-hip-amdgcn-amd-amdhsa-gfx942.bc"

# Check if source BC files exist
if [ ! -f "$WRAPPER_BC" ]; then
    echo "Error: Wrapper BC file not found: $WRAPPER_BC"
    exit 1
fi

if [ ! -f "$INIT_BC" ]; then
    echo "Error: Init BC file not found: $INIT_BC"
    exit 1
fi

if [ ! -f "$MEMORY_BC" ]; then
    echo "Error: Memory BC file not found: $MEMORY_BC"
    exit 1
fi

# Destination path (only need to copy to backend, python has a symlink)
TRITON_BACKEND_LIB="${SCRIPT_DIR}/../3rdparty/triton/third_party/amd/backend/lib"

# Create temp directory for linking
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

echo "Linking all mori_shmem BC files (wrapper + init + memory)..."
${ROCM_PATH}/lib/llvm/bin/llvm-link \
    "$WRAPPER_BC" \
    "$INIT_BC" \
    "$MEMORY_BC" \
    -o "$TEMP_DIR/libmori_shmem_device.bc"

echo "Verifying linked BC file..."
${ROCM_PATH}/lib/llvm/bin/opt -S "$TEMP_DIR/libmori_shmem_device.bc" -o "$TEMP_DIR/libmori_shmem_device.ll"

# Check if globalGpuStates is properly defined (should NOT have weak attribute)
if grep -q "weak.*globalGpuStates" "$TEMP_DIR/libmori_shmem_device.ll"; then
    echo "Warning: globalGpuStates still has weak linkage"
    grep "globalGpuStates" "$TEMP_DIR/libmori_shmem_device.ll" | head -3
fi

# Check for proper global definition (__device__ or __constant__)
if grep -q "@_ZN4mori5shmem15globalGpuStatesE" "$TEMP_DIR/libmori_shmem_device.ll"; then
    echo "✓ globalGpuStates found in linked BC"
    grep "@_ZN4mori5shmem15globalGpuStatesE" "$TEMP_DIR/libmori_shmem_device.ll" | head -1
else
    echo "Error: globalGpuStates not found in linked BC"
    exit 1
fi

echo "Copying unified BC to triton backend..."
mkdir -p "$TRITON_BACKEND_LIB"
cp "$TEMP_DIR/libmori_shmem_device.bc" "$TRITON_BACKEND_LIB/"

echo "Creating symlink in python/triton/backends/amd/lib..."
TRITON_PYTHON_LIB="${SCRIPT_DIR}/../3rdparty/triton/python/triton/backends/amd/lib"
mkdir -p "$TRITON_PYTHON_LIB"

# Remove old symlink if exists
rm -f "$TRITON_PYTHON_LIB/libmori_shmem_device.bc"

# Create symlink (relative path)
cd "$TRITON_PYTHON_LIB"
ln -s ../../../../../third_party/amd/backend/lib/libmori_shmem_device.bc libmori_shmem_device.bc
cd - > /dev/null

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo "  Mori Python module: installed via pip"
echo "  Device BC: $TRITON_BACKEND_LIB/libmori_shmem_device.bc"
echo "  Symlink: $TRITON_PYTHON_LIB/libmori_shmem_device.bc"
echo "=========================================="
