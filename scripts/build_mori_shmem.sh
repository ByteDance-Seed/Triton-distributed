#!/bin/bash
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export MORI_DIR=${MORI_DIR:-${SCRIPT_DIR}/../3rdparty/mori}
export MORI_BUILD_DIR=${MORI_BUILD_DIR:-${MORI_DIR}/build}
export ROCM_PATH=${ROCM_PATH:-/opt/rocm}

# Function to detect GPU architecture
detect_gpu_arch() {
    local arch=""
    
    # Method 1: Try rocm_agent_enumerator
    if [ -x "${ROCM_PATH}/bin/rocm_agent_enumerator" ]; then
        arch=$(${ROCM_PATH}/bin/rocm_agent_enumerator | grep -v "gfx000" | grep "gfx" | head -1)
    fi
    
    # Method 2: Try rocminfo if rocm_agent_enumerator failed
    if [ -z "$arch" ] && command -v rocminfo &> /dev/null; then
        arch=$(rocminfo | grep -oP 'gfx\w+' | head -1)
    fi
    
    # Method 3: Check environment variable
    if [ -z "$arch" ] && [ -n "$AMDGPU_TARGETS" ]; then
        arch=$(echo "$AMDGPU_TARGETS" | tr ',' '\n' | grep "gfx" | head -1)
    fi
    
    # Fallback to gfx942 if detection failed
    if [ -z "$arch" ]; then
        echo "Warning: Could not detect GPU architecture, defaulting to gfx942" >&2
        arch="gfx942"
    fi
    
    echo "$arch"
}

# Detect GPU architecture
GPU_ARCH=$(detect_gpu_arch)
echo "Detected GPU architecture: $GPU_ARCH"

echo ""
echo "=========================================="
echo "Step 1: Build and install mori (including Python interface)"
echo "=========================================="

# Install mori Python interface (this will also build and install C++ library via CMake)
echo "Installing mori..."
cd "${MORI_DIR}"
# pip install -r requirements-build.txt
if pip3 install . --no-build-isolation --verbose; then
    echo "pip3 install completed successfully."
else
    echo "Error: pip3 install failed. Aborting build." >&2
    exit 1
fi

echo "Mori build and installation complete"

echo ""
echo "=========================================="
echo "Step 2: Build device BC file"
echo "=========================================="

# Source BC files from mori build (all components that make up mori_shmem)
WRAPPER_BC="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir/shmem_device_api_wrapper-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.bc"
INIT_BC="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir/init-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.bc"
MEMORY_BC="${MORI_BUILD_DIR}/src/shmem/CMakeFiles/mori_shmem.dir/memory-hip-amdgcn-amd-amdhsa-${GPU_ARCH}.bc"

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

echo "Copying mori_shmem bitcode to destination..."

# Determine destination path (similar to rocshmem build.sh)
if [ -n "$MORI_HOME" ]; then
    DST_PATH="$MORI_HOME/lib"
else
    DST_PATH="${SCRIPT_DIR}/../python/triton_dist/tools/compile"
fi

mkdir -p "$DST_PATH"

if ! cp -f "$TEMP_DIR/libmori_shmem_device.bc" "$DST_PATH/"; then
    echo "Error: Mori bitcode copy failed." >&2
    exit 1
fi

echo "✓ Mori bitcode copied to: $DST_PATH/libmori_shmem_device.bc"

echo ""
echo "=========================================="
echo "  Build complete!"
echo "=========================================="
echo "  Mori Python module: installed via pip"
echo "  Device BC: $DST_PATH/libmori_shmem_device.bc"
echo "=========================================="
