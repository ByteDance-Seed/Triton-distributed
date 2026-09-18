#!/bin/bash
# Build the Triton-distributed out-of-tree plugin (libtriton_dist.so) + companion
# (libtriton_dist_ext) against the pinned Triton 3.7.1 submodule + cached LLVM.
#
# IMPORTANT: the plugin/companion MUST be compiled against the SAME Triton that
# is imported at runtime (the 3rdparty/triton submodule, v3.7.1). Building against
# any other tree (e.g. a stray 3.8 checkout) yields an ABI mismatch that crashes
# at plugin load / first builder call.
set -e

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
_TRITON_SUBMODULE="$REPO_ROOT/3rdparty/triton"

export TRITON_SOURCE_DIR=${TRITON_SOURCE_DIR:-$_TRITON_SUBMODULE}

# Resolve Triton's CMake build dir exactly like python/setup.py:locate_triton()
# -- glob build/cmake* and take the newest. The concrete name encodes the Python
# version and platform (e.g. cmake.linux-x86_64-cpython-3.11), so hardcoding it
# breaks under a different interpreter/arch; discover it instead.
if [ -z "$TRITON_BUILD_DIR" ]; then
  TRITON_BUILD_DIR="$(ls -d "$TRITON_SOURCE_DIR"/build/cmake* 2>/dev/null | sort | tail -1)"
fi
export TRITON_BUILD_DIR
if [ -z "$TRITON_BUILD_DIR" ] || [ ! -d "$TRITON_BUILD_DIR" ]; then
  echo "error: no Triton CMake build dir under $TRITON_SOURCE_DIR/build/cmake* --" \
       "build Triton first (scripts/build_triton.sh)." >&2
  exit 1
fi

# Resolve the LLVM/MLIR install exactly like python/setup.py:locate_llvm() so the
# plugin links the SAME LLVM Triton was built against: prefer the Triton build's
# CMakeCache (LLVM_SYSPATH, else LLVM_DIR=<root>/lib/cmake/llvm -> <root>), else
# the newest ~/.triton/llvm/llvm-* cache. No hardcoded hash/path.
if [ -z "$LLVM_INSTALL_DIR" ] && [ -f "$TRITON_BUILD_DIR/CMakeCache.txt" ]; then
  LLVM_INSTALL_DIR="$(grep -E '^LLVM_SYSPATH(:[^=]*)?=' "$TRITON_BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2-)"
  if [ -z "$LLVM_INSTALL_DIR" ]; then
    _llvm_dir="$(grep -E '^LLVM_DIR(:[^=]*)?=' "$TRITON_BUILD_DIR/CMakeCache.txt" | head -1 | cut -d= -f2-)"
    [ -n "$_llvm_dir" ] && LLVM_INSTALL_DIR="$(cd "$_llvm_dir/../../.." 2>/dev/null && pwd)"
  fi
fi
if [ -z "$LLVM_INSTALL_DIR" ]; then
  LLVM_INSTALL_DIR="$(ls -d "$HOME"/.triton/llvm/llvm-* 2>/dev/null | sort | tail -1)"
fi
export LLVM_INSTALL_DIR
if [ -z "$LLVM_INSTALL_DIR" ] || [ ! -d "$LLVM_INSTALL_DIR" ]; then
  echo "error: could not resolve LLVM_INSTALL_DIR (not in $TRITON_BUILD_DIR/CMakeCache.txt" \
       "nor ~/.triton/llvm/llvm-*); set LLVM_INSTALL_DIR explicitly." >&2
  exit 1
fi

BUILD="$HERE/build"
EXTRA_ARGS="$*"

# Build the full plugin (dialects + TTGPU/LLVM conversions for NVIDIA + AMD +
# SIMT). There are no per-pass/per-backend toggles; callers can still override any
# CMake variable via positional args (e.g. -DTRITON_USE_MACA=ON).
cmake -G Ninja -B "$BUILD" -S "$HERE" \
  -DCMAKE_BUILD_TYPE=Release \
  -DTRITON_SOURCE_DIR="$TRITON_SOURCE_DIR" \
  -DTRITON_BUILD_DIR="$TRITON_BUILD_DIR" \
  -DLLVM_INSTALL_DIR="$LLVM_INSTALL_DIR" \
  $EXTRA_ARGS
cmake --build "$BUILD" --target triton_dist -j 64
# Companion python module: now hosts ALL distributed builder ops (not just
# extern_call), so its build is mandatory -- a failure here means the frontend
# has no way to emit distributed ops on 3.7.1.
cmake --build "$BUILD" --target triton_dist_ext -j 64
echo "=== built ==="
ls -la "$BUILD"/lib*triton_dist* "$BUILD"/*.so 2>/dev/null
find "$BUILD" -name "libtriton_dist.so" -o -name "libtriton_dist_ext*.so" 2>/dev/null

# Deploy into the package's installed layout (triton_dist/lib). _plugin.py
# resolves this location *before* plugin/build, so a stale copy here would
# silently shadow a fresh build -- always overwrite it to keep the two in sync.
DEPLOY_DIR="$HERE/../python/triton_dist/lib"
mkdir -p "$DEPLOY_DIR"
cp -f "$BUILD"/libtriton_dist.so "$DEPLOY_DIR"/
cp -f "$BUILD"/libtriton_dist_ext*.so "$DEPLOY_DIR"/
echo "=== deployed to $DEPLOY_DIR ==="
ls -l "$DEPLOY_DIR"/*.so
