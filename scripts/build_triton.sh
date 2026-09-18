#!/bin/bash
# Build the upstream Triton that Triton-distributed's out-of-tree plugin loads
# against. The 3rdparty/triton submodule tracks github.com/triton-lang/triton
# (NOT a fork). Customizations are kept to the absolute minimum:
#   1. TRITON_EXT_ENABLED=1  -> build libtriton.so with default visibility so the
#      plugin can resolve Triton/MLIR/LLVM symbols at load time (upstream feature,
#      PR #9922). Stock pip wheels are hidden-visibility and cannot host plugins.
#   2. a linker version-script (triton_hide_libstdcxx.map) -> localize libtriton's
#      incidental libstdc++ symbols so they cannot interpose torch's incompatible
#      ones (see that file's header for the full rationale).
#   3. Minimal, op-agnostic source patches under plugin/patches/*.patch, applied
#      in lexical order. Each adds a generic extension hook that genuinely cannot
#      be expressed via the plugin mechanism (triton references no distributed
#      types); see docs/refactor/INTRUSIVE_REMOVAL_MAP.md and the plugin-side
#      consumers:
#        0001-axisinfo-visitor-hook        -> AxisInfo visitor registration. The
#          built-in Coalesce / load-store vectorization passes build
#          ModuleAxisInfoAnalysis with NO callback, so there is no out-of-tree way
#          to teach AxisInfo that the distributed pointer-preserving ops (symm_at /
#          consume_token) keep their operand's alignment/contiguity; without it
#          those loads/stores are not vectorized (a real perf regression).
#          Consumer: plugin/lib/AxisInfoVisitors.cpp.
#        0002-pipeliner-predicate-hook     -> software-pipeliner predication
#          registration. triton::predicateOp hard-codes the built-in ops and
#          report_fatal_error()s on any other registered op, so a distributed.wait
#          moved into a pipelined loop's prologue/epilogue would crash; the hook
#          lets the plugin predicate it (set its optional `pred` operand).
#          Consumer: plugin/lib/PipelinerPredication.cpp.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRITON_DIR="${PROJECT_ROOT}/3rdparty/triton"

# Pinned upstream commit (Triton 3.7.1 release tag v3.7.1). Recorded here for
# traceability; the 3rdparty/triton submodule gitlink should point at this same
# commit.
TRITON_PIN="f797708c0626e5f9840ca5b0a98790e2c7cb09ad"

# Ensure the submodule is checked out at the pinned commit.
if [ ! -e "${TRITON_DIR}/CMakeLists.txt" ]; then
  git -C "${PROJECT_ROOT}" submodule update --init 3rdparty/triton
fi
CUR="$(git -C "${TRITON_DIR}" rev-parse HEAD 2>/dev/null || echo none)"
if [ "${CUR}" != "${TRITON_PIN}" ]; then
  echo "[build_triton] checking out pinned upstream ${TRITON_PIN} (was ${CUR})"
  git -C "${TRITON_DIR}" fetch --no-tags origin "${TRITON_PIN}" 2>/dev/null || \
    git -C "${TRITON_DIR}" fetch --no-tags origin
  git -C "${TRITON_DIR}" checkout --quiet "${TRITON_PIN}"
fi

# Apply the minimal source patches (see header, item 3) idempotently and in
# lexical order: for each, skip if already applied (reverse-check), apply if it
# applies cleanly, warn otherwise.
_PATCH_DIR="${PROJECT_ROOT}/plugin/patches"
if [ -d "${_PATCH_DIR}" ]; then
  for _patch in "${_PATCH_DIR}"/*.patch; do
    [ -e "${_patch}" ] || continue  # no patches present
    _name="$(basename "${_patch}")"
    if git -C "${TRITON_DIR}" apply --reverse --check "${_patch}" 2>/dev/null; then
      echo "[build_triton] patch already applied: ${_name}"
    elif git -C "${TRITON_DIR}" apply --check "${_patch}" 2>/dev/null; then
      echo "[build_triton] applying patch: ${_name}"
      git -C "${TRITON_DIR}" apply "${_patch}"
    else
      echo "[build_triton] WARNING: ${_name} neither applies cleanly nor is" \
           "already applied; the corresponding hook may be missing." >&2
    fi
  done
else
  echo "[build_triton] WARNING: ${_PATCH_DIR} not found; source hooks will be missing." >&2
fi

export TRITON_EXT_ENABLED=1
_TRITON_HIDE_MAP="${SCRIPT_DIR}/triton_hide_libstdcxx.map"
export LDFLAGS="-Wl,--version-script=${_TRITON_HIDE_MAP} ${LDFLAGS}"

# Triton builds with -Werror. With TRITON_EXT_ENABLED=1 the -fvisibility=hidden
# default is dropped, which surfaces a -Wattributes diagnostic in 3.7.1's
# python/src/gluon_ir.cc (the `GluonLayouts` struct has greater visibility than
# its pybind11 `py::handle` fields). This is benign for a host library and is a
# pure source-cleanliness issue upstream fixed later. Disable just that warning
# via CXXFLAGS (read by CMake into CMAKE_CXX_FLAGS *before* triton appends
# -Werror, so the now-disabled warning is never escalated to an error). No
# Triton source is patched.
export CXXFLAGS="-Wno-attributes ${CXXFLAGS}"

# Local version label so the patched wheel is distinguishable from a stock PyPI
# `triton` of the same 3.7.1 (which is hidden-visibility and cannot host the
# plugin), letting triton_dist pin to exactly this build. Triton appends this to
# its version (see 3rdparty/triton/setup.py get_triton_version_suffix).
export TRITON_WHEEL_VERSION_SUFFIX="${TRITON_WHEEL_VERSION_SUFFIX:-+tritondist}"

# Triton 3.7.x drives the build from the repo root setup.py (older layouts used
# python/). Pick whichever exists so this stays robust across version bumps.
if [ -f "${TRITON_DIR}/setup.py" ]; then
  _SETUP_DIR="${TRITON_DIR}"
else
  _SETUP_DIR="${TRITON_DIR}/python"
fi

# Two modes:
#   editable (default) -> `pip install -e` the patched Triton into the current
#     env. Used by the dev flow and by setup.py's ensure_patched_triton().
#   wheel              -> `bdist_wheel` a redistributable patched Triton (left in
#     <triton>/dist/, and additionally copied to TRITON_DIST_TRITON_WHEEL_OUT if
#     set). Used by release packaging (scripts/build_scm.sh), which extracts it
#     into output/python next to triton_dist.
_MODE="${TRITON_DIST_TRITON_INSTALL_MODE:-editable}"
pushd "${_SETUP_DIR}"
if [ "${_MODE}" = "wheel" ]; then
  MAX_JOBS="${MAX_JOBS:-126}" python3 setup.py bdist_wheel
  if [ -n "${TRITON_DIST_TRITON_WHEEL_OUT}" ]; then
    mkdir -p "${TRITON_DIST_TRITON_WHEEL_OUT}"
    cp -f dist/triton-*.whl "${TRITON_DIST_TRITON_WHEEL_OUT}/"
    echo "[build_triton] patched Triton wheel -> ${TRITON_DIST_TRITON_WHEEL_OUT}"
  fi
  ls -l dist/triton-*.whl
  # `setup.py bdist_wheel` does NOT place libtriton.so inplace, so the source
  # tree's python/triton/_C/ stays empty. 
  _tdw_whl="$(ls -t dist/triton-*.whl | head -1)"
  python3 - "${_tdw_whl}" "${TRITON_DIR}" <<'PY'
import os, sys, zipfile
whl, root = sys.argv[1], sys.argv[2]
src_c = os.path.join(root, "python", "triton", "_C")
os.makedirs(src_c, exist_ok=True)
staged = 0
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        base = os.path.basename(n)
        if "_C/" in n and base.startswith("libtriton") and base.endswith(".so"):
            with open(os.path.join(src_c, base), "wb") as f:
                f.write(z.read(n))
            print(f"[build_triton] staged {n} -> {os.path.join(src_c, base)}")
            staged += 1
if not staged:
    sys.stderr.write(f"[build_triton] WARNING: no libtriton*.so found in {whl} to stage\n")
PY
else
  # Editable install into the current environment. Any pre-existing Triton must
  # be removed first or it shadows our patched build at import time (import triton
  # -> the wrong one, so the out-of-tree plugin never loads and
  # passes.plugin.* is missing at runtime):
  #   * a same-named stock `triton` wheel: pip's editable install replaces it, but
  #     we uninstall explicitly so a leftover physical dir can't linger;
  #   * NGC's `pytorch-triton` dist: ships a *physical* dist-packages/triton dir
  #     under a DIFFERENT dist name, so `pip install -e` (dist `triton`) does not
  #     touch it -- it must be uninstalled by name.
  pip3 uninstall -y triton pytorch-triton triton-nightly 2>/dev/null || true
  # Belt-and-suspenders: drop any physical triton package dir a broken/manifest-
  # less install may have left behind (a leftover dir shadows the editable .pth).
  python3 - <<'PY'
import os, shutil, site
cands = []
try:
    cands += list(site.getsitepackages())
except Exception:
    pass
try:
    cands.append(site.getusersitepackages())
except Exception:
    pass
for d in cands:
    p = os.path.join(d, "triton")
    if os.path.isdir(p) and not os.path.islink(p):
        shutil.rmtree(p, ignore_errors=True)
PY
  MAX_JOBS="${MAX_JOBS:-126}" pip3 install -e . --verbose --no-build-isolation
  # Verify the patched build is now THE importable Triton (a fresh interpreter so
  # the editable .pth is honored). Fail loud rather than let a shadowing stock
  # Triton silently break plugin loading downstream.
  python3 - "${TRITON_DIR}" <<'PY'
import os, sys
import triton
real = os.path.realpath(triton.__file__)
want = os.path.realpath(sys.argv[1])
if os.path.commonpath([real, want]) != want:
    sys.stderr.write(
        f"[build_triton] ERROR: 'import triton' -> {real} (v{triton.__version__}), not the "
        f"patched tree under {want}. A stock Triton is shadowing the patched build.\n")
    sys.exit(1)
print(f"[build_triton] verified patched Triton: {triton.__version__} @ {real}")
PY
  # Relax torch's now-stale exact Triton pin. Some torch wheels -- notably ROCm
  # builds (e.g. 2.8.0+rocm hard-pins `Requires-Dist: triton==3.4.0+rocm...`) but
  # also some CUDA/NGC ones -- declare an exact dependency on the Triton they
  # shipped with. We just replaced that Triton with the EXT-enabled build the
  # plugin binds to, so that pin is now unsatisfiable. If left in place, the next
  # unpinned `pip install` (e.g. deepspeed/accelerate in build_e2e_env.sh) sees
  # torch's dependency as broken and "repairs" torch by pulling a stock PyPI
  # wheel -- on ROCm that is a *CUDA* torch (+nvidia-cu* stack) that then dies at
  # runtime with "Found no NVIDIA driver on your system". triton_dist manages
  # Triton out-of-band (see setup.py: TRITON_DEP is empty when self-building), so
  # torch must not force a specific Triton. Drop the stale Requires-Dist so the
  # environment stays internally consistent and torch is never silently swapped.
  python3 - <<'PY'
import glob, os, re, site
mds = set()
dirs = list(site.getsitepackages())
try:
    dirs.append(site.getusersitepackages())
except Exception:
    pass
for d in dirs:
    mds.update(glob.glob(os.path.join(d, "torch-*.dist-info", "METADATA")))
_pat = re.compile(r"(?i)^Requires-Dist:\s*(pytorch-)?triton(-nightly)?\b")
for md in mds:
    try:
        lines = open(md).read().splitlines(keepends=True)
    except OSError:
        continue
    kept = [ln for ln in lines if not _pat.match(ln)]
    if len(kept) != len(lines):
        open(md, "w").writelines(kept)
        print(f"[build_triton] relaxed stale Triton pin in {md} "
              f"({len(lines) - len(kept)} line(s))")
PY
fi
popd
