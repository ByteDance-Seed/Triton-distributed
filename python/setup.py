################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

################################################################################
# Copyright 2025 ByteDance Ltd. and/or its affiliates.
#
# Triton-distributed packaging — out-of-tree Triton plugin model.
#
# This setup builds the distributed extension as out-of-tree artifacts bundled
# into the ``triton_dist`` package:
#
#   * libtriton_dist.so          — the Triton plugin (Distributed + SIMT dialects
#                                   + conversion passes), discovered at runtime via
#                                   TRITON_PLUGIN_PATHS. NOTE: on Triton 3.7.1 the
#                                   op builders are exposed via the companion module
#                                   below (not OpInfo); see plugin/lib/Plugin.cpp.
#   * libtriton_dist_ext*.so      — the companion pybind11 module hosting all
#                                   distributed op builders (incl. extern_call).
#   * libtriton_distributed.so    — OPTIONAL (USE_TRITON_DISTRIBUTED_AOT=1): a
#                                   standalone torch/CUDA/pybind module carrying
#                                   the hand-written MoE CUDA ops + AOT-compiled
#                                   Triton kernels. It has NO Triton-compiler
#                                   dependency.
#
# The plugin can only load into a *patched* upstream Triton (built with
# TRITON_EXT_ENABLED=1 + plugin/patches/*.patch + the libstdc++ version-script); a
# stock PyPI wheel is hidden-visibility and ships no C++ headers. So by default the
#   pip install ./python            (or: pip install -e ./python)
# flow first builds+installs that Triton from the pinned 3rdparty/triton submodule
# (ensure_patched_triton -> scripts/build_triton.sh) and then compiles+bundles the
# extension against it — no separate manual step. Opt out with
# TRITON_DIST_SKIP_TRITON_BUILD=1 (e.g. release packaging builds the Triton wheel
# once up-front and reuses it; see scripts/build_scm.sh).
#
# Triton is otherwise located automatically from the installed package (editable /
# source-backed installs that expose C++ headers + a CMake build dir). Override via
# TRITON_INSTALL_DIR or TRITON_SOURCE_DIR(+TRITON_BUILD_DIR); LLVM via
# LLVM_INSTALL_DIR (else derived from Triton's build CMakeCache).
################################################################################
import glob
import os
import shutil
import subprocess
import sys
from pathlib import Path

from setuptools import Command, Extension, setup
from setuptools.command.build_ext import build_ext
from distutils.command.clean import clean

from build_helpers import get_base_dir


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def check_env_flag(name: str, default: str = "") -> bool:
    return os.getenv(name, default).upper() in ["ON", "1", "YES", "TRUE", "Y"]


def _is_cuda_platform() -> bool:
    return shutil.which("nvidia-smi") is not None


def _is_hip_platform() -> bool:
    return shutil.which("rocm-smi") is not None


def _is_maca_platform() -> bool:
    return shutil.which("mx-smi") is not None


def get_build_type() -> str:
    if check_env_flag("DEBUG"):
        return "Debug"
    if check_env_flag("REL_WITH_DEB_INFO"):
        return "RelWithDebInfo"
    return "Release"


# --------------------------------------------------------------------------- #
# Build the patched (EXT-enabled) upstream Triton the plugin binds to
# --------------------------------------------------------------------------- #
def ensure_patched_triton():
    """Build+install the patched upstream Triton this plugin binds to.

    triton_dist is an out-of-tree plugin, but it can only load into a Triton that
    was built with default symbol visibility (``TRITON_EXT_ENABLED=1``) plus the
    minimal op-agnostic source patches (``plugin/patches/*.patch``) and the
    libstdc++ version-script. A stock PyPI ``triton`` wheel is hidden-visibility
    (cannot host the plugin) and ships no C++ headers (cannot be compiled
    against), so by default we build that Triton from the pinned
    ``3rdparty/triton`` submodule here — i.e. ``pip install ./python`` produces a
    matching Triton + the plugin in one go.

    Skipped when:
      * ``TRITON_DIST_SKIP_TRITON_BUILD`` is set (release packaging builds the
        Triton *wheel* once up-front, then reuses it — see scripts/build_scm.sh);
      * ``TRITON_SOURCE_DIR`` / ``TRITON_INSTALL_DIR`` is set (the caller manages
        Triton explicitly and only wants the plugin compiled against it).
    """
    if check_env_flag("TRITON_DIST_SKIP_TRITON_BUILD"):
        print("-- TRITON_DIST_SKIP_TRITON_BUILD set: not building patched Triton")
        return
    if os.environ.get("TRITON_SOURCE_DIR") or os.environ.get("TRITON_INSTALL_DIR"):
        print("-- TRITON_SOURCE_DIR/TRITON_INSTALL_DIR set: not building patched "
              "Triton (compiling the plugin against the caller-provided tree)")
        return
    script = os.path.join(get_base_dir(), "scripts", "build_triton.sh")
    if not os.path.isfile(script):
        raise RuntimeError(f"Cannot build the patched Triton: {script} not found. Either restore "
                           "it or set TRITON_DIST_SKIP_TRITON_BUILD=1 with a pre-installed "
                           "EXT-enabled Triton (and TRITON_SOURCE_DIR to compile against).")
    print(f"-- Building patched Triton via {script} (set "
          "TRITON_DIST_SKIP_TRITON_BUILD=1 to reuse an existing one)")
    subprocess.check_call(["bash", script])

    # The nested `pip install -e` above makes Triton importable *on disk*, but the
    # editable .pth is NOT picked up by this already-running interpreter, so a
    # later `import triton` in locate_triton() would raise ModuleNotFoundError.
    # This is exactly the single-command path (`pip install ./python`, and the CI
    # `pip install -e python[...]` step) — as opposed to the two-step dev/CI flow
    # that installs Triton in a separate process first. Point locate_triton() at
    # the source tree we just built so it never needs to import Triton in-process.
    triton_src = os.path.join(get_base_dir(), "3rdparty", "triton")
    if os.path.isdir(triton_src):
        os.environ.setdefault("TRITON_SOURCE_DIR", triton_src)
        if not os.environ.get("TRITON_BUILD_DIR"):
            builds = sorted(glob.glob(os.path.join(triton_src, "build", "cmake*")))
            if builds:
                os.environ["TRITON_BUILD_DIR"] = builds[-1]


# --------------------------------------------------------------------------- #
# Locate the installed Triton + LLVM to build the plugin against
# --------------------------------------------------------------------------- #
def locate_triton() -> dict:
    """Resolve how to build the plugin against an *installed* Triton.

    Returns a dict with either ``install_dir`` (mode 1) or ``source_dir`` (+
    optional ``build_dir``, mode 2), matching plugin/CMakeLists.txt's two modes.

    Resolution order:
      1. Explicit env: TRITON_INSTALL_DIR, or TRITON_SOURCE_DIR (+TRITON_BUILD_DIR).
      2. Derived from ``import triton`` — an editable/source-backed install
         exposes the Triton source tree (headers) and its CMake build dir (for
         the tablegen'd ``*.inc``). A plain wheel without dev headers cannot be
         used to *compile* the plugin (it has no headers / generated includes);
         in that case set the env vars above to point at a dev Triton.
    """
    install_dir = os.environ.get("TRITON_INSTALL_DIR")
    source_dir = os.environ.get("TRITON_SOURCE_DIR")
    build_dir = os.environ.get("TRITON_BUILD_DIR")

    if install_dir or source_dir:
        # Compiling against a Triton *source* tree needs its tablegen'd *.inc
        # headers, which live in the CMake build dir (not include/). Callers that
        # set TRITON_SOURCE_DIR but not TRITON_BUILD_DIR (e.g. a two-step CI
        # build that compiles Triton first) would otherwise leave
        # build_dir=None, so -DTRITON_BUILD_DIR is never passed and the plugin
        # fails to find e.g. AttrInterfaces.h.inc. Discover it the same way the
        # `import triton` branch below does.
        if source_dir and not build_dir:
            builds = sorted(glob.glob(str(Path(source_dir) / "build" / "cmake*")))
            build_dir = builds[-1] if builds else None
        return {"install_dir": install_dir, "source_dir": source_dir, "build_dir": build_dir}

    try:
        import triton  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Triton is not importable. Install it first (e.g. `pip install triton`) "
                           "or set TRITON_INSTALL_DIR / TRITON_SOURCE_DIR(+TRITON_BUILD_DIR).") from exc

    pkg = Path(triton.__file__).resolve().parent  # <...>/python/triton
    # editable / source-backed install: <src>/python/triton -> <src>
    cand_src = pkg.parent.parent
    if (cand_src / "include").is_dir() and (pkg / "_C").is_dir():
        builds = sorted(glob.glob(str(cand_src / "build" / "cmake*")))
        return {
            "install_dir": None,
            "source_dir": str(cand_src),
            "build_dir": build_dir or (builds[-1] if builds else None),
        }
    # installed tree that ships headers next to the package
    if (pkg.parent / "include").is_dir():
        return {"install_dir": str(pkg.parent), "source_dir": None, "build_dir": None}

    raise RuntimeError("The installed Triton does not expose C++ headers, so the plugin cannot be "
                       "compiled against it. Point TRITON_SOURCE_DIR(+TRITON_BUILD_DIR) at a Triton "
                       "source checkout built with TRITON_EXT_ENABLED=ON.")


def locate_llvm(build_dir: str = None) -> str:
    """Resolve the LLVM/MLIR install dir used to build the plugin.

    Order: LLVM_INSTALL_DIR env -> Triton build CMakeCache (LLVM_SYSPATH /
    LLVM_DIR) -> newest ~/.triton/llvm/llvm-* cache.
    """
    env = os.environ.get("LLVM_INSTALL_DIR")
    if env:
        return env
    if build_dir:
        cache = Path(build_dir) / "CMakeCache.txt"
        if cache.is_file():
            for line in cache.read_text().splitlines():
                if line.startswith("LLVM_SYSPATH:"):
                    return line.split("=", 1)[1].strip()
                if line.startswith("LLVM_DIR:"):
                    # <root>/lib/cmake/llvm
                    return str(Path(line.split("=", 1)[1].strip()).parents[2])
    cands = sorted(glob.glob(os.path.expanduser("~/.triton/llvm/llvm-*")))
    if cands:
        return cands[-1]
    raise RuntimeError("Could not resolve LLVM; set LLVM_INSTALL_DIR.")


# --------------------------------------------------------------------------- #
# Optional SHMEM backends (HIP / MACA only; NVIDIA uses pip nvshmem wheels)
# --------------------------------------------------------------------------- #
def build_shmem():
    if not (_is_hip_platform() or _is_maca_platform()):
        return  # NVIDIA: nvshmem provided via pip (see DEPS_NVIDIA)
    base = get_base_dir()
    backend = os.getenv("TRITON_DIST_SHMEM_BACKEND", "").lower()
    if _is_hip_platform():
        if backend in ("", "mori_shmem"):
            subprocess.check_call(["bash", os.path.join(base, "scripts", "build_mori_shmem.sh")])
        if backend in ("", "rocshmem"):
            subprocess.check_call(
                ["bash", os.path.join(base, "shmem", "rocshmem_bind", "build.sh"), "--arch", "gfx942"])
    if _is_maca_platform():
        subprocess.check_call(["bash", os.path.join(base, "shmem", "mxshmem_bind", "build.sh")])


class SHMEMBuildOnly(Command):
    description = "Build SHMEM backend bindings only"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        build_shmem()


# --------------------------------------------------------------------------- #
# CMake-driven build of the plugin (+ optional csrc) via setuptools build_ext
# --------------------------------------------------------------------------- #
class CMakeExtension(Extension):
    """A CMake project to configure+build. ``cmake_dir`` is the source dir; the
    produced shared objects are staged into ``dest_pkg_subdir`` of triton_dist."""

    def __init__(self, name: str, cmake_dir: str, dest_pkg_subdir: str):
        super().__init__(name, sources=[])
        self.cmake_dir = os.path.abspath(cmake_dir)
        self.dest_pkg_subdir = dest_pkg_subdir


class CMakeBuild(build_ext):

    def run(self):
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as exc:
            raise RuntimeError("CMake is required to build triton_dist") from exc
        # Build+install the patched Triton first so the plugin compiles/loads
        # against it (no-op when the caller opts out; see ensure_patched_triton).
        ensure_patched_triton()
        build_shmem()
        for ext in self.extensions:
            self._build_cmake(ext)

    def _stage(self, built_so: str, dest_subdir: str):
        """Copy a built .so into both the source tree (for editable installs and
        triton_dist._plugin.find_plugin) and the wheel build dir."""
        name = os.path.basename(built_so)
        dests = [os.path.join(get_base_dir(), "python", "triton_dist", dest_subdir)]
        if self.build_lib:
            dests.append(os.path.join(self.build_lib, "triton_dist", dest_subdir))
        for d in dests:
            os.makedirs(d, exist_ok=True)
            shutil.copy2(built_so, os.path.join(d, name))
            print(f"-- staged {name} -> {d}")

    def _build_cmake(self, ext: CMakeExtension):
        triton = locate_triton()
        llvm = locate_llvm(triton.get("build_dir"))
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)

        cmake_args = [
            f"-DCMAKE_BUILD_TYPE={get_build_type()}",
            f"-DLLVM_INSTALL_DIR={llvm}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if triton.get("install_dir"):
            cmake_args.append(f"-DTRITON_INSTALL_DIR={triton['install_dir']}")
        else:
            cmake_args.append(f"-DTRITON_SOURCE_DIR={triton['source_dir']}")
            if triton.get("build_dir"):
                cmake_args.append(f"-DTRITON_BUILD_DIR={triton['build_dir']}")

        # Per-extension knobs.
        if ext.name == "triton_dist_plugin":
            # The plugin always builds the full dialects + TTGPU/LLVM conversions
            # (NVIDIA + AMD + SIMT), like the old monolithic build and libtriton.so
            # itself -- no per-pass/per-backend toggles. METAX (MACA) is the only
            # switch, since it replaces the NVIDIA/AMD toolchain.
            cmake_args.append("-DTRITON_DIST_BUILD_PY_EXT=ON")
            if check_env_flag("TRITON_USE_MACA"):
                cmake_args.append("-DTRITON_USE_MACA=ON")
        elif ext.name == "triton_dist_csrc":
            cmake_args.append("-DUSE_TRITON_DISTRIBUTED_AOT=ON")

        extra = os.getenv("TRITON_DIST_APPEND_CMAKE_ARGS")
        if extra:
            cmake_args += extra.split()

        subprocess.check_call(["cmake", ext.cmake_dir] + cmake_args, cwd=build_temp)
        build_args = ["--config", get_build_type(), "-j", os.getenv("MAX_JOBS", str(os.cpu_count() or 8))]
        subprocess.check_call(["cmake", "--build", "."] + build_args, cwd=build_temp)

        # Plugin: libtriton_dist.so + companion libtriton_dist_ext*.so.
        # csrc (AOT): libtriton_distributed.so + its libtriton_distributed_kernel.so
        # (linked via $ORIGIN rpath, so both must land in the same dir).
        for pattern in ("libtriton_dist.so", "libtriton_dist_ext*.so", "libtriton_distributed.so",
                        "libtriton_distributed_kernel.so"):
            for so in glob.glob(os.path.join(build_temp, "**", pattern), recursive=True):
                self._stage(so, ext.dest_pkg_subdir)


class CMakeClean(clean):

    def initialize_options(self):
        clean.initialize_options(self)
        self.build_temp = os.path.join(get_base_dir(), "python", "build")


# --------------------------------------------------------------------------- #
# Packages / package data
# --------------------------------------------------------------------------- #
def get_packages():
    candidates = [
        "triton_dist",
        "triton_dist/_C",
        "triton_dist/benchmark",
        "triton_dist/kernels",
        "triton_dist/kernels/nvidia",
        "triton_dist/kernels/amd",
        "triton_dist/kernels/metax",
        "triton_dist/language",
        "triton_dist/language/extra",
        "triton_dist/language/extra/cuda",
        "triton_dist/language/extra/hip",
        "triton_dist/language/extra/maca",
        "triton_dist/mega_triton_kernel",
        "triton_dist/function",
        "triton_dist/function/nvidia",
        "triton_dist/layers",
        "triton_dist/layers/nvidia",
        "triton_dist/layers/amd",
        "triton_dist/models",
        "triton_dist/test",
        "triton_dist/tools",
        "triton_dist/tools/compile",
        "triton_dist/tools/profiler",
        "triton_dist/tools/runtime",
        "triton_dist/tools/tune",
    ]
    if check_env_flag("TRITON_BUILD_LITTLE_KERNEL", "ON"):
        candidates += [
            p for p in (
                "little_kernel",
                "little_kernel/atom",
                "little_kernel/atom/barrier",
                "little_kernel/atom/mma",
                "little_kernel/atom/tma",
                "little_kernel/codegen",
                "little_kernel/codegen/registries",
                "little_kernel/codegen/special_struct",
                "little_kernel/codegen/visitors",
                "little_kernel/core",
                "little_kernel/core/passes",
                "little_kernel/core/passes/utils",
                "little_kernel/core/passes/utils/registries",
                "little_kernel/core/passes/utils/type_inference",
                "little_kernel/language",
                "little_kernel/language/intrin",
                "little_kernel/runtime",
                "little_kernel/benchmark",
                "little_kernel/benchmark/gemm_sm90",
                "little_kernel/benchmark/gemm_sm100",
                "little_kernel/benchmark/compute",
                "little_kernel/benchmark/memory",
                "little_kernel/benchmark/latency",
                "little_kernel/benchmark/warp",
                "little_kernel/benchmark/sm",
                "little_kernel/benchmark/sm90",
            )
        ]
    return [p for p in candidates if os.path.isdir(p)]


package_data = {
    # Bundle the out-of-tree plugin + companion + (optional) AOT module that the
    # CMake build stages into the package. find_plugin() probes triton_dist/lib
    # first for the installed layout. Empty when nothing has been built yet.
    "triton_dist": ["lib/*.so*", "_C/*.so*"],
    "": ["*.so*", "*.a"],
}


def get_git_version_suffix():
    try:
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]).strip().decode()
        if branch.startswith("release"):
            return ""
        sha = subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"]).strip().decode()
        return f"+git{sha}"
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# Runtime dependencies
# --------------------------------------------------------------------------- #
# triton_dist is an out-of-tree plugin *for* Triton: it requires a matching
# installed Triton (see plugin/TRITON_PIN.md for the validated upstream commit).
#
# CRITICAL: when this build self-builds the patched Triton (ensure_patched_triton,
# the default `pip install ./python` path) we must NOT also declare `triton` as a
# pip dependency. Otherwise the outer pip resolves it against the index, downloads
# a *stock* PyPI Triton, and installs it *after* our build step -- clobbering the
# patched (EXT-enabled, plugin-hosting) editable Triton with a hidden-visibility
# one that cannot load the plugin (import triton -> passes.plugin.* missing at
# runtime). ensure_patched_triton installs a satisfying Triton out-of-band, so the
# dependency is redundant there anyway.
#
# For a *published* / skip-build wheel we DO pin the exact co-shipped Triton (stock
# PyPI 3.7.1 is hidden-visibility and cannot host the plugin): release packaging
# sets TRITON_DIST_TRITON_REQUIREMENT="triton==<patched local version>" (e.g.
# 3.7.1+tritondist...); see scripts/build_scm.sh. Mirror ensure_patched_triton's
# skip conditions exactly so the two decisions never disagree.
_self_builds_triton = not (check_env_flag("TRITON_DIST_SKIP_TRITON_BUILD") or os.environ.get("TRITON_SOURCE_DIR")
                           or os.environ.get("TRITON_INSTALL_DIR"))
TRITON_DEP = [] if _self_builds_triton else [os.environ.get("TRITON_DIST_TRITON_REQUIREMENT", "triton>=3.7.1")]

DEPS_NVIDIA = [
    "cuda.core==1.0.1",
    "cuda-python>=12.0",
    "nvidia-nvshmem-cu12==3.6.5",
    "Cython>=0.29.24",
    "nvshmem4py-cu12==0.3.0",
] if _is_cuda_platform() else []
DEPS_HIP = ["hip-python"] if _is_hip_platform() else []
DEPS_TEST = ["nvidia-ml-py>=12.0"] if _is_cuda_platform() else []

ext_modules = [
    CMakeExtension("triton_dist_plugin", os.path.join(get_base_dir(), "plugin"), "lib"),
]
if check_env_flag("USE_TRITON_DISTRIBUTED_AOT"):
    # Standalone torch/CUDA/pybind module (MoE CUDA ops + AOT kernels). Imported
    # as triton_dist._C.libtriton_distributed only when AOT is enabled.
    ext_modules.append(CMakeExtension("triton_dist_csrc", os.path.join(get_base_dir(), "csrc"), "_C"))

setup(
    name=os.environ.get("TRITON_WHEEL_NAME", "triton_dist"),
    # Aligned with the pinned upstream Triton (3.7.1) this plugin builds against.
    version="3.7.1" + get_git_version_suffix() + os.environ.get("TRITON_WHEEL_VERSION_SUFFIX", ""),
    author="ByteDance Seed",
    author_email="zheng.size@bytedance.com",
    description="Triton language and compiler extension for distributed deep learning systems "
    "(out-of-tree Triton plugin).",
    long_description="",
    install_requires=["setuptools>=40.8.0", "packaging"] + TRITON_DEP + DEPS_NVIDIA + DEPS_HIP,
    packages=get_packages(),
    package_data=package_data,
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": CMakeBuild,
        "clean": CMakeClean,
        "build_shmem": SHMEMBuildOnly,
    },
    zip_safe=False,
    keywords=["Compiler", "Deep Learning", "Overlapping", "Distributed"],
    url="https://github.com/ByteDance-Seed/Triton-distributed",
    python_requires=">=3.9,<3.14",
    extras_require={
        "build": ["cmake>=3.20,<4.0", "lit", "ninja", "pybind11"],
        "tests": [
            "autopep8",
            "isort",
            "numpy",
            "pytest",
            "pytest-forked",
            "pytest-xdist",
            "scipy>=1.7.1",
            "llnl-hatchet",
            "transformers",
            "tqdm",
        ] + DEPS_TEST,
        "tutorials": ["matplotlib", "pandas", "tabulate", "chardet"],
    },
)
