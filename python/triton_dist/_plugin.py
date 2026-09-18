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
"""Locate and register the Triton-distributed out-of-tree plugin
(``libtriton_dist.so``) on ``TRITON_PLUGIN_PATHS`` *before* Triton is imported.

Triton loads plugins exactly once, lazily, the first time ``libtriton`` is
initialised (it reads ``TRITON_PLUGIN_PATHS`` and ``dlopen``s each entry, after
which the registration is cached). Therefore this module must run, and the
environment variable must be set, before ``import triton``.

Resolution order for the plugin shared object:
  1. ``TRITON_DIST_PLUGIN_PATH`` env var (explicit override);
  2. installed next to this package (``triton_dist/lib/libtriton_dist.so``);
  3. the in-tree developer build (``<repo>/plugin/build/libtriton_dist.so``).
"""
import os
import sys
import warnings

_PLUGIN_SONAME = "libtriton_dist.so"
# Companion pybind11 module (the distributed create-op builders); a glob since the
# file carries the cpython ABI tag (e.g. libtriton_dist_ext.cpython-311-*.so).
_EXT_GLOB = "libtriton_dist_ext*.so"


def _candidate_paths():
    override = os.environ.get("TRITON_DIST_PLUGIN_PATH")
    if override:
        yield override

    here = os.path.dirname(os.path.abspath(__file__))
    # Installed layout: triton_dist/lib/libtriton_dist.so
    yield os.path.join(here, "lib", _PLUGIN_SONAME)
    # Developer layout: <repo>/plugin/build/libtriton_dist.so
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    yield os.path.join(repo_root, "plugin", "build", _PLUGIN_SONAME)


def find_plugin():
    for path in _candidate_paths():
        if path and os.path.isfile(path):
            return os.path.realpath(path)
    return None


def _ext_dirs():
    here = os.path.dirname(os.path.abspath(__file__))
    yield os.path.join(here, "lib")
    repo_root = os.path.abspath(os.path.join(here, os.pardir, os.pardir))
    yield os.path.join(repo_root, "plugin", "build")


def find_ext_module():
    import glob
    for d in _ext_dirs():
        hits = sorted(glob.glob(os.path.join(d, _EXT_GLOB)))
        if hits:
            return hits[0]
    return None


_EXT_MODULE = None


def _promote_plugin_to_global():
    """Make the already-loaded plugin's symbols globally visible.

    The companion module links ``libtriton`` (MLIR/LLVM/pybind11) via DT_NEEDED but
    leaves the Distributed/SIMT dialect op symbols (e.g.
    ``distributed::ExternCallOp::build`` and their TypeIDs) undefined, to be resolved
    at runtime from the plugin .so. Triton's ``loadPlugins`` ``dlopen``s the plugin
    with *local* scope, so those symbols are not in the global namespace and the
    companion's import fails with ``undefined symbol`` (load-order dependent: it may
    happen to work standalone but fails under more complex import graphs, e.g.
    torchrun).

    ``RTLD_NOLOAD | RTLD_GLOBAL`` re-opens the *already-resident* plugin and promotes
    its symbols to the global scope without reloading or re-resolving anything. This
    is the deterministic counterpart to relying on import order."""
    import ctypes
    plugin = find_plugin()
    if plugin is None:
        return
    try:
        ctypes.CDLL(plugin, mode=os.RTLD_GLOBAL | os.RTLD_NOLOAD)
    except OSError:
        # Not yet resident (unexpected after importing libtriton): load it global
        # but lazily so its own undefined MLIR symbols resolve from libtriton on use.
        ctypes.CDLL(plugin, mode=os.RTLD_GLOBAL | os.RTLD_LAZY)


def load_ext_module():
    """Import the companion extern_call module, caching the result.

    Importing ``triton._C.libtriton`` triggers Triton's one-time ``loadPlugins`` so
    the plugin is resident; we then promote its symbols to the global scope (see
    ``_promote_plugin_to_global``) so the companion's builders can resolve the
    Distributed/SIMT dialect op symbols + their TypeIDs from the plugin. Returns
    ``None`` if unavailable (distributed ops then raise a clear error on use)."""
    global _EXT_MODULE
    if _EXT_MODULE is not None:
        return _EXT_MODULE
    so = find_ext_module()
    if so is None:
        return None
    import importlib.util
    import triton._C.libtriton  # noqa: F401  (triggers loadPlugins -> plugin resident)
    _promote_plugin_to_global()
    spec = importlib.util.spec_from_file_location("libtriton_dist_ext", so)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _EXT_MODULE = mod
    return mod


def require_ext_module():
    """Return the companion module, raising a clear error if unavailable.

    All distributed builder ops are exposed by the companion (libtriton_dist_ext)
    rather than as OpInfo callbacks: on the pinned Triton 3.7.1 the plugin-op
    binding inserts no result slot and returns no Value, so it cannot drive the
    `operands[0]=result slot` builders. The companion applies that convention and
    returns the created Value (see plugin/lib/ext/py_module.cpp)."""
    ext = load_ext_module()
    if ext is None:
        raise RuntimeError("The Triton-distributed companion module (libtriton_dist_ext) is "
                           "required for distributed ops but could not be found. Build it via "
                           "`pip install ./python` (or `pip install -e ./python`).")
    return ext


def register_plugin():
    """Prepend the distributed plugin to ``TRITON_PLUGIN_PATHS``.

    Returns the resolved path, or ``None`` if the plugin could not be found.
    Safe to call multiple times (idempotent)."""
    path = find_plugin()

    # Warn only when the plugin is genuinely at risk of not being loaded: libtriton
    # is already imported (plugins loaded once at first import) AND this plugin is
    # not already on TRITON_PLUGIN_PATHS. When a harness exported TRITON_PLUGIN_PATHS
    # up front (the supported way), import order is irrelevant -- so stay quiet.
    if "triton._C.libtriton" in sys.modules:
        _on_path = path and path in [
            os.path.realpath(p) for p in os.environ.get("TRITON_PLUGIN_PATHS", "").split(":") if p
        ]
        if not _on_path:
            warnings.warn("triton._C.libtriton was imported before triton_dist; the "
                          "distributed plugin may not be registered. Import triton_dist "
                          "before triton, or set TRITON_PLUGIN_PATHS explicitly.")

    if path is None:
        warnings.warn(f"Could not locate {_PLUGIN_SONAME}. Build it via `pip install ./python` "
                      "(or `pip install -e ./python`), or set TRITON_DIST_PLUGIN_PATH. "
                      "Distributed ops/passes will be unavailable.")
        return None

    existing = os.environ.get("TRITON_PLUGIN_PATHS", "")
    entries = [p for p in existing.split(":") if p]
    # Compare by realpath so an already-present symlinked/relative entry for the same
    # file is not duplicated (``path`` is realpath'd by ``find_plugin``).
    if path not in [os.path.realpath(p) for p in entries]:
        entries.insert(0, path)
        os.environ["TRITON_PLUGIN_PATHS"] = ":".join(entries)
    return path
