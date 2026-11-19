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
"""Immutable build identity and atomic cubin cache publication (RF02).

A cached cubin is only reusable when *everything that can change the compiled
binary* is identical.  :class:`BuildIdentity` therefore fingerprints:

* the generated CUDA source content (full SHA-256, never its mtime);
* the **resolved** target architecture (``sm_86`` ...), never the unresolved
  literal ``"auto"`` -- an unresolved target must be resolved by
  :func:`little_kernel.backends.nv.common.runtime.target.resolve_target` before a key exists;
* the exact nvcc argv tokens -- in order, including duplicates and literal
  whitespace (``-DMODE=1`` vs ``-DMODE=2`` and reversed pairs cannot collide);
* the include *search precedence* and the *contents* of every reachable user
  header, whether included with quotes or angle brackets, by content hash and
  resolved path rather than mtime;
* user include paths / forced includes / response files carried inside the
  option list itself;
* a toolchain/compiler identity (nvcc path + version);
* a build-schema version so future changes to fixed compiler flags invalidate
  old entries.

Publication is atomic: every artifact is written to a temporary file in the
cache directory, ``fsync``-ed and then ``os.replace``-d onto its final name.
Concurrent readers that look for ``*.cubin`` can therefore only ever observe
"absent" or a complete cubin, never a partially written one.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Tuple, Union

from little_kernel.runtime.cache import (BuildIdentity as _SharedBuildIdentity, AtomicBinaryCache,
                                         UncacheableBuildError, normalize_options, normalize_include_paths,
                                         scan_user_dependencies, expand_option_dependencies, atomic_publish, _sha256)


@dataclass(frozen=True)
class BuildIdentity(_SharedBuildIdentity):
    toolchain_id: str = "nvcc:unavailable"


class AtomicCubinCache(AtomicBinaryCache):

    def __init__(self, cache_dir):
        super().__init__(cache_dir, binary_suffix=".cubin", source_suffix=".cu")

    def cubin_path(self, identity):
        return self.binary_path(identity)


__all__ = [
    "BuildIdentity",
    "AtomicCubinCache",
    "UncacheableBuildError",
    "compute_build_identity",
    "normalize_options",
    "normalize_include_paths",
    "scan_user_dependencies",
    "expand_option_dependencies",
    "default_toolchain_identity",
    "atomic_publish",
]

# Bump when the fixed compile_cuda command line (-O3/-std=...) or the identity
# serialization changes in a way that invalidates previously cached cubins.
_BUILD_SCHEMA = 3

# Quoted and angle-bracket includes are both tracked; angle headers found in a
# user include directory are user content, not toolchain content.
# A #include whose argument is neither quoted nor angle-bracket is a
# macro-expanded/computed include (e.g. ``#include HEADER``). Its expansion is
# unknowable without running the preprocessor, so caching fails closed rather
# than silently missing the selected header.

# Flags that take an include search directory as the very next argv token.
# Flags with an attached directory, e.g. -I/foo or --include-path=/foo.
_ATTACHED_INC_DIR_PREFIXES = ("-I", "-iquote", "-isystem", "--include-path=", "--system-include=")
# Flags that force-include a header as the next argv token.
# Flags naming an options/command file: paired, attached ("=") and
# comma-separated multiple-file forms, plus the leading-"@" response form.
_OPTIONS_FILE_FLAGS = frozenset({"-optf", "--options-file"})

# Environment variables that can alter nvcc argv or header selection without
# appearing in the Python options. When any non-empty one is present caching
# fails closed (disable cache or pass explicit inputs).
_DEPENDENCY_BEARING_ENV = (
    "NVCC_PREPEND_FLAGS",
    "NVCC_APPEND_FLAGS",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
)

# Bound for recursive options/response-file expansion (prevents cycles from
# overflowing the stack).

# Override for tests/offline builds that must not shell out to nvcc just to
# establish its identity. Production code normally leaves this unset.
_TOOLCHAIN_ENV_VAR = "LITTLE_KERNEL_TOOLCHAIN_ID"

# -----------------------------------------------------------------------------
# Normalization helpers
# -----------------------------------------------------------------------------


def default_toolchain_identity() -> str:
    """Return a stable identity for the available nvcc toolchain.

    The environment variable :data:`_TOOLCHAIN_ENV_VAR` provides an explicit,
    subprocess-free override (used in deterministic CPU tests).  Otherwise the
    nvcc executable path and its ``--version`` output are hashed.  When no
    nvcc is discoverable an explicit ``"nvcc:unavailable"`` marker is used --
    compilation itself will fail downstream, but caching never silently
    pretends two different toolchains are identical.
    """
    override = os.environ.get(_TOOLCHAIN_ENV_VAR)
    if override:
        return override

    import shutil
    import subprocess

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        return "nvcc:unavailable"
    try:
        result = subprocess.run([nvcc, "--version"], capture_output=True, text=True, check=False)
        version_text = result.stdout or result.stderr or ""
    except OSError as exc:
        version_text = f"error:{type(exc).__name__}"
    digest = _sha256(f"{nvcc}\n{version_text}")[:16]
    return f"nvcc:{os.path.basename(nvcc)}:{digest}"


# -----------------------------------------------------------------------------
# Immutable build identity
# -----------------------------------------------------------------------------


def compute_build_identity(
        *,
        kernel_name: str,
        source: str,
        target_arch: str,
        options: Optional[Sequence[str]] = None,
        include_paths: Optional[Sequence[Union[str, Path]]] = None,
        extra_dependencies: Sequence[Any] = (),
        toolchain_id: Optional[str] = None,
) -> BuildIdentity:
    """Compute a complete :class:`BuildIdentity`.

    ``target_arch`` must already be resolved (typically via
    :func:`little_kernel.backends.nv.common.runtime.target.resolve_target`). Dependency-bearing
    flags inside ``options`` (``-I``, forced includes, ``@response`` files)
    are expanded so user header contents cannot evade the key.
    """
    arch = getattr(target_arch, "arch", target_arch)
    if not isinstance(arch, str):
        raise TypeError(f"target_arch must resolve to a string, got {target_arch!r}")

    # Fail closed when an environment variable could alter nvcc argv or header
    # selection outside the options we can see. Disabling the cache bypasses
    # this entirely (compute_build_identity is only called for cache keying).
    untracked = sorted(name for name in _DEPENDENCY_BEARING_ENV if (os.environ.get(name) or "").strip())
    if untracked:
        raise UncacheableBuildError(f"environment variable(s) {untracked} can change nvcc options or "
                                    f"header selection without entering the cache key; unset them, pass "
                                    f"explicit options/include paths, or disable caching")

    ordered_options = normalize_options(options)
    # Use the single shared effective-search policy (including the implicit
    # packaged default include directory the compiler adds when no explicit
    # paths are supplied), preserving exact compiler search order.
    from little_kernel.backends.nv.common.runtime.compiler import effective_include_paths

    ordered_includes = normalize_include_paths(
        effective_include_paths(list(include_paths) if include_paths is not None else None))
    option_system_dirs, option_quote_dirs, forced, response_hashes = (expand_option_dependencies(ordered_options))

    dependencies = scan_user_dependencies(
        source,
        ordered_includes,
        extra_dependencies,
        option_system_dirs=option_system_dirs,
        option_quote_dirs=option_quote_dirs,
        forced_includes=forced,
    )
    dependencies = tuple(sorted(set(dependencies) | {(f"response:{h}", "response") for h in response_hashes}))

    if toolchain_id is None:
        toolchain_id = default_toolchain_identity()
    return BuildIdentity(
        kernel_name=kernel_name,
        source_hash=_sha256(source),
        target_arch=arch,
        options=ordered_options,
        include_paths=ordered_includes,
        dependencies=dependencies,
        toolchain_id=toolchain_id,
    )


# -----------------------------------------------------------------------------
# Atomic publication
# -----------------------------------------------------------------------------
