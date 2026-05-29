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
"""Central CuTeDSL kernel cache with TVM FFI and optional AOT artifacts."""

import functools
import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, Mapping, Optional, Tuple

_DEFAULT_BACKEND = "tvm_ffi"
_ENV_AOT_DIR = "FLASH_COMM_CUTEDSL_AOT_DIR"
_ENV_CACHE_MODE = "FLASH_COMM_CUTEDSL_CACHE_MODE"
_ENV_EXPORT_ON_JIT = "FLASH_COMM_CUTEDSL_EXPORT_ON_JIT"
_ENV_RECORD_KEYS = "FLASH_COMM_CUTEDSL_RECORD_KEYS"

# Cache modes selected by FLASH_COMM_CUTEDSL_CACHE_MODE:
#   auto     : memory -> AOT -> JIT fallback (default).
#   strict   : memory -> AOT, fail on miss; useful for production no-JIT runs.
#   jit_only : memory -> JIT only; ignore AOT load/export paths.
_VALID_CACHE_MODES = {"auto", "strict", "jit_only"}
_RUNTIME_ENVIRONMENT: Optional[Tuple[Tuple[str, Any], ...]] = None


def _normalise_value(value: Any) -> Any:
    """Convert key material into stable JSON-compatible values."""
    if isinstance(value, Mapping):
        return {str(k): _normalise_value(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, (tuple, list)):
        return [_normalise_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _freeze_key_value(value: Any) -> Hashable:
    """Return immutable key material while leaving hashable values untouched."""
    try:
        hash(value)
        return value
    except TypeError:
        pass
    if isinstance(value, Mapping):
        return tuple((str(k), _freeze_key_value(value[k])) for k in sorted(value, key=str))
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_key_value(v) for v in value)
    return str(value)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_normalise_value(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_symbol(name: str) -> str:
    symbol = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not symbol:
        symbol = "kernel"
    if symbol[0].isdigit():
        symbol = f"k_{symbol}"
    return symbol


def _runtime_environment() -> Tuple[Tuple[str, Any], ...]:
    """Best-effort environment fingerprint for generated code."""
    global _RUNTIME_ENVIRONMENT
    if _RUNTIME_ENVIRONMENT is not None:
        return _RUNTIME_ENVIRONMENT

    env: Dict[str, Any] = {}
    try:
        import torch

        env["torch_cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            env["compute_capability"] = torch.cuda.get_device_capability(device)
        else:
            env["compute_capability"] = None
    except Exception as exc:  # pragma: no cover - defensive fingerprinting
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import cutlass

        env["cutlass_version"] = getattr(cutlass, "__version__", None)
        cuda_version = getattr(cutlass, "CUDA_VERSION", None)
        env["cutlass_cuda_version"] = str(cuda_version) if cuda_version is not None else None
    except Exception as exc:  # pragma: no cover
        env["cutlass_error"] = f"{type(exc).__name__}: {exc}"
    try:
        import tvm_ffi

        env["tvm_ffi_version"] = getattr(tvm_ffi, "__version__", None)
    except Exception as exc:  # pragma: no cover
        env["tvm_ffi_error"] = f"{type(exc).__name__}: {exc}"
    _RUNTIME_ENVIRONMENT = tuple(sorted(env.items()))
    return _RUNTIME_ENVIRONMENT


@dataclass(frozen=True, slots=True)
class KernelKey:
    """Stable identity for an in-memory or AOT CuTeDSL artifact."""

    op_name: str
    variant_args: Hashable
    backend: str = _DEFAULT_BACKEND
    environment: Tuple[Tuple[str, Any], ...] = field(default_factory=_runtime_environment)
    compile_options: Tuple[str, ...] = ()
    _hash_value: int = field(init=False, repr=False, compare=False)
    _stable_hash_value: Optional[str] = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        compile_options = ((self.compile_options, )
                           if isinstance(self.compile_options, str) else tuple(self.compile_options))
        object.__setattr__(self, "variant_args", _freeze_key_value(self.variant_args))
        object.__setattr__(self, "environment", tuple(self.environment))
        object.__setattr__(self, "compile_options", compile_options)
        object.__setattr__(
            self,
            "_hash_value",
            hash((self.op_name, self.variant_args, self.backend, self.environment, self.compile_options)),
        )

    def __hash__(self) -> int:
        return self._hash_value

    def payload(self) -> Dict[str, Any]:
        return {
            "op_name": self.op_name,
            "backend": self.backend,
            "environment": self.environment,
            "compile_options": self.compile_options,
            "variant_args": self.variant_args,
        }

    @property
    def hash(self) -> str:
        if self._stable_hash_value is None:
            object.__setattr__(self, "_stable_hash_value", _stable_hash(self.payload()))
        return self._stable_hash_value


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """Declarative request for a compiled CuTeDSL callable."""

    op_name: str
    variant_args: Hashable
    builder: Callable[[], Any]
    backend: str = _DEFAULT_BACKEND
    compile_options: Tuple[str, ...] = ()
    symbol: Optional[str] = None
    _key: KernelKey = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        key = KernelKey(
            op_name=self.op_name,
            variant_args=self.variant_args,
            backend=self.backend,
            compile_options=self.compile_options,
        )
        object.__setattr__(self, "variant_args", key.variant_args)
        object.__setattr__(self, "compile_options", key.compile_options)
        object.__setattr__(self, "_key", key)

    def key(self, environment: Optional[Tuple[Tuple[str, Any], ...]] = None) -> KernelKey:
        if environment is None or environment == self._key.environment:
            return self._key
        return KernelKey(
            op_name=self._key.op_name,
            variant_args=self._key.variant_args,
            backend=self._key.backend,
            environment=environment,
            compile_options=self._key.compile_options,
        )

    def symbol_name(self, key: KernelKey) -> str:
        if self.symbol:
            return _safe_symbol(self.symbol)
        return f"{_safe_symbol(self.op_name)}_{key.hash[:16]}"


@dataclass
class KernelHandle:
    """Callable wrapper that records where the artifact came from."""

    key: KernelKey
    function: Callable[..., Any]
    source: str
    symbol_name: str
    artifact_path: Optional[str] = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.function(*args, **kwargs)


@dataclass
class KernelCacheStats:
    entries: int = 0
    memory_hits: int = 0
    aot_hits: int = 0
    jit_compiles: int = 0
    exports: int = 0
    waits: int = 0
    strict_misses: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "entries": self.entries,
            "memory_hits": self.memory_hits,
            "aot_hits": self.aot_hits,
            "jit_compiles": self.jit_compiles,
            "exports": self.exports,
            "waits": self.waits,
            "strict_misses": self.strict_misses,
        }


class CuTeDSLKernelCache:
    """Unified cache for TVM FFI CuTeDSL artifacts.

    Lookup order is in-process memory, optional AOT shared library, then
    JIT compile. With ``FLASH_COMM_CUTEDSL_EXPORT_ON_JIT=1`` the miss path
    is guarded by a per-key file lock so torchrun ranks can share the
    exported artifact instead of linking the same ``.so`` concurrently.
    """

    def __init__(self, *, aot_dir: Optional[str] = None, mode: Optional[str] = None,
                 export_on_jit: Optional[bool] = None) -> None:
        self._entries: Dict[KernelKey, KernelHandle] = {}
        self._last_key: Optional[KernelKey] = None
        self._last_value: Optional[KernelHandle] = None
        self._stats = KernelCacheStats()
        self._environment = _runtime_environment()
        self.aot_dir = aot_dir if aot_dir is not None else os.getenv(_ENV_AOT_DIR)
        self.mode = mode if mode is not None else os.getenv(_ENV_CACHE_MODE, "auto")
        if self.mode not in _VALID_CACHE_MODES:
            raise ValueError(f"{_ENV_CACHE_MODE} must be one of {_VALID_CACHE_MODES}; got {self.mode}")
        if export_on_jit is None:
            export_on_jit = os.getenv(_ENV_EXPORT_ON_JIT, "0") == "1"
        self.export_on_jit = bool(export_on_jit)

    def get_or_compile(self, spec: KernelSpec) -> KernelHandle:
        key = spec.key(self._environment)
        if self._last_key == key and self._last_value is not None:
            self._stats.memory_hits += 1
            return self._last_value
        cached = self._entries.get(key)
        if cached is not None:
            self._last_key = key
            self._last_value = cached
            self._stats.memory_hits += 1
            return cached

        self._record_key(spec, key)
        loaded = self._try_load_aot(spec, key)
        if loaded is not None:
            self._store(key, loaded)
            self._stats.aot_hits += 1
            return loaded
        if self.mode == "strict":
            self._stats.strict_misses += 1
            raise FileNotFoundError(f"CuTeDSL AOT artifact not found for {spec.op_name} key={key.hash}")

        if self.export_on_jit and self.aot_dir and self.mode != "jit_only":
            handle = self._compile_export_or_wait(spec, key)
        else:
            handle = self._compile(spec, key)
        self._store(key, handle)
        return handle

    def clear(self) -> None:
        self._entries.clear()
        self._last_key = None
        self._last_value = None
        self._stats.entries = 0

    def stats(self) -> Dict[str, Any]:
        op_counts: Dict[str, int] = {}
        source_counts: Dict[str, int] = {}
        for handle in self._entries.values():
            op_counts[handle.key.op_name] = op_counts.get(handle.key.op_name, 0) + 1
            source_counts[handle.source] = source_counts.get(handle.source, 0) + 1
        stats = self._stats.as_dict()
        stats["entries"] = len(self._entries)
        stats["ops"] = op_counts
        stats["sources"] = source_counts
        return stats

    def _store(self, key: KernelKey, handle: KernelHandle) -> None:
        self._entries[key] = handle
        self._last_key = key
        self._last_value = handle
        self._stats.entries = len(self._entries)

    def _compile(self, spec: KernelSpec, key: KernelKey) -> KernelHandle:
        compiled = spec.builder()
        self._stats.jit_compiles += 1
        return KernelHandle(key=key, function=compiled, source="jit", symbol_name=spec.symbol_name(key))

    def _try_load_aot(self, spec: KernelSpec, key: KernelKey) -> Optional[KernelHandle]:
        if not self.aot_dir or self.mode == "jit_only":
            return None
        from .cutedsl_aot import KernelArtifactStore

        store = KernelArtifactStore(self.aot_dir)
        return store.load(spec, key)

    def _compile_export_or_wait(self, spec: KernelSpec, key: KernelKey) -> KernelHandle:
        from .cutedsl_aot import KernelArtifactStore

        store = KernelArtifactStore(self.aot_dir)
        with store.lock(key):
            self._stats.waits += 1
            loaded = store.load(spec, key)
            if loaded is not None:
                self._stats.aot_hits += 1
                return loaded
            handle = self._compile(spec, key)
            exported = store.export(spec, handle)
            self._stats.exports += 1
            return exported

    def _record_key(self, spec: KernelSpec, key: KernelKey) -> None:
        record_path = os.getenv(_ENV_RECORD_KEYS)
        if not record_path:
            return
        payload = {
            "key": key.payload(),
            "key_hash": key.hash,
            "symbol": spec.symbol_name(key),
        }
        os.makedirs(os.path.dirname(os.path.abspath(record_path)), exist_ok=True)
        with open(record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")


_GLOBAL_CUTEDSL_KERNEL_CACHE: Optional[CuTeDSLKernelCache] = None


def get_global_cutedsl_kernel_cache() -> CuTeDSLKernelCache:
    """Return the process-wide CuTeDSL kernel cache."""
    global _GLOBAL_CUTEDSL_KERNEL_CACHE
    if _GLOBAL_CUTEDSL_KERNEL_CACHE is None:
        _GLOBAL_CUTEDSL_KERNEL_CACHE = CuTeDSLKernelCache()
    return _GLOBAL_CUTEDSL_KERNEL_CACHE


def cached_cutedsl_kernel(spec_method: Callable[..., KernelSpec]) -> Callable[..., KernelHandle]:
    """Decorator for op methods that return a :class:`KernelSpec`."""

    @functools.wraps(spec_method)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> KernelHandle:
        return self._cache.get_or_compile(spec_method(self, *args, **kwargs))

    return wrapper
