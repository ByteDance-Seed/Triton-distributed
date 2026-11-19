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
"""Filesystem-described plugins: removed backends are genuinely unavailable.

Discovery reads packaged descriptors, never imports every backend to find one.
The shared registry has no hardcoded vendor/hardware implementation list.
"""
from __future__ import annotations

from functools import lru_cache
from importlib import import_module
import json
from pathlib import Path

from little_kernel.backends.interfaces import BackendComponents, BackendSpec, BackendUnavailable


def load_symbol(reference: str):
    module, separator, attribute = reference.partition(":")
    if not separator:
        raise ValueError(f"Expected module:attribute reference, got {reference!r}")
    loaded = import_module(module)
    return getattr(loaded, attribute) if attribute else loaded


@lru_cache(maxsize=1)
def backend_specs() -> tuple[BackendSpec, ...]:
    specs = []
    for path in sorted(Path(__file__).parent.glob("*/*/backend.json")):
        data = json.loads(path.read_text())
        specs.append(BackendSpec(**data))
    ids = [spec.backend_id for spec in specs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate backend descriptors")
    return tuple(specs)


def available_backends() -> tuple[str, ...]:
    return tuple(spec.backend_id for spec in backend_specs())


def resolve_backend_spec(backend: str, architecture: str | None = None) -> BackendSpec:
    if not isinstance(backend, str) or not backend:
        raise ValueError("A backend id or installed alias is required")
    candidates = [s for s in backend_specs() if backend == s.backend_id or backend in s.aliases]
    if architecture is not None:
        arch = str(getattr(architecture, "arch", architecture))
        matching = [s for s in candidates if arch in s.architectures]
        if matching:
            candidates = matching
        else:
            candidates = []
    if not candidates:
        raise BackendUnavailable(f"Backend {backend!r} (architecture {architecture!r}) is not included; "
                                 f"available backends: {', '.join(available_backends()) or '(none)'}")
    return min(candidates, key=lambda s: (s.priority, s.backend_id))


@lru_cache(maxsize=None)
def _instantiate(backend_id: str) -> BackendComponents:
    spec = resolve_backend_spec(backend_id)
    result = load_symbol(spec.factory)(spec)
    if not isinstance(result, BackendComponents) or result.backend_id != backend_id:
        raise TypeError(f"Backend factory {spec.factory} returned an invalid component set")
    return result


def get_backend(backend: str, architecture: str | None = None) -> BackendComponents:
    return _instantiate(resolve_backend_spec(backend, architecture).backend_id)


def language_symbols(backend: str | None = None) -> dict[str, str]:
    specs = (resolve_backend_spec(backend), ) if backend is not None else backend_specs()
    result = {}
    for spec in sorted(specs, key=lambda s: (s.priority, s.backend_id), reverse=True):
        result.update(spec.language_exports)
    return result


def resolve_language_symbol(name: str, backend: str | None = None):
    symbols = language_symbols(backend)
    if name not in symbols:
        raise AttributeError(f"No installed backend exports language symbol {name!r}")
    return load_symbol(symbols[name])


def runtime_symbols() -> dict[str, str]:
    result = {}
    for spec in backend_specs():
        result.update(spec.runtime_exports)
    return result


def resolve_runtime_symbol(name: str):
    reference = runtime_symbols().get(name)
    if reference is None:
        raise AttributeError(f"No installed backend exports runtime symbol {name!r}")
    return load_symbol(reference)


class LanguageNamespace:
    """Target-specific language view over shared types and backend primitives."""

    def __init__(self, backend_id: str):
        self.backend_id = backend_id

    def __getattr__(self, name):
        from little_kernel import language
        if name in language.COMMON_EXPORTS:
            return getattr(language, name)
        return resolve_language_symbol(name, self.backend_id)

    def __dir__(self):
        from little_kernel import language
        return sorted(set(language.COMMON_EXPORTS) | set(language_symbols(self.backend_id)))
