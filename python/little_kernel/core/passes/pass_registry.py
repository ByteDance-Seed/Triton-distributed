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
"""Pass registry and immutable pipeline specifications (RF07).

A :class:`PassSpec` binds a pass callable to a stable ``pass_id`` together
with its ``requires`` prerequisites and the analyses it ``produces``.  A
:class:`Pipeline` is an immutable, validated ordering of specs.

The registry keeps *private* stage lists: every accessor returns a fresh
copy or an immutable pipeline, so a consumer can never append to or reorder
the global defaults.  Registering the same pass id for a backend twice is a
idempotent no-op rather than a second stage.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from little_kernel.core.passes.dump import dump_ast

PASS_DEBUG = os.getenv("PASS_DEBUG", "").split(",")
PASS_DEBUG_ALL = "all" in PASS_DEBUG

__all__ = [
    "PassSpec",
    "Pipeline",
    "PassRegistry",
    "get_pass_registry",
    "register_pass",
    "get_passes",
    "get_pipeline",
    "empty_pass",
    "debug_wrapper",
]


def empty_pass(tree, ctx, *args, **kwargs):
    """Empty pass that does nothing."""
    return tree


def debug_wrapper(pass_func, pass_name, simple_mode: bool = True):
    """Wrap a pass function with debug output."""

    def wrapper_func(tree, ctx, *args, **kwargs):
        tree = pass_func(tree, ctx, *args, **kwargs)
        if pass_name in PASS_DEBUG or PASS_DEBUG_ALL:
            dump_ast(f"after {pass_name}", simple_mode=simple_mode)(tree, ctx, *args, **kwargs)
        return tree

    wrapper_func.__name__ = getattr(pass_func, "__name__", pass_name)
    return wrapper_func


@dataclass(frozen=True)
class PassSpec:
    """One identified, ordered pipeline stage.

    Attributes
    ----------
    pass_id:
        Stable identifier used for prerequisite validation and diagnostics.
    func:
        Pass callable ``(tree, ctx) -> tree``.
    name:
        Debug display name (defaults to ``pass_id``).
    requires:
        Pass ids that must have executed already.
    produces:
        Analysis/ctx keys the pass publishes (metadata only).
    backend:
        Backend this stage belongs to, if any.
    """

    pass_id: str
    func: Callable
    name: Optional[str] = None
    requires: Tuple[str, ...] = field(default_factory=tuple)
    produces: Tuple[str, ...] = field(default_factory=tuple)
    backend: Optional[str] = None

    def __post_init__(self):
        if not isinstance(self.pass_id, str) or not self.pass_id:
            raise ValueError("PassSpec.pass_id must be a non-empty str")
        if not callable(self.func):
            raise TypeError(f"PassSpec {self.pass_id!r}.func must be callable")
        if not isinstance(self.requires, tuple):
            object.__setattr__(self, "requires", tuple(self.requires))
        if not isinstance(self.produces, tuple):
            object.__setattr__(self, "produces", tuple(self.produces))

    @property
    def display_name(self) -> str:
        return self.name or self.pass_id


class Pipeline:
    """Immutable, validated ordering of :class:`PassSpec` stages."""

    __slots__ = ("_backend", "_stages")

    def __init__(self, backend: str, stages: Sequence[PassSpec]):
        stages = tuple(stages)
        for spec in stages:
            if not isinstance(spec, PassSpec):
                raise TypeError("Pipeline stages must be PassSpec, got "
                                f"{type(spec).__name__}")
        object.__setattr__(self, "_backend", backend)
        object.__setattr__(self, "_stages", stages)
        self.validate()

    def __setattr__(self, name, value):
        raise AttributeError(f"Pipeline is immutable; cannot set {name!r}")

    def __delattr__(self, name):
        raise AttributeError(f"Pipeline is immutable; cannot delete {name!r}")

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def stages(self) -> Tuple[PassSpec, ...]:
        return self._stages

    @property
    def pass_ids(self) -> Tuple[str, ...]:
        return tuple(spec.pass_id for spec in self._stages)

    @property
    def callables(self) -> Tuple[Callable, ...]:
        """Fresh tuple of the bound pass callables for explicit-list callers."""
        return tuple(spec.func for spec in self._stages)

    def validate(self) -> "Pipeline":
        """Reject duplicate stage ids and unsatisfied prerequisites.

        Validation runs at construction time, i.e. *before* any pass can
        execute, so an invalid phase order never partially mutates a
        compilation session.
        """
        seen = set()
        for spec in self._stages:
            duplicated = [pid for pid in spec.requires if pid not in seen]
            if duplicated:
                raise ValueError(f"Pass {spec.pass_id!r} requires prerequisite pass(es) "
                                 f"{tuple(duplicated)} to run first; pipeline order is "
                                 f"{self.pass_ids}")
            if spec.pass_id in seen:
                raise ValueError(f"Duplicate pass id {spec.pass_id!r} in pipeline "
                                 f"{self.pass_ids}")
            seen.add(spec.pass_id)
        return self

    def __len__(self) -> int:
        return len(self._stages)

    def __iter__(self):
        return iter(self._stages)

    def __getitem__(self, index):
        return self._stages[index]

    def __eq__(self, other) -> bool:
        if not isinstance(other, Pipeline):
            return NotImplemented
        return (self._backend == other._backend and self.pass_ids == other.pass_ids)

    def __hash__(self):
        return hash((self._backend, self.pass_ids))

    def __repr__(self) -> str:
        return f"Pipeline(backend={self._backend!r}, stages={self.pass_ids!r})"


class PassRegistry:
    """Registry for compilation passes organized by backend.

    The owned stage lists are private and lock-protected.  Accessors always
    return independent copies or immutable pipelines; registration of a pass
    id already present for the backend is idempotent.
    """

    def __init__(self):
        self._passes: Dict[str, List[PassSpec]] = {}
        self._lock = threading.RLock()

    def register_pass(self, backend: str, pass_func: Callable, pass_name: str = None, enable_debug: bool = True,
                      simple_mode: bool = True, pass_id: str = None, requires: Sequence[str] = (),
                      produces: Sequence[str] = ()) -> PassSpec:
        """
        Register a pass for a specific backend.

        Args:
            backend: Backend name (e.g., "cuda").
            pass_func: Pass function to register.
            pass_name: Optional name for the pass (debug display name).
            enable_debug: Whether to enable the debug wrapper.
            pass_id: Stable id (defaults to pass_name/function name).
                Registering the same id for the same backend twice is a
                no-op: it never appends a second stage.
            requires: Pass ids that must precede this one.
            produces: Analysis keys published by this pass.
        """
        pid = pass_id or pass_name or getattr(pass_func, "__name__", None)
        if not pid:
            raise ValueError("register_pass requires a pass_id, pass_name or a named "
                             "function")
        wrapped = pass_func
        if enable_debug and pass_name:
            wrapped = debug_wrapper(pass_func, pass_name, simple_mode=simple_mode)
        # Bind the canonical pass id onto the stage callable so legacy
        # explicit pass lists obtained from the defaults recover stable ids
        # rather than opaque function names.
        try:
            wrapped._lk_pass_id = pid
        except (AttributeError, TypeError):
            pass
        spec = PassSpec(
            pass_id=pid,
            func=wrapped,
            name=pass_name or pid,
            requires=tuple(requires),
            produces=tuple(produces),
            backend=backend,
        )
        with self._lock:
            specs = self._passes.setdefault(backend, [])
            if any(existing.pass_id == pid for existing in specs):
                # Idempotent: defaults registered twice must not expand the
                # pipeline and previously handed-out references stay valid.
                return next(s for s in specs if s.pass_id == pid)
            specs.append(spec)
        return spec

    def get_pipeline(self, backend: str) -> Pipeline:
        """Return an immutable :class:`Pipeline` snapshot for a backend."""
        with self._lock:
            specs = tuple(self._passes.get(backend, ()))
        return Pipeline(backend, specs)

    def get_passes(self, backend: str) -> List[Callable]:
        """Get a *fresh* list of pass callables for a backend.

        Callers own the returned list and may reorder/extend it; the
        registry's defaults are never exposed for mutation.
        """
        with self._lock:
            return [spec.func for spec in self._passes.get(backend, ())]

    def get_pass_ids(self, backend: str) -> Tuple[str, ...]:
        with self._lock:
            return tuple(spec.pass_id for spec in self._passes.get(backend, ()))

    def register_backend_passes(self, backend: str, passes: List[Callable]):
        """Register multiple passes for a backend at once (idempotent)."""
        for i, pass_func in enumerate(passes):
            pass_name = getattr(pass_func, '__name__', f"pass_{i}")
            self.register_pass(backend, pass_func, pass_name)

    def reset_backend(self, backend: str) -> None:
        """Remove all registered passes for a backend (tests/extensions)."""
        with self._lock:
            self._passes.pop(backend, None)


# Global registry instance
_pass_registry = PassRegistry()


def get_pass_registry() -> PassRegistry:
    """Get the global pass registry."""
    return _pass_registry


def register_pass(backend: str, pass_func: Callable, pass_name: str = None, enable_debug: bool = True,
                  simple_mode: bool = True, pass_id: str = None, requires: Sequence[str] = (),
                  produces: Sequence[str] = ()):
    """Convenience function to register a pass (idempotent per pass id)."""
    return _pass_registry.register_pass(
        backend,
        pass_func,
        pass_name,
        enable_debug,
        simple_mode,
        pass_id=pass_id,
        requires=requires,
        produces=produces,
    )


def get_passes(backend: str) -> List[Callable]:
    """Convenience function: fresh pass-callable list for a backend."""
    return _pass_registry.get_passes(backend)


def get_pipeline(backend: str) -> Pipeline:
    """Convenience function: immutable pipeline snapshot for a backend."""
    return _pass_registry.get_pipeline(backend)
