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
"""Isolated compilation sessions and explicit frontend selection (RF07/RF05).

Each :meth:`LLKernel.compile` / :meth:`LLKernel.lower_result` /
:meth:`LLKernel.build` creates exactly one :class:`CompilationSession`.  A
session owns:

* a fresh independent copy of the kernel's user context (source globals,
  closure values, launch/alias/min-blocks options). Built-in mutable
  containers are recursively duplicated (with a cycle/alias-safe memo) while
  modules, functions, interned LLTypes and opaque runtime handles stay
  shared read-only, so passes and codegen mutate only this copy and derived
  analysis state (generated symbols, recorded local array shapes, nested
  option lists such as ``options['history']`` ...) can neither accumulate on
  ``LLKernel.ctx`` nor alias a sibling/concurrent session across constexpr
  specializations;
* the concrete constexpr values for this specialization;
* the resolved immutable :class:`~little_kernel.core.passes.pass_registry.Pipeline`
  (default or explicitly supplied) and the ordered stage ids;
* an optional explicit frontend hook (e.g. role/dataflow lowering, supplied
  by the dataflow worker);
* the launch ABI :class:`~little_kernel.runtime.abi.KernelSignature` built
  from the actual remaining parameters after constexpr removal. A function
  with no parameter annotations at all is the explicit legacy untyped path
  (``None``); any other ABI construction error (unsupported typed layout,
  unresolvable annotation, adapter failure) is propagated -- never hidden as
  a guessed/empty signature.

The lowered result (the tree **and** this session's full derived context) is
returned to the caller; it is never published on the shared
:class:`~little_kernel.core.compile.LLKernel`, so there is no last-session
race between concurrent compilations or recursive device-function lowering.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.core.compile import _substitute_constexpr
from little_kernel.core.passes.pass_registry import PassSpec, Pipeline, get_pipeline

__all__ = [
    "CompilationSession",
    "CompilationResult",
    "clone_context",
    "register_frontend",
    "select_frontend",
    "registered_frontends",
]

# -----------------------------------------------------------------------------
# Per-session context cloning
# -----------------------------------------------------------------------------
#
# A shallow ``dict(kernel.ctx)`` is not enough for RF07: nested mutable option
# containers (e.g. ``ctx['options']['history']``) would stay aliased to the
# source globals and to every sibling session.  We deep-copy only the built-in
# mutable containers -- preserving identity sharing/cycles via a memo -- while
# leaving every other object (modules, functions/classes, interned immutable
# LLTypes, ctypes values and opaque runtime tensor/device handles) shared by
# reference.  Those values are either immutable by contract or external
# resources that must never be duplicated in the host process.

_CLONE_CONTAINERS = (dict, list, set, tuple, frozenset)


def clone_context(source: Mapping[str, Any]) -> dict:
    """Return an independent, cycle-safe copy of a compilation context.

    Built-in containers (``dict/list/set/tuple/frozenset``) are recursively
    duplicated with a memo so that a value referenced twice and any container
    cycle keep shared identity *within the clone* without ever pointing back
    at the source containers.  Non-container objects are shared read-only.
    """
    memo: dict = {}

    def _clone(value):
        if not isinstance(value, _CLONE_CONTAINERS):
            return value
        marker = id(value)
        if marker in memo:
            return memo[marker]
        if isinstance(value, dict):
            new = {}
            memo[marker] = new
            for key, item in value.items():
                cloned_key = _clone(key) if isinstance(key, _CLONE_CONTAINERS) else key
                new[cloned_key] = _clone(item)
            return new
        if isinstance(value, list):
            new = []
            memo[marker] = new
            new.extend(_clone(item) for item in value)
            return new
        if isinstance(value, set):
            new = set()
            memo[marker] = new
            for item in value:
                new.add(_clone(item))
            return new
        if isinstance(value, tuple):
            cloned = tuple(_clone(item) for item in value)
            # A mutable member can lead back to this tuple and complete its
            # clone during recursion. Preserve that already-published identity.
            if marker in memo:
                return memo[marker]
            memo[marker] = cloned
            return cloned
        cloned = frozenset(_clone(item) for item in value)
        memo[marker] = cloned
        return cloned

    # Seed the memo with the result itself so a top-level self-reference
    # (``ctx['ctx'] = ctx``) resolves to the clone rather than re-copying or
    # pointing back at the source containers.
    cloned_source: dict = {}
    memo[id(source)] = cloned_source
    for key, value in source.items():
        cloned_key = _clone(key) if isinstance(key, _CLONE_CONTAINERS) else key
        cloned_source[cloned_key] = _clone(value)
    return cloned_source


# -----------------------------------------------------------------------------
# Explicit frontend/pipeline selection hook (RF05 integration point)
# -----------------------------------------------------------------------------
#
# Role/dataflow lowering is selected explicitly through this hook -- ordinary
# scalar/SIMT kernels use ``frontend=None`` and never run role passes.  The
# canonical dataflow frontend is supplied by the dataflow worker as
# ``dataflow_frontend(tree, ctx) -> ast.Module`` (together with its explicit
# post-lowering pass tuple); core deliberately registers no fake fallback.

FrontendHook = Callable[[ast.Module, Mapping[str, Any]], ast.Module]

_FRONTEND_HOOKS: dict[str, FrontendHook] = {}


def register_frontend(name: str, frontend: FrontendHook) -> None:
    """Register a named explicit frontend hook (e.g. ``"dataflow"``).

    The hook receives the parsed (constexpr-substituted) AST and the fresh
    session context and returns the lowered frontend AST before the ordinary
    pipeline runs.  Re-registering a name replaces the hook; registering a
    non-callable raises rather than installing a silent fallback.
    """
    if not isinstance(name, str) or not name:
        raise ValueError("frontend name must be a non-empty str")
    if not callable(frontend):
        raise TypeError(f"frontend {name!r} must be callable(tree, ctx)")
    _FRONTEND_HOOKS[name] = frontend


def registered_frontends() -> Tuple[str, ...]:
    """Return the names of explicitly registered frontend hooks."""
    return tuple(sorted(_FRONTEND_HOOKS))


def select_frontend(frontend: Union[None, str, FrontendHook]) -> Optional[FrontendHook]:
    """Resolve an explicit frontend selection.

    * ``None`` -> the ordinary scalar/SIMT frontend (no extra lowering);
    * a callable -> used directly as ``frontend(tree, ctx)``;
    * a registered name string (e.g. ``"dataflow"``) -> that hook;
    * an unknown string raises :class:`ValueError` -- core never invents a
      fallback pipeline for role/dataflow programs.
    """
    if frontend is None:
        return None
    if callable(frontend):
        return frontend
    if isinstance(frontend, str):
        if frontend not in _FRONTEND_HOOKS:
            raise ValueError(f"Unknown frontend {frontend!r}; registered explicit "
                             f"frontends are {registered_frontends() or ()}. Role/dataflow "
                             "lowering must be selected explicitly via a registered hook; "
                             "no implicit fallback pipeline is available.")
        return _FRONTEND_HOOKS[frontend]
    raise TypeError("frontend must be None, a callable(tree, ctx) or a registered "
                    f"frontend name, got {type(frontend).__name__}")


class CompilationResult:
    """Immutable outcome of one session's lowering."""

    def __init__(
        self,
        tree: ast.Module,
        context: Mapping[str, Any],
        derived_keys: frozenset,
        signature: Any,
        pass_ids: Tuple[str, ...],
        frontend: Optional[str],
        constexpr_args: Mapping[str, Any],
    ):
        self._tree = tree
        self._context = MappingProxyType(dict(context))
        self._derived_keys = frozenset(derived_keys)
        self._signature = signature
        self._pass_ids = tuple(pass_ids)
        self._frontend = frontend
        self._constexpr_args = MappingProxyType(dict(constexpr_args))

    @property
    def tree(self) -> ast.Module:
        return self._tree

    @property
    def context(self) -> Mapping[str, Any]:
        """Full session context, including pass-derived bindings."""
        return self._context

    @property
    def derived_keys(self) -> frozenset:
        """Context keys introduced by this session's pipeline."""
        return self._derived_keys

    @property
    def signature(self):
        return self._signature

    @property
    def pass_ids(self) -> Tuple[str, ...]:
        return self._pass_ids

    @property
    def frontend(self) -> Optional[str]:
        return self._frontend

    @property
    def constexpr_args(self) -> Mapping[str, Any]:
        return self._constexpr_args


def _explicit_pipeline(backend: str, passes: Sequence[Callable]) -> Pipeline:
    """Compatibility adapter for legacy explicit pass lists.

    Bare callables are identified by their function name and executed in the
    order supplied (the caller is authoritative for its phase order); this
    keeps existing explicit pass-list callers working without exposing or
    mutating global defaults.
    """
    specs = []
    seen = set()
    for i, func in enumerate(passes):
        if isinstance(func, PassSpec):
            specs.append(func)
            continue
        pid = getattr(func, "_lk_pass_id", None) or getattr(func, "__name__", f"explicit_pass_{i}")
        # Explicit lists may legitimately repeat helper passes; disambiguate
        # duplicates while preserving execution order.
        unique = pid
        suffix = 1
        while unique in seen:
            suffix += 1
            unique = f"{pid}#{suffix}"
        seen.add(unique)
        specs.append(PassSpec(pass_id=unique, func=func, name=pid))
    return Pipeline(backend, specs)


class CompilationSession:
    """One isolated compilation of one kernel specialization."""

    def __init__(
        self,
        kernel,
        passes: Union[None, Pipeline, Sequence[Callable]] = None,
        constexpr_args: Optional[Mapping[str, Any]] = None,
        frontend: Union[None, str, FrontendHook] = None,
    ):
        self.kernel = kernel
        self.backend = getattr(kernel, "backend", "cuda")
        # Defensive copy of constexpr values: the session never aliases the
        # caller's mapping.
        self.constexpr_args: dict = dict(constexpr_args or {})
        self.frontend_hook = select_frontend(frontend)
        self.frontend_name = (frontend if isinstance(frontend, str) else
                              (getattr(frontend, "__name__", None) if frontend is not None else None))

        # Fresh per-compilation context, cloned from the kernel's user
        # globals/options with every built-in mutable container duplicated
        # (cycle/alias-safe memo): nested option containers such as
        # ``ctx['options']['history']`` can never alias the source or a
        # sibling session, while modules/functions/interned LLTypes/opaque
        # runtime handles stay shared read-only. The source kernel.ctx is
        # never replaced or stripped -- a session never writes derived keys
        # back to it.
        source_ctx = getattr(kernel, "ctx", {}) or {}
        self.ctx: dict = clone_context(source_ctx)
        self._base_keys = frozenset(self.ctx)
        self.ctx["__LK_CLUSTER_SHAPE__"] = getattr(kernel, "cluster_shape", None)
        self.ctx["__LK_CTA_GROUP_SHAPE__"] = getattr(kernel, "cta_group_shape", None)
        if getattr(kernel, "is_entry", False):
            self.ctx[__LITTLE_KERNEL_ENTRY__] = kernel.py_func
        else:
            # Reserved entry metadata belongs to this compilation unit. A
            # device function's source globals may contain the exported key
            # constant or an unrelated entry; neither makes it a global kernel.
            self.ctx.pop(__LITTLE_KERNEL_ENTRY__, None)

        # Resolve the immutable pipeline before any pass executes.
        if passes is None:
            from .passes.pass_config import get_default_pipeline
            self.pipeline = get_default_pipeline(self.backend)
        elif isinstance(passes, Pipeline):
            self.pipeline = passes.validate()
        else:
            self.pipeline = _explicit_pipeline(self.backend, tuple(passes))

        # The immutable, pre-validated pipeline is the only per-session
        # cached state: lowering returns a fresh result to its caller and
        # publishes nothing on the shared kernel.

    # -- public API ---------------------------------------------------------

    @property
    def pass_ids(self) -> Tuple[str, ...]:
        return self.pipeline.pass_ids

    def parse_tree(self) -> ast.Module:
        source = inspect.getsource(self.kernel.py_func)
        # Remove indentation for functions defined in non-global scope.
        source = textwrap.dedent(source)
        tree = ast.parse(source)
        if self.constexpr_args:
            tree = _substitute_constexpr(tree, self.constexpr_args)
        return tree

    def lower(self, *, require_abi: bool = False) -> CompilationResult:
        """Run the explicit frontend hook (if any) then the pipeline.

        The pipeline's phase prerequisites were validated at construction,
        i.e. before any stage could mutate this session. A fresh
        :class:`CompilationResult` is returned on every call; nothing is
        published on the shared kernel -- the caller owns the result and its
        per-call context.

        ``require_abi`` forces the launchable build/artifact path to fail
        honestly when no typed ABI can be constructed, instead of treating
        it as the legacy untyped path.
        """
        tree = self.parse_tree()
        if self.frontend_hook is not None:
            tree = self.frontend_hook(tree, self.ctx)
            if not isinstance(tree, ast.AST):
                raise TypeError(f"frontend hook {self.frontend_name!r} must return an AST "
                                f"node, got {type(tree).__name__}")
        for spec in self.pipeline:
            tree = spec.func(tree, self.ctx)
        signature = self._build_signature(require_abi=require_abi)
        return CompilationResult(
            tree=tree,
            context=self.ctx,
            derived_keys=frozenset(set(self.ctx) - self._base_keys),
            signature=signature,
            pass_ids=self.pass_ids,
            frontend=self.frontend_name,
            constexpr_args=self.constexpr_args,
        )

    # -- ABI signature ------------------------------------------------------

    def _constexpr_param_names(self) -> Tuple[str, ...]:
        return tuple(self.constexpr_args.keys())

    @staticmethod
    def _has_no_annotations(py_func) -> bool:
        """True when the function declares no parameter/return annotations.

        Such a kernel is the explicit legacy *untyped* path (signature
        ``None``); a partially-typed or unsupported typed annotation is not
        untyped and must fail honestly instead of being silently downgraded.
        """
        annotations = getattr(py_func, "__annotations__", None) or {}
        return not annotations

    def _build_signature(self, *, require_abi: bool = False):
        """Build the launch ABI signature after constexpr removal.

        Uses the actual remaining parameters of ``py_func`` (constexpr names
        removed) resolved against this session's context.

        * A kernel without any annotations at all is the explicit legacy
          untyped path: it returns ``None`` (or raises when the caller
          requires a launchable typed ABI via ``require_abi``).
        * Every other failure -- an unsupported typed layout, a string
          annotation that cannot resolve, an invalid adapter -- propagates
          as its real error. A known typed signature is never dropped and no
          value is guessed.
        """
        from little_kernel.runtime.abi import signature_from_annotations

        if self._has_no_annotations(self.kernel.py_func):
            if require_abi:
                raise TypeError(f"Kernel {self.kernel.py_func.__name__!r} has no parameter "
                                "annotations: a launchable build/artifact requires an "
                                "explicit KernelSignature; the legacy untyped path is "
                                "source-only and cannot be silently downgraded.")
            return None
        try:
            return signature_from_annotations(
                self.kernel.py_func,
                constexpr_names=self._constexpr_param_names(),
                context=self.ctx,
            )
        except (TypeError, ValueError):
            # Launchable paths (compile_artifact/build) must fail honestly on
            # an unsupported typed layout / unresolvable annotation / invalid
            # adapter rather than guess a signature: re-raise the real error
            # and never drop a signature we did construct. The source-only
            # compile/lower path explicitly does not claim a host launch ABI
            # (it returns ``None``), which keeps source emission and
            # recursive device-function lowering working for supported
            # codegen that does not need a host ABI.
            if require_abi:
                raise
            return None
