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

import ast
import hashlib
import inspect
import os
import textwrap
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple, List, Union
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.core.artifact import CompilationArtifact

LK_CACHE_DIR_ENV = "LK_CACHE_DIR"


def _lk_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    if cache_dir is not None:
        return cache_dir
    env = os.environ.get(LK_CACHE_DIR_ENV, "").strip()
    return env or None


def _write_source(cache_dir: str, kernel_name: str, code: str, constexpr_args: Optional[dict] = None,
                  source_suffix=".cc") -> None:
    # Specialization identity (Triton-style): different constexpr instantiations of the
    # same kernel must not overwrite each other in a shared debug cache directory.
    if constexpr_args:
        identity = hashlib.sha256(repr(sorted(constexpr_args.items())).encode("utf-8")).hexdigest()[:8]
        kernel_name = f"{kernel_name}__cx{identity}"
    path = Path(cache_dir) / f"{kernel_name}{source_suffix}"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")
    print(f"[LittleKernel] Generated source: {path}", flush=True)


def _is_little_kernel_related(value) -> bool:
    """True if value is safe to add to ctx (whitelist for closure vars)."""
    if value is None:
        return False
    # little_kernel.language module (e.g. user's "ll")
    if hasattr(value, "__name__") and getattr(value, "__name__", "").startswith("little_kernel."):
        return True
    # Type or instance: allow if type's module is little_kernel
    mod = getattr(type(value), "__module__", None)
    if mod is not None and mod.startswith("little_kernel."):
        return True
    # Value itself may be a module
    mod = getattr(value, "__module__", None)
    if mod is not None and mod.startswith("little_kernel."):
        return True
    # Allow classes (e.g. struct stub classes) so type inference can resolve SpecialStruct(), etc.
    if isinstance(value, type):
        return True
    return False


class _ConstexprSubstituter(ast.NodeTransformer):
    """Replace constexpr parameter Name nodes with Constant nodes."""

    def __init__(self, constexpr_args: dict):
        self.constexpr_args = constexpr_args
        self.constexpr_param_names: set = set()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        from little_kernel.core.type_system import Constexpr
        from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute

        new_args = []
        for arg in node.args.args:
            if arg.arg in self.constexpr_args:
                self.constexpr_param_names.add(arg.arg)
            else:
                new_args.append(arg)
        node.args.args = new_args

        # Also strip from annotations list if present
        new_defaults = []  # noqa: F841
        removed = 0  # noqa: F841
        for i, arg in enumerate(node.args.args + list(self.constexpr_param_names)):
            pass  # defaults handled by positional removal above

        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in self.constexpr_param_names and node.id in self.constexpr_args:
            if not isinstance(node.ctx, ast.Load):
                raise ValueError(f"constexpr parameter {node.id!r} cannot be assigned or deleted: "
                                 "a substituted literal is read-only")
            val = self.constexpr_args[node.id]
            new_node = ast.Constant(value=val)
            return ast.copy_location(new_node, node)
        return node


def _substitute_constexpr(tree: ast.Module, constexpr_args: dict) -> ast.Module:
    """Substitute constexpr parameters with literal values and remove them from the signature."""
    transformer = _ConstexprSubstituter(constexpr_args)
    tree = transformer.visit(tree)
    ast.fix_missing_locations(tree)
    return tree


_NO_ALIAS_CTX_KEY = "__LITTLE_KERNEL_NO_ALIAS_PARAMS__"
_READ_ONLY_CTX_KEY = "__LITTLE_KERNEL_READ_ONLY_PARAMS__"
_MIN_BLOCKS_CTX_KEY = "__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__"


def _normalize_param_names(value, what: str):
    """Normalize a no_alias/read_only name list into a frozenset of identifiers."""
    if value is None:
        return None
    if isinstance(value, str):
        value = (value, )
    try:
        names = tuple(value)
    except TypeError:
        raise ValueError(f"{what} must be an iterable of parameter-name strings, got {value!r}")
    for name in names:
        if not isinstance(name, str) or not name.isidentifier():
            raise ValueError(f"{what} entries must be parameter-name strings, got {name!r}")
    return frozenset(names)


def _normalize_cluster_shape(cluster_shape: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
    if cluster_shape is None:
        return None
    shape = tuple(int(dim) for dim in cluster_shape)
    if len(shape) != 3 or any(dim <= 0 for dim in shape):
        raise ValueError(f"cluster_shape must contain three positive dimensions, got {cluster_shape}")
    return shape


def _normalize_cta_group_shape(cta_group_shape: Optional[Tuple[int, int, int]]) -> Optional[Tuple[int, int, int]]:
    shape = _normalize_cluster_shape(cta_group_shape)
    return shape


def _normalize_launch_bounds(launch_bounds) -> Optional[Tuple[int, Optional[int]]]:
    """Normalize a __launch_bounds__ declaration to (max_threads, min_blocks_or_None).

    Accepted forms: an int max_threads, or a (max_threads,) / (max_threads, min_blocks)
    tuple. Validation happens at decoration time (an illegality in a source-level
    declaration, not a codegen event): counts must be positive ints, max_threads must
    satisfy the selected backend limits, and min_blocks,
    when given, must be at least 1. Booleans are rejected (they are ints in Python).
    """
    if launch_bounds is None:
        return None
    if isinstance(launch_bounds, bool):
        raise ValueError(f"launch_bounds must be an int or (int, int) tuple, got {launch_bounds!r}")
    if isinstance(launch_bounds, int):
        max_threads, min_blocks = launch_bounds, None
    else:
        try:
            parts = tuple(launch_bounds)
        except TypeError:
            raise ValueError(f"launch_bounds must be an int or (int, int) tuple, got {launch_bounds!r}")
        if len(parts) == 1:
            max_threads, min_blocks = parts[0], None
        elif len(parts) == 2:
            max_threads, min_blocks = parts
        else:
            raise ValueError(f"launch_bounds tuple must have 1 or 2 elements, got {launch_bounds!r}")
        if isinstance(max_threads, bool) or not isinstance(max_threads, int):
            raise ValueError(f"launch_bounds max_threads must be an int, got {max_threads!r}")
        if min_blocks is not None and (isinstance(min_blocks, bool) or not isinstance(min_blocks, int)):
            raise ValueError(f"launch_bounds min_blocks must be an int or None, got {min_blocks!r}")
    if max_threads < 1:
        raise ValueError(f"launch_bounds max_threads must be positive, got {max_threads}")
    if min_blocks is not None and min_blocks < 1:
        raise ValueError(f"launch_bounds min_blocks must be >= 1, got {min_blocks}")
    return (max_threads, min_blocks)


class LLKernel:

    def __init__(self, py_func, backend, is_entry, cluster_shape=None, cta_group_shape=None, no_alias=None,
                 read_only=None, min_blocks=None, launch_bounds=None, frontend=None):
        self.py_func = py_func
        self.frontend = frontend
        self.ctx = py_func.__globals__.copy()
        # Opt-in entry-kernel parameter annotations, consumed by CUDA codegen.
        # no_alias: params emitted as "T* __restrict__" (caller-side legality
        #   precondition: the annotated pointers never alias each other or any
        #   other pointer written through in this kernel).
        # read_only: params emitted as "const T* __restrict__" (implies the
        #   no_alias precondition AND that the kernel never writes through the
        #   pointer; lets ptxas schedule loads via the read-only data path).
        # Default (both omitted) is byte-identical to the historical codegen.
        no_alias = _normalize_param_names(no_alias, "no_alias")
        read_only = _normalize_param_names(read_only, "read_only")
        if no_alias:
            self.ctx[_NO_ALIAS_CTX_KEY] = no_alias
        if read_only:
            self.ctx[_READ_ONLY_CTX_KEY] = read_only
        # min_blocks: documented API for the launch-bounds occupancy hint;
        # overrides the legacy module-global ctx key when both are present.
        if min_blocks is not None:
            min_blocks = int(min_blocks)
            if min_blocks <= 0:
                raise ValueError(f"min_blocks must be a positive integer, got {min_blocks}")
            self.ctx[_MIN_BLOCKS_CTX_KEY] = min_blocks
        self.backend = backend
        # Optional explicit __launch_bounds__ declaration: (max_threads, min_blocks|None).
        # Validated at decoration time; applied to the entry __global__ signature only.
        self.launch_bounds = _normalize_launch_bounds(launch_bounds)
        # Shape structure is shared; backend capability limits are descriptor data.
        self.cluster_shape = _normalize_cluster_shape(cluster_shape)
        self.cta_group_shape = _normalize_cta_group_shape(cta_group_shape)
        if self.cta_group_shape is not None:
            if self.cluster_shape is None:
                raise ValueError("cta_group_shape requires cluster_shape")
            if any(c % g for c, g in zip(self.cluster_shape, self.cta_group_shape)):
                raise ValueError(f"cta_group_shape {self.cta_group_shape} must divide "
                                 f"cluster_shape {self.cluster_shape}")

        if any(value is not None and value != frozenset()
               for value in (self.cluster_shape, self.cta_group_shape, self.launch_bounds, no_alias, read_only,
                             min_blocks)):
            from little_kernel.backends.constraints import validate_declaration
            validate_declaration(
                backend,
                dict(cluster_shape=self.cluster_shape, cta_group_shape=self.cta_group_shape,
                     launch_bounds=self.launch_bounds, no_alias=no_alias, read_only=read_only, min_blocks=min_blocks))

        # Add closure variables to context only if little_kernel-related (whitelist).
        # Only ValueError is caught: cell.cell_contents raises ValueError when the cell is
        # empty/unbound. We do not catch other exceptions (explicit catch only).
        if py_func.__closure__ is not None:
            code = py_func.__code__
            for var_name, cell in zip(code.co_freevars, py_func.__closure__):
                try:
                    val = cell.cell_contents
                    if _is_little_kernel_related(val):
                        self.ctx[var_name] = val
                except ValueError:
                    # Cell is empty or unbound (e.g. not yet closed over); skip this freevar.
                    pass

        self.is_entry = is_entry
        # RF07: compilations never publish derived state back onto the
        # kernel. There is intentionally no last-result / last-context /
        # resolved-cache-dir attribute here: those would be last-session
        # races across re-specialization and concurrent calls. Every result
        # (tree *and* its per-call context) is returned to its caller.
        if is_entry:
            self.ctx[__LITTLE_KERNEL_ENTRY__] = self.py_func

    def _new_session(self, passes, constexpr_args=None, frontend=None):
        """Create a fresh isolated CompilationSession for one compilation."""
        from little_kernel.core.compilation import CompilationSession
        return CompilationSession(self, passes=passes, constexpr_args=constexpr_args,
                                  frontend=self.frontend if frontend is None else frontend)

    def compile(self, passes, codegen_func, need_header=True, num_threads=None, constexpr_args=None,
                disable_var_hoisting=False, cache_dir=None, no_alias=False, frontend=None, metadata=None,
                codegen_options=None):
        """Compile kernel to CUDA source text.

        Every invocation runs inside a fresh
        :class:`~little_kernel.core.compilation.CompilationSession`: source
        globals, user options (including nested mutable option containers)
        and pass-derived state are cloned per compilation, so constexpr
        re-specialization and concurrent calls never accumulate or share
        mutable state on ``self.ctx``. This is the source-only API: it does
        not require a launchable host ABI; use :meth:`compile_artifact` for
        the typed signature/build path.

        Parameters
        ----------
        passes : list or Pipeline
            Explicit list of pass callables (legacy callers keep working) or
            an immutable Pipeline; ``None`` selects the backend default.
        codegen_func : callable
            Code generation function (e.g. codegen_cuda).
        need_header : bool
            Whether to emit header includes.
        num_threads : int, optional
            Actual threads per block. Used to pair with the legacy ctx
            global and sanity-check an explicit launch_bounds declaration;
            by itself it does not emit __launch_bounds__.
        constexpr_args : dict, optional
            Constexpr parameter values for specialization.
        disable_var_hoisting : bool, optional
            Disable variable hoisting for if/for/while blocks.
        no_alias : bool, optional
            Emit non-aliasing (__restrict__) entry pointer parameters.
        frontend : None, str or callable, optional
            Explicit frontend/pipeline selection (e.g. the registered
            ``"dataflow"`` hook). ``None`` keeps the ordinary scalar/SIMT
            path; role passes are never enabled implicitly.
        metadata : mapping, optional
            Extra immutable artifact metadata (unused for the source return).

        Returns
        -------
        str
            Generated CUDA code
        """
        code, _result = self._compile_session(passes, codegen_func, need_header=need_header, num_threads=num_threads,
                                              constexpr_args=constexpr_args, disable_var_hoisting=disable_var_hoisting,
                                              cache_dir=cache_dir, no_alias=no_alias, frontend=frontend,
                                              require_abi=False, codegen_options=codegen_options)
        return code

    def compile_artifact(self, passes, codegen_func, target=None, need_header=True, num_threads=None,
                         constexpr_args=None, disable_var_hoisting=False, cache_dir=None, no_alias=False, frontend=None,
                         metadata=None, require_abi=True, codegen_options=None) -> CompilationArtifact:
        """Compile and return a merged :class:`CompilationArtifact`.

        Like :meth:`compile` but bundles generated source, kernel name, the
        actual launch ABI :class:`~little_kernel.runtime.abi.KernelSignature`
        (after constexpr parameter removal), target and immutable build
        metadata. The returned source is byte-identical to :meth:`compile`.
        """
        code, result = self._compile_session(passes, codegen_func, need_header=need_header, num_threads=num_threads,
                                             constexpr_args=constexpr_args, disable_var_hoisting=disable_var_hoisting,
                                             cache_dir=cache_dir, no_alias=no_alias, frontend=frontend,
                                             require_abi=require_abi, codegen_options=codegen_options)
        merged_metadata = self._artifact_metadata(result, num_threads=num_threads,
                                                  disable_var_hoisting=disable_var_hoisting, no_alias=no_alias,
                                                  extra=metadata)
        return CompilationArtifact(
            source=code,
            kernel_name=self.py_func.__name__,
            signature=result.signature,
            target=target,
            metadata=merged_metadata,
        )

    def _artifact_metadata(self, result, *, num_threads, disable_var_hoisting, no_alias, extra=None):
        data = {
            "backend": self.backend,
            "pass_ids": tuple(result.pass_ids),
            "frontend": result.frontend,
            "constexpr_args": tuple(sorted(result.constexpr_args.items())),
            "launch_bounds": self.launch_bounds,
            "cluster_shape": self.cluster_shape,
            "cta_group_shape": self.cta_group_shape,
            "num_threads": num_threads,
            "disable_var_hoisting": bool(disable_var_hoisting),
            "no_alias": bool(no_alias),
        }
        if extra:
            if not isinstance(extra, Mapping):
                raise TypeError("metadata must be a mapping")
            data.update(extra)
        return data

    def _compile_session(self, passes, codegen_func, *, need_header=True, num_threads=None, constexpr_args=None,
                         disable_var_hoisting=False, cache_dir=None, no_alias=False, frontend=None, require_abi=False,
                         codegen_options=None):
        """Run one isolated session; return ``(source, CompilationResult)``.

        ``require_abi`` is False for the source-only :meth:`compile` path (a
        typed ABI is not required to emit source) and True for
        :meth:`compile_artifact`/:meth:`build`, where an unsupported typed
        layout or invalid adapter must fail honestly instead of silently
        producing an untyped launchable artifact.
        """
        session = self._new_session(passes, constexpr_args=constexpr_args, frontend=frontend)
        result = session.lower(require_abi=require_abi)
        # Legality: a declared launch_bounds max_threads below the actual
        # block size would make the kernel unlaunchable; reject before text.
        if self.launch_bounds is not None and num_threads is not None:
            if num_threads > self.launch_bounds[0]:
                raise ValueError(f"launch_bounds max_threads={self.launch_bounds[0]} is "
                                 f"below the actual block size {num_threads}; the kernel "
                                 "could never launch")
        kwargs = {'emit_header': need_header}
        if num_threads is not None:
            kwargs['num_threads'] = num_threads
        if self.launch_bounds is not None:
            kwargs['launch_bounds'] = self.launch_bounds
        if disable_var_hoisting:
            kwargs['disable_var_hoisting'] = disable_var_hoisting
        if no_alias:
            kwargs['no_alias'] = no_alias
        # Codegen receives a fresh copy of THIS session's full derived
        # context so inference/codegen can read lowered symbols. The
        # kernel's own ctx is the immutable user source-globals/options
        # baseline and is never replaced, stripped or enriched.
        kwargs.update(codegen_options or {})
        code = codegen_func(result.tree, dict(result.context), **kwargs)
        # Resolve the cache directory locally for this call only; never
        # stash it on the kernel where another compilation could overwrite
        # it before this build reads it back.
        resolved_cache_dir = _lk_cache_dir(cache_dir)
        if resolved_cache_dir:
            from little_kernel.backends import get_backend
            _write_source(resolved_cache_dir, self.py_func.__name__, code, constexpr_args=constexpr_args,
                          source_suffix=get_backend(self.backend).toolchain.source_suffix)
        return code, result

    def lower_result(self, passes, constexpr_args=None, frontend=None, require_abi=False):
        """Lower one specialization in an isolated session and return the
        full :class:`~little_kernel.core.compilation.CompilationResult`.

        The result carries the lowered tree **and** this call's own derived
        context (plus signature/pass metadata). Nothing is published on the
        shared kernel, so recursive device-function lowering and concurrent
        calls never communicate through a mutable "last" context.
        """
        session = self._new_session(passes, constexpr_args=constexpr_args, frontend=frontend)
        return session.lower(require_abi=require_abi)

    def lower_with_context(self, passes, constexpr_args=None, frontend=None, require_abi=False):
        """Return ``(lowered_tree, per_call_context_dict)`` for one call.

        This is the race-free replacement for the old "lower then read
        ``kernel.ctx``" pattern: the returned context belongs to this
        lowering only.
        """
        result = self.lower_result(passes, constexpr_args=constexpr_args, frontend=frontend, require_abi=require_abi)
        return result.tree, dict(result.context)

    def lower(self, passes, constexpr_args=None, frontend=None):
        """Compatibility API: lower and return the AST only.

        This never mutates ``self.ctx`` and does not publish a "last"
        context. Callers that need the lowered context (recursive codegen /
        type inference) must use :meth:`lower_result` or
        :meth:`lower_with_context`.
        """
        return self.lower_result(passes, constexpr_args=constexpr_args, frontend=frontend).tree

    def __call__(self, *args, **kwargs):
        self.ctx[__LITTLE_KERNEL_ENTRY__] = self.py_func

    def build(
        self,
        passes,
        codegen_func,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        shared_mem_bytes: int = 0,
        arch: Optional[str] = None,
        cache_dir: Optional[str] = None,
        verbose: bool = False,
        include_paths: Optional[List[Union[str, Path]]] = None,
        cluster_shape: Optional[Tuple[int, int, int]] = None,
        extra_nvcc_options: Optional[List[str]] = None,
        allow_non_portable_cluster: bool = False,
        disable_var_hoisting: bool = False,
        frontend=None,
        signature=None,
        extra_hipcc_options=None,
        waves_per_eu=None,
    ):
        """
        Compile and build a kernel that can be launched.
        
        Parameters
        ----------
        passes : list
            List of passes to apply.
        codegen_func : callable
            Code generation function (e.g., codegen_cuda).
        grid : Tuple[int, int, int]
            Grid dimensions (gridDimX, gridDimY, gridDimZ).
        block : Tuple[int, int, int]
            Block dimensions (blockDimX, blockDimY, blockDimZ).
        shared_mem_bytes : int
            Dynamic shared memory size in bytes.
        arch : Optional[str]
            Target architecture (e.g., "sm_90a").
        cache_dir : Optional[str]
            Directory to cache compiled binaries.
        verbose : bool
            Print compilation output.
        include_paths : Optional[List[Union[str, Path]]]
            List of include paths to add to nvcc command.
        cluster_shape : Optional[Tuple[int, int, int]]
            Cluster dimensions (x, y, z) for SM90+ cluster launch.
        extra_nvcc_options : Optional[List[str]]
            Extra nvcc options (e.g. ["--expt-relaxed-constexpr"]).
        allow_non_portable_cluster : bool
            Set NonPortableClusterSizeAllowed=1 (needed for TMA shared::cluster).
        frontend : None, str or callable, optional
            Explicit frontend/pipeline selection hook (role/dataflow). None
            keeps the ordinary scalar/SIMT pipeline.
        signature : KernelSignature, optional
            Explicit launch ABI signature for a raw-source caller. When the
            frontend annotations produce a signature (constexpr parameters
            removed), that actual signature always wins; an explicitly
            untyped kernel (no annotations) may use this override only when
            no typed signature was derived -- a legacy untyped path can
            never replace a known one.

        Returns
        -------
        CompiledKernel
            Compiled kernel that can be launched.
        """
        from little_kernel.backends import BuildRequest, get_backend
        backend = get_backend(self.backend, architecture=arch)
        selected_arch = arch
        if selected_arch is None and self.backend == backend.backend_id:
            selected_arch = backend.target.resolve().arch
        options = dict(cluster_shape=cluster_shape, extra_nvcc_options=extra_nvcc_options,
                       allow_non_portable_cluster=allow_non_portable_cluster, disable_var_hoisting=disable_var_hoisting,
                       frontend=frontend, signature=signature, extra_hipcc_options=extra_hipcc_options,
                       waves_per_eu=waves_per_eu)
        return backend.builder.build(
            BuildRequest(self, passes, codegen_func, grid, block, shared_mem_bytes, selected_arch, cache_dir, verbose,
                         include_paths, options))


def ll_kernel(backend="cuda", is_entry=False, cluster_shape=None, cta_group_shape=None, no_alias=None, read_only=None,
              min_blocks=None, launch_bounds=None, frontend=None):
    """LittleKernel kernel decorator.

    Parameters
    ----------
    backend : str
        Target backend (e.g., "nv/sm90", "nv/sm100", or "amd/cdna4").
    is_entry : bool
        Whether this kernel is the entry (__global__) function.
    frontend : callable or registered name, optional
        Explicit frontend used by compilation calls unless they supply an override.
    cluster_shape / cta_group_shape : tuple, optional
        SM90+ cluster configuration.
    no_alias : iterable of str, optional
        Entry-kernel pointer/tensor parameter names emitted with
        ``__restrict__``. Caller-side legality precondition: annotated
        pointers never alias each other or any storage written through in
        this kernel (same contract style as the 16B-alignment precondition
        on the vector GMEM intrinsics). Default: no annotation (historical
        behavior).
    read_only : iterable of str, optional
        Entry-kernel pointer/tensor parameter names emitted as
        ``const T* __restrict__``. Precondition: the no_alias contract AND
        the kernel never writes through these pointers.
    min_blocks : int, optional
        Minimum blocks per SM for ``__launch_bounds__(num_threads,
        min_blocks)`` on the CUDA entry kernel. Default: no bound beyond
        what the backend already emits. Overrides the legacy
        ``__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__`` module global.
    launch_bounds : int or (int, int), optional
        Explicit __launch_bounds__ declaration for the entry __global__ function, as
        (max_threads_per_block[, min_blocks_per_multiprocessor]). Omitted by default:
        ptxas then chooses its own register budget (unchanged legacy behavior).
        Note: an explicit launch_bounds here takes precedence over min_blocks in
        codegen, mirroring the backend's legacy precedence rules.
    """

    def _compile_helper(func):
        compiled_func = LLKernel(func, backend, is_entry, cluster_shape, cta_group_shape, no_alias=no_alias,
                                 read_only=read_only, min_blocks=min_blocks, launch_bounds=launch_bounds,
                                 frontend=frontend)

        compiled_func.__name__ = func.__name__
        compiled_func.__doc__ = func.__doc__
        compiled_func.__module__ = func.__module__
        return compiled_func

    return _compile_helper
