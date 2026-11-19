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
import re
import textwrap
from little_kernel.codegen.codegen_base import CppEmitter
import little_kernel.language as ll
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.backends.nv.common.core.types import normalize_entry_grid_constant_param
from little_kernel.backends.nv.common.language.intrin.dtype import half_float_cast_code
from little_kernel.codegen.visitors.call_codegen import _emit_builtin_includes

# ctx keys consumed by the CUDA entry-kernel signature lowering. Populated
# by the ll_kernel decorator kwargs (see little_kernel/core/compile.py); the
# legacy spellings as module-level globals in the frontend file keep working.
NO_ALIAS_CTX_KEY = "__LITTLE_KERNEL_NO_ALIAS_PARAMS__"
READ_ONLY_CTX_KEY = "__LITTLE_KERNEL_READ_ONLY_PARAMS__"
MIN_BLOCKS_CTX_KEY = "__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__"
# Full export/lower-path marker: when present, the entry kernel marks every
# pointer/tensor param not declared may-alias __restrict__ (alias-contract
# policy). Render-layer callers that do not set it keep opt-in-only
# __restrict__ (ctor no_alias=True), so default renders are byte-identical
# to the pre-contract behavior.
ALIAS_CONTRACT_CTX_KEY = "__LITTLE_KERNEL_CUDA_ALIAS_CONTRACT__"

# Public alias for integration modules (merged from incoming branch): the
# frontend-visible context key that requests a minimum blocks-per-SM floor for
# the entry kernel's __launch_bounds__(num_threads, min_blocks). A frontend
# may set it as a module-level global; integrations (e.g. the export spec's
# launch.max_blocks_per_sm field) may set it directly on kernel.ctx.
CUDA_LAUNCH_BOUNDS_MIN_BLOCKS_CTX_KEY = MIN_BLOCKS_CTX_KEY

# Opt-in aliasing hint knob (dff458fb): a frontend may define the module-level
# constant __LITTLE_KERNEL_CUDA_RESTRICT_PARAMS__ = ("name1", "name2", ...)
# and any function parameter with a listed name whose LLType renders as a C
# pointer (Tensor or Pointer) is emitted as `T* __restrict__ name`.
#
# Legality contract (mirrors the __ldg/__ldcs read-only contract): the
# FRONTEND asserts the listed tensors never overlap any other tensor
# parameter for the kernel's lifetime. The compiler performs no aliasing
# analysis; unlisted parameters keep the un-annotated spelling.
_RESTRICT_PARAMS_KEY = "__LITTLE_KERNEL_CUDA_RESTRICT_PARAMS__"


def _ll_known_names():
    """Names provided by the little_kernel.language namespace (intrinsics).

    Calls to these are analyzable locally (store flavors are matched via
    CudaEmitter._MUTATING_CALL_HINTS); calls to unknown bare names are not.
    """
    try:
        return frozenset(dir(ll))
    except Exception:
        return frozenset()


_KNOWN_LL_NAMES = _ll_known_names()
_PY_BUILTINS = frozenset({
    "max",
    "min",
    "int",
    "float",
    "len",
    "abs",
    "round",
    "print",
    "range",
    "enumerate",
    "zip",
    "divmod",
    "pow",
    "bool",
    "str",
    "tuple",
    "list",
    "float32",
    "float64",
    "int32",
    "int64",
    "uint32",
    "uint64",
    "bfloat16",
    "float16",
    "int8",
    "int16",
    "uint8",
    "uint16",
    "void",
})


def _normalize_launch_bounds_signature(launch_bounds):
    """Coerce a __launch_bounds__ spec to (max_threads, min_blocks_or_None) or None.

    Full user-facing validation lives in core.compile._normalize_launch_bounds (at
    decoration time); this local form only guards direct emitter/codegen_cuda callers
    against malformed values so a bad spec cannot silently corrupt the signature.
    """
    if launch_bounds is None:
        return None
    if isinstance(launch_bounds, bool):
        raise ValueError(f"launch_bounds must be an int or (int, int) tuple, got {launch_bounds!r}")
    if isinstance(launch_bounds, int):
        max_threads, min_blocks = launch_bounds, None
    else:
        parts = tuple(launch_bounds)
        if len(parts) == 1:
            max_threads, min_blocks = parts[0], None
        elif len(parts) == 2:
            max_threads, min_blocks = parts
        else:
            raise ValueError(f"launch_bounds tuple must have 1 or 2 elements, got {launch_bounds!r}")
    if not isinstance(max_threads, int) or isinstance(max_threads, bool) or not (1 <= max_threads <= 1024):
        raise ValueError(f"launch_bounds max_threads must be an int in [1, 1024], got {max_threads!r}")
    if min_blocks is not None and (not isinstance(min_blocks, int) or isinstance(min_blocks, bool) or min_blocks < 1):
        raise ValueError(f"launch_bounds min_blocks must be an int >= 1, got {min_blocks!r}")
    return (max_threads, min_blocks)


class CudaEmitter(CppEmitter):
    """Unified CUDA code emitter.

    Generates standalone CUDA code using raw PTX intrinsics.
    No CuTe/CUTLASS dependency. Uses CUtensorMap directly for TMA descriptors.
    """

    def _add_basic_headers(self) -> None:
        self.header_cache.add("<cstdint>")
        self.header_buffer.write("#include <cstdint>\n")

    def _native_scalar_conversion(self, value_cpp, source_type, destination_type):
        return half_float_cast_code(value_cpp, source_type, destination_type)

    def __init__(self, ctx, all_types, num_threads=None, disable_var_hoisting=False, launch_bounds=None,
                 no_alias=False):
        super().__init__(ctx, all_types)
        self.tuple_namespace = "cuda::std"
        self.half_arithmetic_header = "<cuda_fp16.h>"
        self.num_threads = num_threads
        self.launch_bounds = _normalize_launch_bounds_signature(launch_bounds)
        self.disable_var_hoisting = disable_var_hoisting
        # Opt-in: emit __restrict__ on the entry kernel's pointer/tensor params.
        # The caller guarantees the kernel's pointer arguments do not alias;
        # spec-level validation (integrations/vibe_cuda_eval._load_spec) rejects
        # the flag when combined with alias_of outputs. Default False keeps
        # generated source byte-identical for all existing programs.
        self.no_alias = no_alias
        # CUDA requires integer constant expressions for __shared__ and local
        # array extents: provably-static integer locals are emitted as
        # `static constexpr` (analysis: visitors/static_int_analysis.py), and
        # array-allocation builtins reject non-constant extents early with a
        # source-located diagnostic instead of emitting VLA text nvcc rejects.
        self._emit_static_constexpr = True
        self._require_constant_array_extents = True
        # Scalar fp16 +,-,* legalize to __hadd/__hsub/__hmul: the pinned
        # nvcc build compiles the __half operator overloads out via
        # -D__CUDA_NO_HALF_OPERATORS__, while these explicit intrinsics
        # remain available (single-round native fp16 arithmetic).
        self._legalize_fp16_scalar_arith = True
        # Tensor subscripts whose index expression is IR-typed 64-bit emit an
        # explicitly wide element offset (base[(ptrdiff_t)(i)]) so the C
        # usual-arithmetic conversions can never truncate a > INT_MAX element
        # index through a 32-bit intermediate on the way to the address
        # (strictly an explicit-width guarantee: C already promotes a plain
        # int64 index operand, but the emitted form no longer depends on any
        # host-toolchain reinterpretation of the subscript expression).
        # 32-bit-typed and untyped indices keep the historical emission.
        self._wide_i64_subscript_offset = True
        self.header_cache.add("<cuda_runtime.h>")
        self.header_buffer.write("#include <cuda_runtime.h>\n")
        self._conditional_headers_emitted = False

    def _convert_value_for_store(self, value_cpp: str, src_lltype, dst_lltype) -> str:
        """Legalize typed subscript stores whose value type differs from the
        element type.

        Typed store validation accepts C++-style promotion of the value onto
        the element type, so e.g. a float32 literal stored into a bfloat16
        tensor reaches codegen still typed float32 and would emit the
        ill-formed ``__nv_bfloat16 = float`` assignment (torch extension
        builds define the no-conversion macros).  Route the value through
        ``half_float_cast_code`` — the same RNE intrinsic lowering used by
        val_cast/to/type constructors — so the store becomes an explicit
        ``__float2bfloat16_rn``/``__float2half_rn`` conversion.  Same-type
        stores (including values already converted via ll.float_to_bf16),
        non-16-bit targets, and untyped values are unchanged.
        """
        conv = half_float_cast_code(value_cpp, src_lltype, dst_lltype)
        if conv is None:
            return value_cpp
        code, includes = conv
        _emit_builtin_includes(self, includes)
        return code

    def _lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a valid C++ type string."""
        # ------------------------------ Scalar Types ------------------------------
        if hasattr(ll_type, "kind") and ll_type.kind == "int":  # IntType
            if ll_type.special == "bool":
                return "bool"
            if ll_type.special == "binary" or ll_type.bits == 4:
                # TODO: Add proper 4-bit (int4/uint4) support. CUDA has native 4-bit types.
                raise NotImplementedError("4-bit integer types (int4/uint4, binary) are not yet supported. "
                                          "CUDA has native 4-bit support; this will be added in a future release.")
            # Map to stdint types (e.g., uint32 -> uint32_t)
            sign_prefix = "u" if not ll_type.signed else ""
            bit_map = {
                8: f"{sign_prefix}int8_t", 16: f"{sign_prefix}int16_t", 32: f"{sign_prefix}int32_t", 64:
                f"{sign_prefix}int64_t", 128: "unsigned __int128" if not ll_type.signed else "__int128"
            }
            return bit_map[ll_type.bits]

        elif hasattr(ll_type, "kind") and ll_type.kind == "float":  # FloatType
            float_map = {
                "fp4_e2m1": "__nv_fp4_e2m1", "fp8_e5m2": "__nv_fp8_e5m2", "fp8_e4m3": "__nv_fp8_e4m3", "bfloat16":
                "__nv_bfloat16", "float16": "__half",
                # "tfloat32": "tfloat32_t",
                "float32": "float", "float64": "double"
            }
            return float_map[ll_type.fmt]

        elif hasattr(ll_type, "kind") and ll_type.kind == "vector":  # VectorType
            # Fixed-width vector register values map onto CUDA's native vector
            # types. Only combinations that are natively supported by CUDA and
            # that have dedicated bit-level lane intrinsics are legal here.
            vector_map = {
                ("uint32", 2): "uint2",
                ("uint32", 4): "uint4",
            }
            key = (str(ll_type.element_type), ll_type.lanes)
            if key not in vector_map:
                raise NotImplementedError(f"No native CUDA vector type for LLType {ll_type!r}; "
                                          f"supported: {[f'{e}x{n}' for e, n in sorted(vector_map)]}")
            return vector_map[key]

        elif hasattr(ll_type, "kind") and ll_type.kind == "void":  # VoidType
            return "void"

        elif hasattr(ll_type, "kind") and ll_type.kind == "str":  # StringType
            return "std::string"

        # ------------------------------ Annotated Types ------------------------------
        elif ll_type.is_const():  # Const/GridConstant
            inner_cpp = self._lltype_to_cpp(ll_type.inner_type)
            return f"__grid_constant__ {inner_cpp}" if ll_type.is_grid_constant() else f"const {inner_cpp}"

        elif ll_type.is_pointer():  # Pointer
            inner_cpp = self._lltype_to_cpp(ll_type.inner_type)
            return f"{inner_cpp}*"

        # ------------------------------ Composite Types ------------------------------
        elif ll_type.is_tensor():  # TensorType (map to pointer for simplicity)
            elem_cpp = self._lltype_to_cpp(ll_type.element_type)
            return f"{elem_cpp}*"

        elif ll_type.is_struct():  # StructType
            if ll_type not in self.struct_types:
                self.struct_types[ll_type] = self._generate_struct_name(ll_type)
            return self.struct_types[ll_type]

        elif ll_type.is_tuple():  # TupleType (e.g. return type for device functions)
            elem_cpps = [self._lltype_to_cpp(t) for t in ll_type.element_types]
            # libcu++ tuple: usable in device code (std::tuple's members are
            # host-only under libstdc++/nvcc).
            return f"cuda::std::tuple<{', '.join(elem_cpps)}>"

        elif str(ll_type) == "TmaDescriptor":  # TmaDescriptorType
            return "TmaDescriptor"

        elif str(ll_type) == "Wgmma":  # WgmmaType
            raise NotImplementedError("WgmmaType is not supported for C++ generation")

        # ------------------------------ Unsupported LLType ------------------------------
        else:
            raise NotImplementedError(f"Unsupported LLType: {type(ll_type).__name__} (value: {str(ll_type)})")

    # Callee-name hints indicating a call may write through a pointer argument.
    # A pointer/tensor param passed to a matching call is treated as mutated
    # (loses `const`). Inline asm is checked against _ASM_STORE_PATTERN.
    # Over-marking is harmless (param stays non-const); a *missed* store
    # degrades to an nvcc *compile error* (write through const pointer),
    # never silent wrong code.
    _MUTATING_CALL_HINTS = (
        "st_",
        "stmatrix",
        "stsm",
        "atomic",
        "atom_",
        "red_",
        "store",
        "copy",
        "fill",
        "memset",
        "cp_async",
        "tma_store",
    )
    # PTX instructions that write memory as their first operand.
    _ASM_STORE_PATTERN = re.compile(r"\b(st\.|stmatrix|stsm|red\.|atom\.|atomic[s]?\s|cp\.async)")

    @staticmethod
    def _is_known_callee(func, callee: str) -> bool:
        """True when a call target resolves to the ll/lk/math namespace, a
        known language intrinsic, or a Python builtin (none of which can
        store through a pointer argument in an unseen way)."""
        if isinstance(func, ast.Attribute):
            root = func.value
            while isinstance(root, ast.Attribute):
                root = root.value
            return isinstance(root, ast.Name) and root.id in ("ll", "lk", "math")
        if isinstance(func, ast.Name):
            return func.id in _KNOWN_LL_NAMES or func.id in _PY_BUILTINS
        return False

    @staticmethod
    def _base_name(expr):
        """Root Name of an lvalue chain (a[i].b -> a)."""
        while isinstance(expr, (ast.Subscript, ast.Attribute)):
            expr = expr.value
        return expr.id if isinstance(expr, ast.Name) else None

    @staticmethod
    def _names_in(expr):
        for sub in ast.walk(expr):
            if isinstance(sub, ast.Name):
                yield sub.id

    def _find_mutated_pointer_params(self, node: ast.FunctionDef, user_callees=frozenset()) -> set:
        """Names of params this function may write through (or rebind).

        Conservative syntactic scan: a param is mutated if it (or an
        expression derived from it) is an assignment target, or appears
        anywhere inside the arguments of a store-flavored intrinsic call
        (see _MUTATING_CALL_HINTS) or a store-containing inline asm.
        Calls to module-local user functions (in `user_callees`) are NOT
        handled here; exact cross-function propagation happens in
        _compute_param_mutation instead. Used only to decide `const` on
        pointer params; never changes semantics beyond nvcc-enforced
        const-correctness.
        """
        mutated: set = set()

        for sub in ast.walk(node):
            if isinstance(sub, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (sub.targets if isinstance(sub, ast.Assign) else [sub.target])
                for tgt in targets:
                    name = self._base_name(tgt)
                    if name is not None:
                        mutated.add(name)
            elif isinstance(sub, ast.Call):
                func = sub.func
                callee = func.attr if isinstance(func,
                                                 ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
                if callee in user_callees:
                    continue  # handled by cross-function fixpoint
                if callee == "asm":
                    template = (sub.args[0].value if sub.args and isinstance(sub.args[0], ast.Constant)
                                and isinstance(sub.args[0].value, str) else "")
                    if not template or self._ASM_STORE_PATTERN.search(template):
                        for arg in list(sub.args) + [kw.value for kw in sub.keywords]:
                            mutated.update(self._names_in(arg))
                elif any(hint in callee for hint in self._MUTATING_CALL_HINTS):
                    for arg in list(sub.args) + [kw.value for kw in sub.keywords]:
                        mutated.update(self._names_in(arg))
        return mutated

    def _compute_param_mutation(self, module: ast.Module) -> dict:
        """Function name -> set of possibly-mutated param names.

        Cross-function fixpoint: a caller param is mutated when the callee's
        corresponding (module-local) param is mutated. Conservative for
        unknown callees: pointer args passed to a non-intrinsic call we
        cannot resolve are marked mutated, so const is never claimed through
        an unanalyzable path (it is claimed only when syntactically read-only
        is provable from the module).
        """
        funcs = {}
        for stmt in module.body:
            if isinstance(stmt, ast.FunctionDef):
                funcs[stmt.name] = stmt
        names = set(funcs)
        mutated = {name: self._find_mutated_pointer_params(node, user_callees=names) for name, node in funcs.items()}
        changed = True
        while changed:
            changed = False
            for caller_name, node in funcs.items():
                caller_mut = mutated[caller_name]
                for sub in ast.walk(node):
                    if not isinstance(sub, ast.Call):
                        continue
                    func = sub.func
                    callee = func.attr if isinstance(func,
                                                     ast.Attribute) else (func.id if isinstance(func, ast.Name) else "")
                    if callee in funcs:
                        callee_node = funcs[callee]
                        callee_params = [a.arg for a in callee_node.args.args]
                        callee_mut = mutated[callee]
                        before = len(caller_mut)
                        # Positional mapping
                        for i, arg in enumerate(sub.args):
                            if i < len(callee_params) and callee_params[i] in callee_mut:
                                caller_mut.update(self._names_in(arg))
                        # Keyword mapping by param name
                        for kw in sub.keywords:
                            if kw.arg is not None and kw.arg in callee_mut:
                                caller_mut.update(self._names_in(kw.value))
                        changed |= len(caller_mut) != before
                    else:
                        # Unresolvable callee: any pointer arg we cannot
                        # prove read-only conservatively loses const.
                        if not self._is_known_callee(func, callee):
                            before = len(caller_mut)
                            for arg in list(sub.args) + [kw.value for kw in sub.keywords]:
                                caller_mut.update(self._names_in(arg))
                            changed |= len(caller_mut) != before
        return mutated

    def visit_Module(self, node: ast.Module) -> None:
        """Pre-compute cross-function param mutation, then defer to mixin."""
        self._param_mutation_map = self._compute_param_mutation(node)
        super().visit_Module(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definition: initialize scope with parameters, then process body."""
        # Resolve return type
        return_lltype = self.get_type(node.returns)
        return_cpp = self._lltype_to_cpp(return_lltype)

        # Pointer/tensor params get `const` when this function (and its
        # module-local callees) never stores through them; the `__global__`
        # entry kernel additionally gets `__restrict__` on every
        # pointer/tensor param not declared may-alias. Two generic
        # declaration channels compose here (merge of HEAD inference and
        # 70ad465e explicit annotations):
        #   1. automatic: cross-function store analysis marks never-stored
        #      params const; the entry kernel marks params __restrict__ unless
        #      the alias contract (vibe export `_contract_may_alias_params`)
        #      or the frontend escape hatch
        #      __LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__ declares may-alias;
        #   2. explicit: the build API (core/compile.py no_alias=/read_only=)
        #      injects NO_ALIAS_CTX_KEY/READ_ONLY_CTX_KEY; annotated params
        #      must be entry params, name a real parameter, and lower to a
        #      pointer type or codegen raises (legality diagnostics, never
        #      silent misannotation).
        # Legality contract: entry tensor params are distinct host tensors
        # (out-of-place ABI), so restrict is sound for non-may-alias params;
        # const is verified by nvcc at compile time, so a false readonly
        # conclusion is a hard error, not a miscompile.
        is_entry = (__LITTLE_KERNEL_ENTRY__ in self.ctx and self.ctx[__LITTLE_KERNEL_ENTRY__].__name__ == node.name)
        mutation_map = getattr(self, "_param_mutation_map", None)
        if mutation_map is not None and node.name in mutation_map:
            mutated_params = mutation_map[node.name]
        else:
            mutated_params = self._find_mutated_pointer_params(node)
        no_alias = frozenset(self.ctx.get(NO_ALIAS_CTX_KEY) or ())
        read_only = frozenset(self.ctx.get(READ_ONLY_CTX_KEY) or ())
        if is_entry:
            # Legality diagnostic: annotations must name real parameters.
            param_names = {arg.arg for arg in node.args.args}
            unknown = (no_alias | read_only) - param_names
            if unknown:
                raise ValueError(f"no_alias/read_only names {sorted(unknown)} are not parameters "
                                 f"of entry kernel {node.name!r}")
        # is_entry_fn kept as a historical alias of is_entry (29d2c834 merge).
        is_entry_fn = is_entry  # noqa: F841

        # Resolve parameters and populate initial scope
        restrict_raw = (self.ctx or {}).get(_RESTRICT_PARAMS_KEY) or ()
        restrict_names = set(restrict_raw)
        if not all(isinstance(n, str) for n in restrict_names):
            raise TypeError(f"{_RESTRICT_PARAMS_KEY} must be a collection of parameter "
                            f"name strings, got {restrict_raw!r}")
        if not hasattr(self, "_restrict_unmatched"):
            # Tracks listed names that never matched a parameter across the
            # whole module, so a typo fails loudly in get_code().
            self._restrict_unmatched = set(restrict_names)

        cpp_params = []
        old_scope_vars = self.scope_vars
        self.scope_vars = {}  # Reset scope for new function
        # Entry kernels mark tensor/pointer parameters __restrict__: frontends
        # hand each kernel a disjoint set of device allocations, and without
        # the qualifier nvcc must conservatively order every load after every
        # store to any tensor, preventing automatic load batching (measured
        # 0.3-2% on copy-style schedules).  __restrict__ is a noalias PROOF
        # OBLIGATION, not a free qualifier: a parameter opts out whenever the
        # ABI can deliver a shared allocation through it.  Two generic sources
        # opt out per parameter:
        #   1. The vibe-cuda-eval export contract: an output with "alias_of"
        #      or an "alias" transform binds the same tensor under two names,
        #      so integrations/vibe_cuda_eval.py adds every transitively
        #      linked kernel parameter to this set automatically
        #      (_contract_may_alias_params); fresh at::empty outputs and
        #      workspaces and independent inputs stay __restrict__.
        #   2. A frontend that is passed two overlapping buffers through
        #      distinct parameters opts out itself with the module-level
        #      escape hatch
        #        __LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__ = ("x", "y")
        #      (same module-global config pattern as
        #      __LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__).
        may_alias = set(self.ctx.get("__LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__", ()))
        # Alias-contract policy channel: active on the full export/lower path
        # (integrations/vibe_cuda_eval._render_submission sets the key), where
        # the export contract hands each parameter its own device allocation
        # and per-parameter __restrict__ is the accepted default. Render-layer
        # callers (direct codegen/_render_cuda use) leave the key unset and
        # keep the incoming branch's opt-in semantics: __restrict__ only via
        # the explicit ctor no_alias=True knob. The audit/aliased bookkeeping
        # below is contract-only, so opt-in renders carry no policy comment
        # and a flag-off render stays byte-identical to a flag-absent one.
        alias_contract = bool(self.ctx.get(ALIAS_CONTRACT_CTX_KEY, False))
        restricted, aliased = [], []
        for arg in node.args.args:
            arg_lltype = self.get_type(arg.annotation)
            if is_entry:
                # Bare-shorthand entry parameters (TmaDescriptor) name the
                # by-value grid-constant singleton transport; resolve them to
                # the explicit grid_constant[...] form here so the emitted
                # signature is the CUDA-legal `__grid_constant__ TmaDescriptor`
                # and the type the body scope sees matches the export
                # validator's accepted spelling.
                arg_lltype = normalize_entry_grid_constant_param(arg_lltype)
            arg_cpp = self._lltype_to_cpp(arg_lltype)
            annotated = is_entry and (arg.arg in no_alias or arg.arg in read_only)
            # By-value grid-constant structs (the singleton TMA descriptor ABI,
            # const[grid_constant[TmaDescriptor]]) are not pointer/token params
            # but are legal annotation targets: _cuda.py mandates this exact
            # type for singleton descriptors, so the annotation channels must
            # not reject what the ABI check accepted. __restrict__ is still
            # never appended below (the param is by-value, not a pointer).
            gc_type = arg_lltype
            while gc_type.is_const() and not gc_type.is_grid_constant():
                gc_type = gc_type.inner_type
            is_byvalue_grid_constant = gc_type.is_grid_constant()
            if annotated and not (arg_cpp.rstrip().endswith("*") or is_byvalue_grid_constant):
                # Pointer decoration legality (70ad465e): annotations on
                # scalar/struct params raise instead of misannotating.
                raise ValueError(f"no_alias/read_only parameter {arg.arg!r} of entry kernel "
                                 f"{node.name!r} does not lower to a pointer type (got {arg_cpp})")
            if arg.arg in restrict_names and not (arg_lltype.is_pointer() or arg_lltype.is_tensor()):
                raise TypeError(f"{_RESTRICT_PARAMS_KEY}: parameter '{arg.arg}' of "
                                f"function '{node.name}' is annotated but its type "
                                f"({arg_lltype}) does not render as a C pointer; "
                                "__restrict__ only applies to Tensor or Pointer "
                                "parameters")
            if arg_lltype.is_pointer() or arg_lltype.is_tensor():
                # `const` from either proof channel: explicit read_only
                # annotation (frontend proof obligation, like __restrict__)
                # or the never-stored cross-function analysis; nvcc verifies
                # the qualifier either way, so a false conclusion is a hard
                # compile error, not a miscompile.
                qual = ("const " if (arg.arg in read_only or arg.arg not in mutated_params) else "")
                if is_entry:
                    # The constructor no_alias=True knob (incoming-branch
                    # callers) is a blanket frontend assertion equivalent to
                    # clearing may_alias for the whole entry kernel; with the
                    # default False the per-parameter contract applies only
                    # when the export alias-contract policy channel is active.
                    if self.no_alias or (alias_contract and arg.arg not in may_alias):
                        # __restrict__ subsumes the explicit no_alias channel:
                        # every may-alias-free entry pointer/tensor param is
                        # asserted noalias under the out-of-place ABI. This
                        # also covers dff458fb's _RESTRICT_PARAMS_KEY opt-in
                        # for entry params (both are frontend noalias claims).
                        arg_cpp = f"{arg_cpp} __restrict__"
                        restricted.append(arg.arg)
                    elif alias_contract:
                        aliased.append(arg.arg)
                elif arg.arg in restrict_names:
                    # dff458fb knob: non-entry (device) functions are outside
                    # the entry alias policy, so honor the opt-in here.
                    arg_cpp = f"{arg_cpp} __restrict__"
                self._restrict_unmatched.discard(arg.arg)
                cpp_params.append(f"{qual}{arg_cpp} {arg.arg}")
            else:
                cpp_params.append(f"{arg_cpp} {arg.arg}")
            # Add parameter to scope (prevents re-declaration in body)
            self.scope_vars[arg.arg] = arg_lltype

        # Preserve the author's semantic description in the generated source.
        docstring = ast.get_docstring(node)
        if docstring:
            for paragraph in docstring.splitlines():
                if paragraph.strip():
                    for line in textwrap.wrap(paragraph.strip(), width=88):
                        self.writeln_main(f"// {line}")
                else:
                    self.writeln_main("//")

        if is_entry and aliased:
            # Alias-policy audit line: the generated TU self-documents which
            # entry parameters carry __restrict__ (the export contract hands
            # each of them its own device allocation, so noalias holds) and
            # which were exempted because the ABI can deliver a shared
            # allocation through them (export-contract "alias_of"/"alias"
            # transform, or the frontend escape hatch
            # __LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__). Emitted only when an
            # opt-out is active, so programs without aliasing constructs are
            # byte-identical to before.
            self.writeln_main("// alias-policy: __restrict__ (noalias) asserted on [{}]; "
                              "not asserted on [{}] (contract/frontend-declared "
                              "may-alias).".format(", ".join(restricted) or "-", ", ".join(aliased)))

        # Write function signature
        sig = f"{return_cpp} {node.name}({', '.join(cpp_params)}) {{"
        if is_entry:
            # Only constrain register allocation when the frontend explicitly asks
            # for it: via the @ll_kernel(launch_bounds=...) declaration (preferred),
            # or the legacy ctx global __LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__
            # (same string as integration-facing CUDA_LAUNCH_BOUNDS_MIN_BLOCKS_CTX_KEY),
            # or an exact per-thread register budget via the ctx global
            # __LITTLE_KERNEL_CUDA_MAXNREG__ (incoming branch). CUDA forbids
            # combining __launch_bounds__ with __maxnreg__ on the same kernel,
            # so the maxnreg channel is mutually exclusive with BOTH
            # launch_bounds request channels. Other kernels should let ptxas
            # choose its own register budget.
            bounds = self.launch_bounds
            maxnreg_key = "__LITTLE_KERNEL_CUDA_MAXNREG__"
            if maxnreg_key in self.ctx:
                if MIN_BLOCKS_CTX_KEY in self.ctx or bounds is not None:
                    raise ValueError(f"{MIN_BLOCKS_CTX_KEY}/launch_bounds and {maxnreg_key} are mutually "
                                     "exclusive: CUDA forbids combining __launch_bounds__ "
                                     "with __maxnreg__ on the same kernel")
                maxnreg = self.ctx[maxnreg_key]
                if isinstance(maxnreg, bool) or not isinstance(maxnreg, int):
                    raise ValueError(f"{maxnreg_key} must be a positive int (registers "
                                     f"per thread), got {maxnreg!r}")
                if maxnreg <= 0:
                    raise ValueError(f"{maxnreg_key} must be a positive int (registers "
                                     f"per thread), got {maxnreg!r}")
                sig = f"extern \"C\" __global__ __maxnreg__({maxnreg}) {sig}"
            else:
                if bounds is None:
                    if self.num_threads is not None and MIN_BLOCKS_CTX_KEY in self.ctx:
                        bounds = (self.num_threads, int(self.ctx[MIN_BLOCKS_CTX_KEY]))
                if bounds is not None:
                    max_threads, min_blocks = bounds
                    if min_blocks is not None:
                        sig = f"extern \"C\" __global__ __launch_bounds__({max_threads}, {min_blocks}) {sig}"
                    else:
                        sig = f"extern \"C\" __global__ __launch_bounds__({max_threads}) {sig}"
                else:
                    sig = f"extern \"C\" __global__ {sig}"
        else:
            sig = f"static __device__ __forceinline__ {sig}"
        if len(sig) <= 100:
            self.writeln_main(sig)
        else:
            self.writeln_main(sig.rsplit("(", 1)[0] + "(")
            for index, parameter in enumerate(cpp_params):
                comma = "," if index + 1 < len(cpp_params) else ""
                self.writeln_main(f"    {parameter}{comma}")
            self.writeln_main(") {")
        self.indent_level += 1

        # Provably-static integer locals (constant-expression extents).
        self._begin_function_static_consts(node)

        # Process function body (statements like Assign, Return)
        for stmt in node.body:
            if (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)):
                continue
            self.visit(stmt)

        # Cleanup: close function and reset scope
        self.indent_level -= 1
        self.writeln_main("}")
        self.writeln_main("")
        self.scope_vars = old_scope_vars  # Restore scope after function
        self._end_function_static_consts()

    def get_code(self, need_header=True) -> str:
        """Combine only the headers and declarations required by this CUDA program."""
        unmatched = getattr(self, "_restrict_unmatched", None)
        if unmatched:
            raise ValueError(f"{_RESTRICT_PARAMS_KEY}: no parameter was matched for "
                             f"these names: {sorted(unmatched)}. Check for typos; names "
                             "must exactly match a function parameter name.")
        self._emit_struct_definitions()  # Emit structs before main code
        main_code = self.main_buffer.getvalue()
        struct_code = self.struct_buffer.getvalue()
        builtin_code = self.builtin_buffer.getvalue()
        if not self._conditional_headers_emitted:
            body = struct_code + builtin_code + main_code
            optional = []
            if "bfloat" in body or "__nv_bfloat16" in body:
                optional.append("<cuda_bf16.h>")
            if "__half" in body or "half2" in body:
                optional.append("<cuda_fp16.h>")
            if "TmaDescriptor" in body or "CUtensorMap" in body or "CUresult" in body:
                optional.append("<cuda.h>")
            if "std::tuple" in body or "std::make_tuple" in body:
                optional.append("<tuple>")
            if ("cuda::std::tuple" in body or "cuda::std::make_tuple" in body or "cuda::std::get" in body):
                optional.append("<cuda/std/tuple>")
            if "std::string" in body or "std::to_string" in body:
                optional.append("<string>")
            if "ptrdiff_t" in body:
                optional.append("<cstddef>")
            for header in optional:
                if header not in self.header_cache:
                    self.header_cache.add(header)
                    self.header_buffer.write(f"#include {header}\n")
            if "TmaDescriptor" in body:
                self.header_buffer.write("\nusing TmaDescriptor = CUtensorMap;\n")
            self._conditional_headers_emitted = True

        sections = ([self.header_buffer.getvalue()] if need_header else [])
        sections += [struct_code, builtin_code, main_code]
        result = "\n\n".join(part.strip("\n") for part in sections if part.strip())
        return ("// Generated from LittleKernel IR.\n" if need_header else "") + result + "\n"


def codegen_cuda(tree: ast.AST, ctx=None, emit_header=True, num_threads=None, disable_var_hoisting=False,
                 launch_bounds=None, no_alias=False) -> str:
    """Generate CUDA code from AST.

    Parameters
    ----------
    tree : ast.AST
        Python AST to generate code from
    ctx : dict, optional
        Context dictionary (e.g., function.__globals__)
    emit_header : bool, optional
        Whether to emit header includes (default: True)
    num_threads : int, optional
        Actual threads per block. Only used to (a) pair with the legacy ctx global
        __LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ and (b) sanity-check an
        explicit launch_bounds. By itself it does NOT add __launch_bounds__.
    disable_var_hoisting : bool, optional
        If True, disable variable hoisting for if/for/while blocks.
        This can improve register allocation for performance-critical kernels,
        but may cause issues if variables are used outside their defining block.
        Default: False.
    launch_bounds : int or (int, int), optional
        Explicit __launch_bounds__(max_threads[, min_blocks]) on the entry
        __global__ function. None (default) preserves legacy behavior: no
        __launch_bounds__ unless the legacy ctx global is set.
    no_alias : bool, optional
        If True, emit __restrict__ on every entry-kernel pointer/tensor
        parameter (a blanket assertion that bypasses the per-parameter
        may_alias contract; with the default False the contract applies).
        Default: False (generated source is unchanged).

    Returns
    -------
    str
        Generated CUDA code
    """
    if ctx is None:
        ctx = {}
    # Reject malformed launch_bounds before type inference runs.
    launch_bounds = _normalize_launch_bounds_signature(launch_bounds)
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    emitter = CudaEmitter(ctx=ctx, all_types=all_types, num_threads=num_threads,
                          disable_var_hoisting=disable_var_hoisting, launch_bounds=launch_bounds, no_alias=no_alias)
    from little_kernel.codegen.policy import emission_scope
    with emission_scope(emitter):
        emitter.visit(tree)
        return emitter.get_code(need_header=emit_header)
