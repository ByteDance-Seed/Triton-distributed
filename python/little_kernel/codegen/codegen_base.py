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
import builtins as _builtins
import re
from io import StringIO
from typing import Dict, List, Union
import little_kernel.language as ll
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.codegen.registries.type_converter import TypeConverter
from little_kernel.codegen.visitors.expression_codegen import ExpressionCodegenMixin
from little_kernel.codegen.visitors.statement_codegen import StatementCodegenMixin
from little_kernel.codegen.visitors.control_flow_codegen import ControlFlowCodegenMixin
from little_kernel.codegen.visitors.call_codegen import CallCodegenMixin
from little_kernel.codegen.visitors.static_int_analysis import compute_static_int_consts, find_escaped_names
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.language.builtin_base import MEMORY_EFFECTS_ATTR, ORIGINAL_PY_FUNC_ATTR
import inspect as _inspect


class CppEmitter(ExpressionCodegenMixin, StatementCodegenMixin, ControlFlowCodegenMixin, CallCodegenMixin,
                 ast.NodeVisitor):
    """
    C++ code emitter with context-aware LLType resolution and scope tracking.
    - Avoids redundant type declarations for in-scope variables (parameters/previous assignments)
    - Enforces basic type safety for reassignments
    - Resolves namespace aliases via user-provided context (e.g., 'lk.uint32')
    """

    def __init__(self, ctx: Dict[str, object], all_types=None):
        """
        Initialize the C++ emitter with a user-provided context.
        
        Args:
            ctx: A dictionary representing the namespace context (e.g., function.__globals__).
                 Must contain the alias used to import `little_kernel.language` (e.g., 'll' or 'lk').
        """
        self.tuple_namespace = "std"
        self.half_arithmetic_header = None
        # Code buffers (headers -> structs -> main code)
        self.header_buffer = StringIO()
        self.header_cache = set()
        self.struct_buffer = StringIO()
        self.main_buffer = StringIO()
        self.builtin_buffer = StringIO()
        self.builtin_cache = set()
        self.builtin_body_cache = set()  # Track emitted function names to prevent duplicates

        # State management
        self.indent_level = 0
        self.indent_unit = "    "  # 4-space indentation

        # LLType tracking: StructType -> C++ struct name
        self.struct_types: Dict["ll.StructType", str] = {}

        # Special struct tracking: Track which special structs are used
        self.used_special_structs: set = set()

        # Enum tracking: Track which Enum types are used
        self.used_enums: set = set()

        # Context & annotation cache
        self.ctx = ctx  # User's namespace (for alias resolution)
        self.annotation_cache: Dict[ast.AST, "ll.LLType"] = {}

        # Scope tracking: Maps variable names to their LLType (per function scope)
        # Reset when entering/exiting a FunctionDef
        self.scope_vars: Dict[str, "ll.LLType"] = {}
        self.all_types = all_types if all_types is not None else {}

        # Provably-static integer locals of the function currently being
        # emitted (name -> StaticIntConst); see
        # codegen/visitors/static_int_analysis.py.  Backends that require
        # constant-expression array extents (CUDA __shared__/local arrays)
        # set the two flags below to emit such locals as compile-time
        # constants and to diagnose non-constant array extents early.
        self._static_int_consts = {}
        self._emit_static_constexpr = False
        self._require_constant_array_extents = False

        # Constant bindings visible to expression codegen for the currently
        # emitting statements, innermost-first: a source-level fully unrolled
        # ``ll.unroll(range(N))`` loop emits each iteration as a scoped copy
        # whose loop counter is re-declared with the literal trip value, so
        # control-flow codegen pushes ``{counter: trip_value}`` around the
        # copy. ``visit_Subscript`` folds fragment indices against these
        # bindings (plus the provably-static integer locals above) so a
        # compile-time-unrolled fragment element access lowers to the
        # per-element scalar ``<frag>_<i>`` rather than an address-taken
        # array slot. Plain (non-fragment) subscripts are unaffected.
        self._unroll_const_envs: List[Dict[str, int]] = []

        # Add required C++ headers
        self._add_basic_headers()

        # Validate context (ensure LLType-related objects are present)
        self._validate_context()

    def _add_basic_headers(self) -> None:
        """Add C++ headers for std types."""
        headers = [
            "<cstdint>",  # For stdint types (uint32_t, int64_t)
            "<string>",  # For std::string (maps to StringType)
        ]
        for hdr in headers:
            self.header_cache.add(hdr)
            self.header_buffer.write(f"#include {hdr}\n")

    def _validate_context(self) -> None:
        """Validate context contains `little_kernel.language` or LLType objects."""
        has_lltype = any(
            # Check if object is the LLType module
            (hasattr(obj, "__name__") and obj.__name__.startswith("little_kernel.language"))
            # Check if object is an LLType instance
            or (isinstance(obj, (type, object)) and "LLType" in str(type(obj))) for obj in self.ctx.values())

        if not has_lltype:
            raise ValueError("Context (ctx) missing `little_kernel.language` or LLType objects. "
                             "Ensure ctx is the __globals__ of a function that imports `little_kernel.language`.")

    def indent_str(self) -> str:
        """Return current indentation string."""
        return self.indent_unit * self.indent_level

    # ------------------------------ Code Writing Helpers ------------------------------
    def write_main(self, s: str) -> None:
        """Write to main code buffer (functions, classes, statements)."""
        self.main_buffer.write(s)

    def writeln_main(self, s: str = "") -> None:
        """Write line to main code buffer with newline."""
        self.write_main(f"{self.indent_str()}{s}\n")

    def writeln_header(self, s: str = "") -> None:
        """Write line to header buffer with newline."""
        self.header_buffer.write(f"{s}\n")

    def write_struct(self, s: str) -> None:
        """Write to struct definition buffer."""
        self.struct_buffer.write(s)

    def writeln_struct(self, s: str = "") -> None:
        """Write line to struct buffer with newline."""
        self.write_struct(f"{s}\n")

    # C/C++ keywords that must not be treated as function names when matching
    _FUNC_NAME_EXCLUDE = frozenset({"if", "for", "while", "switch", "catch", "return", "else"})

    def writeln_builtin(self, s: str = "") -> None:
        """Write lines to builtin buffer, deduplicating function definitions.
        
        When code from an inner LLKernel emitter is added, it may contain
        function definitions already emitted by the outer emitter. We extract
        function names and skip blocks where all functions are already emitted.
        
        Handles __device__, __forceinline__, pointer types (void*), custom structs,
        and other return types by matching the identifier before the parameter list.
        """
        stripped = s.strip()
        if not stripped:
            return
        # Match identifier before ( and ) { - works for void, void*, uint64_t, Struct_xxx, etc.
        all_matches = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', stripped)
        func_names = [m for m in all_matches if m not in self._FUNC_NAME_EXCLUDE]
        if func_names:
            new_funcs = [f for f in func_names if f not in self.builtin_body_cache]
            if not new_funcs:
                return  # All functions already emitted
            for f in func_names:
                self.builtin_body_cache.add(f)
        else:
            body_key = hash(stripped)
            if body_key in self.builtin_body_cache:
                return
            self.builtin_body_cache.add(body_key)
        self.builtin_buffer.write(f"{s}\n")

    # ------------------------------ LLType to C++ Mapping ------------------------------
    def _generate_struct_name(self, struct_type: "ll.StructType") -> str:
        """Generate a unique, valid C++ struct name from StructType fields."""
        field_suffix = "_".join([f"{name}_{self._lltype_to_cpp(typ)}" for name, typ in struct_type.type_tuple])
        # Sanitize for C++ identifier rules (replace invalid chars)
        valid_suffix = (field_suffix.replace("<", "_").replace(">", "_").replace("*", "_ptr_").replace(" ", ""))
        return f"Struct_{valid_suffix}"

    def _lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a valid C++ type string."""
        if not hasattr(self, '_type_converter'):
            self._type_converter = TypeConverter(self.struct_types)
        return self._type_converter.lltype_to_cpp(ll_type)

    def _native_scalar_conversion(self, value_cpp, source_type, destination_type):
        """Optional target conversion policy; the common path uses C++ casts."""
        return None

    # ------------------------------ Store Legalization ------------------------------
    def _convert_value_for_store(self, value_cpp: str, src_lltype, dst_lltype) -> str:
        """Hook: legalize a subscript-store value whose type differs from the
        target element type.

        ``src_lltype``/``dst_lltype`` are the best-effort LLTypes of the value
        expression and of the tensor/pointer element (either may be None when
        type information is unavailable).  The default keeps the historical
        emission unchanged; backends with ill-formed implicit conversions
        (e.g. CUDA 16-bit floats under the torch no-conversion macros)
        override this to lower through explicit conversion intrinsics.
        """
        return value_cpp

    # ------------------------------ Struct Generation ------------------------------
    def _emit_struct_definitions(self) -> None:
        """
        Auto-generate C++ structs for all tracked StructType instances.
        
        Note: Enum and special struct definitions are now handled via IR nodes
        in the AST (EnumDef and SpecialStructDef), which are processed in visit_Module.
        This method only handles regular LLType struct definitions.
        """

        # Then, emit regular LLType struct definitions
        if not self.struct_types:
            return

        self.writeln_struct("// Auto-generated structs from LLType StructType")
        for ll_struct, cpp_name in self.struct_types.items():
            self.writeln_struct(f"struct {cpp_name} {{")
            # Emit struct fields (name + C++ type)
            for field_name, field_type in ll_struct.type_tuple:
                field_cpp = self._lltype_to_cpp(field_type)
                self.writeln_struct(f"    {field_cpp} {field_name};")
            self.writeln_struct("};")
            self.writeln_struct()  # Blank line between structs

    # ------------------------------ AST Node Visitors ------------------------------
    def get_source_location(self, node: ast.AST) -> str:
        """
        Get source code location information from an AST node.
        
        Returns a string like "line 10, column 5" or "unknown location" if not available.
        """
        if hasattr(node, 'lineno') and node.lineno is not None:
            col_info = f", column {node.col_offset + 1}" if hasattr(
                node, 'col_offset') and node.col_offset is not None else ""
            return f"line {node.lineno}{col_info}"
        return "unknown location"

    def get_type(self, value):
        """Get the LLType of a value."""
        if isinstance(value, ll.LLType):
            return value
        elif isinstance(value, ast.Constant) and isinstance(value.value, ll.LLType):
            return value.value
        assert value in self.all_types, f"Value {ast.dump(value) if isinstance(value, ast.AST) else value} is not in all_types"
        return self.all_types[value]

    # -------------------- Provably-static integer locals --------------------
    def _begin_function_static_consts(self, node: ast.FunctionDef) -> None:
        """Compute provably-static integer locals for the function body about
        to be emitted, and remember the caller's map for restore on exit.

        The analysis is always run when a backend opted into constexpr
        emission or constant-extent enforcement (the CUDA emitter); generic
        emitters keep the flags off and never pay for it.
        """
        self._static_int_consts_outer = getattr(self, "_static_int_consts", {})
        self._current_function_top_stmts_outer = getattr(self, "_current_function_top_stmts", None)
        self._current_function_top_stmts = list(node.body)
        if not (self._emit_static_constexpr or self._require_constant_array_extents):
            self._static_int_consts = {}
            return
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        def resolve_annotation(ann_node):
            return inferencer._resolve_annotation(ann_node)

        def infer_expr_type(expr_node, scope):
            sub = TypeInferencer(ctx=self.ctx, scope_vars=dict(scope))
            try:
                t = sub.visit(expr_node)
            except Exception:
                return None
            return t if isinstance(t, ll.LLType) else None

        escape_predicate = self._make_call_arg_escape_predicate()
        escape_predicate.set_scope(node)
        escaped = find_escaped_names(node, escape_predicate)
        self._static_int_consts = compute_static_int_consts(node, resolve_annotation, infer_expr_type,
                                                            extra_disqualified=escaped)

    def _end_function_static_consts(self) -> None:
        """Restore the enclosing function's static-const map (device-function
        emission is nested within the caller's module walk)."""
        self._static_int_consts = getattr(self, "_static_int_consts_outer", {})
        self._current_function_top_stmts = getattr(self, "_current_function_top_stmts_outer", None)

    # Cache: id(resolved builtin) -> {param_name: True if emitted by address}
    # so repeated calls to the same intrinsic resolve templates once.
    def _make_call_arg_escape_predicate(self):
        """Build the (call_node, arg_expr, func_node) -> bool predicate used
        by the escape analysis to decide whether passing a whole local by
        name to a call may mutate it / take its address.

        Resolution mirrors call emission: the call target is resolved through
        the emitter context to a callable.  ``None`` resolution is NOT one
        case: a bare ``Name`` callee that is *absent* from the emitter
        context refers, by ordinary source lexical semantics, to the real
        Python builtin of that spelling (this source language binds no
        ``builtins`` scope), so the read-only builtin table below applies to
        the identical object from that scope; a context entry that is
        explicitly ``None``, an unresolved AST fragment, or any object that
        is not the identical builtin is an UNKNOWN callee and escapes
        conservatively.  A callee name bound anywhere in the enclosing
        function's lexical scope (a parameter, or any assignment /
        annotated / augmented / loop / with target) is shadowed or
        rebindable and likewise stays conservative even when context-free.

        For a resolved ``@builtin``-decorated intrinsic, per-parameter write
        facts come ONLY from declared intrinsic/IR effects
        (``__memory_effects__``): a W/RW effect whose operand names the
        parameter marks it written, and an explicit R effect marks it
        read-only.  Every parameter with NO declared effect is WRITTEN:
        successfully rendered C++ text, absence of ``asm``/``__asm__``,
        occurrence of the parameter's spelling in the output, and failure to
        match any write regex prove NOTHING — a helper such as ``void
        bump(int& renamed){++renamed;}`` invoked as ``bump(out)`` mutates
        the caller through a mutable reference while ``out`` shows none of
        those spellings.  Anything uninspectable (no signature) escapes ALL
        parameters conservatively.

        Two exceptions stay read-only by construction, bound to the actual
        primitive OBJECTS (never their names or rendered text): the
        allocation-extent callee (``alloc_shared_memory``) reads its bound
        by value, and the genuine Python builtin ``range`` reads a loop
        bound by value — identified by object identity against
        ``builtins.range`` so an equality-spoofing custom object can never
        impersonate it.  This never mis-annotates ``constexpr``; at worst it
        keeps the historical plain-local emission.
        """
        template_cache: Dict[int, Dict[str, bool]] = {}
        ctx = self.ctx

        def _param_addr_map(func_resolved) -> Dict[str, bool]:
            """param_name -> True if that param is *written* by the callee.

            ONLY explicit declared intrinsic/IR effects classify a parameter:
            a W/RW effect whose operand names the parameter marks it written;
            an explicit R effect marks it read-only.  Every parameter with NO
            declared effect — including parameters of a callable whose C++
            renders successfully with no asm and its name spelled in the
            output — defaults to WRITTEN.  Rendered text carries no sound
            purity evidence: a helper `void bump(int& renamed) { ++renamed; }`
            invoked as `bump(out)` mutates the caller through a mutable
            reference while `out` matches no write spelling, so no textual
            pattern is consulted here.  Uninspectable callables (no signature)
            fall back to ``{"*": True}``: every argument escapes.
            """
            key = id(func_resolved)
            if key in template_cache:
                return template_cache[key]
            try:
                py_func = getattr(func_resolved, ORIGINAL_PY_FUNC_ATTR, None)
                sig = _inspect.signature(py_func if py_func is not None else func_resolved)
                param_names = [p.name for p in sig.parameters.values()]
                # Explicit intrinsic/IR effects: W/RW on an operand names the
                # written parameter; explicit R effects name honoured
                # read-only parameters.
                effects = getattr(func_resolved, MEMORY_EFFECTS_ATTR, None)
                if effects is None:
                    effects = getattr(py_func, MEMORY_EFFECTS_ATTR, None) \
                        if py_func is not None else None
                write_operands: set = set()
                read_operands: set = set()
                if effects:
                    from little_kernel.core.memory_effects import Access
                    for eff in effects:
                        operand = getattr(eff, "operand", None)
                        access = getattr(eff, "access", None)
                        if not isinstance(operand, str):
                            continue
                        if access in (Access.W, Access.RW):
                            write_operands.add(operand)
                        elif access is Access.R:
                            read_operands.add(operand)
                addr: Dict[str, bool] = {}
                for name in param_names:
                    if name in write_operands:
                        addr[name] = True
                    elif name in read_operands:
                        addr[name] = False
                    else:
                        # No declared effect: unknown effects are WRITTEN.
                        # A successful C++ render, absence of asm, occurrence
                        # of the parameter name, and failure to match any
                        # write spelling prove nothing (e.g. `void bump(int&
                        # renamed){++renamed;}` called as `bump(out)` writes
                        # the caller through a reference); only an explicit
                        # IR/intrinsic effect or genuine primitive semantics
                        # (the _READ_ONLY_BUILTINS identity check in _escapes,
                        # bound to the real builtin object) can prove
                        # read-only.
                        addr[name] = True
                template_cache[key] = addr
            except Exception:
                # Uninspectable: treat every parameter as address-taken so a
                # whole-Name arg conservatively escapes (never mis-annotates).
                template_cache[key] = {"*": True}
            return template_cache[key]

        def _resolve_call_target(call_node):
            func_expr = call_node.func
            if not isinstance(func_expr, (ast.Attribute, ast.Name)):
                return None
            try:
                resolved = recursive_resolve_attribute(func_expr, ctx)
            except Exception:
                return None
            # Unresolved leaves an AST node behind; treat as unresolvable.
            return None if isinstance(resolved, ast.AST) else resolved

        # Python builtins whose integer arguments are read-only: the builtin
        # neither mutates its arguments nor leaks their addresses.  ``range``
        # is the one the source language uses to hold a compile-time loop
        # bound; keep the set minimal and *object-identity* keyed so name
        # shadowing or inspect.signature quirks never grant purity by
        # spelling alone.
        _READ_ONLY_BUILTINS = (_builtins.range, )

        def _read_only_builtin_for_call(call_node, resolved):
            """The identical read-only builtin object behind this call, or
            None if the call target is not PROVEN to be that builtin.

            * ``resolved`` came from context resolution: it qualifies only
              by object identity (an explicit ``ctx['range'] =
              builtins.range`` binding IS the real builtin).
            * a bare ``Name`` callee ABSENT from the context refers to the
              source language's builtin scope by lexical semantics; qualify
              the identical object from that scope.  A present context
              entry — even ``None`` or an unresolvable AST fragment — is a
              bound (possibly unknown) object, NOT an absent name, and
              never qualifies.
            * a callee name bound anywhere in the enclosing function's
              lexical scope (parameter, plain/annotated/augmented
              assignment, tuple target, loop or with target) is shadowed
              or rebindable: no read-only treatment regardless of context.
            """
            func_expr = call_node.func
            if not isinstance(func_expr, ast.Name):
                return None
            if func_expr.id in _locally_bound:
                # Shadowed in lexical scope: the real builtin is not the
                # guaranteed callee.
                return None
            if resolved is not None:
                # Membership by OBJECT IDENTITY: ``in``/``==`` would execute
                # a custom context object's __eq__ and let an arbitrary
                # binding impersonate the builtin.  Only the identical
                # read-only builtin object qualifies.
                if any(resolved is known for known in _READ_ONLY_BUILTINS):
                    return resolved
                return None
            if isinstance(ctx, dict) and func_expr.id in ctx:
                # A present binding that did not resolve to the builtin
                # (None/AST/other object) is not the absent-name fallback.
                return None
            try:
                candidate = getattr(_builtins, func_expr.id, None)
            except Exception:
                return None
            # Identity, not equality: a custom object equal-by-__eq__ to the
            # builtin is still an unknown callee.
            if any(candidate is known for known in _READ_ONLY_BUILTINS):
                return candidate
            return None

        def _collect_locally_bound(func_node) -> set:
            """Every name bound inside the function's own statements or
            parameter list (not descending into nested def/class scopes)."""
            bound = {a.arg for a in func_node.args.args}
            bound.update(a.arg for a in func_node.args.posonlyargs)
            bound.update(a.arg for a in func_node.args.kwonlyargs)
            for a in (func_node.args.vararg, func_node.args.kwarg):
                if a is not None:
                    bound.add(a.arg)

            def add_target(t):
                for n in _assigned_target_names(t):
                    bound.add(n)

            def walk(stmts):
                for stmt in stmts:
                    if isinstance(stmt, ast.Assign):
                        for tgt in stmt.targets:
                            add_target(tgt)
                    elif isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
                        add_target(stmt.target)
                    elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                        add_target(stmt.target)
                        walk(stmt.body)
                        walk(stmt.orelse)
                    elif isinstance(stmt, (ast.If, ast.While)):
                        walk(stmt.body)
                        walk(stmt.orelse)
                    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                        for item in stmt.items:
                            if item.optional_vars is not None:
                                add_target(item.optional_vars)
                        walk(stmt.body)
                    elif isinstance(stmt, ast.Try):
                        walk(stmt.body)
                        for h in stmt.handlers:
                            if h.name:
                                bound.add(h.name)
                            walk(h.body)
                        walk(stmt.orelse)
                        walk(stmt.finalbody)
                    elif isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        # Nested scopes do not rebind this function's names.
                        continue

            walk(func_node.body)
            return bound

        def _assigned_target_names(t):
            if isinstance(t, ast.Name):
                yield t.id
            elif isinstance(t, (ast.Tuple, ast.List)):
                for elt in t.elts:
                    yield from _assigned_target_names(elt)
            elif isinstance(t, ast.Starred):
                yield from _assigned_target_names(t.value)

        # Lexical bindings of the CURRENT function are filled in when the
        # predicate is attached (see _set_escape_scope below); an empty set
        # keeps the pre-existing behaviour for direct predicate users.
        _locally_bound: set = set()

        # Established primitive value semantics, bound to the actual
        # primitive OBJECTS by identity (never to a name, spelling, rendered
        # text, or task): the allocation-extent primitives consume their
        # extent/size operands as plain Python ints while evaluating the
        # allocation — the extent never reaches emitted C++ as an lvalue, so
        # it cannot be written or address-taken.  A custom object merely
        # equal-by-__eq__ or identically named never qualifies.  Everything
        # NOT listed here keeps the declared-effect-or-conservative default.
        from little_kernel.language.builtin_base import by_value_argument_primitives

        def _primitive_by_value_params(func_resolved):
            """The by-value parameter names of ``func_resolved`` when it IS
            (by object identity) one of the established allocation-extent
            primitives above; None for every other object."""
            for prim, names in by_value_argument_primitives():
                if func_resolved is prim:
                    return names
            return None

        def _escapes(call_node, arg_expr) -> bool:
            resolved = _resolve_call_target(call_node)
            if _read_only_builtin_for_call(call_node, resolved) is not None:
                # The callee is PROVEN (by object identity, with absent-name
                # resolution checked against the context and the function's
                # lexical bindings) to be a genuine Python read-only builtin
                # such as range: it reads its integer arguments by value and
                # never mutates or address-takes them, so a whole-Name bound
                # passed there must not cancel the constant proof.
                return False
            if resolved is None or isinstance(resolved, ll.LLType):
                # Unknown callee (absent/None/AST/custom binding, shadowed
                # name) or a type constructor: we cannot prove the argument
                # is read purely by value.
                return True
            prim_params = _primitive_by_value_params(resolved)
            addr = _param_addr_map(resolved)
            if addr.get("*"):
                return True
            py_func = getattr(resolved, ORIGINAL_PY_FUNC_ATTR, None)
            try:
                sig = _inspect.signature(py_func if py_func is not None else resolved)
                params = list(sig.parameters.values())
            except Exception:
                return True
            # Positional index of this arg among call_node.args; keyword
            # matched by name.  (arg_expr IS the Name node at that slot.)
            # A parameter without positive write evidence stays read-only
            # ONLY when genuinely established: an explicit declared R effect
            # in ``addr``, or the parameter of the identical real
            # allocation-extent primitive in ``prim_params``.  Every
            # missing-position/undeclared entry stays conservative.
            def _param_read_only(pname):
                if not addr.get(pname, True):
                    return True
                return prim_params is not None and pname in prim_params

            for idx, arg in enumerate(call_node.args):
                if arg is arg_expr:
                    if idx < len(params):
                        return _param_read_only(params[idx].name) is False
                    return True
            for kw in call_node.keywords:
                if kw.value is arg_expr:
                    for p in params:
                        if p.name == kw.arg:
                            return _param_read_only(p.name) is False
                    return True
            return True

        def _set_escape_scope(func_node) -> None:
            """Record the enclosing function's local bindings so shadowed
            callee names stay conservative (see _read_only_builtin_for_call).
            """
            _locally_bound.clear()
            _locally_bound.update(_collect_locally_bound(func_node))

        _escapes.set_scope = _set_escape_scope
        return _escapes

    # Note: visit_Assign, visit_Module, visit_FunctionDef, visit_AnnAssign, visit_Expr,
    # visit_Return, visit_Assert, visit_Pass are now in StatementCodegenMixin

    # Note: visit_Name, visit_Constant, visit_Attribute, visit_BinOp, visit_UnaryOp,
    # visit_BoolOp, visit_Compare, visit_IfExp, visit_Subscript, visit_JoinedStr,
    # visit_FormattedValue are now in ExpressionCodegenMixin

    # Note: visit_Call is now in CallCodegenMixin
    # Note: visit_Return, visit_BinOp, visit_UnaryOp, visit_BoolOp, visit_Compare,
    # visit_IfExp, visit_JoinedStr, visit_FormattedValue, visit_Subscript are now in ExpressionCodegenMixin
    # Note: _get_operator_str is now in ExpressionCodegenMixin

    def generic_visit(self, node: ast.AST) -> Union[None, str]:
        """Default handler for unimplemented AST nodes."""
        # inline may result lists of stmts
        if isinstance(node, list):
            for stmt in node:
                self.visit(stmt)
        else:
            try:
                node_dump = ast.dump(node, indent=2)
            except TypeError:
                # Python 3.7 doesn't support indent parameter
                node_dump = ast.dump(node)
            raise NotImplementedError(f"Unsupported AST node type: {type(node).__name__}\n"
                                      f"Node details: {node_dump}")

    def visit_Expression(self, node: ast.Expression) -> str:
        """Visit Expression node (from ast.parse with mode='eval')."""
        return self.visit(node.body)

    # Note: visit_If, visit_While, visit_For, visit_Continue, visit_Break are now in ControlFlowCodegenMixin
    # Note: visit_Subscript, _process_single_index are now in ExpressionCodegenMixin

    # ------------------------------ Final Code Generation ------------------------------
    def get_code(self, need_header=True) -> str:
        """Combine headers, structs, and main code into final C++ output."""
        self._emit_struct_definitions()  # Emit structs before main code
        return (("/*start of generated code*/\n" + self.header_buffer.getvalue() if need_header else "") + "\n" +
                self.struct_buffer.getvalue() + "\n" + self.builtin_buffer.getvalue() + "\n" +
                self.main_buffer.getvalue())


def codegen_cpp(tree: ast.AST, ctx=None, emit_header=True) -> str:
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    emitter = CppEmitter(ctx=ctx, all_types=all_types)
    emitter.visit(tree)
    return emitter.get_code(need_header=emit_header)
