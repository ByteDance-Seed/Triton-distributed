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
"""
Expression codegen visitors for CppEmitter.

This module contains visitor methods for AST expression nodes that return C++ code strings.
"""

import ast
import os
import warnings
from enum import Enum
from typing import TYPE_CHECKING
import little_kernel.language as ll
from little_kernel.core.internal import __LOCAL_ARRAY_SHAPES_CTX_KEY__, __WGMMA_ACC_FRAGMENTS_CTX_KEY__, __MMA_SYNC_FRAGMENTS_CTX_KEY__
from little_kernel.codegen.visitors.static_int_analysis import try_eval_const_int
from little_kernel.core import ir as lk_ir
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.core.passes.utils.error_report import format_error_message
from little_kernel.language.intrin.scalar_literal import _F32Literal, _F64Literal, format_f32_constant, format_f64_constant
from little_kernel.codegen.registries.operator_codegen import get_operator_codegen_registry


def _warn_bare_float_literal(value, lineno):
    """Opt-in probe (LK_WARN_FLOAT_LITERAL=1): a float constant goes through
    the legacy bare-C++-double emission.  Off by default; advisory only."""
    if os.environ.get("LK_WARN_FLOAT_LITERAL"):
        warnings.warn(
            f"float literal {value!r} (line {lineno}) is emitted as a C++ "
            f"double; in fp32 expressions C++ usual arithmetic conversions "
            f"promote the expression to double. Pin the width explicitly with "
            f"ll.f32()/ll.f64(), or set LK_WARN_FLOAT_LITERAL=0 to silence.", stacklevel=2)


# Import PASSES lazily to avoid circular import
def _get_passes():
    from little_kernel.core.passes import PASSES
    return PASSES


if TYPE_CHECKING:
    pass


class ExpressionCodegenMixin:
    """Mixin class for expression codegen methods."""

    # Scalar fp16 +,-,* map to the single-round native __half intrinsics
    # when the target opts in (CUDA): the C++ __half operator overloads
    # are compiled out by -D__CUDA_NO_HALF_OPERATORS__, while these
    # explicit device functions survive it.
    _FP16_SCALAR_ARITH_INTRINSICS = {ast.Add: "__hadd", ast.Sub: "__hsub", ast.Mult: "__hmul"}

    def _fp16_scalar_arith_intrinsic(self, node: ast.BinOp):
        """Return the native __half intrinsic name for a scalar fp16
        BinOp (adding its header on first use), or None when this BinOp
        must keep its ordinary C++ operator spelling."""
        if not getattr(self, "_legalize_fp16_scalar_arith", False):
            return None
        intrinsic = self._FP16_SCALAR_ARITH_INTRINSICS.get(type(node.op))
        if intrinsic is None:
            return None
        try:
            lhs_type = self.get_type(node.left)
            rhs_type = self.get_type(node.right)
            res_type = self.get_type(node)
        except Exception:
            return None
        # Scalar float16 only: register arrays (TensorType) and mixed
        # dtype expressions keep the ambient operator path.
        if not (lhs_type == ll.float16 and rhs_type == ll.float16 and res_type == ll.float16):
            return None
        header = self.half_arithmetic_header
        if header not in self.header_cache:
            self.writeln_header(f"#include {header}")
            self.header_cache.add(header)
        return intrinsic

    @staticmethod
    def _format_float32_literal(value: float) -> str:
        """Render a Python float constant as a C++ single-precision literal.

        LittleKernel's type inference types every Python float constant as
        float32 (ConstantTypeInferencer.infer_constant_type). Emitting a bare
        C++ ``double`` literal (``str(value)``) silently breaks that contract:
        under the usual arithmetic conversions the whole typed-f32 expression
        is promoted to double (including /, which becomes software div.rn.f64
        on device). Emit the f32-rounded value with an 'f' suffix instead, the
        same convention as MLIR emitc's printFloat (TranslateToCpp.cpp), which
        appends 'f' for IEEEsingle semantics.
        """
        import math
        import struct
        try:
            f32 = struct.unpack('<f', struct.pack('<f', value))[0]
        except OverflowError:  # constant overflowed f32 range
            f32 = math.copysign(math.inf, value)
        if math.isinf(f32):
            return '(float)INFINITY' if f32 > 0 else '(-(float)INFINITY)'
        if math.isnan(f32):
            return '(float)NAN'
        bits = struct.pack('<f', f32)
        rendered = None
        for prec in range(1, 10):  # 9 significant digits always round-trip f32
            rendered = f'{f32:.{prec}g}'
            if struct.pack('<f', float(rendered)) == bits:
                break
        if '.' not in rendered and 'e' not in rendered:
            rendered += '.0'  # keep a float literal, not an int
        return rendered + 'f'

    _cpp_precedence = {
        ast.Mult: 13,
        ast.Div: 13,
        ast.FloorDiv: 13,
        ast.Mod: 13,
        ast.Add: 12,
        ast.Sub: 12,
        ast.LShift: 11,
        ast.RShift: 11,
        ast.BitAnd: 8,
        ast.BitXor: 7,
        ast.BitOr: 6,
    }

    def _operand_cpp(self, node: ast.AST, parent_precedence: int, *, right: bool = False) -> str:
        """Keep only parentheses needed to preserve the frontend AST."""
        rendered = self.visit(node)
        if isinstance(node, ast.BinOp):
            precedence = self._cpp_precedence.get(type(node.op), 0)
            if precedence < parent_precedence or (right and precedence == parent_precedence):
                return f"({rendered})"
        elif isinstance(node, ast.BoolOp):
            precedence = 5 if isinstance(node.op, ast.And) else 4
            if precedence < parent_precedence:
                return f"({rendered})"
        elif isinstance(node, (ast.Compare, ast.IfExp)):
            if parent_precedence > 5:
                return f"({rendered})"
        return rendered

    def _is_local_variable(self, name: str) -> bool:
        """
        Check if a variable name is defined locally (should not be resolved from ctx).
        
        Supports both:
        - scope_vars (Dict[str, LLType]) used by codegen
        - scope_manager (ScopeManager) used by passes
        
        This unified check ensures consistent symbol resolution across all components.
        """
        # Check scope_vars (codegen style)
        if hasattr(self, 'scope_vars') and name in self.scope_vars:
            return True
        # Check scope_manager (pass style)
        if hasattr(self, 'scope_manager') and self.scope_manager.is_local(name):
            return True
        return False

    def visit_Name(self, node: ast.Name) -> str:
        """Process variable/identifier reference (return as-is for C++)."""
        # Transitional compatibility: role/dataflow lowering used to hide raw
        # C++ text on ast.Name via ``_lk_cpp_expr``. Route it through the
        # explicit typed escape-hatch path. New lowering constructs
        # ``lk_ir.TargetExpression`` directly and never reaches this branch.
        if lk_ir.is_legacy_cpp_node(node):
            upgraded = lk_ir.upgrade_legacy_node(node, expr_type=lk_ir.legacy_cpp_expr_type(node))
            return self.visit_TargetExpression(upgraded)

        # First check if variable is defined locally
        # Local variables should NOT be resolved from ctx
        if self._is_local_variable(node.id):
            return node.id

        # Not a local variable, check ctx for constant folding
        if node.id in self.ctx:
            ctx_value = self.ctx[node.id]
            # Explicit-width literal markers (ll.f32/ll.f64) fold to
            # suffixed C++ literals; check before the generic branch.
            if isinstance(ctx_value, _F32Literal):
                return format_f32_constant(ctx_value)
            if isinstance(ctx_value, _F64Literal):
                return format_f64_constant(ctx_value)
            # If it's a constant value, convert it to C++ string directly
            if isinstance(ctx_value, float):
                _warn_bare_float_literal(ctx_value, getattr(node, "lineno", 0))
                # Same float32-typed-literal rule as visit_Constant: a bare
                # double literal would promote the expression to fp64. Keep
                # the inf/nan guard from the optional-input branch so folded
                # non-finite constants still emit valid C++.
                import math
                if math.isinf(ctx_value):
                    return "INFINITY" if ctx_value > 0 else "(-INFINITY)"
                if math.isnan(ctx_value):
                    return "NAN"
                node_type = self.all_types.get(node)
                if node_type is not None and getattr(node_type, "kind", None) == "float" \
                        and getattr(node_type, "bits", None) == 32:
                    return str(ctx_value) + "f"
                return str(ctx_value)
            if isinstance(ctx_value, (int, bool)):
                return str(ctx_value)
            elif isinstance(ctx_value, str):
                return f'"{ctx_value}"'
            # Otherwise, return the variable name
        return node.id

    def visit_TargetExpression(self, node) -> str:
        """Emit an explicitly typed backend expression escape hatch.

        The spelling is validated text emitted only at the backend boundary,
        never resolved as an ordinary identifier. Because its internal C++
        precedence is unknown, it is conservatively grouped (parenthesized)
        when embedded in a larger expression tree: a nested ``'1 + 2'`` under
        multiplication must render ``(1 + 2) * 3``, not ``1 + 2 * 3``.
        """
        lk_ir.verify_lowered_node(node)
        all_types = getattr(self, "all_types", None)
        if all_types is not None and node not in all_types:
            all_types[node] = node.result_type
        return lk_ir.grouped_target_spelling(node.spelling)

    def visit_ResourceReference(self, node) -> str:
        """Emit a typed lowered resource reference (pointer-valued).

        The path is a fully validated identifier/member/index expression whose
        precedence is known (member access and indexing bind tightest), so it
        is emitted verbatim and keeps its pointer/address-space semantics.
        """
        lk_ir.verify_lowered_node(node)
        all_types = getattr(self, "all_types", None)
        if all_types is not None and node not in all_types:
            all_types[node] = node.result_type
        return node.resource

    def visit_StageReference(self, node) -> str:
        """Emit a stage-indexed view over a pointer resource."""
        lk_ir.verify_lowered_node(node)
        base_cpp = self.visit(node.base)
        stage_cpp = self.visit(node.stage)
        all_types = getattr(self, "all_types", None)
        if all_types is not None and node not in all_types:
            all_types[node] = node.result_type
        return f"({base_cpp} + ({stage_cpp}) * {node.stride})"

    def visit_Constant(self, node: ast.Constant) -> str:
        """Process constants (map to C++ syntax)."""
        value = node.value

        # Handle None values
        if value is None:
            # Check if this None is used in a pointer context
            # Try to get the type from all_types
            node_type = None
            if node in self.all_types:
                node_type = self.all_types[node]

            # If we have type information, check if it's a pointer
            if node_type is not None:
                if node_type.is_pointer() or node_type.is_tensor():
                    # None for pointer/tensor types → nullptr
                    return "nullptr"
                else:
                    # None for non-pointer types is invalid
                    raise ValueError(f"None value is not allowed for non-pointer type {node_type}. "
                                     f"None can only be used for pointer types (will be converted to nullptr). "
                                     f"Node: {ast.dump(node, indent=2)}")
            else:
                # No type information available - this is ambiguous
                raise ValueError(f"None value encountered without type information. "
                                 f"None can only be used for pointer types (will be converted to nullptr). "
                                 f"Please ensure type information is available, or use explicit type annotations. "
                                 f"Node: {ast.dump(node, indent=2)}")

        if isinstance(value, str):
            if value.startswith("0x"):
                return value  # C++ hex literals (e.g., 0x1234)
            return f'"{value}"'  # C++ string literals (double quotes)
        elif isinstance(value, bool):
            return "true" if value else "false"  # C++ lowercase bools
        elif isinstance(value, int):
            # Try to preserve hex format if the original source was hex
            # Python 3.8+ AST nodes have a 'kind' attribute for hex literals
            if hasattr(node, 'kind') and node.kind == 'hex':
                return hex(value)
            # Note: AST doesn't preserve the original format (e.g., 0x1234 becomes 4660)
            # Without access to the original source code, we can't reliably determine
            # if a value was originally written in hex format.
            # The test expects 0x1234, but we only have the integer value 4660.
            # This is a limitation of using AST - we lose the original format information.
            # For now, we'll output the decimal representation.
            # To properly support this, we would need to:
            # 1. Store original source code alongside AST nodes, or
            # 2. Use a custom parser that preserves format information, or
            # 3. Use tokenize to extract format from source (requires source access)
            return str(value)
        elif isinstance(value, float):
            # Explicit-width literal markers can reach visit_Constant when a
            # pass substitutes a ctx value into the AST as an ast.Constant.
            if isinstance(value, _F32Literal):
                return format_f32_constant(value)
            if isinstance(value, _F64Literal):
                return format_f64_constant(value)
            _warn_bare_float_literal(value, getattr(node, "lineno", 0))
            # Handle special float values (inf, nan)
            import math
            if math.isinf(value):
                return "INFINITY" if value > 0 else "(-INFINITY)"
            elif math.isnan(value):
                return "NAN"
            text = str(value)
            # Typed float literal: Python float literals are inferred as
            # float32 (ConstantTypeInferencer). Emitting a bare str(value)
            # produces a C++ double literal, whose usual arithmetic
            # conversions silently promote the surrounding fp32 expression to
            # FP64 (SM90 fp64 runs at 1/64 throughput and the
            # double->float narrowing at assignment double-rounds). When the
            # inferred node type is a 32-bit float, emit a C++ float literal
            # (<value>f) so the expression remains fp32. 64-bit-float
            # contexts and missing type information keep the legacy double
            # literal. The kind/bits duck-check is used instead of
            # isinstance(ll.FloatType) so structurally-typed nodes also
            # qualify, and the lookup is getattr-defensive for emitters
            # constructed without all_types.  (Supersedes the gpu-07
            # `node_type == ll.float32` variant: ll.float32 itself has
            # kind="float"/bits=32, so both fire on the same nodes.)
            node_type = getattr(self, "all_types", {}).get(node)
            if node_type is not None and getattr(node_type, "kind", None) == "float" \
                    and getattr(node_type, "bits", None) == 32:
                return text + "f"
            return text
        elif isinstance(value, ll.LLType):
            return self._lltype_to_cpp(value)
        else:
            raise NotImplementedError(f"Unsupported constant type: {type(value).__name__} (value: {value})")

    def visit_Num(self, node: ast.Num) -> str:
        """Process numeric constants (Python 3.7 and earlier)."""
        # Convert ast.Num to ast.Constant for compatibility
        const_node = ast.Constant(value=node.n)
        return self.visit_Constant(const_node)

    def visit_Str(self, node: ast.Str) -> str:
        """Process string constants (Python 3.7 and earlier)."""
        # Convert ast.Str to ast.Constant for compatibility
        const_node = ast.Constant(value=node.s)
        return self.visit_Constant(const_node)

    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Process attribute access (e.g., obj.attr)."""
        # Check if this is an enum value access (e.g., GemmType.Normal, IndexType.MN)
        try:
            # Try to resolve the attribute to check if it's an enum value
            attr = recursive_resolve_attribute(node, self.ctx)

            # Check if resolved value is an Enum instance
            if isinstance(attr, Enum):
                enum_class = type(attr)
                enum_name = enum_class.__name__
                enum_member = attr.name
                # Track that this enum is used
                if hasattr(self, 'used_enums'):
                    self.used_enums.add(enum_name)
                # Convert to C++ enum value format: EnumType::EnumValue
                return f"{enum_name}::{enum_member}"

            # Check if the base is an Enum class and attr is an enum member
            base = recursive_resolve_attribute(node.value, self.ctx)
            if isinstance(base, type) and issubclass(base, Enum):
                enum_name = base.__name__
                # Track that this enum is used
                if hasattr(self, 'used_enums'):
                    self.used_enums.add(enum_name)
                # Convert to C++ enum value format: EnumType::EnumValue
                return f"{enum_name}::{node.attr}"
        except Exception:
            # If resolution fails, fall through to normal handling
            pass

        attr = recursive_resolve_attribute(node, self.ctx)
        if isinstance(attr, ast.AST):
            # If it's still an AST node, generate C++ code for it
            base_str = self.visit(node.value)
            return f"{base_str}.{node.attr}"
        # If it's a resolved value (constant), convert to C++ string
        if isinstance(attr, _F32Literal):
            return format_f32_constant(attr)
        if isinstance(attr, _F64Literal):
            return format_f64_constant(attr)
        if isinstance(attr, (int, float, bool)):
            if isinstance(attr, float):
                _warn_bare_float_literal(attr, node.lineno)
            return str(attr)
        elif isinstance(attr, str):
            return f'"{attr}"'
        # A dtype attribute (e.g. ll.float32 used as a value argument) must
        # reach codegen as the C++ type spelling, not the literal source text
        # "ll.float32" (which is not valid C++).  Same conversion as
        # visit_Constant's LLType branch and the val.to(dtype) handling.
        if isinstance(attr, ll.LLType) and hasattr(self, '_lltype_to_cpp'):
            return self._lltype_to_cpp(attr)
        # For other types, generate attribute access
        base_str = self.visit(node.value)
        return f"{base_str}.{node.attr}"

    def visit_BinOp(self, node: ast.BinOp) -> str:
        """Emit binary expressions with C++ precedence rather than nested wrappers."""
        precedence = self._cpp_precedence.get(type(node.op), 0)
        left_cpp = self._operand_cpp(node.left, precedence)
        op_cpp = self._get_operator_str(node.op)
        right_cpp = self._operand_cpp(node.right, precedence, right=True)
        intrinsic = self._fp16_scalar_arith_intrinsic(node)
        if intrinsic is not None:
            return f"{intrinsic}({left_cpp}, {right_cpp})"
        return f"{left_cpp} {op_cpp} {right_cpp}"

    def _get_operator_str(self, op: ast.operator) -> str:
        """Map Python AST operators to C++ operators."""
        registry = get_operator_codegen_registry()
        return registry.get_bin_op(op)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> str:
        """Process unary operations (e.g., not, -x, +x, ~x)."""
        operand_cpp = self._operand_cpp(node.operand, 14)
        if isinstance(node.operand, ast.UnaryOp):
            operand_cpp = f"({operand_cpp})"
        registry = get_operator_codegen_registry()
        op_str = registry.get_un_op(node.op)
        return f"{op_str}{operand_cpp}"

    def visit_BoolOp(self, node: ast.BoolOp) -> str:
        """Process boolean operations (Python 'and'/'or' → C++ '&&'/'||')."""
        registry = get_operator_codegen_registry()
        op_str = registry.get_bool_op(node.op)

        precedence = 5 if isinstance(node.op, ast.And) else 4
        sub_exprs = [self._operand_cpp(expr, precedence) for expr in node.values]
        return f" {op_str} ".join(sub_exprs)

    def visit_Compare(self, node: ast.Compare) -> str:
        """Process comparison expressions (e.g., `a < b <= c`)."""
        sub_conditions = []
        left_node = node.left
        for op, comparator in zip(node.ops, node.comparators):
            precedence = 9 if isinstance(op, (ast.Eq, ast.NotEq, ast.Is, ast.IsNot)) else 10
            left_expr = self._operand_cpp(left_node, precedence)
            op_str = self._get_operator_str(op)
            right_expr = self._operand_cpp(comparator, precedence, right=True)
            sub_conditions.append(f"({left_expr} {op_str} {right_expr})")
            left_node = comparator

        # Conditions already supply their own parentheses; keep the expression readable.
        if len(sub_conditions) == 1:
            return sub_conditions[0][1:-1]
        else:
            return " && ".join(sub_conditions)

    def visit_IfExp(self, node: ast.IfExp) -> str:
        """Process Python ternary expressions (conditional expressions)."""
        condition_cpp = self.visit(node.test)
        true_expr_cpp = self.visit(node.body)
        false_expr_cpp = self.visit(node.orelse)
        return f"({condition_cpp} ? {true_expr_cpp} : {false_expr_cpp})"

    def _wgmma_fragment_element(self, node: ast.Subscript):
        """Lower a subscript of a typed register fragment (WGMMA or mma.sync).

        A fragment declared by ``ll.wgmma_acc_fragment_f32("a", N)``,
        ``ll.mma_acc_fragment_f32("a", N)`` or
        ``ll.mma_operand_fragment_u32("a", N)`` owns N individual register
        scalars ``a_0 .. a_{N-1}`` (there is NO address-taken array), so
        element access is only well defined at a compile-time-unrolled
        index: ``a[i]`` lowers to the scalar name ``a_<i>`` when ``i`` folds
        to a compile-time integer constant (a literal, a provably-static
        integer local, or a source-level-unrolled ``ll.unroll`` loop counter
        binding), mirroring the hand-CUDA register fragment. Returns the
        scalar C string, or None when the base is not a registered fragment.
        A dynamic index on a fragment is a compile error, never a silent
        local-memory demotion.
        """
        if not isinstance(node.value, ast.Name):
            return None
        ctx = getattr(self, "ctx", {})
        fragments = ctx.get(__WGMMA_ACC_FRAGMENTS_CTX_KEY__)
        rec = fragments.get(node.value.id) if fragments else None
        kind = "WGMMA"
        if rec is None:
            mma_fragments = ctx.get(__MMA_SYNC_FRAGMENTS_CTX_KEY__)
            rec = mma_fragments.get(node.value.id) if mma_fragments else None
            kind = "mma.sync"
        if rec is None:
            return None
        num_elems = rec["num_elems"]

        # Constant-fold the index against every binding class visible to
        # codegen: ll.unroll iteration literals (innermost first), then the
        # function's provably-static integer locals.
        index_expr = node.slice
        if isinstance(index_expr, ast.Index):  # Python 3.8 wrapper
            index_expr = index_expr.value
        if isinstance(index_expr, ast.Tuple):
            message = (f"{kind} register fragment {node.value.id!r} is a 1-D "
                       f"per-thread register file of {num_elems} elements; a tuple "
                       f"subscript has no meaning (map to a flat fragment index "
                       f"first, e.g. via a *_accumulator_coordinate helper like "
                       f"wgmma_m64nNk16_accumulator_coordinate or "
                       f"mma_m16n8k16_accumulator_coordinate)")
            formatted = format_error_message(message, node, ctx=self.ctx)
            raise IndexError(formatted)
        # Fold the index against the UNION of every binding class visible to
        # codegen (statics first, then ll.unroll iteration literals
        # outermost-first override): index expressions routinely mix a static
        # base with an unroll counter (e.g. ``base + i``) and must still fold.
        bindings = {}
        statics = getattr(self, "_static_int_consts", {}) or {}
        for name, rec2 in statics.items():
            bindings[name] = rec2.value
        for env in getattr(self, "_unroll_const_envs", ()):
            bindings.update(env)
        value = try_eval_const_int(index_expr, bindings)
        if value is None:
            message = (f"{kind} register fragment {node.value.id!r} requires a "
                       f"compile-time-unrolled (constant) element index, got an "
                       f"expression that does not fold at codegen time; write the "
                       f"access inside ll.unroll(range(...)) or with a literal/"
                       f"provably-static index so the per-element register scalar "
                       f"is selected at compile time")
            formatted = format_error_message(message, node.slice, ctx=self.ctx)
            raise IndexError(formatted)
        if not 0 <= value < num_elems:
            message = (f"{kind} register fragment {node.value.id!r} index {value} "
                       f"out of range for {num_elems}-element fragment")
            formatted = format_error_message(message, node.slice, ctx=self.ctx)
            raise IndexError(formatted)
        return f"{node.value.id}_{value}"

    def visit_Subscript(self, node: ast.Subscript) -> str:
        """Process subscript/indexing operations (e.g., A[i], B[0])."""
        fragment_element = self._wgmma_fragment_element(node)
        if fragment_element is not None:
            return fragment_element
        base_str = self.visit(node.value)

        # Tuple element access: t[<const int>] on a TupleType value lowers to
        # cuda::std::get<i>(t) (e.g. single-name capture of a multi-value builtin).
        if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
            try:
                from little_kernel.core.type_system import TupleType
                base_type = self.get_type(node.value)
            except Exception:
                base_type = None
            if isinstance(base_type, TupleType):
                idx = node.slice.value
                if not 0 <= idx < len(base_type.element_types):
                    raise IndexError(f"Tuple index {idx} out of range for {base_type}")
                return f"{self.tuple_namespace}::get<{idx}>({base_str})"

        # Handle different index types
        if isinstance(node.slice, ast.Index):
            # Python 3.8 and earlier: ast.Index wrapper
            index_elem = node.slice.value
        elif isinstance(node.slice, ast.Tuple):
            # Multi-dimensional indexing (e.g., A[i, j])
            flattened = self._try_flatten_local_array_subscript(node, base_str)
            if flattened is not None:
                return flattened
            indices = [
                self._cpp_wide_subscript_index(elem, self._process_single_index(elem)) for elem in node.slice.elts
            ]
            return f"{base_str}[{']['.join(indices)}]"
        else:
            # Python 3.9+: direct value
            self._check_single_subscript_of_local_array(node)
            index_elem = node.slice

        index_str = self._process_single_index(index_elem)
        index_str = self._cpp_wide_subscript_index(index_elem, index_str)
        return f"{base_str}[{index_str}]"

    def _cpp_wide_subscript_index(self, index_elem: ast.AST, index_str: str) -> str:
        """Emit an explicitly 64-bit element offset for a 64-bit-typed index.

        Guards against backends (``_wide_i64_subscript_offset`` opt-in) whose
        generated C++ must prove the C usual-arithmetic conversions keep the
        subscript 64-bit: a ``base[(ptrdiff_t)(i)]`` element subscript is
        pointer + ptrdiff_t arithmetic, which cannot silently truncate a
        large int64 element index to C int the way a narrow intermediate
        (e.g. ``size_t((int)(i)) * sizeof``) would. The decision is purely
        type-directed (the index expression's LLType); constant Python ints
        without an IR type keep the plain emission, since they are typed by
        the frontend before reaching here when they exceed int32.

        The default keeps every existing emission byte-identical; only
        emitters that opt in (CUDA) wrap 64-bit-typed indices.
        """
        if not getattr(self, "_wide_i64_subscript_offset", False):
            return index_str
        try:
            index_lltype = self._resolve_expr_lltype(index_elem)
        except Exception:
            index_lltype = None
        if (isinstance(index_lltype, ll.IntType) and not index_lltype.special and index_lltype.bits == 64):
            # <cstddef> (ptrdiff_t) is emitted by the backend's conditional
            # header pass (CudaEmitter scans the body for "ptrdiff_t").
            return f"(ptrdiff_t)({index_str})"
        return index_str

    def _local_array_shape(self, node: ast.Subscript):
        """Recorded extents of a local (register-scope) array base, or None."""
        if not isinstance(node.value, ast.Name):
            return None
        shapes = getattr(self, "ctx", {}).get(__LOCAL_ARRAY_SHAPES_CTX_KEY__)
        if not shapes:
            return None
        return shapes.get(node.value.id)

    def _check_single_subscript_of_local_array(self, node: ast.Subscript) -> None:
        """Reject a single-index subscript of a rank>=2 local array.

        Local arrays are declared as flat 1-D C arrays, so `a[i]` on a
        `float a[R * C];` is *valid* C but silently reads element `i` of the
        flat array rather than anything the declared rank-2 shape implies —
        a mis-index, not a compile error. Require the full tuple subscript."""
        shape = self._local_array_shape(node)
        if shape is None or len(shape) < 2:
            return
        message = (f"local array {node.value.id!r} declared with shape "
                   f"{tuple(shape)!r} requires {len(shape)} subscript indices, "
                   f"got 1")
        formatted = format_error_message(message, node, ctx=self.ctx)
        raise IndexError(formatted)

    def _try_flatten_local_array_subscript(self, node: ast.Subscript, base_str: str):
        """Flatten a tuple subscript of a local (register-scope) array.

        Local arrays are declared as flat 1-D C arrays (`float a[R * C];`) by
        alloc_local_memory codegen, so a multi-dimensional subscript `a[i, j]`
        must lower to the matching flat index `a[(i) * C + (j)]` (row-major,
        trailing dims collapsed) rather than the invalid `a[i][j]`. A tuple
        subscript whose arity mismatches the recorded rank is rejected: the
        legacy emission would be invalid C for every flat local declaration.

        Returns the flattened C string, or None when the base is not a recorded
        local array (the caller then keeps the legacy emission).
        """
        shape = self._local_array_shape(node)
        if shape is None:
            return None
        elts = node.slice.elts
        if len(elts) != len(shape):
            message = (f"local array {node.value.id!r} declared with shape "
                       f"{tuple(shape)!r} requires {len(shape)} subscript indices, "
                       f"got {len(elts)}")
            formatted = format_error_message(message, node, ctx=self.ctx)
            raise IndexError(formatted)
        flat_terms = []
        # Row-major flat index: sum over positions of index_k * prod(shape[k+1:]).
        for k, elem in enumerate(elts):
            stride = 1
            for d in shape[k + 1:]:
                stride *= d
            idx_cpp = self._process_single_index(elem)
            if stride == 1:
                flat_terms.append(f"({idx_cpp})")
            else:
                flat_terms.append(f"({idx_cpp}) * {stride}")
        return f"{base_str}[{' + '.join(flat_terms)}]"

    def _process_single_index(self, index_elem: ast.AST) -> str:
        """Helper to process a single index element."""

        # Reject slice indices
        if isinstance(index_elem, ast.Slice):
            message = "Slice indices (e.g., A[1:5]) are not supported yet"
            formatted = format_error_message(message, index_elem, ctx=self.ctx)
            raise NotImplementedError(formatted)

        # Visit the index element to get its string representation
        index_str = self.visit(index_elem)

        # Validate the index type (should be integer-compatible)
        # For constants, infer type from the value
        if isinstance(index_elem, ast.Constant):
            if not isinstance(index_elem.value, int):
                message = (f"Subscript index must be a constant integer, "
                           f"got {type(index_elem.value).__name__}: {index_elem.value}")
                formatted = format_error_message(message, index_elem, ctx=self.ctx)
                raise TypeError(formatted)
        else:
            # For non-constants, check type if available
            try:
                index_type = self.get_type(index_elem)
                if not (isinstance(index_type, ll.IntType) and not index_type.special):
                    message = (f"Subscript index must be a regular integer type (e.g., int32, uint64), "
                               f"got {index_type}")
                    formatted = format_error_message(message, index_elem, ctx=self.ctx)
                    raise TypeError(formatted)
            except (AssertionError, KeyError):
                # If type is not available, assume it's valid (might be a constant or already validated)
                pass

        return index_str

    def visit_JoinedStr(self, node: ast.JoinedStr) -> str:
        """Process f-strings (JoinedStr) → C++ string concatenation."""
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant):
                # String literal part
                parts.append(f'"{value.value}"')
            elif isinstance(value, ast.FormattedValue):
                # Formatted value (e.g., {BLOCK_M})
                formatted_value_cpp = self.visit(value.value)
                if not isinstance(formatted_value_cpp, str):
                    formatted_value_cpp = str(formatted_value_cpp)
                # Check if the result is already a string literal
                if formatted_value_cpp.startswith('"') and formatted_value_cpp.endswith('"'):
                    parts.append(formatted_value_cpp)
                else:
                    # Convert to string using std::to_string
                    parts.append(f'std::to_string({formatted_value_cpp})')
            else:
                # Other types - visit and convert to string
                formatted_value = self.visit(value)
                if isinstance(formatted_value,
                              str) and formatted_value.startswith('"') and formatted_value.endswith('"'):
                    parts.append(formatted_value)
                else:
                    parts.append(f'std::to_string({formatted_value})')

        if len(parts) == 1:
            return parts[0]
        else:
            # Concatenate strings in C++
            return " + ".join(parts)

    def visit_FormattedValue(self, node: ast.FormattedValue) -> str:
        """Process formatted value in f-string (e.g., {variable})."""
        return self.visit(node.value)

    def visit_List(self, node: ast.List) -> str:
        """Process list literals (e.g., [1, 2, 3] or [ArraySize]).
        
        For code generation, lists are typically used for:
        1. Shape parameters (should be folded to constants by const_fold pass)
        2. Array initializations (converted to C++ array initialization)
        
        If all elements are constants, convert to C++ array initialization.
        Otherwise, this might indicate an issue with const folding.
        """
        elements = []
        for elt in node.elts:
            elt_str = self.visit(elt)
            elements.append(elt_str)

        # For now, return as comma-separated values
        # This is typically used for shape parameters that should have been folded
        # If we reach here, it means const_fold didn't fold the list
        return f"{{{', '.join(elements)}}}"

    def visit_Tuple(self, node: ast.Tuple) -> str:
        """Process tuple expressions (e.g., return (a, b, c) -> cuda::std::make_tuple(a, b, c))."""
        elements = [self.visit(elt) for elt in node.elts]
        return f"{self.tuple_namespace}::make_tuple({', '.join(elements)})"
