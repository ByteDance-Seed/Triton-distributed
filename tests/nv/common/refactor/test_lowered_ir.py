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
"""RF06 typed lowered IR tests.

Covers construction contracts, malformed-node rejection, pointer result
preservation, declared effects/source spans, AST clone/traversal/dump
preservation, type inference, C++ codegen and legacy ``_lk_cpp_*`` read
compatibility.
"""

import ast
import copy

import pytest

import little_kernel.language as ll
from little_kernel.backends.nv.common.codegen.emitter import CudaEmitter as CppEmitter
from little_kernel.core import ir
from little_kernel.core.passes.constfold import ConstantFolder
from little_kernel.core.passes.pass_base import CompleteASTMutator
from little_kernel.core.passes.utils.type_inference import TypeInferencer

CTX = {"ll": ll}


def _attr(path):
    parts = path.split(".")
    node = ast.Name(parts[0], ast.Load())
    for part in parts[1:]:
        node = ast.Attribute(node, part, ast.Load())
    return node


def _ptr_annotation(element="float16"):
    return ast.Subscript(_attr("ll.ptr"), _attr(f"ll.{element}"), ast.Load())


def _void_kernel(body):
    return ast.Module(
        body=[
            ast.FunctionDef(name="k", args=ast.arguments([], [], None, [], [], None, []),
                            body=body + [ast.Return(value=None)], decorator_list=[], returns=_attr("ll.void"))
        ], type_ignores=[])


# ----------------------------------------------------------------------------
# Construction and declared contracts
# ----------------------------------------------------------------------------


def test_target_expression_declares_type_effect_and_span():
    span = ir.SourceSpan(lineno=12, col_offset=3)
    node = ir.TargetExpression("rank()", ll.int32, effect=ir.READ_EFFECT, span=span)
    assert isinstance(node, ast.expr)
    assert node.spelling == "rank()"
    assert node.result_type is ll.int32
    assert node.effect is ir.READ_EFFECT
    assert node.span.lineno == 12
    # span mirrors onto standard ast location attributes
    assert node.lineno == 12
    assert ir.is_lowered_node(node)


def test_target_expression_accepts_string_effect():
    node = ir.target_expression("x()", ll.int32, effect="barrier")
    assert node.effect is ir.BARRIER_EFFECT


@pytest.mark.parametrize("spelling", ["", "   ", 7])
def test_target_expression_rejects_empty_spelling(spelling):
    with pytest.raises((TypeError, ValueError)):
        ir.TargetExpression(spelling, ll.int32)


def test_omitted_required_fields_produce_invalid_node_verified():
    # Zero-arg construction exists only for AST cloning; such a node must
    # never pass the verifier (no silent fallback).
    node = ir.TargetExpression()
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(node)


def test_result_type_must_be_lltype_no_int32_catchall():
    # No implicit int32 default: an explicit LLType is mandatory for direct
    # construction, and a non-LLType is rejected.
    with pytest.raises(TypeError):
        ir.TargetExpression("x()", "int32")
    with pytest.raises(TypeError):
        ir.TargetExpression("x()", int)
    node = ir.TargetExpression("x()", None)
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(node)


def test_invalid_effect_rejected_not_defaulted():
    with pytest.raises(TypeError):
        ir.TargetExpression("x()", ll.int32, effect="does-not-exist")


def test_target_include_requires_bracketed_directive():
    assert ir.target_include("<a/b.cuh>").include == "<a/b.cuh>"
    assert ir.target_include('"role.cuh"').include == '"role.cuh"'
    for bad in ["a/b.cuh", "#include <a.h>", "<a>.h", '"unterminated', "<a\nb.h>", ""]:
        with pytest.raises((TypeError, ValueError)):
            ir.target_include(bad)


def test_target_statement_rejects_pure_and_bad_section():
    node = ir.target_statement("f();", effect=ir.WRITE_EFFECT)
    assert node.section == "main"
    with pytest.raises(ValueError):
        ir.target_statement("f();", effect=ir.PURE_EFFECT)
    with pytest.raises(ValueError):
        ir.target_statement("f();", section="header")


def test_target_helper_validates_name():
    h = ir.target_helper("void h(){}", name="h")
    assert h.name == "h"
    with pytest.raises(ValueError):
        ir.target_helper("void x(){}", name="1bad-name")


def test_source_span_from_ast_node_and_synthetic():
    n = ast.parse("x", mode="eval").body
    span = ir.span_from_node(n)
    assert span.lineno == 1
    assert ir.span_from_node(None).is_synthetic
    assert str(ir.SourceSpan(lineno=2, col_offset=0)) == "line 2, column 1"


# ----------------------------------------------------------------------------
# Pointer-result contracts (the core RF06 defect)
# ----------------------------------------------------------------------------


def test_resource_reference_is_typed_pointer_not_int32():
    rr = ir.resource_reference("__lk_smem.A[0]", ll.float16, space="smem")
    assert rr.result_type.is_pointer()
    assert rr.result_type.inner_type is ll.float16
    assert rr.space == "smem"
    assert rr.effect is ir.READ_EFFECT


def test_resource_reference_validates_space_and_resource_path():
    with pytest.raises(ValueError):
        ir.resource_reference("a.b", ll.int32, space="registers")
    # Arbitrary raw text cannot masquerade as a resource name.
    for bad in ["a + b", "a();", "a->b", "x = 1"]:
        with pytest.raises(ValueError):
            ir.resource_reference(bad, ll.int32)
    # Dotted identifier paths and a single trailing index are allowed.
    assert ir.resource_reference("__lk_smem.buffer", ll.int32).resource
    assert ir.resource_reference("buf[7]", ll.int32).resource


def test_stage_reference_retains_pointer_element_type():
    base = ir.resource_reference("buf", ll.float16, space="smem")
    stage = ir.target_expression("pipe.stage()", ll.int32, effect=ir.READ_EFFECT)
    sr = ir.stage_reference(base, stage, 16)
    assert sr.result_type.is_pointer()
    assert sr.result_type.inner_type is ll.float16


def test_stage_reference_requires_consistent_element_type():
    base = ir.resource_reference("buf", ll.float16, space="smem")
    stage = ast.Name("s", ast.Load())
    with pytest.raises(TypeError):
        ir.stage_reference(base, stage, 16, element_type=ll.int32)
    with pytest.raises(ValueError):
        ir.stage_reference(base, stage, -4)


def test_stage_reference_requires_integer_stage():
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    base = ir.resource_reference("buf", ll.float16, space="smem")
    stage = ir.target_expression("flag()", ll.float32, effect=ir.READ_EFFECT)
    sr = ir.stage_reference(base, stage, 8)
    # Constructor does not infer; inference must reject the non-int stage.
    with pytest.raises(Exception):
        inf.infer_node_type(sr)


# ----------------------------------------------------------------------------
# Verifier
# ----------------------------------------------------------------------------


def test_verifier_rejects_corrupted_node():
    node = ir.target_expression("x()", ll.int32)
    node.result_type = "not-a-type"
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(node)


def test_verifier_rejects_resource_reference_without_pointer_result():
    rr = ir.resource_reference("buf", ll.int32)
    rr.result_type = ll.int32  # corrupt the pointer invariant
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(rr)


def test_verifier_collects_errors_without_raising():
    good = ir.target_expression("ok()", ll.int32)
    bad = ir.resource_reference("buf", ll.int32)
    bad.space = "bogus"
    errors = ir.verify_lowered_ir(ast.Module([good, ast.Expr(bad)], []), raise_on_error=False)
    assert len(errors) == 1
    assert isinstance(errors[0], ir.LoweredIRVerificationError)


def test_verifier_rejects_pure_statement_backstop():
    # TargetStatement rejects pure at construction; corrupting one exercises
    # the verifier's independent backstop.
    node = ir.target_statement("f();")
    node.effect = ir.PURE_EFFECT
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(node)


def test_verifier_non_lowered_node_is_passthrough():
    n = ast.Name("x", ast.Load())
    assert ir.verify_lowered_node(n) is n


# ----------------------------------------------------------------------------
# AST cloning / traversal / dumps / source locations
# ----------------------------------------------------------------------------


def test_clone_preserves_types_effects_and_spans():
    te = ir.TargetExpression("rank()", ll.ptr[ll.float16], ir.READ_EFFECT, ir.SourceSpan(lineno=9))
    tree = _void_kernel([ast.Assign([ast.Name("p", ast.Store())], te)])
    clone = copy.deepcopy(tree)
    cloned = clone.body[0].body[0].value
    assert isinstance(cloned, ir.TargetExpression)
    assert cloned.result_type == ll.ptr[ll.float16]
    assert cloned.result_type.is_pointer()
    assert cloned.effect is ir.READ_EFFECT
    assert cloned.lineno == 9
    ir.verify_lowered_ir(clone)


def test_traversal_and_dump_show_semantics():
    inc = ir.target_include("<cuda_fp16.h>")
    stmt = ir.target_statement("__syncthreads();", effect=ir.BARRIER_EFFECT, span=ir.SourceSpan(lineno=5))
    tree = _void_kernel([inc, stmt])
    lowered = list(ir.iter_lowered_nodes(tree))
    assert len(lowered) == 2
    dumped = ast.dump(tree, include_attributes=True)
    assert "TargetInclude" in dumped
    assert "TargetStatement" in dumped
    assert "lineno=5" in dumped
    assert "__syncthreads" in dumped


# ----------------------------------------------------------------------------
# Type inference
# ----------------------------------------------------------------------------


def test_type_inference_uses_declared_types():
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    te = ir.target_expression("pipe.get()", ll.ptr[ll.uint64], effect=ir.READ_EFFECT)
    rr = ir.resource_reference("smem_buf", ll.float16, space="smem")
    stage = ir.target_expression("s()", ll.int32, effect=ir.READ_EFFECT)
    sr = ir.stage_reference(rr, stage, 32)
    assert inf.infer_node_type(te) == ll.ptr[ll.uint64]
    assert inf.infer_node_type(rr).is_pointer()
    assert inf.infer_node_type(sr).inner_type is ll.float16


def test_statement_nodes_infer_without_result():
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    # Must not raise and must not register a value type.
    assert inf.visit(ir.target_statement("f();", effect=ir.ESCAPE_EFFECT)) is None
    assert inf.visit(ir.target_include("<x.h>")) is None


# ----------------------------------------------------------------------------
# Codegen
# ----------------------------------------------------------------------------


def _emit(body, all_types_from_inference=True):
    tree = _void_kernel(body)
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    inf.visit(tree)
    emitter = CppEmitter(ctx=CTX, all_types=inf.all_types)
    emitter.visit(tree)
    return tree, inf, emitter.get_code()


def test_codegen_emits_target_statement_include_helper():
    _, _, code = _emit([
        ir.target_include("<cuda_fp16.h>"),
        ir.target_helper("__device__ int role_h(){return 0;}", name="role_h"),
        ir.target_statement("__syncthreads();", effect=ir.BARRIER_EFFECT),
    ])
    assert "#include <cuda_fp16.h>" in code
    assert "int role_h" in code
    assert "__syncthreads();" in code


def test_codegen_deduplicates_includes_and_helpers():
    _, _, code = _emit([
        ir.target_include("<cuda_fp16.h>"),
        ir.target_include("<cuda_fp16.h>"),
        ir.target_helper("__device__ int dup(){return 1;}", name="dup"),
        ir.target_helper("__device__ int dup(){return 1;}", name="dup"),
    ])
    assert code.count("#include <cuda_fp16.h>") == 1
    assert code.count("int dup") == 1


def test_codegen_pointer_stage_reference_renders_and_types():
    rr = ir.resource_reference("__lk_smem.A[0]", ll.float16, space="smem")
    stage = ir.target_expression("pipe.producer_stage()", ll.int32, effect=ir.READ_EFFECT)
    sr = ir.stage_reference(rr, stage, 8)
    _, inf, code = _emit([
        ast.AnnAssign(ast.Name("p", ast.Store()), _ptr_annotation(), sr, 1),
    ])
    assert inf.all_types[sr].is_pointer()
    # The opaque stage expression is conservatively grouped; resource/stage
    # pointer semantics remain intact.
    assert "__half* p = (__lk_smem.A[0] + ((pipe.producer_stage())) * 8);" \
        in code


def test_codegen_builtin_section_statement():
    stmt = ir.TargetStatement("namespace ns { int v; }", effect=ir.ESCAPE_EFFECT, section="builtin")
    _, _, code = _emit([stmt])
    assert "namespace ns" in code


def test_codegen_does_not_emit_spelling_as_identifier():
    # A target expression is emitted as its validated text, never resolved as
    # a Python identifier from ctx.
    te = ir.target_expression("rank<2>()", ll.int32, effect=ir.READ_EFFECT)
    _, _, code = _emit([ast.Assign([ast.Name("v", ast.Store())], te)])
    assert "rank<2>()" in code


# ----------------------------------------------------------------------------
# Legacy _lk_cpp_* read compatibility (temporary; dataflow worker migration)
# ----------------------------------------------------------------------------


def test_legacy_recognition_and_spelling():
    p = ast.Pass()
    p._lk_cpp_stmt = "f();"
    assert ir.is_legacy_cpp_node(p)
    assert ir.legacy_cpp_kind(p) is ir.LegacyCppKind.STATEMENT
    assert ir.legacy_cpp_spelling(p) == "f();"

    n = ast.Name(id=ir.LEGACY_CPP_EXPR_NAME, ctx=ast.Load())
    n._lk_cpp_expr = "raw()"
    assert ir.legacy_cpp_kind(n) is ir.LegacyCppKind.EXPRESSION
    # Historical int32 default applies only to unmigrated legacy nodes.
    assert ir.legacy_cpp_expr_type(n) is ll.int32


def test_legacy_expr_can_declare_pointer_type():
    n = ast.Name(id=ir.LEGACY_CPP_EXPR_NAME, ctx=ast.Load())
    n._lk_cpp_expr = "ptr()"
    n._lk_cpp_type = ll.ptr[ll.uint64]
    assert ir.legacy_cpp_expr_type(n).is_pointer()


def test_legacy_nodes_still_codegen_and_infer():
    p_inc = ast.Pass()
    p_inc._lk_cpp_include = "<cuda_fp16.h>"
    p_stmt = ast.Pass()
    p_stmt._lk_cpp_stmt = "__syncthreads();"
    n = ast.Name(id=ir.LEGACY_CPP_EXPR_NAME, ctx=ast.Load())
    n._lk_cpp_expr = "legacy_call()"

    tree, inf, code = _emit([
        p_inc,
        p_stmt,
        ast.Assign([ast.Name("a", ast.Store())], n),
    ])
    assert inf.all_types[n] is ll.int32  # legacy-only default
    assert "#include <cuda_fp16.h>" in code
    assert "__syncthreads();" in code
    assert "legacy_call()" in code


def test_legacy_pointer_expr_keeps_pointer_type():
    n = ast.Name(id=ir.LEGACY_CPP_EXPR_NAME, ctx=ast.Load())
    n._lk_cpp_expr = "ptr_call()"
    n._lk_cpp_type = ll.ptr[ll.float16]
    tree = _void_kernel([
        ast.AnnAssign(ast.Name("q", ast.Store()), _ptr_annotation(), n, 1),
    ])
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    inf.visit(tree)
    assert inf.all_types[n].is_pointer()
    emitter = CppEmitter(ctx=CTX, all_types=inf.all_types)
    emitter.visit(tree)
    code = emitter.get_code()
    # Legacy opaque expression is conservatively grouped too (semantics over
    # incidental spelling); pointer type is still honored.
    assert "__half* q = (ptr_call());" in code


def test_plain_pass_remains_noop():
    tree = _void_kernel([ast.Pass()])
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    inf.visit(tree)
    emitter = CppEmitter(ctx=CTX, all_types=inf.all_types)
    emitter.visit(tree)  # must not raise


# ----------------------------------------------------------------------------
# P1: opaque backend expressions are conservatively grouped
# ----------------------------------------------------------------------------


def _emit_assignment_line(expr):
    """Emit ``v = <expr>;`` inside a void kernel and return that line."""
    tree = _void_kernel([ast.Assign([ast.Name("v", ast.Store())], expr)])
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    inf.visit(tree)
    emitter = CppEmitter(ctx=CTX, all_types=inf.all_types)
    emitter.visit(tree)
    lines = [l.strip() for l in emitter.get_code().splitlines() if "v =" in l]
    assert lines, "assignment line was not emitted"
    return lines[0]


def test_opaque_expr_grouped_inside_multiplication():
    # The reported defect: nested '1 + 2' under Mult emitted '1 + 2 * 3' (7).
    opaque = ir.TargetExpression("1 + 2", ll.int32, effect=ir.PURE_EFFECT)
    mult = ast.BinOp(opaque, ast.Mult(), ast.Constant(3))
    line = _emit_assignment_line(mult)
    assert "(1 + 2) * 3" in line
    assert "1 + 2 * 3" not in line


def test_opaque_expr_grouped_inside_ternary():
    opaque = ir.TargetExpression("a ? b : c", ll.int32, effect=ir.PURE_EFFECT)
    ternary = ast.IfExp(ast.Constant(True), opaque, ast.Constant(1))
    line = _emit_assignment_line(ternary)
    assert "(a ? b : c)" in line


def test_opaque_expr_grouped_in_argument_context():
    # An opaque comma-expression as an argument stays one grouped operand.
    opaque = ir.TargetExpression("x, y", ll.int32, effect=ir.PURE_EFFECT)
    emitter = CppEmitter(ctx=CTX, all_types={})
    assert emitter.visit(opaque) == "(x, y)"


def test_grouping_helper_always_wraps():
    assert ir.grouped_target_spelling("1 + 2") == "(1 + 2)"
    assert ir.grouped_target_spelling("rank()") == "(rank())"


def test_resource_and_stage_pointer_semantics_survive_grouping():
    rr = ir.resource_reference("__lk_smem.A[0]", ll.float16, space="smem")
    stage = ir.TargetExpression("pipe.stage()", ll.int32, effect=ir.READ_EFFECT)
    sr = ir.stage_reference(rr, stage, 8)
    line = _emit_assignment_line(sr)
    assert "__lk_smem.A[0]" in line  # validated resource path verbatim
    assert "(pipe.stage())" in line  # opaque stage grouped
    assert sr.result_type.is_pointer()  # pointer semantics retained


# ----------------------------------------------------------------------------
# P2: AST mutators clone-and-transform lowered nodes (never in place)
# ----------------------------------------------------------------------------


def test_constant_folder_clones_stage_reference_and_preserves_input():
    stage = ast.BinOp(ast.Constant(1), ast.Add(), ast.Constant(2))
    original = ir.stage_reference(ir.resource_reference("buf", ll.float16, space="smem"), stage, 16)
    result = ConstantFolder(ctx=CTX, all_types={}).visit(original)
    assert result is not original
    assert isinstance(result, ir.StageReference)
    assert isinstance(result.stage, ast.Constant)
    assert result.stage.value == 3
    # The ORIGINAL stage remains the original BinOp (no in-place mutation).
    assert isinstance(original.stage, ast.BinOp)


def test_mutator_preserves_immutable_identities():
    stage = ast.BinOp(ast.Constant(1), ast.Add(), ast.Constant(2))
    base = ir.resource_reference("buf", ll.float16, space="smem")
    original = ir.stage_reference(base, stage, 16)
    result = ConstantFolder(ctx=CTX, all_types={}).visit(original)
    assert result.result_type is original.result_type
    assert result.result_type.is_pointer()
    assert result.span is original.span
    assert result.effect is original.effect
    assert result.base is not original.base
    assert result.base.result_type is original.base.result_type
    ir.verify_lowered_node(result)


def test_complete_mutator_clones_lowered_identity_transform():
    node = ir.TargetExpression("r()", ll.ptr[ll.float16], effect=ir.READ_EFFECT)
    out = CompleteASTMutator().visit(node)
    assert out is not node
    assert isinstance(out, ir.TargetExpression)
    assert out.spelling == "r()"
    assert out.result_type is node.result_type  # shared immutable type


def test_mutator_does_not_mutate_target_expression_input():
    opaque = ir.TargetExpression("opaque()", ll.int32, effect=ir.READ_EFFECT)
    folded = ConstantFolder(ctx=CTX, all_types={}).visit(ast.BinOp(ast.Constant(4), ast.Add(), ast.Constant(5)))
    assert isinstance(folded, ast.Constant) and folded.value == 9
    cloned = ConstantFolder(ctx=CTX, all_types={}).visit(opaque)
    assert cloned is not opaque
    assert cloned.spelling == opaque.spelling


# ----------------------------------------------------------------------------
# P3: verification rejects incomplete/corrupt nodes at every boundary
# ----------------------------------------------------------------------------


def _blank_resource_reference(element_type):
    node = ir.ResourceReference()  # blank: supported for AST reconstruction
    node.element_type = element_type
    node.result_type = ll.ptr[element_type]
    node.space = "generic"
    node.effect = ir.READ_EFFECT
    node.span = ir.SourceSpan(lineno=1)
    node.symbol = None
    node.resource = None
    return node


def test_missing_resource_path_rejected_by_verifier():
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(_blank_resource_reference(ll.int32))


def test_missing_resource_path_rejected_by_inference():
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    with pytest.raises(ir.LoweredIRVerificationError):
        inf.infer_node_type(_blank_resource_reference(ll.int32))


def test_missing_resource_path_rejected_at_codegen_boundary():
    emitter = CppEmitter(ctx=CTX, all_types={})
    with pytest.raises(ir.LoweredIRVerificationError):
        emitter.visit_ResourceReference(_blank_resource_reference(ll.int32))


def test_corrupted_resource_path_rejected():
    node = ir.resource_reference("buf", ll.int32)
    node.resource = "a + b"  # corrupt after construction
    with pytest.raises(ir.LoweredIRVerificationError):
        ir.verify_lowered_node(node)


def test_valid_resource_and_stage_pass_all_boundaries():
    rr = ir.resource_reference("__lk_smem.buf[0]", ll.float16, space="smem")
    stage = ir.TargetExpression("pipe.stage()", ll.int32, effect=ir.READ_EFFECT)
    sr = ir.stage_reference(rr, stage, 8)
    ir.verify_lowered_node(rr)
    ir.verify_lowered_node(sr)
    inf = TypeInferencer(ctx=CTX, scope_vars={})
    assert inf.infer_node_type(sr).is_pointer()
    emitter = CppEmitter(ctx=CTX, all_types=inf.all_types)
    assert emitter.visit(rr) == "__lk_smem.buf[0]"


def test_ident_path_validator():
    for good in ["a", "a.b", "a.b.c", "buf[7]", "__lk_smem.A[0]", "x[idx]"]:
        assert ir.is_valid_ident_path(good)
    for bad in [None, "", "a + b", "f()", "a,b", "x->y", "a[0]b"]:
        assert not ir.is_valid_ident_path(bad)


def test_opaque_expression_verified_at_codegen_boundary():
    node = ir.TargetExpression("x()", ll.int32)
    node.result_type = "not-a-type"  # corrupt
    emitter = CppEmitter(ctx=CTX, all_types={})
    with pytest.raises(ir.LoweredIRVerificationError):
        emitter.visit_TargetExpression(node)
