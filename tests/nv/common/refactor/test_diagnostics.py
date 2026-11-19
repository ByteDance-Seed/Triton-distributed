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
"""RF16 / nested-return diagnostics regressions (CPU-only).

These tests pin three contracts:

* LLType casts (``ll.int32(x)``) cannot swallow undefined variables or
  invalid sub-expressions -- the operand's program error propagates with
  its source node.
* Return checking uses a function return-context stack, so a ``return``
  nested in if/for/while validates against the function that lexically
  contains it, and a nested function's signature never leaks outward.
* The special-struct resolver keeps its supported optional probes (an
  exact registered-name fallback) but no longer hijacks unrelated names
  through case-insensitive guessing, and unsupported optional probes stay
  distinct from invalid input.
"""

import ast
import textwrap

import pytest

import little_kernel.language as ll
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.core.passes.special_struct_materialize_pass import SpecialStructMaterializeTransformer
from little_kernel.core.passes.utils.type_inference import TypeInferencer, TypeInferenceError, TypeMismatchError, UndefinedVariableError, get_variable_type_from_scope
from little_kernel.core.passes.utils.scope_manager import ScopeManager

CTX = {"ll": ll}


def infer_source(source: str, ctx=None):
    """Run type inference over ``source`` with a minimal ll-only context."""
    tree = ast.parse(textwrap.dedent(source))
    inferencer = TypeInferencer(ctx=ctx if ctx is not None else dict(CTX), scope_vars={})
    inferencer.visit(tree)
    return inferencer


def expect_error(source: str, error_type, *, snippet: str = None):
    tree = ast.parse(textwrap.dedent(source))
    inferencer = TypeInferencer(ctx=dict(CTX), scope_vars={})
    with pytest.raises(error_type) as exc_info:
        inferencer.visit(tree)
    error = exc_info.value
    # Source-located: every failure carries the offending AST node and a
    # rendered Location line; no caller-stack frame guessing is involved.
    assert error.node is not None
    assert "Location:" in str(error)
    if snippet is not None:
        assert snippet in str(error)
    return error


class TestCastDoesNotHideProgramErrors:

    def test_bare_undefined_return_still_raises(self):
        err = expect_error("""
            def f() -> ll.int32:
                return missing
            """, UndefinedVariableError, snippet="missing")
        # The node is the offending Name at its own location, not the return.
        assert isinstance(err.node, ast.Name)
        assert err.node.id == "missing"

    def test_int32_cast_of_undefined_raises(self):
        # RF16 evidence: previously ll.int32(missing) was silently accepted.
        err = expect_error(
            """
            def f() -> ll.int32:
                return ll.int32(missing)
            """, UndefinedVariableError, snippet="missing")
        assert isinstance(err.node, ast.Name)

    def test_nested_cast_of_undefined_raises(self):
        expect_error(
            """
            def f(x: ll.int32) -> ll.int32:
                if x:
                    return ll.int32(missing)
                return 0
            """, UndefinedVariableError)

    def test_cast_preserves_invalid_subexpression(self):
        # An invalid operand inside a cast must surface, not be swallowed by
        # the constructor's argument inference.
        expect_error(
            """
            def f(x: ll.int32) -> ll.int32:
                return ll.float32(x + missing)
            """, UndefinedVariableError)

    def test_valid_cast_argument_is_typed(self):
        # Supported-positive: casts still infer their argument so codegen
        # sees the operand type (the f32-literal contract).
        inferencer = infer_source("""
            def f(x: ll.float32) -> ll.int32:
                return ll.int32(x * 2.0)
            """)
        call_nodes = [n for n in inferencer.all_types if isinstance(n, ast.Call)]
        assert call_nodes
        assert all(isinstance(t, ll.LLType) for t in inferencer.all_types.values())

    def test_cast_argument_binaryop_literal_typed(self):
        inferencer = infer_source("""
            def f(x: ll.float32) -> ll.float16:
                return ll.float16(x * 2.0)
            """)
        binops = [n for n in inferencer.all_types if isinstance(n, ast.BinOp)]
        assert binops, "cast argument subtree must be inferred"


class TestNestedReturnContextStack:

    @pytest.mark.parametrize("wrapper", [
        "if x:\n{body}",
        "for i in range(3):\n{body}",
        "while x:\n{body}",
        "if x:\n    pass\nelse:\n{body}",
    ])
    def test_nested_value_type_mismatch(self, wrapper):
        body = "    return 1.5"
        nested = wrapper.format(body=body)
        source = ("def f(x: ll.int32) -> ll.int32:\n" + "\n".join("    " + line if line else ""
                                                                  for line in nested.splitlines()) + "\n    return 0\n")
        err = expect_error(source, TypeMismatchError)
        assert isinstance(err.node, ast.Return)
        assert "float32" in str(err)
        assert "int32" in str(err)

    def test_nested_value_return_from_void(self):
        expect_error(
            """
            def f(x: ll.int32) -> ll.void:
                if x:
                    return 1
                return
            """, TypeMismatchError, snippet="void")

    def test_nested_bare_return_from_value_function(self):
        expect_error(
            """
            def f(x: ll.int32) -> ll.int32:
                while x:
                    return
                return 0
            """, TypeMismatchError)

    def test_nested_undefined_in_return(self):
        expect_error(
            """
            def f(x: ll.int32) -> ll.int32:
                if x:
                    return unknown_thing
                return 0
            """, UndefinedVariableError)

    def test_valid_nested_returns(self):
        # Direct, if/for/while and else-branch returns that match are fine.
        infer_source("""
            def f(x: ll.int32) -> ll.int32:
                if x:
                    return 1
                for i in range(2):
                    return 2
                while x:
                    return 3
                return 4
            """)

    def test_valid_nested_void_returns(self):
        infer_source("""
            def f(x: ll.int32) -> ll.void:
                if x:
                    return
                for i in range(2):
                    return
                return
            """)

    def test_implicit_numeric_return_coercion_preserved(self):
        # int->int width and int->float implicit conversions accepted.
        infer_source("""
            def f() -> ll.int64:
                if True:
                    return 1
                return 2
            """)
        infer_source("""
            def g() -> ll.float32:
                return 1
            """)

    def test_inner_nested_function_uses_own_signature(self):
        # The inner function's bad return is checked against int32, not the
        # outer function; previously this nested return escaped validation.
        expect_error(
            """
            def outer() -> ll.int32:
                def inner() -> ll.int32:
                    if True:
                        return 1.5
                    return 1
                return 0
            """, TypeMismatchError)

    def test_inner_signature_does_not_leak_to_outer(self):
        # After visiting the nested void function, the outer int32 return
        # must still validate against int32 (no stale single-slot context).
        infer_source("""
            def outer() -> ll.int32:
                def inner() -> ll.void:
                    return
                return 1
            """)

    def test_outer_int_return_after_inner_value_function(self):
        # Conversely, an outer bare return must not be validated against an
        # inner value-returning function's signature.
        expect_error(
            """
            def outer() -> ll.void:
                def inner() -> ll.int32:
                    return 1
                return 2
            """, TypeMismatchError)

    def test_deeply_nested_function_return(self):
        infer_source("""
            def outer() -> ll.void:
                def inner(x: ll.int32) -> ll.int32:
                    for i in range(x):
                        if i:
                            return i
                    return 0
                return
            """)


class TestOptionalProbeVersusInvalidInput:

    def test_required_false_probe_is_none_required_true_raises(self):
        scope = ScopeManager({})
        scope.enter_scope()
        node = ast.Name(id="missing", ctx=ast.Load())
        assert get_variable_type_from_scope(scope, "missing", node, {}, required=False) is None
        with pytest.raises(UndefinedVariableError):
            get_variable_type_from_scope(scope, "missing", node, {})

    def test_supported_exact_struct_name_probe_transforms(self):

        class DemoProbeStruct:
            pass

        ctx = {
            "ll": ll,
        }
        # Register exactly; the constructor itself is intentionally absent
        # from ctx, exercising the supported unresolved-name probe.
        from little_kernel.codegen.registries.special_struct_registry import _SPECIAL_STRUCT_CODEGEN_REGISTRY
        registry = _SPECIAL_STRUCT_CODEGEN_REGISTRY
        saved = registry.get("DemoProbeStruct")
        registry["DemoProbeStruct"] = {"class": DemoProbeStruct}
        try:
            tree = ast.parse("s = DemoProbeStruct()")
            transformed = SpecialStructMaterializeTransformer(ctx).visit(tree)
            assigns = [stmt for stmt in transformed.body if isinstance(stmt, ast.Assign)]
            assert len(assigns) == 1
            assert getattr(assigns[0], "_is_special_struct_decl", False)
            assert assigns[0]._special_struct_name == "DemoProbeStruct"
        finally:
            if saved is None:
                registry.pop("DemoProbeStruct", None)
            else:
                registry["DemoProbeStruct"] = saved

    def test_case_insensitive_name_is_not_hijacked(self):
        from little_kernel.codegen.registries.special_struct_registry import _SPECIAL_STRUCT_CODEGEN_REGISTRY

        class RegisteredStruct:
            pass

        registry = _SPECIAL_STRUCT_CODEGEN_REGISTRY
        saved = registry.get("RegisteredStruct")
        registry["RegisteredStruct"] = {"class": RegisteredStruct}
        try:
            # Lowercase spelling does not resolve in ctx and must not match
            # the registered struct by case-insensitive name guessing.
            tree = ast.parse("s = registeredstruct()")
            transformed = SpecialStructMaterializeTransformer({"ll": ll}).visit(tree)
            assign = transformed.body[0]
            assert not getattr(assign, "_is_special_struct_decl", False)
        finally:
            if saved is None:
                registry.pop("RegisteredStruct", None)
            else:
                registry["RegisteredStruct"] = saved

    def test_resolved_non_struct_callable_not_mistaken_for_struct(self):
        # A callable that resolves in ctx but originates from a different
        # module than the registered class must not be transformed just by
        # a matching name (the old case-insensitive / missing-module guess).
        def helper():
            return None

        class Unrelated:
            pass

        Unrelated.__module__ = "some.other.module"
        ctx = {"ll": ll, "helper": helper}
        from little_kernel.codegen.registries.special_struct_registry import _SPECIAL_STRUCT_CODEGEN_REGISTRY
        registry = _SPECIAL_STRUCT_CODEGEN_REGISTRY
        saved = registry.get("helper")
        registry["helper"] = {"class": Unrelated}
        try:
            tree = ast.parse("x = helper()")
            transformed = SpecialStructMaterializeTransformer(ctx).visit(tree)
            assign = transformed.body[0]
            assert not getattr(assign, "_is_special_struct_decl", False)
        finally:
            if saved is None:
                registry.pop("helper", None)
            else:
                registry["helper"] = saved


class TestSourceLocatedDiagnostics:

    def test_context_attaches_kernel_source_snippet(self):

        def kernel():
            if True:
                return missing  # noqa: F821
            return 0

        ctx = dict(CTX)
        ctx[__LITTLE_KERNEL_ENTRY__] = kernel
        # Parse the actual kernel object so node line numbers align with
        # the source located through the explicit ctx entry.
        import inspect
        tree = ast.parse(textwrap.dedent(inspect.getsource(kernel)))
        inferencer = TypeInferencer(ctx=ctx, scope_vars={})
        with pytest.raises(TypeInferenceError) as exc_info:
            inferencer.visit(tree)
        message = str(exc_info.value)
        assert "Location:" in message
        # The explicit ctx entry locates the real kernel source without
        # inspecting caller stack frames.
        assert "Code snippet:" in message
        assert "missing" in message

    def test_error_type_and_cause_preserved(self):
        err = expect_error(
            """
            def f() -> ll.int32:
                return ll.int32(missing)
            """, UndefinedVariableError)
        assert isinstance(err, UndefinedVariableError)
        assert err.raw_message.startswith("Undefined variable")

    def test_no_frame_based_context_guess_attribute(self):
        # Raised without ctx: formatting must still be deterministic and not
        # pull an unrelated source from the call stack.
        node = ast.Name(id="x", ctx=ast.Load())
        node.lineno = 7
        node.col_offset = 3
        err = UndefinedVariableError("Undefined variable: x", node=node)
        rendered = str(err)
        assert "Line 7" in rendered
        assert "column 3" in rendered
        assert "Code snippet:" not in rendered


class TestExplicitSourceOwnership:
    """RF16 completion: source snippets come only from explicit source_lines
    or the registered compilation entry; never caller frames or an arbitrary
    callable in ctx."""

    @pytest.fixture(autouse=True)
    def _assert_no_frame_scanning(self, monkeypatch):
        # The shared formatter must not touch sys._getframe at all. Replace
        # it with a hard failure for the duration of these tests so any
        # reintroduced caller-frame guessing is caught.
        import little_kernel.core.passes.utils.error_report as error_report

        assert not hasattr(error_report, "sys"), ("error_report must not import sys for caller-frame scanning")

        def _boom(*args, **kwargs):
            raise AssertionError("caller-frame guessing is forbidden")

        monkeypatch.setattr(error_report.inspect, "currentframe", _boom, raising=False)

    @staticmethod
    def _located_node():
        node = ast.Name(id="missing", ctx=ast.Load())
        node.lineno = 1
        node.col_offset = 0
        return node

    @pytest.mark.parametrize("ctx", [
        None,
        {},
        {"ll": ll},
    ])
    def test_absent_source_keeps_location_without_snippet(self, ctx):
        from little_kernel.core.passes.utils.error_report import format_error_message
        rendered = format_error_message("Undefined variable: missing", self._located_node(), ctx=ctx)
        assert "Undefined variable: missing" in rendered
        assert "Location: Line 1" in rendered
        assert "AST node:" in rendered
        assert "Code snippet:" not in rendered

    def test_empty_ctx_direct_exception_construction(self):
        # The exact acceptance reproduction: explicit empty ctx must not
        # re-enter frame/callable guessing.
        err = UndefinedVariableError("missing", node=self._located_node(), ctx={})
        rendered = str(err)
        assert "Code snippet:" not in rendered
        assert "Location: Line 1" in rendered
        assert isinstance(err, UndefinedVariableError)

    def test_unrelated_callable_ctx_is_not_source(self):
        # An arbitrary callable present in ctx (imports/helpers/builtins)
        # must never be picked as the snippet source.
        ctx = {"ll": ll, "an_unrelated_helper": self.test_unrelated_callable_ctx_is_not_source}
        err = UndefinedVariableError("missing", node=self._located_node(), ctx=ctx)
        rendered = str(err)
        assert "Code snippet:" not in rendered
        assert "an_unrelated_helper" not in rendered
        assert "test_unrelated_callable" not in rendered

    def test_get_source_from_ctx_only_uses_explicit_entry(self):
        from little_kernel.core.passes.utils.error_report import get_source_from_ctx
        # No entry, only unrelated callables: None, never a guessed source.
        ctx = {"ll": ll, "helper": self.test_get_source_from_ctx_only_uses_explicit_entry}
        assert get_source_from_ctx(ctx) is None
        assert get_source_from_ctx({}) is None
        assert get_source_from_ctx(None) is None

    def test_explicit_source_lines_win(self):
        from little_kernel.core.passes.utils.error_report import format_error_message
        source = "a = 1\nmissing\nc = 3\n".splitlines(keepends=True)
        node = self._located_node()
        # Explicit source takes precedence even over an unrelated ctx that
        # contains a callable.
        rendered = format_error_message("missing", node, source_lines=source, ctx={"ll": ll, "helper": print})
        assert "Code snippet:" in rendered
        assert "missing" in rendered
        assert "2 | missing" in rendered

    def test_explicit_compilation_entry_supplies_snippet(self):
        from little_kernel.core.passes.utils.error_report import format_error_message
        # A source-string entry is the canonical explicit source owner and
        # needs no host introspection.
        entry_source = "x = 1\ny = missing\n"
        ctx = {
            "ll": ll,
            __LITTLE_KERNEL_ENTRY__: entry_source,
            # An unrelated callable must be ignored in favour of the entry.
            "helper": print,
        }
        node = ast.Name(id="missing", ctx=ast.Load())
        node.lineno = 2
        node.col_offset = 4
        rendered = format_error_message("missing", node, ctx=ctx)
        assert "Code snippet:" in rendered
        assert "y = missing" in rendered

    def test_explicit_entry_used_through_inference_pipeline(self):
        # End-to-end: the inferencer rebuilds the diagnosis with its ctx at
        # the visit boundary, and the explicit entry snippet survives.
        entry_source = ("def f() -> ll.int32:\n"
                        "    if True:\n"
                        "        return missing\n"
                        "    return 0\n")
        ctx = {"ll": ll, __LITTLE_KERNEL_ENTRY__: entry_source}
        inferencer = TypeInferencer(ctx=ctx, scope_vars={})
        with pytest.raises(UndefinedVariableError) as exc_info:
            inferencer.visit(ast.parse(entry_source))
        rendered = str(exc_info.value)
        assert isinstance(exc_info.value, UndefinedVariableError)
        assert "Code snippet:" in rendered
        assert "return missing" in rendered
        assert "Location:" in rendered
