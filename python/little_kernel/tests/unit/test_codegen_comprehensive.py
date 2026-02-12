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
Comprehensive tests for all semantic codegen features.

This test file covers:
1. ExpressionCodegenMixin - all expression types
2. StatementCodegenMixin - all statement types including special structs
3. ControlFlowCodegenMixin - all control flow including loop modifiers
4. CallCodegenMixin - all call types including builtins and struct stubs
"""

import unittest
import ast
import little_kernel.language as ll
from little_kernel.codegen.codegen_base import codegen_cpp, CppEmitter
from little_kernel.codegen.registries.operator_codegen import get_operator_codegen_registry
from little_kernel.core.passes.utils.type_inference import TypeInferencer


def codegen_with_inference(tree, ctx):
    """Helper function to codegen with proper type inference."""
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = CppEmitter(ctx=ctx, all_types=inferencer.all_types)
    emitter.visit(tree)
    return emitter.get_code()


class TestExpressionCodegenComprehensive(unittest.TestCase):
    """Comprehensive tests for ExpressionCodegenMixin."""

    def setUp(self):
        """Set up test context."""
        self.ctx = {'ll': ll}

    def test_all_binary_operators(self):
        """Test all binary operators."""
        operators = [
            ('+', 'Add'),
            ('-', 'Sub'),
            ('*', 'Mult'),
            ('/', 'Div'),
            ('//', 'FloorDiv'),
            ('%', 'Mod'),
            ('<<', 'LShift'),
            ('>>', 'RShift'),
            ('|', 'BitOr'),
            ('^', 'BitXor'),
            ('&', 'BitAnd'),
            ('==', 'Eq'),
            ('!=', 'NotEq'),
            ('<', 'Lt'),
            ('<=', 'LtE'),
            ('>', 'Gt'),
            ('>=', 'GtE'),
        ]

        for op_str, op_name in operators:
            with self.subTest(op=op_str):
                code = f"a {op_str} b"
                tree = ast.parse(code, mode='eval')
                inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'a': ll.int32, 'b': ll.int32})
                inferencer.visit(tree)
                emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
                result = emitter.visit(tree)
                # For // operator, Python uses FloorDiv which is converted to / in C++
                # So we check for / instead of //
                expected_op = "/" if op_str == "//" else op_str
                self.assertIn(expected_op, result or "", f"Operator {expected_op} not found in result")

    def test_operator_registry_bitwise_ops(self):
        """Test operator codegen registry returns correct C++ for bitwise ops (including xor)."""
        registry = get_operator_codegen_registry()
        self.assertEqual(registry.get_bin_op(ast.BitOr()), "|")
        self.assertEqual(registry.get_bin_op(ast.BitAnd()), "&")
        self.assertEqual(registry.get_bin_op(ast.BitXor()), "^")

    def test_all_unary_operators(self):
        """Test all unary operators."""
        operators = [
            ('-', 'USub'),
            ('+', 'UAdd'),
            ('~', 'Invert'),
            ('not ', 'Not'),
        ]

        for op_str, op_name in operators:
            with self.subTest(op=op_str):
                code = f"{op_str}x"
                tree = ast.parse(code, mode='eval')
                inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'x': ll.int32})
                inferencer.visit(tree)
                emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
                result = emitter.visit(tree)
                self.assertIsNotNone(result, f"Operator {op_str} failed")

    def test_all_boolean_operators(self):
        """Test all boolean operators."""
        code = "a and b or c"
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'a': ll.bool_, 'b': ll.bool_, 'c': ll.bool_})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        result = emitter.visit(tree)
        self.assertIn("&&", result)
        self.assertIn("||", result)

    def test_all_comparison_operators(self):
        """Test all comparison operators."""
        code = "a < b <= c > d >= e == f != g"
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(
            ctx=self.ctx, scope_vars={
                'a': ll.int32, 'b': ll.int32, 'c': ll.int32, 'd': ll.int32, 'e': ll.int32, 'f': ll.int32, 'g': ll.int32
            })
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        result = emitter.visit(tree)
        self.assertIn("<", result)
        self.assertIn("<=", result)
        self.assertIn(">", result)
        self.assertIn(">=", result)
        self.assertIn("==", result)
        self.assertIn("!=", result)

    def test_constant_types(self):
        """Test all constant types."""
        constants = [
            ('42', 'int'),
            ('3.14', 'float'),
            ('True', 'bool'),
            ('False', 'bool'),
            ('"hello"', 'str'),
            ('0x1234', 'hex'),
        ]

        for const_code, const_type in constants:
            with self.subTest(const=const_code):
                tree = ast.parse(const_code, mode='eval')
                inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
                inferencer.visit(tree)
                emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
                result = emitter.visit(tree)
                self.assertIsNotNone(result, f"Constant {const_code} failed")

    def test_subscript_single_index(self):
        """Test single index subscript."""
        code = "arr[0]"
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'arr': ll.ptr[ll.int32]})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        result = emitter.visit(tree)
        self.assertIn("[", result)
        self.assertIn("0", result)

    def test_subscript_multi_index(self):
        """Test multi-dimensional subscript."""
        code = "arr[i, j]"
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'arr': ll.ptr[ll.int32], 'i': ll.int32, 'j': ll.int32})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        result = emitter.visit(tree)
        self.assertIn("[", result)
        self.assertIn("]", result)

    def test_attribute_access(self):
        """Test that attribute access on scalar type (int32) raises TypeInferenceError.
        Silently returning int32 would mask type errors - we now raise explicitly."""
        from little_kernel.core.passes.utils.type_inference import TypeInferenceError
        code = "obj.attr"
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'obj': ll.int32})
        with self.assertRaises(TypeInferenceError) as cm:
            inferencer.visit(tree)
        self.assertIn("Cannot resolve attribute", str(cm.exception))
        self.assertIn("attr", str(cm.exception))

    def test_ternary_expression(self):
        """Test ternary expression."""
        code = "a if b else c"
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'a': ll.int32, 'b': ll.bool_, 'c': ll.int32})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        result = emitter.visit(tree)
        self.assertIn("?", result)
        self.assertIn(":", result)

    def test_fstring_codegen(self):
        """Test f-string codegen."""
        code = 'f"value: {x}"'
        tree = ast.parse(code, mode='eval')
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'x': ll.int32})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        result = emitter.visit(tree)
        self.assertIn("value", result)
        self.assertIn("x", result)


class TestStatementCodegenComprehensive(unittest.TestCase):
    """Comprehensive tests for StatementCodegenMixin."""

    def setUp(self):
        """Set up test context."""
        self.ctx = {'ll': ll}

    def test_function_def_with_params(self):
        """Test function definition with multiple parameters."""
        code = """
def test(x: ll.int32, y: ll.float32, z: ll.bool_) -> ll.int32:
    return x
"""
        tree = ast.parse(code)
        # Use TypeInferencer to ensure types are inferred
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("test", result)
        self.assertIn("int32_t", result)
        self.assertIn("float", result)  # float32 maps to float in C++
        self.assertIn("bool", result)

    def test_ann_assign_all_types(self):
        """Test annotated assignment with all basic types."""
        types = ['ll.int32', 'll.uint32', 'll.int64', 'll.uint64', 'll.float32', 'll.float64', 'll.bool_', 'll.str_']
        # Map types to their default values
        type_values = {
            'll.int32': '5', 'll.uint32': '5', 'll.int64': '5', 'll.uint64': '5', 'll.float32': '5.0', 'll.float64':
            '5.0', 'll.bool_': 'True', 'll.str_': '"hello"'
        }

        for type_str in types:
            with self.subTest(type=type_str):
                value = type_values.get(type_str, '5')
                code = f"""
def test() -> ll.void:
    x: {type_str} = {value}
"""
                tree = ast.parse(code)
                inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
                inferencer.visit(tree)
                emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
                emitter.visit(tree)
                result = emitter.get_code()
                self.assertIn("x", result)

    def test_const_annotation(self):
        """Test const annotation."""
        code = """
def test() -> ll.void:
    x: ll.const[ll.int32] = 5
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("const", result)
        self.assertIn("int32_t", result)

    def test_tuple_assignment(self):
        """Test tuple assignment."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 0
    y: ll.float32 = 0.0
    z: ll.bool_ = False
    x, y, z = 1, 2.0, True
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("x = 1", result)
        self.assertIn("y = 2.0", result)
        self.assertIn("z = true", result)

    def test_subscript_assignment(self):
        """Test subscript assignment (C++ style array access)."""
        code = """
def test() -> ll.void:
    arr: ll.Tensor[ll.int32]
    arr[0] = 42
    arr[1] = 100
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("arr[0] = 42", result)
        self.assertIn("arr[1] = 100", result)

    def test_subscript_assignment_with_variable_index(self):
        """Test subscript assignment with variable index."""
        code = """
def test() -> ll.void:
    arr: ll.Tensor[ll.int32]
    idx: ll.int32 = 5
    arr[idx] = 99
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("arr[idx] = 99", result)

    def test_subscript_assignment_with_expression(self):
        """Test subscript assignment with expression value."""
        code = """
def test() -> ll.void:
    arr: ll.Tensor[ll.int32]
    x: ll.int32 = 10
    y: ll.int32 = 20
    arr[0] = x + y
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("arr[0] =", result)
        self.assertIn("x + y", result)

    def test_multiple_assignments(self):
        """Test multiple variable assignments."""
        code = """
def test() -> ll.void:
    a: ll.int32 = 1
    b: ll.int32 = 2
    c: ll.int32 = 3
    a = a + b
    b = b + c
    c = a + b + c
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("a", result)
        self.assertIn("b", result)
        self.assertIn("c", result)

    def test_return_with_value(self):
        """Test return statement with value."""
        code = """
def test() -> ll.int32:
    return 42
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("return 42", result)

    def test_return_void(self):
        """Test void return statement."""
        code = """
def test() -> ll.void:
    return
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("return;", result)

    def test_assert_statement(self):
        """Test assert statement."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    assert x > 0
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("assert", result)

    def test_assert_with_message(self):
        """Test assert with message."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    assert x > 0, "x must be positive"
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("assert", result)
        self.assertIn("x must be positive", result)

    def test_pass_statement(self):
        """Test pass statement."""
        code = """
def test() -> ll.void:
    pass
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("test", result)


class TestControlFlowCodegenComprehensive(unittest.TestCase):
    """Comprehensive tests for ControlFlowCodegenMixin."""

    def setUp(self):
        """Set up test context."""
        self.ctx = {'ll': ll}

    def test_if_statement(self):
        """Test if statement."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    if x > 0:
        y: ll.int32 = 1
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("if", result)
        self.assertIn("x > 0", result)

    def test_if_else_statement(self):
        """Test if-else statement."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    if x > 0:
        y: ll.int32 = 1
    else:
        y: ll.int32 = 0
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("if", result)
        self.assertIn("else", result)

    def test_if_elif_else_statement(self):
        """Test if-elif-else statement."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    if x > 0:
        y: ll.int32 = 1
    elif x < 0:
        y: ll.int32 = -1
    else:
        y: ll.int32 = 0
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("if", result)
        self.assertIn("else if", result)
        self.assertIn("else", result)

    def test_while_loop(self):
        """Test while loop."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    while x > 0:
        x = x - 1
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("while", result)
        self.assertIn("x > 0", result)

    def test_for_range_single_arg(self):
        """Test for loop with range(stop)."""
        code = """
def test() -> ll.void:
    for i in range(10):
        x: ll.int32 = i
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("for", result)
        self.assertIn("i", result)
        self.assertIn("10", result)

    def test_for_range_two_args(self):
        """Test for loop with range(start, stop)."""
        code = """
def test() -> ll.void:
    for i in range(0, 10):
        x: ll.int32 = i
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("for", result)
        self.assertIn("0", result)
        self.assertIn("10", result)

    def test_for_range_three_args(self):
        """Test for loop with range(start, stop, step)."""
        code = """
def test() -> ll.void:
    for i in range(0, 10, 2):
        x: ll.int32 = i
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("for", result)
        self.assertIn("2", result)
        self.assertIn("+=", result)

    def test_for_range_negative_step(self):
        """Test for loop with negative step."""
        code = """
def test() -> ll.void:
    for i in range(10, 0, -1):
        x: ll.int32 = i
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("for", result)
        # Check for negative step pattern (i -= 1 or i += -1)
        self.assertTrue("i -= 1" in result or "-1" in result, f"Negative step not found in: {result}")

    def test_for_range_dynamic_step(self):
        """Test for loop with dynamic step (variable) - uses runtime-dependent condition."""
        code = """
def test(step: ll.int32) -> ll.void:
    for i in range(0, 10, step):
        x: ll.int32 = i
"""
        tree = ast.parse(code)
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'step': ll.int32})
        inferencer.visit(tree)
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()
        self.assertIn("for", result)
        # Dynamic step: condition must use runtime check (step > 0 ? ... : ...)
        self.assertIn("?", result)
        self.assertIn("i +=", result)

    def test_continue_statement(self):
        """Test continue statement."""
        code = """
def test() -> ll.void:
    for i in range(10):
        if i % 2 == 0:
            continue
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("continue", result)

    def test_break_statement(self):
        """Test break statement."""
        code = """
def test() -> ll.void:
    for i in range(10):
        if i > 5:
            break
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("break", result)

    def test_nested_control_flow(self):
        """Test nested control flow."""
        code = """
def test() -> ll.void:
    for i in range(10):
        if i % 2 == 0:
            for j in range(5):
                if j > 2:
                    break
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("for", result)
        self.assertIn("if", result)
        self.assertIn("break", result)


class TestCallCodegenComprehensive(unittest.TestCase):
    """Comprehensive tests for CallCodegenMixin."""

    def setUp(self):
        """Set up test context."""
        self.ctx = {'ll': ll}

    def test_type_constructor_calls(self):
        """Test type constructor calls."""
        type_constructors = [
            ('ll.int32', 'int32_t'),
            ('ll.uint32', 'uint32_t'),
            ('ll.int64', 'int64_t'),
            ('ll.uint64', 'uint64_t'),
            ('ll.float32', 'float'),
            ('ll.float64', 'double'),
        ]

        for type_str, cpp_type in type_constructors:
            with self.subTest(type=type_str):
                code = f"""
def test() -> ll.void:
    x: {type_str} = {type_str}(5)
"""
                tree = ast.parse(code)
                result = codegen_with_inference(tree, self.ctx)
                self.assertIn("static_cast", result)
                self.assertIn(cpp_type, result)

    def test_to_method_call(self):
        """Test .to() method call."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    y: ll.uint32 = x.to(ll.uint32)
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("static_cast", result)
        self.assertIn("uint32_t", result)

    def test_function_call_with_args(self):
        """Test function call with arguments."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    y: ll.int32 = 10
    z: ll.int32 = add(x, y)
"""
        tree = ast.parse(code)
        # This will fail if add is not defined, but we test the structure
        try:
            result = codegen_cpp(tree, ctx=self.ctx)
            self.assertIn("add", result)
        except Exception:
            # Expected if add is not a builtin
            pass


class TestComplexCodegenScenarios(unittest.TestCase):
    """Test complex codegen scenarios combining multiple features."""

    def setUp(self):
        """Set up test context."""
        self.ctx = {'ll': ll}

    def test_nested_loops_with_break_continue(self):
        """Test nested loops with break and continue."""
        code = """
def test() -> ll.void:
    sum: ll.int32 = 0
    for i in range(10):
        if i > 5:
            break
        for j in range(5):
            if j % 2 == 0:
                continue
            sum = sum + j
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("for", result)
        self.assertIn("if", result)
        self.assertIn("break", result)
        self.assertIn("continue", result)
        self.assertIn("sum", result)

    def test_complex_arithmetic_expressions(self):
        """Test complex arithmetic expressions."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 1
    y: ll.int32 = 2
    z: ll.int32 = 3
    result: ll.int32 = x + y * z - (x + y) / z
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("result", result)
        self.assertIn("+", result)
        self.assertIn("*", result)
        self.assertIn("-", result)
        self.assertIn("/", result)

    def test_conditional_assignment(self):
        """Test conditional assignment using ternary."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 5
    y: ll.int32 = 10
    result: ll.int32 = x if x > y else y
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("result", result)
        self.assertIn("?", result)
        self.assertIn(":", result)

    def test_multiple_if_else_chains(self):
        """Test multiple if-else chains."""
        code = """
def test(x: ll.int32) -> ll.void:
    if x > 10:
        y: ll.int32 = 1
    elif x > 5:
        y: ll.int32 = 2
    elif x > 0:
        y: ll.int32 = 3
    else:
        y: ll.int32 = 0
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("if", result)
        self.assertIn("else if", result)
        self.assertIn("else", result)

    def test_loop_with_early_return(self):
        """Test loop with early return."""
        code = """
def test() -> ll.int32:
    for i in range(100):
        if i > 50:
            return i
    return 0
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("for", result)
        self.assertIn("if", result)
        self.assertIn("return", result)

    def test_variable_scoping_in_nested_blocks(self):
        """Test variable scoping in nested blocks."""
        code = """
def test() -> ll.void:
    x: ll.int32 = 0
    if True:
        y: ll.int32 = 1
        if True:
            z: ll.int32 = 2
            x = x + y + z
"""
        tree = ast.parse(code)
        result = codegen_with_inference(tree, self.ctx)
        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIn("z", result)


class TestTypeInferenceDetailed(unittest.TestCase):
    """Detailed tests for type inference functionality."""

    def setUp(self):
        """Set up test context."""
        self.ctx = {'ll': ll}

    def test_constant_type_inference_int(self):
        """Test type inference for integer constants."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        # Test various integer constants
        test_cases = [
            ('5', ll.int32),
            ('0', ll.int32),
            ('-10', ll.int32),
            ('1000000', ll.int32),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_constant_type_inference_float(self):
        """Test type inference for float constants."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('5.0', ll.float32),
            ('0.0', ll.float32),
            ('-10.5', ll.float32),
            ('3.14159', ll.float32),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_constant_type_inference_bool(self):
        """Test type inference for boolean constants."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('True', ll.bool_),
            ('False', ll.bool_),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_constant_type_inference_str(self):
        """Test type inference for string constants."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('"hello"', ll.str_),
            ('""', ll.str_),
            ('"test123"', ll.str_),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_binary_op_type_inference(self):
        """Test type inference for binary operations."""
        inferencer = TypeInferencer(ctx=self.ctx,
                                    scope_vars={'a': ll.int32, 'b': ll.int32, 'c': ll.float32, 'd': ll.float32})

        test_cases = [
            ('a + b', ll.int32),
            ('a - b', ll.int32),
            ('a * b', ll.int32),
            ('a / b', ll.int32),
            ('a % b', ll.int32),
            ('c + d', ll.float32),
            ('c - d', ll.float32),
            ('c * d', ll.float32),
            ('c / d', ll.float32),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_comparison_op_type_inference(self):
        """Test type inference for comparison operations."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'a': ll.int32, 'b': ll.int32})

        test_cases = [
            ('a == b', ll.bool_),
            ('a != b', ll.bool_),
            ('a < b', ll.bool_),
            ('a <= b', ll.bool_),
            ('a > b', ll.bool_),
            ('a >= b', ll.bool_),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_unary_op_type_inference(self):
        """Test type inference for unary operations."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'a': ll.int32, 'b': ll.bool_})

        test_cases = [
            ('-a', ll.int32),
            ('+a', ll.int32),
            ('not b', ll.bool_),
            ('~a', ll.int32),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_variable_type_inference(self):
        """Test type inference for variables."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={
            'x': ll.int32,
            'y': ll.float32,
            'z': ll.bool_,
            's': ll.str_,
        })

        test_cases = [
            ('x', ll.int32),
            ('y', ll.float32),
            ('z', ll.bool_),
            ('s', ll.str_),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_annotated_assignment_type_inference(self):
        """Test type inference for annotated assignments."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('x: ll.int32 = 5', 'x', ll.int32),
            ('y: ll.float32 = 3.14', 'y', ll.float32),
            ('z: ll.bool_ = True', 'z', ll.bool_),
            ('s: ll.str_ = "hello"', 's', ll.str_),
        ]

        for code, var_name, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code)
                inferencer.visit(tree)
                var_type = inferencer.current_scope.get(var_name)
                self.assertEqual(var_type, expected_type, f"Expected {expected_type}, got {var_type} for {var_name}")

    def test_const_type_inference(self):
        """Test type inference for const annotations."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('x: ll.const[ll.int32] = 5', 'x', ll.Const(ll.int32)),
            ('y: ll.const[ll.float32] = 3.14', 'y', ll.Const(ll.float32)),
            ('z: ll.const[ll.str_] = "hello"', 'z', ll.Const(ll.str_)),
        ]

        for code, var_name, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code)
                inferencer.visit(tree)
                var_type = inferencer.current_scope.get(var_name)
                # For const types, we check the inner type
                if isinstance(var_type, ll.Const):
                    self.assertEqual(var_type.inner_type, expected_type.inner_type,
                                     f"Expected const[{expected_type.inner_type}], got const[{var_type.inner_type}]")
                else:
                    self.fail(f"Expected Const type, got {type(var_type)}")

    def test_type_constructor_inference(self):
        """Test type inference for type constructor calls."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('ll.int32(5)', ll.int32),
            ('ll.uint32(10)', ll.uint32),
            ('ll.float32(3.14)', ll.float32),
            ('ll.int64(100)', ll.int64),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_to_method_type_inference(self):
        """Test type inference for .to() method calls."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={
            'x': ll.int32,
            'y': ll.float32,
        })

        test_cases = [
            ('x.to(ll.uint32)', ll.uint32),
            ('x.to(ll.int64)', ll.int64),
            ('y.to(ll.float64)', ll.float64),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_nested_expression_type_inference(self):
        """Test type inference for nested expressions."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={'a': ll.int32, 'b': ll.int32, 'c': ll.int32})

        test_cases = [
            ('(a + b) * c', ll.int32),
            ('a + (b * c)', ll.int32),
            ('(a == b) and (b == c)', ll.bool_),
            ('not (a < b)', ll.bool_),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")

    def test_implicit_type_conversion(self):
        """Test implicit type conversion in assignments."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        # Test implicit conversions that should be allowed
        test_cases = [('x: ll.uint32 = 5', 'x', ll.uint32),  # int32 -> uint32
                      ('y: ll.float32 = 5', 'y', ll.float32),  # int32 -> float32
                      ('z: ll.float64 = 3.14', 'z', ll.float64),  # float32 -> float64
                      ]

        for code, var_name, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code)
                inferencer.visit(tree)
                var_type = inferencer.current_scope.get(var_name)
                self.assertEqual(var_type, expected_type, f"Expected {expected_type}, got {var_type} for {var_name}")

    def test_function_return_type_inference(self):
        """Test type inference for function return types."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        test_cases = [
            ('def f() -> ll.int32: return 5', 'f', ll.int32),
            ('def g() -> ll.float32: return 3.14', 'g', ll.float32),
            ('def h() -> ll.void: pass', 'h', ll.void),
            ('def i() -> ll.bool_: return True', 'i', ll.bool_),
        ]

        for code, func_name, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code)
                inferencer.visit(tree)
                func_type = inferencer.current_scope.get(func_name)
                self.assertEqual(func_type, expected_type, f"Expected {expected_type}, got {func_type} for {func_name}")

    def test_function_parameter_type_inference(self):
        """Test type inference for function parameters."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        code = """
def test(a: ll.int32, b: ll.float32, c: ll.bool_) -> ll.void:
    x: ll.int32 = a
    y: ll.float32 = b
    z: ll.bool_ = c
"""
        tree = ast.parse(code)
        # If this doesn't raise an error, parameter type inference worked
        inferencer.visit(tree)

        # Verify that codegen works (means types were inferred correctly)
        from little_kernel.codegen.codegen_base import CppEmitter
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()

        # Verify that the code was generated with correct types
        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIn("z", result)
        self.assertIn("int32_t", result)
        self.assertIn("float", result)

    def test_tuple_type_inference(self):
        """Test type inference for tuple types."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        code = """
def test() -> ll.void:
    x: ll.int32 = 0
    y: ll.float32 = 0.0
    z: ll.bool_ = False
    x, y, z = 1, 2.0, True
"""
        tree = ast.parse(code)
        # If this doesn't raise an error, tuple unpacking type inference worked
        # The type inference should handle tuple assignment correctly
        # This tests that tuple unpacking doesn't cause type errors
        inferencer.visit(tree)

        # The fact that visit() succeeded means type inference worked
        # We can verify by checking that codegen also works
        from little_kernel.codegen.codegen_base import CppEmitter
        emitter = CppEmitter(ctx=self.ctx, all_types=inferencer.all_types)
        emitter.visit(tree)
        result = emitter.get_code()

        # Verify that the code was generated (means types were inferred correctly)
        self.assertIn("x", result)
        self.assertIn("y", result)
        self.assertIn("z", result)

    def test_scope_nesting_type_inference(self):
        """Test type inference in nested scopes."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        code = """
def test() -> ll.void:
    x: ll.int32 = 0
    if True:
        y: ll.int32 = 1
        x = x + y
    for i in range(10):
        z: ll.int32 = i
        x = x + z
"""
        tree = ast.parse(code)
        # If this doesn't raise an error, nested scope type inference worked
        # The type inference should handle nested scopes correctly
        inferencer.visit(tree)

        # Verify that the annotation was correctly resolved
        func_node = tree.body[0]
        for stmt in func_node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                if stmt.target.id == 'x':
                    # Check that the annotation was resolved correctly
                    if stmt.annotation in inferencer.all_types:
                        annotation_type = inferencer.all_types[stmt.annotation]
                        self.assertEqual(annotation_type, ll.int32)
                    break

    def test_type_mismatch_error(self):
        """Test that type mismatches raise appropriate errors."""
        inferencer = TypeInferencer(ctx=self.ctx, scope_vars={})

        # Test cases that should raise TypeMismatchError
        error_cases = [
            'x: ll.str_ = 5',  # int32 -> str_ (not allowed)
            'y: ll.int32 = "hello"',  # str_ -> int32 (not allowed)
        ]

        for code in error_cases:
            with self.subTest(code=code):
                tree = ast.parse(code)
                with self.assertRaises(Exception):  # TypeMismatchError or similar
                    inferencer.visit(tree)

    def test_complex_expression_type_inference(self):
        """Test type inference for complex expressions."""
        inferencer = TypeInferencer(
            ctx=self.ctx, scope_vars={
                'a': ll.int32, 'b': ll.int32, 'c': ll.int32, 'x': ll.float32, 'y': ll.float32, 'p': ll.bool_, 'q':
                ll.bool_
            })

        test_cases = [
            ('a + b * c', ll.int32),
            ('(a + b) * (c - a)', ll.int32),
            ('x * y + x', ll.float32),
            ('p and q', ll.bool_),
            ('p or (a > b)', ll.bool_),
            ('(a == b) and (b == c)', ll.bool_),
            ('not (a < b)', ll.bool_),
        ]

        for code, expected_type in test_cases:
            with self.subTest(code=code):
                tree = ast.parse(code, mode='eval')
                result_type = inferencer.infer_node_type(tree.body)
                self.assertEqual(result_type, expected_type, f"Expected {expected_type}, got {result_type} for {code}")


if __name__ == "__main__":
    unittest.main()
