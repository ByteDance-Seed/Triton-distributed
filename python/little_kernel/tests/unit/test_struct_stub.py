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
Unit tests for struct_stub decorator functionality.

Tests cover:
- Struct stub registration
- Method configuration with @struct_method
- Type annotations
- @staticmethod and @classmethod support
- Tuple return values
- Custom codegen functions
"""

import unittest
import ast
import little_kernel.language as ll
from little_kernel.language.intrin.struct_stub import (struct_stub, struct_method, get_struct_stub_info,
                                                       get_all_struct_stubs, is_struct_stub)
from little_kernel.codegen.codegen_cuda import codegen_cuda
from little_kernel.core.passes import PASSES
from little_kernel.core.compile import ll_kernel


class TestStructStubRegistration(unittest.TestCase):
    """Test struct stub registration and basic functionality."""

    def test_simple_struct_stub(self):
        """Test basic struct stub registration."""

        @struct_stub(cpp_struct_name="MyStruct", includes=['"my_struct.h"'])
        class MyStruct:

            def __init__(self, arg1, arg2):
                pass

        # Verify registration
        self.assertTrue(is_struct_stub(MyStruct))
        stub_info = get_struct_stub_info("MyStruct")
        self.assertIsNotNone(stub_info)
        self.assertEqual(stub_info.cpp_struct_name, "MyStruct")
        self.assertEqual(stub_info.includes, ['"my_struct.h"'])
        self.assertIsNone(stub_info.namespace)
        self.assertIsNone(stub_info.template_params)

    def test_struct_stub_with_namespace(self):
        """Test struct stub with namespace."""

        @struct_stub(cpp_struct_name="Tensor", includes=['<cute/tensor.hpp>'], namespace="cute")
        class CuteTensor:

            def __init__(self, shape, dtype):
                pass

        stub_info = get_struct_stub_info("CuteTensor")
        self.assertIsNotNone(stub_info)
        self.assertEqual(stub_info.cpp_struct_name, "Tensor")
        self.assertEqual(stub_info.namespace, "cute")
        self.assertIn('<cute/tensor.hpp>', stub_info.includes)

    def test_struct_stub_with_template_params(self):
        """Test struct stub with template parameters."""

        @struct_stub(cpp_struct_name="Array", includes=['"array.h"'], template_params=["typename T", "int N"])
        class Array:

            def __init__(self, data):
                pass

        stub_info = get_struct_stub_info("Array")
        self.assertIsNotNone(stub_info)
        self.assertEqual(stub_info.template_params, ["typename T", "int N"])


class TestStructMethodConfiguration(unittest.TestCase):
    """Test @struct_method decorator and method configuration."""

    def test_method_with_type_annotations(self):
        """Test method with type annotations."""

        @struct_stub(cpp_struct_name="MyStruct", includes=['"my_struct.h"'])
        class MyStruct:

            @struct_method(return_type=ll.bool_, param_types=[ll.int32, ll.int32])
            def method(self, arg1, arg2):
                pass

        stub_info = get_struct_stub_info("MyStruct")
        self.assertIn("method", stub_info.methods)
        method_info = stub_info.methods["method"]
        self.assertEqual(method_info.return_type, ll.bool_)
        self.assertEqual(len(method_info.param_types), 2)
        self.assertFalse(method_info.is_static)
        self.assertFalse(method_info.is_classmethod)

    def test_method_with_different_cpp_name(self):
        """Test method with different C++ name."""

        @struct_stub(cpp_struct_name="MyStruct", includes=['"my_struct.h"'])
        class MyStruct:

            @struct_method(cpp_method_name="cpp_method_name", return_type=ll.int32)
            def python_method(self, arg1):
                pass

        stub_info = get_struct_stub_info("MyStruct")
        method_info = stub_info.methods["python_method"]
        self.assertEqual(method_info.cpp_method_name, "cpp_method_name")
        self.assertEqual(method_info.python_method_name, "python_method")

    def test_static_method(self):
        """Test @staticmethod support."""

        @struct_stub(cpp_struct_name="MyStruct", includes=['"my_struct.h"'])
        class MyStruct:

            @staticmethod
            @struct_method(is_static=True, return_type=ll.int32, param_types=[ll.int32, ll.int32])
            def static_method(arg1, arg2) -> ll.void:
                pass

        stub_info = get_struct_stub_info("MyStruct")
        method_info = stub_info.methods["static_method"]
        self.assertTrue(method_info.is_static)
        self.assertFalse(method_info.is_classmethod)

    def test_classmethod(self):
        """Test @classmethod support."""

        @struct_stub(cpp_struct_name="MyStruct", includes=['"my_struct.h"'])
        class MyStruct:

            @classmethod
            @struct_method(is_classmethod=True, return_type=ll.bool_)
            def class_method(cls, arg1):
                pass

        stub_info = get_struct_stub_info("MyStruct")
        method_info = stub_info.methods["class_method"]
        self.assertTrue(method_info.is_classmethod)
        self.assertFalse(method_info.is_static)

    def test_tuple_return_method(self):
        """Test method returning tuple."""

        @struct_stub(cpp_struct_name="MyStruct", includes=['"my_struct.h"'])
        class MyStruct:

            @struct_method(is_tuple_return=True, tuple_return_types=[ll.bool_, ll.int32, ll.int32], param_types=[])
            def get_next(self):
                pass

        stub_info = get_struct_stub_info("MyStruct")
        method_info = stub_info.methods["get_next"]
        self.assertTrue(method_info.is_tuple_return)
        self.assertEqual(len(method_info.tuple_return_types), 3)
        self.assertEqual(method_info.tuple_return_types[0], ll.bool_)
        self.assertEqual(method_info.tuple_return_types[1], ll.int32)
        self.assertEqual(method_info.tuple_return_types[2], ll.int32)

    def test_custom_codegen(self):
        """Test method with custom codegen function."""

        def custom_codegen(node: ast.Call, emitter) -> str:
            args_cpp = [emitter.visit(arg) for arg in node.args]
            return f"custom_cpp_function({', '.join(args_cpp)})"

        @struct_stub(cpp_struct_name="SpecialStruct", includes=['"special.h"'])
        class SpecialStruct:

            @struct_method(custom_codegen=custom_codegen, return_type=ll.void)
            def special_method(self, arg1):
                pass

        stub_info = get_struct_stub_info("SpecialStruct")
        method_info = stub_info.methods["special_method"]
        self.assertIsNotNone(method_info.custom_codegen)
        self.assertEqual(method_info.custom_codegen, custom_codegen)


class TestStructStubRegistry(unittest.TestCase):
    """Test struct stub registry functionality."""

    def test_get_all_struct_stubs(self):
        """Test retrieving all registered struct stubs."""

        @struct_stub(cpp_struct_name="TestStruct1", includes=['"test1.h"'])
        class TestStruct1:
            pass

        @struct_stub(cpp_struct_name="TestStruct2", includes=['"test2.h"'])
        class TestStruct2:
            pass

        all_stubs = get_all_struct_stubs()
        self.assertIn("TestStruct1", all_stubs)
        self.assertIn("TestStruct2", all_stubs)
        self.assertEqual(all_stubs["TestStruct1"].cpp_struct_name, "TestStruct1")
        self.assertEqual(all_stubs["TestStruct2"].cpp_struct_name, "TestStruct2")

    def test_get_struct_stub_info(self):
        """Test retrieving struct stub info by name."""

        @struct_stub(cpp_struct_name="TestStruct", includes=['"test.h"'], namespace="test")
        class TestStruct:
            pass

        stub_info = get_struct_stub_info("TestStruct")
        self.assertIsNotNone(stub_info)
        self.assertEqual(stub_info.cpp_struct_name, "TestStruct")
        self.assertEqual(stub_info.namespace, "test")

        # Test non-existent struct
        non_existent = get_struct_stub_info("NonExistent")
        self.assertIsNone(non_existent)


class TestStructStubCodegen(unittest.TestCase):
    """Test actual code generation for struct stubs."""

    def setUp(self):
        """Set up common imports for codegen tests."""

        self.codegen_cuda = codegen_cuda
        self.PASSES = PASSES

    def test_struct_constructor_codegen(self):
        """Test code generation for struct constructor."""
        from little_kernel.language.intrin.struct_stub import struct_stub

        @struct_stub(cpp_struct_name="TestStruct", includes=['"test_struct.h"'])
        class TestStruct:

            def __init__(self, arg1, arg2):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            x = 10
            y = 20
            my_struct = TestStruct(x, y)  # noqa: F841

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that include is generated
        self.assertIn('#include "test_struct.h"', code)

        # Check that constructor call is generated
        self.assertIn('TestStruct my_struct', code)
        # Check that arguments are passed (either as variables or constants)
        self.assertTrue('my_struct(x, y)' in code or 'my_struct(10, 20)' in code or 'my_struct(x,y)' in code
                        or 'my_struct(10,20)' in code)

    def test_struct_method_call_codegen(self):
        """Test code generation for struct method call."""
        from little_kernel.language.intrin.struct_stub import struct_stub, struct_method

        @struct_stub(cpp_struct_name="TestStruct", includes=['"test_struct.h"'])
        class TestStruct:

            def __init__(self, arg1):
                pass

            @struct_method(return_type=ll.int32, param_types=[ll.int32])
            def method(self, arg):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            my_struct = TestStruct(10)
            result = my_struct.method(5)  # noqa: F841

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that method call is generated
        self.assertIn('my_struct.method', code)
        self.assertIn('result', code)

    def test_static_method_codegen(self):
        """Test code generation for static method."""
        from little_kernel.language.intrin.struct_stub import struct_stub, struct_method

        @struct_stub(cpp_struct_name="TestStruct", includes=['"test_struct.h"'])
        class TestStruct:

            @staticmethod
            @struct_method(is_static=True, return_type=ll.int32, param_types=[ll.int32, ll.int32])
            def static_method(arg1, arg2):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            result = TestStruct.static_method(10, 20)  # noqa: F841

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that static method call is generated
        self.assertIn('TestStruct::static_method', code)

    def test_tuple_return_codegen(self):
        """Test code generation for tuple return method."""
        from little_kernel.language.intrin.struct_stub import struct_stub, struct_method

        @struct_stub(cpp_struct_name="TestStruct", includes=['"test_struct.h"'])
        class TestStruct:

            def __init__(self):
                pass

            @struct_method(is_tuple_return=True, tuple_return_types=[ll.bool_, ll.int32, ll.int32], param_types=[])
            def get_next(self):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            my_struct = TestStruct()
            has_next, idx1, idx2 = my_struct.get_next()

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that tuple unpacking is generated correctly
        self.assertIn('bool has_next', code)
        self.assertIn('int32_t idx1', code)
        self.assertIn('int32_t idx2', code)
        self.assertIn('my_struct.get_next(has_next, idx1, idx2)', code)

    def test_custom_codegen(self):
        """Test code generation with custom codegen function."""
        from little_kernel.language.intrin.struct_stub import struct_stub, struct_method

        def custom_codegen(node: ast.Call, emitter) -> str:
            args_cpp = [emitter.visit(arg) for arg in node.args]
            return f"custom_cpp_function({', '.join(args_cpp)})"

        @struct_stub(cpp_struct_name="SpecialStruct", includes=['"special.h"'])
        class SpecialStruct:

            def __init__(self):
                pass

            @struct_method(custom_codegen=custom_codegen, return_type=ll.void)
            def special_method(self, arg1):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            my_struct = SpecialStruct()
            my_struct.special_method(42)

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that custom codegen is used
        self.assertIn('custom_cpp_function', code)

    def test_namespace_codegen(self):
        """Test code generation for struct with namespace."""
        from little_kernel.language.intrin.struct_stub import struct_stub

        @struct_stub(cpp_struct_name="Tensor", includes=['<cute/tensor.hpp>'], namespace="cute")
        class CuteTensor:

            def __init__(self, shape, dtype):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            tensor = CuteTensor(32, 64)  # noqa: F841

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that namespace is used
        self.assertIn('cute::Tensor', code)
        self.assertIn('#include <cute/tensor.hpp>', code)

    def test_method_with_different_cpp_name_codegen(self):
        """Test code generation for method with different C++ name."""
        from little_kernel.language.intrin.struct_stub import struct_stub, struct_method

        @struct_stub(cpp_struct_name="TestStruct", includes=['"test_struct.h"'])
        class TestStruct:

            def __init__(self):
                pass

            @struct_method(cpp_method_name="cpp_method", return_type=ll.int32)
            def python_method(self):
                pass

        @ll_kernel(backend="cuda")
        def test_kernel() -> ll.void:
            my_struct = TestStruct()
            result = my_struct.python_method()  # noqa: F841

        code = test_kernel.compile(self.PASSES["cuda"], self.codegen_cuda)

        # Check that C++ method name is used
        self.assertIn('my_struct.cpp_method', code)
        self.assertNotIn('my_struct.python_method', code)


if __name__ == "__main__":
    unittest.main()
