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
Generic Python class to C++ struct code generator.

This module provides a generic mechanism to convert Python class AST
to C++ struct definition. The actual implementation is split across
multiple modules in the special_struct package.
"""

# Re-export for backward compatibility
from little_kernel.codegen.special_struct.struct_converter import PythonClassToCppStructConverter
from little_kernel.codegen.special_struct.generic_codegen import (generic_special_struct_declaration_codegen,
                                                                  generic_special_struct_method_codegen,
                                                                  generic_special_struct_tuple_unpack_codegen,
                                                                  generic_special_struct_definition_codegen)
from little_kernel.codegen.special_struct.utils import (generate_cpp_struct_from_python_class, _resolve_template_param)

__all__ = [
    'PythonClassToCppStructConverter',
    'generic_special_struct_declaration_codegen',
    'generic_special_struct_method_codegen',
    'generic_special_struct_tuple_unpack_codegen',
    'generic_special_struct_definition_codegen',
    'generate_cpp_struct_from_python_class',
    '_resolve_template_param',
]
