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

################################################################################
#
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
# included in all copies or substantial portions of the core portions of the Software.
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

# Codegen registry module exports
from .type_mapping_registry import get_type_mapping_registry, TypeMappingRegistry
from .operator_codegen import get_operator_codegen_registry, OperatorCodegenRegistry
from .loop_modifier_codegen import (get_loop_modifier_codegen, register_loop_modifier_codegen, LoopModifierCodegen)
from .type_converter import TypeConverter
from .enum_registry import get_enum, register_enum, generate_enum_cpp_definition
from .special_struct_registry import (is_special_struct, get_special_struct_name, get_special_struct_codegen,
                                      get_all_registered_special_structs, register_special_struct,
                                      register_special_struct_codegen)

__all__ = [
    'get_type_mapping_registry',
    'TypeMappingRegistry',
    'get_operator_codegen_registry',
    'OperatorCodegenRegistry',
    'get_loop_modifier_codegen',
    'register_loop_modifier_codegen',
    'LoopModifierCodegen',
    'TypeConverter',
    'get_enum',
    'register_enum',
    'generate_enum_cpp_definition',
    'is_special_struct',
    'get_special_struct_name',
    'get_special_struct_codegen',
    'get_all_registered_special_structs',
    'register_special_struct',
    'register_special_struct_codegen',
]
