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

from typing import Dict, Callable, Optional
import little_kernel.language as ll


class TypeMappingRegistry:
    """Registry for mapping LLType to C++ type strings."""

    def __init__(self):
        # TODO: 4-bit (int4/uint4) - CUDA has native support, to be added in future
        self._int_type_mappings: Dict[int, str] = {
            8: "int8_t", 16: "int16_t", 32: "int32_t", 64: "int64_t", 128: "__int128"
        }
        self._uint_type_mappings: Dict[int, str] = {
            8: "uint8_t", 16: "uint16_t", 32: "uint32_t", 64: "uint64_t", 128: "unsigned __int128"
        }
        self._float_type_mappings: Dict[str, str] = {
            "fp4_e2m1": "__fp4", "fp8_e5m2": "__fp8_e5m2", "fp8_e4m3": "__fp8_e4m3", "bfloat16": "__bf16", "float16":
            "__half", "tfloat32": "__tfloat32", "float32": "float", "float64": "double"
        }
        self._special_type_mappings: Dict[str, str] = {
            "bool": "bool", "binary": "uint8_t",  # Unused: 4-bit raises in map_int_type
            "void": "void", "str": "std::string", "TmaDescriptor": "cute::TmaDescriptor"
        }
        self._custom_handlers: Dict[type, Callable] = {}

    def register_int_type(self, bits: int, cpp_type: str, signed: bool = True):
        """Register an integer type mapping."""
        if signed:
            self._int_type_mappings[bits] = cpp_type
        else:
            self._uint_type_mappings[bits] = cpp_type

    def register_float_type(self, fmt: str, cpp_type: str):
        """Register a float type mapping."""
        self._float_type_mappings[fmt] = cpp_type

    def register_special_type(self, name: str, cpp_type: str):
        """Register a special type mapping."""
        self._special_type_mappings[name] = cpp_type

    def register_custom_handler(self, lltype_class: type, handler: Callable):
        """Register a custom handler for a specific LLType class."""
        self._custom_handlers[lltype_class] = handler

    def map_int_type(self, ll_type: ll.IntType) -> str:
        """Map IntType to C++ type string."""
        if ll_type.special == "bool":
            return self._special_type_mappings.get("bool", "bool")
        if ll_type.special == "binary" or ll_type.bits == 4:
            # TODO: Add proper 4-bit (int4/uint4) support. CUDA has native 4-bit types.
            raise NotImplementedError("4-bit integer types (int4/uint4, binary) are not yet supported. "
                                      "CUDA has native 4-bit support; this will be added in a future release.")

        sign_prefix = "u" if not ll_type.signed else ""
        mappings = self._uint_type_mappings if not ll_type.signed else self._int_type_mappings

        if ll_type.bits in mappings:
            return mappings[ll_type.bits]

        # Fallback: construct from sign and bits
        if ll_type.bits == 128:
            return "unsigned __int128" if not ll_type.signed else "__int128"
        return f"{sign_prefix}int{ll_type.bits}_t"

    def map_float_type(self, ll_type: ll.FloatType) -> str:
        """Map FloatType to C++ type string."""
        return self._float_type_mappings.get(ll_type.fmt, f"float{ll_type.fmt}")

    def map_special_type(self, name: str) -> Optional[str]:
        """Map special type name to C++ type string."""
        return self._special_type_mappings.get(name)

    def get_custom_handler(self, lltype_class: type) -> Optional[Callable]:
        """Get custom handler for a LLType class."""
        return self._custom_handlers.get(lltype_class)


# Global registry instance
_type_mapping_registry = TypeMappingRegistry()


def get_type_mapping_registry() -> TypeMappingRegistry:
    """Get the global type mapping registry."""
    return _type_mapping_registry
