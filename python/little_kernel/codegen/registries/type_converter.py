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

import little_kernel.language as ll
from .type_mapping_registry import get_type_mapping_registry


class TypeConverter:
    """Converts LLType to C++ type strings using registry."""

    def __init__(self, struct_types: dict = None):
        """
        Initialize type converter.
        
        Args:
            struct_types: Dictionary mapping StructType to C++ struct names
        """
        self.registry = get_type_mapping_registry()
        self.struct_types = struct_types if struct_types is not None else {}

    def _generate_struct_name(self, struct_type: "ll.StructType") -> str:
        """Generate a unique, valid C++ struct name from StructType fields."""
        field_suffix = "_".join([f"{name}_{self.lltype_to_cpp(typ)}" for name, typ in struct_type.type_tuple])
        valid_suffix = (field_suffix.replace("<", "_").replace(">", "_").replace("*", "_ptr_").replace(" ", ""))
        return f"Struct_{valid_suffix}"

    def lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a valid C++ type string."""
        # Scalar Types
        if hasattr(ll_type, "kind") and ll_type.kind == "int":
            return self.registry.map_int_type(ll_type)

        elif hasattr(ll_type, "kind") and ll_type.kind == "float":
            return self.registry.map_float_type(ll_type)

        elif hasattr(ll_type, "kind") and ll_type.kind == "void":
            return self.registry.map_special_type("void") or "void"

        elif hasattr(ll_type, "kind") and ll_type.kind == "str":
            return self.registry.map_special_type("str") or "std::string"

        # Annotated Types
        elif ll_type.is_const():
            inner_cpp = self.lltype_to_cpp(ll_type.inner_type)
            return f"__grid_constant__ {inner_cpp}" if ll_type.is_grid_constant() else f"const {inner_cpp}"

        elif ll_type.is_pointer():
            inner_cpp = self.lltype_to_cpp(ll_type.inner_type)
            return f"{inner_cpp}*"

        # Composite Types
        elif ll_type.is_tensor():
            elem_cpp = self.lltype_to_cpp(ll_type.element_type)
            return f"{elem_cpp}*"

        elif ll_type.is_struct():
            if ll_type not in self.struct_types:
                self.struct_types[ll_type] = self._generate_struct_name(ll_type)
            return self.struct_types[ll_type]

        # Special types
        type_str = str(ll_type)
        if type_str == "TmaDescriptor":
            return self.registry.map_special_type("TmaDescriptor") or "cute::TmaDescriptor"
        elif type_str == "Wgmma":
            raise NotImplementedError("WgmmaType is not supported for C++ generation")

        # Unsupported LLType
        raise NotImplementedError(f"Unsupported LLType: {type(ll_type).__name__} (value: {str(ll_type)})")
