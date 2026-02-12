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
IR nodes for representing enum and special struct definitions in the AST.

These nodes are inserted into the Module body during materialize passes
and processed during codegen to generate C++ definitions.
"""

import ast
from typing import Dict, Any, Optional
from enum import Enum


class IRNodeType(Enum):
    """Types of IR nodes that can be inserted into the AST."""
    ENUM_DEF = "enum_def"
    SPECIAL_STRUCT_DEF = "special_struct_def"


class EnumDef(ast.AST):
    """
    IR node representing an enum definition.
    
    This node is inserted into the Module body during materialize pass
    and processed during codegen to generate C++ enum class definition.
    """
    _fields = ('enum_name', 'enum_class', 'ast_node')

    def __init__(self, enum_name: str, enum_class: type, ast_node: Optional[ast.AST] = None):
        """
        Initialize an EnumDef node.
        
        Args:
            enum_name: Name of the enum (e.g., 'GemmType')
            enum_class: The Python Enum class object
            ast_node: Optional AST node that triggered this enum definition
        """
        self.enum_name = enum_name
        self.enum_class = enum_class
        self.ast_node = ast_node
        self._ir_node_type = IRNodeType.ENUM_DEF


class SpecialStructDef(ast.AST):
    """
    IR node representing a special struct definition.
    
    This node is inserted into the Module body during materialize pass
    and processed during codegen to generate C++ struct definition.
    """
    _fields = ('struct_name', 'struct_info', 'ast_node')

    def __init__(self, struct_name: str, struct_info: Dict[str, Any], ast_node: Optional[ast.AST] = None):
        """
        Initialize a SpecialStructDef node.
        
        Args:
            struct_name: Name of the special struct (e.g., 'Scheduler')
            struct_info: Dictionary containing struct information (class, etc.)
            ast_node: Optional AST node that triggered this struct definition
        """
        self.struct_name = struct_name
        self.struct_info = struct_info
        self.ast_node = ast_node
        self._ir_node_type = IRNodeType.SPECIAL_STRUCT_DEF


def is_ir_node(node: ast.AST) -> bool:
    """Check if a node is an IR node (EnumDef or SpecialStructDef)."""
    return hasattr(node, '_ir_node_type')


def get_ir_node_type(node: ast.AST) -> Optional[IRNodeType]:
    """Get the IR node type if the node is an IR node."""
    if is_ir_node(node):
        return getattr(node, '_ir_node_type', None)
    return None
