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
Utility for preserving special attributes on AST nodes across passes.

This module provides a unified mechanism for managing special attributes
that need to be preserved when AST nodes are copied or transformed by passes.
All passes should use these utilities to ensure special attributes are
not lost during AST transformations.
"""

import ast
from little_kernel.language.builtin_base import AST_SPECIAL_ATTRS

# Import AST_SPECIAL_ATTRS from builtin_base.py where it's unified with INTERNAL_ATTR
# This ensures all special attributes are managed in one place


def copy_node_with_attrs(node: ast.AST, new_node: ast.AST) -> ast.AST:
    """
    Copy special attributes from source node to target node.
    
    Args:
        node: Source AST node (may have special attributes)
        new_node: Target AST node (will receive special attributes)
    
    Returns:
        The target node with special attributes copied
    """
    if not isinstance(node, ast.AST) or not isinstance(new_node, ast.AST):
        return new_node

    for attr in AST_SPECIAL_ATTRS:
        if hasattr(node, attr):
            setattr(new_node, attr, getattr(node, attr))

    return new_node


def create_node_with_attrs(node_type: type, source_node: ast.AST, **kwargs) -> ast.AST:
    """
    Create a new AST node of the specified type, preserving special attributes from source.
    
    Args:
        node_type: The AST node class to instantiate (e.g., ast.Assign)
        source_node: Source node to copy special attributes from
        **kwargs: Arguments to pass to the node constructor
    
    Returns:
        New AST node with special attributes preserved
    """
    new_node = node_type(**kwargs)
    ast.copy_location(new_node, source_node)
    return copy_node_with_attrs(source_node, new_node)


def has_special_attrs(node: ast.AST) -> bool:
    """
    Check if a node has any special attributes.
    
    Args:
        node: AST node to check
    
    Returns:
        True if node has any special attributes, False otherwise
    """
    if not isinstance(node, ast.AST):
        return False

    return any(hasattr(node, attr) for attr in AST_SPECIAL_ATTRS)


def get_special_attrs(node: ast.AST) -> dict:
    """
    Get all special attributes from a node as a dictionary.
    
    Args:
        node: AST node to extract attributes from
    
    Returns:
        Dictionary of special attributes (only those that exist on the node)
    """
    if not isinstance(node, ast.AST):
        return {}

    attrs = {}
    for attr in AST_SPECIAL_ATTRS:
        if hasattr(node, attr):
            attrs[attr] = getattr(node, attr)

    return attrs


def set_special_attrs(node: ast.AST, attrs: dict) -> ast.AST:
    """
    Set special attributes on a node from a dictionary.
    
    Args:
        node: AST node to set attributes on
        attrs: Dictionary of attribute names to values
    
    Returns:
        The node with attributes set
    """
    if not isinstance(node, ast.AST):
        return node

    for attr, value in attrs.items():
        if attr in AST_SPECIAL_ATTRS:
            setattr(node, attr, value)

    return node
