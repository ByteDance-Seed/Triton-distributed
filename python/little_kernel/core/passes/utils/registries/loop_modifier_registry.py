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

import ast
from typing import Dict, Callable, Optional, Any
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute


def _unwrap_unroll_dyn(iter_node: ast.Call, ctx: Dict[str, Any]) -> ast.AST:
    """Unwrap ll.unroll_dyn(range(...), count) to the inner range() call.

    The count argument is validated by the codegen handler (it must be a
    positive integer literal); here we only need the iterator for tracing
    and type inference.
    """
    if len(iter_node.args) < 1:
        raise ValueError("Loop modifier 'unroll_dyn' requires a range() iterator and an "
                         "unroll count (e.g. ll.unroll_dyn(range(lo, hi, step), 4))")
    return iter_node.args[0]


class LoopModifier:
    """Represents a loop modifier (e.g., ll.unroll, ll.parallel, ll.serial)."""

    def __init__(self, name: str, unwrap: bool = True, handler: Optional[Callable] = None):
        """
        Initialize a loop modifier.
        
        Args:
            name: Name of the modifier (e.g., "unroll", "parallel", "serial")
            unwrap: Whether to unwrap the inner iterator (default: True)
            handler: Optional handler function for custom processing
        """
        self.name = name
        self.unwrap = unwrap
        self.handler = handler

    def process(self, iter_node: ast.Call, ctx: Dict[str, Any]) -> ast.AST:
        """
        Process the loop modifier and return the inner iterator.
        
        Args:
            iter_node: The AST Call node representing the modifier call
            ctx: Context dictionary for resolving attributes
        
        Returns:
            The inner iterator AST node
        """
        if self.handler:
            return self.handler(iter_node, ctx)

        if self.unwrap and len(iter_node.args) == 1:
            return iter_node.args[0]

        return iter_node


class LoopModifierRegistry:
    """Registry for loop modifiers that affect loop behavior."""

    def __init__(self):
        self._modifiers: Dict[str, LoopModifier] = {}
        self._register_default_modifiers()

    def _register_default_modifiers(self):
        """Register default loop modifiers."""
        # Register ll.unroll
        self.register_modifier("unroll", LoopModifier("unroll", unwrap=True))
        self.register_modifier("no_unroll", LoopModifier("no_unroll", unwrap=True))
        # Register ll.unroll_dyn: ll.unroll_dyn(range(...), count) unwraps to the
        # range; the count argument is consumed by the codegen handler.
        self.register_modifier("unroll_dyn", LoopModifier("unroll_dyn", unwrap=True, handler=_unwrap_unroll_dyn))

    def register_modifier(self, name: str, modifier: LoopModifier):
        """
        Register a loop modifier.
        
        Args:
            name: Name of the modifier (e.g., "unroll", "parallel")
            modifier: LoopModifier instance
        """
        self._modifiers[name] = modifier

    def is_modifier_call(self, node: ast.Call, ctx: Dict[str, Any]) -> bool:
        """
        Check if a Call node represents a loop modifier.
        
        Args:
            node: AST Call node to check
            ctx: Context dictionary for resolving attributes
        
        Returns:
            True if the node is a loop modifier call
        """
        if not isinstance(node.func, ast.Attribute):
            return False

        attr_name = node.func.attr

        # Check if it's a registered modifier by name
        if attr_name in self._modifiers:
            return True

        # Try to resolve and check if it matches a registered modifier
        try:
            resolved_func = recursive_resolve_attribute(node.func, ctx)
            if hasattr(resolved_func, '__name__'):
                func_name = resolved_func.__name__
                if func_name in self._modifiers:
                    return True
        except Exception:
            pass

        return False

    def get_modifier(self, node: ast.Call, ctx: Dict[str, Any]) -> Optional[LoopModifier]:
        """
        Get the LoopModifier for a Call node.
        
        Args:
            node: AST Call node representing the modifier call
            ctx: Context dictionary for resolving attributes
        
        Returns:
            LoopModifier if found, None otherwise
        """
        if not isinstance(node.func, ast.Attribute):
            return None

        attr_name = node.func.attr

        # Direct lookup by attribute name
        if attr_name in self._modifiers:
            return self._modifiers[attr_name]

        # Try to resolve and check
        try:
            resolved_func = recursive_resolve_attribute(node.func, ctx)
            if hasattr(resolved_func, '__name__'):
                func_name = resolved_func.__name__
                if func_name in self._modifiers:
                    return self._modifiers[func_name]
        except Exception:
            pass

        return None

    def unwrap_modifier(self, node: ast.Call, ctx: Dict[str, Any]) -> ast.AST:
        """
        Unwrap a loop modifier call to get the inner iterator.
        
        Args:
            node: AST Call node representing the modifier call
            ctx: Context dictionary for resolving attributes
        
        Returns:
            The inner iterator AST node
        
        Raises:
            ValueError: If the modifier requires exactly 1 argument but got different number
        """
        modifier = self.get_modifier(node, ctx)
        if modifier is None:
            return node

        # Handler-based modifiers (e.g. unroll_dyn) own their argument
        # validation; the generic 1-argument rule only applies otherwise.
        if modifier.handler is None and len(node.args) != 1:
            raise ValueError(f"Loop modifier '{modifier.name}' requires exactly 1 argument (got {len(node.args)})")

        return modifier.process(node, ctx)


# Global registry instance
_loop_modifier_registry = LoopModifierRegistry()


def get_loop_modifier_registry() -> LoopModifierRegistry:
    """Get the global loop modifier registry."""
    return _loop_modifier_registry


def register_loop_modifier(name: str, unwrap: bool = True, handler: Optional[Callable] = None):
    """
    Convenience function to register a loop modifier.
    
    Args:
        name: Name of the modifier (e.g., "unroll", "parallel")
        unwrap: Whether to unwrap the inner iterator (default: True)
        handler: Optional handler function for custom processing
    """
    modifier = LoopModifier(name, unwrap=unwrap, handler=handler)
    _loop_modifier_registry.register_modifier(name, modifier)
