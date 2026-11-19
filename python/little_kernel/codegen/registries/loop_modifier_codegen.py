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
from typing import Dict, Callable, Optional
from little_kernel.core.passes.utils.registries.loop_modifier_registry import get_loop_modifier_registry


class LoopModifierCodegen:
    """Handles code generation for loop modifiers."""

    def __init__(self):
        self.registry = get_loop_modifier_registry()
        self._codegen_handlers: Dict[str, Callable] = {}
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default codegen handlers for loop modifiers."""
        # Register unroll handler: generates #pragma unroll
        self.register_codegen("unroll", self._codegen_unroll)
        self.register_codegen("no_unroll", self._codegen_no_unroll)
        self.register_codegen("unroll_dyn", self._codegen_unroll_dyn)

    def register_codegen(self, modifier_name: str, handler: Callable):
        """
        Register a codegen handler for a loop modifier.
        
        Args:
            modifier_name: Name of the modifier (e.g., "unroll", "parallel")
            handler: Function that takes (emitter, node) and generates code
        """
        self._codegen_handlers[modifier_name] = handler

    def _codegen_unroll(self, emitter, node: ast.For) -> Optional[str]:
        """
        Generate code for ll.unroll modifier.
        
        Args:
            emitter: CppEmitter instance
            node: ast.For node
        
        Returns:
            C++ pragma string or None
        """
        return "#pragma unroll"

    def _codegen_no_unroll(self, emitter, node: ast.For) -> Optional[str]:
        return "#pragma unroll 1"

    @staticmethod
    def _resolve_unroll_count(node: ast.For) -> int:
        """Extract the trace-time positive-integer unroll count.

        node.iter is still the un-unwrapped ll.unroll_dyn(range(...), count)
        call when the modifier codegen runs (process_modifier precedes the
        registry unwrap in visit_For), so the count is args[1] or the
        keyword `count`.
        """
        iter_node = node.iter
        args = list(getattr(iter_node, "args", []))
        keywords = list(getattr(iter_node, "keywords", []))
        count_node = None
        if len(args) >= 2:
            count_node = args[1]
        else:
            for kw in keywords:
                if kw.arg == "count":
                    count_node = kw.value
                    break
        if count_node is None:
            raise ValueError("ll.unroll_dyn requires an unroll count: "
                             "ll.unroll_dyn(range(...), 4)")
        # Python-level unary minus on a literal (e.g. -4) arrives as UnaryOp.
        negative = False
        if isinstance(count_node, ast.UnaryOp) and isinstance(count_node.op, ast.USub):
            negative = True
            count_node = count_node.operand
        if not (isinstance(count_node, ast.Constant) and isinstance(count_node.value, int)
                and not isinstance(count_node.value, bool)):
            raise ValueError("ll.unroll_dyn count must be an integer literal at trace "
                             "time (e.g. ll.unroll_dyn(range(...), 4))")
        count = -count_node.value if negative else count_node.value
        if count < 1:
            raise ValueError(f"ll.unroll_dyn count must be a positive integer (got {count})")
        return count

    def _codegen_unroll_dyn(self, emitter, node: ast.For) -> Optional[str]:
        """Emit `#pragma unroll <count>`: partial unroll for dynamic-trip loops."""
        return f"#pragma unroll {self._resolve_unroll_count(node)}"

    def _codegen_parallel(self, emitter, node: ast.For) -> Optional[str]:
        """
        Example handler for ll.parallel modifier (for future use).
        Can be registered when ll.parallel is implemented.
        """
        return "#pragma omp parallel for"

    def _codegen_serial(self, emitter, node: ast.For) -> Optional[str]:
        """
        Example handler for ll.serial modifier (for future use).
        Serial loops don't need pragma, but can be registered for consistency.
        """
        return None

    def process_modifier(self, emitter, node: ast.For) -> Optional[str]:
        """
        Process loop modifier and generate corresponding C++ code.
        
        Args:
            emitter: CppEmitter instance
            node: ast.For node
        
        Returns:
            C++ code string (e.g., "#pragma unroll") or None
        """
        iter_node = node.iter
        if not isinstance(iter_node, ast.Call):
            return None

        modifier = self.registry.get_modifier(iter_node, emitter.ctx)
        if modifier is None:
            return None

        handler = self._codegen_handlers.get(modifier.name)
        if handler:
            return handler(emitter, node)

        return None


# Global codegen instance
_loop_modifier_codegen = LoopModifierCodegen()


def get_loop_modifier_codegen() -> LoopModifierCodegen:
    """Get the global loop modifier codegen instance."""
    return _loop_modifier_codegen


def register_loop_modifier_codegen(modifier_name: str, handler: Callable):
    """
    Convenience function to register a codegen handler for a loop modifier.
    
    Args:
        modifier_name: Name of the modifier (e.g., "unroll", "parallel")
        handler: Function that takes (emitter, node) and generates code
    """
    _loop_modifier_codegen.register_codegen(modifier_name, handler)
