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

from typing import Dict, List, Callable
from .dump import dump_ast
import os

PASS_DEBUG = os.getenv("PASS_DEBUG", "").split(",")
PASS_DEBUG_ALL = "all" in PASS_DEBUG


def empty_pass(tree, ctx, *args, **kwargs):
    """Empty pass that does nothing."""
    return tree


def debug_wrapper(pass_func, pass_name, simple_mode: bool = True):
    """Wrap a pass function with debug output."""

    def wrapper_func(tree, ctx, *args, **kwargs):
        tree = pass_func(tree, ctx, *args, **kwargs)
        if pass_name in PASS_DEBUG or PASS_DEBUG_ALL:
            dump_ast(f"after {pass_name}", simple_mode=simple_mode)(tree, ctx, *args, **kwargs)
        return tree

    return wrapper_func


class PassRegistry:
    """Registry for compilation passes organized by backend."""

    def __init__(self):
        self._passes: Dict[str, List[Callable]] = {}

    def register_pass(self, backend: str, pass_func: Callable, pass_name: str = None, enable_debug: bool = True,
                      simple_mode: bool = True):
        """
        Register a pass for a specific backend.
        
        Args:
            backend: Backend name (e.g., "cuda")
            pass_func: Pass function to register
            pass_name: Optional name for the pass (for debugging)
            enable_debug: Whether to enable debug wrapper
        """
        if backend not in self._passes:
            self._passes[backend] = []

        if enable_debug and pass_name:
            pass_func = debug_wrapper(pass_func, pass_name, simple_mode=simple_mode)

        self._passes[backend].append(pass_func)

    def get_passes(self, backend: str) -> List[Callable]:
        """Get all passes for a backend."""
        return self._passes.get(backend, [])

    def register_backend_passes(self, backend: str, passes: List[Callable]):
        """Register multiple passes for a backend at once."""
        for i, pass_func in enumerate(passes):
            pass_name = getattr(pass_func, '__name__', f"pass_{i}")
            self.register_pass(backend, pass_func, pass_name)


# Global registry instance
_pass_registry = PassRegistry()


def get_pass_registry() -> PassRegistry:
    """Get the global pass registry."""
    return _pass_registry


def register_pass(backend: str, pass_func: Callable, pass_name: str = None, enable_debug: bool = True,
                  simple_mode: bool = True):
    """Convenience function to register a pass."""
    _pass_registry.register_pass(backend, pass_func, pass_name, enable_debug, simple_mode)


def get_passes(backend: str) -> List[Callable]:
    """Convenience function to get passes for a backend."""
    return _pass_registry.get_passes(backend)
