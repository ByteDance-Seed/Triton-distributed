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
import inspect
from typing import Dict, Any, Callable
from collections import OrderedDict

from little_kernel.core.type_system import get_dtype_size
from .utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.language.intrin.memory import empty, zeros, align_memory
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from .utils.extract_empty_params import extract_empty_params, get_shape_size


class MemoryAnalyzer(ast.NodeVisitor):
    """
    Analyzes AST nodes to track memory allocation via `ll.empty` calls.
    - Calculates total bytes used for each scope (`local`/`shared`/`dynamic_shared`).
    - Records detailed info for each allocated variable: scope, dtype, size (bytes), and offset in its scope.
    """

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx  # Global context of the function (to resolve variables like BLOCK_M, kNumStages)

        # Track cumulative byte offsets for each scope (used to calculate variable offsets)
        self.scope_offsets: Dict[str, int] = {"local": 0, "shared": 0, "dynamic_shared": 0}

        # Store detailed info for each variable allocated via ll.empty
        self.variables: Dict[str, Dict[str, Any]] = OrderedDict()

        # Valid scope types for ll.empty
        self.valid_scopes = {"local", "shared", "dynamic_shared"}

        # Alignment requirement for memory offsets (bytes)
        self.align_bytes: Dict[str, int] = {"local": -1, "shared": -1, "dynamic_shared": -1}

        # record the first call related to memory
        self.first_mem_call: Dict[str, ast.AST] = {"local": None, "shared": None, "dynamic_shared": None}

    def _handle_align_call(self, node: ast.Call) -> None:
        """
        Handle `ll.align_memory` calls, extract align_bytes and scope, then update memory analysis.
        """
        align_memory_signature = inspect.signature(align_memory)
        valid = isinstance(node.parent, ast.Expr)
        if valid:
            parent = node.parent
            while parent is not None and not isinstance(parent, ast.FunctionDef):
                parent = getattr(parent, "parent", None)
            valid = parent is not None and self.ctx[__LITTLE_KERNEL_ENTRY__].__name__ == parent.name
        if not valid:
            raise ValueError(
                f"ll.align_memory must be called inside the body of main function, see {ast.unparse(node)}")

        if len(node.args) + len(node.keywords) < len(align_memory_signature.parameters):
            raise ValueError(f"ll.align_memory requires at least {len(align_memory_signature.parameters)} arguments")

        align_bytes = None
        scope = None
        for kw in node.keywords:
            if kw.arg == "align_bytes":
                align_bytes = kw.value
                break
        if align_bytes is None:
            assert len(node.args) > 0, "ll.align_memory requires at least an align_bytes argument"
            align_bytes = node.args[0]

        assert isinstance(align_bytes, ast.Constant), "ll.align_memory align_bytes must be a constant"
        assert isinstance(align_bytes.value, int), "ll.align_memory align_bytes must be an integer"

        align_bytes = align_bytes.value

        for kw in node.keywords:
            if kw.arg == "scope":
                scope = kw.value
                break
        if scope is None:
            assert len(node.args) > 1, "ll.align_memory requires at least a scope argument"
            scope = node.args[1]
        assert isinstance(scope, ast.Constant), "ll.align_memory scope must be a constant"
        assert isinstance(scope.value, str), "ll.align_memory scope must be a string"
        assert scope.value in self.valid_scopes, f"ll.align_memory scope must be one of {self.valid_scopes}, got {scope.value}"
        scope = scope.value

        if self.align_bytes[scope] < 0:
            self.align_bytes[scope] = align_bytes
        else:
            raise RuntimeError(
                f"ll.align_memory align_bytes for scope {scope} has already been set to {self.align_bytes[scope]}, cannot be changed to {align_bytes}"
            )

        # record the call if it's the first
        if self.first_mem_call[scope] is None:
            self.first_mem_call[scope] = node.parent

    def _handle_zeros_call(self, node: ast.Call) -> None:
        """Handle `ll.zeros` like ll.empty (same shape/dtype/scope, alloc+zero)."""
        self._handle_empty_like_call(node, "ll.zeros")

    def _handle_empty_call(self, node: ast.Call) -> None:
        """Handle `ll.empty` calls."""
        self._handle_empty_like_call(node, "ll.empty")

    def _handle_empty_like_call(self, node: ast.Call, api_name: str) -> None:
        """
        Handle `ll.empty` / `ll.zeros` calls, extract shape, dtype, and scope, then update memory analysis.
        """
        # Trace back to find the variable name being assigned (e.g., "D_smem_tensor" in "D_smem_tensor = ll.empty(...)")
        var_name = None
        var_expr = None
        if isinstance(node.parent, ast.Assign) and len(node.parent.targets) == 1:
            target = node.parent.targets[0]
            if isinstance(target, ast.Name):
                var_name = target.id
                var_expr = target
            elif isinstance(target, ast.Subscript):
                assert isinstance(
                    target.slice, ast.Constant
                ), f"Empty assign to array element must use constant subscripts, but get {ast.dump(target.slice)}"
                var_name = f"{target.value.id}[{target.slice.value}]"
                var_expr = target
            else:
                raise NotImplementedError(f"Unsupported target type for ll.empty: {type(target).__name__}")
        if not var_name:
            raise RuntimeError(
                f"{api_name} call is not assigned to a variable: {ast.dump(node)}\nMaybe should use flatten_empty pass first."
            )

        # Find enclosing FunctionDef (may be nested inside if/while/for)
        parent = node.parent
        while parent is not None and not isinstance(parent, ast.FunctionDef):
            parent = getattr(parent, "parent", None)
        valid = parent is not None and self.ctx[__LITTLE_KERNEL_ENTRY__].__name__ == parent.name
        if not valid:
            raise ValueError(
                f"{api_name} must be called inside the body of main function, see {ast.unparse(node)}\nMaybe should use flatten_empty pass first."
            )

        shape_expr, dtype_expr, scope_expr = extract_empty_params(node)
        assert isinstance(dtype_expr,
                          ast.Constant), f"{api_name} dtype must be a constant, got {type(dtype_expr).__name__}"
        dtype = dtype_expr.value
        assert isinstance(scope_expr,
                          ast.Constant), f"{api_name} scope must be a constant, got {type(scope_expr).__name__}"
        scope = scope_expr.value
        total_elements = get_shape_size(shape_expr)

        # Calculate total bytes for this allocation
        element_size = get_dtype_size(dtype)
        total_bytes = int(total_elements * element_size)
        assert total_bytes > 0, f"{api_name} call for {var_name} with shape {shape_expr} and dtype {dtype} results in zero bytes allocation"

        # Record offset (current offset before allocation) and update scope offset
        offset = self.scope_offsets[scope]
        self.scope_offsets[scope] += total_bytes

        # Store detailed info about this variable (empty and zeros both go here)
        if var_name in self.variables:
            raise ValueError(f"{api_name} call for {var_name} has already been recorded")
        self.variables[var_name] = ({
            "name": var_name, "expr": var_expr, "scope": scope, "dtype": dtype, "element_size": element_size,
            "total_elements": total_elements, "total_bytes": total_bytes, "offset":
            offset,  # Offset within its scope (bytes)
            "is_zeros": (api_name == "ll.zeros"),  # Pass uses this to replace with alloc_local_zeros
        })

        # record the call if it's the first
        if self.first_mem_call[scope] is None:
            self.first_mem_call[scope] = node.parent

    def visit_Call(self, node: ast.Call) -> None:
        """
        Override to track `ll.empty` calls. Extracts:
        - Variable name (target of the assignment)
        - Shape (to calculate total elements)
        - Dtype (to calculate element size)
        - Scope (local/shared/dynamic_shared)
        Then computes total bytes and updates scope offsets.
        """
        # Check if the call is to attribute `ll.empty`
        func = recursive_resolve_attribute(node.func, self.ctx)
        if isinstance(func, Callable) and hasattr(func, '__name__'):

            # For align
            if func.__name__ == align_memory.__name__:
                self._handle_align_call(node)
            # For alloc by empty
            elif func.__name__ == empty.__name__:
                self._handle_empty_call(node)
            elif func.__name__ == zeros.__name__:
                self._handle_zeros_call(node)

        self.generic_visit(node)

    def print_summary(self) -> None:
        """Print a summary of shared memory usage and variable details."""
        print("=== Shared Memory Usage Summary ===")
        # Print total bytes per scope
        for scope in self.valid_scopes:
            print(f"Total {scope} memory: {self.scope_offsets[scope]} bytes, align to {self.align_bytes[scope]} bytes")

        for scope in self.valid_scopes:
            if self.first_mem_call[scope] is not None:
                print(f"First {scope} memory call: {ast.unparse(self.first_mem_call[scope])}")

        # Print detailed variable info
        print("\n=== Variable Details ===")
        for var_name, var in self.variables.items():
            print(f"Variable: {var['name']}\n"
                  f"  Expr: {ast.unparse(var['expr'])}\n"
                  f"  Scope: {var['scope']}\n"
                  f"  Dtype: {var['dtype']}\n"
                  f"  Elements: {var['total_elements']} (each {var['element_size']} bytes)\n"
                  f"  Total: {var['total_bytes']} bytes\n"
                  f"  Offset in {var['scope']}: {var['offset']} bytes\n"
                  f"  Reinterpret as: {var['dtype']} at offset {var['offset']} in {var['scope']} memory\n"
                  "-----------------------------------------")
