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
from typing import Dict, Any

from little_kernel.core.type_system import Tensor
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__, __INTERNAL_DYN_SHMEM__
from little_kernel.language.intrin.memory import (
    alloc_dynamic_shared_memory,
    alloc_local_memory,
    alloc_local_zeros,
    alloc_shared_memory,
    slice_dynamic_shared_memory,
)
from .mem_analysis import MemoryAnalyzer
from .pass_base import CompleteASTMutator
from .utils.add_parent_reference import add_parent_references
from .utils.preserve_attributes import create_node_with_attrs
from .utils.update_context import update_context


def _target_to_load_expr(node: ast.AST) -> ast.AST:
    """Convert assignment target (Store ctx) to expression for use as Call argument (Load ctx).
    Required for A[0] etc.: ast.Name(id='A[0]') is invalid; must use ast.Subscript."""
    if isinstance(node, ast.Name):
        return ast.Name(id=node.id, ctx=ast.Load())
    elif isinstance(node, ast.Subscript):
        return ast.Subscript(
            value=_target_to_load_expr(node.value),
            slice=node.slice,
            ctx=ast.Load(),
        )
    else:
        raise NotImplementedError(f"Unsupported target type for mem alloc: {type(node).__name__}")


class InsertMemAlloc(CompleteASTMutator):

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.alloc_dyn_name = update_context(self.ctx, alloc_dynamic_shared_memory.__name__,
                                             alloc_dynamic_shared_memory)
        self.slice_dyn_name = update_context(self.ctx, slice_dynamic_shared_memory.__name__,
                                             slice_dynamic_shared_memory)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        new_node = super().generic_visit(node)
        if __LITTLE_KERNEL_ENTRY__ in self.ctx and new_node.name == self.ctx[__LITTLE_KERNEL_ENTRY__].__name__:
            add_parent_references(new_node)
            mem_ana = MemoryAnalyzer(self.ctx)
            mem_ana.visit(new_node)
            # add memory alloc for dynamic_shared memory
            alloc_dynamic_shared_memory_name = __INTERNAL_DYN_SHMEM__
            for scope in mem_ana.valid_scopes:
                # only insert alloc for shared_dynamic memory
                if scope != "dynamic_shared":
                    continue
                if mem_ana.first_mem_call[scope] is not None:
                    align_bytes = mem_ana.align_bytes[scope]
                    alloc_bytes = mem_ana.scope_offsets[scope]
                    num_stmts = len(new_node.body)
                    for i in range(num_stmts):
                        if new_node.body[i] == mem_ana.first_mem_call[scope]:
                            alloc_stmt = ast.Expr(value=ast.Call(
                                func=ast.Name(id=self.alloc_dyn_name, ctx=ast.Load()),
                                args=[
                                    ast.Constant(value=alloc_bytes),
                                    ast.Constant(value=align_bytes),
                                    ast.Name(id=alloc_dynamic_shared_memory_name, ctx=ast.Load()),
                                ],
                                keywords=[],
                            ))
                            new_node.body.insert(i, alloc_stmt)
                            break
            # slice dynamic_shared memory
            num_stmts = len(new_node.body)
            for i in range(num_stmts):
                for var_name, var in mem_ana.variables.items():
                    is_target = True
                    is_target = is_target and isinstance(new_node.body[i], ast.Assign)
                    is_target = is_target and len(new_node.body[i].targets) == 1
                    if is_target:
                        if isinstance(new_node.body[i].targets[0], ast.Name):
                            is_target = is_target and var_name == new_node.body[i].targets[0].id
                        elif isinstance(new_node.body[i].targets[0], ast.Subscript) and isinstance(
                                new_node.body[i].targets[0].slice, ast.Constant):
                            is_target = is_target and var_name == f"{new_node.body[i].targets[0].value.id}[{new_node.body[i].targets[0].slice.value}]"
                        else:
                            is_target = False
                    if var["scope"] == "dynamic_shared" and is_target:
                        original_assign = new_node.body[i]
                        var_expr = _target_to_load_expr(var["expr"])
                        slice_stmt = create_node_with_attrs(
                            ast.Expr, original_assign, value=ast.Call(
                                func=ast.Name(id=self.slice_dyn_name, ctx=ast.Load()),
                                args=[
                                    var_expr,
                                    ast.Constant(value=Tensor[var["dtype"]]),
                                    ast.Constant(value=var["offset"]),
                                    ast.Constant(value=var["total_bytes"]),
                                    ast.Name(id=alloc_dynamic_shared_memory_name, ctx=ast.Load()),
                                ],
                                keywords=[],
                            ))
                        new_node.body[i] = slice_stmt
                        break
                    elif var["scope"] == "local" and is_target:
                        original_assign = new_node.body[i]
                        var_expr = _target_to_load_expr(var["expr"])
                        # zeros(..., scope="local") -> single alloc_local_zeros (decl+init in one, stable codegen)
                        if var.get("is_zeros"):
                            alloc_fn = alloc_local_zeros
                        else:
                            alloc_fn = alloc_local_memory
                        obj_name = update_context(self.ctx, alloc_fn.__name__, alloc_fn)
                        new_node.body[i] = create_node_with_attrs(
                            ast.Expr, original_assign, value=ast.Call(
                                func=ast.Name(id=obj_name, ctx=ast.Load()),
                                args=[
                                    var_expr,
                                    ast.Constant(value=var["dtype"]),
                                    ast.Constant(value=var["total_elements"]),
                                ],
                                keywords=[],
                            ))
                        break
                    elif var["scope"] == "shared" and is_target:
                        original_assign = new_node.body[i]
                        var_expr = _target_to_load_expr(var["expr"])
                        obj_name = update_context(self.ctx, alloc_shared_memory.__name__, alloc_shared_memory)
                        align_bytes = mem_ana.align_bytes.get("shared", -1)
                        if align_bytes < 0:
                            align_bytes = 0
                        slice_stmt = create_node_with_attrs(
                            ast.Expr, original_assign, value=ast.Call(
                                func=ast.Name(id=obj_name, ctx=ast.Load()),
                                args=[
                                    var_expr,
                                    ast.Constant(value=var["dtype"]),
                                    ast.Constant(value=var["total_elements"]),
                                    ast.Constant(value=align_bytes),
                                ],
                                keywords=[],
                            ))
                        new_node.body[i] = slice_stmt
                        break
            # Recursively replace ll.empty in nested blocks (if/while/for)
            self._replace_empty_assigns_in_body(new_node.body, mem_ana)
        new_node = ast.fix_missing_locations(new_node)
        return new_node

    def _replace_empty_assigns_in_body(self, body: list, mem_ana: MemoryAnalyzer) -> None:
        """Replace ll.empty assigns with alloc/slice in body and nested blocks. No hardcoded sizes."""
        i = 0
        while i < len(body):
            stmt = body[i]
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                var_name = None
                if isinstance(target, ast.Name):
                    var_name = target.id
                elif isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant):
                    var_name = f"{target.value.id}[{target.slice.value}]"
                if var_name and var_name in mem_ana.variables:
                    var = mem_ana.variables[var_name]
                    scope = var["scope"]
                    var_expr = _target_to_load_expr(var["expr"])
                    if scope == "local":
                        alloc_fn = alloc_local_zeros if var.get("is_zeros") else alloc_local_memory
                        obj_name = update_context(self.ctx, alloc_fn.__name__, alloc_fn)
                        body[i] = create_node_with_attrs(
                            ast.Expr, stmt, value=ast.Call(
                                func=ast.Name(id=obj_name, ctx=ast.Load()),
                                args=[
                                    var_expr,
                                    ast.Constant(value=var["dtype"]),
                                    ast.Constant(value=var["total_elements"]),
                                ],
                                keywords=[],
                            ))
                    elif scope == "shared":
                        obj_name = update_context(self.ctx, alloc_shared_memory.__name__, alloc_shared_memory)
                        align_bytes = mem_ana.align_bytes.get("shared", -1)
                        if align_bytes < 0:
                            align_bytes = 0
                        body[i] = create_node_with_attrs(
                            ast.Expr, stmt, value=ast.Call(
                                func=ast.Name(id=obj_name, ctx=ast.Load()),
                                args=[
                                    var_expr,
                                    ast.Constant(value=var["dtype"]),
                                    ast.Constant(value=var["total_elements"]),
                                    ast.Constant(value=align_bytes),
                                ],
                                keywords=[],
                            ))
                    elif scope == "dynamic_shared":
                        body[i] = create_node_with_attrs(
                            ast.Expr, stmt, value=ast.Call(
                                func=ast.Name(id=self.slice_dyn_name, ctx=ast.Load()),
                                args=[
                                    var_expr,
                                    ast.Constant(value=Tensor[var["dtype"]]),
                                    ast.Constant(value=var["offset"]),
                                    ast.Constant(value=var["total_bytes"]),
                                    ast.Name(id=__INTERNAL_DYN_SHMEM__, ctx=ast.Load()),
                                ],
                                keywords=[],
                            ))
            elif isinstance(stmt, ast.If):
                self._replace_empty_assigns_in_body(stmt.body, mem_ana)
                self._replace_empty_assigns_in_body(stmt.orelse, mem_ana)
            elif isinstance(stmt, ast.While):
                self._replace_empty_assigns_in_body(stmt.body, mem_ana)
            elif isinstance(stmt, ast.For):
                self._replace_empty_assigns_in_body(stmt.body, mem_ana)
            i += 1


def insert_mem_alloc(tree: ast.AST, ctx: Dict[str, Any]) -> ast.AST:
    insert_alloc = InsertMemAlloc(ctx)
    new_ast = insert_alloc.visit(tree)
    return new_ast
