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
import json
from typing import Dict, Any

from little_kernel.core.type_system import Tensor
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__, __INTERNAL_DYN_SHMEM__, __LOCAL_ARRAY_SHAPES_CTX_KEY__
from little_kernel.backends.common.gpu.language.intrin.allocation import alloc_dynamic_shared_memory, alloc_local_memory, alloc_local_zeros, alloc_local_full, alloc_shared_memory, slice_dynamic_shared_memory
from little_kernel.backends.common.gpu.passes.mem_analysis import MemoryAnalyzer
from little_kernel.core.passes.pass_base import CompleteASTMutator
from little_kernel.core.passes.utils.add_parent_reference import add_parent_references
from little_kernel.core.passes.utils.preserve_attributes import create_node_with_attrs
from little_kernel.core.passes.utils.update_context import update_context


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


def _alloc_stmt_target(stmt: ast.stmt):
    """Return the single target of an allocation assignment statement.

    Both plain assignments (``a = ll.empty(...)``) and annotated assignments
    (``a: ll.Tensor[dtype] = ll.empty(...)``) bind one buffer name; annotation-
    only statements without a value cannot allocate and yield None."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        return stmt.targets[0]
    if isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        return stmt.target
    return None


def _target_matches_var(target: ast.AST, var_name: str) -> bool:
    """Whether an Assign/AnnAssign target binds the analyzed buffer name."""
    if isinstance(target, ast.Name):
        return var_name == target.id
    if (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)):
        return var_name == f"{target.value.id}[{target.slice.value}]"
    return False


class InsertMemAlloc(CompleteASTMutator):

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.alloc_dyn_name = update_context(self.ctx, alloc_dynamic_shared_memory.__name__,
                                             alloc_dynamic_shared_memory)
        self.slice_dyn_name = update_context(self.ctx, slice_dynamic_shared_memory.__name__,
                                             slice_dynamic_shared_memory)

    def _record_local_array_shape(self, var: Dict[str, Any]) -> None:
        """Publish the declared extents of a local array so codegen can keep
        subscript emission consistent with the flattened 1-D C declaration:
        rank>=2 arrays get row-major flattened subscripts, and any subscript
        whose arity does not match the recorded rank is rejected instead of
        emitting invalid (`a[i][j]` on `float a[N]`) or silently mis-indexed C."""
        shape = var.get("shape")
        if var["scope"] != "local" or not shape:
            return
        if not all(isinstance(d, int) for d in shape):
            raise ValueError(f"local-scope array {var['name']!r} must have fully static "
                             f"integer extents, got shape {shape!r}")
        shapes = self.ctx.setdefault(__LOCAL_ARRAY_SHAPES_CTX_KEY__, {})
        prev = shapes.get(var["name"])
        if prev is not None and prev != tuple(shape):
            raise ValueError(f"conflicting recorded shapes for local array {var['name']!r}: "
                             f"{prev!r} vs {tuple(shape)!r}")
        shapes[var["name"]] = tuple(shape)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        new_node = super().generic_visit(node)
        if __LITTLE_KERNEL_ENTRY__ in self.ctx and new_node.name == self.ctx[__LITTLE_KERNEL_ENTRY__].__name__:
            # Local-array shape metadata describes this lowering only. The
            # same LLKernel.ctx is reused across constexpr specializations
            # (e.g. widths 2, 4, 2), and a name reused with a different
            # trace-time extent is legitimate per specialization; keeping
            # prior specializations here made the third compile raise a
            # spurious conflicting-shape error. Start each entry lowering
            # with a fresh table; genuinely conflicting shapes *within* one
            # lowering are still rejected in _record_local_array_shape.
            self.ctx[__LOCAL_ARRAY_SHAPES_CTX_KEY__] = {}
            add_parent_references(new_node)
            mem_ana = MemoryAnalyzer(self.ctx)
            mem_ana.visit(new_node)
            # Early CPU resource legality: the sum of independent static
            # shared allocations must stay within ptxas's static .shared
            # limit; the launch dynamic-shared reservation is a separate
            # resource and never offsets this sum.
            mem_ana.audit_static_shared_memory()
            # add memory alloc for dynamic_shared memory
            alloc_dynamic_shared_memory_name = __INTERNAL_DYN_SHMEM__
            for scope in mem_ana.valid_scopes:
                # only insert alloc for shared_dynamic memory
                if scope != "dynamic_shared":
                    continue
                if mem_ana.first_mem_call[scope] is not None:
                    align_bytes = mem_ana.align_bytes[scope]
                    if align_bytes < 0:
                        # No ll.align_memory statement: the dynamic shared
                        # memory base is naturally 16-byte aligned. Keep this
                        # at 16: codegen_alloc_dynamic_shared_memory currently
                        # emits __align__(align_bytes) unconditionally, so a 0
                        # here would produce the invalid __align__(0) attribute.
                        align_bytes = 16
                    alloc_bytes = mem_ana.scope_offsets[scope]
                    # Runtime-sized (shape-expression) dynamic_shared
                    # allocations: serialized so the launch wrapper can
                    # host-evaluate the true reservation and legality checks.
                    # Static bytes above stay the only compile-time constant.
                    dyn_entries = [{
                        "elems_src": var["dyn_elems_src"],
                        # get_dtype_size returns bits/8 as float; the host
                        # legality schema requires integral element sizes.
                        "elem_size": int(var["element_size"]),
                        "max_bytes": var["total_bytes"],
                    }
                                   for var in mem_ana.variables.values()
                                   if var["scope"] == "dynamic_shared" and var.get("dyn_elems_src") is not None]
                    if dyn_entries:
                        aligned_args = [
                            ast.Constant(value=alloc_bytes),
                            ast.Constant(value=align_bytes),
                            ast.Name(id=alloc_dynamic_shared_memory_name, ctx=ast.Load()),
                            ast.Constant(value=json.dumps(dyn_entries)),
                        ]
                    else:
                        aligned_args = [
                            ast.Constant(value=alloc_bytes),
                            ast.Constant(value=align_bytes),
                            ast.Name(id=alloc_dynamic_shared_memory_name, ctx=ast.Load()),
                        ]
                    num_stmts = len(new_node.body)
                    for i in range(num_stmts):
                        if new_node.body[i] == mem_ana.first_mem_call[scope]:
                            alloc_stmt = ast.Expr(value=ast.Call(
                                func=ast.Name(id=self.alloc_dyn_name, ctx=ast.Load()),
                                args=aligned_args,
                                keywords=[],
                            ))
                            new_node.body.insert(i, alloc_stmt)
                            break
            # slice dynamic_shared memory
            num_stmts = len(new_node.body)
            for i in range(num_stmts):
                for var_name, var in mem_ana.variables.items():
                    target = _alloc_stmt_target(new_node.body[i])
                    is_target = target is not None and _target_matches_var(target, var_name)
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
                        self._record_local_array_shape(var)
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
                        if var.get("align"):
                            align_bytes = max(align_bytes, var["align"])
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
            target = _alloc_stmt_target(stmt)
            if target is not None:
                var_name = None
                if isinstance(target, ast.Name):
                    var_name = target.id
                elif (isinstance(target, ast.Subscript) and isinstance(target.slice, ast.Constant)):
                    var_name = f"{target.value.id}[{target.slice.value}]"
                if var_name and var_name in mem_ana.variables:
                    var = mem_ana.variables[var_name]
                    scope = var["scope"]
                    if scope == "local":
                        self._record_local_array_shape(var)
                    var_expr = _target_to_load_expr(var["expr"])
                    if scope == "local":
                        if var.get("fill_value") is not None:
                            # ll.full case - use alloc_local_full with fill_value
                            alloc_fn = alloc_local_full
                            obj_name = update_context(self.ctx, alloc_fn.__name__, alloc_fn)
                            body[i] = create_node_with_attrs(
                                ast.Expr, stmt, value=ast.Call(
                                    func=ast.Name(id=obj_name, ctx=ast.Load()),
                                    args=[
                                        var_expr,
                                        ast.Constant(value=var["dtype"]),
                                        ast.Constant(value=var["total_elements"]),
                                        ast.Constant(value=var["fill_value"]),
                                    ],
                                    keywords=[],
                                ))
                        else:
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
                        if var.get("align"):
                            align_bytes = max(align_bytes, var["align"])
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
