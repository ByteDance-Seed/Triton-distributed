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
from typing import Dict, List, Any
import little_kernel.language as ll
from .utils.extract_empty_params import extract_empty_params
from .utils.add_parent_reference import add_parent_references
from .utils.update_context import update_context
from .utils.resolve_attribute import recursive_resolve_attribute
from .pass_base import CompleteASTMutator
from .utils.preserve_attributes import create_node_with_attrs
from .constfold import const_fold


class FlattenEmpty(CompleteASTMutator):
    """
    Transforms list comprehensions of `ll.empty` calls into explicit tensor array initializations.
    Converts:
    A_smem_tensors = [ll.empty(shape, dtype, scope) for _ in range(8)]
    Into:
    A_smem_tensors = ll.empty([8], dtype=ll.Tensor[dtype], scope="local")
    A_smem_tensors[0] = ll.empty(shape, dtype, scope)
    A_smem_tensors[1] = ll.empty(shape, dtype, scope)
    ...
    A_smem_tensors[7] = ll.empty(shape, dtype, scope)
    """

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx  # Function's global context for evaluating constants/variables
        # self.ll_module = None  # Will hold the actual ll module from context
        # self._resolve_ll_module()
        self.ll_module_name = "flatten_empty_ll"
        update_context(self.ctx, self.ll_module_name, ll)

    # def _resolve_ll_module(self) -> None:
    #     """Resolve the actual ll module from the context (handles aliases)"""
    #     for name, obj in self.ctx.items():
    #         if hasattr(obj, 'empty') and hasattr(obj, 'Tensor'):
    #             self.ll_module = obj
    #             return
    #     raise ValueError("Could not find 'll' module with 'empty' and 'Tensor' in context")

    def _evaluate_expr(self, expr: ast.AST) -> Any:
        """Evaluate an AST expression using the function's global context"""
        try:
            compiled = compile(ast.fix_missing_locations(ast.Module(body=[ast.Expr(value=expr)], type_ignores=[])),
                               filename="<expr_evaluation>", mode="exec")
            eval_namespace = self.ctx.copy()
            exec(compiled, eval_namespace)
            return eval_namespace["__return__"]
        except Exception as e:
            raise RuntimeError(f"Failed to evaluate expression: {ast.dump(expr, indent=2)}\nError: {str(e)}")

    def _create_tensor_dtype_ast(self, elem_dtype_expr: ast.AST) -> ast.Attribute:
        """Create AST node for ll.Tensor[elem_dtype]"""
        # Get the alias used for ll (e.g., 'll' or 'lk') from context
        tensor = ast.Attribute(value=ast.Name(id=self.ll_module_name, ctx=ast.Load()), attr="Tensor", ctx=ast.Load())
        slice = ast.Subscript(value=tensor, slice=elem_dtype_expr, ctx=ast.Load())
        return slice

    def visit_ListComp(self, node: ast.ListComp) -> Any:
        """Visit list comprehensions and transform if they match the ll.empty pattern"""
        # First, recursively process children (for nested structures)
        self.generic_visit(node)

        # Check if parent is a single-target assignment
        if not (isinstance(node.parent, ast.Assign) and len(node.parent.targets) == 1
                and isinstance(node.parent.targets[0], ast.Name)):
            return node  # Not an assignment we care about

        target_var = node.parent.targets[0].id  # Variable name (e.g., A_smem_tensors)

        # Check if list comprehension element is an ll.empty call
        if not isinstance(node.elt, ast.Call):
            return node  # Not an ll.empty call

        func = recursive_resolve_attribute(node.elt.func, self.ctx)
        if func.__name__ != ll.empty.__name__:
            return node

        # Extract parameters from the ll.empty call
        try:
            shape_expr, dtype_expr, scope_expr = extract_empty_params(node.elt)
        except ValueError as e:
            raise RuntimeError(f"Invalid ll.empty call in list comprehension: {str(e)}")

        # Analyze the loop generator (must be range-based)
        if len(node.generators) != 1:
            return node  # Only support single generator
        gen = node.generators[0]

        # Check if iterator is range()
        if not (isinstance(gen.iter, ast.Call) and isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range"):
            return node  # Only support range() iterators

        # Evaluate range parameters to get element count.
        # Skip transformation if args are not constants or evaluable (e.g. range(BLOCK_M // 16)
        # that wasn't folded); avoids AssertionError and keeps compiler robust.
        range_values = []
        for arg in gen.iter.args:
            if isinstance(arg, ast.Constant):
                range_values.append(arg.value)
            else:
                try:
                    val = self._evaluate_expr(arg)
                    range_values.append(int(val))
                except Exception:
                    return node  # Cannot evaluate - skip this optimization
        if len(range_values) == 1:
            count = range_values[0]
        elif len(range_values) == 2:
            count = range_values[1] - range_values[0]
        else:
            return node  # Only support range(N) or range(start, end)

        if count <= 0:
            return node  # Invalid range - skip optimization

        # Create replacement nodes
        new_nodes: List[ast.AST] = []

        # 1. Create array definition: target_var = ll.empty([count], dtype=ll.Tensor[elem_dtype], scope="local")
        array_shape = ast.List(elts=[ast.Constant(value=count)], ctx=ast.Load())
        array_dtype = self._create_tensor_dtype_ast(dtype_expr)
        array_scope = ast.Constant(value="local")

        # Use self.ll_module_name (injected into ctx) - node.elt.func may be ast.Attribute (ll.empty)
        # or ast.Name (empty from "from ... import empty"), and Name has no .value
        ll_ref = ast.Name(id=self.ll_module_name, ctx=ast.Load())
        array_empty_call = ast.Call(
            func=ast.Attribute(value=ll_ref, attr="empty", ctx=ast.Load()), args=[array_shape],
            keywords=[ast.keyword(arg="dtype", value=array_dtype),
                      ast.keyword(arg="scope", value=array_scope)])

        # Create array definition assignment, preserving attributes from parent Assign node
        array_def_assign = create_node_with_attrs(ast.Assign, node.parent,
                                                  targets=[ast.Name(id=target_var,
                                                                    ctx=ast.Store())], value=array_empty_call)
        new_nodes.append(array_def_assign)

        # 2. Create element initializations: target_var[i] = ll.empty(...)
        for i in range(count):
            # Create index node: target_var[i]
            index = ast.Subscript(value=ast.Name(id=target_var, ctx=ast.Load()), slice=ast.Constant(value=i),
                                  ctx=ast.Store())

            # Create ll.empty call (reuse original parameters)
            elem_empty_call = ast.Call(
                func=ast.Attribute(value=ll_ref, attr="empty", ctx=ast.Load()),
                args=[shape_expr],  # Reuse original shape
                keywords=[
                    ast.keyword(arg="dtype", value=dtype_expr),  # Reuse original dtype
                    ast.keyword(arg="scope", value=scope_expr)  # Reuse original scope
                ])

            # Create assignment node, preserving attributes from parent Assign node
            elem_assign = create_node_with_attrs(ast.Assign, node.parent, targets=[index], value=elem_empty_call)
            new_nodes.append(elem_assign)

        # Return the new nodes to replace the original ListComp assignment
        return new_nodes

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.targets = self._process_list(node.targets)
        node_copy.value = self.visit(node.value)
        if isinstance(node.value, ast.ListComp) and not isinstance(node_copy.value, ast.ListComp):
            return node_copy.value
        return ast.fix_missing_locations(node_copy)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.name = node.name  # Preserve unless mutated
        node_copy.args = self.visit(node.args)
        node_copy.body = self._process_list(node.body)
        node_copy.decorator_list = self._process_list(node.decorator_list)
        node_copy.returns = self.visit(node.returns)
        node_copy.type_comment = node.type_comment  # Primitive, no mutation
        return ast.fix_missing_locations(node_copy)


def flatten_empty(tree: ast.AST, ctx: Dict[str, Any]) -> ast.AST:
    """
    Flatten list comprehensions of ll.empty calls into explicit tensor array initializations.
    """
    # Add parent references to AST nodes
    add_parent_references(tree)

    transformer = FlattenEmpty(ctx)
    tree = transformer.visit(tree)
    return const_fold(tree, ctx)
