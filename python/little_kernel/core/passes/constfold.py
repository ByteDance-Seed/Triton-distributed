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
from enum import Enum
from typing import Dict, Any, Callable

from .pass_base import CompleteASTMutator
from .utils.preserve_attributes import create_node_with_attrs
from little_kernel.language.builtin_base import (
    BUILTIN_ATTR,
    CONST_FUNC_ATTR,
)
from ..type_system import LLType, ConstTypeHelper, ConstAnnotationType, const_annotation
from .utils.resolve_attribute import recursive_resolve_attribute
from .utils.type_inference import TypeInferencer
from .utils.error_report import raise_pass_error, get_source_from_ctx
from .utils.scope_manager import ScopeManager


class ConstantFolder(CompleteASTMutator):

    def __init__(self, ctx=None, all_types=None):
        super().__init__()
        self.ctx = ctx if ctx is not None else {}
        self.scope_manager = ScopeManager(self.ctx)
        self.defines = {}  # Global constants (from ctx) or folded local constants
        # self.kernel_defines was removed in favor of ScopeManager
        self.is_rvalue = False
        self.is_rvalue_stack = [False]
        self.in_kernel_function = False  # Track if we're inside a kernel function
        self.source_lines = None  # Cache source lines for error reporting
        # Map AST comparison operators to their corresponding Python operations
        self._op_mapping = {
            ast.Eq: lambda a, b: a == b,
            ast.NotEq: lambda a, b: a != b,
            ast.Lt: lambda a, b: a < b,
            ast.LtE: lambda a, b: a <= b,
            ast.Gt: lambda a, b: a > b,
            ast.GtE: lambda a, b: a >= b,
            ast.Is: lambda a, b: a is b,
            ast.IsNot: lambda a, b: a is not b,
            ast.In: lambda a, b: a in b,
            ast.NotIn: lambda a, b: a not in b,
        }
        self._op_handlers: Dict[type[ast.operator], Callable[[Any, Any], Any]] = {
            ast.Add: lambda a, b: a + b,  # a + b
            ast.Sub: lambda a, b: a - b,  # a - b
            ast.Mult: lambda a, b: a * b,  # a * b
            ast.Div: lambda a, b: a / b,  # a / b
            ast.FloorDiv: lambda a, b: a // b,  # a // b
            ast.Mod: lambda a, b: a % b,  # a % b
            ast.Pow: lambda a, b: a**b,  # a **b
            ast.LShift: lambda a, b: a << b,  # a << b
            ast.RShift: lambda a, b: a >> b,  # a >> b
            ast.BitOr: lambda a, b: a | b,  # a | b
            ast.BitXor: lambda a, b: a ^ b,  # a ^ b
            ast.BitAnd: lambda a, b: a & b,  # a & b
            ast.MatMult: lambda a, b: a @ b,  # a @ b (matrix multiplication)
        }
        self._bool_handlers: Dict[type[ast.operator], Callable[[Any, Any], bool]] = {
            ast.And: lambda *values: all(values),
            ast.Or: lambda *values: any(values),
        }
        self.all_types = all_types if all_types is not None else {}

    def switch_on_use(self):
        self.is_rvalue_stack.append(self.is_rvalue)
        self.is_rvalue = True

    def switch_off_use(self):
        assert len(self.is_rvalue_stack) > 0
        self.is_rvalue = self.is_rvalue_stack.pop()

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        """
        Fold Compare nodes into Constants if all operands are Constants.
        Example: 3 < 5 -> Constant(value=True)
        Example: 2 + 2 == 4 (after folding) -> Constant(value=True)
        """
        self.switch_on_use()
        # First, recursively process child nodes (in case they contain foldable expressions)
        processed_left = self.visit(node.left)
        processed_comparators = [self.visit(comp) for comp in node.comparators]
        processed_ops = node.ops  # Operators are not AST nodes to process
        self.switch_off_use()

        # Check if all operands are Constants after processing
        if not isinstance(processed_left, ast.Constant):
            # Left operand is not a constant → return original Compare (with processed children)
            new_compare = create_node_with_attrs(ast.Compare, node, left=processed_left, ops=processed_ops,
                                                 comparators=processed_comparators)
            return new_compare

        for comp in processed_comparators:
            if not isinstance(comp, ast.Constant):
                # At least one comparator is not a constant → return original Compare
                new_compare = create_node_with_attrs(ast.Compare, node, left=processed_left, ops=processed_ops,
                                                     comparators=processed_comparators)
                return new_compare

        # Extract constant values
        left_val = processed_left.value
        comparator_vals = [comp.value for comp in processed_comparators]

        # Evaluate chained comparisons (e.g., a < b < c → (a < b) and (b < c))
        try:
            result = True
            current_val = left_val
            for op, comp_val in zip(processed_ops, comparator_vals):
                # Get the operation function (e.g., Lt → lambda a,b: a < b)
                op_type = type(op)
                op_func = self._op_mapping.get(op_type)
                if not op_func:
                    # Unknown operator → cannot fold
                    raise_pass_error(f"Unsupported operator: {op_type.__name__}", node=node, ctx=self.ctx)

                # Evaluate current comparison step
                step_result = op_func(current_val, comp_val)
                if not step_result:
                    # Short-circuit: if any step fails, overall result is False
                    result = False
                    break

                # For chained comparisons, next step uses current comparator as left value
                current_val = comp_val

        except Exception as e:
            # Handle errors (e.g., incompatible types like 5 < "string")
            print(f"Warning: Could not fold comparison: {e}")
            new_compare = create_node_with_attrs(ast.Compare, node, left=processed_left, ops=processed_ops,
                                                 comparators=processed_comparators)
            return new_compare

        # Return folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original location info for consistency
        ast.copy_location(folded_constant, node)

        return folded_constant

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        """
        Fold BinOp nodes into Constants if both operands are Constants.
        Example: 2 + 3 → Constant(value=5)
        Example: (4 * 5) - 6 → Constant(value=14)
        """
        # First, recursively process left and right operands (handles nested BinOps)
        self.switch_on_use()
        processed_left = self.visit(node.left)
        processed_right = self.visit(node.right)
        self.switch_off_use()

        # Check if both operands are Constants after processing
        if not (isinstance(processed_left, ast.Constant) and isinstance(processed_right, ast.Constant)):
            # At least one operand is not a constant → return original BinOp (with processed children)
            new_binop = create_node_with_attrs(ast.BinOp, node, left=processed_left, op=node.op, right=processed_right)
            return new_binop

        # Extract constant values
        left_val = processed_left.value
        right_val = processed_right.value
        op_type = type(node.op)

        # Evaluate the binary operation
        try:
            # Get the handler function for this operator (e.g., Add → lambda a,b: a+b)
            op_handler = self._op_handlers.get(op_type)
            if not op_handler:
                raise_pass_error(f"Unsupported operator: {op_type.__name__}", node=node, ctx=self.ctx)

            # Compute the result
            result = op_handler(left_val, right_val)

        except Exception as e:
            # Handle errors (e.g., division by zero, incompatible types like "a" + 5)
            print(f"Warning: Could not fold BinOp {ast.unparse(node)}: {e}")
            new_binop = create_node_with_attrs(ast.BinOp, node, left=processed_left, op=node.op, right=processed_right)
            return new_binop

        # Return the folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original node's location (line/column numbers)
        ast.copy_location(folded_constant, node)
        return folded_constant

    def visit_BoolOp(self, node: ast.BoolOp) -> ast.AST:
        """
        Fold BoolOp nodes into Constants if both operands are Constants.
        Example: True and False → Constant(value=False)
        """
        # First, recursively process left and right operands (handles nested BinOps)
        self.switch_on_use()
        processed_values = [self.visit(v) for v in node.values]
        self.switch_off_use()

        # Check if all operands are Constants after processing
        for processed_v in processed_values:
            if not isinstance(processed_v, ast.Constant):
                # At least one operand is not a constant → return original BoolOp (with processed children)
                new_boolop = create_node_with_attrs(ast.BoolOp, node, op=node.op, values=processed_values)
                return new_boolop

        # Extract constant values
        const_values = [v.value for v in processed_values]
        op_type = type(node.op)

        # Evaluate the handler operation
        try:
            # Get the handler function for this operator
            op_handler = self._bool_handlers.get(op_type)
            if not op_handler:
                raise_pass_error(f"Unsupported operator: {op_type.__name__}", node=node, ctx=self.ctx)

            # Compute the result
            result = op_handler(*const_values)

        except Exception as e:
            # Handle errors (e.g., division by zero, incompatible types like "a" + 5)
            print(f"Warning: Could not fold BoolOp {ast.unparse(node)}: {e}")
            new_boolop = create_node_with_attrs(ast.BoolOp, node, op=node.op, values=processed_values)
            return new_boolop

        # Return the folded result as a Constant node
        folded_constant = ast.Constant(value=result)
        # Preserve original node's location (line/column numbers)
        ast.copy_location(folded_constant, node)
        return folded_constant

    def visit_IfExp(self, node: ast.IfExp) -> ast.AST:
        self.switch_on_use()
        new_test = self.visit(node.test)
        if isinstance(new_test, ast.Constant):
            if new_test.value:
                ret = self.visit(node.body)
                self.switch_off_use()
                return ret
            else:
                ret = self.visit(node.orelse)
                self.switch_off_use()
                return ret
        new_if = create_node_with_attrs(ast.IfExp, node, test=new_test, body=self.visit(node.body),
                                        orelse=self.visit(node.orelse))
        self.switch_off_use()
        return new_if

    def visit_AnnAssign(self, node: ast.AnnAssign) -> ast.AST:
        """
        Process annotated assignments (e.g., "x: const = 5" or "x: int32 = 5").
        """
        # Check if annotation is const (either ConstTypeHelper or ConstAnnotationType)
        is_const_annotation = False
        if node.annotation is not None:
            # Try to resolve the annotation
            try:
                resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx, self.scope_manager)
                # Check if it's const (ConstTypeHelper or ConstAnnotationType)
                if (isinstance(resolved_ann, ConstTypeHelper) or isinstance(resolved_ann, ConstAnnotationType)
                        or resolved_ann == const_annotation):
                    is_const_annotation = True
            except Exception:
                # If resolution fails, check for "const" as a Name node (fallback)
                if isinstance(node.annotation, ast.Name) and node.annotation.id == "const":
                    is_const_annotation = True
                elif isinstance(node.annotation, ast.Attribute) and node.annotation.attr == "const":
                    # Could be ll.const
                    try:
                        resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx, self.scope_manager)
                        if (isinstance(resolved_ann, ConstTypeHelper) or isinstance(resolved_ann, ConstAnnotationType)
                                or resolved_ann == const_annotation):
                            is_const_annotation = True
                    except Exception:
                        pass

        # Process value (with const folding enabled for const annotations)
        self.switch_on_use()
        new_value = self.visit(node.value) if node.value else None
        self.switch_off_use()

        # Process target
        new_target = self.visit(node.target)

        # If it's a const annotation, try to fold the value
        if is_const_annotation:
            # Keep trying to fold until we get a constant
            max_fold_iterations = 10
            for _ in range(max_fold_iterations):
                if isinstance(new_value, ast.Constant):
                    # Successfully folded to constant
                    if isinstance(new_target, ast.Name):
                        # Store in defines for later use
                        self.defines[new_target.id] = new_value.value
                    break
                elif isinstance(new_value, (ast.BinOp, ast.UnaryOp, ast.Compare)):
                    # Try to fold again
                    self.switch_on_use()
                    new_value = self.visit(new_value)
                    self.switch_off_use()
                else:
                    # Can't fold further
                    break

        # For const annotations that are successfully folded, we can convert to Assign
        # But for non-const annotations or if we want to preserve AnnAssign for codegen,
        # we should keep it as AnnAssign
        if is_const_annotation:
            if isinstance(new_value, ast.Constant):
                node_copy = self._copy_node(node)
                node_copy.target = new_target
                node_copy.value = new_value
                return node_copy
            else:
                raise_pass_error(f"Can't assign non-const value to const variable: {ast.unparse(node)}", node=node,
                                 ctx=self.ctx)
        else:
            # Non-const type annotation (e.g., "x: uint32 = 5")
            # Preserve the AnnAssign node for codegen to handle
            node_copy = self._copy_node(node)
            node_copy.target = new_target
            node_copy.value = new_value
            node_copy.annotation = node.annotation  # Keep the annotation
            return node_copy

    def visit_Assign(self, node):
        # Skip special struct declarations (e.g., Scheduler(...)) - they should not be const folded
        if hasattr(node, '_is_special_struct_decl') and node._is_special_struct_decl:
            # Preserve the node and its attributes completely
            # However, we still need to visit the constructor arguments to fold constants like Enum values
            # But we need to be careful not to modify the call_node structure itself
            node_copy = self._copy_node(node)
            node_copy.targets = self._process_list(node.targets)
            # Visit the value to fold constants in arguments (e.g., SchedulerGemmType.Normal)
            # This will fold Enum attributes and other constants in the constructor call
            node_copy.value = self.visit(node.value)
            return node_copy

        new_targets = self._process_list(node.targets)
        self.switch_on_use()
        new_value = self.visit(node.value)
        self.switch_off_use()
        reserved_targets = []
        reserved_values = []

        # Check if target has const annotation (e.g., "x: const = 5")
        # Note: This won't happen in regular Assign, only in AnnAssign
        is_const_annotation = False

        if isinstance(new_value, ast.Tuple):
            assert isinstance(new_targets[0], ast.Tuple), type(new_targets[0])
            for t, v in zip(new_targets[0].elts, new_value.elts):
                if not (isinstance(v, ast.Name) and v.id == t.id):
                    # avoid self-assignment
                    reserved_targets.append(t)
                    reserved_values.append(v)
                elif isinstance(v, ast.Constant) and not self.in_kernel_function:
                    # Only fold constants if not in kernel function (or if explicitly marked as const)
                    if is_const_annotation:
                        self.defines[t.id] = v.value
            if len(reserved_targets) == 0:
                return None
            reserved_targets = [ast.Tuple(elts=reserved_targets, ctx=new_targets[0].ctx)]
            reserved_values = ast.Tuple(elts=reserved_values, ctx=new_value.ctx)
        else:
            # Check for self-assignment only if target is a Name
            if isinstance(new_value, ast.Name) and isinstance(node.targets[0], ast.Name):
                if new_value.id == node.targets[0].id:
                    # avoid self-assignment
                    return None
            reserved_targets = new_targets
            reserved_values = new_value
            # Only fold constants if:
            # 1. Not in kernel function, OR
            # 2. Explicitly marked with const annotation
            if isinstance(reserved_values, ast.Constant):
                if not self.in_kernel_function:
                    # Global constant
                    if isinstance(reserved_targets[0], ast.Name):
                        self.defines[reserved_targets[0].id] = reserved_values.value
                else:
                    # Kernel variable - it is a local definition
                    # ScopeManager already tracks this, so we don't need kernel_defines logic.
                    pass

        new_assign = create_node_with_attrs(ast.Assign, node, targets=reserved_targets, value=reserved_values)
        return new_assign

    def visit_Attribute(self, node):

        def _recursive_visit_attribute(node):
            if isinstance(node, ast.Attribute):
                val = _recursive_visit_attribute(node.value)
                if isinstance(val, ast.AST):
                    return node
                # otherwise, try to get the attribute
                try:
                    attr = getattr(val, node.attr)
                    if hasattr(attr, BUILTIN_ATTR) and getattr(attr, BUILTIN_ATTR, False):  # builtin
                        # builtin function/class is reserved
                        return node
                    else:
                        return attr
                except AttributeError:
                    # Attribute not found, return node to preserve structure
                    return node
            if isinstance(node, ast.Name):
                # Check scope manager for local shadowing
                if self.scope_manager.is_local(node.id):
                    # It's local, so return AST node (don't resolve from ctx)
                    return node

                if node.id in self.ctx:
                    return self.ctx[node.id]
                # Also check in defines (for constants defined in the code)
                if node.id in self.defines:
                    return self.defines[node.id]
            else:
                return node

        attr = _recursive_visit_attribute(node)
        if isinstance(attr, ast.AST):
            return node
        # Handle Enum instances - they should be folded to ast.Constant
        if isinstance(attr, Enum):
            # Store the Enum instance in ast.Constant
            # The codegen will handle converting it to the appropriate C++ representation
            ret = ast.Constant(value=attr)
            ast.copy_location(ret, node)
            return ret
        # Handle LLType instances - preserve them as LLType objects, not strings
        # This is critical for type inference to work correctly
        if isinstance(attr, LLType):
            ret = ast.Constant(value=attr)
            ast.copy_location(ret, node)
            return ret
        # For other resolved values, convert to ast.Constant
        # But avoid converting LLType to string - this would lose type information
        ret = ast.Constant(value=attr)
        ast.copy_location(ret, node)
        return ret

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if self.is_rvalue:

            # Check scope manager first
            if self.scope_manager.is_local(node.id):
                # If it's local, we might have folded it (in self.defines)
                if node.id in self.defines:
                    value = self.defines[node.id]
                    if isinstance(value, (int, float, str, LLType)):
                        return ast.Constant(value=value)
                # Otherwise, it's a runtime local variable, return node
                return node

            # Not local, check globals
            if node.id in self.ctx:
                value = self.ctx[node.id]
                if isinstance(value, (int, float, str, LLType)):
                    return ast.Constant(value=value)
            elif node.id in self.defines:
                # Global constant defined in this module
                value = self.defines[node.id]
                if isinstance(value, (int, float, str, LLType)):
                    return ast.Constant(value=value)
        return node

    def visit_FunctionDef(self, node):
        # Check if this is a kernel function (has @ll_kernel decorator)
        is_kernel = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func = decorator.func
                if isinstance(func, ast.Name) and func.id == "ll_kernel":
                    is_kernel = True
                    break
            elif isinstance(decorator, ast.Name) and decorator.id == "ll_kernel":
                is_kernel = True
                break

        # Track if we're in a kernel function
        old_in_kernel = self.in_kernel_function
        if is_kernel:
            self.in_kernel_function = True

        # Scope management
        self.scope_manager.enter_scope()
        self.scope_manager.define_from_ast(node.args)
        # Scan body for locals (Python behavior: assignment anywhere makes it local)
        self.scope_manager.scan_for_locals(node.body)

        new_name = node.name  # Preserve unless mutated
        new_args = self.visit(node.args)
        new_body = self._process_list(node.body)
        new_decorator_list = self._process_list(node.decorator_list)
        self.switch_on_use()
        new_returns = self.visit(node.returns)
        self.switch_off_use()
        new_type_comment = node.type_comment  # Primitive, no mutation
        new_node = ast.FunctionDef(name=new_name, args=new_args, body=new_body, decorator_list=new_decorator_list,
                                   returns=new_returns, type_comment=new_type_comment)
        ast.copy_location(new_node, node)

        # Exit scope
        self.scope_manager.exit_scope()

        # Restore state
        if is_kernel:
            self.in_kernel_function = old_in_kernel

        return new_node

    def visit_Call(self, node):
        self.switch_on_use()
        # new_func = self.visit(node.func)
        new_args = self._process_list(node.args)
        new_kwargs = self._process_list(node.keywords)
        self.switch_off_use()

        value = recursive_resolve_attribute(node.func, self.ctx, self.scope_manager)
        if isinstance(value, Callable):
            if hasattr(value, CONST_FUNC_ATTR) and getattr(value, CONST_FUNC_ATTR, False):
                for arg in new_args:
                    assert isinstance(
                        arg, ast.Constant
                    ), f"Const function should only take Constant as arguments, but get {arg} {ast.unparse(arg)}"
                call_args = [arg.value for arg in new_args]
                kw_call_args = {}
                for kw in new_kwargs:
                    assert isinstance(kw, ast.keyword)
                    assert isinstance(
                        kw.value,
                        ast.Constant), f"Const function should only take Constant as kw arguments, but get {kw.value}"
                    kw_call_args[kw.arg] = kw.value.value
                ret = value(*call_args, **kw_call_args)
                return ast.Constant(value=ret)
            elif hasattr(value, "__const_type_func__") and value.__const_type_func__:
                assert node in self.all_types, f"Call node {ast.dump(node)} is not in all_types"
                ret_type = self.all_types[node]
                return ast.Constant(value=ret_type)
        # Preserve special attributes (e.g., _is_special_struct_method)
        new_call = create_node_with_attrs(ast.Call, node, func=node.func, args=new_args, keywords=new_kwargs)
        return new_call

    def visit_Assert(self, node):
        new_test = self.visit(node.test)
        new_msg = self.visit(node.msg) if node.msg else None

        # If the test is a constant (after folding), evaluate it at compile time
        if isinstance(new_test, ast.Constant):
            if not new_test.value:
                # Compile-time assert failed
                msg_str = ""
                if new_msg is not None:
                    if isinstance(new_msg, ast.Constant):
                        msg_str = new_msg.value
                    elif isinstance(new_msg, ast.JoinedStr):
                        # Try to evaluate the f-string at compile time
                        try:
                            # Build the string from parts
                            parts = []
                            for val in new_msg.values:
                                if isinstance(val, ast.Constant):
                                    parts.append(str(val.value))
                                elif isinstance(val, ast.FormattedValue):
                                    if isinstance(val.value, ast.Constant):
                                        parts.append(str(val.value.value))
                                    elif isinstance(val.value, ast.Name):
                                        # Check scope
                                        if self.scope_manager.is_local(val.value.id):
                                            pass  # Local
                                        elif val.value.id in self.ctx:
                                            parts.append(str(self.ctx[val.value.id]))
                                        elif val.value.id in self.defines:
                                            parts.append(str(self.defines[val.value.id]))
                                    elif isinstance(val.value, ast.Attribute):
                                        # Try to resolve attribute
                                        try:
                                            attr_val = recursive_resolve_attribute(val.value, self.ctx,
                                                                                   self.scope_manager)
                                            if not isinstance(attr_val, ast.AST):
                                                parts.append(str(attr_val))
                                        except Exception:
                                            pass
                            msg_str = "".join(parts)
                        except Exception:
                            msg_str = ast.unparse(new_msg)
                    else:
                        msg_str = ast.unparse(new_msg) if new_msg else ""

                raise_pass_error(f"Compile-time Assert Failed:\n{msg_str}", node=node, ctx=self.ctx,
                                 error_class=AssertionError)
            # Assert passed at compile time, remove it
            return None

        # Test is not a constant - keep as runtime assert (only allowed in kernel functions)
        if not self.in_kernel_function:
            raise_pass_error(
                f"assert outside kernel should be static (compile-time constant), "
                f"but the test results is {ast.dump(new_test)}", node=node, ctx=self.ctx, error_class=AssertionError)

        # In kernel function, keep the assert as-is (will be codegen'd as runtime assert)
        return ast.Assert(test=new_test, msg=new_msg)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node_copy = self._copy_node(node)
        node_copy.arg = node.arg  # String (parameter name)
        # keep annotation
        node_copy.annotation = self.visit(node.annotation) if node.annotation else None
        node_copy.type_comment = self.visit(node.type_comment)
        return node_copy

    def visit_Subscript(self, node):
        new_value = self.visit(node.value)
        new_slice = self.visit(node.slice)
        if isinstance(new_value, ast.Constant) and isinstance(new_slice, ast.Constant):
            return ast.Constant(value=new_value.value[new_slice.value])
        return ast.Subscript(value=new_value, slice=new_slice, ctx=node.ctx)


def const_fold(tree: ast.AST, ctx: Dict[str, Any] = None) -> ast.AST:
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    folder = ConstantFolder(ctx=ctx, all_types=all_types)
    # Try to get source lines for error reporting
    folder.source_lines = get_source_from_ctx(ctx)
    return ast.fix_missing_locations(folder.visit(tree))
