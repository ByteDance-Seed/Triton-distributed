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

################################################################################
#
# Copyright (c) 2025-2026 ByteDance Ltd. and/or its affiliates
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
# EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################
"""
Simple builtin decorator for defining builtin functions with inline assembly.

This module provides the `simple_builtin` decorator that allows users to define
builtin functions using a simplified syntax with inline assembly.

Example:
    @simple_builtin
    def load_shared(ptr: ll.Tensor[ll.uint64], offset: int) -> ll.uint64:
        ret: ll.uint64 = 0
        asm(
            "ld.shared.u64 %0, [%1];",
            ["=l", "l"],
            [ret, ptr + offset]
        )
        return ret
"""

import ast
import inspect
import textwrap
from typing import Callable, List, Optional, Tuple
from little_kernel.core.type_system import LLType
from .builtin_base import builtin, Builtin


def asm(asm_template: str, constraints: List[str], operands: List):
    """
    Placeholder function for inline assembly in simple_builtin functions.
    
    This function should never be called at runtime. It's only used as a marker
    in the function body to indicate inline assembly code.
    
    Args:
        asm_template: Assembly template string (e.g., "ld.shared.u64 %0, [%1];")
        constraints: List of constraint strings (e.g., ["=l", "l"])
        operands: List of operands (variables/expressions)
    
    Returns:
        This function should never return (raises RuntimeError if called).
    """
    raise RuntimeError("asm() should never be called at runtime. It's only used in simple_builtin functions.")


def _find_asm_call(node: ast.AST) -> Optional[ast.Call]:
    """Find the first asm() call in an AST node.
    
    Supports:
    - asm(...)
    - ll.asm(...)
    - ll.asm_volatile(...)
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # Check for asm() or ll.asm() or ll.asm_volatile()
            if isinstance(child.func, ast.Name) and child.func.id == "asm":
                return child
            elif isinstance(child.func, ast.Attribute):
                # Check for ll.asm, ll.asm_volatile, or similar
                if isinstance(child.func.value, ast.Name) and child.func.value.id in ["ll", "lasm"]:
                    if child.func.attr in ["asm", "asm_volatile"]:
                        return child
    return None


def _extract_asm_info(asm_call: ast.Call) -> Tuple[str, List[str], List[ast.AST]]:
    """
    Extract assembly template, constraints, and operands from an asm() call.
    
    Args:
        asm_call: AST node representing asm() call
    
    Returns:
        Tuple of (template, constraints, operands)
    """
    if len(asm_call.args) != 3:
        raise ValueError("asm() call must have exactly 3 arguments: (template, constraints, operands)")

    # Extract template (first arg, should be a string constant)
    template_node = asm_call.args[0]
    if not isinstance(template_node, ast.Constant) or not isinstance(template_node.value, str):
        raise ValueError("asm() first argument must be a string constant (assembly template)")
    template = template_node.value

    # Extract constraints (second arg, should be a list of strings)
    constraints_node = asm_call.args[1]
    if not isinstance(constraints_node, ast.List):
        raise ValueError("asm() second argument must be a list of constraint strings")
    constraints = []
    for elt in constraints_node.elts:
        if not isinstance(elt, ast.Constant) or not isinstance(elt.value, str):
            raise ValueError("asm() constraints must be a list of string constants")
        constraints.append(elt.value)

    # Extract operands (third arg, should be a list of expressions)
    operands_node = asm_call.args[2]
    if not isinstance(operands_node, ast.List):
        raise ValueError("asm() third argument must be a list of operands")
    operands = operands_node.elts

    return template, constraints, operands


def _find_return_variable(func_ast: ast.FunctionDef) -> Optional[str]:
    """Find the return variable name from function body."""
    # Look for pattern: ret: type = 0; ...; return ret
    ret_var = None
    for stmt in func_ast.body:
        if isinstance(stmt, ast.AnnAssign):
            # Check if this is a return variable declaration
            if isinstance(stmt.target, ast.Name):
                ret_var = stmt.target.id
        elif isinstance(stmt, ast.Return):
            if isinstance(stmt.value, ast.Name):
                # Return variable found
                return stmt.value.id
    return ret_var


def _generate_cpp_function_body(func_name: str, params: List[inspect.Parameter], return_type: Optional[LLType],
                                asm_template: str, constraints: List[str], operands: List[ast.AST],
                                ret_var_name: Optional[str]) -> str:
    """
    Generate C++ function body with inline assembly.
    
    Args:
        func_name: Function name
        params: Function parameters
        return_type: Return type (LLType)
        asm_template: Assembly template string
        constraints: List of constraint strings
        operands: List of operand AST nodes (for generating operand placeholders)
        ret_var_name: Name of return variable (if any)
    
    Returns:
        C++ function body as string
    """
    # Generate parameter declarations
    param_decls = []
    for param in params:
        # Convert parameter type annotation to C++ type
        # This is a simplified version - in practice, we'd need to resolve types properly
        param_type_str = "auto"  # Placeholder, will be resolved during codegen
        param_decls.append(f"{param_type_str} {param.name}")

    # Generate return type
    return_type_str = "auto"  # Placeholder, will be resolved during codegen
    if return_type is None:
        return_type_str = "void"

    # Generate function body
    body_lines = []

    # Declare return variable if needed
    if ret_var_name and return_type is not None:
        body_lines.append(f"    {return_type_str} {ret_var_name} = 0;")

    # Generate inline assembly
    # Format: asm volatile("template" : "output constraints" : "input constraints" : "clobbered");
    output_constraints = []
    input_constraints = []

    # Separate output and input constraints
    for i, constraint in enumerate(constraints):
        if constraint.startswith("="):
            # Output constraint
            output_constraints.append(f'"{constraint}"({operands[i] if i < len(operands) else "?"})')
        else:
            # Input constraint
            input_constraints.append(f'"{constraint}"({operands[i] if i < len(operands) else "?"})')

    # Build asm statement
    asm_parts = [f'asm volatile("{asm_template}"']
    if output_constraints:
        asm_parts.append(f"        : {', '.join(output_constraints)}")
    if input_constraints:
        asm_parts.append(f"        : {', '.join(input_constraints)}")
    asm_parts.append(");")

    body_lines.append("    " + "".join(asm_parts))

    # Return statement
    if ret_var_name:
        body_lines.append(f"    return {ret_var_name};")

    return "\n".join(body_lines)


def _lltype_to_cpp_simple(ll_type) -> str:
    """
    Simple LLType to C++ type conversion for simple_builtin.
    This is a simplified version that doesn't require a full emitter context.
    """
    import little_kernel.language as ll
    from little_kernel.core.type_system import LLType, TensorType

    if not isinstance(ll_type, LLType):
        # Try to resolve from annotation
        if isinstance(ll_type, type):
            # It's a type class, try to get an instance
            try:
                if hasattr(ll, ll_type.__name__):
                    ll_type = getattr(ll, ll_type.__name__)
            except Exception:
                pass

    if isinstance(ll_type, TensorType):
        elem_cpp = _lltype_to_cpp_simple(ll_type.element_type)
        return f"{elem_cpp}*"

    # Scalar types
    if hasattr(ll_type, "kind"):
        if ll_type.kind == "int":
            if ll_type.special == "binary" or ll_type.bits == 4:
                # TODO: Add proper 4-bit (int4/uint4) support. CUDA has native 4-bit types.
                raise NotImplementedError("4-bit integer types (int4/uint4, binary) are not yet supported. "
                                          "CUDA has native 4-bit support; this will be added in a future release.")
            sign_prefix = "u" if not ll_type.signed else ""
            bit_map = {
                8: f"{sign_prefix}int8_t", 16: f"{sign_prefix}int16_t", 32: f"{sign_prefix}int32_t", 64:
                f"{sign_prefix}int64_t", 128: "unsigned __int128" if not ll_type.signed else "__int128"
            }
            return bit_map.get(ll_type.bits, f"{sign_prefix}int32_t")
        elif ll_type.kind == "float":
            float_map = {
                "fp4_e2m1": "__nv_fp4_e2m1", "fp8_e5m2": "__nv_fp8_e5m2", "fp8_e4m3": "__nv_fp8_e4m3", "bfloat16":
                "__nv_bfloat16", "float16": "__half", "float32": "float", "float64": "double"
            }
            return float_map.get(ll_type.fmt, "float")
        elif ll_type.kind == "void":
            return "void"
        elif ll_type.kind == "str":
            return "std::string"

    # Fallback: try to convert from string representation
    type_str = str(ll_type)
    if "uint64" in type_str or "int64" in type_str:
        return "int64_t" if "int64" in type_str else "uint64_t"
    elif "uint32" in type_str or "int32" in type_str:
        return "int32_t" if "int32" in type_str else "uint32_t"
    elif "float32" in type_str:
        return "float"
    elif "float64" in type_str:
        return "double"

    # Last resort: use auto
    return "auto"


def _generate_codegen_func(func: Callable, asm_template: str, constraints: List[str], operands_ast: List[ast.AST],
                           ret_var_name: Optional[str], func_ast: ast.FunctionDef) -> Callable:
    """
    Generate a codegen function for the simple_builtin.
    
    The codegen function will be called during code generation with C++ code strings
    for the function arguments. It needs to generate a C++ function body that includes
    the inline assembly, with operands properly referencing the function parameters.
    
    Args:
        func: Original function
        asm_template: Assembly template
        constraints: Constraint strings
        operands_ast: Operand AST nodes
        ret_var_name: Return variable name
        func_ast: Function AST node
    
    Returns:
        Codegen function that generates Builtin object
    """
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    # Resolve parameter types from annotations
    param_types = []
    import little_kernel.core.type_system as ll_types
    import little_kernel.language as ll

    for param in params:
        if param.annotation != inspect.Parameter.empty:
            # Try to resolve annotation to LLType
            try:
                # Get the annotation value
                ann = param.annotation
                # If it's a string, try to resolve from context
                if isinstance(ann, str):
                    # Try to resolve from ll module
                    if hasattr(ll, ann):
                        resolved = getattr(ll, ann)
                        if isinstance(resolved, ll_types.LLType):
                            param_types.append(resolved)
                        else:
                            param_types.append(None)
                    else:
                        # Common Python type mappings
                        type_map = {
                            "int": ll.int32,
                            "float": ll.float32,
                            "bool": ll.bool_,
                        }
                        if ann in type_map:
                            param_types.append(type_map[ann])
                        else:
                            param_types.append(None)
                else:
                    # Try to use as-is if it's already an LLType
                    if isinstance(ann, ll_types.LLType):
                        param_types.append(ann)
                    elif isinstance(ann, type):
                        # Python builtin types
                        type_map = {
                            int: ll.int32,
                            float: ll.float32,
                            bool: ll.bool_,
                        }
                        if ann in type_map:
                            param_types.append(type_map[ann])
                        else:
                            param_types.append(None)
                    else:
                        param_types.append(None)
            except Exception:
                param_types.append(None)
        else:
            param_types.append(None)

    # Resolve return type
    return_type = None
    if func_ast.returns:
        try:
            import little_kernel.core.type_system as ll_types
            # Try to resolve return annotation
            return_annotation = sig.return_annotation
            if return_annotation != inspect.Signature.empty:
                if isinstance(return_annotation, ll_types.LLType):
                    return_type = return_annotation
                elif isinstance(return_annotation, type):
                    # Try to get from ll module
                    import little_kernel.language as ll
                    if hasattr(ll, return_annotation.__name__):
                        return_type = getattr(ll, return_annotation.__name__)
        except Exception:
            pass

    # Store operand AST nodes for later processing
    # We'll need to convert them to C++ code during codegen
    def codegen_func(*args_cpp, **kwargs_cpp):
        """
        Generated codegen function.
        args_cpp are the C++ code strings for function arguments.
        """
        # Map parameter names to their C++ code
        param_map = {}
        for i, param in enumerate(params):
            if i < len(args_cpp):
                param_map[param.name] = args_cpp[i]
            else:
                param_map[param.name] = f"/* {param.name} */"

        # Generate function body
        body_lines = []

        # Declare return variable if needed
        if ret_var_name:
            # Use return type if available, otherwise use auto
            ret_type_cpp = _lltype_to_cpp_simple(return_type) if return_type else "auto"
            # Initialize with appropriate zero value
            if return_type and hasattr(return_type, "kind"):
                if return_type.kind == "float":
                    init_value = "0.0f" if return_type.fmt == "float32" else "0.0"
                else:
                    init_value = "0"
            else:
                init_value = "0"
            body_lines.append(f"    {ret_type_cpp} {ret_var_name} = {init_value};")

        # Convert operands to C++ code
        # For simple cases (Name nodes), use parameter map
        # For complex expressions, we'll need to generate C++ code
        operand_cpp = []
        for operand_ast in operands_ast:
            if isinstance(operand_ast, ast.Name):
                # Direct variable reference - use parameter map or variable name
                var_name = operand_ast.id
                if var_name in param_map:
                    operand_cpp.append(param_map[var_name])
                elif var_name == ret_var_name:
                    operand_cpp.append(ret_var_name)
                else:
                    # Local variable - use as-is
                    operand_cpp.append(var_name)
            elif isinstance(operand_ast, ast.BinOp):
                # Binary operation - generate C++ code
                # This is simplified - in practice, we'd need a full expression codegen
                left = operand_ast.left.id if isinstance(operand_ast.left, ast.Name) else "?"
                right = operand_ast.right.id if isinstance(operand_ast.right, ast.Name) else "?"
                op_map = {
                    ast.Add: "+",
                    ast.Sub: "-",
                    ast.Mult: "*",
                    ast.Div: "/",
                    ast.Mod: "%",
                }
                op_str = op_map.get(type(operand_ast.op), "?")
                operand_cpp.append(f"({left} {op_str} {right})")
            else:
                # Fallback - use placeholder
                operand_cpp.append("/* operand */")

        # Separate output and input constraints
        output_constraints = []
        input_constraints = []
        for i, constraint in enumerate(constraints):
            operand = operand_cpp[i] if i < len(operand_cpp) else "?"
            if constraint.startswith("="):
                output_constraints.append(f'"{constraint}"({operand})')
            else:
                input_constraints.append(f'"{constraint}"({operand})')

        # Build asm statement
        asm_lines = [f'    asm volatile("{asm_template}"']
        if output_constraints:
            asm_lines.append(f"        : {', '.join(output_constraints)}")
        if input_constraints:
            asm_lines.append(f"        : {', '.join(input_constraints)}")
        asm_lines.append("    );")

        body_lines.extend(asm_lines)

        # Return statement
        if ret_var_name:
            body_lines.append(f"    return {ret_var_name};")

        # Generate function signature with parameter types
        # Use resolved types instead of auto
        param_decls = []
        for i, param in enumerate(params):
            param_type = param_types[i] if i < len(param_types) else None
            param_type_cpp = _lltype_to_cpp_simple(param_type) if param_type else "auto"
            param_decls.append(f"{param_type_cpp} {param.name}")

        func_body = "\n".join(body_lines)

        # Generate return type
        return_type_str = _lltype_to_cpp_simple(return_type) if return_type else "auto"
        if not ret_var_name:
            return_type_str = "void"

        return Builtin(
            body=f"""
__device__ {return_type_str} {func.__name__}({', '.join(param_decls)}) {{
{func_body}
}}
""", includes=[], return_val=f"{func.__name__}({', '.join(args_cpp)})")

    return codegen_func


def _generate_eval_return_type(func: Callable, func_ast: ast.FunctionDef) -> Callable:
    """
    Generate eval_return_type function from function signature.
    
    Args:
        func: Function with type annotations
        func_ast: Function AST node
    
    Returns:
        eval_return_type function
    """
    sig = inspect.signature(func)
    return_annotation = sig.return_annotation
    return_ast = func_ast.returns if func_ast.returns else None

    def eval_return_type(*arg_types, **kwarg_types):
        """Generated eval_return_type function."""
        # Try to resolve return type from annotation
        if return_annotation == inspect.Signature.empty:
            return None

        # If we have an AST node, try to resolve it using type inference
        if return_ast is not None:
            try:
                # Try to resolve using recursive_resolve_attribute
                from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
                # We need a context - use func.__globals__ as fallback
                ctx = func.__globals__ if hasattr(func, '__globals__') else {}
                resolved = recursive_resolve_attribute(return_ast, ctx)
                if isinstance(resolved, LLType):
                    return resolved
            except Exception:
                pass

        # Fallback: try to evaluate return annotation directly
        if isinstance(return_annotation, LLType):
            return return_annotation
        elif isinstance(return_annotation, type):
            # Try to instantiate if it's a type
            try:
                if issubclass(return_annotation, LLType):
                    # It's an LLType class, but we need an instance
                    # This won't work directly, need proper resolution
                    pass
            except Exception:
                pass

        # Last resort: try to get from context
        if isinstance(return_annotation, str):
            ctx = func.__globals__ if hasattr(func, '__globals__') else {}
            if return_annotation in ctx:
                resolved = ctx[return_annotation]
                if isinstance(resolved, LLType):
                    return resolved

        # If all else fails, raise an error
        raise ValueError(f"Cannot resolve return type from annotation: {return_annotation}. "
                         f"Please ensure the return type is a valid LLType or can be resolved from context.")

    return eval_return_type


def simple_builtin(func: Callable) -> Callable:
    """
    Decorator for defining builtin functions with inline assembly.
    
    The decorated function should:
    1. Have type annotations for all parameters and return type
    2. Contain exactly one asm() call in its body
    3. Follow the pattern: ret: type = 0; asm(...); return ret
    
    Example:
        @simple_builtin
        def load_shared(ptr: ll.Tensor[ll.uint64], offset: int) -> ll.uint64:
            ret: ll.uint64 = 0
            asm(
                "ld.shared.u64 %0, [%1];",
                ["=l", "l"],
                [ret, ptr + offset]
            )
            return ret
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function registered as builtin
    """
    # Parse function source to extract asm() call
    source = inspect.getsource(func)
    source = textwrap.dedent(source)
    func_ast = ast.parse(source)

    if len(func_ast.body) != 1 or not isinstance(func_ast.body[0], ast.FunctionDef):
        raise ValueError("simple_builtin can only be applied to function definitions")

    func_ast = func_ast.body[0]

    # Find asm() call
    asm_call = _find_asm_call(func_ast)
    if asm_call is None:
        # Debug: print all calls in the function
        all_calls = []
        for node in ast.walk(func_ast):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    all_calls.append(f"Name call: {node.func.id}")
                elif isinstance(node.func, ast.Attribute):
                    all_calls.append(f"Attribute call: {ast.dump(node.func)}")
                else:
                    all_calls.append(f"Other call: {ast.dump(node.func)}")
        body_dump = "\n".join([ast.dump(stmt, indent=2) for stmt in func_ast.body])
        raise ValueError(f"simple_builtin function must contain exactly one asm() call. "
                         f"Found calls: {all_calls}. "
                         f"Function body statements:\n{body_dump}")

    # Extract asm information
    asm_template, constraints, operands_ast = _extract_asm_info(asm_call)

    # Find return variable
    ret_var_name = _find_return_variable(func_ast)

    # Generate codegen function
    codegen_func = _generate_codegen_func(func, asm_template, constraints, operands_ast, ret_var_name, func_ast)

    # Generate eval_return_type
    eval_return_type = _generate_eval_return_type(func, func_ast)

    # Register as builtin
    decorated = builtin(eval_return_type=eval_return_type, codegen_func=codegen_func)(func)

    return decorated
