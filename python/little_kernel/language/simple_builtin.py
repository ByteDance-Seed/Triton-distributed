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
from little_kernel.language.builtin_base import builtin, Builtin


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


def _extract_asm_info(asm_call: ast.Call) -> Tuple[str, List[str], List[ast.AST], List[str]]:
    """
    Extract assembly template, constraints, operands, and optional clobbers from asm() call.
    
    Args:
        asm_call: AST node representing asm() call
    
    Returns:
        Tuple of (template, constraints, operands, clobbers)
    """
    if len(asm_call.args) < 3:
        raise ValueError("asm() call must have at least 3 arguments: (template, constraints, operands)")

    # Extract template (first arg, should be a string constant)
    # ast.Constant (Py3.8+) or ast.Str (Py3.7)
    template_node = asm_call.args[0]
    if isinstance(template_node, ast.Constant) and isinstance(template_node.value, str):
        template = template_node.value
    elif hasattr(ast, 'Str') and isinstance(template_node, ast.Str):
        template = template_node.s
    else:
        raise ValueError("asm() first argument must be a string constant (assembly template)")

    # Extract constraints (second arg, should be a list of strings)
    constraints_node = asm_call.args[1]
    if not isinstance(constraints_node, ast.List):
        raise ValueError("asm() second argument must be a list of constraint strings")

    def _str_val(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if hasattr(ast, 'Str') and isinstance(n, ast.Str):
            return n.s
        return None

    constraints = []
    for elt in constraints_node.elts:
        v = _str_val(elt)
        if v is None:
            raise ValueError("asm() constraints must be a list of string constants")
        constraints.append(v)

    # Extract operands (third arg): list, listcomp, or listcomp + list
    operands_node = asm_call.args[2]
    operands = _extract_operands_list(operands_node)

    # Optional 4th arg: clobbers (e.g. ["memory"]). It may be spelled
    # positionally (asm(..., ["memory"])) or by keyword
    # (asm(..., clobbers=["memory"])). The keyword form must not be dropped:
    # without it the generated body loses scheduler-visible barriers such as
    # "memory", letting the backend hoist memory operations across the asm.
    clobbers = []
    if len(asm_call.args) >= 4:
        clobbers.extend(_extract_clobbers_list(asm_call.args[3]))
    for kw in asm_call.keywords:
        if kw.arg == "clobbers":
            clobbers.extend(_extract_clobbers_list(kw.value))
            break

    return template, constraints, operands, clobbers


def _extract_clobbers_list(node: ast.AST) -> List[str]:
    """Extract string clobbers from a list literal node ([] on failure)."""
    clobbers = []
    if isinstance(node, ast.List):
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                clobbers.append(elt.value)
            elif hasattr(ast, 'Str') and isinstance(elt, ast.Str):
                clobbers.append(elt.s)
    return clobbers


def _extract_operands_list(node: ast.AST) -> List[ast.AST]:
    """Extract operands from list, listcomp, or BinOp(listcomp + list)."""
    if isinstance(node, ast.List):
        return node.elts
    if isinstance(node, ast.ListComp):
        operands = _unroll_listcomp_operands(node)
        if operands is None:
            raise ValueError("asm() listcomp operands must be [arr[i] for i in range(N)] "
                             "with constant N")
        return operands
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        # listcomp + list or list + list
        left = _extract_operands_list(node.left)
        right = _extract_operands_list(node.right)
        return left + right
    raise ValueError("asm() third argument must be a list or listcomp of operands")


def _unroll_listcomp_operands(node: ast.ListComp) -> Optional[List[ast.AST]]:
    """Unroll [arr[i] for i in range(N)] to [arr[0], arr[1], ..., arr[N-1]]."""
    if len(node.generators) != 1:
        return None
    gen = node.generators[0]
    if not isinstance(gen.iter, ast.Call):
        return None
    if not (isinstance(gen.iter.func, ast.Name) and gen.iter.func.id == "range"):
        return None
    if len(gen.iter.args) != 1:
        return None
    n_arg = gen.iter.args[0]
    n = None
    if isinstance(n_arg, ast.Constant) and isinstance(n_arg.value, int):
        n = n_arg.value
    elif hasattr(ast, 'Num') and isinstance(n_arg, ast.Num):
        n = n_arg.n if isinstance(getattr(n_arg, 'n', None), int) else None
    if n is None:
        return None
    if not isinstance(gen.target, ast.Name):
        return None
    idx_name = gen.target.id
    if not isinstance(node.elt, ast.Subscript):
        return None
    if not isinstance(node.elt.value, ast.Name):
        return None
    arr_name = node.elt.value.id
    slc = node.elt.slice
    if hasattr(ast, "Index") and isinstance(slc, ast.Index):
        slc = slc.value
    if not isinstance(slc, ast.Name) or slc.id != idx_name:
        return None
    # Build [arr[0], arr[1], ..., arr[n-1]]
    result = []
    for i in range(n):
        if hasattr(ast, 'Constant'):
            slice_val = ast.Constant(value=i)
        else:
            slice_val = ast.Index(ast.Num(i))
        result.append(ast.Subscript(
            value=ast.Name(id=arr_name, ctx=ast.Load()),
            slice=slice_val,
            ctx=ast.Load(),
        ))
    return result


def _apply_operand_transform(operand_cpp: str, constraint: str) -> str:
    """Apply operand transform from constraint suffix. E.g. 'r:cvta_shared' -> wrap."""
    if ":" not in constraint:
        return operand_cpp
    parts = constraint.split(":", 1)
    suffix = parts[1] if len(parts) > 1 else ""
    if suffix == "cvta_shared":
        return f"((uint32_t)__cvta_generic_to_shared({operand_cpp}))"
    if suffix == "as_u64":
        return f"((uint64_t)({operand_cpp}))"
    if suffix == "as_u64_addr":
        # TMA descriptor: need address for CUtensorMap_st (no direct uint64_t cast)
        return f"(reinterpret_cast<uint64_t>(static_cast<const void*>(&({operand_cpp}))))"
    return operand_cpp


def _operand_ast_to_cpp(operand_ast: ast.AST, param_map: dict, ret_var_name: Optional[str]) -> str:
    """Convert operand AST to C++ code string."""
    if isinstance(operand_ast, ast.Name):
        var_name = operand_ast.id
        if var_name in param_map:
            return param_map[var_name]
        elif var_name == ret_var_name:
            return ret_var_name
        return var_name
    elif isinstance(operand_ast, ast.Subscript):
        # arr[i] - value is array, slice is index
        if isinstance(operand_ast.value, ast.Name):
            base = operand_ast.value.id
            base_cpp = param_map.get(base, base)
            # slice: Index(Constant) in older Python, or directly Constant in 3.9+
            slc = operand_ast.slice
            if hasattr(ast, "Index") and isinstance(slc, ast.Index):
                slc = slc.value
            if isinstance(slc, ast.Constant):
                return f"{base_cpp}[{slc.value}]"
            if isinstance(slc, ast.Name):
                idx_cpp = param_map.get(slc.id, slc.id)
                return f"{base_cpp}[{idx_cpp}]"
        return "/* operand */"
    elif isinstance(operand_ast, ast.BinOp):
        left = _operand_ast_to_cpp(operand_ast.left, param_map, ret_var_name)
        right = _operand_ast_to_cpp(operand_ast.right, param_map, ret_var_name)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
        }
        op_str = op_map.get(type(operand_ast.op), "?")
        return f"({left} {op_str} {right})"
    elif isinstance(operand_ast, ast.Constant):
        v = operand_ast.value
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            return f"{v}" if v != 0.0 else "0.0"
        return str(v)
    return "/* operand */"


def _return_ast_to_cpp(node: ast.AST, param_map: dict) -> str:
    """Convert return expression AST to C++ code."""
    if isinstance(node, ast.Name):
        return param_map.get(node.id, node.id)
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        return str(v)
    if isinstance(node, ast.Compare):
        # e.g. ready != 0
        if len(node.ops) == 1 and len(node.comparators) == 1:
            left = _return_ast_to_cpp(node.left, param_map)
            right = _return_ast_to_cpp(node.comparators[0], param_map)
            op_map = {
                ast.Eq: "==",
                ast.NotEq: "!=",
                ast.Lt: "<",
                ast.LtE: "<=",
                ast.Gt: ">",
                ast.GtE: ">=",
            }
            op_str = op_map.get(type(node.ops[0]), "?")
            return f"({left} {op_str} {right})"
    if isinstance(node, ast.BinOp):
        left = _return_ast_to_cpp(node.left, param_map)
        right = _return_ast_to_cpp(node.right, param_map)
        op_map = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.Mod: "%",
            ast.BitAnd: "&",
            ast.BitOr: "|",
            ast.BitXor: "^",
        }
        op_str = op_map.get(type(node.op), "?")
        return f"({left} {op_str} {right})"
    return "?"


def _find_return_info(func_ast: ast.FunctionDef) -> Tuple[Optional[str], Optional[ast.AST]]:
    """Find return info: (ret_var_name, return_expr_ast).
    If return is simple Name (e.g. return ret), returns (ret, None).
    If return has expression (e.g. return ready != 0), returns (None, expr_ast).
    """
    for stmt in func_ast.body:
        if isinstance(stmt, ast.Return) and stmt.value is not None:
            if isinstance(stmt.value, ast.Name):
                return (stmt.value.id, None)
            return (None, stmt.value)
    return (None, None)


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
    from little_kernel.core.type_system import LLType, TensorType, TupleType

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

    if isinstance(ll_type, TupleType):
        elem_cpps = [_lltype_to_cpp_simple(t) for t in ll_type.element_types]
        # libcu++ tuple: usable in device code (std::tuple's members are
        # host-only under libstdc++/nvcc).
        return f"cuda::std::tuple<{', '.join(elem_cpps)}>"

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
            from little_kernel.codegen.policy import cpp_type
            return cpp_type(ll_type)
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


def _get_output_var_from_operands(constraints: List[str], operands_ast: List[ast.AST]) -> Optional[str]:
    """Get the name of the first output operand (constraint starts with '=')."""
    for i, c in enumerate(constraints):
        if c.startswith("=") and i < len(operands_ast):
            op = operands_ast[i]
            if isinstance(op, ast.Name):
                return op.id
    return None


def _resolve_ann_lltype(ann_ast: ast.AST, func: Callable):
    """Resolve a local variable annotation (e.g. ll.float32) to an LLType."""
    try:
        from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
        ctx = getattr(func, "__globals__", {}) or {}
        resolved = recursive_resolve_attribute(ann_ast, ctx)
        if isinstance(resolved, LLType):
            return resolved
    except Exception:
        pass
    return None


def _resolve_return_lltype(func: Callable, func_ast: ast.FunctionDef) -> Optional[LLType]:
    """Resolve the return annotation of a simple_builtin to an LLType.

    Supports scalar annotations (``-> ll.float32``) and tuple annotations
    (``-> ll.Tuple[ll.float32, ll.uint32, ...]``) for multi-value returns.
    Returns None when no annotation is present or it cannot be resolved.
    """
    from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
    from little_kernel.core.type_system import TupleType, TupleTypeHelper
    ctx = getattr(func, "__globals__", {}) or {}
    ann = func_ast.returns
    if ann is None:
        return None
    # Tuple annotation: ll.Tuple[T0, T1, ...]
    if isinstance(ann, ast.Subscript):
        try:
            base = recursive_resolve_attribute(ann.value, ctx)
        except Exception:
            base = None
        if isinstance(base, TupleTypeHelper):
            slc = ann.slice
            if hasattr(ast, "Index") and isinstance(slc, ast.Index):
                slc = slc.value
            elt_nodes = slc.elts if isinstance(slc, ast.Tuple) else [slc]
            elt_types = []
            for elt in elt_nodes:
                t = _resolve_ann_lltype(elt, func)
                if t is None:
                    raise ValueError(f"simple_builtin tuple return element annotation "
                                     f"does not resolve to an LLType: {ast.dump(elt)}")
                elt_types.append(t)
            return TupleType(tuple(elt_types))
        return None
    resolved = _resolve_ann_lltype(ann, func)
    return resolved


def _collect_local_decls(func_ast: ast.FunctionDef, func: Callable, param_names) -> List[Tuple[str, str]]:
    """Collect annotated local variable declarations from a simple_builtin body.

    Every ``name: ll.<dtype> = <constant>`` statement in the frontend source
    becomes a C++ local declaration with the matching scalar type. This makes
    inline-asm blocks with several output operands (e.g. a vectorized
    ``ld.global.nc.v4.f32`` writing four registers) expressible: previously
    only the single returned variable was declared.
    """
    decls = []
    seen = set(param_names)
    for stmt in func_ast.body:
        if not isinstance(stmt, ast.AnnAssign) or not isinstance(stmt.target, ast.Name):
            continue
        name = stmt.target.id
        if name in seen:
            continue
        seen.add(name)
        vtype = _resolve_ann_lltype(stmt.annotation, func)
        type_cpp = _lltype_to_cpp_simple(vtype) if vtype is not None else "auto"
        init = None
        if stmt.value is not None and isinstance(stmt.value, ast.Constant):
            value = stmt.value.value
            if isinstance(value, bool):
                init = "true" if value else "false"
            elif isinstance(value, int):
                init = str(value)
            elif isinstance(value, float):
                init = f"{value!r}" + ("f" if type_cpp == "float" else "")
        if init is None:
            if vtype is not None and getattr(vtype, "kind", None) == "float":
                init = "0.0f" if getattr(vtype, "fmt", None) == "float32" else "0.0"
            else:
                init = "0"
        decls.append((name, f"    {type_cpp} {name} = {init};"))
    return decls


def _generate_codegen_func(func: Callable, asm_template: str, constraints: List[str], operands_ast: List[ast.AST],
                           ret_var_name: Optional[str], return_expr_ast: Optional[ast.AST], clobbers: List[str],
                           func_ast: ast.FunctionDef, output_param_names: frozenset = frozenset()) -> Callable:
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
        # Prefer the AST annotation (handles ll.Tuple[...] subscripts and is
        # immune to `from __future__ import annotations` stringification).
        return_type = _resolve_return_lltype(func, func_ast)
    if return_type is None and func_ast.returns:
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
        args_cpp are the C++ code strings for function arguments (used in return_val/call).
        For the function BODY we use parameter names, not call-site args.
        """
        # For function body: use param names (body is the emitted definition)
        param_map = {p.name: p.name for p in params}

        # When return_expr_ast: need output var from first output operand
        output_var = ret_var_name
        if output_var is None and return_expr_ast is not None:
            output_var = _get_output_var_from_operands(constraints, operands_ast)
            if output_var:
                param_map[output_var] = output_var  # local var uses its own name

        # Generate function body
        body_lines = []

        # Declare all annotated locals from the frontend body first. Inline
        # asm with several output operands writes to several locals, so every
        # one of them must exist in the generated C++ scope.
        declared_locals = set()
        for local_name, decl_line in _collect_local_decls(func_ast, func, {p.name for p in params}):
            body_lines.append(decl_line)
            declared_locals.add(local_name)

        # Declare return/output variable if needed
        if output_var and output_var not in declared_locals:
            # Use return type if available, otherwise use auto
            # For return_expr (e.g. ready!=0), output var is often uint32
            var_type = return_type
            if return_expr_ast is not None:
                var_type = None  # infer from first output operand - use uint32 for pred
            ret_type_cpp = _lltype_to_cpp_simple(var_type) if var_type else "uint32_t"
            if var_type and hasattr(var_type, "kind"):
                if var_type.kind == "float":
                    init_value = "0.0f" if var_type.fmt == "float32" else "0.0"
                else:
                    init_value = "0"
            else:
                init_value = "0"
            body_lines.append(f"    {ret_type_cpp} {output_var} = {init_value};")

        # Convert operands to C++ code
        # For simple cases (Name nodes), use parameter map
        # For complex expressions, we'll need to generate C++ code
        operand_cpp = []
        for operand_ast in operands_ast:
            cpp = _operand_ast_to_cpp(operand_ast, param_map, output_var)
            operand_cpp.append(cpp)

        # Separate output and input constraints, apply operand transforms
        output_constraints = []
        input_constraints = []
        for i, constraint in enumerate(constraints):
            operand = operand_cpp[i] if i < len(operand_cpp) else "?"
            operand = _apply_operand_transform(operand, constraint)
            base_constraint = constraint.split(":")[0]  # "r:cvta_shared" -> "r"
            if base_constraint.startswith("="):
                output_constraints.append(f'"{base_constraint}"({operand})')
            else:
                input_constraints.append(f'"{base_constraint}"({operand})')

        # Build asm: asm volatile("..." : outputs : inputs [: clobbers]);
        # Escape template for C++: newlines->\n, backslash->\\, quote->\"
        asm_escaped = (asm_template.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n"))
        asm_lines = [f'    asm volatile("{asm_escaped}"']
        if output_constraints:
            asm_lines.append(f"        : {', '.join(output_constraints)}")
        else:
            asm_lines.append('        :')
        if input_constraints:
            asm_lines.append(f"        : {', '.join(input_constraints)}")
        else:
            asm_lines.append('        :')
        if clobbers:
            clobber_str = ", ".join(f'"{c}"' for c in clobbers)
            asm_lines.append(f"        : {clobber_str}")
        asm_lines.append("    );")

        body_lines.extend(asm_lines)

        # Return statement
        if isinstance(return_expr_ast, ast.Tuple):
            # Multi-value return: return a, b -> return cuda::std::make_tuple(a, b)
            ret_elts = [_return_ast_to_cpp(elt, param_map) for elt in return_expr_ast.elts]
            body_lines.append(f"    return cuda::std::make_tuple({', '.join(ret_elts)});")
        elif return_expr_ast is not None:
            ret_cpp = _return_ast_to_cpp(return_expr_ast, param_map)
            body_lines.append(f"    return {ret_cpp};")
        elif output_var:
            body_lines.append(f"    return {output_var};")

        # Generate function signature with parameter types
        # Use resolved types instead of auto
        param_decls = []
        for i, param in enumerate(params):
            param_type = param_types[i] if i < len(param_types) else None
            param_type_cpp = _lltype_to_cpp_simple(param_type) if param_type else "auto"
            if param.name in output_param_names:
                # Inline-asm output operand ("=r"/"=f"/...) names this scalar
                # parameter: it MUST lower as a C++ reference so the asm result
                # is written back into the caller's local instead of a dead
                # by-value copy (previously a silent miscompile). Without a
                # resolvable type we cannot form a valid reference, so fail
                # loudly rather than emitting `auto` (which would not compile,
                # or worse silently drop the output if it ever did).
                if param_type_cpp == "auto":
                    raise ValueError(f"simple_builtin '{func.__name__}': inline-asm output "
                                     f"parameter '{param.name}' needs a resolvable type "
                                     f"annotation to emit a reference (T&); the annotation "
                                     f"did not resolve to an LLType.")
                param_decls.append(f"{param_type_cpp}& {param.name}")
            else:
                param_decls.append(f"{param_type_cpp} {param.name}")

        func_body = "\n".join(body_lines)

        # Generate return type
        return_type_str = _lltype_to_cpp_simple(return_type) if return_type else "auto"
        if not output_var and not return_expr_ast:
            return_type_str = "void"

        signature = (f"static __device__ __forceinline__ {return_type_str} "
                     f"{func.__name__}({', '.join(param_decls)}) {{")
        if len(signature) > 100:
            signature = (f"static __device__ __forceinline__ {return_type_str} "
                         f"{func.__name__}(\n" + ",\n".join("    " + decl for decl in param_decls) + "\n) {")

        return Builtin(body=f"""
{signature}
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
        if return_annotation == inspect.Signature.empty and return_ast is None:
            return None

        # Prefer the AST annotation resolver (handles ll.Tuple[...] too)
        if return_ast is not None:
            resolved = _resolve_return_lltype(func, func_ast)
            if isinstance(resolved, LLType):
                return resolved

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

    Multi-output asm is supported: a scalar parameter named by an inline-asm
    output constraint ("=r"/"=f"/...) is emitted as a C++ reference (T&), so
    the asm result is written back into the caller's local. The call-site
    argument for such a parameter must be an lvalue (a named local).
    Example:

        @simple_builtin
        def ld_nc_f32x4(ptr: ll.Tensor[ll.float32], off: ll.int32,
                        r0: ll.uint32, r1: ll.uint32, r2: ll.uint32,
                        r3: ll.uint32) -> ll.void:
            ll.asm("ld.global.nc.v4.b32 {%0,%1,%2,%3}, [%4];",
                   ["=r", "=r", "=r", "=r", "l"], [r0, r1, r2, r3, ptr + off])
    
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
    asm_template, constraints, operands_ast, clobbers = _extract_asm_info(asm_call)

    # Multi-output inline asm support (requests: task 0002 cycle-003-kernel-00
    # R1b / cycle-004-kernel-02 R1; task 0000 cycle-005-kernel-02 R1 -- all
    # asking for the requesters' preferred option (a), "emit T& references for
    # = operands"). A scalar parameter named by an inline-asm OUTPUT constraint
    # ("=r"/"=f"/...) previously lowered BY VALUE, so the asm result was
    # written to a local copy and silently lost at the call site -- a
    # miscompile that passed compilation and only failed numbers. We now mark
    # those parameters so codegen emits them as C++ references (T&); the asm
    # then writes through to the caller's local. The call-site argument must
    # be an lvalue (a named frontend local), otherwise nvcc fails loudly --
    # never silently. Single output-then-return builtins and all-input
    # (store) builtins were already correct and are unchanged.
    _param_names = set(inspect.signature(func).parameters.keys())
    _output_param_names = set()
    for _i, _c in enumerate(constraints):
        if _c.split(":", 1)[0].startswith("=") and _i < len(operands_ast):
            _op = operands_ast[_i]
            if isinstance(_op, ast.Name) and _op.id in _param_names:
                _output_param_names.add(_op.id)

    # Find return info: simple ret var or return expression
    ret_var_name, return_expr_ast = _find_return_info(func_ast)

    # Generate codegen function
    codegen_func = _generate_codegen_func(func, asm_template, constraints, operands_ast, ret_var_name, return_expr_ast,
                                          clobbers, func_ast, output_param_names=frozenset(_output_param_names))

    # Generate eval_return_type
    eval_return_type = _generate_eval_return_type(func, func_ast)

    # Register as builtin
    decorated = builtin(eval_return_type=eval_return_type, codegen_func=codegen_func)(func)

    return decorated
