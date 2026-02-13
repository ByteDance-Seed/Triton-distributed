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
Generic code generators for special struct operations.

This module provides generic code generation functions for special struct
declarations, method calls, tuple unpacking, and struct definitions.
"""

import ast
import inspect
from little_kernel.core.passes.utils.type_inference import TypeInferencer
import little_kernel.language as ll
from .struct_converter import PythonClassToCppStructConverter
from .utils import _resolve_template_param


def generic_special_struct_declaration_codegen(node: ast.Assign, emitter) -> str:
    """
    Generic code generator for special struct declarations.
    
    Extracts template parameters and constructor arguments from special_struct_info
    and generates C++ struct declaration.
    """
    struct_info = node._special_struct_info
    struct_name = node._special_struct_name
    var_name = struct_info['var_name']

    # Get the class object to extract template parameters
    from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
    codegen_info = get_special_struct_codegen(struct_name)
    if not codegen_info or 'class' not in codegen_info:
        raise ValueError(f"Cannot find class object for special struct '{struct_name}'")

    class_obj = codegen_info['class']

    # Parse the class to get template parameters
    scheduler_file = inspect.getfile(class_obj)
    with open(scheduler_file, 'r') as f:
        source = f.read()
    class_ast = ast.parse(source)

    # Find the class and extract template parameters from __init__
    template_params = []
    constructor_args = []

    for node_item in ast.walk(class_ast):
        if isinstance(node_item, ast.ClassDef) and node_item.name == struct_name:
            for item in node_item.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    # Extract template parameters from __init__ arguments
                    for arg in item.args.args:
                        if arg.arg == 'self':
                            continue
                        if arg.annotation:
                            try:
                                from little_kernel.core.type_system import Template
                                type_inferencer = TypeInferencer(emitter.ctx if hasattr(emitter, 'ctx') else {}, {})
                                ann_type = type_inferencer._resolve_annotation(arg.annotation)
                                if isinstance(ann_type, Template):
                                    param_name = arg.arg
                                    if param_name in struct_info:
                                        # Pass emitter in ctx so _resolve_template_param can track enums
                                        resolve_ctx = dict(emitter.ctx) if hasattr(emitter, 'ctx') else {}
                                        resolve_ctx['emitter'] = emitter
                                        param_value = _resolve_template_param(struct_info[param_name], resolve_ctx)
                                        template_params.append(param_value)

                                        # Also track Enum types directly from ast.Constant
                                        if isinstance(struct_info[param_name], ast.Constant):
                                            enum_value = struct_info[param_name].value
                                            from enum import Enum
                                            if isinstance(enum_value, Enum):
                                                enum_class = type(enum_value)
                                                enum_name = enum_class.__name__
                                                if hasattr(emitter, 'used_enums'):
                                                    emitter.used_enums.add(enum_name)
                            except Exception:
                                pass
                    break

    # Extract constructor arguments (non-template parameters)
    # Get parameter order from __init__ signature
    param_order = []
    for node_item in ast.walk(class_ast):
        if isinstance(node_item, ast.ClassDef) and node_item.name == struct_name:
            for item in node_item.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for arg in item.args.args:
                        if arg.arg == 'self':
                            continue
                        # Check if it's a template parameter
                        is_template = False
                        if arg.annotation:
                            try:
                                from little_kernel.core.type_system import Template
                                type_inferencer = TypeInferencer(emitter.ctx if hasattr(emitter, 'ctx') else {}, {})
                                ann_type = type_inferencer._resolve_annotation(arg.annotation)
                                if isinstance(ann_type, Template):
                                    is_template = True
                            except Exception:
                                pass
                        if not is_template:
                            param_order.append(arg.arg)
                    break
            break

    # Extract constructor arguments in the correct order
    for param_name in param_order:
        if param_name in struct_info:
            value = struct_info[param_name]
            # Skip special keys
            if param_name in ['var_name', 'struct_name', 'method_calls', 'ast_node']:
                continue
            # This is a constructor argument
            if isinstance(value, ast.AST):
                arg_cpp = emitter.visit(value)
            elif isinstance(value, str):
                arg_cpp = value
            else:
                raise ValueError(
                    f"Expected AST node for constructor argument '{param_name}', but got {type(value).__name__}: {value}."
                )
            constructor_args.append(arg_cpp)

    # Generate template struct declaration
    if template_params:
        template_str = ", ".join(template_params)
        constructor_str = ", ".join(constructor_args)
        return f"{struct_name}<{template_str}> {var_name}({constructor_str});"
    else:
        constructor_str = ", ".join(constructor_args)
        return f"{struct_name} {var_name}({constructor_str});"


def generic_special_struct_method_codegen(node: ast.Call, emitter) -> str:
    """Generic code generator for special struct method calls."""
    var_name = node._special_struct_var
    method_name = node._method_name

    # Generate method arguments - convert keyword arguments to positional based on method signature
    # Get the method signature to determine parameter order
    from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen

    # Get struct name from node or emitter context
    struct_name = getattr(node, '_special_struct_name', None)
    if not struct_name:
        # Try to get from emitter's special_struct_vars
        if hasattr(emitter, 'scope_vars') and var_name in emitter.scope_vars:
            var_type = emitter.scope_vars[var_name]
            from little_kernel.core.type_system import SpecialStructType
            if isinstance(var_type, SpecialStructType):
                struct_name = var_type.struct_name

    method_args_cpp = []

    if struct_name:
        codegen_info = get_special_struct_codegen(struct_name)
        if codegen_info and 'class' in codegen_info:
            class_obj = codegen_info['class']
            if hasattr(class_obj, method_name):
                import inspect
                try:
                    method = getattr(class_obj, method_name)
                    sig = inspect.signature(method)
                    param_names = [p for p in sig.parameters.keys() if p != 'self']

                    # Build keyword map
                    keyword_map = {kw.arg: kw.value for kw in node.keywords if kw.arg}

                    # Map all parameters in order, using positional args first, then keywords, then defaults
                    for i, param_name in enumerate(param_names):
                        if i < len(node.args):
                            # Use positional argument
                            method_args_cpp.append(emitter.visit(node.args[i]))
                        elif param_name in keyword_map:
                            # Use keyword argument
                            method_args_cpp.append(emitter.visit(keyword_map[param_name]))
                        elif param_name in sig.parameters:
                            param = sig.parameters[param_name]
                            if param.default != inspect.Parameter.empty:
                                # Use default value
                                default_value = param.default
                                # Handle enum types
                                from enum import Enum
                                if isinstance(default_value, Enum):
                                    # Convert enum to C++ enum value (e.g., IndexType.MN -> IndexType::MN)
                                    enum_class = type(default_value)
                                    enum_name = enum_class.__name__
                                    enum_member = default_value.name
                                    # Track that this enum is used
                                    if hasattr(emitter, 'used_enums'):
                                        emitter.used_enums.add(enum_name)
                                    method_args_cpp.append(f"{enum_name}::{enum_member}")
                                else:
                                    default_ast = ast.Constant(value=default_value)
                                    method_args_cpp.append(emitter.visit(default_ast))
                            else:
                                # Required parameter missing - this is an error
                                raise ValueError(
                                    f"Missing required parameter '{param_name}' for method '{method_name}'")
                except Exception as e:
                    # Don't silently fallback - raise the error so we can debug
                    import traceback
                    raise RuntimeError(f"Failed to process keyword arguments for {struct_name}.{method_name}: {e}\n"
                                       f"Traceback: {traceback.format_exc()}") from e
            else:
                raise ValueError(f"Method '{method_name}' not found in {struct_name}")
        else:
            raise ValueError(f"Special struct '{struct_name}' not found in registry")
    else:
        raise ValueError(f"Special struct name not found for method call {method_name}")

    # Check return type to determine if we need reference parameters
    reference_params = getattr(node, '_reference_params', None)

    # For methods with multiple return values, declare and add reference parameters
    if reference_params:
        from little_kernel.codegen.special_struct.struct_converter import PythonClassToCppStructConverter
        converter = PythonClassToCppStructConverter("", ctx=emitter.ctx if hasattr(emitter, 'ctx') else {})
        for param_info in reference_params:
            param_name = param_info['name']
            param_type = param_info['type']
            # Declare variable if not already in scope
            if param_name not in emitter.scope_vars:
                param_cpp_type = converter._lltype_to_cpp_type(param_type)
                # Initialize based on type
                if param_cpp_type == "bool":
                    init_value = "false"
                elif param_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                    init_value = "0"
                else:
                    init_value = "{}"  # Default initialization
                emitter.writeln_main(f"{param_cpp_type} {param_name} = {init_value};")
                emitter.scope_vars[param_name] = param_type
            # Add as reference parameter
            method_args_cpp.append(param_name)

    # For single return value, it's returned directly, not as reference parameter
    # Regular method call - tuple unpacking is handled separately
    return f"{var_name}.{method_name}({', '.join(method_args_cpp)})"


def generic_special_struct_tuple_unpack_codegen(target: ast.Tuple, value_node: ast.Call, emitter) -> None:
    """
    Generic code generator for tuple unpacking from special struct method calls.
    
    Uses type inference to determine return structure and generates appropriate C++ code.
    """
    var_name = value_node._special_struct_var
    method_name = value_node._method_name

    # Extract target variable names
    target_names = []
    for t in target.elts:
        if isinstance(t, ast.Name):
            target_names.append(t.id)
        else:
            raise NotImplementedError(f"Tuple unpacking target must be variable names, got {type(t).__name__}")

    if not target_names:
        raise ValueError("Empty tuple unpacking target")

    # Get return type from type inference
    from little_kernel.core.type_system import TupleType

    # Try to get return type from all_types
    return_type = None
    if hasattr(emitter, 'all_types') and value_node in emitter.all_types:
        return_type = emitter.all_types[value_node]

    # If not found, try to infer from the method
    if return_type is None and hasattr(emitter, 'all_types'):
        if isinstance(value_node.func, ast.Attribute):
            if isinstance(value_node.func.value, ast.Name):
                var_name_in_ast = value_node.func.value.id
                if var_name_in_ast in emitter.scope_vars:
                    var_type = emitter.scope_vars[var_name_in_ast]
                    from little_kernel.core.type_system import SpecialStructType
                    if isinstance(var_type, SpecialStructType):
                        from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
                        codegen_info = get_special_struct_codegen(var_type.struct_name)
                        if codegen_info and 'class' in codegen_info and codegen_info['class'] is not None:
                            class_obj = codegen_info['class']
                            if hasattr(class_obj, method_name):
                                method = getattr(class_obj, method_name)
                                if hasattr(method, '__annotations__') and 'return' in method.__annotations__:
                                    inferencer = TypeInferencer(emitter.ctx if hasattr(emitter, 'ctx') else {}, {})
                                    try:
                                        return_type = inferencer._resolve_annotation(
                                            ast.parse(f"x: {method.__annotations__['return']}", mode='eval').body)
                                    except Exception:
                                        pass

    # Generate method arguments
    args_cpp = [emitter.visit(arg) for arg in value_node.args]

    # All targets are reference parameters, function returns void
    # Declare variables if needed
    # For tuple returns with multiple elements:
    # - First element is returned directly
    # - Other elements are reference parameters
    call_args = list(args_cpp)

    if isinstance(return_type, TupleType) and len(return_type.element_types) > 1:
        # Multiple return values: first is returned, others are reference parameters
        # Declare and initialize reference parameters (skip first element which is returned)
        ref_param_names = []
        # target_names[0] gets the return value, target_names[1:] are reference parameters
        # return_type.element_types[0] is returned, return_type.element_types[1:] are reference parameters
        for idx in range(1, len(target_names)):
            ref_name = target_names[idx]
            # Map to return_type index: target_names[1] -> return_type.element_types[1], etc.
            return_type_idx = idx  # target_names[0] -> return_type[0] (returned), target_names[1] -> return_type[1] (ref param)

            # This is a reference parameter
            if ref_name not in emitter.scope_vars:
                # Get type from return_type (index return_type_idx)
                if return_type_idx < len(return_type.element_types):
                    element_type = return_type.element_types[return_type_idx]
                    converter = PythonClassToCppStructConverter("", ctx=emitter.ctx if hasattr(emitter, 'ctx') else {})
                    var_cpp_type = converter._lltype_to_cpp_type(element_type)
                    var_lltype = element_type
                else:
                    var_cpp_type = "int32_t"
                    var_lltype = ll.int32

                # Initialize based on type
                if var_cpp_type == "bool":
                    init_value = "false"
                elif var_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                    init_value = "0"
                else:
                    init_value = "{}"
                emitter.writeln_main(f"{var_cpp_type} {ref_name} = {init_value};")
                emitter.scope_vars[ref_name] = var_lltype

            ref_param_names.append(ref_name)

        # Add reference parameters to call args
        call_args.extend(ref_param_names)

        # If there are more return elements than targets, add dummy variables for the rest
        if len(return_type.element_types) > len(target_names) + 1:
            remaining = len(return_type.element_types) - len(target_names) - 1
            for i in range(remaining):
                dummy_name = f"_unused_ret_{method_name}_{i}"
                if dummy_name not in emitter.scope_vars:
                    actual_idx = len(target_names) + 1 + i
                    if actual_idx < len(return_type.element_types):
                        element_type = return_type.element_types[actual_idx]
                        converter = PythonClassToCppStructConverter("",
                                                                    ctx=emitter.ctx if hasattr(emitter, 'ctx') else {})
                        var_cpp_type = converter._lltype_to_cpp_type(element_type)
                        # Initialize based on type
                        if var_cpp_type == "bool":
                            init_value = "false"
                        elif var_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                            init_value = "0"
                        else:
                            init_value = "{}"
                        emitter.writeln_main(f"{var_cpp_type} {dummy_name} = {init_value};")
                        emitter.scope_vars[dummy_name] = element_type
                call_args.append(dummy_name)

        # Assign first return value to first target variable
        first_target = target_names[0] if target_names else None
        if first_target:
            # Declare first target if needed
            if first_target not in emitter.scope_vars:
                first_elem_type = return_type.element_types[0]
                converter = PythonClassToCppStructConverter("", ctx=emitter.ctx if hasattr(emitter, 'ctx') else {})
                var_cpp_type = converter._lltype_to_cpp_type(first_elem_type)
                if var_cpp_type == "bool":
                    init_value = "false"
                elif var_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                    init_value = "0"
                else:
                    init_value = "{}"
                emitter.writeln_main(f"{var_cpp_type} {first_target} = {init_value};")
                emitter.scope_vars[first_target] = first_elem_type

            # Call method and assign first return value
            emitter.writeln_main(f"{first_target} = {var_name}.{method_name}({', '.join(call_args)});")
        else:
            # No first target - just call the method
            emitter.writeln_main(f"{var_name}.{method_name}({', '.join(call_args)});")
    else:
        # Single return value or single element tuple - handled differently
        # This case should not happen in tuple unpacking, but handle it anyway
        # Declare all targets as reference parameters
        for ref_name in target_names:
            if ref_name not in emitter.scope_vars:
                var_cpp_type = "int32_t"  # Default
                var_lltype = ll.int32
                if isinstance(return_type, TupleType) and len(return_type.element_types) > 0:
                    element_type = return_type.element_types[0]
                    converter = PythonClassToCppStructConverter("", ctx=emitter.ctx if hasattr(emitter, 'ctx') else {})
                    var_cpp_type = converter._lltype_to_cpp_type(element_type)
                    var_lltype = element_type
                emitter.writeln_main(f"{var_cpp_type} {ref_name};")
                emitter.scope_vars[ref_name] = var_lltype
        call_args.extend(target_names)
        emitter.writeln_main(f"{var_name}.{method_name}({', '.join(call_args)});")


def generic_special_struct_definition_codegen(struct_name: str, emitter) -> str:
    """
    Generic code generator for special struct definitions.
    
    Generates C++ struct definition by automatically converting the Python class to C++ struct.
    Template parameters are automatically extracted from type annotations.
    """
    from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen

    # Get the class object
    codegen_info = get_special_struct_codegen(struct_name)
    if not codegen_info or 'class' not in codegen_info:
        raise ValueError(f"Cannot find class object for special struct '{struct_name}'")

    class_obj = codegen_info['class']

    # Get class module path
    class_file = inspect.getfile(class_obj)

    # Parse class file and convert class to struct
    with open(class_file, 'r') as f:
        source = f.read()
    class_ast = ast.parse(source)

    # Get context from emitter for type inference
    ctx = emitter.ctx if hasattr(emitter, 'ctx') else {}

    # Convert class to C++ struct
    converter = PythonClassToCppStructConverter(struct_name, template_params=None, ctx=ctx)
    converter.visit(class_ast)

    return converter.code.getvalue()
