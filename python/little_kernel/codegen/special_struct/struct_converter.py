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
Python class to C++ struct converter.

This module converts Python class AST to C++ struct definition.
"""

import ast
from typing import Dict, Any, Optional, List, Tuple
from io import StringIO
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
import little_kernel.language as ll
from little_kernel.codegen.registries.type_converter import TypeConverter
from .translator import PythonToCppTranslator


class PythonClassToCppStructConverter(ast.NodeVisitor):
    """
    Convert Python class AST to C++ struct definition.
    
    This visitor traverses a Python class definition and generates
    the corresponding C++ struct with member variables and methods.
    """

    def __init__(self, class_name: str, template_params: Optional[List[str]] = None, ctx: Optional[Dict[str,
                                                                                                        Any]] = None):
        self.class_name = class_name
        self.template_params = template_params or []
        self.ctx = ctx or {}
        self.code = StringIO()
        self.indent_level = 0
        self.indent_unit = "    "
        self.member_vars: List[Tuple[str, str, Optional[str]]] = []  # (name, type, default_value)
        self.methods: List[ast.FunctionDef] = []
        self.helper_functions: List[ast.FunctionDef] = []
        self.enums: List[ast.ClassDef] = []  # Enum classes in the module
        # Type inferencer for analyzing return types
        self.type_inferencer = TypeInferencer(self.ctx, {})
        # Type converter for LLType to C++ conversion
        self.type_converter = TypeConverter()
        # Map from parameter name to template parameter name (e.g., 'block_m' -> 'BLOCK_M')
        self.template_param_map: Dict[str, str] = {}
        # Track which methods are static (have @staticmethod decorator)
        self.static_methods: set = set()

    def _writeln(self, line: str = ""):
        """Write a line with proper indentation."""
        if line:
            self.code.write(" " * (self.indent_level * len(self.indent_unit)) + line + "\n")
        else:
            self.code.write("\n")

    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition and extract members and methods."""
        if node.name != self.class_name:
            return

        # Track which methods are static and their visibility
        self.static_methods = set()
        self.private_methods = set()
        self.public_methods = set()

        # First pass: collect member variables from __init__ and identify methods
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                # Check for decorators
                is_static = False
                is_private = False
                is_public = False

                for decorator in item.decorator_list:
                    # Check for @staticmethod
                    if isinstance(decorator, ast.Name) and decorator.id == "staticmethod":
                        is_static = True
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "staticmethod":
                        is_static = True
                    # Check for @private decorator (if exists)
                    elif isinstance(decorator, ast.Name) and decorator.id == "private":
                        is_private = True
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "private":
                        is_private = True
                    # Check for @public decorator (if exists)
                    elif isinstance(decorator, ast.Name) and decorator.id == "public":
                        is_public = True
                    elif isinstance(decorator, ast.Attribute) and decorator.attr == "public":
                        is_public = True

                if is_static:
                    self.static_methods.add(item.name)

                # Determine visibility based on decorator or naming convention
                if item.name == "__init__":
                    # __init__ is always public (used for constructor)
                    self._extract_member_vars_from_init(item)
                    self.methods.append(item)
                elif is_public:
                    # Explicitly marked as public
                    self.methods.append(item)
                elif is_private or item.name.startswith("_"):
                    # Explicitly marked as private OR starts with underscore (Python convention)
                    self.helper_functions.append(item)
                    self.private_methods.add(item.name)
                else:
                    # No decorator and doesn't start with _ = public
                    self.methods.append(item)
                    self.public_methods.add(item.name)

        # Generate C++ code
        self._generate_cpp_struct()

    def _extract_member_vars_from_init(self, init_method: ast.FunctionDef):
        """Extract member variables and template parameters from __init__ method."""
        # First, extract template parameters from __init__ parameters
        template_params_list = []
        for arg in init_method.args.args:
            if arg.arg == 'self':
                continue
            # Check if parameter has template annotation
            if arg.annotation:
                try:
                    from little_kernel.core.type_system import Template
                    ann_type = self.type_inferencer._resolve_annotation(arg.annotation)
                    if isinstance(ann_type, Template):
                        # This is a template parameter
                        inner_type = ann_type.inner_type
                        # Try to get the actual type name from the annotation AST
                        cpp_type = None
                        if isinstance(arg.annotation, ast.Subscript) and isinstance(arg.annotation.slice, ast.Name):
                            # template[GemmType] -> GemmType
                            enum_name = arg.annotation.slice.id
                            # Check if it's an Enum class
                            try:
                                resolved_enum = recursive_resolve_attribute(arg.annotation.slice, self.ctx)
                                if isinstance(resolved_enum, ast.Constant):
                                    resolved_enum = resolved_enum.value
                                from enum import Enum
                                if isinstance(resolved_enum, type) and issubclass(resolved_enum, Enum):
                                    cpp_type = enum_name  # Use enum class name directly
                            except Exception:
                                pass

                        if cpp_type is None:
                            cpp_type = self._lltype_to_cpp_type(inner_type)

                        # Convert parameter name to template parameter name
                        template_name = arg.arg.upper() if arg.arg.islower() else arg.arg
                        if not template_name.startswith('k'):
                            template_name = f"k{template_name[0].upper()}{template_name[1:]}" if template_name[
                                0].islower() else template_name
                        template_params_list.append((arg.arg, template_name, cpp_type))
                except Exception:
                    pass

        # Update template_params with extracted template parameters
        if template_params_list:
            self.template_params = [
                f"{cpp_type} {template_name}" for _, template_name, cpp_type in template_params_list
            ]
            # Build mapping from parameter name to template parameter name
            self.template_param_map = {
                param_name: template_name
                for param_name, template_name, _ in template_params_list
            }

        # Build a map from parameter name to parameter annotation for type lookup
        param_type_map = {}
        for arg in init_method.args.args:
            if arg.arg == 'self':
                continue
            if arg.annotation:
                try:
                    ann_type = self.type_inferencer._resolve_annotation(arg.annotation)
                    param_type_map[arg.arg] = ann_type
                except Exception:
                    pass

        # Extract member variables from self.xxx assignments
        for stmt in init_method.body:
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                            # Found self.xxx = ...
                            var_name = target.attr

                            # Check if this variable was marked as template parameter
                            is_template = any(param_name == var_name for param_name, _, _ in template_params_list)

                            if not is_template:
                                # Regular member variable
                                # Check if the value is a parameter name - if so, use parameter's type annotation
                                var_type = None
                                if isinstance(stmt.value, ast.Name):
                                    param_name = stmt.value.id
                                    if param_name in param_type_map:
                                        # Use the parameter's type annotation
                                        param_lltype = param_type_map[param_name]
                                        var_type = self._lltype_to_cpp_type(param_lltype)

                                # If not found from parameter type, infer from value
                                if var_type is None:
                                    var_type = self._infer_cpp_type(stmt.value)

                                default_value = self._get_default_value(stmt.value)
                                self.member_vars.append((var_name, var_type, default_value))

    def _infer_cpp_type(self, value_node: ast.AST) -> str:
        """Infer C++ type from Python value node."""
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return "bool"
            elif isinstance(value_node.value, int):
                # Use int32_t for signed integers (default), uint32_t only for explicitly unsigned
                return "int32_t" if value_node.value < 0 else "int32_t"
            elif isinstance(value_node.value, float):
                return "float"
        elif isinstance(value_node, ast.Name):
            if value_node.id == "True" or value_node.id == "False":
                return "bool"
        elif isinstance(value_node, ast.Call):
            # Check if this is a type constructor call (ll.uint32(value), ll.int32(value), etc.)
            try:
                from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
                from little_kernel.core.type_system import LLType
                func_resolved = recursive_resolve_attribute(value_node.func, self.ctx)
                if isinstance(func_resolved, LLType):
                    # Type constructor call: ll.uint32(value) returns uint32_t
                    return self._lltype_to_cpp_type(func_resolved)
            except Exception:
                pass

            # Try to infer return type from function call using type inferencer
            if self.type_inferencer:
                try:
                    ll_type = self.type_inferencer.infer_node_type(value_node)
                    return self._lltype_to_cpp_type(ll_type)
                except Exception:
                    pass
            # Fallback: try to resolve function and check if it's a builtin with eval_return_type
            try:
                from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
                from little_kernel.language.builtin_base import EVAL_RETURN_TYPE_ATTR
                func_obj = recursive_resolve_attribute(value_node.func, self.ctx)
                if callable(func_obj) and hasattr(func_obj, EVAL_RETURN_TYPE_ATTR):
                    eval_return_type = getattr(func_obj, EVAL_RETURN_TYPE_ATTR)
                    # Infer argument types
                    arg_types = []
                    for arg in value_node.args:
                        try:
                            if self.type_inferencer:
                                arg_ll_type = self.type_inferencer.infer_node_type(arg)
                                arg_types.append(arg_ll_type)
                            else:
                                arg_types.append(None)
                        except Exception:
                            arg_types.append(None)
                    # Call eval_return_type to get return type
                    return_type = eval_return_type(*arg_types)
                    return self._lltype_to_cpp_type(return_type)
            except Exception:
                pass
        # Check for pointer types (grouped_layout)
        if hasattr(value_node, 'id') and 'layout' in str(value_node):
            return "int32_t*"
        return "int32_t"  # Default to signed integer

    def _get_default_value(self, value_node: ast.AST) -> Optional[str]:
        """Get default value string for C++ initialization."""
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return "true" if value_node.value else "false"
            elif isinstance(value_node.value, int):
                return str(value_node.value)
            elif isinstance(value_node.value, float):
                return str(value_node.value)
        elif isinstance(value_node, ast.NameConstant):  # Python < 3.8
            if value_node.value is True:
                return "true"
            elif value_node.value is False:
                return "false"
            elif value_node.value is None:
                return None
        elif isinstance(value_node, ast.UnaryOp) and isinstance(value_node.op, ast.USub):
            if isinstance(value_node.operand, ast.Constant):
                return f"-{value_node.operand.value}"
        return None

    def _generate_cpp_struct(self):
        """Generate the complete C++ struct definition."""
        # Generate struct template
        if self.template_params:
            self._writeln("template <" + ", ".join(self.template_params) + ">")
        self._writeln(f"struct {self.class_name} {{")
        self.indent_level += 1

        # Generate member variables
        for var_name, var_type, default_value in self.member_vars:
            if default_value is not None:
                self._writeln(f"{var_type} {var_name} = {default_value};")
            else:
                self._writeln(f"{var_type} {var_name};")

        if self.member_vars:
            self._writeln()

        # Generate methods
        for method in self.methods:
            self._generate_method(method)
            self._writeln()

        # Generate helper methods (private methods)
        if self.helper_functions:
            self._writeln("private:")
            for method in self.helper_functions:
                self._generate_method(method, is_private=False)
                self._writeln()

        self.indent_level -= 1
        self._writeln("};")

    def _generate_method(self, method_ast: ast.FunctionDef, is_private: bool = False):
        """Generate C++ method from Python method AST."""
        # Generate method signature
        method_name = method_ast.name
        if method_name == "__init__":
            method_name = self.class_name  # Constructor

        # Analyze return value
        return_info = self._analyze_return_value(method_ast)

        # Determine return type based on return value count
        # For constructor, there's no return type
        is_constructor = (method_name == self.class_name)
        if is_constructor:
            return_type = ""
        elif return_info:
            if return_info['is_tuple']:
                tuple_types = return_info['tuple_types']
                # If tuple has multiple elements, first one is returned, others are reference parameters
                if len(tuple_types) > 1:
                    return_type = tuple_types[0]  # First element is returned
                else:
                    # Single element tuple - return it directly
                    return_type = tuple_types[0]
            else:
                # Single return value - return it directly
                return_type = return_info['return_type']
        else:
            return_type = "void"

        # Generate parameters
        params = []
        for arg in method_ast.args.args:
            if arg.arg == 'self':
                continue

            # Skip template parameters in constructor
            if is_constructor:
                if arg.annotation:
                    try:
                        from little_kernel.core.type_system import Template
                        ann_type = self.type_inferencer._resolve_annotation(arg.annotation)
                        if isinstance(ann_type, Template):
                            continue
                    except Exception:
                        pass

            param_type = "int32_t"  # Default to signed integer
            if arg.annotation:
                if isinstance(arg.annotation, ast.Name):
                    if arg.annotation.id == "bool":
                        param_type = "bool"
                    elif arg.annotation.id == "int":
                        param_type = "int32_t"  # Default int is signed
                    else:
                        # Check if it's an enum type
                        enum_found = False
                        try:
                            resolved = recursive_resolve_attribute(arg.annotation, self.ctx)
                            from enum import Enum
                            if isinstance(resolved, type) and issubclass(resolved, Enum):
                                param_type = resolved.__name__  # Use enum class name directly
                                enum_found = True
                        except Exception:
                            pass
                        # If not found via resolve, check if it's a registered enum
                        if not enum_found:
                            from little_kernel.codegen.registries.enum_registry import get_all_registered_enums
                            registered_enums = get_all_registered_enums()
                            if arg.annotation.id in registered_enums:
                                param_type = arg.annotation.id  # Use enum class name directly
                            # Also check by class name
                            for reg_name, reg_class in registered_enums.items():
                                if reg_class.__name__ == arg.annotation.id:
                                    param_type = reg_name  # Use registered name
                                    break
                else:
                    # Try to infer type from annotation
                    try:
                        # First check if annotation is an enum type directly
                        resolved_ann = recursive_resolve_attribute(arg.annotation, self.ctx)
                        from enum import Enum
                        if isinstance(resolved_ann, type) and issubclass(resolved_ann, Enum):
                            param_type = resolved_ann.__name__  # Use enum class name directly
                        else:
                            ann_type = self.type_inferencer._resolve_annotation(arg.annotation)
                            from little_kernel.core.type_system import Template
                            if not isinstance(ann_type, Template):
                                param_type = self._lltype_to_cpp_type(ann_type)
                    except Exception:
                        pass
            # Check for reference parameters (mutable)
            param_name = arg.arg
            if method_name != self.class_name:  # Not constructor
                # Check if parameter is modified in method body
                if self._is_parameter_modified(method_ast, arg.arg):
                    param_type = f"{param_type}&"
            params.append(f"{param_type} {param_name}")

        # Add return values as reference parameters (only for tuple returns with multiple elements)
        # For single return value, it's returned directly, not as reference parameter
        if return_info and return_info['is_tuple']:
            tuple_types = return_info['tuple_types']
            tuple_names = return_info.get('tuple_names', [])
            # If tuple has multiple elements, first one is returned, others are reference parameters
            if len(tuple_types) > 1:
                for i in range(1, len(tuple_types)):  # Skip first element (it's returned)
                    ret_type = tuple_types[i]
                    param_name = tuple_names[i] if i < len(tuple_names) else f"ret_{i}"
                    params.append(f"{ret_type}& {param_name}")

        # Check if this is a static method
        is_static = method_name in self.static_methods

        # Generate method signature with __device__ modifier
        qualifiers = "__device__ __forceinline__"
        if is_static:
            qualifiers = "static " + qualifiers
        const_qualifier = " const" if self._is_const_method(method_ast) else ""
        # For constructor, no return type; for other methods, include return type
        if is_constructor:
            self._writeln(f"{qualifiers} {method_name}({', '.join(params)}){const_qualifier} {{")
        else:
            self._writeln(f"{qualifiers} {return_type} {method_name}({', '.join(params)}){const_qualifier} {{")

        # Generate method body
        self.indent_level += 1
        # For static methods, don't use "this->" since there's no instance.
        # For constructor AND regular methods, use "this->" to avoid shadowing when
        # parameter names match member names (e.g. self.attr = attr -> this->attr = attr).
        struct_var_name = ""
        if not is_static:
            struct_var_name = "this->"

        # Apply constfold and inline passes to method body before translation
        # Create a temporary module AST with the method body for passes
        method_body_statements = []
        for stmt in method_ast.body:
            # Skip docstrings
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and isinstance(
                    stmt.value.value, str):
                continue
            method_body_statements.append(stmt)

        # Create a temporary module for passes
        temp_module = ast.Module(body=method_body_statements, type_ignores=[])
        ast.fix_missing_locations(temp_module)

        # Ensure intrin functions are in ctx for passes
        pass_ctx = dict(self.ctx)  # Copy ctx
        # Add all registered intrin functions to ctx
        try:
            from little_kernel.language.builtin_base import get_intrin_ctx
            intrin_ctx = get_intrin_ctx()
            for name, func in intrin_ctx.items():
                if name not in pass_ctx:
                    pass_ctx[name] = func
        except Exception:
            # If getting intrin ctx fails, continue without intrin functions
            pass

        # Apply constfold pass
        from little_kernel.core.passes.constfold import const_fold
        try:
            temp_module = const_fold(temp_module, ctx=pass_ctx)
        except Exception:
            # If constfold fails, continue with original body
            # Debug: print error but don't fail
            # print(f"Warning: constfold failed for {method_name}: {e}\n{traceback.format_exc()}")
            pass

        # Apply inline pass
        from little_kernel.core.passes.inline import inline
        try:
            temp_module = inline(temp_module, ctx=pass_ctx)
        except Exception:
            # If inline fails, continue with original body
            # Debug: print error but don't fail
            # print(f"Warning: inline failed for {method_name}: {e}\n{traceback.format_exc()}")
            pass

        # Ensure intrin functions are in ctx for translator
        translator_ctx = dict(self.ctx)  # Copy ctx
        # Add all registered intrin functions to ctx
        try:
            from little_kernel.language.builtin_base import get_intrin_ctx
            intrin_ctx = get_intrin_ctx()
            for name, func in intrin_ctx.items():
                if name not in translator_ctx:
                    translator_ctx[name] = func
        except Exception:
            # If getting intrin ctx fails, continue without intrin functions
            pass

        translator = PythonToCppTranslator(struct_var_name=struct_var_name, return_info=return_info,
                                           type_inferencer=self.type_inferencer, ctx=translator_ctx,
                                           class_name=self.class_name, static_methods=self.static_methods)
        # Pass template parameter mapping to translator
        if hasattr(self, 'template_param_map'):
            translator.template_param_map = self.template_param_map

        # Add method parameters to scope_vars to avoid redeclaration
        for arg in method_ast.args.args:
            if arg.arg == 'self':
                continue
            # Skip template parameters in constructor
            if is_constructor:
                if arg.annotation:
                    try:
                        from little_kernel.core.type_system import Template
                        ann_type = self.type_inferencer._resolve_annotation(arg.annotation)
                        if isinstance(ann_type, Template):
                            continue
                    except Exception:
                        pass
            # Infer parameter type
            param_lltype = ll.int32  # Default
            if arg.annotation:
                try:
                    param_lltype = self.type_inferencer._resolve_annotation(arg.annotation)
                    from little_kernel.core.type_system import Template
                    if isinstance(param_lltype, Template):
                        continue  # Skip template parameters
                except Exception:
                    pass
            translator.scope_vars[arg.arg] = param_lltype

        # Add return value reference parameters to scope_vars
        if return_info and return_info['is_tuple']:
            tuple_types = return_info['tuple_types']
            tuple_names = return_info.get('tuple_names', [])
            if len(tuple_types) > 1:
                for i in range(1, len(tuple_types)):  # Skip first element (it's returned)
                    param_name = tuple_names[i] if i < len(tuple_names) else f"ret_{i}"
                    param_lltype = tuple_types[i]
                    translator.scope_vars[param_name] = param_lltype

        # Translate processed body
        for stmt in temp_module.body:
            translator.visit(stmt)
        # Copy translated code
        translated_lines = translator.code.getvalue().split('\n')
        for line in translated_lines:
            if line.strip():
                self._writeln(line)
        self.indent_level -= 1
        self._writeln("}")

    def _analyze_return_value(self, method_ast: ast.FunctionDef) -> Optional[Dict[str, Any]]:
        """Analyze return statements to determine return value structure."""
        return_statements = []
        for stmt in ast.walk(method_ast):
            if isinstance(stmt, ast.Return) and stmt.value:
                return_statements.append(stmt.value)

        if not return_statements:
            return None

        # Analyze first return statement
        ret_value = return_statements[0]

        # Check if method has return type annotation
        annotation_return_info = None
        if method_ast.returns:
            try:
                annotation_return_info = self._analyze_return_from_annotation(method_ast.returns, ret_value,
                                                                              method_ast.name)
            except Exception:
                pass

        if isinstance(ret_value, ast.Tuple):
            # Return tuple - try to infer types from actual return values
            tuple_types = []
            tuple_names = []
            actual_return_types_inferred = True

            # First, try to infer types from actual return values
            for i, elt in enumerate(ret_value.elts):
                try:
                    ll_type = self.type_inferencer.infer_node_type(elt)
                    cpp_type = self._lltype_to_cpp_type(ll_type)
                    tuple_types.append(cpp_type)
                except Exception:
                    # If type inference fails, try to infer from the expression itself
                    # For Name nodes, they're likely int32_t (default for variables)
                    if isinstance(elt, ast.Name):
                        # Variable names - default to int32_t
                        tuple_types.append("int32_t")
                    elif isinstance(elt, (ast.BinOp, ast.UnaryOp)):
                        # Arithmetic operations - default to int32_t
                        tuple_types.append("int32_t")
                    else:
                        actual_return_types_inferred = False
                        break

                # Try to infer name from context
                if isinstance(elt, ast.Name):
                    tuple_names.append(elt.id)
                elif isinstance(elt, ast.Attribute):
                    tuple_names.append(elt.attr)
                elif isinstance(elt, ast.Constant):
                    # Constant values - check if it's a bool constant (True/False)
                    if isinstance(elt.value, bool):
                        tuple_types[i] = "bool"  # Override with bool type
                    # Try to infer name from method name
                    if i == 0 and tuple_types and tuple_types[0] == "bool":
                        method_name = method_ast.name
                        if method_name.startswith("get_"):
                            base_name = method_name[4:]
                            tuple_names.append(f"has_{base_name}")
                        elif method_name.startswith("is_"):
                            tuple_names.append(method_name)
                        elif method_name.startswith("has_"):
                            tuple_names.append(method_name)
                        else:
                            tuple_names.append(f"has_{method_name}")
                    else:
                        tuple_names.append(f"ret_{i}")
                else:
                    tuple_names.append(f"ret_{i}")

            # If annotation exists and has different count than actual return, use annotation
            # (annotation is more authoritative for function signature)
            if annotation_return_info and annotation_return_info['is_tuple']:
                annotation_tuple_types = annotation_return_info['tuple_types']
                annotation_tuple_names = annotation_return_info.get('tuple_names', [])
                # If annotation has different count, use annotation types
                if len(annotation_tuple_types) != len(tuple_types):
                    tuple_types = annotation_tuple_types
                    # Merge names: use actual return names where available, fill rest with annotation names or defaults
                    # When annotation has more elements than actual return, map actual return values to annotation positions
                    # For example: annotation (bool, int32, int32) but actual return (m_block_idx, n_block_idx)
                    # Should map: [0] = bool (default), [1] = m_block_idx, [2] = n_block_idx
                    merged_names = []
                    actual_idx = 0  # Index into actual return values
                    for i in range(len(tuple_types)):
                        # First element might be bool (from annotation) but not in actual return
                        if i == 0 and tuple_types[0] == "bool" and actual_idx < len(tuple_names):
                            # Skip bool element, use default name
                            if annotation_tuple_names and len(annotation_tuple_names) > 0:
                                merged_names.append(annotation_tuple_names[0])
                            else:
                                method_name = method_ast.name if hasattr(method_ast, 'name') else "unknown"
                                if method_name.startswith("get_"):
                                    base_name = method_name[4:]
                                    merged_names.append(f"has_{base_name}")
                                else:
                                    merged_names.append(f"has_{method_name}")
                        elif actual_idx < len(tuple_names):
                            # Use actual return name
                            merged_names.append(tuple_names[actual_idx])
                            actual_idx += 1
                        elif i < len(annotation_tuple_names):
                            # Use annotation name if available
                            merged_names.append(annotation_tuple_names[i])
                        else:
                            # Generate default name
                            merged_names.append(f"ret_{i}")
                    tuple_names = merged_names
                elif actual_return_types_inferred and tuple_types:
                    # Counts match - use actual return types (they're more accurate)
                    pass  # Already using actual return types
            # If we couldn't infer from actual return, use annotation if available
            elif not actual_return_types_inferred or not tuple_types:
                if annotation_return_info and annotation_return_info['is_tuple']:
                    tuple_types = annotation_return_info['tuple_types']
                    tuple_names = annotation_return_info.get('tuple_names',
                                                             [f"ret_{i}" for i in range(len(tuple_types))])
                else:
                    raise ValueError(
                        f"Cannot infer return type for tuple in method '{method_ast.name}'. "
                        f"Please add return type annotation: def {method_ast.name}(...) -> Tuple[type1, type2, ...]:")

            return {'is_tuple': True, 'tuple_types': tuple_types, 'tuple_names': tuple_names, 'return_ast': ret_value}
        else:
            # Single return value
            # Try annotation first if available
            if annotation_return_info and not annotation_return_info['is_tuple']:
                return annotation_return_info

            try:
                ll_type = self.type_inferencer.infer_node_type(ret_value)
                ret_type = self._lltype_to_cpp_type(ll_type)
            except Exception as e:
                # If annotation available, use it even if inference fails
                if annotation_return_info:
                    return annotation_return_info
                raise ValueError(f"Cannot infer return type for method '{method_ast.name}'. "
                                 f"Please add return type annotation: def {method_ast.name}(...) -> ReturnType:") from e
            return {'is_tuple': False, 'return_type': ret_type, 'return_ast': ret_value}

    def _analyze_return_from_annotation(self, annotation: ast.AST, ret_value: ast.AST,
                                        method_name: Optional[str] = None) -> Dict[str, Any]:
        """Analyze return type from annotation (e.g., -> Tuple[bool, int, int])."""
        try:
            ll_type = self.type_inferencer._resolve_annotation(annotation)

            # Check if it's a TupleType
            from little_kernel.core.type_system import TupleType
            if isinstance(ll_type, TupleType):
                # Extract element types and convert to C++ types
                tuple_types = []
                for element_ll_type in ll_type.element_types:
                    cpp_type = self._lltype_to_cpp_type(element_ll_type)
                    tuple_types.append(cpp_type)

                # Extract names from return value
                tuple_names = []
                if isinstance(ret_value, ast.Tuple):
                    for i, elt in enumerate(ret_value.elts):
                        if isinstance(elt, ast.Name):
                            tuple_names.append(elt.id)
                        elif isinstance(elt, ast.Attribute):
                            tuple_names.append(elt.attr)
                        elif isinstance(elt, ast.Constant):
                            # Constant values don't have names - try to infer from method name
                            if i == 0 and tuple_types and tuple_types[0] == "bool" and method_name:
                                if method_name.startswith("get_"):
                                    base_name = method_name[4:]
                                    tuple_names.append(f"has_{base_name}")
                                elif method_name.startswith("is_"):
                                    tuple_names.append(method_name)
                                elif method_name.startswith("has_"):
                                    tuple_names.append(method_name)
                                else:
                                    tuple_names.append(f"has_{method_name}")
                            else:
                                tuple_names.append(f"ret_{i}")
                        else:
                            tuple_names.append(f"ret_{i}")
                else:
                    tuple_names = [f"ret_{i}" for i in range(len(tuple_types))]

                return {
                    'is_tuple': True, 'tuple_types': tuple_types, 'tuple_names': tuple_names, 'return_ast': ret_value,
                    'is_bool_first': tuple_types and tuple_types[0] == "bool"
                }

            # Single return type annotation
            ret_type = self._lltype_to_cpp_type(ll_type)
            return {'is_tuple': False, 'return_type': ret_type, 'return_ast': ret_value}
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f"Cannot resolve return type annotation. Please ensure the annotation is valid. "
                             f"Annotation: {ast.dump(annotation)}, Error: {e}") from e

    def _lltype_to_cpp_type(self, ll_type: "ll.LLType") -> str:
        """Convert LLType to C++ type string."""
        return self.type_converter.lltype_to_cpp(ll_type)

    def _is_const_method(self, method_ast: ast.FunctionDef) -> bool:
        """Check if method should be const (doesn't modify member variables)."""
        const_keywords = ['get_', 'is_', 'has_']
        if any(method_ast.name.startswith(kw) for kw in const_keywords):
            # Check if it actually modifies members
            for stmt in ast.walk(method_ast):
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Attribute):
                            if isinstance(target.value, ast.Name) and target.value.id == 'self':
                                return False  # Modifies member
        return False  # Default to non-const for safety

    def _is_parameter_modified(self, method_ast: ast.FunctionDef, param_name: str) -> bool:
        """Check if a parameter is modified in the method body."""
        for stmt in ast.walk(method_ast):
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name) and target.id == param_name:
                        return True
        return False
