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
Python to C++ translator for special struct method bodies.

This module provides a translator that converts Python AST nodes to C++ code
for use within special struct method bodies.
"""

import ast
from typing import Dict, Any, Optional, Callable
from io import StringIO
from little_kernel.core.passes.utils.type_inference import TypeInferencer
import little_kernel.language as ll
from little_kernel.codegen.visitors.expression_codegen import ExpressionCodegenMixin
from little_kernel.codegen.visitors.control_flow_codegen import ControlFlowCodegenMixin
from little_kernel.codegen.visitors.call_codegen import CallCodegenMixin
from little_kernel.codegen.registries.type_converter import TypeConverter


class PythonToCppTranslator(ExpressionCodegenMixin, ControlFlowCodegenMixin, CallCodegenMixin, ast.NodeVisitor):
    """
    Translates Python AST nodes to C++ code.
    
    This translator reuses existing codegen mixins but adapts them for special struct codegen:
    - Handles self.xxx -> this->xxx or struct_var_name.xxx conversion
    - Handles template parameters
    - Uses StringIO for output instead of separate buffers
    - Handles special return_info for tuple return values
    """

    def __init__(self, struct_var_name: str = "self", ctx: Optional[Dict[str, Any]] = None,
                 return_info: Optional[Dict[str, Any]] = None, type_inferencer: Optional[TypeInferencer] = None,
                 class_name: Optional[str] = None, static_methods: Optional[set] = None):
        self.struct_var_name = struct_var_name
        self.ctx = ctx or {}
        self.return_info = return_info
        self.type_inferencer = type_inferencer
        self.code = StringIO()
        self.indent_level = 0
        self.indent_unit = "    "
        # Map from parameter name to template parameter name (e.g., 'block_m' -> 'BLOCK_M')
        self.template_param_map: Dict[str, str] = {}

        # Class information for handling static method calls
        self.class_name = class_name
        self.static_methods = static_methods or set()

        # Required attributes for mixins
        self.scope_vars: Dict[str, ll.LLType] = {}
        self.all_types: Dict[ast.AST, ll.LLType] = {}
        # For ExpressionCodegenMixin.visit_Constant
        if type_inferencer and hasattr(type_inferencer, 'all_types'):
            self.all_types = type_inferencer.all_types

        # Required attributes for CallCodegenMixin
        self.builtin_cache: set = set()
        self.header_cache: set = set()

        # Type converter for LLType to C++ conversion
        self.type_converter = TypeConverter()

    def _writeln(self, line: str = ""):
        """Write a line with proper indentation."""
        if line:
            self.code.write(" " * (self.indent_level * len(self.indent_unit)) + line + "\n")
        else:
            self.code.write("\n")

    def write_main(self, s: str):
        """Write to code buffer without newline (for mixin compatibility)."""
        self.code.write(s)

    def writeln_main(self, s: str = ""):
        """Write a line to main buffer (for mixin compatibility)."""
        self._writeln(s)

    def writeln_builtin(self, s: str = ""):
        """Write a line to builtin buffer (for mixin compatibility)."""
        # For special struct codegen, builtins are inlined, so we don't need a separate buffer
        # But we still need this method for CallCodegenMixin compatibility
        pass

    def writeln_header(self, s: str = ""):
        """Write a line to header buffer (for mixin compatibility)."""
        # For special struct codegen, headers are handled at the struct level
        # But we still need this method for CallCodegenMixin compatibility
        pass

    def _lltype_to_cpp(self, ll_type: ll.LLType) -> str:
        """Convert LLType to C++ type string (for mixin compatibility)."""
        return self.type_converter.lltype_to_cpp(ll_type)

    def get_type(self, value):
        """Get the LLType of a value (for mixin compatibility)."""
        if isinstance(value, ll.LLType):
            return value
        elif isinstance(value, ast.Constant) and isinstance(value.value, ll.LLType):
            return value.value
        if value in self.all_types:
            return self.all_types[value]
        # Fallback for special struct codegen
        return ll.int32

    # Override ExpressionCodegenMixin methods to handle self and template parameters
    def visit_Name(self, node: ast.Name) -> str:
        """Process variable/identifier reference with special handling for self and template params."""
        if node.id == 'self':
            return self.struct_var_name
        # Check if it's a template parameter mapped in template_param_map (e.g., 'block_m' -> 'BLOCK_M')
        if hasattr(self, 'template_param_map') and node.id in self.template_param_map:
            return self.template_param_map[node.id]
        # Check if it's a template parameter (kGemmType, BLOCK_M, etc.)
        if node.id.startswith('k') or node.id.isupper():
            return node.id
        # Use parent implementation for context resolution
        return super().visit_Name(node)

    def visit_Attribute(self, node: ast.Attribute) -> str:
        """Process attribute access with special handling for self.xxx and enum values."""
        # Check if this is an enum value access (e.g., GemmType.Normal, IndexType.MN)
        try:
            from enum import Enum
            from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
            # Try to resolve the attribute to check if it's an enum value
            attr = recursive_resolve_attribute(node, self.ctx)

            # Check if resolved value is an Enum instance
            if isinstance(attr, Enum):
                enum_class = type(attr)
                enum_name = enum_class.__name__
                enum_member = attr.name
                # Track that this enum is used
                if hasattr(self, 'used_enums'):
                    self.used_enums.add(enum_name)
                # Convert to C++ enum value format: EnumType::EnumValue
                return f"{enum_name}::{enum_member}"

            # Check if the base is an Enum class and attr is an enum member
            base = recursive_resolve_attribute(node.value, self.ctx)
            if isinstance(base, type) and issubclass(base, Enum):
                enum_name = base.__name__
                # Track that this enum is used
                if hasattr(self, 'used_enums'):
                    self.used_enums.add(enum_name)
                # Convert to C++ enum value format: EnumType::EnumValue
                return f"{enum_name}::{node.attr}"
        except Exception:
            # If resolution fails, fall through to normal handling
            pass

        # Check if this is self.xxx first (before visiting value)
        if isinstance(node.value, ast.Name) and node.value.id == 'self':
            # Check if this is a template parameter
            if hasattr(self, 'template_param_map') and node.attr in self.template_param_map:
                # self.xxx where xxx is a template parameter -> use template parameter name
                return self.template_param_map[node.attr]
            # self.xxx -> this->xxx (or struct_var_name.xxx for constructor)
            return f"{self.struct_var_name}{node.attr}"

        # Try to resolve as enum value one more time with different approach
        # Check if node.value is a Name that might be an enum class
        if isinstance(node.value, ast.Name):
            try:
                from little_kernel.codegen.registries.enum_registry import get_all_registered_enums
                registered_enums = get_all_registered_enums()
                # Check if node.value.id matches an enum class name
                if node.value.id in registered_enums:
                    enum_class = registered_enums[node.value.id]
                    # Check if node.attr is a valid enum member
                    if hasattr(enum_class, node.attr):
                        enum_member = getattr(enum_class, node.attr)
                        from enum import Enum
                        if isinstance(enum_member, Enum):
                            enum_name = enum_class.__name__
                            # Track that this enum is used
                            if hasattr(self, 'used_enums'):
                                self.used_enums.add(enum_name)
                            # Convert to C++ enum value format: EnumType::EnumValue
                            return f"{enum_name}::{node.attr}"
            except Exception:
                pass

        # Use parent implementation for other cases
        return super().visit_Attribute(node)

    def _visit_expr(self, node: ast.AST) -> str:
        """Visit an expression node using mixin's visit method."""
        return self.visit(node)

    def visit_Call(self, node: ast.Call) -> str:
        """Process function calls, with special handling for same-class method calls."""
        # First, try to resolve as builtin/intrin function (e.g., blockIdx_x, cdiv)
        # This must come before same-class method check to handle builtins correctly
        func_resolved = None
        if isinstance(node.func, ast.Attribute) or isinstance(node.func, ast.Name):
            try:
                from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
                func_resolved = recursive_resolve_attribute(node.func, self.ctx)
            except Exception:
                pass

        # Check if it's a builtin function (must check before same-class method check)
        if func_resolved is not None and isinstance(func_resolved, Callable):
            from little_kernel.language.builtin_base import BUILTIN_ATTR
            if hasattr(func_resolved, BUILTIN_ATTR) and getattr(func_resolved, BUILTIN_ATTR, False):
                # Handle as builtin - use parent's builtin handling
                # But we need to call the parent's visit_Call which will handle it properly
                return super().visit_Call(node)

        # Then check if this is a method call on the same class (e.g., self._get_num_1d_blocks_per_group(...))
        # Only check if func is an Attribute (not a Name like blockIdx_x)
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name):
                # Check if it's self.method_name(...) or ClassName.method_name(...)
                if node.func.value.id == 'self' or (self.class_name and node.func.value.id == self.class_name):
                    method_name = node.func.attr
                    args_cpp = [self.visit(arg) for arg in node.args]

                    # Check if it's a static method of the same class
                    if method_name in self.static_methods:
                        # Generate static method call: ClassName::method_name(...)
                        if self.class_name:
                            return f"{self.class_name}::{method_name}({', '.join(args_cpp)})"
                        else:
                            # Fallback: just method name (shouldn't happen, but handle it)
                            return f"{method_name}({', '.join(args_cpp)})"
                    else:
                        # It's an instance method call: this->method_name(...)
                        # Use struct_var_name (which is "this->" for instance methods)
                        return f"{self.struct_var_name}{method_name}({', '.join(args_cpp)})"

        # Fall back to CallCodegenMixin's generic handling (for builtins, struct stubs, etc.)
        return super().visit_Call(node)

    def visit_Tuple(self, node: ast.Tuple) -> str:
        """Process tuple expressions."""
        elts = [self.visit(elt) for elt in node.elts]
        return f"({', '.join(elts)})"

    def visit_If(self, node: ast.If) -> None:
        """Generate C++ if statement with proper variable declaration handling."""
        # Use shared _scan_for_assigned_vars from ControlFlowCodegenMixin
        vars_in_if = self._scan_for_assigned_vars(node.body)
        vars_in_else = {}
        if node.orelse:
            vars_in_else = self._scan_for_assigned_vars(node.orelse)

        # Variables used in both if and else blocks need to be declared before if
        vars_to_declare_before_if = {}
        all_vars = set(vars_in_if.keys()) | set(vars_in_else.keys())
        for var_name in all_vars:
            if var_name not in self.scope_vars:
                # Use type from if block if available, else from else block
                if var_name in vars_in_if:
                    vars_to_declare_before_if[var_name] = vars_in_if[var_name]
                elif var_name in vars_in_else:
                    vars_to_declare_before_if[var_name] = vars_in_else[var_name]

        # Declare variables before if statement
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare_before_if.items():
            self._writeln(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        # Generate condition expression
        test_cpp = self._visit_expr(node.test)
        # Check if this should be constexpr if
        is_constexpr = self._is_constexpr_condition(node.test)
        if_keyword = "if constexpr" if is_constexpr else "if"
        self._writeln(f"{if_keyword} ({test_cpp}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self._writeln("}")
        if node.orelse:
            if len(node.orelse) == 1 and isinstance(node.orelse[0], ast.If):
                self.write_main(" else ")
                self.visit(node.orelse[0])
            else:
                self._writeln("else {")
                self.indent_level += 1
                for stmt in node.orelse:
                    self.visit(stmt)
                self.indent_level -= 1
                self._writeln("}")

    def _is_constexpr_condition(self, test: ast.AST) -> bool:
        """Check if condition should use constexpr if (template parameters, enum comparisons, or ll.const)."""
        # Check if condition is wrapped in ll.const(...)
        if isinstance(test, ast.Call):
            if isinstance(test.func, ast.Attribute):
                if isinstance(test.func.value, ast.Name) and test.func.value.id == 'll' and test.func.attr == 'const':
                    return True

        # Check for template parameters or enum comparisons
        if isinstance(test, ast.Compare):
            left = test.left
            # Check if comparing with template parameter or enum
            if isinstance(left, ast.Attribute):
                if isinstance(left.value, ast.Name):
                    # Check if it's a template parameter (kGemmType, etc.)
                    if left.value.id.startswith('k') or left.value.id.isupper():
                        return True
            elif isinstance(left, ast.Name):
                if left.id.startswith('k') or left.id.isupper():
                    return True
        return False

    def visit_While(self, node: ast.While) -> None:
        """Generate C++ while loop (reuse mixin but adapt output)."""
        # Pre-scan for variables assigned in loop body (incl. nested blocks).
        # Python has function-level scope; C++ scopes vars to loop block - hoist before loop.
        vars_to_declare = self._scan_for_assigned_vars(node.body)
        for var_name, (var_cpp_type, var_lltype) in vars_to_declare.items():
            self._writeln(f"{var_cpp_type} {var_name};")
            self.scope_vars[var_name] = var_lltype

        test_cpp = self.visit(node.test)
        self._writeln(f"while ({test_cpp}) {{")
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self._writeln("}")

    def visit_For(self, node: ast.For) -> None:
        """Generate C++ for loop (reuse mixin but adapt output)."""
        # Only support simple for loops: for x in [a, b, ...]
        if isinstance(node.iter, ast.List):
            # for candidate in [8, 16]:
            target = node.target.id if isinstance(node.target, ast.Name) else "item"
            # Pre-scan for variables assigned in loop body (incl. nested blocks).
            # Python has function-level scope; C++ scopes vars to loop block - hoist before loop.
            vars_to_declare = self._scan_for_assigned_vars(node.body)
            vars_to_declare = {k: v for k, v in vars_to_declare.items() if k != target}
            for var_name, (var_cpp_type, var_lltype) in vars_to_declare.items():
                self._writeln(f"{var_cpp_type} {var_name};")
                self.scope_vars[var_name] = var_lltype

            values = [self.visit(elt) for elt in node.iter.elts]
            self._writeln(f"for (const auto& {target}: {{{', '.join(values)}}}) {{")
            self.indent_level += 1
            for stmt in node.body:
                self.visit(stmt)
            self.indent_level -= 1
            self._writeln("}")
        else:
            raise NotImplementedError(f"Unsupported for loop: {ast.dump(node.iter)}")

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Generate C++ annotated assignment (e.g., "x: uint32_t = 5")."""
        if not isinstance(node.target, ast.Name):
            raise NotImplementedError(
                f"Annotated assignment target must be a single variable, got {type(node.target).__name__}")

        var_name = node.target.id

        # Try to resolve the annotation to check if it's const and extract type
        is_const_annotation = False
        const_inner_type = None
        try:
            from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
            resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx)
            from little_kernel.core.type_system import ConstTypeHelper, ConstAnnotationType, const_annotation, Const

            # Check if it's const[int32] (Const type with inner_type)
            if isinstance(resolved_ann, Const):
                is_const_annotation = True
                const_inner_type = resolved_ann.inner_type
            # Check if it's just "const" (ConstTypeHelper or ConstAnnotationType)
            elif (isinstance(resolved_ann, ConstTypeHelper) or isinstance(resolved_ann, ConstAnnotationType)
                  or resolved_ann == const_annotation):
                is_const_annotation = True
                const_inner_type = None  # Need to infer from value
        except Exception:
            # Fallback: check for "const" as a Name node
            if isinstance(node.annotation, ast.Name) and node.annotation.id == "const":
                is_const_annotation = True
                const_inner_type = None

        # Handle "const" annotation
        if is_const_annotation:
            if node.value is not None:
                value_cpp = self.visit(node.value)
                if var_name in self.scope_vars:
                    self._writeln(f"{var_name} = {value_cpp};")
                else:
                    # Use inner_type if specified (const[int32]), otherwise infer from value (const)
                    if const_inner_type is not None:
                        var_lltype = const_inner_type
                    else:
                        var_lltype = self.get_type(node.value)
                    var_cpp_type = self._lltype_to_cpp(var_lltype)
                    self._writeln(f"const {var_cpp_type} {var_name} = {value_cpp};")
                    self.scope_vars[var_name] = var_lltype
        else:
            # Regular type annotation - resolve the annotation AST node to LLType
            try:
                from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
                resolved_ann = recursive_resolve_attribute(node.annotation, self.ctx)
                from little_kernel.core.type_system import LLType
                if isinstance(resolved_ann, LLType):
                    target_type = resolved_ann
                else:
                    # Fallback: try to get from all_types if it's already there
                    if node.annotation in self.all_types:
                        target_type = self.get_type(node.annotation)
                    else:
                        raise ValueError(
                            f"Failed to resolve annotation {ast.dump(node.annotation)} to LLType, got {type(resolved_ann).__name__}"
                        )
            except Exception as e:
                # Fallback: try to get from all_types
                if node.annotation in self.all_types:
                    target_type = self.get_type(node.annotation)
                else:
                    raise ValueError(
                        f"Failed to resolve type annotation for {var_name}: {e}. Annotation: {ast.dump(node.annotation)}"
                    )

            target_cpp_type = self._lltype_to_cpp(target_type)

            if node.value is not None:
                value_cpp = self.visit(node.value)
                if var_name in self.scope_vars:
                    existing_type = self.scope_vars[var_name]
                    if existing_type != target_type:
                        self._writeln(f"{var_name} = static_cast<{target_cpp_type}>({value_cpp});")
                    else:
                        self._writeln(f"{var_name} = {value_cpp};")
                else:
                    self._writeln(f"{target_cpp_type} {var_name} = {value_cpp};")
                    self.scope_vars[var_name] = target_type
            else:
                # Just declaration without value (e.g., "x: int32")
                self._writeln(f"{target_cpp_type} {var_name};")
                self.scope_vars[var_name] = target_type

    def visit_Assign(self, node: ast.Assign) -> None:
        """Generate C++ assignment."""
        if len(node.targets) != 1:
            raise NotImplementedError("Multiple assignment targets not supported")

        target = node.targets[0]
        value_cpp = self.visit(node.value)

        if isinstance(target, ast.Attribute):
            if isinstance(target.value, ast.Name) and target.value.id == 'self':
                # Check if this is a template parameter assignment (should be skipped)
                if hasattr(self, 'template_param_map') and target.attr in self.template_param_map:
                    # This is a template parameter, skip the assignment
                    # Template parameters are compile-time constants, not member variables
                    return
                # self.xxx = ... -> this->xxx = ... (or struct_var.xxx for constructor)
                self._writeln(f"{self.struct_var_name}{target.attr} = {value_cpp};")
            else:
                target_cpp = self.visit(target.value)
                self._writeln(f"{target_cpp}.{target.attr} = {value_cpp};")
        elif isinstance(target, ast.Name):
            var_name = target.id
            # Check if variable is already in scope
            if var_name in self.scope_vars:
                # Variable already declared, just assign
                self._writeln(f"{var_name} = {value_cpp};")
            else:
                # Variable not declared, need to declare it first
                try:
                    value_type = self.get_type(node.value)
                    var_cpp_type = self._lltype_to_cpp(value_type)
                    self._writeln(f"{var_cpp_type} {var_name} = {value_cpp};")
                    self.scope_vars[var_name] = value_type
                except Exception:
                    # If type inference fails, use default int32_t
                    self._writeln(f"int32_t {var_name} = {value_cpp};")
                    self.scope_vars[var_name] = ll.int32
        elif isinstance(target, ast.Tuple):
            # Tuple unpacking: a, b = ... or a, b = xxx.func()
            if isinstance(node.value, ast.Tuple):
                # Direct tuple: a, b = (x, y)
                targets = [t.id if isinstance(t, ast.Name) else self.visit(t) for t in target.elts]
                values = [self.visit(v) for v in node.value.elts]
                for t, v in zip(targets, values):
                    self._writeln(f"{t} = {v};")
            elif isinstance(node.value, ast.Call):
                # Function call returning tuple: a, b = xxx.func()
                # In C++, this becomes: xxx.func(a, b) where a and b are reference parameters
                self._handle_tuple_unpacking_from_call(target, node.value)
            else:
                # Other cases - try to handle as expression
                # If it's a single value, we can't unpack it
                raise NotImplementedError(
                    f"Tuple unpacking from {type(node.value).__name__} not supported: {ast.dump(node.value)}")
        else:
            raise NotImplementedError(f"Unsupported assignment target: {type(target).__name__}")

    def visit_Return(self, node: ast.Return) -> None:
        """Generate C++ return statement.
        
        If method returns a tuple, convert to reference parameter assignments.
        """
        if not node.value:
            self._writeln("return;")
            return

        if self.return_info and self.return_info.get('is_tuple'):
            # Handle tuple return: first element is returned, others are reference parameters
            tuple_types = self.return_info.get('tuple_types', [])
            tuple_names = self.return_info.get('tuple_names', [])

            if isinstance(node.value, ast.Tuple):
                tuple_elts = node.value.elts
                if len(tuple_types) > 1:
                    # Multiple elements: first is returned, others are reference parameters
                    # IMPORTANT: Assign reference parameters BEFORE return statement
                    # Map actual return values to annotation positions
                    # If annotation has more elements than actual return, first element might be bool (not in actual return)
                    first_is_bool = (len(tuple_types) > 0 and tuple_types[0] == "bool"
                                     and len(tuple_elts) < len(tuple_types))

                    # Assign reference parameters (skip first element which is returned)
                    # tuple_elts[0] is returned, tuple_elts[1:] are assigned to reference parameters
                    for i in range(1, len(tuple_types)):
                        param_name = tuple_names[i] if i < len(tuple_names) else f"ret_{i}"
                        # Map actual return values: tuple_elts[i] corresponds to tuple_names[i]
                        # tuple_elts[0] is returned, so tuple_elts[1] -> tuple_names[1], etc.
                        map_idx = i  # Direct mapping: tuple_elts[i] -> tuple_names[i]

                        if map_idx < len(tuple_elts):
                            # Use actual return value
                            elt_cpp = self.visit(tuple_elts[map_idx])
                            # Only assign if the value is different from the parameter name
                            # (avoid redundant assignments like "param = param")
                            if elt_cpp != param_name:
                                self._writeln(f"{param_name} = {elt_cpp};")
                        else:
                            # Actual return has fewer elements than annotation - use default value
                            param_type = tuple_types[i]
                            if param_type == "bool":
                                self._writeln(f"{param_name} = false;")
                            elif param_type in ["int32_t", "int64_t", "uint32_t", "uint64_t"]:
                                self._writeln(f"{param_name} = 0;")
                            else:
                                self._writeln(f"{param_name} = {{}};")  # Default initialization

                    # Return first element (bool if annotation says so, or first actual return value)
                    if first_is_bool:
                        # First element is bool (not in actual return) - return default
                        self._writeln("return true;")  # or false, depending on logic
                    elif len(tuple_elts) > 0:
                        first_elt_cpp = self.visit(tuple_elts[0])
                        self._writeln(f"return {first_elt_cpp};")
                    else:
                        # No actual return values - return default based on first type
                        first_type = tuple_types[0] if tuple_types else "int32_t"
                        if first_type == "bool":
                            self._writeln("return false;")
                        elif first_type in ["int32_t", "int64_t", "uint32_t", "uint64_t"]:
                            self._writeln("return 0;")
                        else:
                            self._writeln("return {};")
                else:
                    # Single element tuple - return it directly
                    first_elt_cpp = self.visit(tuple_elts[0])
                    self._writeln(f"return {first_elt_cpp};")
            else:
                # Single value return (but return_info says it's a tuple - might be variable)
                value_cpp = self.visit(node.value)
                if len(tuple_types) > 1:
                    # Multiple elements expected - assign to reference parameters first, then return
                    # Assign to reference parameters (skip first)
                    for i in range(1, len(tuple_types)):
                        param_name = tuple_names[i] if i < len(tuple_names) else f"ret_{i}"
                        param_type = tuple_types[i]
                        if param_type == "bool":
                            self._writeln(f"{param_name} = false;")
                        elif param_type in ["int32_t", "int64_t", "uint32_t", "uint64_t"]:
                            self._writeln(f"{param_name} = 0;")
                        else:
                            self._writeln(f"{param_name} = {{}};")
                    # Return the single value as first element
                    self._writeln(f"return {value_cpp};")
                else:
                    # Single element - return it
                    self._writeln(f"return {value_cpp};")
        elif self.return_info and not self.return_info.get('is_tuple'):
            # Single return value - return it directly (not as reference parameter)
            value_cpp = self.visit(node.value)
            self._writeln(f"return {value_cpp};")
        else:
            # No return info or simple return
            value_cpp = self.visit(node.value)
            self._writeln(f"return {value_cpp};")

    def visit_Expr(self, node: ast.Expr) -> None:
        """Generate C++ expression statement."""
        expr_cpp = self.visit(node.value)
        self._writeln(f"{expr_cpp};")

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        """Generate C++ augmented assignment."""
        target_cpp = self.visit(node.target)
        value_cpp = self.visit(node.value)
        op_map = {
            ast.Add: "+=",
            ast.Sub: "-=",
            ast.Mult: "*=",
            ast.Div: "/=",
            ast.Mod: "%=",
        }
        op_str = op_map.get(type(node.op), "?=")
        self._writeln(f"{target_cpp} {op_str} {value_cpp};")

    def visit_Break(self, node: ast.Break) -> None:
        """Generate C++ break statement."""
        self._writeln("break;")

    def visit_Continue(self, node: ast.Continue) -> None:
        """Generate C++ continue statement."""
        self._writeln("continue;")

    def visit_Pass(self, node: ast.Pass) -> None:
        """Generate nothing for pass statement."""
        pass

    def _handle_tuple_unpacking_from_call(self, target: ast.Tuple, call: ast.Call):
        """
        Handle tuple unpacking from function call: a, b = xxx.func() -> a = xxx.func(..., b).
        First target receives return value, others are reference parameters.
        """
        # Extract target variable names
        target_names = []
        for t in target.elts:
            if isinstance(t, ast.Name):
                target_names.append(t.id)
            else:
                raise NotImplementedError(f"Tuple unpacking target must be variable names, got {type(t).__name__}")

        if not target_names:
            raise ValueError("Empty tuple unpacking target")

        # Generate function call
        func_cpp = self.visit(call.func)
        args_cpp = [self.visit(arg) for arg in call.args]

        # For tuple returns: first target receives return value, others are reference parameters
        if len(target_names) > 1:
            # Multiple return values: first is returned, others are reference parameters
            first_target = target_names[0]
            ref_params = target_names[1:]

            # Declare first target if needed
            if first_target not in self.scope_vars:
                # Try to infer type from call
                try:
                    from little_kernel.core.type_system import TupleType
                    if hasattr(self, 'all_types') and call in self.all_types:
                        return_type = self.all_types[call]
                        if isinstance(return_type, TupleType) and len(return_type.element_types) > 0:
                            first_elem_type = return_type.element_types[0]
                            var_cpp_type = self._lltype_to_cpp(first_elem_type)
                            var_lltype = first_elem_type
                        else:
                            var_cpp_type = "int32_t"
                            var_lltype = ll.int32
                    else:
                        import little_kernel.language as ll
                        var_cpp_type = "int32_t"
                        var_lltype = ll.int32
                except Exception:
                    var_cpp_type = "int32_t"
                    var_lltype = ll.int32

                # Initialize based on type
                if var_cpp_type == "bool":
                    init_value = "false"
                elif var_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                    init_value = "0"
                else:
                    init_value = "{}"
                self._writeln(f"{var_cpp_type} {first_target} = {init_value};")
                self.scope_vars[first_target] = var_lltype

            # Declare reference parameters if needed
            for ref_name in ref_params:
                if ref_name not in self.scope_vars:
                    # Try to infer type from call
                    try:
                        from little_kernel.core.type_system import TupleType
                        if hasattr(self, 'all_types') and call in self.all_types:
                            return_type = self.all_types[call]
                            if isinstance(return_type, TupleType) and len(
                                    return_type.element_types) > len(target_names) - 1:
                                ref_idx = target_names.index(ref_name)  # Index in target_names
                                if ref_idx < len(return_type.element_types):
                                    ref_elem_type = return_type.element_types[ref_idx]
                                    var_cpp_type = self._lltype_to_cpp(ref_elem_type)
                                    var_lltype = ref_elem_type
                                else:
                                    var_cpp_type = "int32_t"
                                    var_lltype = ll.int32
                            else:
                                var_cpp_type = "int32_t"
                                var_lltype = ll.int32
                        else:
                            var_cpp_type = "int32_t"
                            var_lltype = ll.int32
                    except Exception:
                        var_cpp_type = "int32_t"
                        var_lltype = ll.int32

                    # Initialize based on type
                    if var_cpp_type == "bool":
                        init_value = "false"
                    elif var_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                        init_value = "0"
                    else:
                        init_value = "{}"
                    self._writeln(f"{var_cpp_type} {ref_name} = {init_value};")
                    self.scope_vars[ref_name] = var_lltype

            # Generate call: first_target = func_cpp(args_cpp + ref_params)
            all_args = args_cpp + ref_params
            call_str = f"{func_cpp}({', '.join(all_args)})"
            self._writeln(f"{first_target} = {call_str};")
        else:
            # Single target - just assign return value
            first_target = target_names[0]
            if first_target not in self.scope_vars:
                # Try to infer type from call
                try:
                    if hasattr(self, 'all_types') and call in self.all_types:
                        return_type = self.all_types[call]
                        from little_kernel.core.type_system import TupleType
                        if isinstance(return_type, TupleType) and len(return_type.element_types) > 0:
                            first_elem_type = return_type.element_types[0]
                            var_cpp_type = self._lltype_to_cpp(first_elem_type)
                            var_lltype = first_elem_type
                        else:
                            var_cpp_type = self._lltype_to_cpp(return_type)
                            var_lltype = return_type
                    else:
                        import little_kernel.language as ll
                        var_cpp_type = "int32_t"
                        var_lltype = ll.int32
                except Exception:
                    var_cpp_type = "int32_t"
                    var_lltype = ll.int32

                # Initialize based on type
                if var_cpp_type == "bool":
                    init_value = "false"
                elif var_cpp_type in ("int32_t", "int64_t", "uint32_t", "uint64_t"):
                    init_value = "0"
                else:
                    init_value = "{}"
                self._writeln(f"{var_cpp_type} {first_target} = {init_value};")
                self.scope_vars[first_target] = var_lltype

            call_str = f"{func_cpp}({', '.join(args_cpp)})"
            self._writeln(f"{first_target} = {call_str};")
