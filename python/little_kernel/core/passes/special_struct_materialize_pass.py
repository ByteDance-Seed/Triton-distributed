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
Special struct materialize pass for translating special struct constructor calls.

This pass:
1. Recognizes special struct constructor calls (e.g., Scheduler(), MyStruct())
2. Replaces them with special struct declarations and initializations
3. Translates special struct method calls to C++ code by parsing class AST
"""

import ast
from typing import Dict, Any, List, Optional, Set
from .pass_base import CompleteASTMutator
from .utils.resolve_attribute import recursive_resolve_attribute
from .utils.error_report import raise_pass_error
from .utils.ir_nodes import SpecialStructDef, EnumDef
from little_kernel.codegen.registries.special_struct_registry import get_all_registered_special_structs
from little_kernel.codegen.registries.enum_registry import get_all_registered_enums


def _python_value_to_ast_node(value: Any) -> ast.AST:
    """
    Convert a Python default value to an AST node.
    
    Handles:
    - Basic types (int, float, bool, str, None) -> ast.Constant
    - Enum instances -> ast.Attribute (e.g., GemmType.Normal)
    - Other types -> attempt ast.Constant, fallback to error
    """
    from enum import Enum

    # Handle None
    if value is None:
        return ast.Constant(value=None)

    # Handle basic types that ast.Constant supports
    if isinstance(value, (int, float, bool, str)):
        return ast.Constant(value=value)

    # Handle Enum instances
    if isinstance(value, Enum):
        # Convert Enum instance to ast.Attribute
        # e.g., GemmType.Normal -> ast.Attribute(value=ast.Name(id='GemmType'), attr='Normal')
        enum_class = type(value)
        enum_class_name = enum_class.__name__
        enum_member_name = value.name
        return ast.Attribute(value=ast.Name(id=enum_class_name, ctx=ast.Load()), attr=enum_member_name, ctx=ast.Load())

    # For other types, try ast.Constant as fallback
    # This might work for some types, but may fail for complex objects
    try:
        return ast.Constant(value=value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert Python default value {type(value).__name__} ({value}) to AST node. "
                         f"Only basic types (int, float, bool, str, None) and Enum instances are supported. "
                         f"Error: {e}")


class SpecialStructMaterializeTransformer(CompleteASTMutator):
    """
    AST transformer that handles special struct constructor calls.
    
    This transformer:
    - Recognizes special struct constructor calls and replaces them with struct declarations
    - Translates special struct method calls to appropriate C++ code
    - Works generically for any registered special struct
    """

    def __init__(self, ctx: Dict[str, Any]):
        super().__init__()
        self.ctx = ctx
        self.special_struct_vars: Dict[str, Dict[str, Any]] = {}  # var_name -> struct_info
        self.special_struct_counter = 0  # For generating unique struct names

        # Track which structs and enums are used (for adding definitions to IR)
        self.used_special_structs: Set[str] = set()
        self.used_enums: Set[str] = set()

        # Get all registered special structs
        self.registered_structs = get_all_registered_special_structs()

    def _is_special_struct_constructor(self, func: Any) -> Optional[str]:
        """
        Check if function is a special struct constructor.
        
        Returns the struct name if it's a registered special struct constructor, None otherwise.
        
        This handles:
        1. Direct class constructor calls (e.g., Scheduler() -> Scheduler)
        2. Factory functions that return a struct instance (detected by checking return type annotation)
        """
        if func is None:
            return None

        if not hasattr(func, '__name__'):
            return None

        func_name = func.__name__
        func_module = getattr(func, '__module__', None)

        # Check if it's a registered special struct constructor
        for struct_name, reg_info in self.registered_structs.items():
            class_obj = reg_info.get('class')
            if not class_obj:
                continue

            class_module = getattr(class_obj, '__module__', None)

            # Case 1: Direct class constructor call (e.g., Scheduler() -> Scheduler)
            # Check if the resolved function is the class itself
            if func == class_obj:
                return struct_name

            # Case 1b: Function name matches struct name (e.g., MyStruct() -> MyStruct)
            if func_name.lower() == struct_name.lower() or func_name == struct_name:
                if func_module and class_module and func_module == class_module:
                    return struct_name
                # Also check if it's the constructor directly (module check might fail)
                # This handles cases where the class is in ctx but module info is missing
                return struct_name

            # Case 2: Function is a factory function that returns the struct
            # Check by return type annotation
            if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
                return_type = func.__annotations__['return']
                # Check if return type is the class itself or a string name
                if return_type == class_obj:
                    # Function returns the class directly
                    if func_module and class_module and func_module == class_module:
                        return struct_name
                elif isinstance(return_type, str) and return_type == struct_name:
                    # Return type is a string matching struct name
                    if func_module and class_module and func_module == class_module:
                        return struct_name
                elif hasattr(return_type, '__name__') and return_type.__name__ == struct_name:
                    # Return type is a class with matching name
                    if func_module and class_module and func_module == class_module:
                        return struct_name

        return None

    def _resolve_callable(self, node: ast.AST) -> Optional[Any]:
        """Resolve a callable from AST node."""
        try:
            result = recursive_resolve_attribute(node, self.ctx)
            return result
        except Exception:
            return None

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """Handle assignments that might contain special struct constructor calls."""
        # Check if this is a special struct constructor call BEFORE visiting
        if isinstance(node.value, ast.Call):
            # Try to resolve the function and check if it's a special struct constructor
            # This works for both Name (e.g., Scheduler(...)) and Attribute access (e.g., module.Scheduler(...))
            try:
                func = self._resolve_callable(node.value.func)
                if func:
                    struct_name = self._is_special_struct_constructor(func)
                    if struct_name:
                        return self._transform_special_struct_call(node, node.value, struct_name)
            except Exception:
                pass

            # Fallback: Check by name if resolution fails
            # This handles cases where the class can't be resolved from ctx but we know it's a constructor
            if isinstance(node.value.func, ast.Name):
                func_name = node.value.func.id
                # Check if it matches any registered special struct name
                for struct_name in self.registered_structs.keys():
                    if func_name.lower() == struct_name.lower() or func_name == struct_name:
                        # Try to resolve one more time, but if it fails, still transform it
                        # This handles cases where the class exists but can't be resolved from ctx
                        try:
                            func = self._resolve_callable(node.value.func)
                            if func and self._is_special_struct_constructor(func) == struct_name:
                                return self._transform_special_struct_call(node, node.value, struct_name)
                        except Exception:
                            # If resolution fails but name matches a registered struct, assume it's the constructor
                            # This is safe because we're checking against registered structs
                            return self._transform_special_struct_call(node, node.value, struct_name)

            # Also check if it's an Attribute access (e.g., from module import Scheduler)
            if isinstance(node.value.func, ast.Attribute):
                # Try to resolve and check
                try:
                    func = self._resolve_callable(node.value.func)
                    if func:
                        struct_name = self._is_special_struct_constructor(func)
                        if struct_name:
                            return self._transform_special_struct_call(node, node.value, struct_name)
                except Exception:
                    pass

        # Visit the value first (for other cases)
        new_value = self.visit(node.value)

        # Process targets
        new_targets = self._process_list(node.targets)

        # Create new assignment node
        node_copy = self._copy_node(node)
        node_copy.targets = new_targets
        node_copy.value = new_value
        return node_copy

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        """Handle attribute access on special struct objects and enum values."""
        new_value = self.visit(node.value)

        # Check if this is accessing a special struct method
        if isinstance(new_value, ast.Name):
            var_name = new_value.id
            if var_name in self.special_struct_vars:
                # This is a special struct method call
                # We'll handle this in visit_Call
                return node

        # Check if this is an enum value access (e.g., GemmType.Normal)
        # We need to check the original node.value, not the visited one
        if isinstance(node.value, ast.Name):
            enum_name = node.value.id
            # Try to resolve and check if it's an enum class
            try:
                resolved = recursive_resolve_attribute(node.value, self.ctx)
                from enum import Enum
                if isinstance(resolved, type) and issubclass(resolved, Enum):
                    # Track this enum for adding definition to IR
                    self.used_enums.add(enum_name)
            except Exception:
                pass

        # Also check if the resolved attribute itself is an enum value
        try:
            resolved_attr = recursive_resolve_attribute(node, self.ctx)
            from enum import Enum
            if isinstance(resolved_attr, Enum):
                enum_class = type(resolved_attr)
                enum_name = enum_class.__name__
                self.used_enums.add(enum_name)
        except Exception:
            pass

        node_copy = self._copy_node(node)
        node_copy.value = new_value
        return node_copy

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        """Handle constants, including enum values."""
        # Check if the constant value is an enum instance
        from enum import Enum
        if isinstance(node.value, Enum):
            enum_class = type(node.value)
            enum_name = enum_class.__name__
            # Add to used_enums set
            self.used_enums.add(enum_name)
            # Also check if it's registered with a different name
            registered_enums = get_all_registered_enums()
            for reg_name, reg_class in registered_enums.items():
                if reg_class == enum_class:
                    # If registered with different name, use registered name
                    if reg_name != enum_name:
                        self.used_enums.add(reg_name)
        # Call parent to get proper node copy
        return super().visit_Constant(node)

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        """Handle keyword arguments, including enum values."""
        # Visit the value to detect enum constants
        new_value = self.visit(node.value)
        # Create new keyword node
        node_copy = self._copy_node(node)
        node_copy.arg = node.arg
        node_copy.value = new_value
        return node_copy

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """Handle method calls on special struct objects."""
        new_func = self.visit(node.func)
        new_args = self._process_list(node.args)
        new_keywords = self._process_list(node.keywords)

        # Check if this is a special struct method call
        if isinstance(new_func, ast.Attribute):
            if isinstance(new_func.value, ast.Name):
                var_name = new_func.value.id
                if var_name in self.special_struct_vars:
                    # This is a special struct method call
                    struct_info = self.special_struct_vars[var_name]
                    struct_name = struct_info.get('struct_name')
                    return self._transform_special_struct_method_call(node, var_name, struct_name, new_func.attr,
                                                                      new_args, new_keywords)

        # Regular call
        node_copy = self._copy_node(node)
        node_copy.func = new_func
        node_copy.args = new_args
        node_copy.keywords = new_keywords
        return node_copy

    def _transform_special_struct_call(self, assign_node: ast.Assign, call_node: ast.Call, struct_name: str) -> ast.AST:
        """
        Transform special struct constructor call into struct declaration.
        
        Example:
            scheduler = Scheduler(...)
        becomes:
            Scheduler<...> scheduler(...);
        """
        if len(assign_node.targets) != 1:
            raise_pass_error(f"{struct_name} constructor must be assigned to a single variable", node=assign_node,
                             ctx=self.ctx)

        target = assign_node.targets[0]
        if not isinstance(target, ast.Name):
            raise_pass_error(f"{struct_name} constructor target must be a variable name", node=target, ctx=self.ctx)

        var_name = target.id

        # Extract arguments from call
        struct_info = self._extract_struct_args(call_node, struct_name)

        # Store struct info for later method call translation
        struct_info['var_name'] = var_name
        struct_info['struct_name'] = struct_name
        struct_info['ast_node'] = assign_node
        self.special_struct_vars[var_name] = struct_info

        # Track this special struct for adding definition to IR
        self.used_special_structs.add(struct_name)

        # Mark this assignment as a special struct declaration for codegen
        new_assign = self._copy_node(assign_node)
        new_assign._is_special_struct_decl = True
        new_assign._special_struct_name = struct_name
        new_assign._special_struct_info = struct_info
        # Keep the call node - codegen will handle it via _is_special_struct_decl
        new_assign.value = call_node
        return new_assign

    def _extract_struct_args(self, call_node: ast.Call, struct_name: str) -> Dict[str, Any]:
        """
        Extract struct arguments from constructor call.
        
        This is a generic implementation that extracts all arguments (positional and keyword)
        and stores them in a dictionary. The actual argument names depend on the struct's __init__.
        """
        args = call_node.args
        keywords = {kw.arg: kw.value for kw in call_node.keywords if kw.arg}

        result = {}

        # Get the class to understand its __init__ signature
        reg_info = self.registered_structs.get(struct_name)
        if reg_info and reg_info.get('class'):
            class_obj = reg_info['class']
            try:
                import inspect
                sig = inspect.signature(class_obj.__init__)
                param_names = list(sig.parameters.keys())
                # Skip 'self'
                param_names = [p for p in param_names if p != 'self']

                # Map positional arguments to parameter names
                # Ensure all args are AST nodes
                for i, arg in enumerate(args):
                    if i < len(param_names):
                        if not isinstance(arg, ast.AST):
                            raise ValueError(
                                f"Expected AST node for positional argument {i} of {struct_name} constructor, "
                                f"but got {type(arg).__name__}: {arg}")
                        param_name = param_names[i]
                        result[param_name] = arg

                        # Track enum if positional argument is an enum constant
                        if isinstance(arg, ast.Constant):
                            from enum import Enum
                            if isinstance(arg.value, Enum):
                                enum_class = type(arg.value)
                                enum_name = enum_class.__name__
                                self.used_enums.add(enum_name)
                        elif isinstance(arg, ast.Attribute):
                            # Check if it's an enum value access (e.g., GemmType.Normal)
                            try:
                                resolved = recursive_resolve_attribute(arg, self.ctx)
                                from enum import Enum
                                if isinstance(resolved, Enum):
                                    enum_class = type(resolved)
                                    enum_name = enum_class.__name__
                                    self.used_enums.add(enum_name)
                            except Exception:
                                pass

                        # Check parameter type annotation for template[EnumType]
                        # This is important because template parameters may have enum types
                        try:
                            param = sig.parameters[param_name]
                            if param.annotation != inspect.Parameter.empty:
                                # Try to get actual annotation from __annotations__ or resolve AST
                                annotation_obj = None

                                # First, try to get from __annotations__ (more reliable)
                                if hasattr(class_obj.__init__, '__annotations__'):
                                    annotations = class_obj.__init__.__annotations__
                                    if param_name in annotations:
                                        annotation_obj = annotations[param_name]

                                # If not found, try to resolve AST annotation
                                if annotation_obj is None:
                                    try:
                                        # param.annotation might be a string or AST
                                        if isinstance(param.annotation, str):
                                            # Try to evaluate the string
                                            try:
                                                annotation_obj = eval(param.annotation, class_obj.__init__.__globals__)
                                            except Exception:
                                                pass
                                        elif isinstance(param.annotation, ast.Subscript):
                                            # Try to resolve AST
                                            try:
                                                annotation_obj = recursive_resolve_attribute(
                                                    param.annotation.slice, self.ctx)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                # Check if annotation is template[EnumType]
                                from little_kernel.core.type_system import Template
                                if annotation_obj is not None:
                                    # Check if it's a Template type
                                    if isinstance(annotation_obj, Template):
                                        inner_type = annotation_obj.inner_type
                                        # Check if inner_type is an enum class
                                        from enum import Enum
                                        if isinstance(inner_type, type) and issubclass(inner_type, Enum):
                                            enum_name = inner_type.__name__
                                            self.used_enums.add(enum_name)
                                    # Also check if annotation_obj itself is an enum class
                                    from enum import Enum
                                    if isinstance(annotation_obj, type) and issubclass(annotation_obj, Enum):
                                        enum_name = annotation_obj.__name__
                                        self.used_enums.add(enum_name)

                                # Fallback: parse AST if annotation_obj is None
                                if annotation_obj is None and isinstance(param.annotation, ast.Subscript):
                                    if isinstance(param.annotation.slice, ast.Name):
                                        enum_name = param.annotation.slice.id
                                        # Try to resolve from ctx or registered enums
                                        if enum_name in self.ctx:
                                            enum_obj = self.ctx[enum_name]
                                            from enum import Enum
                                            if isinstance(enum_obj, type) and issubclass(enum_obj, Enum):
                                                self.used_enums.add(enum_name)
                                        else:
                                            # Check registered enums
                                            from little_kernel.codegen.registries.enum_registry import get_enum
                                            enum_obj = get_enum(enum_name)
                                            if enum_obj:
                                                self.used_enums.add(enum_name)
                        except Exception:
                            # Silently ignore errors in annotation resolution
                            pass

                # Add keyword arguments
                # Ensure all keyword values are AST nodes
                for kw_arg, kw_value in keywords.items():
                    if not isinstance(kw_value, ast.AST):
                        raise ValueError(
                            f"Expected AST node for keyword argument '{kw_arg}' of {struct_name} constructor, "
                            f"but got {type(kw_value).__name__}: {kw_value}")
                    # Track enum if keyword value is an enum constant
                    if isinstance(kw_value, ast.Constant):
                        from enum import Enum
                        if isinstance(kw_value.value, Enum):
                            enum_class = type(kw_value.value)
                            enum_name = enum_class.__name__
                            self.used_enums.add(enum_name)
                    elif isinstance(kw_value, ast.Attribute):
                        # Check if it's an enum value access (e.g., GemmType.Normal)
                        try:
                            resolved = recursive_resolve_attribute(kw_value, self.ctx)
                            from enum import Enum
                            if isinstance(resolved, Enum):
                                enum_class = type(resolved)
                                enum_name = enum_class.__name__
                                self.used_enums.add(enum_name)
                        except Exception:
                            pass

                    # Check keyword parameter type annotation for template[EnumType]
                    if kw_arg in sig.parameters:
                        try:
                            param = sig.parameters[kw_arg]
                            if param.annotation != inspect.Parameter.empty:
                                # Try to get actual annotation from __annotations__ or resolve AST
                                annotation_obj = None

                                # First, try to get from __annotations__ (more reliable)
                                if hasattr(class_obj.__init__, '__annotations__'):
                                    annotations = class_obj.__init__.__annotations__
                                    if kw_arg in annotations:
                                        annotation_obj = annotations[kw_arg]

                                # If not found, try to resolve AST annotation
                                if annotation_obj is None:
                                    try:
                                        # param.annotation might be a string or AST
                                        if isinstance(param.annotation, str):
                                            # Try to evaluate the string
                                            try:
                                                annotation_obj = eval(param.annotation, class_obj.__init__.__globals__)
                                            except Exception:
                                                pass
                                        elif isinstance(param.annotation, ast.Subscript):
                                            # Try to resolve AST
                                            try:
                                                annotation_obj = recursive_resolve_attribute(
                                                    param.annotation.slice, self.ctx)
                                            except Exception:
                                                pass
                                    except Exception:
                                        pass

                                # Check if annotation is template[EnumType]
                                from little_kernel.core.type_system import Template
                                if annotation_obj is not None:
                                    # Check if it's a Template type
                                    if isinstance(annotation_obj, Template):
                                        inner_type = annotation_obj.inner_type
                                        # Check if inner_type is an enum class
                                        from enum import Enum
                                        if isinstance(inner_type, type) and issubclass(inner_type, Enum):
                                            enum_name = inner_type.__name__
                                            self.used_enums.add(enum_name)
                                    # Also check if annotation_obj itself is an enum class
                                    from enum import Enum
                                    if isinstance(annotation_obj, type) and issubclass(annotation_obj, Enum):
                                        enum_name = annotation_obj.__name__
                                        self.used_enums.add(enum_name)

                                # Fallback: parse AST if annotation_obj is None
                                if annotation_obj is None and isinstance(param.annotation, ast.Subscript):
                                    if isinstance(param.annotation.slice, ast.Name):
                                        enum_name = param.annotation.slice.id
                                        # Try to resolve from ctx or registered enums
                                        if enum_name in self.ctx:
                                            enum_obj = self.ctx[enum_name]
                                            from enum import Enum
                                            if isinstance(enum_obj, type) and issubclass(enum_obj, Enum):
                                                self.used_enums.add(enum_name)
                                        else:
                                            # Check registered enums
                                            from little_kernel.codegen.registries.enum_registry import get_enum
                                            enum_obj = get_enum(enum_name)
                                            if enum_obj:
                                                self.used_enums.add(enum_name)
                        except Exception:
                            # Silently ignore errors in annotation resolution
                            pass
                result.update(keywords)

                # Fill in defaults for missing parameters
                for param_name, param in sig.parameters.items():
                    if param_name != 'self' and param_name not in result:
                        if param.default != inspect.Parameter.empty:
                            # Convert Python default value to AST node
                            default_ast = _python_value_to_ast_node(param.default)
                            result[param_name] = default_ast
                            # Track enum if default value is an enum
                            from enum import Enum
                            if isinstance(param.default, Enum):
                                enum_class = type(param.default)
                                enum_name = enum_class.__name__
                                self.used_enums.add(enum_name)
            except ValueError:
                # Re-raise ValueError as-is
                raise
            except Exception:
                # If signature inspection fails, just use positional and keyword args
                # But still validate they are AST nodes
                for i, arg in enumerate(args):
                    if not isinstance(arg, ast.AST):
                        raise ValueError(f"Expected AST node for positional argument {i} of {struct_name} constructor, "
                                         f"but got {type(arg).__name__}: {arg}")
                    result[f'arg_{i}'] = arg
                for kw_arg, kw_value in keywords.items():
                    if not isinstance(kw_value, ast.AST):
                        raise ValueError(
                            f"Expected AST node for keyword argument '{kw_arg}' of {struct_name} constructor, "
                            f"but got {type(kw_value).__name__}: {kw_value}")
                result.update(keywords)
        else:
            # Fallback: just store positional and keyword arguments
            # But still validate they are AST nodes
            for i, arg in enumerate(args):
                if not isinstance(arg, ast.AST):
                    raise ValueError(f"Expected AST node for positional argument {i} of {struct_name} constructor, "
                                     f"but got {type(arg).__name__}: {arg}")
                result[f'arg_{i}'] = arg
            for kw_arg, kw_value in keywords.items():
                if not isinstance(kw_value, ast.AST):
                    raise ValueError(f"Expected AST node for keyword argument '{kw_arg}' of {struct_name} constructor, "
                                     f"but got {type(kw_value).__name__}: {kw_value}")
            result.update(keywords)

        return result

    def _transform_special_struct_method_call(self, call_node: ast.Call, var_name: str, struct_name: str,
                                              method_name: str, args: List[ast.AST],
                                              keywords: List[ast.keyword]) -> ast.AST:
        """
        Transform special struct method calls.
        
        This method marks the call node for special handling in codegen.
        For methods with multiple return values, it also declares reference parameter variables.
        """
        struct_info = self.special_struct_vars[var_name]

        # Check method signature for enum default values and return type
        from little_kernel.codegen.registries.special_struct_registry import get_special_struct_codegen
        codegen_info = get_special_struct_codegen(struct_name)
        return_type_info = None
        if codegen_info and 'class' in codegen_info:
            class_obj = codegen_info['class']
            if hasattr(class_obj, method_name):
                import inspect
                try:
                    method = getattr(class_obj, method_name)
                    sig = inspect.signature(method)
                    # Check default values for enum types
                    from enum import Enum
                    for param_name, param in sig.parameters.items():
                        if param.default != inspect.Parameter.empty:
                            default_value = param.default
                            if isinstance(default_value, Enum):
                                enum_class = type(default_value)
                                enum_name = enum_class.__name__
                                self.used_enums.add(enum_name)

                    # Get return type
                    return_annotation = sig.return_annotation
                    if return_annotation != inspect.Signature.empty:
                        from little_kernel.core.passes.utils.type_inference import TypeInferencer
                        inferencer = TypeInferencer(self.ctx, {})
                        try:
                            return_type_info = inferencer._resolve_annotation(return_annotation)
                        except Exception:
                            # Try string annotation
                            if isinstance(return_annotation, str):
                                try:
                                    # Try to evaluate in method's context
                                    method_globals = getattr(method, '__globals__', {})
                                    if hasattr(class_obj, '__module__'):
                                        import sys
                                        module = sys.modules.get(class_obj.__module__)
                                        if module:
                                            method_globals.update(module.__dict__)
                                    return_type_obj = eval(return_annotation, method_globals)
                                    import little_kernel.language as ll
                                    if return_type_obj == ll.bool_:
                                        return_type_info = ll.bool_
                                    elif return_type_obj == ll.int32:
                                        return_type_info = ll.int32
                                    elif return_type_obj == ll.int64:
                                        return_type_info = ll.int64
                                    else:
                                        return_type_info = return_type_obj
                                except Exception:
                                    pass
                except Exception:
                    pass  # Ignore errors in signature inspection

        # Mark this call as a special struct method call for codegen
        call_node._is_special_struct_method = True
        call_node._special_struct_name = struct_name
        call_node._special_struct_var = var_name
        call_node._special_struct_info = struct_info
        call_node._method_name = method_name
        call_node._return_type_info = return_type_info  # Store return type info

        # For methods with multiple return values, declare reference parameter variables
        # These will be declared before the call in codegen
        if return_type_info:
            from little_kernel.core.type_system import TupleType
            if isinstance(return_type_info, TupleType) and len(return_type_info.element_types) > 1:
                # Multiple return values: first is returned, others are reference parameters
                # Store info for codegen to declare these variables
                call_node._reference_params = []
                # Try to infer names from tuple unpacking context or use defaults
                for i in range(1, len(return_type_info.element_types)):
                    param_name = f"_ret_{method_name}_{i}"
                    call_node._reference_params.append(
                        {'name': param_name, 'type': return_type_info.element_types[i], 'index': i})

        # Store method call info
        if 'method_calls' not in struct_info:
            struct_info['method_calls'] = []
        struct_info['method_calls'].append(
            {'method': method_name, 'node': call_node, 'args': args, 'keywords': keywords})

        return call_node

    def visit_Module(self, node: ast.Module) -> ast.Module:
        """Visit Module and add IR nodes for special struct and enum definitions."""
        # First, visit all statements to collect used structs and enums
        # This will trigger visit_Constant for enum values
        new_body = []
        for stmt in node.body:
            new_stmt = self.visit(stmt)
            if new_stmt is not None:
                if isinstance(new_stmt, list):
                    new_body.extend(new_stmt)
                else:
                    new_body.append(new_stmt)

        # Add IR definitions to the beginning of the module
        # IMPORTANT: Enum definitions must come BEFORE special struct definitions
        # because special structs may use enums as template parameters
        ir_definitions = []

        # First, add enum definitions (they may be used by special structs)
        registered_enums = get_all_registered_enums()
        # Match enum names from used_enums to registered enums
        # Try both by class name and by registered name
        processed_enum_classes = set()  # Track processed enum classes to avoid duplicates
        for enum_name in sorted(self.used_enums):
            # First try direct lookup
            enum_class = registered_enums.get(enum_name)
            if enum_class:
                # Check if we've already processed this enum class
                if enum_class not in processed_enum_classes:
                    enum_def_node = EnumDef(enum_name=enum_name, enum_class=enum_class, ast_node=None)
                    ir_definitions.append(enum_def_node)
                    processed_enum_classes.add(enum_class)
            else:
                # Try to find by class name in registry
                found = False
                for reg_name, reg_class in registered_enums.items():
                    if reg_class.__name__ == enum_name:
                        # Check if we've already processed this enum class
                        if reg_class not in processed_enum_classes:
                            enum_def_node = EnumDef(enum_name=reg_name,  # Use registered name
                                                    enum_class=reg_class, ast_node=None)
                            ir_definitions.append(enum_def_node)
                            processed_enum_classes.add(reg_class)
                        found = True
                        break

                if not found:
                    # If not found in registry, try to find in ctx
                    try:
                        if enum_name in self.ctx:
                            enum_obj = self.ctx[enum_name]
                            from enum import Enum
                            if isinstance(enum_obj, type) and issubclass(enum_obj, Enum):
                                # Check if we've already processed this enum class
                                if enum_obj not in processed_enum_classes:
                                    enum_def_node = EnumDef(enum_name=enum_name, enum_class=enum_obj, ast_node=None)
                                    ir_definitions.append(enum_def_node)
                                    processed_enum_classes.add(enum_obj)
                    except Exception:
                        pass

        # Then, add special struct definitions (they may use enums defined above)
        for struct_name in sorted(self.used_special_structs):
            reg_info = self.registered_structs.get(struct_name)
            if reg_info:
                struct_def_node = SpecialStructDef(struct_name=struct_name, struct_info=reg_info, ast_node=None)
                ir_definitions.append(struct_def_node)

        # Combine IR definitions with the rest of the body
        # IR definitions (enum and struct) go first, then regular statements
        # IMPORTANT: ir_definitions are added to Module body, NOT to any class body
        final_body = ir_definitions + new_body

        new_module = self._copy_node(node)
        new_module.body = final_body
        return new_module


def special_struct_materialize_pass(tree: ast.AST, ctx: Dict[str, Any]) -> ast.AST:
    """
    Special struct materialize pass entry point.
    
    This pass transforms special struct constructor calls into struct
    declarations and translates special struct method calls.
    """
    transformer = SpecialStructMaterializeTransformer(ctx)
    return ast.fix_missing_locations(transformer.visit(tree))
