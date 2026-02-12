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
import re
from io import StringIO
from typing import Dict, Union
import little_kernel.language as ll
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from .registries.type_converter import TypeConverter
from .visitors.expression_codegen import ExpressionCodegenMixin
from .visitors.statement_codegen import StatementCodegenMixin
from .visitors.control_flow_codegen import ControlFlowCodegenMixin
from .visitors.call_codegen import CallCodegenMixin


class CppEmitter(ExpressionCodegenMixin, StatementCodegenMixin, ControlFlowCodegenMixin, CallCodegenMixin,
                 ast.NodeVisitor):
    """
    C++ code emitter with context-aware LLType resolution and scope tracking.
    - Avoids redundant type declarations for in-scope variables (parameters/previous assignments)
    - Enforces basic type safety for reassignments
    - Resolves namespace aliases via user-provided context (e.g., 'lk.uint32')
    """

    def __init__(self, ctx: Dict[str, object], all_types=None):
        """
        Initialize the C++ emitter with a user-provided context.
        
        Args:
            ctx: A dictionary representing the namespace context (e.g., function.__globals__).
                 Must contain the alias used to import `little_kernel.language` (e.g., 'll' or 'lk').
        """
        # Code buffers (headers -> structs -> main code)
        self.header_buffer = StringIO()
        self.header_cache = set()
        self.struct_buffer = StringIO()
        self.main_buffer = StringIO()
        self.builtin_buffer = StringIO()
        self.builtin_cache = set()
        self.builtin_body_cache = set()  # Track emitted function names to prevent duplicates

        # State management
        self.indent_level = 0
        self.indent_unit = "    "  # 4-space indentation

        # LLType tracking: StructType -> C++ struct name
        self.struct_types: Dict["ll.StructType", str] = {}

        # Special struct tracking: Track which special structs are used
        self.used_special_structs: set = set()

        # Enum tracking: Track which Enum types are used
        self.used_enums: set = set()

        # Context & annotation cache
        self.ctx = ctx  # User's namespace (for alias resolution)
        self.annotation_cache: Dict[ast.AST, "ll.LLType"] = {}

        # Scope tracking: Maps variable names to their LLType (per function scope)
        # Reset when entering/exiting a FunctionDef
        self.scope_vars: Dict[str, "ll.LLType"] = {}
        self.all_types = all_types if all_types is not None else {}

        # Add required C++ headers
        self._add_basic_headers()

        # Validate context (ensure LLType-related objects are present)
        self._validate_context()

    def _add_basic_headers(self) -> None:
        """Add C++ headers for std types."""
        headers = [
            "<cstdint>",  # For stdint types (uint32_t, int64_t)
            "<string>",  # For std::string (maps to StringType)
        ]
        for hdr in headers:
            self.header_cache.add(hdr)
            self.header_buffer.write(f"#include {hdr}\n")

    def _validate_context(self) -> None:
        """Validate context contains `little_kernel.language` or LLType objects."""
        has_lltype = any(
            # Check if object is the LLType module
            (hasattr(obj, "__name__") and obj.__name__.startswith("little_kernel.language"))
            # Check if object is an LLType instance
            or (isinstance(obj, (type, object)) and "LLType" in str(type(obj))) for obj in self.ctx.values())

        if not has_lltype:
            raise ValueError("Context (ctx) missing `little_kernel.language` or LLType objects. "
                             "Ensure ctx is the __globals__ of a function that imports `little_kernel.language`.")

    def indent_str(self) -> str:
        """Return current indentation string."""
        return self.indent_unit * self.indent_level

    # ------------------------------ Code Writing Helpers ------------------------------
    def write_main(self, s: str) -> None:
        """Write to main code buffer (functions, classes, statements)."""
        self.main_buffer.write(s)

    def writeln_main(self, s: str = "") -> None:
        """Write line to main code buffer with newline."""
        self.write_main(f"{self.indent_str()}{s}\n")

    def writeln_header(self, s: str = "") -> None:
        """Write line to header buffer with newline."""
        self.header_buffer.write(f"{s}\n")

    def write_struct(self, s: str) -> None:
        """Write to struct definition buffer."""
        self.struct_buffer.write(s)

    def writeln_struct(self, s: str = "") -> None:
        """Write line to struct buffer with newline."""
        self.write_struct(f"{s}\n")

    # C/C++ keywords that must not be treated as function names when matching
    _FUNC_NAME_EXCLUDE = frozenset({"if", "for", "while", "switch", "catch", "return", "else"})

    def writeln_builtin(self, s: str = "") -> None:
        """Write lines to builtin buffer, deduplicating function definitions.
        
        When code from an inner LLKernel emitter is added, it may contain
        function definitions already emitted by the outer emitter. We extract
        function names and skip blocks where all functions are already emitted.
        
        Handles __device__, __forceinline__, pointer types (void*), custom structs,
        and other return types by matching the identifier before the parameter list.
        """
        stripped = s.strip()
        if not stripped:
            return
        # Match identifier before ( and ) { - works for void, void*, uint64_t, Struct_xxx, etc.
        all_matches = re.findall(r'(\w+)\s*\([^)]*\)\s*\{', stripped)
        func_names = [m for m in all_matches if m not in self._FUNC_NAME_EXCLUDE]
        if func_names:
            new_funcs = [f for f in func_names if f not in self.builtin_body_cache]
            if not new_funcs:
                return  # All functions already emitted
            for f in func_names:
                self.builtin_body_cache.add(f)
        else:
            body_key = hash(stripped)
            if body_key in self.builtin_body_cache:
                return
            self.builtin_body_cache.add(body_key)
        self.builtin_buffer.write(f"{s}\n")

    # ------------------------------ LLType to C++ Mapping ------------------------------
    def _generate_struct_name(self, struct_type: "ll.StructType") -> str:
        """Generate a unique, valid C++ struct name from StructType fields."""
        field_suffix = "_".join([f"{name}_{self._lltype_to_cpp(typ)}" for name, typ in struct_type.type_tuple])
        # Sanitize for C++ identifier rules (replace invalid chars)
        valid_suffix = (field_suffix.replace("<", "_").replace(">", "_").replace("*", "_ptr_").replace(" ", ""))
        return f"Struct_{valid_suffix}"

    def _lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a valid C++ type string."""
        if not hasattr(self, '_type_converter'):
            self._type_converter = TypeConverter(self.struct_types)
        return self._type_converter.lltype_to_cpp(ll_type)

    # ------------------------------ Struct Generation ------------------------------
    def _emit_struct_definitions(self) -> None:
        """
        Auto-generate C++ structs for all tracked StructType instances.
        
        Note: Enum and special struct definitions are now handled via IR nodes
        in the AST (EnumDef and SpecialStructDef), which are processed in visit_Module.
        This method only handles regular LLType struct definitions.
        """

        # Then, emit regular LLType struct definitions
        if not self.struct_types:
            return

        self.writeln_struct("// Auto-generated structs from LLType StructType")
        for ll_struct, cpp_name in self.struct_types.items():
            self.writeln_struct(f"struct {cpp_name} {{")
            # Emit struct fields (name + C++ type)
            for field_name, field_type in ll_struct.type_tuple:
                field_cpp = self._lltype_to_cpp(field_type)
                self.writeln_struct(f"    {field_cpp} {field_name};")
            self.writeln_struct("};")
            self.writeln_struct()  # Blank line between structs

    # ------------------------------ AST Node Visitors ------------------------------
    def get_source_location(self, node: ast.AST) -> str:
        """
        Get source code location information from an AST node.
        
        Returns a string like "line 10, column 5" or "unknown location" if not available.
        """
        if hasattr(node, 'lineno') and node.lineno is not None:
            col_info = f", column {node.col_offset + 1}" if hasattr(
                node, 'col_offset') and node.col_offset is not None else ""
            return f"line {node.lineno}{col_info}"
        return "unknown location"

    def get_type(self, value):
        """Get the LLType of a value."""
        if isinstance(value, ll.LLType):
            return value
        elif isinstance(value, ast.Constant) and isinstance(value.value, ll.LLType):
            return value.value
        assert value in self.all_types, f"Value {ast.dump(value) if isinstance(value, ast.AST) else value} is not in all_types"
        return self.all_types[value]

    # Note: visit_Assign, visit_Module, visit_FunctionDef, visit_AnnAssign, visit_Expr,
    # visit_Return, visit_Assert, visit_Pass are now in StatementCodegenMixin

    # Note: visit_Name, visit_Constant, visit_Attribute, visit_BinOp, visit_UnaryOp,
    # visit_BoolOp, visit_Compare, visit_IfExp, visit_Subscript, visit_JoinedStr,
    # visit_FormattedValue are now in ExpressionCodegenMixin

    # Note: visit_Call is now in CallCodegenMixin
    # Note: visit_Return, visit_BinOp, visit_UnaryOp, visit_BoolOp, visit_Compare,
    # visit_IfExp, visit_JoinedStr, visit_FormattedValue, visit_Subscript are now in ExpressionCodegenMixin
    # Note: _get_operator_str is now in ExpressionCodegenMixin

    def generic_visit(self, node: ast.AST) -> Union[None, str]:
        """Default handler for unimplemented AST nodes."""
        # inline may result lists of stmts
        if isinstance(node, list):
            for stmt in node:
                self.visit(stmt)
        else:
            try:
                node_dump = ast.dump(node, indent=2)
            except TypeError:
                # Python 3.7 doesn't support indent parameter
                node_dump = ast.dump(node)
            raise NotImplementedError(f"Unsupported AST node type: {type(node).__name__}\n"
                                      f"Node details: {node_dump}")

    def visit_Expression(self, node: ast.Expression) -> str:
        """Visit Expression node (from ast.parse with mode='eval')."""
        return self.visit(node.body)

    # Note: visit_If, visit_While, visit_For, visit_Continue, visit_Break are now in ControlFlowCodegenMixin
    # Note: visit_Subscript, _process_single_index are now in ExpressionCodegenMixin

    # ------------------------------ Final Code Generation ------------------------------
    def get_code(self, need_header=True) -> str:
        """Combine headers, structs, and main code into final C++ output."""
        self._emit_struct_definitions()  # Emit structs before main code
        return (("/*start of generated code*/\n" + self.header_buffer.getvalue() if need_header else "") + "\n" +
                self.struct_buffer.getvalue() + "\n" + self.builtin_buffer.getvalue() + "\n" +
                self.main_buffer.getvalue())


def codegen_cpp(tree: ast.AST, ctx=None, emit_header=True) -> str:
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    emitter = CppEmitter(ctx=ctx, all_types=all_types)
    emitter.visit(tree)
    return emitter.get_code(need_header=emit_header)
