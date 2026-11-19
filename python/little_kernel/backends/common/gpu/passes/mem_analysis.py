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
import inspect
from dataclasses import dataclass, field
from typing import Dict, Any, Callable, List, Tuple
from collections import OrderedDict

from little_kernel.core.type_system import get_dtype_size
from little_kernel.core.passes.utils.resolve_attribute import recursive_resolve_attribute
from little_kernel.backends.common.gpu.language.intrin.allocation import empty, zeros, full, align_memory
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.core.passes.utils.extract_empty_params import extract_empty_align, extract_empty_params, get_shape_size, split_shape_dims, extract_max_elems

# ptxas caps *static* .shared data at 48 KiB (0xc000) on every supported CUDA
# architecture, independently of any dynamic shared-memory opt-in. The
# cudaFuncSetAttribute(MaxDynamicSharedMemorySize) carveout raises the limit
# of the `extern __shared__` reservation only; it can never cover a static
# `__shared__` declaration. This is exactly the boundary ptxas reports as
# "uses too much shared data (... 0xc000 max)".
STATIC_SHARED_MEMORY_LIMIT_BYTES: int = 48 * 1024


@dataclass
class StaticSharedMemoryReport:
    """Typed result of the static shared-memory resource legality audit.

    static_bytes: sum of every independent scope="shared" allocation,
        including inter-buffer alignment padding; this is the quantity
        ptxas accounts against STATIC_SHARED_MEMORY_LIMIT_BYTES.
    dynamic_floor_bytes: compile-time-known scope="dynamic_shared" floor;
        reported for transparency only — it is a separate resource and is
        never subtracted from the static sum (a launch dynamic-shared byte
        count does not rewrite or alias independent static allocations).
    buffers: (name, offset, bytes) for each independent static allocation,
        in declaration order, so diagnostics are actionable.
    """
    static_bytes: int
    dynamic_floor_bytes: int
    limit_bytes: int
    buffers: List[Tuple[str, int, int]] = field(default_factory=list)

    @property
    def legal(self) -> bool:
        return self.static_bytes <= self.limit_bytes


class MemoryAnalyzer(ast.NodeVisitor):
    """
    Analyzes AST nodes to track memory allocation via `ll.empty` calls.
    - Calculates total bytes used for each scope (`local`/`shared`/`dynamic_shared`).
    - Records detailed info for each allocated variable: scope, dtype, size (bytes), and offset in its scope.
    """

    def __init__(self, ctx: Dict[str, Any]):
        self.ctx = ctx  # Global context of the function (to resolve variables like BLOCK_M, kNumStages)

        # Track cumulative byte offsets for each scope (used to calculate variable offsets)
        self.scope_offsets: Dict[str, int] = {"local": 0, "shared": 0, "dynamic_shared": 0}

        # Store detailed info for each variable allocated via ll.empty
        self.variables: Dict[str, Dict[str, Any]] = OrderedDict()

        # Valid scope types for ll.empty
        self.valid_scopes = {"local", "shared", "dynamic_shared"}

        # Alignment requirement for memory offsets (bytes)
        self.align_bytes: Dict[str, int] = {"local": -1, "shared": -1, "dynamic_shared": -1}

        # record the first call related to memory
        self.first_mem_call: Dict[str, ast.AST] = {"local": None, "shared": None, "dynamic_shared": None}

        # Whether a shape-expression (runtime-sized) dynamic_shared allocation
        # has been seen; subsequent dynamic_shared allocations would get
        # offsets unknown at compile time and are rejected.
        self._dyn_shared_dynamic_seen: bool = False

    def _handle_align_call(self, node: ast.Call) -> None:
        """
        Handle `ll.align_memory` calls, extract align_bytes and scope, then update memory analysis.
        """
        align_memory_signature = inspect.signature(align_memory)
        valid = isinstance(node.parent, ast.Expr)
        if valid:
            parent = node.parent
            while parent is not None and not isinstance(parent, ast.FunctionDef):
                parent = getattr(parent, "parent", None)
            valid = parent is not None and self.ctx[__LITTLE_KERNEL_ENTRY__].__name__ == parent.name
        if not valid:
            raise ValueError(
                f"ll.align_memory must be called inside the body of main function, see {ast.unparse(node)}")

        if len(node.args) + len(node.keywords) < len(align_memory_signature.parameters):
            raise ValueError(f"ll.align_memory requires at least {len(align_memory_signature.parameters)} arguments")

        align_bytes = None
        scope = None
        for kw in node.keywords:
            if kw.arg == "align_bytes":
                align_bytes = kw.value
                break
        if align_bytes is None:
            assert len(node.args) > 0, "ll.align_memory requires at least an align_bytes argument"
            align_bytes = node.args[0]

        assert isinstance(align_bytes, ast.Constant), "ll.align_memory align_bytes must be a constant"
        assert isinstance(align_bytes.value, int), "ll.align_memory align_bytes must be an integer"

        align_bytes = align_bytes.value

        for kw in node.keywords:
            if kw.arg == "scope":
                scope = kw.value
                break
        if scope is None:
            assert len(node.args) > 1, "ll.align_memory requires at least a scope argument"
            scope = node.args[1]
        assert isinstance(scope, ast.Constant), "ll.align_memory scope must be a constant"
        assert isinstance(scope.value, str), "ll.align_memory scope must be a string"
        assert scope.value in self.valid_scopes, f"ll.align_memory scope must be one of {self.valid_scopes}, got {scope.value}"
        scope = scope.value

        if self.align_bytes[scope] < 0:
            self.align_bytes[scope] = align_bytes
        elif self.align_bytes[scope] == align_bytes:
            # Idempotent re-application: layers of a frontend may each state
            # the same base alignment requirement.
            pass
        elif self.scope_offsets[scope] == 0:
            # No buffer has been laid out in this scope yet, so the base
            # alignment may still be raised without moving any buffer.
            self.align_bytes[scope] = align_bytes
        else:
            raise RuntimeError(
                f"ll.align_memory align_bytes for scope {scope} has already been set to {self.align_bytes[scope]}, cannot be changed to {align_bytes} after buffers were allocated"
            )

        # record the call if it's the first
        if self.first_mem_call[scope] is None:
            self.first_mem_call[scope] = node.parent

    def _handle_zeros_call(self, node: ast.Call) -> None:
        """Handle `ll.zeros` like ll.empty (same shape/dtype/scope, alloc+zero)."""
        self._handle_empty_like_call(node, "ll.zeros")

    def _handle_full_call(self, node: ast.Call) -> None:
        """Handle `ll.full` like ll.empty (same shape/dtype/scope, alloc+fill)."""
        self._handle_empty_like_call(node, "ll.full")

    def _handle_empty_call(self, node: ast.Call) -> None:
        """Handle `ll.empty` calls."""
        self._handle_empty_like_call(node, "ll.empty")

    def _handle_empty_like_call(self, node: ast.Call, api_name: str) -> None:
        """
        Handle `ll.empty` / `ll.zeros` calls, extract shape, dtype, and scope, then update memory analysis.
        """
        # Trace back to find the variable name being assigned (e.g., "D_smem_tensor" in
        # "D_smem_tensor = ll.empty(...)" or in the annotated form
        # "D_smem_tensor: ll.Tensor[dtype] = ll.empty(...)").
        var_name = None
        var_expr = None
        parent = node.parent
        target = None
        if isinstance(parent, ast.Assign) and len(parent.targets) == 1:
            target = parent.targets[0]
        elif (isinstance(parent, ast.AnnAssign) and parent.value is node):
            target = parent.target
        if target is not None:
            if isinstance(target, ast.Name):
                var_name = target.id
                var_expr = target
            elif isinstance(target, ast.Subscript):
                assert isinstance(
                    target.slice, ast.Constant
                ), f"Empty assign to array element must use constant subscripts, but get {ast.dump(target.slice)}"
                var_name = f"{target.value.id}[{target.slice.value}]"
                var_expr = target
            else:
                raise NotImplementedError(f"Unsupported target type for ll.empty: {type(target).__name__}")
        if not var_name:
            raise RuntimeError(
                f"{api_name} call is not assigned to a variable: {ast.dump(node)}\nMaybe should use flatten_empty pass first."
            )

        # Find enclosing FunctionDef (may be nested inside if/while/for)
        parent = node.parent
        while parent is not None and not isinstance(parent, ast.FunctionDef):
            parent = getattr(parent, "parent", None)
        valid = parent is not None and self.ctx[__LITTLE_KERNEL_ENTRY__].__name__ == parent.name
        if not valid:
            raise ValueError(
                f"{api_name} must be called inside the body of main function, see {ast.unparse(node)}\nMaybe should use flatten_empty pass first."
            )

        shape_expr, dtype_expr, scope_expr = extract_empty_params(node)
        assert isinstance(dtype_expr,
                          ast.Constant), f"{api_name} dtype must be a constant, got {type(dtype_expr).__name__}"
        dtype = dtype_expr.value
        assert isinstance(scope_expr,
                          ast.Constant), f"{api_name} scope must be a constant, got {type(scope_expr).__name__}"
        scope = scope_expr.value
        if not isinstance(scope, str) or scope not in self.valid_scopes:
            # Reject unknown scopes here instead of dying with an opaque
            # KeyError from the scope-offset tables below. There is no
            # separate "register" scope: a fixed-size scope="local" array
            # accessed only with compile-time-constant indices (e.g. via
            # ll.unroll) is scalarized into physical registers by nvcc, and
            # constant-index "+f"/"+r" inline-asm constraints bind its
            # elements as a register operand list (the WGMMA accumulator
            # pattern); a runtime-indexed local array stays in .local memory.
            raise ValueError(f"{api_name} scope must be one of {sorted(self.valid_scopes)}, "
                             f"got {scope!r}; register-resident fixed-size fragment arrays "
                             "use scope=\"local\" with compile-time-constant indices")
        dims = split_shape_dims(shape_expr)
        dyn_dims = sorted(i for i, d in enumerate(dims) if isinstance(d, str))
        max_elems = extract_max_elems(node, api_name)
        dyn_elems_src = None
        if dyn_dims:
            # Dynamic extents (shape-expression strings) are only legal for a
            # trailing dynamic_shared allocation: the launch-time byte size is
            # host-evaluated (vibe_cuda_eval), while every compile-time offset
            # stays constant. max_elems is the static legality bound on the
            # total element count of the allocation.
            if scope != "dynamic_shared":
                raise ValueError(f"{api_name} shape-expression dimensions are only supported "
                                 f"for scope=\"dynamic_shared\", got scope={scope!r}")
            if max_elems is None:
                raise ValueError(f"{api_name} with a shape-expression dimension requires a "
                                 "static max_elems legality bound")
            if api_name != "ll.empty":
                raise ValueError(f"{api_name} does not support shape-expression dimensions "
                                 f"(use ll.empty for runtime-sized dynamic_shared allocations)")
            if self._dyn_shared_dynamic_seen:
                raise ValueError("only one dynamic_shared allocation may have "
                                 "shape-expression dimensions, and it must be the last "
                                 "dynamic_shared allocation (later offsets would be "
                                 "unknown at compile time)")
            self._dyn_shared_dynamic_seen = True
            parts = [f"({dims[i]})" for i in dyn_dims]
            const_product = 1
            for d in dims:
                if isinstance(d, int):
                    const_product *= d
            if const_product != 1:
                parts.insert(0, str(const_product))
            dyn_elems_src = " * ".join(parts)
            total_elements = max_elems  # capacity bound; static byte size is 0
        elif max_elems is not None:
            raise ValueError(f"{api_name} max_elems requires at least one shape-expression "
                             "dimension")
        else:
            if scope == "dynamic_shared" and self._dyn_shared_dynamic_seen:
                raise ValueError("a static dynamic_shared allocation may not follow a "
                                 "shape-expression allocation (its offset would be "
                                 "unknown at compile time)")
            total_elements = 1
            for d in dims:
                total_elements *= d

        # Extract fill_value for ll.full
        fill_value = None
        if api_name == "ll.full":
            fill_value_node = None
            for kw in node.keywords:
                if kw.arg == "fill_value":
                    fill_value_node = kw.value
                    break
            if fill_value_node is None and len(node.args) > 3:
                fill_value_node = node.args[3]
            assert fill_value_node is not None, "ll.full requires fill_value argument"
            # Extract the actual value from the AST node
            # Handle both Constant and UnaryOp (e.g., -float('inf'))
            if isinstance(fill_value_node, ast.Constant):
                fill_value = fill_value_node.value
            elif isinstance(fill_value_node, ast.UnaryOp) and isinstance(fill_value_node.op, ast.USub):
                # Handle -constant case (e.g., -float('inf'))
                if isinstance(fill_value_node.operand, ast.Constant):
                    fill_value = -fill_value_node.operand.value
                elif isinstance(fill_value_node.operand, ast.Call):
                    # Handle -float('inf') case
                    call = fill_value_node.operand
                    if isinstance(call.func, ast.Name) and call.func.id == 'float':
                        if len(call.args) == 1 and isinstance(call.args[0], ast.Constant):
                            fill_value = -float(call.args[0].value)
            if fill_value is None:
                raise ValueError(f"ll.full fill_value must be a constant or -constant, got {ast.dump(fill_value_node)}")

        # Calculate total bytes for this allocation. For shape-expression
        # (runtime-sized) allocations this is the max_elems capacity, used
        # only for legality checks and slice size comments — the static
        # scope offset must not advance, the real reservation is evaluated
        # host-side at launch time.
        element_size = get_dtype_size(dtype)
        total_bytes = int(total_elements * element_size)
        assert total_bytes > 0, f"{api_name} call for {var_name} with shape {shape_expr} and dtype {dtype} results in zero bytes allocation"

        # Optional per-buffer start alignment (e.g. align=16 so this buffer is
        # a legal cp.async / 128-bit vectorized smem destination).
        align = extract_empty_align(node)
        if align is not None and scope == "local":
            raise ValueError(f"{api_name} align is not supported for scope local; "
                             "use scope shared or dynamic_shared")
        base_align = self.align_bytes[scope]
        if base_align < 0:
            # CUDA guarantees the dynamic shared memory base is at least
            # 16-byte aligned; static shared buffers are naturally aligned.
            base_align = 16
        if align is not None and align > base_align:
            raise ValueError(f"{api_name} align={align} for {var_name} exceeds the base "
                             f"alignment {base_align} of scope {scope}; raise it first with "
                             f"ll.align_memory({align}, scope=\"{scope}\")")

        # Record offset (current offset before allocation, aligned up to the
        # buffer's own requirement) and update scope offset.
        offset = self.scope_offsets[scope]
        if align is not None:
            offset = ((offset + align - 1) // align) * align
        if dyn_elems_src is None:
            self.scope_offsets[scope] = offset + total_bytes

        # Store detailed info about this variable (empty and zeros both go here)
        if var_name in self.variables:
            prev = self.variables[var_name]
            # Report the descriptor fields that differ between the two
            # declarations so duplicate-allocation failures are actionable
            # without re-running the pipeline by hand.
            prev_desc = (prev["scope"], prev["dtype"], prev["total_elements"], prev["total_bytes"])
            curr_desc = (scope, dtype, total_elements, total_bytes)
            diff_lines = []
            for field, before, after in zip(("scope", "dtype", "total_elements", "total_bytes"), prev_desc, curr_desc):
                if before != after:
                    diff_lines.append(f"    {field}: first={before!r} second={after!r}")
            diff_text = ("\n".join(diff_lines) if diff_lines else "    identical scope/dtype/shape/bytes"
                         " (re-declaration of one buffer)")
            raise ValueError(f"{api_name} call for {var_name} has already been recorded"
                             f" (ir_memory_analysis_duplicate); a second memory allocation"
                             f" with the same variable name is not supported.\n"
                             f"  first definition:  {prev.get('call_site', '<unavailable>')}\n"
                             f"  second definition: {ast.unparse(node)}\n"
                             f"  descriptor diff:\n{diff_text}\n"
                             f"  If both declarations are meant to be the same buffer,"
                             f" declare it once and reuse the variable; otherwise rename"
                             f" one of the declarations.")
        self.variables[var_name] = ({
            "name": var_name, "expr": var_expr, "scope": scope, "dtype": dtype, "element_size": element_size,
            "total_elements": total_elements, "total_bytes": total_bytes,
            # Static per-dimension extents (ints) in declaration order. Dynamic
            # (shape-expression) dims contribute their string form and only
            # occur for dynamic_shared; local-scope shapes are always all-int.
            "shape": tuple(dims), "offset": offset,  # Offset within its scope (bytes)
            "align": align,  # Optional per-buffer start alignment
            "is_zeros": (api_name == "ll.zeros"),  # Pass uses this to replace with alloc_local_zeros
            "is_full": (api_name == "ll.full"),  # Pass uses this to replace with alloc_local_full
            "fill_value": fill_value,  # AST node for fill value (for ll.full)
            # Shape-expression (runtime-sized) dynamic_shared allocation:
            # dyn_elems_src is an element-count source string in the launch
            # shape-expression mini-language (host-evaluated at launch);
            # max_elems is its static legality bound.
            "dyn_elems_src": dyn_elems_src, "max_elems": max_elems, "call_site":
            ast.unparse(node),  # Source snippet for diagnostics only
        })

        # record the call if it's the first
        if self.first_mem_call[scope] is None:
            self.first_mem_call[scope] = node.parent

    def visit_Call(self, node: ast.Call) -> None:
        """
        Override to track `ll.empty` calls. Extracts:
        - Variable name (target of the assignment)
        - Shape (to calculate total elements)
        - Dtype (to calculate element size)
        - Scope (local/shared/dynamic_shared)
        Then computes total bytes and updates scope offsets.
        """
        # Check if the call is to attribute `ll.empty`
        func = recursive_resolve_attribute(node.func, self.ctx)
        if isinstance(func, Callable) and hasattr(func, '__name__'):

            # For align
            if func.__name__ == align_memory.__name__:
                self._handle_align_call(node)
            # For alloc by empty
            elif func.__name__ == empty.__name__:
                self._handle_empty_call(node)
            elif func.__name__ == zeros.__name__:
                self._handle_zeros_call(node)
            elif func.__name__ == full.__name__:
                self._handle_full_call(node)

        self.generic_visit(node)

    def static_shared_memory_report(self) -> StaticSharedMemoryReport:
        """Assemble the typed static shared-memory resource report.

        Independent ``scope="shared"`` allocations are *summed*: the IR gives
        no aliasing/liveness-union guarantee between distinct declarations,
        so a launch-config dynamic ``shared_memory_bytes`` reservation (which
        configures the separate ``extern __shared__`` region) must never be
        treated as covering, rewriting or unioning these static arrays.
        """
        buffers = [(var["name"], var["offset"], var["total_bytes"])
                   for var in self.variables.values()
                   if var["scope"] == "shared"]
        return StaticSharedMemoryReport(
            static_bytes=self.scope_offsets["shared"],
            dynamic_floor_bytes=self.scope_offsets["dynamic_shared"],
            limit_bytes=STATIC_SHARED_MEMORY_LIMIT_BYTES,
            buffers=buffers,
        )

    def audit_static_shared_memory(self, kernel_name: str = None) -> StaticSharedMemoryReport:
        """Early CPU legality gate for static shared memory.

        Raises ValueError before any CUDA/ptxas invocation when the sum of
        independent static shared allocations exceeds the 48 KiB static
        .shared limit that ptxas enforces. This never raises for the
        compile-time-known ``dynamic_shared`` floor, whose true size is
        host-evaluated at launch and raised by the device opt-in carveout.
        """
        report = self.static_shared_memory_report()
        if report.legal:
            return report
        kernel_name = kernel_name or (self.ctx.get(__LITTLE_KERNEL_ENTRY__).__name__
                                      if self.ctx.get(__LITTLE_KERNEL_ENTRY__) is not None else "<entry>")
        lines = [f"  {name}: {nbytes} bytes at offset {offset}" for name, offset, nbytes in report.buffers]
        raise ValueError(f"entry kernel {kernel_name!r} requests "
                         f"{report.static_bytes} bytes (0x{report.static_bytes:x}) of "
                         f"static shared memory, exceeding the "
                         f"{report.limit_bytes}-byte (0x{report.limit_bytes:x}) static "
                         f".shared limit ptxas enforces on every supported CUDA "
                         f"architecture (ir_shared_memory_static_resource_legality).\n" +
                         ("\n".join(lines) + "\n" if lines else "") +
                         "Independent scope=\"shared\" allocations are summed by the "
                         "compiler — they are never unioned or aliased, and a launch "
                         "shared_memory_bytes reservation configures the separate "
                         "`extern __shared__` (dynamic) region only; it cannot cover or "
                         "rewrite these static arrays.\n"
                         "Remediation (without raising hardware limits): fold buffers "
                         "with non-overlapping lifetimes into one allocation and slice "
                         "it, and/or declare buffers that must exceed 48 KiB with "
                         "scope=\"dynamic_shared\" (host launch reservation bounded by "
                         "the device opt-in shared-memory limit).")

    def print_summary(self) -> None:
        """Print a summary of shared memory usage and variable details."""
        print("=== Shared Memory Usage Summary ===")
        # Print total bytes per scope
        for scope in self.valid_scopes:
            print(f"Total {scope} memory: {self.scope_offsets[scope]} bytes, align to {self.align_bytes[scope]} bytes")

        for scope in self.valid_scopes:
            if self.first_mem_call[scope] is not None:
                print(f"First {scope} memory call: {ast.unparse(self.first_mem_call[scope])}")

        # Print detailed variable info
        print("\n=== Variable Details ===")
        for var_name, var in self.variables.items():
            print(f"Variable: {var['name']}\n"
                  f"  Expr: {ast.unparse(var['expr'])}\n"
                  f"  Scope: {var['scope']}\n"
                  f"  Dtype: {var['dtype']}\n"
                  f"  Elements: {var['total_elements']} (each {var['element_size']} bytes)\n"
                  f"  Total: {var['total_bytes']} bytes\n"
                  f"  Offset in {var['scope']}: {var['offset']} bytes\n"
                  f"  Reinterpret as: {var['dtype']} at offset {var['offset']} in {var['scope']} memory\n"
                  "-----------------------------------------")
