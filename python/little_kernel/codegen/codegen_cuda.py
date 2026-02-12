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
from .codegen_base import CppEmitter
import little_kernel.language as ll
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__


class CudaEmitter(CppEmitter):
    """Unified CUDA code emitter.

    Generates standalone CUDA code using raw PTX intrinsics.
    No CuTe/CUTLASS dependency. Uses CUtensorMap directly for TMA descriptors.
    """

    def __init__(self, ctx, all_types, num_threads=None):
        super().__init__(ctx, all_types)
        self.num_threads = num_threads
        headers = [
            "<cuda_runtime.h>",
            "<cuda_bf16.h>",
            "<cuda_fp16.h>",  # __half (float16)
            "<cuda.h>",
        ]
        for hdr in headers:
            self.header_cache.add(hdr)
            self.header_buffer.write(f"#include {hdr}\n")
        # TmaDescriptor is CUtensorMap from the CUDA Driver API
        self.header_buffer.write("\nusing TmaDescriptor = CUtensorMap;\n\n")

    def _lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a valid C++ type string."""
        # ------------------------------ Scalar Types ------------------------------
        if hasattr(ll_type, "kind") and ll_type.kind == "int":  # IntType
            if ll_type.special == "bool":
                return "bool"
            if ll_type.special == "binary" or ll_type.bits == 4:
                # TODO: Add proper 4-bit (int4/uint4) support. CUDA has native 4-bit types.
                raise NotImplementedError("4-bit integer types (int4/uint4, binary) are not yet supported. "
                                          "CUDA has native 4-bit support; this will be added in a future release.")
            # Map to stdint types (e.g., uint32 -> uint32_t)
            sign_prefix = "u" if not ll_type.signed else ""
            bit_map = {
                8: f"{sign_prefix}int8_t", 16: f"{sign_prefix}int16_t", 32: f"{sign_prefix}int32_t", 64:
                f"{sign_prefix}int64_t", 128: "unsigned __int128" if not ll_type.signed else "__int128"
            }
            return bit_map[ll_type.bits]

        elif hasattr(ll_type, "kind") and ll_type.kind == "float":  # FloatType
            float_map = {
                "fp4_e2m1": "__nv_fp4_e2m1", "fp8_e5m2": "__nv_fp8_e5m2", "fp8_e4m3": "__nv_fp8_e4m3", "bfloat16":
                "__nv_bfloat16", "float16": "__half",
                # "tfloat32": "tfloat32_t",
                "float32": "float", "float64": "double"
            }
            return float_map[ll_type.fmt]

        elif hasattr(ll_type, "kind") and ll_type.kind == "void":  # VoidType
            return "void"

        elif hasattr(ll_type, "kind") and ll_type.kind == "str":  # StringType
            return "std::string"

        # ------------------------------ Annotated Types ------------------------------
        elif ll_type.is_const():  # Const/GridConstant
            inner_cpp = self._lltype_to_cpp(ll_type.inner_type)
            return f"__grid_constant__ {inner_cpp}" if ll_type.is_grid_constant() else f"const {inner_cpp}"

        elif ll_type.is_pointer():  # Pointer
            inner_cpp = self._lltype_to_cpp(ll_type.inner_type)
            return f"{inner_cpp}*"

        # ------------------------------ Composite Types ------------------------------
        elif ll_type.is_tensor():  # TensorType (map to pointer for simplicity)
            elem_cpp = self._lltype_to_cpp(ll_type.element_type)
            return f"{elem_cpp}*"

        elif ll_type.is_struct():  # StructType
            if ll_type not in self.struct_types:
                self.struct_types[ll_type] = self._generate_struct_name(ll_type)
            return self.struct_types[ll_type]

        elif str(ll_type) == "TmaDescriptor":  # TmaDescriptorType
            return "TmaDescriptor"

        elif str(ll_type) == "Wgmma":  # WgmmaType
            raise NotImplementedError("WgmmaType is not supported for C++ generation")

        # ------------------------------ Unsupported LLType ------------------------------
        else:
            raise NotImplementedError(f"Unsupported LLType: {type(ll_type).__name__} (value: {str(ll_type)})")

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Process function definition: initialize scope with parameters, then process body."""
        # Resolve return type
        return_lltype = self.get_type(node.returns)
        return_cpp = self._lltype_to_cpp(return_lltype)

        # Resolve parameters and populate initial scope
        cpp_params = []
        old_scope_vars = self.scope_vars
        self.scope_vars = {}  # Reset scope for new function
        for arg in node.args.args:
            arg_lltype = self.get_type(arg.annotation)
            arg_cpp = self._lltype_to_cpp(arg_lltype)
            cpp_params.append(f"{arg_cpp} {arg.arg}")
            # Add parameter to scope (prevents re-declaration in body)
            self.scope_vars[arg.arg] = arg_lltype

        # Write function signature
        sig = f"{return_cpp} {node.name}({', '.join(cpp_params)}) {{"
        if __LITTLE_KERNEL_ENTRY__ in self.ctx and self.ctx[__LITTLE_KERNEL_ENTRY__].__name__ == node.name:
            # Add __launch_bounds__ if num_threads is provided
            if self.num_threads is not None:
                sig = f"__global__ __launch_bounds__({self.num_threads}, 1) {sig}"
            else:
                sig = f"__global__ {sig}"
        else:
            sig = f"__device__ {sig}"
        self.writeln_main(sig)
        self.indent_level += 1

        # Process function body (statements like Assign, Return)
        for stmt in node.body:
            self.visit(stmt)

        # Cleanup: close function and reset scope
        self.indent_level -= 1
        self.writeln_main("}")
        self.writeln_main("")
        self.scope_vars = old_scope_vars  # Restore scope after function

    def get_code(self, need_header=True) -> str:
        """Combine headers, structs, and main code into final CUDA output with extern "C" wrapper."""
        self._emit_struct_definitions()  # Emit structs before main code

        # Wrap main code in extern "C" to avoid C++ name mangling for kernel functions
        main_code = self.main_buffer.getvalue()
        if main_code.strip():
            # Only wrap if there's actual code
            main_code = "extern \"C\" {\n" + main_code + "}\n"

        return (("/*start of generated code*/\n" + self.header_buffer.getvalue() if need_header else "") + "\n" +
                self.struct_buffer.getvalue() + "\n" + self.builtin_buffer.getvalue() + "\n" + main_code)


def codegen_cuda(tree: ast.AST, ctx=None, emit_header=True, num_threads=None) -> str:
    """Generate CUDA code from AST.
    
    Parameters
    ----------
    tree : ast.AST
        Python AST to generate code from
    ctx : dict, optional
        Context dictionary (e.g., function.__globals__)
    emit_header : bool, optional
        Whether to emit header includes (default: True)
    num_threads : int, optional
        Number of threads per block for __launch_bounds__ attribute.
        If provided, will add __launch_bounds__(num_threads) to __global__ functions.
        If None, no __launch_bounds__ will be generated.
    
    Returns
    -------
    str
        Generated CUDA code
    """
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    all_types = inferencer.all_types
    emitter = CudaEmitter(ctx=ctx, all_types=all_types, num_threads=num_threads)
    emitter.visit(tree)
    return emitter.get_code(need_header=emit_header)
