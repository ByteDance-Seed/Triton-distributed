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
from typing import Optional

from little_kernel.codegen.codegen_base import CppEmitter
import little_kernel.language as ll
from little_kernel.core.passes.utils.type_inference import TypeInferencer
from little_kernel.core.internal import __LITTLE_KERNEL_ENTRY__
from little_kernel.backends.amd.cdna4.core.types import AmdBufferRsrcType
from little_kernel.core.type_system import VectorType

# ext_vector_type typedef name and C element type of each supported VectorType
_VECTOR_ELEMENT_CPP = {"int32": "int", "uint32": "unsigned", "float32": "float"}


def hip_vector_type_name(vec: VectorType) -> str:
    return f"lk_{vec.element_type}x{vec.lanes}"


def _vector_typedefs() -> str:
    lines = []
    for elem, cpp in _VECTOR_ELEMENT_CPP.items():
        for lanes in (2, 4, 8, 16):
            lines.append(f"typedef {cpp} lk_{elem}x{lanes} __attribute__((ext_vector_type({lanes})));")
    return "\n".join(lines)


class HipEmitter(CppEmitter):
    """HIP C++ emitter for AMD GPUs (gfx9 CDNA).

    Generates HIP source that uses clang AMDGPU builtins and inline AMDGCN assembly; no library
    dependency beyond the HIP headers.
    """

    def __init__(self, ctx, all_types, num_threads=None, disable_var_hoisting=False, waves_per_eu=None):
        super().__init__(ctx, all_types)
        self.num_threads = num_threads
        self.disable_var_hoisting = disable_var_hoisting
        self.waves_per_eu = waves_per_eu
        headers = [
            "<hip/hip_runtime.h>",
            "<hip/hip_bf16.h>",
            "<hip/hip_fp16.h>",
            "<hip/hip_fp8.h>",
            "<tuple>",
        ]
        for hdr in headers:
            self.header_cache.add(hdr)
            self.header_buffer.write(f"#include {hdr}\n")
        self.header_buffer.write("\n" + _vector_typedefs() + "\n")
        # LDS pointer for the global->LDS DMA builtins
        self.header_buffer.write("typedef __attribute__((address_space(3))) void* lk_lds_ptr_t;\n\n")

    def _lltype_to_cpp(self, ll_type: "ll.LLType") -> str:
        """Convert an LLType instance to a HIP C++ type string."""
        if hasattr(ll_type, "kind") and ll_type.kind == "int":
            if ll_type.special == "bool":
                return "bool"
            if ll_type.special == "binary" or ll_type.bits == 4:
                raise NotImplementedError("4-bit integer types are not supported by the HIP backend")
            sign_prefix = "u" if not ll_type.signed else ""
            bit_map = {
                8: f"{sign_prefix}int8_t",
                16: f"{sign_prefix}int16_t",
                32: f"{sign_prefix}int32_t",
                64: f"{sign_prefix}int64_t",
                128: "unsigned __int128" if not ll_type.signed else "__int128",
            }
            return bit_map[ll_type.bits]

        elif hasattr(ll_type, "kind") and ll_type.kind == "float":
            float_map = {
                "fp8_e5m2": "__hip_fp8_e5m2",
                "fp8_e4m3": "__hip_fp8_e4m3",
                "bfloat16": "__hip_bfloat16",
                "float16": "__half",
                "float32": "float",
                "float64": "double",
            }
            if ll_type.fmt not in float_map:
                raise NotImplementedError(f"{ll_type.fmt} is not supported by the HIP backend")
            return float_map[ll_type.fmt]

        elif hasattr(ll_type, "kind") and ll_type.kind == "void":
            return "void"

        elif hasattr(ll_type, "kind") and ll_type.kind == "str":
            return "std::string"

        elif isinstance(ll_type, VectorType):
            if str(ll_type.element_type) not in _VECTOR_ELEMENT_CPP or ll_type.lanes not in (2, 4, 8, 16):
                raise NotImplementedError(f"vector type {ll_type} is not supported by the HIP backend")
            return hip_vector_type_name(ll_type)

        elif isinstance(ll_type, AmdBufferRsrcType):
            return "__amdgpu_buffer_rsrc_t"

        elif ll_type.is_const():  # HIP has no __grid_constant__: kernel arguments are already constant
            return f"const {self._lltype_to_cpp(ll_type.inner_type)}"

        elif ll_type.is_pointer():
            return f"{self._lltype_to_cpp(ll_type.inner_type)}*"

        elif ll_type.is_tensor():
            return f"{self._lltype_to_cpp(ll_type.element_type)}*"

        elif ll_type.is_struct():
            if ll_type not in self.struct_types:
                self.struct_types[ll_type] = self._generate_struct_name(ll_type)
            return self.struct_types[ll_type]

        elif ll_type.is_tuple():
            elem_cpps = [self._lltype_to_cpp(t) for t in ll_type.element_types]
            return f"std::tuple<{', '.join(elem_cpps)}>"

        raise NotImplementedError(
            f"Unsupported LLType for the HIP backend: {type(ll_type).__name__} (value: {str(ll_type)})")

    def _kernel_attributes(self) -> str:
        attrs = []
        if self.num_threads is not None:
            attrs.append(f"amdgpu_flat_work_group_size({self.num_threads}, {self.num_threads})")
        if self.waves_per_eu is not None:
            attrs.append(f"amdgpu_waves_per_eu({self.waves_per_eu}, {self.waves_per_eu})")
        return f"__attribute__(({', '.join(attrs)})) " if attrs else ""

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return_cpp = self._lltype_to_cpp(self.get_type(node.returns))
        cpp_params = []
        old_scope_vars = self.scope_vars
        self.scope_vars = {}
        for arg in node.args.args:
            arg_lltype = self.get_type(arg.annotation)
            cpp_params.append(f"{self._lltype_to_cpp(arg_lltype)} {arg.arg}")
            self.scope_vars[arg.arg] = arg_lltype

        sig = f"{return_cpp} {node.name}({', '.join(cpp_params)}) {{"
        entry = self.ctx.get(__LITTLE_KERNEL_ENTRY__)
        if entry is not None and entry.__name__ == node.name:
            # Only the __global__ entry takes C linkage so that hipModuleGetFunction
            # resolves it via an unmangled lookup; device helpers keep C++ linkage,
            # which avoids cross-TU ODR collisions on unmangled helper names and keeps
            # the C++ templates/overloads pulled in from the HIP headers legal.
            sig = f"extern \"C\" __global__ {self._kernel_attributes()}{sig}"
        else:
            sig = f"__device__ __forceinline__ {sig}"
        self.writeln_main(sig)
        self.indent_level += 1
        for stmt in node.body:
            self.visit(stmt)
        self.indent_level -= 1
        self.writeln_main("}")
        self.writeln_main("")
        self.scope_vars = old_scope_vars

    def get_code(self, need_header=True) -> str:
        """Combine headers, structs, builtins and the kernel(s)."""
        self._emit_struct_definitions()
        main_code = self.main_buffer.getvalue()
        return (("/*start of generated code*/\n" + self.header_buffer.getvalue() if need_header else "") + "\n" +
                self.struct_buffer.getvalue() + "\n" + self.builtin_buffer.getvalue() + "\n" + main_code)


def codegen_hip(tree: ast.AST, ctx=None, emit_header=True, num_threads=None, disable_var_hoisting=False,
                waves_per_eu: Optional[int] = None) -> str:
    """Generate HIP C++ from a lowered LittleKernel AST.

    Parameters
    ----------
    num_threads : int, optional
        Work-group size; emitted as ``amdgpu_flat_work_group_size(n, n)`` on the kernel.
    waves_per_eu : int, optional
        Emitted as ``amdgpu_waves_per_eu(w, w)``: the register budget the compiler may use per wave
        (1: all 512 VGPR + AGPR of a SIMD lane).
    disable_var_hoisting : bool
        Keep declarations at their first assignment instead of hoisting them out of blocks.
    """
    if ctx is None:
        ctx = {}
    inferencer = TypeInferencer(ctx=ctx, scope_vars={})
    inferencer.visit(tree)
    emitter = HipEmitter(ctx=ctx, all_types=inferencer.all_types, num_threads=num_threads,
                         disable_var_hoisting=disable_var_hoisting, waves_per_eu=waves_per_eu)
    from little_kernel.codegen.policy import emission_scope
    with emission_scope(emitter):
        emitter.visit(tree)
        return emitter.get_code(need_header=emit_header)
