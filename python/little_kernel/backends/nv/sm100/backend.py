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
"""Instantiate hardware components for this backend."""
from little_kernel.backends.interfaces import BackendComponents
from little_kernel.backends.shared import FunctionGenerator, FunctionToolchain, FunctionBuilder, StandardPassProvider, LazyRuntimeProvider, ArchitecturePolicy


def make_backend(spec):
    from little_kernel.backends.nv.common.codegen.emitter import codegen_cuda
    from little_kernel.backends.nv.common.runtime.compiler import compile_cuda
    from little_kernel.backends.nv.common.runtime.builder import build_kernel
    return BackendComponents(
        spec, FunctionGenerator(codegen_cuda), StandardPassProvider(gpu_allocations=True, static_unroll=False),
        FunctionToolchain('.cu', compile_cuda), FunctionBuilder(build_kernel, ('extra_hipcc_options', 'waves_per_eu')),
        LazyRuntimeProvider('little_kernel.backends.nv.common.runtime.cuda_runtime:CUDARuntime', 'device'),
        ArchitecturePolicy(spec, 'sm_100a', 'little_kernel.backends.nv.common.runtime.target:resolve_target',
                           'little_kernel.backends.nv.common.runtime.compiler:get_cuda_arch'))
