/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
#include "mlir/Transforms/Passes.h"
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h"
#include "TritonDistributed/Conversion/TritonDistributedToTritonGPU/Passes.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "python/src/passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_distributed_passes_ttgpuir_for_nvidia(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_2("add_convert_triton_distributed_to_llvm",
                     createConvertTritonDistributedToLLVMPass, int, int);
}

void init_triton_distributed_passes_ttgpuir_for_amd(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_1("add_lib_device_to_llvmir",
                     createConvertLibDeviceToLLVMPass, bool);
  ADD_PASS_WRAPPER_2("add_distributed_to_llvm",
                     createConvertAMDDistributedToLLVMPass, const std::string &,
                     bool);
}

void init_triton_distributed_passes_ttir(py::module &&m) {
  using namespace mlir::triton;
  ADD_PASS_WRAPPER_4("add_convert_to_ttgpuir_ext",
                     createConvertTritonDistributedToTritonGPUPass,
                     const std::string &, int, int, int);
}

void init_triton_distributed_passes(py::module &&m) {
  init_triton_distributed_passes_ttir(m.def_submodule("ttir"));
  auto ttgpuir = m.def_submodule("ttgpuir");
  init_triton_distributed_passes_ttgpuir_for_nvidia(
      ttgpuir.def_submodule("nvidia"));
  init_triton_distributed_passes_ttgpuir_for_amd(ttgpuir.def_submodule("amd"));
}
