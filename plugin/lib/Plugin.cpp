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

/// Triton-distributed plugin entry point.
///
/// Bundles the Distributed + SIMT dialects and their conversion passes into a
/// single shared library loadable via TRITON_PLUGIN_PATHS. The op *builders* do
/// NOT live here -- they are in the companion module (libtriton_dist_ext); this
/// .so owns only the dialects, passes, and the AxisInfo visitor registration.
///
/// NOTE: this build targets Triton 3.7.1 with two minimal, op-agnostic source
/// patches that are unavoidable for parity (neither is expressible via the
/// plugin mechanism): (1) the AxisInfo visitor-registration hook (see
/// AxisInfoVisitors.cpp) that keeps load/store vectorization on distributed
/// pointers, and (2) the software-pipeliner predication hook (see
/// PipelinerPredication.cpp) that lets the pipeliner predicate
/// distributed.wait. All other extension points use the unpatched plugin
/// mechanism.
///
/// Follows the hand-written PluginInfo style of triton-ext/utlx so that one
/// .so can register multiple dialects and passes.

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/DialectRegistry.h"
#include "triton/Tools/PluginUtils.h"
#include "triton/Version.h"

#include "AxisInfoVisitors.h"
#include "PipelinerPredication.h"
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"

#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h"
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/TritonDistributedToLLVMPass.h"
#include "TritonDistributed/Conversion/TritonDistributedToTritonGPU/Passes.h"
#include "TritonDistributed/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPUPass.h"
#include <string>

#ifndef TRITON_DIST_VERSION
#define TRITON_DIST_VERSION "0.1.0"
#endif

using namespace mlir::triton;

// ---------------------------------------------------------------------------
// Pass registration
//
// The full lowering is always compiled (see plugin/CMakeLists.txt): the
// Distributed/SIMT -> TritonGPU conversion (shared by all backends) plus the
// Distributed/SIMT -> LLVM conversion. Only the LLVM stage differs by
// toolchain: NVIDIA + AMD by default, or METAX when USE_MACA is defined (a
// distinct toolchain that replaces NVIDIA/AMD, mirroring Triton's own
// TRITON_USE_MACA).
// ---------------------------------------------------------------------------
// addPass args: [target, numWarps, threadsPerWarp, numCTAs, enableSourceRemat]
static void
addConvertTritonDistributedToTritonGPU(mlir::PassManager *pm,
                                       const std::vector<std::string> &args) {
  std::string target = args.size() >= 1 ? args[0] : "";
  int numWarps = args.size() >= 2 ? std::stoi(args[1]) : 4;
  int threadsPerWarp = args.size() >= 3 ? std::stoi(args[2]) : 32;
  int numCTAs = args.size() >= 4 ? std::stoi(args[3]) : 1;
  bool enableSourceRemat = args.size() >= 5 ? (std::stoi(args[4]) != 0) : false;
  pm->addPass(mlir::triton::createConvertTritonDistributedToTritonGPUPass(
      target, numWarps, threadsPerWarp, numCTAs, enableSourceRemat));
}
static void registerConvertTritonDistributedToTritonGPUFn() {
  mlir::triton::registerConvertTritonDistributedToTritonGPU();
}

#ifdef USE_MACA
// addPass args: [computeCapability]
static void
addConvertMETAXDistributedToLLVM(mlir::PassManager *pm,
                                 const std::vector<std::string> &args) {
  int computeCapability = args.size() >= 1 ? std::stoi(args[0]) : 80;
  pm->addPass(
      mlir::triton::createConvertMETAXDistributedToLLVMPass(computeCapability));
}
static void registerConvertMETAXDistributedToLLVMFn() {
  mlir::triton::registerConvertMETAXDistributedToLLVM();
}
#else
// NVIDIA: addPass args: [computeCapability, ptxVersion]
static void
addConvertTritonDistributedToLLVM(mlir::PassManager *pm,
                                  const std::vector<std::string> &args) {
  int computeCapability = args.size() >= 1 ? std::stoi(args[0]) : 80;
  int ptxVersion = args.size() >= 2 ? std::stoi(args[1]) : 80;
  pm->addPass(mlir::triton::createConvertTritonDistributedToLLVMPass(
      computeCapability, ptxVersion));
}
static void registerConvertTritonDistributedToLLVMFn() {
  mlir::triton::registerConvertTritonDistributedToLLVM();
}
// The AMD backend lowers distributed/SIMT ops with a dedicated pass (a superset
// of the standard TritonGPU->LLVM `add_to_llvmir`), plus an "ext" builtin-func
// pass that lowers the distributed `__triton_hip_*` extern calls (ld/st/atomic/
// syncthreads/v4_b32). These mirror the fork's `add_distributed_to_llvm` /
// `add_builtin_func_to_llvmir_ext` (see the removed python/src/passes.cc); the
// AMD frontend hook (jit.py) splices them into HIPBackend.make_llir.
// addPass args: [arch, ftz]
static void
addConvertAMDDistributedToLLVM(mlir::PassManager *pm,
                               const std::vector<std::string> &args) {
  std::string arch = args.size() >= 1 ? args[0] : "";
  bool ftz = args.size() >= 2 ? (std::stoi(args[1]) != 0) : true;
  pm->addPass(mlir::triton::createConvertAMDDistributedToLLVMPass(arch, ftz));
}
static void registerConvertAMDDistributedToLLVMFn() {
  mlir::triton::registerConvertAMDDistributedToLLVM();
}
// addPass args: [ftz]
static void
addConvertBuiltinFuncToLLVMExt(mlir::PassManager *pm,
                               const std::vector<std::string> &args) {
  bool ftz = args.size() >= 1 ? (std::stoi(args[0]) != 0) : true;
  pm->addPass(mlir::triton::createConvertBuiltinFuncToLLVMExtPass(ftz));
}
static void registerConvertBuiltinFuncToLLVMExtFn() {
  mlir::triton::registerConvertBuiltinFuncToLLVMExt();
}
#endif

// ---------------------------------------------------------------------------
// Dialect registration
// ---------------------------------------------------------------------------
static void registerDistributedDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::triton::distributed::DistributedDialect>();
}
static void registerSIMTDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::triton::simt::SIMTDialect>();
}
// The SIMT extract/insert op builders emit builtin MLIR tensor dialect ops
// (tensor.extract / tensor.insert) for per-thread element access. Upstream
// Triton does not load the tensor dialect into its context, so register it
// here to make those ops available in the frontend MLIRContext.
static void registerTensorDialect(mlir::DialectRegistry *registry) {
  registry->insert<mlir::tensor::TensorDialect>();
}

// ---------------------------------------------------------------------------
// Plugin entry point
// ---------------------------------------------------------------------------
TRITON_PLUGIN_API plugin::PluginInfo *tritonGetPluginInfo() {
  // Register the distributed AxisInfo visitors and the pipeliner predicator
  // once, at plugin load. loadPlugins calls this after libtriton is resident,
  // so the patched extension-hook symbols resolve.
  static const bool extensionsRegistered = [] {
    ::triton::dist::registerDistributedAxisInfoVisitors();
    ::triton::dist::registerDistributedPipelinerPredication();
    return true;
  }();
  (void)extensionsRegistered;

  static plugin::DialectInfo dialects[] = {
      {"DistributedDialect", TRITON_DIST_VERSION, registerDistributedDialect},
      {"SIMTDialect", TRITON_DIST_VERSION, registerSIMTDialect},
      {"TensorDialect", TRITON_DIST_VERSION, registerTensorDialect},
  };

  static plugin::PassInfo passes[] = {
      {"convert_triton_distributed_to_tritongpu", TRITON_DIST_VERSION,
       addConvertTritonDistributedToTritonGPU,
       registerConvertTritonDistributedToTritonGPUFn},
#ifdef USE_MACA
      {"convert_metax_distributed_to_llvm", TRITON_DIST_VERSION,
       addConvertMETAXDistributedToLLVM,
       registerConvertMETAXDistributedToLLVMFn},
#else
      {"convert_triton_distributed_to_llvm", TRITON_DIST_VERSION,
       addConvertTritonDistributedToLLVM,
       registerConvertTritonDistributedToLLVMFn},
      {"convert_amd_distributed_to_llvm", TRITON_DIST_VERSION,
       addConvertAMDDistributedToLLVM, registerConvertAMDDistributedToLLVMFn},
      {"convert_builtin_func_to_llvmir_ext", TRITON_DIST_VERSION,
       addConvertBuiltinFuncToLLVMExt, registerConvertBuiltinFuncToLLVMExtFn},
#endif
  };

  // No OpInfo ops are registered. On the pinned Triton 3.7.1, the plugin-op
  // binding in ir.cc passes its argument vector by value with no result slot
  // and no return value (see ext/py_module.cpp for the exact binding), so it
  // can neither realise the `operands[0]=result slot` convention the builders
  // use nor return a created Value to Python. Every distributed op is therefore
  // driven from the companion pybind11 module (libtriton_dist_ext), which owns
  // the op builders and applies the slot/return convention locally. This plugin
  // .so owns only the dialects and passes -- the parts of the plugin mechanism
  // that work natively on 3.7.1.

  static plugin::PluginInfo info = {
      TRITON_PLUGIN_API_VERSION,
      "TritonDistributed",
      TRITON_DIST_VERSION,
      passes,
      sizeof(passes) / sizeof(passes[0]),
      dialects,
      sizeof(dialects) / sizeof(dialects[0]),
      /*ops=*/nullptr,
      /*numOps=*/0,
      TRITON_VERSION,
  };
  return &info;
}
