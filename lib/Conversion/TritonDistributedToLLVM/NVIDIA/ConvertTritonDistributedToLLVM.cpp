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
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h"
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTTRITONDISTRIBUTEDTOLLVM
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

// some code adapt from
// triton/third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TritonGPUToLLVM.cpp
class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<NVVM::NVVMDialect>();
    addLegalDialect<mlir::triton::nvgpu::NVGPUDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    // distributed and simt
    addIllegalDialect<triton::simt::SIMTDialect>();
    addIllegalDialect<triton::distributed::DistributedDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();

    // Warp specialization is lowered later.
    addLegalOp<triton::gpu::WarpSpecializeOp>();
    addLegalOp<triton::gpu::WarpYieldOp>();
    addLegalOp<triton::gpu::WarpSpecializePartitionsOp>();
    addLegalOp<triton::gpu::WarpReturnOp>();
  }
};

struct ConvertTritonDistributedToLLVM
    : public triton::impl::ConvertTritonDistributedToLLVMBase<
          ConvertTritonDistributedToLLVM> {
  using ConvertTritonDistributedToLLVMBase::ConvertTritonDistributedToLLVMBase;

  ConvertTritonDistributedToLLVM(int32_t computeCapability)
      : ConvertTritonDistributedToLLVMBase({computeCapability}) {}
  ConvertTritonDistributedToLLVM(int32_t computeCapability, int32_t ptxVersion)
      : ConvertTritonDistributedToLLVMBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability, ptxVersion);

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    // Distributed ops
    mlir::triton::NVIDIA::populateDistributedOpToLLVMPatterns(
        typeConverter, patterns, benefit, targetInfo);
    // SIMT ops
    mlir::triton::populateSIMTOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, benefit);
    // pattern of TritonDistributed need to be legalized to llvm through the
    // following pattern
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, benefit);
    mlir::arith::populateCeilFloorDivExpandOpsPatterns(patterns);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();

    // Ensure warp group code is isolated from above.
    makeAllWarpGroupsIsolatedFromAbove(mod);
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToLLVMPass() {
  return std::make_unique<ConvertTritonDistributedToLLVM>();
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertTritonDistributedToLLVM>(computeCapability);
}
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToLLVMPass(int32_t computeCapability,
                                         int32_t ptxVersion) {
  return std::make_unique<ConvertTritonDistributedToLLVM>(computeCapability,
                                                          ptxVersion);
}

} // namespace triton
} // namespace mlir
