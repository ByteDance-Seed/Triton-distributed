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
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/AMDGPU/Utils/Chipset.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Pass/Pass.h"
#include "third_party/amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "third_party/amd/lib/TritonAMDGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/SchedInstructions.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Analysis/Membar.h"
#include <algorithm>
#include <cstdlib>

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTAMDDISTRIBUTEDTOLLVM
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::NVIDIA;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<ROCDL::ROCDLDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addIllegalDialect<triton::nvidia_gpu::TritonNvidiaGPUDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<triton::distributed::DistributedDialect>();
    addIllegalDialect<triton::simt::SIMTDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
    addLegalOp<triton::amdgpu::InstructionSchedHint>();

    addDynamicallyLegalOp<ROCDL::BlockDimXOp, ROCDL::BlockDimYOp,
                          ROCDL::BlockDimZOp>([](Operation *op) {
      Type retTy = op->getResult(0).getType();
      return retTy.isInteger(64);
    });
  }
};

struct ConvertAMDDistributedToLLVM
    : public triton::impl::ConvertAMDDistributedToLLVMBase<
          ConvertAMDDistributedToLLVM> {
  explicit ConvertAMDDistributedToLLVM(StringRef targetArch, bool ftz) {
    this->arch = targetArch.str();
    this->ftz = ftz;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    AMD::TargetInfo targetInfo(this->arch.getValue());
    if (targetInfo.getISAFamily() == AMD::ISAFamily::Unknown) {
      mod.emitError("unsupported target: '") << this->arch.getValue() << "'";
      return signalPassFailure();
    }

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);

    TritonGPUToLLVMTypeConverter typeConverter(context, option, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);

    int numCTAs = triton::gpu::TritonGPUDialect::getNumCTAs(mod);
    int threadsPerWarp = triton::gpu::TritonGPUDialect::getThreadsPerWarp(mod);

    ModuleAxisInfoAnalysis axisInfoAnalysis(mod);

    // Emit logics to get threadId/blockIds/linearized clusterCTAId etc. and
    // cache the values. The reason to do it here is that cluster_ctaid is
    // currently implemented via inline asm, and thus cannot be CSEed.
    // clusterCTAId will be emitted only when numCTAs is larger than 1, and
    // other values will be DCEed if not used hereafter.
    OpBuilder::InsertPoint indexInsertPoint;

    RewritePatternSet patterns(context);
    int commonBenefit = patternBenefitPrioritizeOverLLVMConversions;
    int AMDBenefit = commonBenefit + 1;

    AMD::populateMemoryOpToLLVMPatterns(typeConverter, patterns, targetInfo,
                                        AMDBenefit);
    mlir::triton::populateMemoryOpToLLVMPatterns(typeConverter, targetInfo,
                                                 patterns, commonBenefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, commonBenefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, commonBenefit);
    mlir::triton::populateControlFlowOpToLLVMPattern(typeConverter, patterns,
                                                     targetInfo, commonBenefit);
    mlir::triton::populateSPMDOpToLLVMPattern(typeConverter, patterns,
                                              targetInfo, commonBenefit);
    AMD::populateSPMDOpToLLVMPattern(typeConverter, patterns, AMDBenefit);

    mlir::triton::AMD::populateTritonAMDGPUToLLVMPatterns(typeConverter,
                                                          patterns, AMDBenefit);
    mlir::triton::AMD::populateUpcastMXFPToLLVMPatterns(typeConverter, patterns,
                                                        targetInfo, AMDBenefit);

    // TODO(thomas): this should probably be done in a separate step to not
    // interfere with our own lowering of arith ops. Add arith/math's patterns
    // to help convert scalar expression to LLVM.
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);

    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    FailureOr<mlir::amdgpu::Chipset> maybeChipset =
        mlir::amdgpu::Chipset::parse(this->arch);
    if (failed(maybeChipset)) {
      emitError(UnknownLoc::get(&getContext()),
                "Invalid AMDGPU chipset name: " + this->arch);
      return signalPassFailure();
    }
    // Native lowering patterns
    mlir::populateGpuToROCDLConversionPatterns(
        typeConverter, patterns, mlir::gpu::amd::HIP, *maybeChipset);

    // Distributed ops - determine SHMEM backend from environment variable
    std::string shmemLibname = "librocshmem_device"; // default to rocshmem
    std::string shmemLibpath = "";

    // Check environment variable to determine backend
    const char *shmemBackendEnv = std::getenv("TRITON_DIST_SHMEM_BACKEND");
    if (shmemBackendEnv != nullptr) {
      std::string shmemBackend(shmemBackendEnv);
      // Convert to lowercase for case-insensitive comparison
      std::transform(shmemBackend.begin(), shmemBackend.end(),
                     shmemBackend.begin(), ::tolower);

      if (shmemBackend == "mori_shmem") {
        shmemLibname = "libmori_shmem_device";
      }
    }

    mlir::triton::AMD::populateDistributedOpToLLVMPatterns(
        typeConverter, patterns, commonBenefit, targetInfo, shmemLibname,
        shmemLibpath);

    // SIMT ops
    mlir::triton::populateSIMTOpToLLVMPatterns(typeConverter, targetInfo,
                                               patterns, commonBenefit);

    mlir::triton::AMD::populateROCDLBlockDimPattern(typeConverter, patterns,
                                                    commonBenefit);

    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertAMDDistributedToLLVMPass(StringRef targetArch,
                                                    bool ftz) {
  return std::make_unique<ConvertAMDDistributedToLLVM>(targetArch, ftz);
}
