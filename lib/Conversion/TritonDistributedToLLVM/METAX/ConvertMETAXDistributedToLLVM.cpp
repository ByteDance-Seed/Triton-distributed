#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h"
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/GPUToMACA/GPUToMACAPass.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/MACADialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/TypeConverter.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_CONVERTMETAXDISTRIBUTEDTOLLVM
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace mlir::triton::METAX;

namespace {

class TritonLLVMConversionTarget : public ConversionTarget {
public:
  explicit TritonLLVMConversionTarget(MLIRContext &ctx)
      : ConversionTarget(ctx) {
    addLegalDialect<LLVM::LLVMDialect>();
    addLegalDialect<MACA::MACADialect>();
    addIllegalDialect<triton::TritonDialect>();
    addIllegalDialect<triton::gpu::TritonGPUDialect>();
    addLegalDialect<mlir::scf::SCFDialect>();
    addIllegalDialect<mlir::gpu::GPUDialect>();
    addIllegalDialect<triton::distributed::DistributedDialect>();
    addLegalOp<mlir::UnrealizedConversionCastOp>();
  }
};

struct ConvertMETAXDistributedToLLVM
    : public triton::impl::ConvertMETAXDistributedToLLVMBase<
          ConvertMETAXDistributedToLLVM> {
  ConvertMETAXDistributedToLLVM(int32_t computeCapability)
      : ConvertMETAXDistributedToLLVMBase({computeCapability}) {}

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    TargetInfo targetInfo(computeCapability);

    mlir::LowerToLLVMOptions option(context);
    option.overrideIndexBitwidth(32);
    TritonGPUToLLVMTypeConverter typeConverter(context, option);

    RewritePatternSet patterns(context);
    int benefit = patternBenefitPrioritizeOverLLVMConversions;

    // Distributed ops
    mlir::triton::METAX::populateDistributedOpToLLVMPatterns(
        typeConverter, patterns, benefit, targetInfo);
    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateMathToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateGpuToMACAConversionPatterns(typeConverter, patterns);
    mlir::cf::populateControlFlowToLLVMConversionPatterns(typeConverter,
                                                          patterns);
    mlir::triton::populateViewOpToLLVMPatterns(typeConverter, patterns,
                                               benefit);
    mlir::triton::populateAssertOpToLLVMPattern(typeConverter, patterns,
                                                targetInfo, benefit);
    mlir::triton::populateMemoryOpToLLVMPattern(typeConverter, targetInfo,
                                                patterns, benefit);
    mlir::triton::populateMakeRangeOpToLLVMPattern(typeConverter, targetInfo,
                                                   patterns, benefit);
    mlir::triton::METAX::populateDistributedOpToLLVMPatterns(
        typeConverter, patterns, benefit, targetInfo);
    TritonLLVMConversionTarget convTarget(*context);
    if (failed(applyPartialConversion(mod, convTarget, std::move(patterns))))
      return signalPassFailure();
  }
};

} // anonymous namespace

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMETAXDistributedToLLVMPass(int32_t computeCapability) {
  return std::make_unique<ConvertMETAXDistributedToLLVM>(computeCapability);
}

} // namespace triton
} // namespace mlir
