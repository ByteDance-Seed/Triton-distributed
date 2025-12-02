/*
 * Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
 */
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <cassert>

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTBUILTINFUNCTOLLVMEXT
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h.inc"
} // namespace mlir::triton

using namespace mlir;

namespace {

// adapt from
// triton/third_party/amd/lib/TritonAMDGPUToLLVM/BuiltinFuncToLLVM.cpp
class CallOpConversion : public OpRewritePattern<LLVM::CallOp> {
public:
  CallOpConversion(mlir::MLIRContext *context, bool ftz)
      : OpRewritePattern<LLVM::CallOp>(context, 1), ftz(ftz) {}

  LogicalResult
  matchAndRewrite(LLVM::CallOp callOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (isWrappedLLVMIntrinsic(callOp)) {
      return convertToLLVMIntrinsic(callOp, rewriter);
    } else {
      return failure();
    }
  }

private:
  bool isWrappedLLVMIntrinsic(LLVM::CallOp callOp) const {
    if (std::optional<StringRef> callee = callOp.getCallee()) {
      if (callee.value().starts_with("__triton_hip_")) {
        return true;
      }
    }
    return false;
  }

  std::optional<SmallVector<StringRef>>
  matchPrefixAndSplitRemainder(StringRef name, StringRef prefix) const {
    if (name.starts_with(prefix)) {
      StringRef remainder = name.substr(prefix.size());
      SmallVector<StringRef> parts;

      static const SmallVector<StringRef> protectedPatterns = {"acq_rel",
                                                               "seq_cst"};

      while (!remainder.empty()) {
        bool foundProtected = false;

        for (StringRef pattern : protectedPatterns) {
          if (remainder.starts_with(pattern)) {
            if (remainder.size() == pattern.size() ||
                remainder[pattern.size()] == '_') {
              parts.push_back(remainder.substr(0, pattern.size()));
              remainder = remainder.substr(pattern.size());
              if (remainder.starts_with('_')) {
                remainder = remainder.substr(1);
              }
              foundProtected = true;
              break;
            }
          }
        }

        if (foundProtected)
          continue;

        size_t pos = remainder.find('_');
        if (pos == StringRef::npos) {
          parts.push_back(remainder);
          break;
        }
        parts.push_back(remainder.substr(0, pos));
        remainder = remainder.substr(pos + 1);
      }

      return parts;
    }
    return std::nullopt;
  }

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();
    auto strToMemoryOrder = [](StringRef str) {
      if (str == "monotonic" || str == "relaxed")
        return LLVM::AtomicOrdering::monotonic;
      else if (str == "acquire")
        return LLVM::AtomicOrdering::acquire;
      else if (str == "release")
        return LLVM::AtomicOrdering::release;
      else if (str == "acq_rel")
        return LLVM::AtomicOrdering::acq_rel;
      else if (str == "seq_cst")
        return LLVM::AtomicOrdering::seq_cst;
      else
        llvm_unreachable("unknown memory order string");
    };

    auto strToScope = [](StringRef str) {
      if (str == "workgroup")
        return "workgroup-one-as";
      else if (str == "agent")
        return "agent-one-as";
      else if (str == "system")
        return "";
      else
        llvm_unreachable("unknown scope string");
    };

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();
    auto buildAtomicLoad =
        [&rewriter, &loc,
         returnType](Value inputPtr, LLVM::AtomicOrdering ordering,
                     std::optional<StringRef> scopeStr = std::nullopt) {
          assert(llvm::isa<LLVM::LLVMPointerType>(inputPtr.getType()) &&
                 "expected pointer type for atomic load");
          int alignment = returnType.getIntOrFloatBitWidth() / 8;
          return rewriter.create<LLVM::LoadOp>(
              loc, returnType, inputPtr, /*alignment=*/alignment,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariant =*/false, /*isInvariantGroup=*/false, ordering,
              scopeStr.value_or(StringRef()));
        };

    auto buildAtomicStore =
        [&rewriter, &loc](Value value, Value inputPtr,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          int32_t alignment = value.getType().getIntOrFloatBitWidth() / 8;
          return rewriter.create<LLVM::StoreOp>(
              loc, value, inputPtr, /*alignment=*/alignment,
              /*isVolatile =*/false, /*isNonTemporal*/ false,
              /*isInvariantGroup=*/false, ordering,
              scopeStr.value_or(StringRef()));
        };

    auto buildAtomicFetchAdd =
        [&rewriter, &loc](Value atomicAddr, Value value,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          int32_t alignment = value.getType().getIntOrFloatBitWidth() / 8;
          return rewriter.create<LLVM::AtomicRMWOp>(
              loc, LLVM::AtomicBinOp::add, atomicAddr, value, ordering,
              scopeStr.value_or(StringRef()), /*alignment=*/alignment);
        };

    auto buildAtomicCompareExchangeStrong =
        [&rewriter, &loc](Value atomicAddr, Value cmpVal, Value value,
                          LLVM::AtomicOrdering successOrdering,
                          LLVM::AtomicOrdering failureOrdering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          int32_t alignment = cmpVal.getType().getIntOrFloatBitWidth() / 8;
          auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
              loc, atomicAddr, cmpVal, value, successOrdering, failureOrdering,
              scopeStr.value_or(StringRef()), /*alignment=*/alignment);
          auto atomPtrVal = rewriter.create<LLVM::ExtractValueOp>(
              loc, cmpxchg, SmallVector<int64_t>{0});
          return atomPtrVal;
        };

    Operation *replacementOp = nullptr;
    // syncthreads
    if (calleeName == "__triton_hip_syncthreads") {
      assert(operands.size() == 0);
      (void)rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::release,
                                           "workgroup");
      (void)rewriter.create<ROCDL::SBarrierOp>(loc);
      (void)rewriter.create<LLVM::FenceOp>(loc, LLVM::AtomicOrdering::acquire,
                                           "workgroup");
      Value zero = rewriter.create<LLVM::ConstantOp>(
          loc, i64_ty, IntegerAttr::get(i64_ty, 0));
      replacementOp = zero.getDefiningOp();
    }

    // load
    if (auto maybeParts =
            matchPrefixAndSplitRemainder(calleeName, "__triton_hip_load_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 2 &&
             "expected load function to have 2 parts after prefix");
      LLVM::AtomicOrdering memOrder = strToMemoryOrder(parts[0]);
      auto scopeStr = strToScope(parts[1]);
      assert(operands.size() == 1 && "expected load to have 1 operand");

      replacementOp = buildAtomicLoad(operands[0], memOrder, scopeStr);
    }

    // store
    else if (auto maybeParts = matchPrefixAndSplitRemainder(
                 calleeName, "__triton_hip_store_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 2 &&
             "expected store function to have 2 parts after prefix");
      LLVM::AtomicOrdering memOrder = strToMemoryOrder(parts[0]);
      auto scopeStr = strToScope(parts[1]);
      assert(operands.size() == 2 && "expected store to have 2 operands");
      buildAtomicStore(operands[1], operands[0], memOrder, scopeStr);
      rewriter.eraseOp(callOp);
      return mlir::success();
    }

    // atomic add
    else if (auto maybeParts = matchPrefixAndSplitRemainder(
                 calleeName, "__triton_hip_atom_add_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 2 &&
             "expected atomic add function to have 2 parts after prefix");
      assert(operands.size() == 2 && "expected atomic add to have 2 operands");
      LLVM::AtomicOrdering memOrder = strToMemoryOrder(parts[0]);
      auto scopeStr = strToScope(parts[1]);
      replacementOp =
          buildAtomicFetchAdd(operands[0], operands[1], memOrder, scopeStr);
    }

    // atomic cas
    else if (auto maybeParts = matchPrefixAndSplitRemainder(
                 calleeName, "__triton_hip_atom_cas_")) {
      auto parts = maybeParts.value();
      assert(parts.size() == 3 &&
             "expected atomic cas function to have 3 parts after prefix");
      assert(operands.size() == 3 && "expected atomic cas to have 3 operands");
      LLVM::AtomicOrdering successOrdering = strToMemoryOrder(parts[0]);
      LLVM::AtomicOrdering failureOrdering = strToMemoryOrder(parts[1]);
      auto scopeStr = strToScope(parts[2]);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], successOrdering,
          failureOrdering, scopeStr);
    }

    if (replacementOp) {
      rewriter.replaceOp(callOp, replacementOp);
      return mlir::success();
    }

    return mlir::failure();
  }

private:
  bool ftz;
};

struct ConvertBuiltinFuncToLLVMExt
    : public triton::impl::ConvertBuiltinFuncToLLVMExtBase<
          ConvertBuiltinFuncToLLVMExt> {
  explicit ConvertBuiltinFuncToLLVMExt(bool ftz) { this->ftz = ftz; }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    GreedyRewriteConfig config;
    config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Aggressive);

    RewritePatternSet patterns(context);
    patterns.add<CallOpConversion>(context, this->ftz);

    if (mlir::applyPatternsGreedily(mod, std::move(patterns), config)
            .failed()) {
      mod.emitError("failed to convert builtins/externs to llvm");
      signalPassFailure();
    }
  }
};

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMExtPass(bool ftz) {
  return std::make_unique<ConvertBuiltinFuncToLLVMExt>(ftz);
}

} // namespace mlir::triton
