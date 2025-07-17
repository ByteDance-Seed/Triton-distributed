/*
 * Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
 */
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

namespace mlir::triton {
#define GEN_PASS_DEF_CONVERTLIBDEVICETOLLVM
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

  LogicalResult convertToLLVMIntrinsic(LLVM::CallOp callOp,
                                       mlir::PatternRewriter &rewriter) const {
    StringRef calleeName = callOp.getCallee().value();

    auto operands = callOp.getOperands();
    auto result = callOp.getResult();

    LLVM::LLVMFunctionType calleeType = callOp.getCalleeFunctionType();
    Type returnType = calleeType.getReturnType();

    auto loc = callOp.getLoc();
    auto buildAtomicLoad =
        [&rewriter, &loc](Type dtype, Value inputPtr, int align,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          return rewriter.create<LLVM::LoadOp>(
              loc, dtype, inputPtr, /*alignment=*/align,
              /*isVolatile=*/false, /*isNonTemporal=*/false,
              /*isInvariant =*/false, /*isInvariantGroup=*/false, ordering,
              scopeStr.value_or(StringRef()));
        };

    auto buildAtomicStore =
        [&rewriter, &loc](Value value, Value inputPtr, int align,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          return rewriter.create<LLVM::StoreOp>(
              loc, value, inputPtr, /*alignment=*/align,
              /*isVolatile =*/false, /*isNonTemporal*/ false,
              /*isInvariantGroup=*/false, ordering,
              scopeStr.value_or(StringRef()));
        };

    auto buildAtomicFetchAdd =
        [&rewriter, &loc](Value atomicAddr, Value value,
                          LLVM::AtomicOrdering ordering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          return rewriter.create<LLVM::AtomicRMWOp>(
              loc, LLVM::AtomicBinOp::add, atomicAddr, value, ordering,
              scopeStr.value_or(StringRef()), /*alignment=*/4);
        };

    auto buildAtomicCompareExchangeStrong =
        [&rewriter, &loc](Value atomicAddr, Value compare, Value value,
                          LLVM::AtomicOrdering successOrdering,
                          LLVM::AtomicOrdering failureOrdering,
                          std::optional<StringRef> scopeStr = std::nullopt) {
          // Prepare the value for the atomic operation.
          auto cmpVal = rewriter.create<LLVM::LoadOp>(loc, i32_ty, compare,
                                                      /*alignment=*/4);
          auto cmpxchg = rewriter.create<LLVM::AtomicCmpXchgOp>(
              loc, atomicAddr, cmpVal, value, successOrdering, failureOrdering,
              scopeStr.value_or(StringRef()), /*alignment=*/4);
          // Extract the result value and condition from the struct.
          // 0th is the old value at the ptr, 1st is the compare result: true if
          // equal.
          auto atomPtrVal = rewriter.create<LLVM::ExtractValueOp>(
              loc, cmpxchg, SmallVector<int64_t>{0});
          auto equalToCmpVal = rewriter.create<LLVM::ExtractValueOp>(
              loc, cmpxchg, SmallVector<int64_t>{1});

          Block *curBlock = rewriter.getInsertionBlock();
          Block *endBlock =
              rewriter.splitBlock(curBlock, rewriter.getInsertionPoint());
          Block *trueBlock = rewriter.createBlock(endBlock);
          rewriter.setInsertionPointToEnd(curBlock);
          rewriter.create<LLVM::CondBrOp>(loc, equalToCmpVal, trueBlock,
                                          endBlock);

          // If the compare was successful, store the value at the atomic
          // address.
          rewriter.setInsertionPointToStart(trueBlock);
          (void)rewriter.create<LLVM::StoreOp>(loc, value, atomicAddr,
                                               /*alignment=*/4);

          rewriter.create<LLVM::BrOp>(loc, endBlock);
          rewriter.setInsertionPointToStart(endBlock);
          // Return the value at the atomic address regardless of the compare
          // result.
          return atomPtrVal;
        };

    Operation *replacementOp = nullptr;
    if (calleeName == "__triton_hip_load_acquire_workgroup") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(i32_ty, operands[0], 8, LLVM::AtomicOrdering::acquire,
                          "workgroup-one-as");
    } else if (calleeName == "__triton_hip_load_relaxed_workgroup") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(i32_ty, operands[0], 8,
                          LLVM::AtomicOrdering::monotonic, "workgroup-one-as");
    }

    else if (calleeName == "__triton_hip_load_acquire_agent") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(i32_ty, operands[0], 8, LLVM::AtomicOrdering::acquire,
                          "agent-one-as");
    } else if (calleeName == "__triton_hip_load_relaxed_agent") {
      assert(operands.size() == 1);
      replacementOp =
          buildAtomicLoad(i32_ty, operands[0], 8,
                          LLVM::AtomicOrdering::monotonic, "agent-one-as");
    } else if (calleeName == "__triton_hip_load_acquire_system") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(i32_ty, operands[0], 4,
                                      LLVM::AtomicOrdering::acquire);
    } else if (calleeName == "__triton_hip_load_relaxed_system") {
      assert(operands.size() == 1);
      replacementOp = buildAtomicLoad(i32_ty, operands[0], 4,
                                      LLVM::AtomicOrdering::monotonic);
    }

    else if (calleeName == "__triton_hip_store_release_workgroup") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i32_ty, IntegerAttr::get(i32_ty, 1));
      (void)buildAtomicStore(one, operands[0], 8, LLVM::AtomicOrdering::release,
                             "workgroup-one-as");
      replacementOp = one.getDefiningOp();
    } else if (calleeName == "__triton_hip_store_relaxed_workgroup") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i32_ty, IntegerAttr::get(i32_ty, 1));
      (void)buildAtomicStore(one, operands[0], 8,
                             LLVM::AtomicOrdering::monotonic,
                             "workgroup-one-as");
      replacementOp = one.getDefiningOp();
    }

    else if (calleeName == "__triton_hip_store_release_agent") {
      assert(operands.size() == 2);
      (void)buildAtomicStore(operands[1], operands[0], 4,
                             LLVM::AtomicOrdering::release, "agent-one-as");
      replacementOp = operands[1].getDefiningOp();
    } else if (calleeName == "__triton_hip_store_relaxed_agent") {
      assert(operands.size() == 1);
      Value one = rewriter.create<LLVM::ConstantOp>(
          loc, i32_ty, IntegerAttr::get(i32_ty, 1));
      (void)buildAtomicStore(one, operands[0], 8,
                             LLVM::AtomicOrdering::monotonic, "agent-one-as");
      replacementOp = one.getDefiningOp();
    }

    else if (calleeName == "__triton_hip_store_release_system") {
      assert(operands.size() == 2);
      (void)buildAtomicStore(operands[1], operands[0], 4,
                             LLVM::AtomicOrdering::release);
      // FIXME: should store-like ops have returns ?
      replacementOp = operands[1].getDefiningOp();
    } else if (calleeName == "__triton_hip_store_relaxed_system") {
      assert(operands.size() == 2);
      (void)buildAtomicStore(operands[1], operands[0], 4,
                             LLVM::AtomicOrdering::monotonic);
      replacementOp = operands[1].getDefiningOp();
    }

    // define internal noundef i64 @syncthreads()() #1 !dbg !51 {
    // entry:
    //   fence syncscope("workgroup") release, !dbg !52
    //   tail call void @llvm.amdgcn.s.barrier(), !dbg !60
    //   fence syncscope("workgroup") acquire, !dbg !61
    //   ret i64 0, !dbg !62
    // }
    else if (calleeName == "__triton_hip_syncthreads") {
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

    else if (calleeName == "__triton_hip_red_add_release_agent") {
      assert(operands.size() == 2);
      replacementOp =
          buildAtomicFetchAdd(operands[0], operands[1],
                              LLVM::AtomicOrdering::release, "agent-one-as");
    } else if (calleeName == "__triton_hip_atom_add_acquire_agent") {
      assert(operands.size() == 2);
      replacementOp =
          buildAtomicFetchAdd(operands[0], operands[1],
                              LLVM::AtomicOrdering::acquire, "agent-one-as");
    } else if (calleeName == "__triton_hip_atom_add_relaxed_agent") {
      assert(operands.size() == 2);
      replacementOp =
          buildAtomicFetchAdd(operands[0], operands[1],
                              LLVM::AtomicOrdering::monotonic, "agent-one-as");
    } else if (calleeName == "__triton_hip_atom_add_acqrel_agent") {
      assert(operands.size() == 2);
      replacementOp =
          buildAtomicFetchAdd(operands[0], operands[1],
                              LLVM::AtomicOrdering::acq_rel, "agent-one-as");
    }

    else if (calleeName == "__triton_hip_red_add_release_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::release);
    } else if (calleeName == "__triton_hip_atom_add_acquire_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::acquire);
    } else if (calleeName == "__triton_hip_atom_add_relaxed_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::monotonic);
    } else if (calleeName == "__triton_hip_atom_add_acqrel_system") {
      assert(operands.size() == 2);
      replacementOp = buildAtomicFetchAdd(operands[0], operands[1],
                                          LLVM::AtomicOrdering::acq_rel);
    }

    else if (calleeName == "__triton_hip_atom_cas_acquire_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acquire,
          LLVM::AtomicOrdering::monotonic, "agent-one-as");
    } else if (calleeName == "__triton_hip_atom_cas_release_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::release,
          LLVM::AtomicOrdering::monotonic, "agent-one-as");
    } else if (calleeName == "__triton_hip_atom_cas_relaxed_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2],
          LLVM::AtomicOrdering::monotonic, LLVM::AtomicOrdering::monotonic,
          "agent-one-as");
    } else if (calleeName == "__triton_hip_atom_cas_acqrel_relaxed_agent") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acq_rel,
          LLVM::AtomicOrdering::monotonic, "agent-one-as");
    }

    else if (calleeName == "__triton_hip_atom_cas_acquire_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acquire,
          LLVM::AtomicOrdering::monotonic);
    } else if (calleeName == "__triton_hip_atom_cas_release_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::release,
          LLVM::AtomicOrdering::monotonic);
    } else if (calleeName == "__triton_hip_atom_cas_relaxed_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2],
          LLVM::AtomicOrdering::monotonic, LLVM::AtomicOrdering::monotonic);
    } else if (calleeName == "__triton_hip_atom_cas_acqrel_relaxed_system") {
      assert(operands.size() == 3);
      replacementOp = buildAtomicCompareExchangeStrong(
          operands[0], operands[1], operands[2], LLVM::AtomicOrdering::acq_rel,
          LLVM::AtomicOrdering::monotonic);
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

struct ConvertLibDeviceToLLVM
    : public triton::impl::ConvertLibDeviceToLLVMBase<ConvertLibDeviceToLLVM> {
  explicit ConvertLibDeviceToLLVM(bool ftz) { this->ftz = ftz; }

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
createConvertLibDeviceToLLVMPass(bool ftz) {
  return std::make_unique<ConvertLibDeviceToLLVM>(ftz);
}

} // namespace mlir::triton
