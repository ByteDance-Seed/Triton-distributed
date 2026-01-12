#include "TritonDistributed/Conversion/TritonDistributedToLLVM/TritonDistributedToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "third_party/metax/lib/TritonMETAXGPUToLLVM/Utility.h"

#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {

bool useMXSHMEMLibrary(StringRef libname) {
  return libname == "libmxshmem_device";
}

Operation *CreateMXSHMEMOp(ConversionPatternRewriter &rewriter,
                           Operation *curOp, const StringRef &symbol,
                           StringRef libname, StringRef libpath,
                           ValueRange inputOperands, Type retType) {
  auto loc = curOp->getLoc();
  SmallVector<Value> llvmOpearands;

  // generic(addrspace=0) address space is required by func in mxshmem bitcode.
  // if address space is inconsistent, always-inline will not work.
  for (auto val : inputOperands) {
    if (auto ptrTy = llvm::dyn_cast<LLVM::LLVMPointerType>(val.getType())) {
      assert((ptrTy.getAddressSpace() == 0 || ptrTy.getAddressSpace() == 1) &&
             "wrong address space.");
      Value ptrAfterCast = val;
      ptrAfterCast = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0), val);
      llvmOpearands.push_back(ptrAfterCast);
    } else {
      llvmOpearands.push_back(val);
    }
  }

  Type llvmRetType = retType;
  if (auto retPtrType = llvm::dyn_cast<LLVM::LLVMPointerType>(retType)) {
    assert((retPtrType.getAddressSpace() == 0 ||
            retPtrType.getAddressSpace() == 1) &&
           "wrong address space.");
    llvmRetType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  }

  Type funcType =
      mlir::triton::gpu::getFunctionType(llvmRetType, llvmOpearands);

  LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
      rewriter, curOp, symbol, funcType, libname, libpath);
  auto op = LLVM::createLLVMCallOp(rewriter, loc, funcOp, llvmOpearands);
  if (retType == llvmRetType)
    return op;

  auto castRet =
      rewriter.create<LLVM::AddrSpaceCastOp>(loc, retType, op->getResult(0));
  return castRet;
}

template <typename DistOp>
class GenericOpToMXSHMEMDevice : public ConvertOpToLLVMPattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  GenericOpToMXSHMEMDevice(const LLVMTypeConverter &converter,
                           const PatternBenefit &benefit, StringRef calleeName,
                           StringRef libname = "", StringRef libpath = "")
      : ConvertOpToLLVMPattern<DistOp>(converter, benefit),
        calleeName(calleeName), libname(libname), libpath(libpath) {}

  LogicalResult
  matchAndRewrite(DistOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op->getNumResults() > 1)
      return failure();
    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
    auto newOperands = adaptor.getOperands();
    Type retType =
        op->getNumResults() == 0
            ? voidTy
            : this->getTypeConverter()->convertType(op->getResult(0).getType());
    auto mxshmemOp = CreateMXSHMEMOp(rewriter, op, calleeName, libname, libpath,
                                     newOperands, retType);
    auto newResult = mxshmemOp->getResult(0);
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, newResult);
    }

    return success();
  }

private:
  StringRef calleeName;
  StringRef libname;
  StringRef libpath;
};

template <typename... Args>
void registerGenericOpToMXSHMEMDevice(RewritePatternSet &patterns,
                                      LLVMTypeConverter &typeConverter,
                                      PatternBenefit benefit,
                                      StringRef calleeName, StringRef libname,
                                      StringRef libpath) {
  patterns.add<GenericOpToMXSHMEMDevice<Args>...>(typeConverter, benefit,
                                                  calleeName, libname, libpath);
}

struct WaitOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::WaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto type = op->getOperand(0).getType();
    assert(isa<triton::PointerType>(type) && "must be a pointer type");
    auto ptree_type = dyn_cast<triton::PointerType>(type).getPointeeType();
    auto intType = dyn_cast<mlir::IntegerType>(ptree_type);
    assert(intType && "barrier ptr must be integer type.");
    const size_t barrier_width = intType.getWidth() / 8;
    std::string scope = "";
    if (op.getScope() == triton::MemSyncScope::CTA) {
      scope = "block";
    } else if (op.getScope() == triton::MemSyncScope::GPU) {
      scope = "device";
    } else if (op.getScope() == triton::MemSyncScope::SYSTEM) {
      scope = "system";
    }
    LLVM::AtomicOrdering semantic;
    if (op.getSemantic() == triton::MemSemantic::ACQUIRE) {
      semantic = LLVM::AtomicOrdering::acquire;
    } else if (op.getSemantic() == triton::MemSemantic::RELAXED) {
      semantic = LLVM::AtomicOrdering::monotonic;
    } else if (op.getSemantic() == triton::MemSemantic::RELEASE) {
      semantic = LLVM::AtomicOrdering::release;
    } else if (op.getSemantic() == triton::MemSemantic::ACQUIRE_RELEASE) {
      semantic = LLVM::AtomicOrdering::acq_rel;
    }
    auto barrier_ptr = adaptor.getBarrierPtr();
    auto num_barriers = adaptor.getNumBarriers();
    auto wait_val = adaptor.getWaitValue();
    auto tid = tid_val();
    Value warpSize = i32_val(64);
    Value laneId = urem(tid, warpSize);

    auto pred = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               laneId, num_barriers);
    // create if
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{}, pred,
                                           /*hasElse=*/false);
    // if then block:
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    // calculate offset of each barrier.
    auto addr =
        gep(ptr_ty(rewriter.getContext(), 1), intType, barrier_ptr, laneId);
    // load init val for while op.
    Value init = load(intType, addr, barrier_width, false, false, false,
                      semantic, scope);

    // create while
    auto whileOp =
        rewriter.create<scf::WhileOp>(loc, TypeRange{intType}, // return type
                                      ValueRange{init}         // init args
        );

    // create condition
    Block *beforeBlock;
    if (whileOp.getBefore().empty()) {
      whileOp.getBefore().push_back(new Block());
    }
    beforeBlock = whileOp.getBeforeBody();
    if (beforeBlock->getNumArguments() == 0) {
      beforeBlock->addArgument(intType, loc);
    }
    rewriter.setInsertionPointToStart(beforeBlock);
    Value arg = beforeBlock->getArgument(0);
    auto whileCond = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, arg, wait_val);
    rewriter.create<scf::ConditionOp>(loc, whileCond, ValueRange{arg});
    // create loop after region
    Block *afterBlock;
    if (whileOp.getAfter().empty()) {
      whileOp.getAfter().push_back(new Block());
    }
    afterBlock = whileOp.getAfterBody();
    if (afterBlock->getNumArguments() == 0) {
      afterBlock->addArgument(intType, loc);
    }
    rewriter.setInsertionPointToStart(afterBlock);
    // update argument
    Value ret = load(intType, addr, barrier_width, false, false, false,
                     semantic, scope);
    rewriter.create<scf::YieldOp>(loc, ValueRange{ret});

    // barrier
    rewriter.setInsertionPointAfter(ifOp);
    StringRef funcName("llvm.mxc.barrier.inst");
    Value voidVal = undef(void_ty(op.getContext()));
    ValueRange voidVals = {};
    mlir::LLVM::createBuiltinFunc<triton::distributed::WaitOp>(
        rewriter, loc, op, funcName, getVoidType(), voidVals);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConsumeTokenOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::ConsumeTokenOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::ConsumeTokenOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getInput());
    return success();
  }
};

class NotifyOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::NotifyOp> {
public:
  NotifyOpConversion(const LLVMTypeConverter &converter,
                     const PatternBenefit &benefit, StringRef libname = "",
                     StringRef libpath = "")
      : ConvertOpToLLVMPattern<triton::distributed::NotifyOp>(converter,
                                                              benefit),
        libname(libname), libpath(libpath) {}

  LogicalResult
  matchAndRewrite(triton::distributed::NotifyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }

private:
  StringRef libname;
  StringRef libpath;
};

class SymmAtOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::SymmAtOp> {
public:
  SymmAtOpConversion(const LLVMTypeConverter &converter,
                     const PatternBenefit &benefit, bool inlinePtx = false,
                     StringRef libname = "", StringRef libpath = "")
      : ConvertOpToLLVMPattern<triton::distributed::SymmAtOp>(converter,
                                                              benefit),
        inlinePtx(inlinePtx), libname(libname), libpath(libpath) {}

  LogicalResult
  matchAndRewrite(triton::distributed::SymmAtOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return success();
  }

private:
  bool inlinePtx;
  StringRef libname;
  StringRef libpath;
};

class ExternCallConversion
    : public ConvertOpToLLVMPattern<triton::distributed::ExternCallOp> {
public:
  ExternCallConversion(const LLVMTypeConverter &converter,
                       const PatternBenefit &benefit)
      : ConvertOpToLLVMPattern<triton::distributed::ExternCallOp>(converter,
                                                                  benefit) {}

  LogicalResult
  matchAndRewrite(triton::distributed::ExternCallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();

    if (op->getNumResults() > 1) {
      llvm::errs() << "ExternCallConversion does not support multi outs.";
      return failure();
    }

    LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
    auto newOperands = adaptor.getOperands();
    Type retType =
        op->getNumResults() == 0
            ? voidTy
            : this->getTypeConverter()->convertType(op->getResult(0).getType());
    StringRef funcName = op.getSymbol();
    StringRef libname = op.getLibname();
    StringRef libpath = op.getLibpath();

    Operation *externCallOp;
    if (useMXSHMEMLibrary(op.getLibname())) {
      externCallOp = CreateMXSHMEMOp(rewriter, op, funcName, libname, libpath,
                                     newOperands, retType);
    } else {
      Type funcType = mlir::triton::gpu::getFunctionType(retType, newOperands);
      LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
          rewriter, op, funcName, funcType, libname, libpath);
      externCallOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, newOperands);
    }

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, externCallOp->getResult(0));
    }

    return success();
  }
};

} // namespace

void mlir::triton::METAX::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo,
    std::string MXSHMEMLibname, std::string MXSHMEMLibpath) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter,
                                                           benefit);
  registerGenericOpToMXSHMEMDevice<triton::distributed::GetRankOp>(
      patterns, typeConverter, benefit, "mxshmem_my_pe", MXSHMEMLibname,
      MXSHMEMLibpath);
  registerGenericOpToMXSHMEMDevice<triton::distributed::GetNumRanksOp>(
      patterns, typeConverter, benefit, "mxshmem_n_pes", MXSHMEMLibname,
      MXSHMEMLibpath);
  registerGenericOpToMXSHMEMDevice<triton::distributed::SymmAtOp>(
      patterns, typeConverter, benefit, "mxshmem_ptr", MXSHMEMLibname,
      MXSHMEMLibpath);
  patterns.add<NotifyOpConversion>(typeConverter, benefit, MXSHMEMLibname,
                                   MXSHMEMLibpath);
  patterns.add<ExternCallConversion>(typeConverter, benefit);
}
