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
#include "TritonDistributed/Conversion/TritonDistributedToLLVM/TritonDistributedToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"

#include "third_party/amd/lib/TritonAMDGPUToLLVM/Utility.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {

bool useROCSHMEMLibrary(StringRef libname) {
  return libname == "librocshmem_device";
}

template <typename DistOp>
class GenericOpToROCSHMEMDevice : public ConvertOpToLLVMPattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  GenericOpToROCSHMEMDevice(const LLVMTypeConverter &converter,
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
    Type funcType = mlir::triton::gpu::getFunctionType(retType, newOperands);
    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, calleeName, funcType, libname, libpath);
    auto newResult =
        LLVM::createLLVMCallOp(rewriter, loc, funcOp, newOperands).getResult();
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
    std::string funcName = op.getSymbol().str();

    StringRef libname = op.getLibname();
    StringRef libpath = op.getLibpath();

    SmallVector<Value> llvmOpearands;

    for (auto val : newOperands) {
      if (auto ptrTy = llvm::dyn_cast<LLVM::LLVMPointerType>(val.getType())) {
        assert((ptrTy.getAddressSpace() == 0 || ptrTy.getAddressSpace() == 1) &&
               "wrong address space.");
        Value ptrAfterCast = val;
        ptrAfterCast = rewriter.create<LLVM::AddrSpaceCastOp>(
            loc, LLVM::LLVMPointerType::get(rewriter.getContext()), val);
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
      llvmRetType = LLVM::LLVMPointerType::get(rewriter.getContext());
    }

    Type funcType =
        mlir::triton::gpu::getFunctionType(llvmRetType, llvmOpearands);

    LLVM::LLVMFuncOp funcOp = mlir::triton::gpu::appendOrGetExternFuncOp(
        rewriter, op, funcName, funcType, libname, libpath);
    auto callOp = LLVM::createLLVMCallOp(rewriter, loc, funcOp, llvmOpearands);

    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      if (retType == llvmRetType) {
        auto newResult = callOp.getResult();
        rewriter.replaceOp(op, newResult);
      } else {

        auto castOp =
            rewriter
                .create<LLVM::AddrSpaceCastOp>(loc, retType, callOp.getResult())
                ->getResult(0);
        rewriter.replaceOp(op, castOp);
      }
    }

    return success();
  }
};

template <typename... Args>
void registerGenericOpToROCSHMEMDevice(RewritePatternSet &patterns,
                                       LLVMTypeConverter &typeConverter,
                                       PatternBenefit benefit,
                                       StringRef calleeName, StringRef libname,
                                       StringRef libpath) {
  patterns.add<GenericOpToROCSHMEMDevice<Args>...>(
      typeConverter, benefit, calleeName, libname, libpath);
}

struct WaitOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::WaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op->getLoc();
    auto type = op.getBarrierPtr().getType();
    assert(isa<triton::PointerType>(type) && "must be a pointer type");
    auto ptreeType = dyn_cast<triton::PointerType>(type).getPointeeType();
    auto intType = dyn_cast<mlir::IntegerType>(ptreeType);
    if (!intType) {
      return op->emitError("barrier ptr must be integer type.");
    }
    const size_t barrierWidth = intType.getWidth();
    unsigned int numBytes = barrierWidth / 8;

    StringRef syncGroup;
    if (adaptor.getScope() == triton::MemSyncScope::CTA) {
      syncGroup = "workgroup-one-as";
    } else if (adaptor.getScope() == triton::MemSyncScope::GPU) {
      syncGroup = "agent-one-as";
    } else if (adaptor.getScope() == triton::MemSyncScope::SYSTEM) {
      syncGroup = StringRef();
    }

    LLVM::AtomicOrdering ordering = LLVM::AtomicOrdering::acquire;
    if (adaptor.getSemantic() == triton::MemSemantic::ACQUIRE) {
      ordering = LLVM::AtomicOrdering::acquire;
    } else if (adaptor.getSemantic() == triton::MemSemantic::RELAXED) {
      ordering = LLVM::AtomicOrdering::monotonic;
    } else if (adaptor.getSemantic() == triton::MemSemantic::RELEASE) {
      ordering = LLVM::AtomicOrdering::release;
    } else if (adaptor.getSemantic() == triton::MemSemantic::ACQUIRE_RELEASE) {
      ordering = LLVM::AtomicOrdering::acq_rel;
    }

    // convert waitOp to the following ops:
    /*
    ^init_block:
      tid = rocdl.threadIdx.x
      cf.br entry_block(tid)

    ^entry_block(i: i32):
      pred = i < num_barrier
      cd.cond_br (pred, loop_block, yield)

    ^loop_block:
      ptr = ptr + i
      val = load(ptr)
      pred = val != x
      cf.cond_br (pred, loop_block, jump_block)

    ^jump_block:
      val = i + block_size
      cf.br entry_block(val)

    ^yield_block:
      gpu.barrier
    */
    Block *initBlock = op->getBlock();
    Block *whileEntryBlock =
        rewriter.splitBlock(initBlock, rewriter.getInsertionPoint());
    Block *loopBlock = rewriter.splitBlock(whileEntryBlock, op->getIterator());
    Block *jumpBlock = rewriter.splitBlock(loopBlock, op->getIterator());
    Block *yieldBlock = rewriter.splitBlock(jumpBlock, op->getIterator());

    auto b = ::mlir::triton::TritonLLVMOpBuilder(loc, rewriter);

    // init block
    rewriter.setInsertionPointToEnd(initBlock);
    Value workIDX = rewriter.create<ROCDL::ThreadIdXOp>(loc, i32_ty);
    rewriter.create<cf::BranchOp>(loc, whileEntryBlock, workIDX);

    // while entry block
    whileEntryBlock->addArgument(workIDX.getType(), loc);
    rewriter.setInsertionPointToEnd(whileEntryBlock);
    Value index = whileEntryBlock->getArgument(0);
    Value whileCond = b.icmp_slt(index, adaptor.getNumBarriers());
    rewriter.create<cf::CondBranchOp>(loc, whileCond, loopBlock, yieldBlock);

    // loop block
    rewriter.setInsertionPointToEnd(loopBlock);
    auto basePtr = adaptor.getBarrierPtr();
    auto elemLlvmTy = this->getTypeConverter()->convertType(intType);
    auto newPtr = b.gep(basePtr.getType(), elemLlvmTy, basePtr, index);
    LLVM::LoadOp loadOp = rewriter.create<LLVM::LoadOp>(
        loc, elemLlvmTy, newPtr, /*alignment=*/numBytes,
        /*isVolatile=*/false, /*isNonTemporal=*/false,
        /*isInvariant =*/false, /*isInvariantGroup=*/false, ordering,
        syncGroup);

    Value pred = rewriter.create<LLVM::ICmpOp>(loc, LLVM::ICmpPredicate::ne,
                                               loadOp.getResult(),
                                               adaptor.getWaitValue());
    rewriter.create<cf::CondBranchOp>(loc, pred, loopBlock, jumpBlock);

    // jump block
    rewriter.setInsertionPointToEnd(jumpBlock);
    // why not use the i32 BlockDimXOp:
    // there is a bug in the upstream createDimGetterFunctionCall, which will
    // cause type mismatch(create binary operator with two operands of differing
    // type) when LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    Value blockSize =
        rewriter.create<ROCDL::BlockDimXOp>(loc, rewriter.getIntegerType(64));
    blockSize = b.trunc(i32_ty, blockSize);

    Value barrier_idx = b.add(whileEntryBlock->getArgument(0), blockSize);
    rewriter.create<cf::BranchOp>(loc, whileEntryBlock, barrier_idx);

    // yieldBlock
    rewriter.setInsertionPointToStart(yieldBlock);
    auto preBarrier = rewriter.create<mlir::gpu::BarrierOp>(loc);

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

} // namespace

void mlir::triton::AMD::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo,
    std::string ROCSHMEMLibname, std::string ROCSHMEMLibpath) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter,
                                                           benefit);

  // convert to rocshmem device func call
  registerGenericOpToROCSHMEMDevice<triton::distributed::GetRankOp>(
      patterns, typeConverter, benefit, "rocshmem_my_pe_wrapper",
      ROCSHMEMLibname, ROCSHMEMLibpath);
  registerGenericOpToROCSHMEMDevice<triton::distributed::GetNumRanksOp>(
      patterns, typeConverter, benefit, "rocshmem_n_pes_wrapper",
      ROCSHMEMLibname, ROCSHMEMLibpath);
  registerGenericOpToROCSHMEMDevice<triton::distributed::SymmAtOp>(
      patterns, typeConverter, benefit, "rocshmem_ptr_wrapper", ROCSHMEMLibname,
      ROCSHMEMLibpath);
  patterns.add<ExternCallConversion>(typeConverter, benefit);
}
