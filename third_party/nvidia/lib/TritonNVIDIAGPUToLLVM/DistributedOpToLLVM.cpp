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
#include "PatternTritonGPUOpToLLVM.h"
#include "TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "third_party/distributed/dialect/include/Dialect/Distributed/IR/Dialect.h"

#include "Utility.h"
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {

bool useNVSHMEMLibrary(StringRef libname) {
  return libname == "libnvshmem_device";
}

Operation *CreateNVSHMEMOp(RewriterBase &rewriter, Operation *curOp,
                           const StringRef &symbol, StringRef libname,
                           StringRef libpath, ValueRange inputOperands,
                           Type retType) {
  auto loc = curOp->getLoc();
  SmallVector<Value> llvmOpearands;

  // generic(addrspace=0) address space is required by func in nvshmem bitcode.
  // if address space is inconsistent, always-inline will not work.
  for (auto val : inputOperands) {
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
      rewriter, curOp, symbol, funcType, libname, libpath);
  auto op = LLVM::createLLVMCallOp(rewriter, loc, funcOp, llvmOpearands);
  if (retType == llvmRetType)
    return op;

  auto castRet =
      rewriter.create<LLVM::AddrSpaceCastOp>(loc, retType, op->getResult(0));
  return castRet;
}

template <typename DistOp>
class GenericOpToNVSHMEMDevice : public ConvertOpToLLVMPattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  GenericOpToNVSHMEMDevice(const LLVMTypeConverter &converter,
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
    auto nvshmemOp = CreateNVSHMEMOp(rewriter, op, calleeName, libname, libpath,
                                     newOperands, retType);
    auto newResult = nvshmemOp->getResult(0);
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
void registerGenericOpToNVSHMEMDevice(RewritePatternSet &patterns,
                                      LLVMTypeConverter &typeConverter,
                                      PatternBenefit benefit,
                                      StringRef calleeName, StringRef libname,
                                      StringRef libpath) {
  patterns.add<GenericOpToNVSHMEMDevice<Args>...>(typeConverter, benefit,
                                                  calleeName, libname, libpath);
}

struct WaitOpConversion
    : public ConvertOpToLLVMPattern<triton::distributed::WaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::distributed::WaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto type = op->getOperand(0).getType();
    assert(isa<triton::PointerType>(type) && "must be a pointer type");
    auto ptree_type = dyn_cast<triton::PointerType>(type).getPointeeType();
    auto intType = dyn_cast<mlir::IntegerType>(ptree_type);
    assert(intType && "barrier ptr must be integer type.");
    const size_t barrier_width = intType.getWidth();
    std::string scope = "";
    if (op.getScope() == triton::MemSyncScope::CTA) {
      scope = "cta";
    } else if (op.getScope() == triton::MemSyncScope::GPU) {
      scope = "gpu";
    } else if (op.getScope() == triton::MemSyncScope::SYSTEM) {
      scope = "sys";
    }
    std::string semantic = "";
    if (op.getSemantic() == triton::MemSemantic::ACQUIRE) {
      semantic = "acquire";
    } else if (op.getSemantic() == triton::MemSemantic::RELAXED) {
      semantic = "relaxed";
    } else if (op.getSemantic() == triton::MemSemantic::RELEASE) {
      semantic = "release";
    } else if (op.getSemantic() == triton::MemSemantic::ACQUIRE_RELEASE) {
      semantic = "acq_rel";
    }
    const std::string ld_ptx = "ld.global."s + semantic + "."s + scope + ".b"s +
                               std::to_string(barrier_width);
    const std::string bit_w = std::to_string(barrier_width);
    const std::string byte_w = std::to_string(barrier_width / 8);
    // TODO(zhengsize): how about more barriers?
    // we only consider warp sync now
    // so numBarriers should be <= WARP_SIZE
    // otherwise, the behavior is undefined
    const std::string ptx =
        "{                                                              \n\t"s +
        ".reg .pred %p<2>;                                              \n\t"s +
        ".reg .b32 %th<2>;                                              \n\t"s +
        ".reg .u64 %addr<2>;                                            \n\t"s +
        ".reg .b"s + bit_w + " %tmp<1>;                                 \n\t"s +
        "mov.u32 %th1, $1;                                              \n\t"s +
        "mov.u32 %th0, %tid.x;                                          \n\t"s +
        "rem.u32 %th0, %th0, 32;                                        \n\t"s +
        "mul.wide.s32 %addr1, %th0, "s + byte_w + ";                    \n\t"s +
        "add.u64 %addr0, $0, %addr1;                                    \n\t"s +
        "setp.lt.u32 %p0, %th0, %th1;                                   \n\t"s +
        "@!%p0 bra.uni skipLoop;                                        \n\t"s +
        "waitLoop:                                                      \n\t"s +
        "  "s + ld_ptx + " %tmp0, [%addr0];                             \n\t"s +
        "  setp.eq.b"s + bit_w + " %p0, %tmp0, $2;                      \n\t"s +
        "  @!%p0 bra.uni waitLoop;                                      \n\t"s +
        "skipLoop:                                                      \n\t"s +
        "bar.warp.sync 0xffffffff;                                      \n\t"s +
        "}                                                              \n\t"s;

    std::string regTy = barrier_width == 64 ? "l" : "r";
    auto &waitOp = *ptxBuilder.create<>(ptx);
    waitOp({ptxBuilder.newOperand(adaptor.getBarrierPtr(), "l"),
            ptxBuilder.newOperand(adaptor.getNumBarriers(), "r"),
            ptxBuilder.newOperand(adaptor.getWaitValue(), regTy)},
           /*onlyAttachMLIRArgs=*/true);
    auto voidTy = void_ty(op->getContext());
    ptxBuilder.launch(rewriter, loc, voidTy);
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
    auto loc = op->getLoc();
    bool isIntraNode =
        op.getCommScope() != ::mlir::triton::distributed::CommScope::INTER_NODE;
    auto signalType = op.getSigAddr().getType();
    ::mlir::triton::PTXBuilder ptxBuilder;
    auto b = ::mlir::triton::TritonLLVMOpBuilder(loc, rewriter);
    Value threadId = rewriter.create<NVVM::ThreadIdXOp>(loc, i32_ty);
    Value pred = b.icmp_eq(threadId, b.i32_val(0));
    Block *prevBlock = op->getBlock();

    Block *ifBlock = rewriter.splitBlock(prevBlock, op->getIterator());
    rewriter.setInsertionPointToStart(ifBlock);

    Block *thenBlock = rewriter.splitBlock(ifBlock, op->getIterator());
    rewriter.setInsertionPointToEnd(ifBlock);
    rewriter.create<cf::BranchOp>(loc, thenBlock);
    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<cf::CondBranchOp>(loc, pred, ifBlock, thenBlock);
    rewriter.setInsertionPointToStart(ifBlock);

    rewriter.setInsertionPointToStart(ifBlock);
    if (isIntraNode) {
      // remote ptr
      bool isIntraRank =
          op.getCommScope() == ::mlir::triton::distributed::CommScope::GPU;
      Type retType = this->getTypeConverter()->convertType(signalType);
      Value remotePtr;
      if (isIntraRank) {
        remotePtr = adaptor.getSigAddr();
      } else {
        remotePtr =
            CreateNVSHMEMOp(rewriter, op, "nvshmem_ptr", libname, libpath,
                            {adaptor.getSigAddr(), adaptor.getRank()}, retType)
                ->getResult(0);
      }
      ::mlir::triton::PointerType sigalElemType =
          llvm::cast<::mlir::triton::PointerType>(signalType);
      const size_t signalWidth =
          sigalElemType.getPointeeType().getIntOrFloatBitWidth();
      std::string semantic = "relaxed";
      std::string stScope = isIntraRank ? "gpu" : "sys";
      std::string membarScope = isIntraRank ? "gl" : "sys";
      const std::string memBarPtx = "membar." + membarScope + ";\n\t";

      if (adaptor.getSigOp() == ::mlir::triton::distributed::SignalOp::SET) {
        std::string opType = "st";
        const std::string stSignalPtx =
            opType + "." + semantic + "." + stScope + ".global.b" +
            std::to_string(signalWidth) + " [$0], $1" + ";\n\t";
        const std::string ptx = memBarPtx + stSignalPtx;
        auto &notifyPtxOp = *ptxBuilder.create<>(ptx);
        notifyPtxOp({ptxBuilder.newOperand(remotePtr, "l"),
                     ptxBuilder.newOperand(adaptor.getSignalVal(), "l")}, // u64
                    /*onlyAttachMLIRArgs=*/true);
        auto voidTy = void_ty(op->getContext());
        ptxBuilder.launch(rewriter, loc, voidTy);
      } else {
        std::string opType = "add";
        // Operation .add requires .u32 or .s32 or .u64 or .f64 or f16 or f16x2
        // or .f32 or .bf16 or .bf16x2 type for instruction 'atom'
        const std::string stSignalPtx =
            "atom." + semantic + "." + stScope + ".global." + opType + ".u" +
            std::to_string(signalWidth) + " $0, [$1], $2" + ";\n\t";
        const std::string ptx = memBarPtx + stSignalPtx;
        auto &notifyPtxOp = *ptxBuilder.create<>(ptx);
        notifyPtxOp({ptxBuilder.newOperand("=l"),
                     ptxBuilder.newOperand(remotePtr, "l"),
                     ptxBuilder.newOperand(adaptor.getSignalVal(), "l")}, // u64
                    /*onlyAttachMLIRArgs=*/true);
        ptxBuilder.launch(rewriter, loc, retType);
      }
    } else {
      LLVM::LLVMVoidType voidTy = void_ty(op->getContext());
      // NVSHMEM_SIGNAL_SET = 9
      // NVSHMEM_SIGNAL_ADD = 10
      int32_t v = -1;
      if (adaptor.getSigOp() == ::mlir::triton::distributed::SignalOp::SET) {
        v = 9;
      } else if (adaptor.getSigOp() ==
                 ::mlir::triton::distributed::SignalOp::ADD) {
        v = 10;
      }
      Value sigOp = mlir::LLVM::createConstantI32(loc, rewriter, v);
      auto nvshmemxSignalOp =
          CreateNVSHMEMOp(rewriter, op, "nvshmemx_signal_op", libname, libpath,
                          {adaptor.getSigAddr(), adaptor.getSignalVal(), sigOp,
                           adaptor.getRank()},
                          voidTy);
    }
    rewriter.eraseOp(op);
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
    auto loc = op->getLoc();
    ::mlir::triton::PTXBuilder ptxBuilder;
    // inline ptx to aviod function call.
    // we can remove it if `nvshmem_ptr` can be inlined in llvm
    if (inlinePtx) {
      const std::string nvshmemPtxPtx =
          "{                                                        \n\t"s +
          " .reg.b64 %nvshmem_ptr_wrapper_in_0;                     \n\t"s +
          " .reg.b32 %nvshmem_ptr_wrapper_in_1;                     \n\t"s +
          " .reg.b64 %nvshmem_ptr_wrapper_out_0;                    \n\t"s +
          " mov.b64 %nvshmem_ptr_wrapper_in_0, $1;                  \n\t"s +
          " mov.b32 %nvshmem_ptr_wrapper_in_1, $2;                  \n\t"s +
          " {                                                       \n\t"s +
          "   .reg .pred  %p<5>;                                    \n\t"s +
          "   .reg .b32 %r<2>;                                      \n\t"s +
          "   .reg .b64 %rd<14>;                                    \n\t"s +
          "   mov.b64 %rd5, %nvshmem_ptr_wrapper_in_0;              \n\t"s +
          "   mov.b32 %r1, %nvshmem_ptr_wrapper_in_1;               \n\t"s +
          "   mov.u64 %rd13, 0;                                     \n\t"s +
          "   ld.const.u64  %rd6, [nvshmemi_device_state_d+40];     \n\t"s +
          "   sub.s64 %rd1, %rd5, %rd6;                             \n\t"s +
          "   setp.gt.u64 %p1, %rd6, %rd5;                          \n\t"s +
          "   ld.const.u64  %rd7, [nvshmemi_device_state_d+48];     \n\t"s +
          "   setp.ge.u64 %p2, %rd1, %rd7;                          \n\t"s +
          "   or.pred %p3, %p1, %p2;                                \n\t"s +
          "   @%p3 bra  L__BB6_2;                                   \n\t"s +
          "   ld.const.u64 %rd10, [nvshmemi_device_state_d+56];     \n\t"s +
          "   mul.wide.s32 %rd11, %r1, 8;                           \n\t"s +
          "   add.s64 %rd9, %rd10, %rd11;                           \n\t"s +
          "   // begin inline asm                                   \n\t"s +
          "   ld.global.nc.u64 %rd8, [%rd9];                        \n\t"s +
          "   // end inline asm                                     \n\t"s +
          "   setp.eq.s64 %p4, %rd8, 0;                             \n\t"s +
          "   add.s64 %rd12, %rd8, %rd1;                            \n\t"s +
          "   selp.b64 %rd13, %rd8, %rd12, %p4;                     \n\t"s +
          " L__BB6_2:                                               \n\t"s +
          "   mov.b64 %nvshmem_ptr_wrapper_out_0, %rd13;            \n\t"s +
          " }                                                       \n\t"s +
          " mov.b64 $0, %nvshmem_ptr_wrapper_out_0;                 \n\t"s +
          "}                                                        \n\t";

      auto &nvshmemPtr = *ptxBuilder.create<>(nvshmemPtxPtx);
      nvshmemPtr({ptxBuilder.newOperand("=l"),
                  ptxBuilder.newOperand(adaptor.getSymmAddr(), "l"),
                  ptxBuilder.newOperand(adaptor.getRank(), "r")},
                 /*onlyAttachMLIRArgs=*/true);

      // addrspace = 1 means global memory
      auto ptxResult = ptxBuilder.launch(
          rewriter, loc, ptr_ty(rewriter.getContext(), /*addrspace=*/1));
      rewriter.replaceOp(op, ptxResult);
    } else {
      Type retType =
          this->getTypeConverter()->convertType(op->getResult(0).getType());
      auto nvshmemOp = CreateNVSHMEMOp(rewriter, op, "nvshmem_ptr", libname,
                                       libpath, adaptor.getOperands(), retType);
      auto newResult = nvshmemOp->getResult(0);
      rewriter.replaceOp(op, newResult);
    }
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
    if (useNVSHMEMLibrary(op.getLibname())) {
      externCallOp = CreateNVSHMEMOp(rewriter, op, funcName, libname, libpath,
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

void mlir::triton::NVIDIA::populateDistributedOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo,
    std::string NVSHMEMLibname, std::string NVSHMEMLibpath) {
  patterns.add<WaitOpConversion, ConsumeTokenOpConversion>(typeConverter,
                                                           benefit);

  // convert to nvshmem device func call
  registerGenericOpToNVSHMEMDevice<triton::distributed::GetRankOp>(
      patterns, typeConverter, benefit, "nvshmem_my_pe", NVSHMEMLibname,
      NVSHMEMLibpath);
  registerGenericOpToNVSHMEMDevice<triton::distributed::GetNumRanksOp>(
      patterns, typeConverter, benefit, "nvshmem_n_pes", NVSHMEMLibname,
      NVSHMEMLibpath);
  registerGenericOpToNVSHMEMDevice<triton::distributed::SymmAtOp>(
      patterns, typeConverter, benefit, "nvshmem_ptr", NVSHMEMLibname,
      NVSHMEMLibpath);
  patterns.add<NotifyOpConversion>(typeConverter, benefit, NVSHMEMLibname,
                                   NVSHMEMLibpath);
  patterns.add<ExternCallConversion>(typeConverter, benefit);
}
