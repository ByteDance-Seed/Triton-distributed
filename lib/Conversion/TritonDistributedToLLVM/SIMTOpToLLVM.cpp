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
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"

#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/Utility.h"
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {
#ifndef USE_MACA

// Compute element-wise linear strides for a trivial (vec=perPhase=maxPhase=1)
// swizzled shared-memory buffer. `SharedMemoryObject::getStrides` was removed
// upstream when shared layouts moved to the LinearLayout model, so we
// reconstruct the exact strides from the buffer's shape and shared encoding.
//
// Two encoding conventions reach this lowering and BOTH must be handled:
//   1. SIMT-region promotion buffers carry a FULL-rank order (a permutation of
//      all dims), e.g. `order=[0,1]` (dim 0 contiguous == column-major). The
//      physical stride is order-driven and must be honored -- treating these as
//      row-major transposes the tile and corrupts the result.
//   2. Frontend makeSwizzledMemDescType buffers give a rank>1 buffer a rank-1
//      shared encoding (leading dims are multibuffer/batch indices), so `order`
//      covers only the trailing `order.size()` dims, e.g. `order=[0]` on a
//      rank-2 shape.
//
// General rule: assign order-driven contiguous strides within the trailing
// encoded block (fastest dim first), then row-major strides to the leading
// batch dims on top of that block. This reproduces the removed getStrides for
// full-rank orders and stays in-bounds for short (multibuffer) orders.
SmallVector<Value> computeSharedStrides(triton::gpu::MemDescType sharedTy,
                                        Location loc, RewriterBase &rewriter) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto shape = sharedTy.getShape();
  auto order = triton::gpu::getOrder(sharedTy);
  unsigned rank = shape.size();
  unsigned encRank = order.size();
  assert(encRank <= rank && encRank >= 1 &&
         "shared encoding rank must be in [1, buffer rank]");
  unsigned batch = rank - encRank; // leading (un-encoded) multibuffer dims

  SmallVector<int64_t> strideInt(rank, 1);
  int64_t running = 1;
  // Trailing encoded block: `order` lists its dims fastest-first, indexed
  // relative to the block, so the absolute buffer dim is `batch + order[k]`.
  for (unsigned k = 0; k < encRank; ++k) {
    unsigned absDim = batch + order[k];
    assert(absDim < rank && "encoding order out of range");
    strideInt[absDim] = running;
    running *= shape[absDim];
  }
  // Leading batch dims stack row-major (outermost slowest) above the block.
  for (int d = static_cast<int>(batch) - 1; d >= 0; --d) {
    strideInt[d] = running;
    running *= shape[d];
  }

  SmallVector<Value> strides(rank);
  for (unsigned i = 0; i < rank; ++i)
    strides[i] = b.i32_val(strideInt[i]);
  return strides;
}

Value getSharedMemAddress(RewriterBase &rewriter,
                          const SharedMemoryObject &smemObj,
                          const SmallVector<Value> &indices,
                          triton::gpu::MemDescType sharedTy, Type elemLlvmTy,
                          Location loc) {
  auto sharedEnc =
      cast<triton::gpu::SharedEncodingTrait>(sharedTy.getEncoding());
  auto b = TritonLLVMOpBuilder(loc, rewriter);
  auto smemBase = smemObj.getBase();
  auto smemOffsets = smemObj.getOffsets();
  assert(smemOffsets.size() == indices.size());
  auto smemStrides = computeSharedStrides(sharedTy, loc, rewriter);
  for (size_t i = 0; i < smemOffsets.size(); ++i) {
    smemOffsets[i] = b.add(smemOffsets[i], indices[i]);
  }
  Value offset = dot(rewriter, loc, smemOffsets, smemStrides);

  auto base = smemObj.getBase();
  auto elemPtrTy = base.getType();
  Value addr = b.gep(elemPtrTy, elemLlvmTy, base, offset);
  return addr;
}

struct LoadSharedOpPattern
    : public ConvertOpToLLVMPattern<triton::simt::LoadSharedOp> {
  explicit LoadSharedOpPattern(LLVMTypeConverter &typeConverter,
                               const TargetInfoBase &targetInfo,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::simt::LoadSharedOp>(typeConverter,
                                                           benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::simt::LoadSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto elemLlvmTy = typeConverter->convertType(srcTy.getElementType());

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         elemLlvmTy, rewriter);

    Value addr = getSharedMemAddress(rewriter, smemObj, adaptor.getIndices(),
                                     srcTy, elemLlvmTy, loc);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // ctaId is std::optional<Value> in Triton 3.7.1: pass std::nullopt for a
    // local (non-remote) access. Passing Value() would wrap a *null* Value in a
    // non-empty optional, wrongly taking the remote `mapa` path and producing
    // invalid IR (silent ConvertTritonDistributedToLLVM failure).
    // NOTE (AMD): targetInfo.loadDShared emits a triton::amdgpu::MaskedLoadOp
    // intermediate; the AMD conversion pass must also run
    // populateMaskedOpsToLLVMPatterns to lower it (see
    // ConvertAMDDistributedToLLVM).
    Value val =
        targetInfo.loadDShared(rewriter, loc, addr, /*ctaId=*/std::nullopt,
                               elemLlvmTy, /*pred=*/b.true_val());
    rewriter.replaceOp(op, val);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct StoreSharedOpPattern
    : public ConvertOpToLLVMPattern<triton::simt::StoreSharedOp> {
  explicit StoreSharedOpPattern(LLVMTypeConverter &typeConverter,
                                const TargetInfoBase &targetInfo,
                                PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::simt::StoreSharedOp>(typeConverter,
                                                            benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::simt::StoreSharedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcTy = op.getDest().getType();
    auto elemLlvmTy = typeConverter->convertType(srcTy.getElementType());

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getDest(),
                                                         elemLlvmTy, rewriter);

    Value addr = getSharedMemAddress(rewriter, smemObj, adaptor.getIndices(),
                                     srcTy, elemLlvmTy, loc);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    // ctaId is std::optional<Value> in Triton 3.7.1: std::nullopt = local
    // store.
    targetInfo.storeDShared(rewriter, loc, addr, /*ctaId=*/std::nullopt,
                            adaptor.getValue(),
                            /*pred=*/b.true_val());
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct MemDescToPtrOpPattern
    : public ConvertOpToLLVMPattern<triton::simt::MemDescToPtrOp> {
  explicit MemDescToPtrOpPattern(LLVMTypeConverter &typeConverter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::simt::MemDescToPtrOp>(typeConverter,
                                                             benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::simt::MemDescToPtrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto srcTy = op.getSrc().getType();
    auto elemLlvmTy = typeConverter->convertType(srcTy.getElementType());

    auto smemObj = LLVM::getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                         elemLlvmTy, rewriter);

    SmallVector<Value> indices(adaptor.getIndices().begin(),
                               adaptor.getIndices().end());
    Value addr;
    if (indices.empty()) {
      addr = smemObj.getBase();
    } else {
      addr = getSharedMemAddress(rewriter, smemObj, indices, srcTy, elemLlvmTy,
                                 loc);
    }

    // `addr` is NVVM shared (AS 3). Frontend must use `pointer_type(...,
    // address_space=0)` for `memdesc_to_ptr` so the converted result is LLVM
    // generic (AS 0).
    auto resultTy = typeConverter->convertType(op.getResult().getType());
    auto genericPtrTy = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    if (resultTy != genericPtrTy)
      return op.emitOpError("memdesc_to_ptr result must be LLVM pointer in "
                            "address space 0 "
                            "(generic); got ")
             << resultTy;
    Value p = rewriter.create<LLVM::AddrSpaceCastOp>(loc, genericPtrTy, addr);
    rewriter.replaceOp(op, p);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};

struct SIMTExecRegionPattern
    : public ConvertOpToLLVMPattern<triton::simt::SIMTExecRegionOp> {
  explicit SIMTExecRegionPattern(LLVMTypeConverter &typeConverter,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::simt::SIMTExecRegionOp>(typeConverter,
                                                               benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::simt::SIMTExecRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() > 0) {
      llvm_unreachable("Unsupported SIMTExecRegionOp.");
      return failure();
    }

    Block *prevBlock = op->getBlock();
    Block *nextBlock = rewriter.splitBlock(prevBlock, op->getIterator());

    rewriter.setInsertionPointToEnd(prevBlock);
    rewriter.create<LLVM::BrOp>(op->getLoc(), &op.getDefaultRegion().front());

    op.getDefaultRegion().walk([&](simt::BlockYieldOp yieldOp) {
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(yieldOp, yieldOp.getOperands(),
                                              nextBlock);
    });

    nextBlock->getParent()->getBlocks().splice(
        nextBlock->getIterator(), op.getDefaultRegion().getBlocks());
    rewriter.eraseOp(op);
    return success();
  }

protected:
  const TargetInfoBase &targetInfo;
};
#endif

} // namespace

void mlir::triton::populateSIMTOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
#ifndef USE_MACA
  patterns
      .add<LoadSharedOpPattern, StoreSharedOpPattern, MemDescToPtrOpPattern>(
          typeConverter, targetInfo, benefit);
  patterns.add<SIMTExecRegionPattern>(typeConverter, targetInfo, benefit);
#endif
}
