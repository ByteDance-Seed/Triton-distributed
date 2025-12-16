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
#include <string>

using namespace mlir;
using namespace mlir::triton;
using namespace std::literals;

namespace {

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
  auto smemStrides = smemObj.getStrides(sharedTy, loc, rewriter);
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
    Value val = targetInfo.loadDShared(rewriter, loc, addr, std::nullopt,
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
    targetInfo.storeDShared(rewriter, loc, addr, std::nullopt,
                            adaptor.getValue(),
                            /*pred=*/b.true_val());
    rewriter.eraseOp(op);
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

} // namespace

void mlir::triton::populateSIMTOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<LoadSharedOpPattern, StoreSharedOpPattern>(typeConverter,
                                                          targetInfo, benefit);
  patterns.add<SIMTExecRegionPattern>(typeConverter, targetInfo, benefit);
}
