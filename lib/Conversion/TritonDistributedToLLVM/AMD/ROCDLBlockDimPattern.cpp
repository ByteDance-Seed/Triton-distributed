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
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/MLIRTypes.h"

using namespace mlir;

namespace {
// Convert rocdl.dim operations whose result is not i64 into i64 + trunc
template <typename OpTy>
struct ROCDLBlockDimPattern : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto origType = op.getResult().getType();
    if (origType.isInteger(64) || origType.isIndex())
      return failure();

    auto i64Type = rewriter.getI64Type();
    auto newDimOp = rewriter.create<OpTy>(op.getLoc(), i64Type);

    Value trunc = rewriter.create<LLVM::TruncOp>(op.getLoc(), origType,
                                                 newDimOp.getResult());
    rewriter.replaceOp(op, trunc);
    return success();
  }
};

using ROCDLBlockDimXPattern = ROCDLBlockDimPattern<ROCDL::BlockDimXOp>;
using ROCDLBlockDimYPattern = ROCDLBlockDimPattern<ROCDL::BlockDimYOp>;
using ROCDLBlockDimZPattern = ROCDLBlockDimPattern<ROCDL::BlockDimZOp>;

} // namespace

namespace mlir::triton::AMD {
void populateROCDLBlockDimPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit) {
  patterns
      .add<ROCDLBlockDimXPattern, ROCDLBlockDimYPattern, ROCDLBlockDimZPattern>(
          typeConverter, benefit);
}
} // namespace mlir::triton::AMD