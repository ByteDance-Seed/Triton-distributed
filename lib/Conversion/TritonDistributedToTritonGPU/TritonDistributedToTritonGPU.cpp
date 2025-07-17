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
#include "TritonDistributed/Conversion/TritonDistributedToTritonGPU/TritonDistributedToTritonGPUPass.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "llvm/ADT/APSInt.h"
#include <numeric>

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"

#define GEN_PASS_CLASSES
#include "TritonDistributed/Conversion/TritonDistributedToTritonGPU/Passes.h.inc"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "third_party/proton/dialect/include/Dialect/Proton/IR/Dialect.h"

#define DEBUG_TYPE "convert-triton-to-tritongpu"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace {

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

// pass named attrs (e.g., tt.contiguity) from Triton to Triton
static void addNamedAttrs(Operation *op, DictionaryAttr dictAttrs) {
  for (const NamedAttribute attr : dictAttrs.getValue())
    if (!op->hasAttr(attr.getName()))
      op->setAttr(attr.getName(), attr.getValue());
}

template <class Op> struct GenericOpPattern : public OpConversionPattern<Op> {
  using OpConversionPattern<Op>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> retTypes;
    if (failed(this->getTypeConverter()->convertTypes(op->getResultTypes(),
                                                      retTypes)))
      return failure();
    rewriter.replaceOpWithNewOp<Op>(op, retTypes, adaptor.getOperands(),
                                    op->getAttrs());

    return success();
  }
};

class ArithConstantPattern : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type retType = getTypeConverter()->convertType(op.getType());
    auto retShapedType = cast<ShapedType>(retType);
    auto value = dyn_cast<DenseElementsAttr>(adaptor.getValue());
    if (isa<RankedTensorType>(retShapedType)) {
      assert(value && "expected a dense elements attribute");
      // This is a hack. We just want to add encoding.
      value = value.reshape(retShapedType);
    }
    addNamedAttrs(rewriter.replaceOpWithNewOp<arith::ConstantOp>(
                      op, retShapedType, value),
                  adaptor.getAttributes());
    return success();
  }
};

void populateArithPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      TritonGPUConversionTarget &target) {
  // --------------
  // Add legality and rewrite pattern rules for operations
  // from the Arith dialect. The basic premise is that
  // Arith operations require both inputs to have the same
  // non-null encoding
  // --------------
  MLIRContext *context = patterns.getContext();
  // TODO: there's probably a better way to avoid adding all ops one-by-one
  patterns.add<
      ArithConstantPattern, GenericOpPattern<arith::AddIOp>,
      GenericOpPattern<arith::SubIOp>, GenericOpPattern<arith::MulIOp>,
      GenericOpPattern<arith::DivUIOp>, GenericOpPattern<arith::DivSIOp>,
      GenericOpPattern<arith::CeilDivUIOp>,
      GenericOpPattern<arith::CeilDivSIOp>,
      GenericOpPattern<arith::FloorDivSIOp>, GenericOpPattern<arith::RemUIOp>,
      GenericOpPattern<arith::RemSIOp>, GenericOpPattern<arith::AndIOp>,
      GenericOpPattern<arith::OrIOp>, GenericOpPattern<arith::XOrIOp>,
      GenericOpPattern<arith::ShLIOp>, GenericOpPattern<arith::ShRUIOp>,
      GenericOpPattern<arith::ShRSIOp>, // NegFOp
      // Floating point
      GenericOpPattern<arith::AddFOp>, GenericOpPattern<arith::SubFOp>,
      // MaxMin
      GenericOpPattern<arith::MaximumFOp>, GenericOpPattern<arith::MaxNumFOp>,
      GenericOpPattern<arith::MaxSIOp>, GenericOpPattern<arith::MaxUIOp>,
      GenericOpPattern<arith::MinimumFOp>, GenericOpPattern<arith::MinNumFOp>,
      GenericOpPattern<arith::MinSIOp>, GenericOpPattern<arith::MinUIOp>,
      // Floating point
      GenericOpPattern<arith::MulFOp>, GenericOpPattern<arith::DivFOp>,
      GenericOpPattern<arith::RemFOp>,
      // Cmp
      GenericOpPattern<arith::CmpIOp>, GenericOpPattern<arith::CmpFOp>,
      // Select
      GenericOpPattern<arith::SelectOp>,
      // Cast Ops
      GenericOpPattern<arith::TruncIOp>, GenericOpPattern<arith::TruncFOp>,
      GenericOpPattern<arith::ExtUIOp>, GenericOpPattern<arith::ExtSIOp>,
      GenericOpPattern<arith::ExtFOp>, GenericOpPattern<arith::SIToFPOp>,
      GenericOpPattern<arith::FPToSIOp>, GenericOpPattern<arith::FPToUIOp>,
      GenericOpPattern<arith::UIToFPOp>>(typeConverter, context);
}

void populateMathPatternsAndLegality(TritonGPUTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     TritonGPUConversionTarget &target) {
  MLIRContext *context = patterns.getContext();
  // Rewrite rule
  patterns.add<GenericOpPattern<math::ExpOp>, GenericOpPattern<math::Exp2Op>,
               GenericOpPattern<math::FloorOp>, GenericOpPattern<math::CeilOp>,
               GenericOpPattern<math::CosOp>, GenericOpPattern<math::SinOp>,
               GenericOpPattern<math::LogOp>, GenericOpPattern<math::Log2Op>,
               GenericOpPattern<math::ErfOp>, GenericOpPattern<math::AbsFOp>,
               GenericOpPattern<math::AbsIOp>, GenericOpPattern<math::SqrtOp>,
               GenericOpPattern<math::RsqrtOp>, GenericOpPattern<math::FmaOp>>(
      typeConverter, context);
}

//
// Triton patterns
//
struct TritonExpandDimsPattern
    : public OpConversionPattern<triton::ExpandDimsOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ExpandDimsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Type retType = op.getType());
    RankedTensorType argType =
        cast<RankedTensorType>(adaptor.getSrc().getType());
    Attribute _argEncoding = argType.getEncoding();
    if (!_argEncoding)
      return failure();
    auto argEncoding = cast<triton::gpu::BlockedEncodingAttr>(_argEncoding);
    // return shape
    auto retShape = argType.getShape().vec();
    retShape.insert(retShape.begin() + op.getAxis(), 1);
    // return encoding
    auto retSizePerThread = llvm::to_vector(argEncoding.getSizePerThread());
    retSizePerThread.insert(retSizePerThread.begin() + op.getAxis(), 1);
    auto retThreadsPerWarp = to_vector(argEncoding.getThreadsPerWarp());
    retThreadsPerWarp.insert(retThreadsPerWarp.begin() + op.getAxis(), 1);
    auto retWarpsPerCTA = to_vector(argEncoding.getWarpsPerCTA());
    retWarpsPerCTA.insert(retWarpsPerCTA.begin() + op.getAxis(), 1);
    SmallVector<unsigned, 4> retOrder(retShape.size());
    std::iota(retOrder.begin(), retOrder.end(), 0);

    auto argCTALayout = argEncoding.getCTALayout();
    auto retCTAsPerCGA = insertOne(argCTALayout.getCTAsPerCGA(), op.getAxis());
    auto retCTASplitNum =
        insertOne(argCTALayout.getCTASplitNum(), op.getAxis());
    auto retCTAOrder = insertOrder(argCTALayout.getCTAOrder(), op.getAxis());
    auto retCTALayout = triton::gpu::CTALayoutAttr::get(
        getContext(), retCTAsPerCGA, retCTASplitNum, retCTAOrder);

    triton::gpu::BlockedEncodingAttr retEncoding =
        triton::gpu::BlockedEncodingAttr::get(getContext(), retSizePerThread,
                                              retThreadsPerWarp, retWarpsPerCTA,
                                              retOrder, retCTALayout);
    // convert operand to slice of return type
    Attribute newArgEncoding = triton::gpu::SliceEncodingAttr::get(
        getContext(), op.getAxis(), retEncoding);
    RankedTensorType newArgType = RankedTensorType::get(
        argType.getShape(), argType.getElementType(), newArgEncoding);
    // construct new op
    auto newSrc = rewriter.create<triton::gpu::ConvertLayoutOp>(
        op.getLoc(), newArgType, adaptor.getSrc());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(
                      op, newSrc, adaptor.getAxis()),
                  adaptor.getAttributes());
    return success();
  }

private:
  template <typename T>
  SmallVector<T> insertOne(ArrayRef<T> vec, unsigned axis) const {
    SmallVector<T> res(vec.begin(), vec.end());
    res.insert(res.begin() + axis, 1);
    return res;
  }

  // Example:    order = [   0, 2, 1, 3], dim = 2
  //          resOrder = [2, 0, 3, 1, 4]
  SmallVector<unsigned> insertOrder(ArrayRef<unsigned> order,
                                    unsigned axis) const {
    SmallVector<unsigned> resOrder(order.begin(), order.end());
    for (unsigned i = 0; i < resOrder.size(); ++i)
      if (resOrder[i] >= axis)
        ++resOrder[i];
    resOrder.insert(resOrder.begin(), axis);
    return resOrder;
  }
};

struct TritonDotPattern : public OpConversionPattern<triton::DotOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType origType = op.getType();
    auto origShape = origType.getShape();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();
    int numWarps = typeConverter->getNumWarps();
    int threadsPerWarp = typeConverter->getThreadsPerWarp();
    int numCTAs = typeConverter->getNumCTAs();
    auto rank = origShape.size();
    SmallVector<unsigned> retSizePerThread(rank, 1);
    auto numElements = product<int64_t>(origShape);
    if (numElements / (numWarps * threadsPerWarp) >= 4) {
      retSizePerThread[rank - 1] = 2;
      retSizePerThread[rank - 2] = 2;
    }
    if (numElements / (numWarps * threadsPerWarp) >= 16) {
      retSizePerThread[rank - 1] = 4;
      retSizePerThread[rank - 2] = 4;
    }
    retSizePerThread[rank - 1] = std::min(
        retSizePerThread[rank - 1], static_cast<unsigned>(origShape[rank - 1]));
    retSizePerThread[rank - 2] = std::min(
        retSizePerThread[rank - 2], static_cast<unsigned>(origShape[rank - 2]));

    SmallVector<unsigned> retOrder(rank);
    for (unsigned i = 0; i < rank; ++i)
      retOrder[i] = rank - 1 - i;
    Attribute dEncoding = triton::gpu::BlockedEncodingAttr::get(
        getContext(), origShape, retSizePerThread, retOrder, numWarps,
        threadsPerWarp, numCTAs);
    RankedTensorType retType =
        RankedTensorType::get(origShape, origType.getElementType(), dEncoding);
    // a & b must be of smem layout
    auto aType = cast<RankedTensorType>(adaptor.getA().getType());
    auto bType = cast<RankedTensorType>(adaptor.getB().getType());
    Type aEltType = aType.getElementType();
    Type bEltType = bType.getElementType();
    Attribute aEncoding = aType.getEncoding();
    Attribute bEncoding = bType.getEncoding();
    if (!aEncoding || !bEncoding)
      return failure();
    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value c = adaptor.getC();
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(aEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 0, dEncoding, aEltType);
      auto dstType =
          RankedTensorType::get(aType.getShape(), aEltType, encoding);
      a = rewriter.create<triton::gpu::ConvertLayoutOp>(a.getLoc(), dstType, a);
    }
    if (!mlir::isa<triton::gpu::DotOperandEncodingAttr>(bEncoding)) {
      Attribute encoding = triton::gpu::DotOperandEncodingAttr::get(
          getContext(), 1, dEncoding, bEltType);
      auto dstType =
          RankedTensorType::get(bType.getShape(), bEltType, encoding);
      b = rewriter.create<triton::gpu::ConvertLayoutOp>(b.getLoc(), dstType, b);
    }
    c = rewriter.create<triton::gpu::ConvertLayoutOp>(c.getLoc(), retType, c);

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::DotOp>(
                      op, retType, a, b, c, adaptor.getInputPrecision(),
                      adaptor.getMaxNumImpreciseAcc()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonCatPattern : public OpConversionPattern<triton::CatOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // The cat op satisfy two conditions:
    // 1. output.numel = lhs.numel + rhs.numel
    // 2. output.total_elems_per_thread =
    // next_power_of_2(lhs.total_elems_per_thread + rhs.total_elems_per_thread)
    // For now, this behaves like generic, but this
    // will evolve when we add support for `can_reorder=False`.
    auto retType = cast<RankedTensorType>(
        this->getTypeConverter()->convertType(op.getType()));
    auto retEncoding =
        cast<triton::gpu::BlockedEncodingAttr>(retType.getEncoding());
    auto lhsType = adaptor.getLhs().getType();
    auto rhsType = adaptor.getRhs().getType();
    auto lhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(lhsType);
    auto rhsTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(rhsType);
    auto retTotalElemsPerThread = triton::gpu::getTotalElemsPerThread(retType);
    auto retShape = retType.getShape();
    auto retOrder = retEncoding.getOrder();
    auto retThreadsPerWarp = retEncoding.getThreadsPerWarp();
    auto retWarpsPerCTA = retEncoding.getWarpsPerCTA();
    // Get new retSizePerThread if ret elems per thread is not enough.
    // We have to round it up to the next power of 2 due to triton's tensor size
    // constraint.
    auto newRetTotalElemsPerThread =
        nextPowOf2(lhsTotalElemsPerThread + rhsTotalElemsPerThread);
    auto newRetSizePerThread = llvm::to_vector(retEncoding.getSizePerThread());
    newRetSizePerThread[retOrder[0]] *=
        newRetTotalElemsPerThread / retTotalElemsPerThread;
    triton::gpu::BlockedEncodingAttr newRetEncoding =
        triton::gpu::BlockedEncodingAttr::get(
            getContext(), newRetSizePerThread, retThreadsPerWarp,
            retWarpsPerCTA, retOrder, retEncoding.getCTALayout());
    auto newRetType = RankedTensorType::get(retShape, retType.getElementType(),
                                            newRetEncoding);
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::CatOp>(
                      op, newRetType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonJoinOpPattern : public OpConversionPattern<triton::JoinOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(JoinOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Simply rely on type inference for this op.  (Notably, GenericOpPattern
    // does not do this, instead it assigns the default layout to the ins and
    // outs.)
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::JoinOp>(
                      op, adaptor.getLhs(), adaptor.getRhs()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonSplitOpPattern : public OpConversionPattern<triton::SplitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const {
    auto src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = dyn_cast<BlockedEncodingAttr>(srcTy.getEncoding());
    int rank = srcEnc.getOrder().size();
    auto typeConverter = getTypeConverter<TritonGPUTypeConverter>();

    // The operand to split must have:
    //  - a blocked layout, with
    //  - sizePerThread = 2 in the last dimension,
    //  - threadsPerWarp, warpsPerCTA, and CTAsPerCGA = 1 in the last dim, and
    //  - the last dimension minor.
    // If that's not the case, add a convert before the split.
    if (!srcEnc || srcEnc.getSizePerThread().back() != 2 ||
        srcEnc.getOrder().front() != rank - 1) {
      // If we take the default encoding for the op's result (i.e. post-split)
      // and add 1 to the end of each dim, that gives us what we want.  Other
      // than making a legal src encoding, our choice of layout doesn't matter;
      // it'll get fixed by RemoveLayoutConversions.
      auto defaultEnc = getDefaultBlockedEncoding(
          getContext(),
          cast<RankedTensorType>(op.getResult(0).getType()).getShape(),
          typeConverter->getNumWarps(), typeConverter->getThreadsPerWarp(),
          typeConverter->getNumCTAs());

      auto append = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res(vals);
        res.push_back(val);
        return res;
      };
      auto prepend = [&](ArrayRef<unsigned> vals, unsigned val) {
        SmallVector<unsigned> res;
        res.push_back(val);
        res.append(vals.begin(), vals.end());
        return res;
      };

      srcEnc = BlockedEncodingAttr::get(
          getContext(), append(defaultEnc.getSizePerThread(), 2),
          append(defaultEnc.getThreadsPerWarp(), 1),
          append(defaultEnc.getWarpsPerCTA(), 1),
          prepend(defaultEnc.getOrder(), rank - 1),
          CTALayoutAttr::get(getContext(),
                             append(defaultEnc.getCTAsPerCGA(), 1),
                             append(defaultEnc.getCTASplitNum(), 1),
                             prepend(defaultEnc.getCTAOrder(), rank - 1)));
      srcTy = RankedTensorType::get(srcTy.getShape(), srcTy.getElementType(),
                                    srcEnc);
      src = rewriter.create<ConvertLayoutOp>(op.getLoc(), srcTy, src);
    }

    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::SplitOp>(op, src),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonTransPattern : public OpConversionPattern<TransOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TransOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = adaptor.getSrc();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto srcEnc = srcTy.getEncoding();
    if (!srcEnc)
      return failure();
    addNamedAttrs(rewriter.replaceOpWithNewOp<TransOp>(op, src, op.getOrder()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonBroadcastPattern
    : public OpConversionPattern<triton::BroadcastOp> {
  using OpConversionPattern::OpConversionPattern;

  // This creates a tensor with the new shape but the argument's layout
  LogicalResult
  matchAndRewrite(BroadcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(adaptor.getSrc().getType());
    auto srcEncoding = srcType.getEncoding();
    if (!srcEncoding)
      return failure();
    Type retType = RankedTensorType::get(
        op.getType().getShape(), op.getType().getElementType(), srcEncoding);
    // Type retType = this->getTypeConverter()->convertType(op.getType());
    addNamedAttrs(rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
                      op, retType, adaptor.getOperands()),
                  adaptor.getAttributes());
    return success();
  }
};

struct TritonReducePattern : public OpConversionPattern<triton::ReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newReduce = rewriter.create<triton::ReduceOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis());
    addNamedAttrs(newReduce, adaptor.getAttributes());

    auto &newCombineOp = newReduce.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newReduce.getResult());
    return success();
  }
};

struct TritonScanPattern : public OpConversionPattern<triton::ScanOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newScan = rewriter.create<triton::ScanOp>(
        op.getLoc(), adaptor.getOperands(), adaptor.getAxis(), op.getReverse());
    addNamedAttrs(newScan, adaptor.getAttributes());

    auto &newCombineOp = newScan.getCombineOp();
    rewriter.cloneRegionBefore(op.getCombineOp(), newCombineOp,
                               newCombineOp.end());
    rewriter.replaceOp(op, newScan.getResult());
    return success();
  }
};

class TritonFuncOpPattern : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    TypeConverter::SignatureConversion result(op.getNumArguments());
    auto newOp = rewriter.replaceOpWithNewOp<triton::FuncOp>(
        op, op.getName(), op.getFunctionType());
    addNamedAttrs(newOp, adaptor.getAttributes());
    rewriter.inlineRegionBefore(op.getBody(), newOp.getBody(),
                                newOp.getBody().end());
    // Convert just the entry block. The remaining unstructured control flow is
    // converted by br patterns.
    if (!newOp.getBody().empty())
      rewriter.applySignatureConversion(&newOp.getBody().front(), result,
                                        converter);
    return success();
  }
};

class TritonCallOpPattern : public OpConversionPattern<triton::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = rewriter.replaceOpWithNewOp<triton::CallOp>(
        op, op.getCallee(), op.getResultTypes(), adaptor.getOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());
    return success();
  }
};

class TritonReturnOpPattern : public OpConversionPattern<ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReturnOp op, ReturnOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

void populateTritonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns, unsigned numCTAs) {
  MLIRContext *context = patterns.getContext();
  patterns.insert< // TODO: view should have custom pattern that views the
                   // layout
      // clang-format off
      GenericOpPattern<triton::AdvanceOp>,
      GenericOpPattern<triton::MakeTensorPtrOp>,
      GenericOpPattern<triton::ReshapeOp>,
      GenericOpPattern<triton::BitcastOp>,
      GenericOpPattern<triton::FpToFpOp>,
      GenericOpPattern<triton::IntToPtrOp>,
      GenericOpPattern<triton::PtrToIntOp>,
      GenericOpPattern<triton::SplatOp>,
      GenericOpPattern<triton::AddPtrOp>,
      TritonBroadcastPattern,
      TritonCatPattern,
      TritonJoinOpPattern,
      TritonSplitOpPattern,
      GenericOpPattern<triton::ClampFOp>,
      GenericOpPattern<triton::PreciseSqrtOp>,
      GenericOpPattern<triton::PreciseDivFOp>,
      GenericOpPattern<triton::MulhiUIOp>,
      GenericOpPattern<triton::ElementwiseInlineAsmOp>,
      TritonReducePattern,
      GenericOpPattern<triton::ReduceReturnOp>,
      TritonScanPattern,
      GenericOpPattern<triton::ScanReturnOp>,
      GenericOpPattern<triton::MakeRangeOp>,
      TritonExpandDimsPattern,
      TritonTransPattern,
      TritonDotPattern,
      GatherScatterOpPattern<DescriptorGatherOp>,
      GatherScatterOpPattern<DescriptorScatterOp>,
      GenericOpPattern<triton::LoadOp>,
      GenericOpPattern<triton::StoreOp>,
      GenericOpPattern<triton::HistogramOp>,
      GenericOpPattern<triton::GatherOp>,
      GenericOpPattern<triton::ExternElementwiseOp>,
      GenericOpPattern<triton::PrintOp>,
      GenericOpPattern<triton::AssertOp>,
      GenericOpPattern<triton::AtomicCASOp>,
      GenericOpPattern<triton::AtomicRMWOp>,
      GenericOpPattern<triton::DescriptorLoadOp>,
      GenericOpPattern<triton::DescriptorStoreOp>,
      GenericOpPattern<triton::DescriptorReduceOp>,
      // this assumes the right layout will be set later for dot scaled.
      GenericOpPattern<triton::DotScaledOp>,
      GenericOpPattern<triton::CallOp>,
      GenericOpPattern<ReturnOp>,
      TritonFuncOpPattern
      // clang-format on
      >(typeConverter, context);
}
// Proton patterns
// NOTE: Because Proton's inputs are scalars and not tensors this conversion
// isn't strictly necessary however you could envision a case where we pass in
// tensors in for Triton object specific tracing operations in which case we
// would need to fill in the OpConversionPattern
void populateProtonPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<triton::proton::RecordOp>>(typeConverter,
                                                           context);
}

// Distributed patterns
void populateDistributedPatterns(TritonGPUTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<triton::distributed::WaitOp>>(typeConverter,
                                                              context);
  patterns.add<GenericOpPattern<triton::distributed::ConsumeTokenOp>>(
      typeConverter, context);
}

Value promoteToShared(Value val, RewriterBase &rewriter, Location loc) {
  auto tensorTy = dyn_cast<RankedTensorType>(val.getType());
  if (!tensorTy)
    return val;
  auto encoding = getSharedEncoding(tensorTy);
  Attribute sharedMemorySpace =
      triton::gpu::SharedMemorySpaceAttr::get(tensorTy.getContext());
  MemDescType newMemDescType =
      MemDescType::get(tensorTy.getShape(), tensorTy.getElementType(), encoding,
                       sharedMemorySpace, /*mutableMemory=*/true);
  Value alloc = rewriter.create<triton::gpu::LocalAllocOp>(loc, newMemDescType);
  rewriter.create<triton::gpu::LocalStoreOp>(loc, val, alloc);
  return alloc;
};

struct SIMTExecRegionPattern
    : public OpConversionPattern<simt::SIMTExecRegionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(simt::SIMTExecRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp = cast<simt::SIMTExecRegionOp>(
        rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // ops in simt region still use distributed tensor, will be promoted later.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }

    // promote operands to shared memory
    SmallVector<Value> newOperands = adaptor.getOperands();
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(newOp);
      for (size_t i = 0; i < newOp->getNumOperands(); ++i) {
        auto curOperand = newOperands[i];
        newOperands[i] = promoteToShared(curOperand, rewriter, op->getLoc());
      }
    }
    newOp->setOperands(newOperands);

    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    int32_t cnt = 0;
    for (Type type : op.getResultTypes()) {
      Type newType = newOperands[cnt].getType();
      newResultTypes.push_back(newType);
      cnt += 1;
    }
    for (auto t : llvm::zip(newOp.getResults(), newOperands))
      std::get<0>(t).setType(std::get<1>(t).getType());

    // store result into distributed tensor
    SmallVector<Value> newResults = newOp.getResults();
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(newOp);
      for (size_t i = 0; i < op.getNumResults(); ++i) {
        auto curResult = newResults[i];
        auto memDescType = dyn_cast<MemDescType>(curResult.getType());
        if (memDescType && memDescType.getMemorySpace() &&
            isa<SharedMemorySpaceAttr>(memDescType.getMemorySpace())) {
          Type oldDistType =
              typeConverter->convertType(op->getResult(i).getType());
          Value output = rewriter.create<LocalLoadOp>(op->getLoc(), oldDistType,
                                                      curResult);
          newResults[i] = output;
        }
      }
    }
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

// SIMT patterns
void populateSIMTPatterns(TritonGPUTypeConverter &typeConverter,
                          RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<SIMTExecRegionPattern, GenericOpPattern<simt::BlockYieldOp>>(
      typeConverter, context);
}

void populateTensorPatterns(TritonGPUTypeConverter &typeConverter,
                            RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<tensor::ExtractOp>,
               GenericOpPattern<tensor::InsertOp>>(typeConverter, context);
}

//
// SCF patterns
//
// This is borrowed from ConvertForOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
struct SCFForPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());

    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFIfPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // TODO: Generalize this to any type conversion, not just 1:1.
    //
    // We need to implement something more sophisticated here that tracks which
    // types convert to which other types and does the appropriate
    // materialization logic.
    // For example, it's possible that one result type converts to 0 types and
    // another to 2 types, so newResultTypes would at least be the right size to
    // not crash in the llvm::zip call below, but then we would set the the
    // wrong type on the SSA values! These edge cases are also why we cannot
    // safely use the TypeConverter::convertTypes helper here.
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    // See comments in the ForOp pattern for why we clone without regions and
    // then inline.
    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    // Update the operands and types.
    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from ConvertFIfOpTypes in
//    SCF/Transforms/StructuralTypeConversions.cpp
class SCFWhilePattern : public OpConversionPattern<scf::WhileOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::WhileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto *converter = getTypeConverter();
    assert(converter);
    SmallVector<Type> newResultTypes;
    if (failed(converter->convertTypes(op.getResultTypes(), newResultTypes)))
      return failure();

    auto newOp = rewriter.create<scf::WhileOp>(op.getLoc(), newResultTypes,
                                               adaptor.getOperands());
    for (auto i : {0u, 1u}) {
      auto &dstRegion = newOp.getRegion(i);
      rewriter.inlineRegionBefore(op.getRegion(i), dstRegion, dstRegion.end());
      if (failed(rewriter.convertRegionTypes(&dstRegion, *converter)))
        return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

class SCFConditionPattern : public OpConversionPattern<scf::ConditionOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::ConditionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.modifyOpInPlace(op,
                             [&]() { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

void populateSCFPatterns(TritonGPUTypeConverter &typeConverter,
                         RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<GenericOpPattern<scf::YieldOp>, SCFForPattern, SCFIfPattern,
               SCFWhilePattern, SCFConditionPattern>(typeConverter, context);
}

// CF

class CFBranchPattern : public OpConversionPattern<cf::BranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::BranchOp op, cf::BranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::BranchOp>(
        op, op.getSuccessor(), adaptor.getOperands());
    if (failed(rewriter.convertRegionTypes(newOp.getSuccessor()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

class CFCondBranchPattern : public OpConversionPattern<cf::CondBranchOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(cf::CondBranchOp op, cf::CondBranchOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto converter = getTypeConverter();
    auto newOp = rewriter.replaceOpWithNewOp<cf::CondBranchOp>(
        op, adaptor.getCondition(), op.getTrueDest(),
        adaptor.getTrueDestOperands(), op.getFalseDest(),
        adaptor.getFalseDestOperands());
    addNamedAttrs(newOp, adaptor.getAttributes());

    if (failed(rewriter.convertRegionTypes(newOp.getTrueDest()->getParent(),
                                           *converter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(newOp.getFalseDest()->getParent(),
                                           *converter)))
      return failure();
    return success();
  }
};

void populateCFPatterns(TritonGPUTypeConverter &typeConverter,
                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  patterns.add<CFCondBranchPattern, CFBranchPattern>(typeConverter, context);
}
//

Value unrealizedCastMaterialization(OpBuilder &builder, Type type,
                                    ValueRange inputs, Location loc) {
  auto cast = builder.create<UnrealizedConversionCastOp>(loc, type, inputs);
  return cast.getResult(0);
}

class SIMTRegionTypeConverter : public TypeConverter {
public:
  SIMTRegionTypeConverter(MLIRContext *context, int numWarps,
                          int threadsPerWarp, int numCTAs)
      : context(context), numWarps(numWarps), threadsPerWarp(threadsPerWarp),
        numCTAs(numCTAs) {
    addConversion([](Type type) { return type; });

    // Add encoding for tensor
    addConversion([this](RankedTensorType tensorType) -> MemDescType {
      // types with encoding are already in the right format
      // TODO: check for layout encodings more specifically
      if (!tensorType.getEncoding()) {
        ArrayRef<int64_t> shape = tensorType.getShape();
        triton::gpu::BlockedEncodingAttr encoding =
            getDefaultBlockedEncoding(this->context, shape, this->numWarps,
                                      this->threadsPerWarp, this->numCTAs);
        tensorType =
            RankedTensorType::get(shape, tensorType.getElementType(), encoding);
      }
      auto encoding = getSharedEncoding(tensorType);
      Attribute sharedMemorySpace =
          triton::gpu::SharedMemorySpaceAttr::get(tensorType.getContext());
      MemDescType memDescType =
          MemDescType::get(tensorType.getShape(), tensorType.getElementType(),
                           encoding, sharedMemorySpace, /*mutableMemory=*/true);
      return memDescType;
    });

    // Add encoding for tensor pointer
    addConversion([this](triton::PointerType ptrType) -> triton::PointerType {
      // Check whether tensor pointer `tt.ptr<tensor<>>`
      auto pointeeTensorType =
          dyn_cast<RankedTensorType>(ptrType.getPointeeType());
      if (pointeeTensorType == nullptr)
        return ptrType;

      // Add layout into the tensor
      auto convertedTensorType = convertType(pointeeTensorType);
      return triton::PointerType::get(convertedTensorType,
                                      ptrType.getAddressSpace());
    });

    addSourceMaterialization(unrealizedCastMaterialization);
    addTargetMaterialization(unrealizedCastMaterialization);
  }
  int getNumWarps() const { return numWarps; }
  int getThreadsPerWarp() const { return threadsPerWarp; }
  int getNumCTAs() const { return numCTAs; }

private:
  MLIRContext *context;
  int numWarps;
  int threadsPerWarp;
  int numCTAs;
};

struct TensorExtractPromotionPattern
    : public OpConversionPattern<tensor::ExtractOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::ExtractOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    simt::SIMTExecRegionOp simtExecRegionOp =
        op->getParentOfType<simt::SIMTExecRegionOp>();
    if (!simtExecRegionOp)
      return failure();

    auto isValueOutOfRegion = [&](Value value) -> bool {
      Region *defRegion;

      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        Block *block = blockArg.getOwner();
        defRegion = block->getParent();
        for (Region &region : simtExecRegionOp->getRegions()) {
          if (region.isAncestor(defRegion)) {
            return false;
          }
        }
      } else {
        Operation *defOp = value.getDefiningOp();
        if (!defOp)
          return true;
        if (auto curRegionOp =
                defOp->getParentOfType<simt::SIMTExecRegionOp>()) {
          return curRegionOp != simtExecRegionOp;
        } else {
          return true;
        }
      }
      return true;
    };

    Value src;
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(simtExecRegionOp);
      // some tensors that are only read by op within simt region are not
      // captured by SIMTExecRegionOp
      if (isValueOutOfRegion(adaptor.getTensor())) {
        // promote the origin tensor, not the converted.
        src = promoteToShared(op.getTensor(), rewriter, op->getLoc());
      } else {
        src = adaptor.getTensor();
      }
    }

    SmallVector<Type> retTypes;
    auto srcType = src.getType();
    if (auto memDescType = dyn_cast<MemDescType>(srcType)) {
      auto newOp = rewriter.create<simt::LoadSharedOp>(op->getLoc(), src,
                                                       adaptor.getIndices());
      rewriter.replaceOp(op, newOp.getResult());
    } else {
      return failure();
    }

    return success();
  }
};

struct TensorInsertPromotionPattern
    : public OpConversionPattern<tensor::InsertOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tensor::InsertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = adaptor.getDest().getType();
    SmallVector<Type> retTypes;
    if (auto memDescType = dyn_cast<MemDescType>(srcType)) {
      auto newOp = rewriter.create<simt::StoreSharedOp>(
          op->getLoc(), adaptor.getScalar(), adaptor.getDest(),
          adaptor.getIndices());
      rewriter.replaceOp(op, adaptor.getDest());
    } else {
      return failure();
    }
    return success();
  }
};

// This is borrowed from SCFForOpPattern
struct SCFForOpPromotionPattern : public OpConversionPattern<scf::ForOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(scf::ForOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto newOp =
        cast<scf::ForOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }
    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    newOp->setOperands(adaptor.getOperands());
    // Update the result types to the new converted types.
    SmallVector<Type> newResultTypes;
    for (Type type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));

    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

// This is borrowed from SCFIfOpPattern
class SCFIfOpPromotionPattern : public OpConversionPattern<scf::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Type> newResultTypes;
    for (auto type : op.getResultTypes()) {
      Type newType = typeConverter->convertType(type);
      if (!newType)
        return rewriter.notifyMatchFailure(op, "not a 1:1 type conversion");
      newResultTypes.push_back(newType);
    }

    scf::IfOp newOp =
        cast<scf::IfOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().end());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().end());

    newOp->setOperands(adaptor.getOperands());
    for (auto t : llvm::zip(newOp.getResults(), newResultTypes))
      std::get<0>(t).setType(std::get<1>(t));
    rewriter.replaceOp(op, newOp.getResults());
    return success();
  }
};

struct SIMTExecRegionPromotionPattern
    : public OpConversionPattern<simt::SIMTExecRegionOp> {
  using OpConversionPattern::OpConversionPattern;
  // Ref: ConvertForOpTypes
  LogicalResult
  matchAndRewrite(simt::SIMTExecRegionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // move to SIMTExecRegionPattern?
    IRMapping mapping;
    SmallVector<int64_t> argMapIndices;
    SmallVector<Value> newInitArgs;
    SmallVector<Value> newResults(op->getNumResults());
    auto &region = op.getDefaultRegion();
    int32_t cnt = 0;
    for (auto t : llvm::zip(region.getArguments(), adaptor.getOperands())) {
      auto arg = std::get<0>(t);
      auto operand = std::get<1>(t);
      if (llvm::isa<RankedTensorType>(arg.getType())) {
        mapping.map(arg, operand);
      } else {
        newInitArgs.push_back(operand);
        argMapIndices.push_back(cnt);
      }
      newResults[cnt] = operand;
      cnt += 1;
    }
    auto newOp = rewriter.create<mlir::triton::simt::SIMTExecRegionOp>(
        op->getLoc(), newInitArgs);
    auto &newRegion = newOp.getDefaultRegion();
    for (size_t i = 0; i < newRegion.getNumArguments(); ++i) {
      mapping.map(adaptor.getOperands()[argMapIndices[i]],
                  newRegion.getArgument(i));
      newResults[argMapIndices[i]] = newOp->getResult(i);
    }
    // Erase the empty block that was inserted by the builder.
    rewriter.eraseBlock(&newRegion.front());
    // Clone the loop body and remap the block arguments of the collapsed loops
    // (inlining does not support a cancellable block argument mapping).
    rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
                               newOp.getRegion().begin(), mapping);
    if (auto yieldOp =
            dyn_cast<simt::BlockYieldOp>(newRegion.front().getTerminator())) {
      SmallVector<Value> yieldResult;
      for (auto v : argMapIndices) {
        yieldResult.push_back(yieldOp->getOperand(v));
      }
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPoint(yieldOp);
      rewriter.replaceOpWithNewOp<simt::BlockYieldOp>(yieldOp, yieldResult);
    }
    // region.cloneInto(&newRegion, bvm);
    // rewriter.cloneRegionBefore(op.getRegion(), newOp.getRegion(),
    //                            newOp.getRegion().end(), bvm);
    // auto newOp = cast<simt::SIMTExecRegionOp>(
    //     rewriter.cloneWithoutRegions(*op.getOperation()));
    // rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
    //                             newOp.getRegion().end());

    // Now, update all the types.

    // Convert the types of block arguments within the given region. This
    // replaces each block with a new block containing the updated signature.
    // The entry block may have a special conversion if `entryConversion` is
    // provided. On success, the new entry block to the region is returned for
    // convenience. Otherwise, failure is returned.
    if (failed(rewriter.convertRegionTypes(&newOp.getRegion(),
                                           *getTypeConverter()))) {
      return rewriter.notifyMatchFailure(op, "could not convert body types");
    }

    // Change the clone to use the updated operands. We could have cloned with
    // a IRMapping, but this seems a bit more direct.
    rewriter.replaceOp(op, newResults);

    return success();
  }
};

void populateSIMTReigonPromotionPattern(SIMTRegionTypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  MLIRContext *context = patterns.getContext();
  // tensor dialect
  patterns.add<TensorInsertPromotionPattern, TensorExtractPromotionPattern>(
      typeConverter, context);

  // scf dialect
  patterns.add<SCFForOpPromotionPattern, GenericOpPattern<scf::YieldOp>,
               SCFIfOpPromotionPattern>(typeConverter, context);

  // simt dialect
  patterns.add<SIMTExecRegionPromotionPattern,
               GenericOpPattern<simt::BlockYieldOp>>(typeConverter, context);
}

// Modified from the upstream ConvertTritonToTritonGPU, add simt and distributed
// extensions
class ConvertTritonDistributedToTritonGPU
    : public ConvertTritonDistributedToTritonGPUBase<
          ConvertTritonDistributedToTritonGPU> {
public:
  ConvertTritonDistributedToTritonGPU() = default;
  // constructor with some parameters set explicitly.
  ConvertTritonDistributedToTritonGPU(const std::string &target, int numWarps,
                                      int threadsPerWarp, int numCTAs,
                                      bool enableSourceRemat) {
    this->numWarps = numWarps;
    this->threadsPerWarp = threadsPerWarp;
    this->numCTAs = numCTAs;
    this->target = target;
    this->enableSourceRemat = enableSourceRemat;
  }

  void runOnOperation() override {
    if (target.getValue().empty()) {
      mlir::emitError(
          getOperation().getLoc(),
          "'convert-triton-to-tritongpu' requires 'target' option to be set");
      return signalPassFailure();
    }

    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();
    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, enableSourceRemat);
    TritonGPUConversionTarget target(*context, typeConverter);

    // triton distributed extension
    target.addDynamicallyLegalDialect<tensor::TensorDialect,
                                      triton::simt::SIMTDialect,
                                      triton::distributed::DistributedDialect>(
        [&](Operation *op) {
          bool hasLegalRegions = true;
          for (auto &region : op->getRegions()) {
            hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
          }
          if (hasLegalRegions && typeConverter.isLegal(op)) {
            return true;
          }
          return false;
        });

    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    populateArithPatternsAndLegality(typeConverter, patterns, target);
    populateMathPatternsAndLegality(typeConverter, patterns, target);
    populateTritonPatterns(typeConverter, patterns, numCTAs);
    populateProtonPatterns(typeConverter, patterns);
    populateDistributedPatterns(typeConverter, patterns);
    populateTensorPatterns(typeConverter, patterns);
    populateSIMTPatterns(typeConverter, patterns);
    // TODO: can we use
    //    mlir::scf::populateSCFStructurealTypeConversionsAndLegality(...) here?
    populateSCFPatterns(typeConverter, patterns);
    populateCFPatterns(typeConverter, patterns);
    patterns.insert<GenericOpPattern<ub::PoisonOp>>(typeConverter, context);

    auto inti = llvm::APSInt(32, false);

    Builder b(&getContext());
    mod->setAttr(AttrNumWarpsName, b.getI32IntegerAttr(numWarps));
    mod->setAttr(AttrNumThreadsPerWarp, b.getI32IntegerAttr(threadsPerWarp));
    mod->setAttr(AttrNumCTAsName, b.getI32IntegerAttr(numCTAs));
    mod->setAttr(AttrTargetName, b.getStringAttr(this->target.getValue()));

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();

    // update layouts
    //  broadcast src => multicast, dst => broadcasted
    // if (failed(target.refineLayouts(mod, numWarps)))
    //   return signalPassFailure();

    LLVM_DEBUG({ DBGS() << "after TritonGPUConversion = \n" << mod << "\n"; });

    // promote memory space in simt region to ttg.memdesc
    {
      ConversionTarget target(*context);
      target.addLegalDialect<triton::gpu::TritonGPUDialect>();
      target.addLegalOp<UnrealizedConversionCastOp>();

      // Some ops from SCF are illegal
      target.addIllegalOp<scf::ExecuteRegionOp, scf::ParallelOp, scf::ReduceOp,
                          scf::ReduceReturnOp>();

      SIMTRegionTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                            numCTAs);
      target.addDynamicallyLegalDialect<
          arith::ArithDialect, math::MathDialect, triton::TritonDialect,
          cf::ControlFlowDialect, scf::SCFDialect, mlir::gpu::GPUDialect,
          triton::simt::SIMTDialect, triton::distributed::DistributedDialect,
          ub::UBDialect>([&](Operation *op) {
        if (!op->getParentOfType<triton::simt::SIMTExecRegionOp>() &&
            !isa<triton::simt::SIMTExecRegionOp>(op)) {
          return true;
        }
        bool hasLegalRegions = true;
        for (auto &region : op->getRegions()) {
          hasLegalRegions = hasLegalRegions && typeConverter.isLegal(&region);
        }
        if (hasLegalRegions && typeConverter.isLegal(op)) {
          return true;
        }
        return false;
      });

      RewritePatternSet patterns(context);
      populateSIMTReigonPromotionPattern(typeConverter, patterns);
      if (failed(applyPartialConversion(getOperation(), target,
                                        std::move(patterns)))) {
        signalPassFailure();
      }
    }

    LLVM_DEBUG({ DBGS() << "after simt promotion = \n" << mod << "\n"; });

    // clean up: ForOp/IfOp dead arg elimination
    {
      RewritePatternSet cleanUpPatterns(context);
      populateForOpDeadArgumentElimination(cleanUpPatterns);
      scf::ForOp::getCanonicalizationPatterns(cleanUpPatterns, context);
      scf::IfOp::getCanonicalizationPatterns(cleanUpPatterns, context);
      ConvertLayoutOp::getCanonicalizationPatterns(cleanUpPatterns, context);
      if (applyPatternsGreedily(getOperation(), std::move(cleanUpPatterns))
              .failed()) {
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonDistributedToTritonGPUPass(
    const std::string &target, int numWarps, int threadsPerWarp, int numCTAs,
    bool enableSourceRemat) {
  return std::make_unique<::ConvertTritonDistributedToTritonGPU>(
      target, numWarps, threadsPerWarp, numCTAs, enableSourceRemat);
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createConvertTritonDistributedToTritonGPUPass() {
  return std::make_unique<::ConvertTritonDistributedToTritonGPU>();
}
