/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include "TritonDistributed/Conversion/TritonDistributedToHIVM/TritonDistributedToHIVMPass.h"
#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"
#include "ascend/include/Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include <string>
#include <type_traits>

using namespace mlir;
using namespace mlir::triton;

namespace {

/// Generic template pattern to convert distributed ops to hivm.custom
/// The callee name is derived from the distributed op name with a prefix
template <typename DistOp>
class DistributedOpToHIVM : public OpRewritePattern<DistOp> {
public:
  using OpAdaptor = typename DistOp::Adaptor;

  DistributedOpToHIVM(MLIRContext *context, StringRef customPrefix = "",
                      PatternBenefit benefit = PatternBenefit(1))
      : OpRewritePattern<DistOp>(context, benefit), customPrefix(customPrefix) {
  }

  void inferCoreType(DistOp op) const {
    if constexpr (std::is_same_v<DistOp, distributed::ExternCallOp>) {
      auto externCallOp = llvm::cast<distributed::ExternCallOp>(op);
      if (llvm::is_contained(ASCEND::mixTypeList, externCallOp.getSymbol())) {
        op->setAttr(hivm::TCoreTypeAttr::name,
                    hivm::TCoreTypeAttr::get(op->getContext(),
                                             hivm::TCoreType::CUBE_AND_VECTOR));
        return;
      }
      if (llvm::is_contained(ASCEND::vecTypeList, externCallOp.getSymbol())) {
        op->setAttr(hivm::TCoreTypeAttr::name,
                    hivm::TCoreTypeAttr::get(op->getContext(),
                                             hivm::TCoreType::VECTOR));
        return;
      }
    }
    op->setAttr(hivm::TCoreTypeAttr::name,
                hivm::TCoreTypeAttr::get(op->getContext(),
                                         hivm::TCoreType::CUBE_OR_VECTOR));
  }

  std::string getTypeName(Type type) const {
    if (auto fpTy = llvm::dyn_cast<FloatType>(type)) {
      if (fpTy.isBF16()) {
        return "bfloat16";
      } else if (fpTy.isF16()) {
        return "half";
      } else if (fpTy.isF32()) {
        return "float";
      }
    }
    if (auto intTy = llvm::dyn_cast<IntegerType>(type)) {
      switch (intTy.getWidth()) {
      case 8:
        return "int8";
      case 16:
        return "int16";
      case 32:
        return "int32";
      case 64:
        return "int64";
      }
    }
    assert(false && "unsupport type");
  }

  LogicalResult matchAndRewrite(DistOp op,
                                PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    inferCoreType(op);
    std::string symbolName;
    if (auto sym = op->template getAttrOfType<StringAttr>("symbol")) {
      symbolName = sym.str();
    }

    llvm::TypeSwitch<Operation *, void>(op)
        .Case([&](distributed::SymmAtOp) {
          auto elemTy = llvm::cast<triton::PointerType>(op->getOperand(0).getType())
                            .getPointeeType();
          symbolName = "aclshmem_" + getTypeName(elemTy) + "_ptr";
        })
        .Case([&](distributed::GetRankOp) {
          symbolName = "aclshmem_my_pe";
        })
        .Case([&](distributed::GetNumRanksOp) {
          symbolName = "aclshmem_n_pes";
        })
        .Case([&](distributed::NotifyOp notifyOp) {
          auto typeName = getTypeName(notifyOp.getSignalVal().getType());
          symbolName = "aclshmem_" + typeName + "_p";
        });

    if (symbolName == "aclshmem_ptr") {
      auto elemTy = llvm::cast<triton::PointerType>(op->getOperand(0).getType())
                        .getPointeeType();
      symbolName = "aclshmem_" + getTypeName(elemTy) + "_ptr";
    }

    if (symbolName.empty()) {
      symbolName = op->getName().getStringRef().drop_front(
          ASCEND::distributedDialectPrefixLen);
    }
    std::string customName = customPrefix + "." + symbolName;
    auto operands = op->getOperands();
    llvm::SmallVector<Value> customResults;
    for (auto res : op->getResults()) {
      if (auto tensorTy = llvm::dyn_cast<RankedTensorType>(res.getType())) {
        auto emptyOp = rewriter.create<tensor::EmptyOp>(
            op->getLoc(), tensorTy.getShape(), tensorTy.getElementType());
        customResults.emplace_back(emptyOp);
      }
    }
    auto customOp =
        rewriter.create<hivm::CustomOp>(loc, op->getResultTypes(), customName,
                                        operands, customResults, ValueRange{});
    customOp->setAttrs(op->getAttrs());
    customOp->setAttr("hivm.is_distributed", rewriter.getUnitAttr());
    auto pipePair = ASCEND::pipeMap.find(customName);
    if (pipePair != ASCEND::pipeMap.end()) {
      customOp.setPipe(pipePair->getSecond());
    } else {
      customOp.setPipe(hivm::PIPE::PIPE_S);
    }
    customOp.setVFMode(hivm::VFMode::SIMD);
    customOp->setAttr("symbol", rewriter.getStringAttr(symbolName));
    if (llvm::isa<distributed::ConsumeTokenOp>(op) &&
        llvm::isa<RankedTensorType>(customOp->getResult(0).getType())) {
      auto annotationOp = rewriter.create<annotation::MarkOp>(
          op->getLoc(), customOp->getResult(0));
      annotationOp->setAttr(ConverterUtils::continuousAttrName,
                            rewriter.getUnitAttr());
      customOp->setAttr(ConverterUtils::customSrcPtrIndexAttrName,
                        rewriter.getDenseI32ArrayAttr({0}));
    }
    // Replace the original op with the custom op
    if (op->getNumResults() == 0) {
      rewriter.eraseOp(op);
    } else {
      rewriter.replaceOp(op, customOp);
    }

    return success();
  }

private:
  std::string customPrefix;
};

/// Helper function to register distributed op to hivm.custom patterns
template <typename... Args>
void registerDistributedOpToHIVM(RewritePatternSet &patterns,
                                 StringRef calleePrefix = "",
                                 PatternBenefit benefit = PatternBenefit(1)) {
  patterns.add<DistributedOpToHIVM<Args>...>(patterns.getContext(),
                                             calleePrefix, benefit);
}

} // namespace

/// Populate the pattern set with distributed op to hivm.custom patterns
void mlir::triton::ASCEND::populateDistributedOpToHIVMPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  // Use the template function to register all distributed ops
  // All ops convert to hivm.custom with the name "dist.<op_name>"
  registerDistributedOpToHIVM<distributed::GetRankOp,
                              distributed::GetNumRanksOp, distributed::SymmAtOp,
                              distributed::WaitOp, distributed::ConsumeTokenOp,
                              distributed::NotifyOp, distributed::ExternCallOp>(
      patterns, "dist", benefit);
}
