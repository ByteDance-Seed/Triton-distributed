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

/// Restores the fork's software-pipeline predication of `distributed.wait`,
/// out-of-tree, via Triton's pipeliner extension hook
/// (registerExtraPredicateOpFn).
///
/// The software pipeliner predicates the side-effecting ops it relocates into a
/// loop's prologue/epilogue (and across the steady-state boundary) by calling
/// triton::predicateOp. That routine handles the built-in ops through a
/// hard-coded dyn_cast chain and report_fatal_error()s on any *other*
/// registered op. Because `distributed.wait` carries a Read memory effect it is
/// NOT skipped as effect-free, so an un-predicated `distributed.wait` inside a
/// pipelined loop would otherwise crash compilation.
///
/// `distributed.wait` already lowers an optional `pred` operand (false => skip
/// the spin-wait entirely, see NVIDIA/DistributedOpToLLVM.cpp), which is
/// exactly the behavior the pipeliner needs at a masked-off iteration. This
/// consumer registers a predicator that sets/ANDs that operand, mirroring
/// upstream's ttng::WaitBarrierOp handling in predicateOp. It is the 3.7.1
/// equivalent of the fork's intrusive `predicateOp` branch (and of 3.8's
/// PredicatedOpInterface): the op stays in the distributed dialect and the only
/// triton-side change is one op-agnostic registration hook (see the
/// 3rdparty/triton patch).
#include "PipelinerPredication.h"

#include "triton/Dialect/Triton/IR/Utility.h" // getPredMask
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace triton::dist {

void registerDistributedPipelinerPredication() {
  registerExtraPredicateOpFn([](RewriterBase &rewriter, Operation *op,
                                Value pred) -> Operation * {
    auto waitOp = dyn_cast<distributed::WaitOp>(op);
    if (!waitOp)
      return nullptr; // decline; let the next predicator (or the error) run
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPoint(waitOp);
    // AND the incoming stage predicate with any predicate already on the op.
    Value mask = pred;
    if (Value currentPred = waitOp.getPred())
      mask = getPredMask(rewriter, currentPred.getType(), currentPred, pred);
    waitOp.getPredMutable().assign(mask);
    return op;
  });
}

} // namespace triton::dist
