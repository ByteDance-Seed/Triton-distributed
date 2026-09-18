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

/// Restores the fork's AxisInfo behavior for distributed pointer-preserving
/// ops, out-of-tree, via Triton's AxisInfo extension hook
/// (registerExtraVisitorCallback).
///
/// `distributed.symm_at` (local ptr -> symmetric remote ptr) and
/// `distributed.consume_token` (ordering barrier returning its input) both
/// leave the pointer's alignment/contiguity unchanged. They are converted with
/// a GenericOpPattern and survive (TritonGPU-encoded) into TTGIR, so
/// make_ttgir's Coalesce / load-store vectorization passes encounter them.
/// Without a visitor the built-in AxisInfo analysis treats them as opaque ->
/// pessimistic AxisInfo -> the loads/stores through the produced pointer are
/// NOT vectorized/coalesced (a perf loss; correctness is unaffected since
/// pessimistic is always safe).
///
/// Upstream 3.7.1 offers no way to inject visitors into the built-in passes
/// (they build ModuleAxisInfoAnalysis with no callback), so a tiny op-agnostic
/// hook was added to Triton's AxisInfo (see the 3rdparty/triton patch). This
/// file is the plugin-side consumer: it lives in the plugin .so (alongside the
/// dialect that owns the op TypeIDs) and registers the two visitors at plugin
/// load.
#include "AxisInfoVisitors.h"

#include "triton/Analysis/AxisInfo.h"

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace {

// Result axis info == operand[0]'s axis info (mirrors Triton's
// CastOpAxisInfoVisitor passthrough). operand[0] is the source pointer for both
// `symm_at` (symmAddr) and `consume_token` (input).
template <typename OpTy>
class PassthroughAxisInfoVisitor : public AxisInfoVisitor {
public:
  bool match(Operation *op) override { return isa<OpTy>(op); }

  AxisInfo
  getAxisInfo(Operation *op,
              ArrayRef<const dataflow::Lattice<AxisInfo> *> operands) override {
    return operands[0]->getValue();
  }
};

} // namespace

namespace triton::dist {

void registerDistributedAxisInfoVisitors() {
  axisinfo::registerExtraVisitorCallback([](AxisInfoVisitorList &visitors) {
    visitors.append<PassthroughAxisInfoVisitor<distributed::ConsumeTokenOp>,
                    PassthroughAxisInfoVisitor<distributed::SymmAtOp>>();
  });
}

} // namespace triton::dist
