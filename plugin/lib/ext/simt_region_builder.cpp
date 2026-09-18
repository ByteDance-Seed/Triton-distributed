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

/// Definitions of the SIMT execution-region builders (create-op companion).
///
/// Compiled into the companion module libtriton_dist_ext. It includes the SIMT
/// dialect headers and calls SIMTExecRegionOp/BlockYieldOp::create, but those
/// headers only DECLARE their `TypeID`s; the single TypeID DEFINE and the SIMT
/// dialect registration live in the dialect TU compiled into the plugin
/// (libtriton_dist.so), resolved at runtime from the RTLD_GLOBAL plugin -- so
/// there is exactly one TypeID owner.
#include "simt_region_builder.h"

#include "python/src/ir.h" // TritonOpBuilder (full definition)

#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"

#include "TritonDistributed/Dialect/SIMT/IR/Dialect.h"

namespace triton::dist {

namespace simt = ::mlir::triton::simt;

mlir::OpState createSimtExecRegionOp(TritonOpBuilder &self,
                                     std::vector<mlir::Value> &initArgs) {
  return self.create<simt::SIMTExecRegionOp>(initArgs);
}

mlir::Block *getSimtRegionEntryBlock(mlir::OpState op) {
  return &mlir::cast<simt::SIMTExecRegionOp>(op.getOperation())
              .getDefaultRegion()
              .front();
}

mlir::OpState createBlockYieldOp(TritonOpBuilder &self,
                                 std::vector<mlir::Value> &yields) {
  return self.create<simt::BlockYieldOp>(yields);
}

} // namespace triton::dist
