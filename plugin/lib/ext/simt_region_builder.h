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

/// Builders for the SIMT execution region -- part of the create-op companion
/// module (libtriton_dist_ext).
///
/// Like extern_call_builder.h, these declarations carry NO SIMT dialect header
/// so callers (py_module.cpp) need not pull them in. The definitions
/// (simt_region_builder.cpp) include the dialect headers (DECLARE'd TypeIDs
/// only); the single TypeID DEFINE for SIMTExecRegionOp/BlockYieldOp and the
/// SIMT dialect registration live in the dialect TU compiled into the plugin
/// (libtriton_dist.so), resolved at runtime from the RTLD_GLOBAL plugin.
///
/// A `simt.simt_exec_region` carries a region with block arguments and results,
/// so (unlike the value-in/value-out OpInfo callbacks) the Python frontend must
/// thread loop-carried (liveins) values through it: create the op with init
/// args, bind the entry-block arguments, emit the body, then `block_yield` the
/// updated values and read them back from the op's results. That needs op/block
/// handles, which only a companion builder method (not OpInfo) can return.
#pragma once

#include <vector>

#include "mlir/IR/OpDefinition.h" // mlir::OpState
#include "mlir/IR/Value.h"        // mlir::Value

namespace mlir {
class Block;
} // namespace mlir

class TritonOpBuilder;

namespace triton::dist {

/// Create a `simt.simt_exec_region` whose operands/results/entry-block
/// arguments all mirror `initArgs` (one per loop-carried value). The insertion
/// point is left unchanged; the caller moves it into the entry block.
mlir::OpState createSimtExecRegionOp(TritonOpBuilder &self,
                                     std::vector<mlir::Value> &initArgs);

/// The entry (and only) block of the region carried by a
/// `simt.simt_exec_region`.
mlir::Block *getSimtRegionEntryBlock(mlir::OpState op);

/// Terminate the region with `simt.block_yield`, forwarding `yields` as the
/// region's results (count/types must match the op's results).
mlir::OpState createBlockYieldOp(TritonOpBuilder &self,
                                 std::vector<mlir::Value> &yields);

} // namespace triton::dist
