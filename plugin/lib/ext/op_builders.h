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

/// Distributed/SIMT op builders -- part of the create-op companion module
/// (libtriton_dist_ext), not the plugin .so.
///
/// Each builder follows the upstream Triton plugin-op convention used by
/// triton-ext/utlx: it receives a `std::vector<mlir::Value>` where
/// `operands[0]` is the (initially empty) result slot and `operands[1..]` are
/// the caller's arguments. Attribute enums / scalar metadata are encoded as
/// `arith.constant` integer Values.
///
/// On the pinned Triton 3.7.1 these are NOT registered as OpInfo callbacks (its
/// plugin-op binding inserts no result slot and returns nothing); instead the
/// pybind layer (py_module.cpp) prepends the slot, calls these, and returns
/// `operands[0]`. These TUs include only the dialect *headers* (DECLARE'd
/// TypeIDs); the single TypeID DEFINE lives in the dialect TU compiled into the
/// plugin .so, which this module resolves at runtime via the RTLD_GLOBAL
/// plugin.
#ifndef TRITON_DIST_EXT_OP_BUILDERS_H
#define TRITON_DIST_EXT_OP_BUILDERS_H

#include "python/src/ir.h"
#include <vector>

namespace triton::dist::ops {

using ::TritonOpBuilder;
using mlir::Value;

void createDistributedWait(TritonOpBuilder &self, std::vector<Value> &operands);
void createConsumeToken(TritonOpBuilder &self, std::vector<Value> &operands);
void createGetRank(TritonOpBuilder &self, std::vector<Value> &operands);
void createGetNumRanks(TritonOpBuilder &self, std::vector<Value> &operands);
void createSymmAt(TritonOpBuilder &self, std::vector<Value> &operands);
void createNotify(TritonOpBuilder &self, std::vector<Value> &operands);

// SIMT / shared-memory / tensor ops (value-only; result types derived from
// operand types, so no separate type channel is required).
void createExtract(TritonOpBuilder &self, std::vector<Value> &operands);
void createInsert(TritonOpBuilder &self, std::vector<Value> &operands);
void createLoadShared(TritonOpBuilder &self, std::vector<Value> &operands);
void createStoreShared(TritonOpBuilder &self, std::vector<Value> &operands);
void createMemDescToPtr(TritonOpBuilder &self, std::vector<Value> &operands);
void createLocalDealloc(TritonOpBuilder &self, std::vector<Value> &operands);
void createLocalAlloc(TritonOpBuilder &self, std::vector<Value> &operands);
void createMemDescSubview(TritonOpBuilder &self, std::vector<Value> &operands);

// SIMT thread/block intrinsics (value-producing). The region op itself is built
// by simt_region_builder.cpp (it returns op/block handles).
void createGetThreadId(TritonOpBuilder &self, std::vector<Value> &operands);
void createGetBlockSize(TritonOpBuilder &self, std::vector<Value> &operands);

} // namespace triton::dist::ops

#endif // TRITON_DIST_EXT_OP_BUILDERS_H
