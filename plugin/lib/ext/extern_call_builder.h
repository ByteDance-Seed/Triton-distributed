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

/// Builder for `distributed.extern_call` -- part of the create-op companion
/// module (libtriton_dist_ext).
///
/// This *declaration* is deliberately free of any Triton-distributed dialect
/// header so callers (e.g. py_module.cpp) need not pull them in. The definition
/// (extern_call_builder.cpp) includes the dialect headers, whose op `TypeID`s
/// are only DECLARE'd there; the single TypeID DEFINE lives in the dialect TU
/// compiled into the plugin `libtriton_dist.so`, resolved at runtime from the
/// RTLD_GLOBAL plugin. So the dialect/op registration still has exactly one
/// owner -- no second `TypeID<ExternCallOp>` is instantiated here.
#pragma once

#include <string>
#include <vector>

#include "mlir/IR/OpDefinition.h" // mlir::OpState
#include "mlir/IR/Types.h"        // mlir::Type
#include "mlir/IR/Value.h"        // mlir::Value

class TritonOpBuilder;

namespace triton::dist {

/// Build a variadic-result `distributed.extern_call` carrying the lib/symbol
/// string attributes. Returns the OpState so the Python frontend can read
/// get_result(i) (0 results => void call, 1 => scalar return; >1 rejected by
/// the op's lowering).
mlir::OpState
createExternCallOp(TritonOpBuilder &self, const std::string &libName,
                   const std::string &libPath, const std::string &symbol,
                   std::vector<mlir::Value> &argList,
                   std::vector<mlir::Type> &retTypes, bool isPure);

} // namespace triton::dist
