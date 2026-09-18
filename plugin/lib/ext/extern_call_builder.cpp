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

/// Definition of `distributed.extern_call` construction (create-op companion).
///
/// Compiled into the companion module libtriton_dist_ext. It includes the
/// Distributed dialect headers and calls ExternCallOp::create, but those
/// headers only DECLARE `TypeID<ExternCallOp>`; the single TypeID DEFINE and
/// the op registration live in the dialect TU compiled into the plugin
/// (libtriton_dist.so) and resolve at runtime from the RTLD_GLOBAL plugin. So
/// there is still one TypeID owner -- no cross-library duplication hazard.
#include "extern_call_builder.h"

#include "python/src/ir.h" // TritonOpBuilder (full definition)

#include "TritonDistributed/Dialect/Distributed/IR/Dialect.h"

namespace triton::dist {

mlir::OpState
createExternCallOp(TritonOpBuilder &self, const std::string &libName,
                   const std::string &libPath, const std::string &symbol,
                   std::vector<mlir::Value> &argList,
                   std::vector<mlir::Type> &retTypes, bool isPure) {
  return self.create<::mlir::triton::distributed::ExternCallOp>(
      retTypes, argList, libName, libPath, symbol, isPure);
}

} // namespace triton::dist
