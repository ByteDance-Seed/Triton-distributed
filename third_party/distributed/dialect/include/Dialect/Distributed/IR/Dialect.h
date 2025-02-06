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
#ifndef TRITON_DIALECT_DISTRIBUTED_IR_DIALECT_H_
#define TRITON_DIALECT_DISTRIBUTED_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Traits.h"

// clang-format off
#include "distributed/dialect/include/Dialect/Distributed/IR/Dialect.h.inc"
#include "distributed/dialect/include/Dialect/Distributed/IR/DistributedEnums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "distributed/dialect/include/Dialect/Distributed/IR/DistributedAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "distributed/dialect/include/Dialect/Distributed/IR/Ops.h.inc"

namespace mlir {
namespace triton {
namespace distributed {} // namespace distributed
} // namespace triton
} // namespace mlir

#endif // TRITON_DIALECT_DISTRIBUTED_IR_DIALECT_H_
