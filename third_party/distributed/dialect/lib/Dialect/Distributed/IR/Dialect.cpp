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

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/Distributed/IR/Dialect.h"
#include "Dialect/Distributed/IR/Dialect.cpp.inc"
// clang-format on

using namespace mlir;
using namespace mlir::triton::distributed;

void mlir::triton::distributed::DistributedDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/Distributed/IR/DistributedAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/Distributed/IR/Ops.cpp.inc"
      >();
}

#include "Dialect/Distributed/IR/DistributedEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/Distributed/IR/DistributedAttrDefs.cpp.inc"
