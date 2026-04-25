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
#ifndef TRITON_DISTRIBUTED_CONVERSION_TRITONDISTRIBUTEDTOHIVM_H
#define TRITON_DISTRIBUTED_CONVERSION_TRITONDISTRIBUTEDTOHIVM_H

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/PatternMatch.h"
#include <memory>

namespace mlir {
class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

namespace ASCEND {
static const llvm::SmallVector<std::string> mixTypeList = {
    "aclshmem_barrier_all", "aclshmem_barrier"};
static const llvm::SmallVector<std::string> vecTypeList = {
    "aclshmemx_barrier_all_vec"};
static const llvm::DenseMap<llvm::StringRef, hivm::PIPE> pipeMap;
static const int distributedDialectPrefixLen = 12;
void populateDistributedOpToHIVMPatterns(RewritePatternSet &patterns,
                                         PatternBenefit benefit);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToHIVMPass();
} // namespace ASCEND

} // namespace triton
} // namespace mlir

#endif
