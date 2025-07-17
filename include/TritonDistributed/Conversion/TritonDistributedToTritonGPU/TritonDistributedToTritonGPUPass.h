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
#ifndef TRITON_CONVERSION_TRITONDISTRIBUTEDTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONDISTRIBUTEDTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>
#include <optional>
#include <string>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToTritonGPUPass(const std::string &target,
                                              int numWarps,
                                              int threadsPerWarp = 32,
                                              int numCTAs = 1,
                                              bool enableSourceRemat = false);

} // namespace triton
} // namespace mlir

#endif
