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
#ifndef TRITON_DISTRIBUTED_CONVERSION_TRITONDISTRIBUTEDTOLLVM_H
#define TRITON_DISTRIBUTED_CONVERSION_TRITONDISTRIBUTEDTOLLVM_H

#include <memory>
#include <optional>
#include <string>

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "third_party/amd/lib/TritonAMDGPUToLLVM/TargetInfo.h"
#include "third_party/nvidia/lib/TritonNVIDIAGPUToLLVM/TargetInfo.h"
#ifdef USE_MACA
#include "third_party/metax/lib/TritonMETAXGPUToLLVM/TargetInfo.h"
#endif
#include "triton/Conversion/MLIRTypes.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

namespace NVIDIA {
void populateDistributedOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit,
                                         const TargetInfo &targetInfo,
                                         std::string NVSHMEMLibname = "",
                                         std::string NVSHMEMLibpath = "");
} // namespace NVIDIA

namespace AMD {
void populateDistributedOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit,
                                         const TargetInfo &targetInfo,
                                         std::string ROCSHMEMLibname = "",
                                         std::string ROCSHMEMLibpath = "");

void populateROCDLBlockDimPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);
} // namespace AMD

void populateSIMTOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  const TargetInfoBase &targetInfo,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToLLVMPass(int32_t computeCapability);
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonDistributedToLLVMPass(int32_t computeCapability,
                                         int32_t ptxVersion);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertBuiltinFuncToLLVMExtPass(bool ftz);

std::unique_ptr<OperationPass<ModuleOp>>
createConvertAMDDistributedToLLVMPass(StringRef targetArch, bool ftz);

#ifdef USE_MACA
namespace METAX {
void populateDistributedOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit,
                                         const TargetInfo &targetInfo,
                                         std::string MXSHMEMLibname = "",
                                         std::string MXSHMEMLibpath = "");
} // namespace METAX

std::unique_ptr<OperationPass<ModuleOp>>
createConvertMETAXDistributedToLLVMPass(int32_t computeCapability);
#endif // USE_MACA

} // namespace triton
} // namespace mlir

#endif
