//
// Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
//
#ifndef TRITONDISTRIBUTED_LLVM_CONVERSION_PASSES
#define TRITONDISTRIBUTED_LLVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// NVIDIA Backend
def ConvertTritonDistributedToLLVM : Pass<"convert-triton-distributed-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonDistributed to LLVM";
    let description = [{
        This pass converts the TritonDistributed Dialect into LLVM Dialect.
    }];
    let constructor = "mlir::triton::createConvertTritonDistributedToLLVMPass()";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect",
                             "triton::simt::SIMTDialect",
                             "triton::distributed::DistributedDialect",
                             "mlir::triton::nvgpu::NVGPUDialect",
                             "mlir::NVVM::NVVMDialect"];

    let options = [
        Option<"computeCapability", "compute-capability",
               "int32_t", /*default*/"80",
               "device compute capability">,
        Option<"ptxVersion", "ptx-version",
               "int32_t", /*default*/"80",
               "PTX version">,
    ];
}


// AMD Backend
def ConvertLibDeviceToLLVM : Pass<"convert-lib-device-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert Lib Device (User Defined Ops) to LLVM";
    let constructor = "mlir::triton::createConvertLibDeviceToLLVMPass(/*ftz=*/true)";

    let dependentDialects = ["mlir::LLVM::LLVMDialect"];

    let options = [
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

def ConvertAMDDistributedToLLVM : Pass<"convert-amd-distributed-to-llvm", "mlir::ModuleOp"> {
    let summary = "Convert TritonDistributed to LLVM";
    let description = [{
        The `convert-amd-distributed-to-llvm` pass performs lowering of
        the distributed op for AMD backend.
    }];

    let constructor = "mlir::triton::createConvertAMDDistributedToLLVMPass(\"\", /*ftz=*/true)";

    let dependentDialects = ["mlir::arith::ArithDialect",
                             "mlir::math::MathDialect",
                             "mlir::gpu::GPUDialect",
                             "mlir::scf::SCFDialect",
                             "mlir::LLVM::LLVMDialect",
                             "mlir::triton::TritonDialect",
                             "mlir::triton::gpu::TritonGPUDialect",
                             "mlir::ROCDL::ROCDLDialect",
                             "mlir::cf::ControlFlowDialect",
                             "triton::distributed::DistributedDialect"];

    let options = [
        Option<"arch", "arch", "std::string", /*default*/"\"\"",
               "gfx target device architecture, e.g., gfx942">,
        Option<"ftz", "ftz", "bool", /*default*/"true",
               "flush denorms for math functions">,
    ];
}

#endif // TRITONDISTRIBUTED_LLVM_CONVERSION_PASSES
