//
// Modification Copyright 2026 Huawei Technologies Ltd. and/or its affiliates.
//
#ifndef TRITONDISTRIBUTED_HIVM_CONVERSION_PASSES
#define TRITONDISTRIBUTED_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

// ASCEND Backend
def ConvertTritonDistributedToHIVM : Pass<"convert-triton-distributed-to-hivm", "mlir::ModuleOp"> {
    let summary = "Convert TritonDistributed to HIVM";
    let description = [{
        This pass converts TritonDistributed Dialect operations to hivm.custom operations.
    }];
    let constructor = "mlir::triton::ASCEND::createConvertTritonDistributedToHIVMPass()";

    let dependentDialects = ["hivm::HIVMDialect",
                             "triton::distributed::DistributedDialect"];

    let options = [
        Option<"patternBenefitPrioritizeOverLLVMConversions", "pattern-benefit-prioritize-over-llvm-conversions",
               "int", /*default*/"10",
               "pattern benefit to prioritize over LLVM conversions">,
    ];
}
#endif // TRITONDISTRIBUTED_HIVM_CONVERSION_PASSES
