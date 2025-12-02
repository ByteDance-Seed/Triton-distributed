add_triton_library(TritonDistributedToLLVM
    NVIDIA/ConvertTritonDistributedToLLVM.cpp
    NVIDIA/DistributedOpToLLVM.cpp
    SIMTOpToLLVM.cpp
    AMD/BuiltinFuncToLLVMExt.cpp
    AMD/DistributedOpToLLVM.cpp
    AMD/ConvertAMDDistributedToLLVM.cpp

    DEPENDS
    TritonDistributedToLLVMConversionPassIncGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    TritonGPUToLLVM
    TritonProtonToLLVM
    DistributedIR
    SIMTIR
    TritonAMDGPUIR
    TritonAMDGPUToLLVM
    TritonAMDGPUDialectToLLVM
    TritonAMDGPUTransforms
    TritonAMDUtils
    TritonAMDAnalysis
)
