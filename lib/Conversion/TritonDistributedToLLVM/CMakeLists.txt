set(TRITON_SOURCES
    NVIDIA/ConvertTritonDistributedToLLVM.cpp
    NVIDIA/DistributedOpToLLVM.cpp
    SIMTOpToLLVM.cpp
    AMD/BuiltinFuncToLLVMExt.cpp
    AMD/DistributedOpToLLVM.cpp
    AMD/ConvertAMDDistributedToLLVM.cpp
    AMD/ROCDLBlockDimPattern.cpp
)
set(TRITON_LINK_LIBS
    TritonGPUToLLVM
    DistributedIR
    SIMTIR
    TritonAMDGPUIR
    TritonAMDGPUToLLVM
    TritonAMDGPUDialectToLLVM
    TritonAMDGPUTransforms
    TritonAMDUtils
    TritonAMDAnalysis
)

if(TRITON_USE_MACA)
    add_triton_library(TritonDistributedToLLVM
        ${TRITON_SOURCES} 
        METAX/DistributedOpToLLVM.cpp
        METAX/ConvertMETAXDistributedToLLVM.cpp

        DEPENDS
        TritonDistributedToLLVMConversionPassIncGen
        NVGPUAttrDefsIncGen

        LINK_LIBS PUBLIC
        ${TRITON_LINK_LIBS}
        TritonMETAXGPUToLLVM
        TritonMETAXGPUTransforms
    )
else()
    add_triton_library(TritonDistributedToLLVM
        ${TRITON_SOURCES}

        DEPENDS
        TritonDistributedToLLVMConversionPassIncGen
        NVGPUAttrDefsIncGen

        LINK_LIBS PUBLIC
        ${TRITON_LINK_LIBS}
        TritonProtonToLLVM
    )
endif()
