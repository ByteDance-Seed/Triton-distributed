if(TRITON_USE_MACA)
    add_triton_library(TritonDistributedToLLVM
        METAX/DistributedOpToLLVM.cpp
        METAX/ConvertMETAXDistributedToLLVM.cpp

        DEPENDS
        TritonDistributedToLLVMConversionPassIncGen

        LINK_LIBS PUBLIC
        TritonGPUToLLVM
        DistributedIR
        TritonMETAXGPUToLLVM
        TritonMETAXGPUTransforms
    )
else()
    add_triton_library(TritonDistributedToLLVM
        NVIDIA/ConvertTritonDistributedToLLVM.cpp
        NVIDIA/DistributedOpToLLVM.cpp
        SIMTOpToLLVM.cpp
        AMD/LibDeviceToLLVM.cpp
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
endif()
