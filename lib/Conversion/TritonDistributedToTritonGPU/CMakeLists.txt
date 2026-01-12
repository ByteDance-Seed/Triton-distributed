if(TRITON_USE_MACA)
    add_triton_library(TritonDistributedToTritonGPU
        METAX/TritonDistributedToTritonGPU.cpp

        DEPENDS
        TritonDistributedConversionPassIncGen
        TritonConversionPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        TritonIR
        DistributedIR
        TritonGPUIR
        TritonGPUTransforms
    )
else()
    add_triton_library(TritonDistributedToTritonGPU
        TritonDistributedToTritonGPU.cpp

        DEPENDS
        TritonDistributedConversionPassIncGen
        TritonConversionPassIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRPass
        MLIRTransforms
        TritonIR
        ProtonIR
        DistributedIR
        SIMTIR
        TritonGPUIR
        TritonGPUTransforms
    )
endif()
