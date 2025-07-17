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
