if(TRITON_USE_MACA)
    add_triton_library(TritonDistributedToTritonGPU
        METAX/TritonDistributedToTritonGPU.cpp

        # Also #includes the Distributed/SIMT dialect TableGen'd headers; depend on
        # their IncGen targets so a parallel build cannot compile before the .inc
        # files exist (see TritonDistributedToLLVM/CMakeLists.txt for details).
        DEPENDS
        TritonDistributedConversionPassIncGen
        TritonConversionPassIncGen
        DistributedTableGen
        DistributedAttrDefsIncGen
        SIMTTableGen
        SIMTAttrDefsIncGen

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
else()
    add_triton_library(TritonDistributedToTritonGPU
        TritonDistributedToTritonGPU.cpp

        # Also #includes the Distributed/SIMT dialect TableGen'd headers; depend on
        # their IncGen targets so a parallel build cannot compile before the .inc
        # files exist (see TritonDistributedToLLVM/CMakeLists.txt for details).
        DEPENDS
        TritonDistributedConversionPassIncGen
        TritonConversionPassIncGen
        DistributedTableGen
        DistributedAttrDefsIncGen
        SIMTTableGen
        SIMTAttrDefsIncGen

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
