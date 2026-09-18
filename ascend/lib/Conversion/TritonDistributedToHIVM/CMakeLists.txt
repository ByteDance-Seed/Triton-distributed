
add_triton_library(TritonDistributedToHIVM
    ASCEND/DistributedOpToHIVM.cpp
    ASCEND/ConvertTritonDistributedToHIVM.cpp

    # Also #includes the Distributed dialect TableGen'd headers; depend on their
    # IncGen targets so a parallel build cannot compile before the .inc files exist
    # (see TritonDistributedToLLVM/CMakeLists.txt for details).
    DEPENDS
    TritonDistributedToHIVMConversionPassIncGen
    DistributedTableGen
    DistributedAttrDefsIncGen

    LINK_LIBS PUBLIC
    DistributedIR
    BiShengIRHIVMDialect
)
