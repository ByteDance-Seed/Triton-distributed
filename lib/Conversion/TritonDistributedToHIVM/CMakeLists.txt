
add_triton_library(TritonDistributedToHIVM
    ASCEND/DistributedOpToHIVM.cpp
    ASCEND/ConvertTritonDistributedToHIVM.cpp

    DEPENDS
    TritonDistributedToHIVMConversionPassIncGen

    LINK_LIBS PUBLIC
    DistributedIR
    BiShengIRHIVMDialect
)
