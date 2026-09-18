if(TRITON_USE_MACA)
    add_triton_library(TritonDistributedToLLVM
        METAX/DistributedOpToLLVM.cpp
        METAX/ConvertMETAXDistributedToLLVM.cpp

        # These sources #include the Distributed dialect's TableGen'd headers
        # (Dialect.h -> DistributedEnums.h.inc, ...). Linking DistributedIR only
        # orders the *link*, not the *compile* of these objects, so under a
        # parallel build the .cpp can compile before the .inc is generated
        # (fatal error: ... .h.inc: No such file or directory). Depend on the
        # IncGen targets directly so the headers always exist first.
        DEPENDS
        TritonDistributedToLLVMConversionPassIncGen
        DistributedTableGen
        DistributedAttrDefsIncGen

        LINK_LIBS PUBLIC
        TritonGPUToLLVM
        DistributedIR
        TritonMETAXGPUToLLVM
        TritonMETAXGPUTransforms
    )
else()
    # NVIDIA + AMD + SIMT lowering, always built together (mirrors the old
    # monolithic build and libtriton.so, which always ships both backends). The
    # TritonAMDGPU* link libs below are dropped by add_triton_library and resolved
    # from libtriton.so at load time.
    add_triton_library(TritonDistributedToLLVM
        NVIDIA/ConvertTritonDistributedToLLVM.cpp
        NVIDIA/DistributedOpToLLVM.cpp
        SIMTOpToLLVM.cpp
        AMD/BuiltinFuncToLLVMExt.cpp
        AMD/DistributedOpToLLVM.cpp
        AMD/ConvertAMDDistributedToLLVM.cpp
        AMD/ROCDLBlockDimPattern.cpp

        # These sources #include the Distributed and SIMT dialects' TableGen'd
        # headers (Dialect.h -> DistributedEnums.h.inc / SIMT Dialect.h.inc, ...).
        # Linking DistributedIR/SIMTIR only orders the *link*, not the *compile*
        # of these objects, so under a parallel build (-j) the .cpp can compile
        # before the .inc is generated (fatal error: ... .h.inc: No such file or
        # directory -- observed on a clean nvcr build). Depend on the IncGen
        # targets directly so the generated headers always exist first.
        DEPENDS
        TritonDistributedToLLVMConversionPassIncGen
        NVGPUAttrDefsIncGen
        DistributedTableGen
        DistributedAttrDefsIncGen
        SIMTTableGen
        SIMTAttrDefsIncGen

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
