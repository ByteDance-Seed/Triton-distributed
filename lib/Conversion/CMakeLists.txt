if(NOT TRITON_USE_ASCEND)
  add_subdirectory(TritonDistributedToLLVM)
  add_subdirectory(TritonDistributedToTritonGPU)
endif()

if(TRITON_USE_ASCEND)
  add_subdirectory(TritonDistributedToHIVM)
endif()
