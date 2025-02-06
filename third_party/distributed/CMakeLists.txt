cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

project(DynamicGPUDetection)
#project(Distributed LANGUAGES C CXX CUDA)

# Detect if NVCC (NVIDIA CUDA compiler) is available
execute_process(
    COMMAND nvcc --version
    RESULT_VARIABLE CUDA_FOUND
    OUTPUT_QUIET ERROR_QUIET
)

# Detect if HIPCC (AMD ROCm compiler) is available
execute_process(
    COMMAND hipcc --version
    RESULT_VARIABLE HIP_FOUND
    OUTPUT_QUIET ERROR_QUIET
)

# Set project language dynamically
if (CUDA_FOUND EQUAL 0)
    message(STATUS "CUDA compiler detected, setting project language to CUDA.")
    project(DynamicGPUDetection LANGUAGES C CXX CUDA)
elseif (HIP_FOUND EQUAL 0)
    message(STATUS "HIP compiler detected, setting project language to HIP.")
    project(DynamicGPUDetection LANGUAGES C CXX)
else()
    message(FATAL_ERROR "Neither CUDA nor HIP compiler found. Please install CUDA or ROCm.")
endif()

set(DISTRIBUTED_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/csrc")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============ Add check for includes here (for future) =============

# ============ Dependencies =============
find_package(Python3 REQUIRED COMPONENTS Development.Module Interpreter)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY DISTRIBUTED_LIBS "")

# ============ Define a function to create object libraries ============
function(add_distributed_library name)
  add_library(${name} SHARED ${ARGN})

  target_link_libraries(${name} PUBLIC Python3::Module pybind11::headers)

  target_include_directories(${name}
    PUBLIC
      "${DISTRIBUTED_SRC_DIR}/include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "DISTRIBUTED_LIBS"
  set_property(GLOBAL APPEND PROPERTY DISTRIBUTED_LIBS ${name})
endfunction()

string(TOLOWER "${USE_TRITON_DISTRIBUTED_AOT}" USE_TRITON_DISTRIBUTED_AOT_LOWER)
if(USE_TRITON_DISTRIBUTED_AOT_LOWER STREQUAL "on" OR USE_TRITON_DISTRIBUTED_AOT_LOWER STREQUAL "1" OR USE_TRITON_DISTRIBUTED_AOT_LOWER STREQUAL "true")
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import triton; from packaging.version import Version; assert Version(triton.__version__) >= Version('3.0.0')"
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    COMMAND_ERROR_IS_FATAL ANY
  )
  # set(TRITON_AOT_GENERATED "triton_aot_generated")
  # file(REMOVE_RECURSE "${CMAKE_CURRENT_SOURCE_DIR}/csrc/${TRITON_AOT_GENERATED}")
  # set(FLASH_DECODE_KERNEL ${PROJECT_SOURCE_DIR}/distributed/kernels/nvidia/flash_decode.py:kernel_gqa_fwd_batch_decode_split_kv_persistent ${PROJECT_SOURCE_DIR}/distributed/kernels/nvidia/flash_decode.py:kernel_gqa_fwd_batch_decode_split_kv ${PROJECT_SOURCE_DIR}/distributed/kernels/nvidia/flash_decode.py:kernel_gqa_fwd_batch_decode_combine_kv)
  # execute_process(
  #   COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/distributed/tools/compile_aot.py --workspace ${CMAKE_CURRENT_SOURCE_DIR}/csrc/${TRITON_AOT_GENERATED} --kernels ${FLASH_DECODE_KERNEL}  --library triton_distributed_kernel
  #   WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
  #   COMMAND_ERROR_IS_FATAL ANY
  # )
  add_definitions(-DUSE_TRITON_DISTRIBUTED_AOT)
endif()

# ============ Add subdirectory with actual code that calls add_distributed_library ============
if (CUDA_FOUND EQUAL 0)
  add_subdirectory("${DISTRIBUTED_SRC_DIR}")
endif()

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_distributed_obj_libs GLOBAL PROPERTY DISTRIBUTED_LIBS)

if(NOT _distributed_obj_libs)
  message(WARNING "No object libraries were defined in 'DISTRIBUTED_LIBS'!")
endif()
