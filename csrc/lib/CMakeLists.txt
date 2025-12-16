cmake_minimum_required(VERSION 3.12)

find_package(
  Python3
  COMPONENTS Interpreter Development.Module
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

find_package(pybind11 REQUIRED HINTS "${Python3_SITELIB}")

file(GLOB LIB_FILES "*.cc")
message(STATUS "lib files: ${LIB_FILES}")
file(GLOB TRITON_AOT_PYBIND_FILES "../triton_aot_generated/pybind/*.cc")
message(STATUS "triton AoT pybind files: ${TRITON_AOT_PYBIND_FILES}")

add_library(triton_distributed SHARED ${LIB_FILES} ${TRITON_AOT_PYBIND_FILES})
target_link_libraries(triton_distributed PUBLIC pybind11::headers Python3::Module)

#
target_include_directories(triton_distributed PUBLIC ".")
target_include_directories(triton_distributed PUBLIC "../triton_aot_generated")

set(NVSHMEM_HOME "")
function(set_nvshmem_home)
  # 1. Check if NVSHMEM_HOME environment variable is set
  if(DEFINED ENV{NVSHMEM_HOME})
      set(NVSHMEM_HOME "$ENV{NVSHMEM_HOME}" PARENT_SCOPE)
      message(STATUS "Found NVSHMEM_HOME from environment variable: $ENV{NVSHMEM_HOME}")
      return()
  endif()

  # 2. Try to find from Python command
  find_program(PYTHON_EXECUTABLE python3)
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} -c
              "import nvidia.nvshmem, pathlib; print(pathlib.Path(nvidia.nvshmem.__path__[0]))"
      OUTPUT_VARIABLE PYTHON_NVSHMEM_PATH
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  if(PYTHON_NVSHMEM_PATH AND NOT PYTHON_NVSHMEM_PATH STREQUAL "")
      set(NVSHMEM_HOME "${PYTHON_NVSHMEM_PATH}" PARENT_SCOPE)
      message(STATUS "Found NVSHMEM_HOME from Python nvidia-nvshmem-cu12: ${PYTHON_NVSHMEM_PATH}")
      return()
  endif()
  # not found
  message(WARNING "NVSHMEM_HOME could not be determined.")

endfunction()


if(USE_TRITON_DISTRIBUTED_AOT)
  set_nvshmem_home()
  find_library(NVSHMEM_HOST_LIB
    NAMES libnvshmem_host.so.3 nvshmem_host
    PATHS ${NVSHMEM_HOME}/lib
  )
  target_link_libraries(triton_distributed PUBLIC triton_distributed_kernel)
  target_link_libraries(triton_distributed PUBLIC ${NVSHMEM_HOST_LIB} PRIVATE -lnvshmem_device)
  target_include_directories(triton_distributed PUBLIC ${NVSHMEM_HOME}/include)
  target_link_directories(triton_distributed PUBLIC "${NVSHMEM_HOME}/lib")

  # Get the link directory of triton_distributed
  get_target_property(ORIGINAL_LINK_DIRS triton_distributed LINK_DIRECTORIES)

  # Both set absolute and relateive link path
  set_target_properties(triton_distributed PROPERTIES
      INSTALL_RPATH "$ORIGIN;${ORIGINAL_LINK_DIRS}"
      BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()
