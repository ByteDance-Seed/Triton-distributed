cmake_minimum_required(VERSION 3.12)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)

execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c"
          "from __future__ import print_function; import os; import pybind11;
print(os.path.dirname(pybind11.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE PYBIND11_DIR)
message("PYTHON_EXECUTABLE:${PYTHON_EXECUTABLE}")
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message("PYBIND11_DIR: ${PYBIND11_DIR}")
  message(FATAL_ERROR "Pybind11 config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${PYBIND11_DIR})
find_package(pybind11 REQUIRED)

# find torch
execute_process(
  COMMAND ${PYTHON_EXECUTABLE} "-c"
          "from __future__ import print_function; import os; import torch;
print(os.path.dirname(torch.__file__),end='');"
  RESULT_VARIABLE _PYTHON_SUCCESS
  OUTPUT_VARIABLE TORCH_DIR)
if(NOT _PYTHON_SUCCESS MATCHES 0)
  message("PY:${PYTHONPATH}")
  message(FATAL_ERROR "Torch config Error.")
endif()
list(APPEND CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_DIR}/lib")
message(STATUS "torch lib: ${TORCH_DIR}/lib")

if(TORCH_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

file(GLOB CU_FILES "moe_utils.cu")
message(STATUS "cu files: ${CU_FILES}")

file(GLOB LIB_FILES "*.cc")
message(STATUS "lib files: ${LIB_FILES}")
file(GLOB PYBIND_FILES "../triton_aot_generated/pybind/*.cc")
message(STATUS "pybind files: ${PYBIND_FILES}")

add_distributed_library(triton_distributed ${LIB_FILES} ${PYBIND_FILES} ${CU_FILES})
include_directories(${CUDA_INCLUDE_DIRS})
target_include_directories(triton_distributed PUBLIC ${TORCH_INCLUDE_DIRS} ".")
target_include_directories(triton_distributed PUBLIC ${TORCH_INCLUDE_DIRS} "..")
target_include_directories(triton_distributed PUBLIC ${TORCH_INCLUDE_DIRS} "../triton_aot_generated")
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

target_link_libraries(triton_distributed PUBLIC torch ${TORCH_PYTHON_LIBRARY} CUDA::cudart)
string(TOLOWER "${USE_TRITON_DISTRIBUTED_AOT}" USE_TRITON_DISTRIBUTED_AOT_LOWER)
if(USE_TRITON_DISTRIBUTED_AOT_LOWER STREQUAL "on" OR USE_TRITON_DISTRIBUTED_AOT_LOWER STREQUAL "1" OR USE_TRITON_DISTRIBUTED_AOT_LOWER STREQUAL "true")
  target_link_libraries(triton_distributed PUBLIC triton_distributed_kernel)
endif()