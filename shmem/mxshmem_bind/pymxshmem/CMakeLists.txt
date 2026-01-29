cmake_minimum_required(VERSION 3.12)

project(pymxshmem LANGUAGES CXX CUDA)

message(FATAL_ERROR "camake start")
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

if(TORCH_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

find_package(MXSHMEM REQUIRED)

set(SOURCES
    src/pymxshmem.cc
    src/flush_l2c.cu
)

pybind11_add_module(pymxshmem ${SOURCES})

include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

set_target_properties(pymxshmem PROPERTIES CXX_STANDARD 17
                                           CUDA_RESOLVE_DEVICE_SYMBOLS ON)

target_link_libraries(pymxshmem PRIVATE mxshmem::mxshmem_host
                                        mxshmem::mxshmem_device
                                        torch
                                        ${TORCH_PYTHON_LIBRARY})

target_include_directories(pymxshmem PRIVATE ${MXSHMEM_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})
target_compile_options(pymxshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
target_compile_options(pymxshmem PRIVATE "-DUSE_MACA")
