cmake_minimum_required(VERSION 3.12)

project(pymxshmem LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
list(APPEND CMAKE_MODULE_PATH "${TRITON_SRC_DIR}/cmake")

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

find_package(CUDA REQUIRED)
find_package(CUDAToolkit REQUIRED)


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

set(SOURCES
    src/pymxshmem.cc
    src/flush_l2c.cu
)

# get pybind11 lib suffix
execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "-c"
    "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
  OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
  ERROR_VARIABLE PYTHON_MODULE_EXTENSION_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

add_library(_pymxshmem MODULE ${SOURCES})
set_target_properties(_pymxshmem PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")

include_directories(${CUDA_INCLUDE_DIRS})
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")

set_target_properties(_pymxshmem PROPERTIES CXX_STANDARD 17
                                           CUDA_RESOLVE_DEVICE_SYMBOLS ON)

find_package(MXSHMEM REQUIRED)
target_link_libraries(_pymxshmem PRIVATE MXSHMEM::mxshmem_host
                                         MXSHMEM::mxshmem_device
                                         torch
                                         ${TORCH_PYTHON_LIBRARY}
                                         Python3::Module
                                         pybind11::headers)
target_include_directories(_pymxshmem PRIVATE ${MXSHMEM_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS})

set(CXX_FLAGS
    -O3
    -DTORCH_CUDA=1
    -fvisibility=hidden
    -Wno-deprecated-declarations
    -fdiagnostics-color=always
)
target_compile_options(_pymxshmem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-rdc=true>)
target_compile_options(_pymxshmem PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${CXX_FLAGS}>)
target_compile_options(_pymxshmem PRIVATE "-DUSE_MACA")
