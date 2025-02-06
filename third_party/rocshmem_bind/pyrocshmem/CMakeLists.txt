cmake_minimum_required(VERSION 3.12)

## Add build of pyrocshmem
project(pyrocshmem LANGUAGES CXX)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

###############################################################################
# GLOBAL COMPILE FLAGS
###############################################################################
if (DEFINED ENV{ROCM_PATH})
  set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE STRING "ROCm install directory")
else()
  set(ROCM_PATH "/opt/rocm" CACHE STRING "ROCm install directory")
endif()

# force reset cxx to hipcc
set(CMAKE_CXX_COMPILER ${ROCM_PATH}/bin/hipcc)
message("CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}")
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -ggdb")

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION FALSE)

###############################################################################
# HIP
###############################################################################
find_package(hip REQUIRED)

###############################################################################
# MPI
###############################################################################
find_package(MPI REQUIRED)
if(MPI_FOUND)
    message(STATUS "MPI found!")
    message(STATUS "MPI_INCLUDE_DIRS: ${MPI_INCLUDE_DIR}")
    message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
else()
    message(FATAL_ERROR "MPI not found.")
endif()

# find pybind
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

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wpointer-arith")
set(CMAKE_FLAGS "${CMAKE_FLAGS} -Wpointer-arith")

find_package(ROCSHMEM REQUIRED)
if(ROCSHMEM_FOUND)
    message(STATUS "ROCSHMEM found!")
    message(STATUS "ROCSHMEM_INCLUDE_DIRS: ${rocshmem_INCLUDE_DIR}")
    message(STATUS "ROCSHMEM_LIBRARIES: ${rocshmem_LIBRARIES}")
else()
    message(FATAL_ERROR "ROCSHMEM not found.")
endif()

pybind11_add_module(pyrocshmem src/pyrocshmem.cc)
include_directories(${ROCM_INCLUDE_DIRS})
message(STATUS "ROCM include directories: ${ROCM_INCLUDE_DIRS}")

set_target_properties(pyrocshmem PROPERTIES CXX_STANDARD 17
                                           CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(pyrocshmem PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)
target_link_libraries(pyrocshmem PRIVATE roc::rocshmem torch ${TORCH_PYTHON_LIBRARY})
target_link_libraries(pyrocshmem PUBLIC hip::device hip::host ${MPI_mpi_LIBRARY} ${MPI_mpicxx_LIBRARY})
message(STATUS "MPI_CXX_HEADER_DIR=${MPI_CXX_HEADER_DIR}")
message(STATUS "MPI_mpi_LIBRARY=${MPI_mpi_LIBRARY}")
message(STATUS "MPI_mpicxx_LIBRARY=${MPI_mpicxx_LIBRARY}")
target_include_directories(pyrocshmem PRIVATE ${ROCSHMEM_INCLUDE_DIRS} ${TORCH_INCLUDE_DIRS} ${MPI_CXX_HEADER_DIR})
target_compile_options(pyrocshmem PRIVATE -fgpu-rdc)
# FIXME: link options copied from rocshmem's example. Can we add compile/link options for rocm in a more generic way ?
target_link_options(pyrocshmem PRIVATE -fgpu-rdc -lamdhip64 -lhsa-runtime64)
