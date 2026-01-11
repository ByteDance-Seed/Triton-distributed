cmake_minimum_required(VERSION 3.12)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)

if(DEFINED ENV{MACA_PATH})
    set(MACA_PATH $ENV{MACA_PATH})
    message(STATUS "Using maca path from environment")
else()
    message(STATUS "Using default maca path")
endif()
message(STATUS "Using MACA_PATH: ${MACA_PATH}")

find_path(MACA_INCLUDE_DIR
    NAMES mcr/maca.h mcr/mc_runtime.h
    HINTS ${MACA_PATH}/include
    PATHS /opt/maca/include
)
if(MACA_INCLUDE_DIR MATCHES "/mcr$")
    get_filename_component(MACA_INCLUDE_DIR ${MACA_INCLUDE_DIR} DIRECTORY)
endif()

find_path(MACA_LIBRARY_DIR
    NAMES libmcruntime.so
    HINTS ${MACA_PATH}/lib
    PATHS /opt/maca/lib
)

message(STATUS "Found MACA_INCLUDE_DIR: ${MACA_INCLUDE_DIR}")
message(STATUS "Found MACA_LIBRARY_DIR: ${MACA_LIBRARY_DIR}")

include_directories(${MACA_INCLUDE_DIR})
include_directories(${MACA_INCLUDE_DIR}/mcr/)

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

file(GLOB LIB_FILES "pybind.cc")
message(STATUS "lib files: ${LIB_FILES}")
pybind11_add_module(maca SHARED ${LIB_FILES})

target_link_libraries(maca PUBLIC Python3::Module pybind11::headers ${MACA_LIBRARY_DIR}/libmcruntime.so)
