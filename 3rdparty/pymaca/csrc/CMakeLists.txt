cmake_minimum_required(VERSION 3.12)

find_package(
  Python3
  COMPONENTS Interpreter Development
  REQUIRED)
find_program(PYTHON_EXECUTABLE NAMES python3 python)
find_package(pybind11 CONFIG REQUIRED HINTS "${Python3_SITELIB}")

# find MACA_PATH
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

# get pybind11 lib suffix
execute_process(
  COMMAND
    "${PYTHON_EXECUTABLE}" "-c"
    "import sys, importlib; s = importlib.import_module('distutils.sysconfig' if sys.version_info < (3, 10) else 'sysconfig'); print(s.get_config_var('EXT_SUFFIX') or s.get_config_var('SO'))"
  OUTPUT_VARIABLE PYTHON_MODULE_EXTENSION
  ERROR_VARIABLE PYTHON_MODULE_EXTENSION_ERR
  OUTPUT_STRIP_TRAILING_WHITESPACE)

file(GLOB LIB_FILES "pybind.cc")
message(STATUS "lib files: ${LIB_FILES}")

add_library(maca SHARED ${LIB_FILES})
set_target_properties(maca PROPERTIES PREFIX "" SUFFIX "${PYTHON_MODULE_EXTENSION}")
target_link_libraries(maca PUBLIC Python3::Module pybind11::headers ${MACA_LIBRARY_DIR}/libmcruntime.so)
