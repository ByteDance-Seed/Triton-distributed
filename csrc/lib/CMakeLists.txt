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

if(USE_TRITON_DISTRIBUTED_AOT)
  target_link_libraries(triton_distributed PUBLIC triton_distributed_kernel)
  # Get the link directory of triton_distributed
  get_target_property(ORIGINAL_LINK_DIRS triton_distributed LINK_DIRECTORIES)

  # Both set absolute and relateive link path
  set_target_properties(triton_distributed PROPERTIES
      INSTALL_RPATH "$ORIGIN;${ORIGINAL_LINK_DIRS}"
      BUILD_WITH_INSTALL_RPATH TRUE
  )
endif()
