cmake_minimum_required(VERSION 3.17 FATAL_ERROR)

set(DISTRIBUTED_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============ Add check for includes here (for future) =============

# ============ Define a GLOBAL property to store object-libraries ============
set_property(GLOBAL PROPERTY DISTRIBUTED_LIBS "")

# ============ Define a function to create object libraries ============
function(add_distributed_library name)
  add_library(${name} SHARED ${ARGN})

  target_link_libraries(${name} PUBLIC Python3::Module pybind11::headers)

  target_include_directories(${name}
    PUBLIC
      "${DISTRIBUTED_SRC_DIR}/../include"
  )

  # If HIP is AMD-based
  target_compile_definitions(${name} PRIVATE __HIP_PLATFORM_AMD__)

  # Append this library name to the GLOBAL property "DISTRIBUTED_LIBS"
  set_property(GLOBAL APPEND PROPERTY DISTRIBUTED_LIBS ${name})
endfunction()

# ============ Add subdirectory with actual code that calls add_distributed_library ============

if(USE_TRITON_DISTRIBUTED_AOT)
    add_subdirectory(triton_aot_generated)
endif()
add_subdirectory(lib)

# ============ Collect all object libraries from property and build final shared lib ============
get_property(_distributed_obj_libs GLOBAL PROPERTY DISTRIBUTED_LIBS)

if(NOT _distributed_obj_libs)
  message(WARNING "No object libraries were defined in 'DISTRIBUTED_LIBS'!")
endif()
