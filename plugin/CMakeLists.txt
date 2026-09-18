# ===========================================================================
# Triton-distributed out-of-tree plugin
#
# Builds two decoupled shared libraries against a separately-built libtriton.so:
#   libtriton_dist.so      — plugin entry point + Distributed/SIMT dialects +
#                            conversion passes + AxisInfo visitors. Loaded via
#                            TRITON_PLUGIN_PATHS. Sole owner of the op TypeIDs.
#   libtriton_dist_ext*.so — pybind11 create-op companion (lib/ext/*): op
#                            builders + bindings; resolves op symbols/TypeIDs
#                            from the plugin at runtime (RTLD_GLOBAL).
#
# Upstream Triton 3.7.1 carries two managed, op-agnostic patches under
# plugin/patches/ (AxisInfo visitor-registration hook + software-pipeliner
# predication hook), applied by scripts/build_triton.sh; no distributed-specific
# code lives in Triton (the plugin-side consumers are lib/AxisInfoVisitors.cpp and
# lib/PipelinerPredication.cpp).
#
# Inputs (env or -D):
#   TRITON_SOURCE_DIR  : Triton source tree (for headers + python/src/ir.h)
#   TRITON_BUILD_DIR   : Triton CMake build dir (for tablegen'd headers)
#   LLVM_INSTALL_DIR   : LLVM/MLIR install (provides MLIR/LLVM CMake packages)
# Triton library (libtriton.so) is located under
#   ${TRITON_SOURCE_DIR}/python/triton/_C and linked for Triton-internal symbols.
# ===========================================================================
cmake_minimum_required(VERSION 3.20)
project(triton_dist_plugin LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(POLICY CMP0116)
  cmake_policy(SET CMP0116 OLD)
endif()

# --- Resolve inputs from environment if not passed via -D -------------------
foreach(_v TRITON_INSTALL_DIR TRITON_SOURCE_DIR TRITON_BUILD_DIR LLVM_INSTALL_DIR)
  if(NOT ${_v} AND DEFINED ENV{${_v}})
    set(${_v} "$ENV{${_v}}")
  endif()
endforeach()

# Two supported modes (mirrors triton-ext):
#   1. TRITON_INSTALL_DIR        -> an installed Triton (headers + lib in one tree)
#   2. TRITON_SOURCE_DIR (+ TRITON_BUILD_DIR) -> a Triton source checkout + its
#      CMake build dir (the build dir provides the tablegen'd *.inc headers).
# The install path is preferred for CI/wheels; the source+build path for dev.
if(NOT TRITON_INSTALL_DIR AND NOT TRITON_SOURCE_DIR)
  message(FATAL_ERROR "Set TRITON_INSTALL_DIR, or TRITON_SOURCE_DIR (+ TRITON_BUILD_DIR)")
endif()
if(NOT LLVM_INSTALL_DIR)
  message(FATAL_ERROR "Set LLVM_INSTALL_DIR to the LLVM/MLIR install")
endif()

# --- Find MLIR / LLVM -------------------------------------------------------
find_package(MLIR REQUIRED CONFIG PATHS "${LLVM_INSTALL_DIR}/lib/cmake/mlir" NO_DEFAULT_PATH)
message(STATUS "Found MLIR ${MLIR_DIR}")
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

# --- Locate libtriton.so + headers (mode 1: install, mode 2: source+build) --
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${LLVM_INCLUDE_DIRS})
if(TRITON_INSTALL_DIR)
  find_library(TRITON_LIB triton
    PATHS "${TRITON_INSTALL_DIR}/lib" "${TRITON_INSTALL_DIR}/python/triton/_C"
    NO_DEFAULT_PATH)
  if(NOT TRITON_LIB)
    message(FATAL_ERROR "Could not find libtriton.so under ${TRITON_INSTALL_DIR}")
  endif()
  include_directories("${TRITON_INSTALL_DIR}/include")
  include_directories("${TRITON_INSTALL_DIR}")
  include_directories("${TRITON_INSTALL_DIR}/include/triton/third_party")
  include_directories("${TRITON_INSTALL_DIR}/include/third_party")
  include_directories("${TRITON_INSTALL_DIR}/include/third_party/nvidia/include")
  include_directories("${TRITON_INSTALL_DIR}/include/third_party/amd/include")
else()
  find_library(TRITON_LIB triton
    PATHS "${TRITON_SOURCE_DIR}/python/triton/_C" NO_DEFAULT_PATH)
  if(NOT TRITON_LIB)
    message(FATAL_ERROR "Could not find libtriton.so under ${TRITON_SOURCE_DIR}/python/triton/_C")
  endif()
  # Triton headers (upstream source)
  include_directories("${TRITON_SOURCE_DIR}/include")
  include_directories("${TRITON_SOURCE_DIR}")                         # for python/src/ir.h
  include_directories("${TRITON_SOURCE_DIR}/third_party")
  include_directories("${TRITON_SOURCE_DIR}/third_party/nvidia/include")
  include_directories("${TRITON_SOURCE_DIR}/third_party/amd/include")
  include_directories("${TRITON_SOURCE_DIR}/third_party/proton/Dialect/include")
  # The tablegen'd Triton headers (e.g. AttrInterfaces.h.inc) live in the CMake
  # build dir. If the caller pointed us at a source tree but forgot to pass
  # TRITON_BUILD_DIR, fall back to the newest build/cmake* under it so the plugin
  # still compiles instead of failing with "No such file or directory".
  if(NOT TRITON_BUILD_DIR)
    file(GLOB _triton_build_dirs "${TRITON_SOURCE_DIR}/build/cmake*")
    if(_triton_build_dirs)
      list(SORT _triton_build_dirs)
      list(GET _triton_build_dirs -1 TRITON_BUILD_DIR)
      message(STATUS "TRITON_BUILD_DIR not set; discovered ${TRITON_BUILD_DIR}")
    endif()
  endif()
  if(TRITON_BUILD_DIR)
    include_directories("${TRITON_BUILD_DIR}/include")                # tablegen'd triton headers
    include_directories("${TRITON_BUILD_DIR}/third_party")
    include_directories("${TRITON_BUILD_DIR}/third_party/proton/Dialect/include")
    include_directories("${TRITON_BUILD_DIR}/third_party/nvidia/include") # NVGPU tablegen
  endif()
endif()
message(STATUS "Found Triton library: ${TRITON_LIB}")
# Triton-distributed headers (this repo)
set(TRITON_DIST_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")
include_directories("${TRITON_DIST_ROOT}/include")
include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")          # tablegen'd dist headers

add_compile_options(-fPIC -Wno-covered-switch-default)

# --- Collect object libraries (filled by subdirectories) -------------------
# We reuse the existing tablegen .td and dialect/conversion .cpp sources via a
# local add_triton_library() shim that produces OBJECT libraries and drops
# Triton-internal LINK_LIBS (resolved at load time from libtriton.so).
set_property(GLOBAL PROPERTY TRITON_DIST_OBJECTS "")

function(add_triton_library name)
  cmake_parse_arguments(ARG "" "" "DEPENDS;LINK_LIBS" ${ARGN})
  add_library(${name} OBJECT ${ARG_UNPARSED_ARGUMENTS})
  # Only depend on tablegen targets that exist in this out-of-tree build;
  # Triton-internal IncGen targets (TritonConversionPassIncGen, ...) already
  # have their generated headers in ${TRITON_BUILD_DIR}/include.
  foreach(dep ${ARG_DEPENDS})
    if(TARGET ${dep})
      add_dependencies(${name} ${dep})
    endif()
  endforeach()
  # Only keep MLIR*/LLVM* link libs (header/usage requirements); Triton-internal
  # targets (TritonIR, TritonGPUIR, ...) are not available out-of-tree and are
  # resolved from libtriton.so at dlopen time.
  foreach(lib ${ARG_LINK_LIBS})
    if(lib MATCHES "^(MLIR|LLVM)")
      target_link_libraries(${name} PUBLIC ${lib})
    endif()
  endforeach()
  llvm_update_compile_flags(${name})
  set_property(GLOBAL APPEND PROPERTY TRITON_DIST_OBJECTS ${name})
endfunction()

# Build model: exactly like the removed monolithic CMakeLists (and libtriton.so
# itself, which always ships TRITON_CODEGEN_BACKENDS "nvidia;amd"), the plugin
# ALWAYS compiles the full lowering -- the Distributed/SIMT dialects + the
# TritonGPU and LLVM conversions for NVIDIA + AMD + SIMT. There are no per-pass /
# per-backend build toggles and no auto-detection: the AMD TritonAMDGPU* symbols
# resolve from libtriton.so at load time, so building the AMD lowering is free of
# runtime cost when unused. METAX (MACA) is the ONLY switch, because it is a
# distinct toolchain that REPLACES the NVIDIA/AMD path (mirrors Triton's own
# TRITON_USE_MACA).
option(TRITON_USE_MACA "Build the METAX (MACA) LLVM lowering instead of NVIDIA/AMD" OFF)
if(TRITON_USE_MACA)
  add_compile_definitions(USE_MACA)
endif()

# include/ runs tablegen (generates the dialect + pass headers).
add_subdirectory("${TRITON_DIST_ROOT}/include/TritonDistributed"
                 "${CMAKE_CURRENT_BINARY_DIR}/include/TritonDistributed")
add_subdirectory("${TRITON_DIST_ROOT}/lib/Dialect/Distributed/IR"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/Distributed/IR")
add_subdirectory("${TRITON_DIST_ROOT}/lib/Dialect/SIMT/IR"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib/Dialect/SIMT/IR")
add_subdirectory("${TRITON_DIST_ROOT}/lib/Conversion/TritonDistributedToTritonGPU"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib/Conversion/TritonDistributedToTritonGPU")
add_subdirectory("${TRITON_DIST_ROOT}/lib/Conversion/TritonDistributedToLLVM"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib/Conversion/TritonDistributedToLLVM")

# --- Plugin shared library --------------------------------------------------
get_property(_dist_objs GLOBAL PROPERTY TRITON_DIST_OBJECTS)
set(_obj_targets "")
foreach(o ${_dist_objs})
  list(APPEND _obj_targets $<TARGET_OBJECTS:${o}>)
endforeach()

# The plugin .so carries ONLY the MLIR plugin proper: the Distributed/SIMT
# dialects (TypeID owners, via the lib/Dialect object libraries) + the conversion
# passes + the plugin entry point. The op *builders* (create-op subsystem) live in
# the separate companion module below (libtriton_dist_ext). The builders only
# *reference* the dialect ops, whose TypeIDs are DECLARE'd in headers and DEFINE'd
# once in the dialect TUs compiled here; the companion resolves those symbols at
# runtime from the RTLD_GLOBAL plugin, so there is still a single TypeID owner.
# AxisInfoVisitors.cpp / PipelinerPredication.cpp register the distributed
# AxisInfo visitors and the pipeliner predicator via Triton's extension hooks
# (they reference the dialect op TypeIDs, so they belong here).
set(_plugin_srcs lib/Plugin.cpp lib/AxisInfoVisitors.cpp lib/PipelinerPredication.cpp)

add_library(triton_dist SHARED ${_plugin_srcs} ${_obj_targets})
set_target_properties(triton_dist PROPERTIES OUTPUT_NAME triton_dist)
target_link_libraries(triton_dist PRIVATE ${TRITON_LIB})
target_include_directories(triton_dist PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/lib")
add_dependencies(triton_dist DistributedTableGen DistributedAttrDefsIncGen)
# The plugin registers mlir::tensor::TensorDialect (Plugin.cpp) and is the single
# owner of the tensor dialect TypeIDs/op symbols, which the companion's SIMT
# extract/insert builders create at runtime (resolved from this RTLD_GLOBAL .so).
# Upstream libtriton.so links these dialects but exports neither the dialects'
# registration nor their (MLIR_DEFINE_EXPLICIT_TYPE_ID, non-weak, hidden) TypeID
# symbols, so we statically link the archives here to make TensorDialect
# self-contained and its symbols exported. Closure: tensor -> {arith, complex,
# affine}; complex -> arith. arith's TypeID is already weak-instantiated by the
# plugin's own TUs; the rest are pulled in here. Remaining MLIR deps resolve from
# libtriton at load time (--allow-shlib-undefined below).
foreach(_mlir_dialect_lib MLIRTensorDialect MLIRComplexDialect)
  target_link_libraries(triton_dist PRIVATE
    "${LLVM_INSTALL_DIR}/lib/lib${_mlir_dialect_lib}.a")
endforeach()
# Allow Triton/MLIR/LLVM symbols to remain undefined at link time; resolved from
# the loading process (libtriton.so, RTLD_GLOBAL) at runtime.
target_link_options(triton_dist PRIVATE -Wl,--allow-shlib-undefined)
# NOTE: We intentionally do NOT restrict the plugin's exported symbols with a
# version script / --exclude-libs. loadPlugins dlopens the plugin RTLD_GLOBAL;
# MLIR identifies types/traits/interfaces by TypeID, whose storage is emitted as
# *weak* symbols (MLIR_DEFINE_EXPLICIT_TYPE_ID) that MUST stay exported so the
# dynamic linker merges the plugin's and libtriton's copies into one. Hiding them
# gives the plugin a private TypeID for e.g. OpTrait::IsTerminator, so a SIMT
# block_yield built here registers its terminator trait under a TypeID the
# libtriton-side verifier never checks -> "block with no terminator". The
# libstdc++ ABI interposition that motivated hiding symbols is instead fixed at
# its source: libtriton localizes those instantiations (scripts/build_triton.sh +
# triton_hide_libstdcxx.map).

# --- Companion Python extension module (the create-op subsystem) ------------
# A separate pybind11 module that owns the distributed op builders and exposes
# them to the Python frontend. It exists (rather than plain OpInfo callbacks)
# because on Triton 3.7.1 the plugin-op binding inserts no result slot and returns
# no Value, so it cannot drive value-returning / attribute-carrying ops. Upstream
# 3.7.1 also binds its IR with pybind11 and registers the wrapper types
# module-local; this module re-declares the few it must return and relies on
# pybind11's foreign module-local loading for the rest (see ext/py_module.cpp). It
# MUST be built with the SAME pybind11 as libtriton (so the module-local handshake
# matches). Resolves MLIR/LLVM from libtriton and the Distributed/SIMT/tensor op
# symbols from libtriton_dist.so (both RTLD_GLOBAL at runtime).
option(TRITON_DIST_BUILD_PY_EXT "Build the distributed create-op companion module" ON)
if(TRITON_DIST_BUILD_PY_EXT)
  find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)
  # Use the same pybind11 Triton built against (its CMake dir is discovered from
  # the installed pybind11 package); keeps PYBIND11_INTERNALS_ID compatible.
  if(NOT pybind11_DIR AND NOT DEFINED ENV{pybind11_DIR})
    execute_process(
      COMMAND "${Python_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
      OUTPUT_VARIABLE _pybind11_cmake_dir OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(pybind11_DIR "${_pybind11_cmake_dir}")
  endif()
  find_package(pybind11 CONFIG REQUIRED)
  # The companion hosts the whole "create-op" subsystem: the op builders + the
  # pybind11 bindings that expose them to the Python frontend. The builders
  # include the dialect headers (DECLARE'd TypeIDs only) and call op::create; the
  # op symbols + their single TypeID DEFINE'itions resolve at runtime from the
  # RTLD_GLOBAL plugin .so. No dialect TU is compiled here, so there is no second
  # TypeID owner (the duplicate-TypeID hazard the legacy companion hit).
  pybind11_add_module(triton_dist_ext
    lib/ext/py_module.cpp
    lib/ext/op_builders.cpp
    lib/ext/extern_call_builder.cpp
    lib/ext/simt_region_builder.cpp)
  set_target_properties(triton_dist_ext PROPERTIES OUTPUT_NAME libtriton_dist_ext)
  target_include_directories(triton_dist_ext PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/lib/ext")
  # Builders include the tablegen'd dialect headers, so wait for them.
  add_dependencies(triton_dist_ext DistributedTableGen DistributedAttrDefsIncGen)
  # libtriton provides MLIR/LLVM/pybind11 symbols; the Distributed/SIMT (and the
  # statically-linked tensor) dialect ops resolve from the RTLD_GLOBAL plugin .so
  # at runtime (loaded first by triton_dist/_plugin.py).
  target_link_libraries(triton_dist_ext PRIVATE ${TRITON_LIB})
  target_link_options(triton_dist_ext PRIVATE -Wl,--allow-shlib-undefined)
endif()
