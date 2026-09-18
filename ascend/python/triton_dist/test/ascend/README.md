### Usage

1. **Build the project**
   Run the build script from the `Triton-distributed/` root directory:
```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} TRITON_BUILD_WITH_CLANG_LLD=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" pip install -e ./python --verbose --no-build-isolation
```

1. **Run the example program**
   Navigate to the example directory and execute the run script:
```bash
   pytest python/triton_dist/test/ascend/ -m dist
```
