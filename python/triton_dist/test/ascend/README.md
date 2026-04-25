### 使用方式

1. **编译项目**
   在 `Triton-distributed/` 根目录下执行编译脚本：
   ```bash
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} TRITON_BUILD_WITH_CLANG_LLD=ON TRITON_BUILD_PROTON=OFF TRITON_BUILD_LITTLE_KERNEL=OFF TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" pip install -e ./python --verbose --no-build-isolation
   ```

1. **运行示例程序**
   进入示例目录并执行运行脚本：
   ```bash
   pytest python/triton_dist/test/ascend/ -m dist
   ```
