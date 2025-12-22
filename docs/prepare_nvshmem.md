# Download and fix NVSHMEM


1. Download NVSHMEM 3.2.5 Source Code [NVSHMEM Open Source Packages](https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz)
    ```sh
    cd /workspace
    wget https://developer.nvidia.com/downloads/assets/secure/nvshmem/nvshmem_src_3.2.5-1.txz
    ```

2. Extract to designated location
    ```sh
    tar -xvf nvshmem_src_3.2.5-1.txz
    ```

3. Bitcode Bug Fix: [BUG with nvshmem 3.2.5 for bitcode compiling](https://forums.developer.nvidia.com/t/bug-with-nvshmem-3-2-5-for-bitcode-compiling/327847)

    > Note: This step is because of NVSHMEM license requirements, it is illegal to release any modified codes or patch.

    File: ```src/include/non_abi/device/common/nvshmemi_common_device.cuh``` (Line 287)
    ```diff
    - dst = (void *)(dst_p + nelems);
    - src = (void *)(src_p + nelems);

    +#ifdef __clang_llvm_bitcode_lib__
    +    dst = (void *)(dst_p + nelems * 4);
    +    src = (void *)(src_p + nelems * 4);
    +#else
    +    dst = (void *)(dst_p + nelems);
    +    src = (void *)(src_p + nelems);
    +#endif
    ```

4. Clang Compilation Error Fix

    > Note: This step is because of NVSHMEM license requirements, it is illegal to release any modified codes or patch.

    File: ```src/include/device_host/nvshmem_common.cuh``` (Line 41)
    ```diff
    - __device__ int __nvvm_reflect(const char *s);
    + __device__ int __nvvm_reflect(const void *s);
    ```

5. Setup `NVSHMEM_SRC` environment variable
    ```sh
    export NVSHMEM_SRC=/workspace/nvshmem_src
    ```
