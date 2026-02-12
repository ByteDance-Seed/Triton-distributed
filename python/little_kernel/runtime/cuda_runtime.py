################################################################################
#
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
################################################################################

import ctypes
import sys
import os
import subprocess
import tempfile
from typing import Optional, Tuple, List, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# CUDA Driver API types
CUresult = ctypes.c_int
CUdevice = ctypes.c_int
CUcontext = ctypes.c_void_p
CUmodule = ctypes.c_void_p
CUfunction = ctypes.c_void_p
CUstream = ctypes.c_void_p
CUdeviceptr = ctypes.c_void_p

# CUDA Driver API constants
CUDA_SUCCESS = 0
CUDA_ERROR_NOT_FOUND = 500
CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97
CU_FUNC_CACHE_PREFER_SHARED = 1
CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8
CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 12

# CUDA Launch Attribute IDs (for cluster launch)
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4


class CUDARuntime:
    """CUDA Driver API runtime wrapper."""

    def __init__(self):
        self._libcuda = None
        self._ctx = None
        self._device = None
        self._load_library()
        self._init_context()

    def _load_library(self):
        """Load libcuda.so library with robust path detection.
        
        Tries multiple methods to find libcuda:
        1. Environment variable LITTLE_KERNEL_LIBCUDA_PATH
        2. Standard library name (libcuda.so.1) - relies on system loader
        3. ldconfig cache (Linux)
        4. Common CUDA installation paths
        5. LD_LIBRARY_PATH
        """
        if sys.platform == "win32":
            lib_names = ["nvcuda.dll"]
        else:
            lib_names = ["libcuda.so.1", "libcuda.so"]

        # Try environment variable first
        env_libcuda_path = os.environ.get("LITTLE_KERNEL_LIBCUDA_PATH")
        if env_libcuda_path:
            if os.path.exists(env_libcuda_path):
                try:
                    self._libcuda = ctypes.CDLL(env_libcuda_path)
                except OSError as e:
                    raise RuntimeError(
                        f"Failed to load libcuda from LITTLE_KERNEL_LIBCUDA_PATH={env_libcuda_path}: {e}")
            else:
                raise RuntimeError(f"LITTLE_KERNEL_LIBCUDA_PATH={env_libcuda_path} does not exist")
        else:
            # Try standard library names first (relies on system loader)
            for lib_name in lib_names:
                try:
                    self._libcuda = ctypes.CDLL(lib_name)
                    break
                except OSError:
                    continue

            # On Linux, try to find via ldconfig if standard names failed
            if self._libcuda is None and sys.platform != "win32":
                try:
                    libs_output = subprocess.check_output(["/sbin/ldconfig", "-p"], stderr=subprocess.DEVNULL).decode()
                    # Parse ldconfig output: libcuda.so.1 (libc6,x86-64) => /usr/lib/x86_64-linux-gnu/libcuda.so.1
                    for line in libs_output.splitlines():
                        if "libcuda.so" in line:
                            # Extract path (last field after =>)
                            if "=>" in line:
                                lib_path = line.split("=>")[-1].strip()
                                if os.path.exists(lib_path):
                                    try:
                                        self._libcuda = ctypes.CDLL(lib_path)
                                        break
                                    except OSError:
                                        continue
                except (subprocess.CalledProcessError, FileNotFoundError):
                    pass

                # Try common CUDA installation paths
                if self._libcuda is None:
                    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                    if cuda_home:
                        for lib_name in lib_names:
                            lib_path = os.path.join(cuda_home, "lib64", lib_name)
                            if os.path.exists(lib_path):
                                try:
                                    self._libcuda = ctypes.CDLL(lib_path)
                                    break
                                except OSError:
                                    continue
                            if self._libcuda is None:
                                # Also try lib/ directory
                                lib_path = os.path.join(cuda_home, "lib", lib_name)
                                if os.path.exists(lib_path):
                                    try:
                                        self._libcuda = ctypes.CDLL(lib_path)
                                        break
                                    except OSError:
                                        continue

                # Try LD_LIBRARY_PATH
                if self._libcuda is None:
                    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                    for lib_dir in ld_library_path.split(":"):
                        if not lib_dir:
                            continue
                        for lib_name in lib_names:
                            lib_path = os.path.join(lib_dir, lib_name)
                            if os.path.exists(lib_path):
                                try:
                                    self._libcuda = ctypes.CDLL(lib_path)
                                    break
                                except OSError:
                                    continue
                        if self._libcuda is not None:
                            break

        # If all methods failed, provide helpful error message
        if self._libcuda is None:
            error_msg = "Failed to load libcuda library. Tried:\n"
            error_msg += f"  1. Standard library names: {lib_names}\n"
            if sys.platform != "win32":
                error_msg += "  2. ldconfig cache\n"
                cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
                if cuda_home:
                    error_msg += f"  3. CUDA_HOME/CUDA_PATH: {cuda_home}\n"
                ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
                if ld_library_path:
                    error_msg += f"  4. LD_LIBRARY_PATH: {ld_library_path}\n"
            error_msg += "\nSuggestions:\n"
            error_msg += "  - Set LITTLE_KERNEL_LIBCUDA_PATH environment variable to the full path of libcuda.so.1\n"
            error_msg += "  - Ensure CUDA driver is installed and GPU is available\n"
            if sys.platform != "win32":
                error_msg += "  - Run 'ldconfig' (requires sudo) to refresh the linker cache\n"
                error_msg += "  - Add CUDA library directory to LD_LIBRARY_PATH\n"
            raise RuntimeError(error_msg)

        # Define function signatures (only reached if library loaded successfully)
        self._libcuda.cuInit.argtypes = [ctypes.c_uint]
        self._libcuda.cuInit.restype = CUresult

        self._libcuda.cuDeviceGet.argtypes = [ctypes.POINTER(CUdevice), ctypes.c_int]
        self._libcuda.cuDeviceGet.restype = CUresult

        self._libcuda.cuCtxCreate.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
        self._libcuda.cuCtxCreate.restype = CUresult

        self._libcuda.cuModuleLoadData.argtypes = [ctypes.POINTER(CUmodule), ctypes.c_char_p]
        self._libcuda.cuModuleLoadData.restype = CUresult

        self._libcuda.cuModuleGetFunction.argtypes = [ctypes.POINTER(CUfunction), CUmodule, ctypes.c_char_p]
        self._libcuda.cuModuleGetFunction.restype = CUresult

        self._libcuda.cuLaunchKernel.argtypes = [
            CUfunction, ctypes.c_uint,  # gridDimX
            ctypes.c_uint,  # gridDimY
            ctypes.c_uint,  # gridDimZ
            ctypes.c_uint,  # blockDimX
            ctypes.c_uint,  # blockDimY
            ctypes.c_uint,  # blockDimZ
            ctypes.c_uint,  # sharedMemBytes
            CUstream,  # hStream
            ctypes.POINTER(ctypes.c_void_p),  # kernelParams
            ctypes.POINTER(ctypes.c_void_p),  # extra
        ]
        self._libcuda.cuLaunchKernel.restype = CUresult

        self._libcuda.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, CUdevice]
        self._libcuda.cuDeviceGetAttribute.restype = CUresult

        self._libcuda.cuFuncSetCacheConfig.argtypes = [CUfunction, ctypes.c_int]
        self._libcuda.cuFuncSetCacheConfig.restype = CUresult

        self._libcuda.cuFuncSetAttribute.argtypes = [CUfunction, ctypes.c_int, ctypes.c_int]
        self._libcuda.cuFuncSetAttribute.restype = CUresult

        self._libcuda.cuStreamSynchronize.argtypes = [CUstream]
        self._libcuda.cuStreamSynchronize.restype = CUresult

        # Context management for multi-GPU
        self._libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(CUcontext), CUdevice]
        self._libcuda.cuDevicePrimaryCtxRetain.restype = CUresult
        self._libcuda.cuCtxSetCurrent.argtypes = [CUcontext]
        self._libcuda.cuCtxSetCurrent.restype = CUresult

        self._libcuda.cuGetErrorString.argtypes = [CUresult, ctypes.POINTER(ctypes.c_char_p)]
        self._libcuda.cuGetErrorString.restype = CUresult

        # Try to load cuLaunchKernelEx for cluster launch support
        try:
            # CUlaunchAttributeValue union - must be 64 bytes to match CUDA header
            class CUlaunchAttributeValue(ctypes.Union):
                _fields_ = [("clusterDim", ctypes.c_uint * 3),  # x, y, z
                            ("cooperative", ctypes.c_uint), ("priority", ctypes.c_int),
                            ("pad", ctypes.c_ubyte * 64),  # Ensure union is 64 bytes
                            ]

            # CUlaunchAttribute structure
            # CUDA header: char pad[8 - sizeof(CUlaunchAttributeID)] = 4 bytes
            class CUlaunchAttribute(ctypes.Structure):
                _fields_ = [("id", ctypes.c_int),  # 4 bytes
                            ("pad", ctypes.c_char * 4),  # 4 bytes padding
                            ("value", CUlaunchAttributeValue),  # 64 bytes
                            ]

            # CUlaunchConfig structure
            class CUlaunchConfig(ctypes.Structure):
                _fields_ = [
                    ("gridDimX", ctypes.c_uint),
                    ("gridDimY", ctypes.c_uint),
                    ("gridDimZ", ctypes.c_uint),
                    ("blockDimX", ctypes.c_uint),
                    ("blockDimY", ctypes.c_uint),
                    ("blockDimZ", ctypes.c_uint),
                    ("sharedMemBytes", ctypes.c_uint),
                    ("hStream", CUstream),
                    ("attrs", ctypes.POINTER(CUlaunchAttribute)),
                    ("numAttrs", ctypes.c_uint),
                ]

            self.CUlaunchConfig = CUlaunchConfig
            self.CUlaunchAttribute = CUlaunchAttribute

            self._libcuda.cuLaunchKernelEx.argtypes = [
                ctypes.POINTER(CUlaunchConfig),
                CUfunction,
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_void_p),
            ]
            self._libcuda.cuLaunchKernelEx.restype = CUresult
            self._has_launch_kernel_ex = True
        except AttributeError:
            # cuLaunchKernelEx not available in this CUDA version
            self._has_launch_kernel_ex = False

        # Try to load optional enumeration functions (available in newer CUDA versions)
        try:
            self._libcuda.cuModuleGetFunctionCount.argtypes = [ctypes.POINTER(ctypes.c_uint), CUmodule]
            self._libcuda.cuModuleGetFunctionCount.restype = CUresult
            self._libcuda.cuModuleEnumerateFunctions.argtypes = [
                ctypes.POINTER(CUfunction), ctypes.POINTER(ctypes.c_uint), CUmodule
            ]
            self._libcuda.cuModuleEnumerateFunctions.restype = CUresult
            # cuFuncGetName returns const char** (pointer to char*)
            self._libcuda.cuFuncGetName.argtypes = [ctypes.POINTER(ctypes.c_char_p), CUfunction]
            self._libcuda.cuFuncGetName.restype = CUresult
            self._has_enumeration_api = True
        except AttributeError:
            # Enumeration API not available in this CUDA version
            self._has_enumeration_api = False

    def _init_context(self):
        """Initialize CUDA context."""
        # Use PyTorch's context if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            # PyTorch manages the context, we just need to get the device
            self._device = torch.cuda.current_device()
            # Retain the primary context so we can set it later for Driver API calls
            ctx = CUcontext()
            device = CUdevice()
            err = self._libcuda.cuDeviceGet(ctypes.byref(device), self._device)
            self._check_error(err, "cuDeviceGet")
            err = self._libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), device)
            self._check_error(err, "cuDevicePrimaryCtxRetain")
            self._ctx = ctx
            return

        # Otherwise, create our own context
        err = self._libcuda.cuInit(0)
        self._check_error(err, "cuInit")

        device = CUdevice()
        err = self._libcuda.cuDeviceGet(ctypes.byref(device), 0)
        self._check_error(err, "cuDeviceGet")
        self._device = device

        ctx = CUcontext()
        err = self._libcuda.cuCtxCreate(ctypes.byref(ctx), 0, device)
        self._check_error(err, "cuCtxCreate")
        self._ctx = ctx

    def _ensure_context(self):
        """Ensure this runtime's CUDA context is the current one."""
        if hasattr(self, '_ctx') and self._ctx:
            self._libcuda.cuCtxSetCurrent(self._ctx)

    def _check_error(self, err: int, func_name: str):
        """Check CUDA error and raise exception if needed."""
        if err != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            self._libcuda.cuGetErrorString(err, ctypes.byref(error_str))
            raise RuntimeError(f"CUDA error in {func_name}: {error_str.value.decode()}")

    def load_module(self, cubin: bytes) -> CUmodule:
        """Load CUBIN module into the correct GPU context."""
        self._ensure_context()
        module = CUmodule()
        err = self._libcuda.cuModuleLoadData(ctypes.byref(module), cubin)
        self._check_error(err, "cuModuleLoadData")
        return module

    def enumerate_functions(self, module: CUmodule, cubin_data: Optional[bytes] = None) -> List[str]:
        """Enumerate all function symbols in a module.
        
        Args:
            module: CUDA module handle
            cubin_data: Optional CUBIN binary data for fallback enumeration using cuobjdump
        
        Returns:
            List of function symbol names
        """
        symbols = []

        # Try using CUDA Driver API enumeration (if available)
        if self._has_enumeration_api:
            try:
                num_funcs = ctypes.c_uint()
                err = self._libcuda.cuModuleGetFunctionCount(ctypes.byref(num_funcs), module)
                if err == CUDA_SUCCESS and num_funcs.value > 0:
                    # Create array of CUfunction pointers
                    func_array_type = CUfunction * num_funcs.value
                    func_list = func_array_type()
                    num_funcs_ref = ctypes.byref(num_funcs)
                    err = self._libcuda.cuModuleEnumerateFunctions(func_list, num_funcs_ref, module)
                    if err == CUDA_SUCCESS:
                        for i in range(num_funcs.value):
                            func_name_ptr = ctypes.c_char_p()  # Pointer to char*
                            err = self._libcuda.cuFuncGetName(ctypes.byref(func_name_ptr), func_list[i])
                            if err == CUDA_SUCCESS and func_name_ptr.value:
                                symbols.append(func_name_ptr.value.decode("utf-8"))
                        return symbols
            except Exception:
                # Silently fail and try fallback methods
                pass

        # Fallback: try using cuobjdump if cubin data is available
        if cubin_data:
            try:
                # Write cubin to temporary file
                with tempfile.NamedTemporaryFile(suffix='.cubin', delete=False) as tmp_file:
                    tmp_file.write(cubin_data)
                    tmp_path = tmp_file.name

                try:
                    # Try to find cuobjdump
                    cuobjdump_paths = [
                        os.path.join(os.environ.get('CUDA_HOME', ''), 'bin', 'cuobjdump'),
                        '/usr/local/cuda/bin/cuobjdump', 'cuobjdump',  # Try in PATH
                    ]

                    cuobjdump = None
                    for path in cuobjdump_paths:
                        if os.path.exists(path) or path == 'cuobjdump':
                            try:
                                result = subprocess.run([path, '-symbols', tmp_path], capture_output=True, text=True,
                                                        timeout=5)
                                if result.returncode == 0:
                                    cuobjdump = path
                                    break
                            except (FileNotFoundError, subprocess.TimeoutExpired):
                                continue

                    if cuobjdump:
                        result = subprocess.run([cuobjdump, '-symbols', tmp_path], capture_output=True, text=True,
                                                timeout=5)
                        if result.returncode == 0:
                            # Parse cuobjdump output
                            illegal_names = {"vprintf", "__instantiate_kernel", "__internal", "__assertfail"}
                            for line in result.stdout.split('\n'):
                                if 'STT_FUNC' in line and 'STO_ENTRY' in line:
                                    # Extract symbol name (last field)
                                    parts = line.split()
                                    if parts:
                                        symbol_name = parts[-1]
                                        if symbol_name not in illegal_names:
                                            symbols.append(symbol_name)
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

                if symbols:
                    return symbols
            except Exception:
                pass

        return symbols

    def get_function(self, module: CUmodule, kernel_name: str, cubin_data: Optional[bytes] = None) -> CUfunction:
        """Get kernel function from module.
        
        Args:
            module: CUDA module handle
            kernel_name: Name of the kernel function to retrieve
            cubin_data: Optional CUBIN binary data for fallback symbol enumeration
        
        Returns:
            CUfunction handle
        """
        func = CUfunction()
        err = self._libcuda.cuModuleGetFunction(ctypes.byref(func), module, kernel_name.encode("utf-8"))

        # If function not found, try to find mangled name or enumerate symbols
        if err == CUDA_ERROR_NOT_FOUND:
            symbols = self.enumerate_functions(module, cubin_data)
            print(symbols, flush=True)

            # Try to find a mangled name that contains the original function name
            mangled_match = None
            if symbols:
                # Look for mangled names that contain the original function name
                # C++ mangled names typically have the form: _Z<length><name>...
                for sym in symbols:
                    # Check if symbol contains the kernel name (for mangled names)
                    if kernel_name in sym:
                        # Prefer exact match at the start of mangled name (after _Z<length>)
                        # Or prefer shorter mangled names (less likely to be a false match)
                        if mangled_match is None or len(sym) < len(mangled_match):
                            mangled_match = sym

                # If found a match, try to use it
                if mangled_match:
                    print(f"Warning: Function '{kernel_name}' not found, but found mangled name '{mangled_match}'")
                    print("Attempting to use mangled name...")
                    err = self._libcuda.cuModuleGetFunction(ctypes.byref(func), module, mangled_match.encode("utf-8"))
                    if err == CUDA_SUCCESS:
                        print(f"Successfully loaded function using mangled name '{mangled_match}'")
                        return func
                    else:
                        print(f"Failed to load function using mangled name '{mangled_match}'")

            # If still not found, print all available symbols
            print(f"\nError: Function '{kernel_name}' not found in module.")
            if symbols:
                print(f"Available symbols in module ({len(symbols)} total):")
                for sym in symbols:
                    print(f"  - {sym}")
                print("\nNote: The function name may be mangled. Try using one of the symbols above,")
                print("      or use extern \"C\" linkage for the kernel function.")
            else:
                print("Note: Could not enumerate module symbols. This may be due to:")
                print("  1. CUDA Driver API enumeration not available in this CUDA version")
                print("  2. cuobjdump tool not available")
                print("  3. Function name may be mangled (C++ name mangling)")
                print("  4. Try using extern \"C\" linkage for the kernel function")
            print()

        self._check_error(err, "cuModuleGetFunction")
        return func

    def configure_function(self, func: CUfunction, shared_mem_bytes: int, cluster_dim: Optional[Tuple[int, int,
                                                                                                      int]] = None):
        """Configure function attributes for shared memory and cluster launch.
        
        Sets MaxDynamicSharedMemorySize and optionally NonPortableClusterSizeAllowed.
        """
        # Set NonPortableClusterSizeAllowed if cluster launch is requested
        if cluster_dim is not None:
            err = self._libcuda.cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1)
            self._check_error(err, "cuFuncSetAttribute (NonPortableCluster)")

        if shared_mem_bytes > 0:
            # Always set the shared memory size attribute before launch
            # This matches DeepGEMM's behavior
            err = self._libcuda.cuFuncSetAttribute(func, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                                   shared_mem_bytes)
            self._check_error(err, "cuFuncSetAttribute (shared memory)")

            # If shared memory exceeds 48KB default limit, also set cache config
            if shared_mem_bytes > 49152:  # 48KB default limit
                # Get max shared memory to verify we can use it
                max_shared = ctypes.c_int()
                err = self._libcuda.cuDeviceGetAttribute(ctypes.byref(max_shared),
                                                         CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                                                         self._device)
                if err == CUDA_SUCCESS and max_shared.value > 49152:
                    self._libcuda.cuFuncSetCacheConfig(func, CU_FUNC_CACHE_PREFER_SHARED)

    def launch_kernel(
        self,
        func: CUfunction,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: List[Any],
        shared_mem_bytes: int = 0,
        stream: Optional[CUstream] = None,
        cluster_dim: Optional[Tuple[int, int, int]] = None,
    ):
        """Launch CUDA kernel."""
        # Convert arguments to pointers
        # CUDA Driver API expects pointers to arguments
        arg_ptrs = []
        arg_data = []  # Keep references to avoid garbage collection

        for arg in args:
            # Check for TMA descriptor (CUtensorMap/cute::TmaDescriptor structure) first
            # TMA descriptors are 128-byte structures that need special handling
            # According to DeepGEMM: void *ptr_args[] = { &args... };
            # So we need to pass a pointer to the structure, not the structure itself
            if isinstance(arg, ctypes.Structure):
                # TMA descriptor or other structure: pass pointer to structure
                # Keep reference to prevent garbage collection
                arg_data.append(arg)
                # For __grid_constant__ parameters, cuLaunchKernel expects a pointer to the structure
                # DeepGEMM uses: void *ptr_args[] = { &args... };
                # So we pass the address of the structure (like &tensor_map_a in C++)
                # ctypes.byref() gives us a pointer to the structure
                struct_ptr = ctypes.cast(ctypes.byref(arg), ctypes.c_void_p)
                arg_ptrs.append(struct_ptr)
            elif isinstance(arg, (int, np.integer)):
                # Integer arguments: create a c_int32 or c_int64 and pass pointer
                val = int(arg)
                if -2**31 <= val < 2**31:
                    c_val = ctypes.c_int32(val)
                else:
                    c_val = ctypes.c_int64(val)
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
            elif hasattr(arg, "data_ptr"):  # PyTorch tensor
                # For pointer arguments, we need to pass a pointer to the pointer
                # DeepGEMM uses: void *ptr_args[] = { &args... };
                # So for void* ptr, we need &ptr (pointer to pointer)
                ptr_val = ctypes.c_void_p(arg.data_ptr())
                arg_data.append(ptr_val)  # Keep reference
                # Get pointer to the pointer value
                ptr_to_ptr = ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p)
                arg_ptrs.append(ptr_to_ptr)
            elif isinstance(arg, np.ndarray):
                # NumPy array: get pointer to data
                if not arg.flags.c_contiguous:
                    arg = np.ascontiguousarray(arg)
                ptr_val = ctypes.c_void_p(arg.ctypes.data)
                arg_data.append(ptr_val)  # Keep reference
                # Get pointer to the pointer value
                ptr_to_ptr = ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p)
                arg_ptrs.append(ptr_to_ptr)
            elif isinstance(arg, ctypes.c_void_p):
                # Already a void* pointer, need pointer to this pointer
                arg_data.append(arg)  # Keep reference
                ptr_to_ptr = ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p)
                arg_ptrs.append(ptr_to_ptr)
            elif isinstance(arg, int) and arg > 0:
                # Integer that might be a pointer address (e.g., from ctypes.addressof)
                # WARNING: If this is from ctypes.addressof() for a TMA descriptor,
                # the original object must remain alive during kernel execution!
                # Better approach: pass the CUtensorMap object directly instead of using addressof()
                # Convert to void pointer, then get pointer to pointer
                ptr_val = ctypes.c_void_p(arg)
                arg_data.append(ptr_val)  # Keep reference
                ptr_to_ptr = ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p)
                arg_ptrs.append(ptr_to_ptr)
            elif isinstance(arg, (float, np.floating)):
                # Float arguments
                c_val = ctypes.c_float(float(arg))
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        # Prepare kernel parameters
        # kernelParams should be a pointer to an array of void* pointers (void**)
        # According to CUDA API: kernelParams is void** (pointer to array of void*)
        # We need to explicitly cast the array to POINTER(ctypes.c_void_p)
        if len(arg_ptrs) > 0:
            kernel_params_array = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
            arg_data.append(kernel_params_array)  # Keep reference to prevent GC
            # Explicitly cast array to pointer type expected by cuLaunchKernel
            kernel_params = ctypes.cast(kernel_params_array, ctypes.POINTER(ctypes.c_void_p))
        else:
            kernel_params = None

        # Ensure correct GPU context is active before launch
        self._ensure_context()

        # Configure function attributes before launch
        self.configure_function(func, shared_mem_bytes, cluster_dim)

        # Launch kernel
        # Use cuLaunchKernelEx if cluster_dim is specified, otherwise use cuLaunchKernel
        if cluster_dim is not None and self._has_launch_kernel_ex:
            # Use cuLaunchKernelEx for cluster launch
            # Create launch config structure
            config = self.CUlaunchConfig()
            config.gridDimX = grid[0]
            config.gridDimY = grid[1]
            config.gridDimZ = grid[2]
            config.blockDimX = block[0]
            config.blockDimY = block[1]
            config.blockDimZ = block[2]
            config.sharedMemBytes = shared_mem_bytes
            config.hStream = stream or ctypes.c_void_p(0)
            config.numAttrs = 0
            config.attrs = None

            # Build launch attributes list
            attrs_list = []

            # Set cluster dimension attribute if needed
            if cluster_dim is not None and (cluster_dim[0] > 1 or cluster_dim[1] > 1 or cluster_dim[2] > 1):
                cluster_attr = self.CUlaunchAttribute()
                cluster_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                cluster_attr.value.clusterDim = (ctypes.c_uint * 3)(cluster_dim[0], cluster_dim[1], cluster_dim[2])
                attrs_list.append(cluster_attr)
                arg_data.append(cluster_attr)  # Keep reference to prevent GC

            if attrs_list:
                attrs_array = (self.CUlaunchAttribute * len(attrs_list))(*attrs_list)
                config.attrs = ctypes.cast(attrs_array, ctypes.POINTER(self.CUlaunchAttribute))
                config.numAttrs = len(attrs_list)
                arg_data.append(attrs_array)  # Keep reference to prevent GC

            arg_data.append(config)  # Keep reference to prevent GC

            err = self._libcuda.cuLaunchKernelEx(ctypes.byref(config), func, kernel_params,  # kernelParams
                                                 None,  # extra
                                                 )
            self._check_error(err, "cuLaunchKernelEx")
        else:
            # Use regular cuLaunchKernel (no cluster support)
            if cluster_dim is not None:
                print(
                    f"[DEBUG] Warning: cluster_dim={cluster_dim} specified but cuLaunchKernelEx not available, ignoring cluster config",
                    flush=True)

            err = self._libcuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],  # gridDim
                                               block[0], block[1], block[2],  # blockDim
                                               shared_mem_bytes,  # sharedMemBytes
                                               stream or ctypes.c_void_p(0),  # stream
                                               kernel_params,  # kernelParams - pointer to array of void* pointers
                                               None,  # extra
                                               )
            self._check_error(err, "cuLaunchKernel")

    def synchronize(self, stream: Optional[CUstream] = None):
        """Synchronize CUDA stream."""
        if stream:
            err = self._libcuda.cuStreamSynchronize(stream)
            self._check_error(err, "cuStreamSynchronize")
        elif TORCH_AVAILABLE:
            torch.cuda.synchronize()


class KernelLauncher:
    """Helper class to launch kernels with PyTorch integration."""

    def __init__(self, runtime: CUDARuntime, module: CUmodule, kernel_name: str, shared_mem_bytes: int = 0,
                 cubin_data: Optional[bytes] = None):
        self.runtime = runtime
        self.module = module
        self.kernel_name = kernel_name
        self.shared_mem_bytes = shared_mem_bytes
        self.cubin_data = cubin_data  # Store cubin data for symbol enumeration
        self._func = None

    @property
    def func(self) -> CUfunction:
        """Lazy load function."""
        if self._func is None:
            self._func = self.runtime.get_function(self.module, self.kernel_name, self.cubin_data)
            self.runtime.configure_function(self._func, self.shared_mem_bytes)
        return self._func

    def __call__(
        self,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        *args,
        stream: Optional[Any] = None,
        cluster_dim: Optional[Tuple[int, int, int]] = None,
    ):
        """Launch kernel.
        
        Parameters
        ----------
        grid, block : Tuple[int, int, int]
            Grid and block dimensions.
        *args : Any
            Kernel arguments.
        stream : Optional[Any]
            CUDA stream.
        cluster_dim : Optional[Tuple[int, int, int]]
            Cluster dimensions for SM90+ cluster launch.
"""
        # Convert PyTorch stream if needed
        cuda_stream = None
        if stream is not None:
            if TORCH_AVAILABLE and isinstance(stream, torch.cuda.Stream):
                cuda_stream = ctypes.c_void_p(stream.cuda_stream)
            else:
                cuda_stream = stream

        self.runtime.launch_kernel(
            self.func,
            grid,
            block,
            args,
            self.shared_mem_bytes,
            cuda_stream,
            cluster_dim=cluster_dim,
        )
