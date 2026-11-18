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
"""CUDA Driver runtime wrapper.

RF04 / RF01 integration contract (see docs/refactor-plan.md):

* ``CUDARuntime(*, context=None, device=None)`` owns a modern Driver context.
  The standalone no-Torch path creates it with the versioned
  ``cuCtxCreate_v2`` API (never the unversioned ``cuCtxCreate``) and validates
  the setup via checked ``cuCtxSetCurrent``/``cuCtxGetCurrent`` before use.
  An explicit ``context=`` is *borrowed*: made current but never released.
* ``CUDARuntime.close()`` / context-manager support unload owned modules
  first and release/retain-balance the owned context exactly once.
  ``load_module(..., borrowed=True)`` tracks a handle owned elsewhere without
  ever unloading it; a module is never unloaded while a live
  :class:`KernelLauncher` depends on it.
* ``synchronize(stream=None)`` performs a checked ``cuCtxSynchronize`` for the
  default context and a checked ``cuStreamSynchronize`` for an explicit
  stream, independent of whether Torch is installed.
* :meth:`CUDARuntime.launch_kernel` and :class:`KernelLauncher` accept a
  trailing optional ``signature`` (``runtime.abi.KernelSignature``). When
  supplied, arguments are marshalled by ``runtime.abi.pack_arguments`` and
  value-based guessing can never override it; omitting it selects the
  explicit legacy untyped path.

Integration hook for the cache worker (``runtime/kernel.py``, owned
separately and intentionally unchanged here): ``CompiledKernel`` should store
an optional trailing ``signature`` from its construction/artifact and pass it
positionally/by keyword into ``KernelLauncher(..., signature=sig)``; the
launcher then forwards it to ``launch_kernel`` automatically. Raw-source
callers that pass no signature keep the legacy path unchanged.
"""

import ctypes
import sys
import os
import shutil
import subprocess
import tempfile
import threading
from typing import Optional, Tuple, List, Any, Dict
import numpy as np

from little_kernel.runtime import abi as launch_abi
from little_kernel.runtime.lifecycle import ContextOwnership, ManagedContext, ManagedModule, ResourceClosedError

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
CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14

# CUDA Launch Attribute IDs (for cluster launch)
CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4


class CUDARuntime:
    """CUDA Driver API runtime wrapper."""

    def __init__(self, *, context: Optional[Any] = None, device: Optional[int] = None):
        """Open the Driver and establish a usable, owned CUDA context.

        Parameters
        ----------
        context:
            Optional pre-existing context handle. An integer/``c_void_p`` is
            *borrowed* -- made current for Driver calls but never released by
            this runtime. When omitted, Torch's primary context is retained
            (when available) or a fresh modern context is created.
        device:
            Device ordinal. Defaults to the Torch current device or 0.
        """
        self._libcuda = None
        self._ctx = None
        self._device = None
        self._managed_context: Optional[ManagedContext] = None
        self._managed_modules: Dict[int, ManagedModule] = {}
        self._modules_lock = threading.RLock()
        self._closed = False
        self._load_library()
        self._init_context(context=context, device=device)

    # -- lifecycle ----------------------------------------------------------

    @property
    def managed_context(self) -> Optional[ManagedContext]:
        """The owned/borrowed context wrapper (RF04)."""
        return self._managed_context

    @property
    def device(self) -> int:
        """The selected logical CUDA device ordinal."""
        return self._device

    def _release_retained_primary(self, managed: ManagedContext) -> None:
        release = getattr(self._libcuda, self._primary_ctx_release_name)
        err = release(CUdevice(managed.device))
        self._check_error(err, self._primary_ctx_release_name)

    def _release_created_context(self, managed: ManagedContext) -> None:
        err = self._libcuda.cuCtxDestroy_v2(CUcontext(managed.handle))
        self._check_error(err, "cuCtxDestroy_v2")

    def _validate_bound_context(self, managed: ManagedContext) -> None:
        """Validate that the context is actually usable as current.

        Creation/retain can report success while the context never becomes
        current on a mixed-version Driver; this check fails fast with the
        real error instead of surfacing CUDA_ERROR_INVALID_CONTEXT (201) at
        the first later allocation.
        """
        err = self._libcuda.cuCtxSetCurrent(CUcontext(managed.handle))
        self._check_error(err, "cuCtxSetCurrent")
        current = CUcontext()
        err = self._libcuda.cuCtxGetCurrent(ctypes.byref(current))
        self._check_error(err, "cuCtxGetCurrent")
        if not current.value or int(current.value or 0) != managed.handle:
            raise RuntimeError("CUDA context setup validation failed: the retained/created "
                               f"context 0x{managed.handle:x} did not become current "
                               f"(reported 0x{int(current.value or 0):x}); refusing to run "
                               "on an unbound context")

    def close(self) -> None:
        """Release owned modules and the owned context exactly once.

        Ordering and failure semantics (RF04):

        * If any module still has live launcher dependents, close refuses
          *before* unloading a module or destroying the owning context, so
          live launchers are never left dangling.
        * Owned modules are released first; the owned context is released
          only once every module has been released. A borrowed context is
          never released.
        * On any unload/destroy failure, ownership tracking is preserved and
          the runtime stays usable/open so ``close`` can be retried after
          the caller fixes the problem (e.g. detaching the live launcher).
          Successfully released resources are not released twice.
        """
        if self._closed:
            return

        # Phase 1: preflight live dependents before touching anything.
        with self._modules_lock:
            live = [(handle, module.live_users)
                    for handle, module in self._managed_modules.items()
                    if module.live_users]
        if live:
            detail = ", ".join(f"0x{handle:x}: {users} launcher(s)" for handle, users in live)
            raise RuntimeError("Refusing to close CUDARuntime while live launchers still "
                               f"depend on loaded modules ({detail}); detach/close them "
                               "first. The context and modules remain owned and usable.")

        errors = []

        # Phase 2: release modules. Failed unloads stay tracked for retry.
        with self._modules_lock:
            pending = [module for module in self._managed_modules.values() if not module.closed]
        for module in pending:
            try:
                module.close()
            except Exception as exc:
                errors.append(exc)

        with self._modules_lock:
            for handle in list(self._managed_modules):
                if self._managed_modules[handle].closed:
                    self._managed_modules.pop(handle, None)
            modules_remaining = bool(self._managed_modules)

        # Phase 3: release the context only after every module is gone.
        context_failed = False
        if (not modules_remaining and self._managed_context is not None and not self._managed_context.closed):
            try:
                self._managed_context.close()
            except Exception as exc:
                context_failed = True
                errors.append(exc)

        if errors:
            # Ownership tracking is intentionally retained: close is
            # retryable and no owned resource was discarded on failure.
            raise RuntimeError("Errors while closing CUDARuntime (resources remain owned "
                               "and close() may be retried): " + "; ".join(repr(e) for e in errors))

        if not context_failed:
            self._managed_context = None
            self._ctx = None
            self._closed = True

    def __enter__(self) -> "CUDARuntime":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

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
            if self._libcuda is None and sys.platform != "win32" and (ldconfig := shutil.which('ldconfig')):
                try:
                    libs_output = subprocess.check_output([ldconfig, "-p"], stderr=subprocess.DEVNULL).decode()
                    # Parse the loader's resolved library path after the arrow.
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

        # Modern (versioned) context APIs. The unversioned cuCtxCreate uses a
        # legacy handle/flag ABI and mixing it with modern calls such as
        # cuMemAlloc_v2 leaves the new context unbound (observed on a root
        # H800 probe: cuMemAlloc_v2 -> CUDA_ERROR_INVALID_CONTEXT 201). We
        # therefore bind the v2 entry point exclusively.
        self._libcuda.cuCtxCreate_v2.argtypes = [ctypes.POINTER(CUcontext), ctypes.c_uint, CUdevice]
        self._libcuda.cuCtxCreate_v2.restype = CUresult

        self._libcuda.cuCtxDestroy_v2.argtypes = [CUcontext]
        self._libcuda.cuCtxDestroy_v2.restype = CUresult

        # Release matching the cuDevicePrimaryCtxRetain retain call.
        release_name = ("cuDevicePrimaryCtxRelease_v2"
                        if hasattr(self._libcuda, "cuDevicePrimaryCtxRelease_v2") else "cuDevicePrimaryCtxRelease")
        self._primary_ctx_release_name = release_name

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

        # Default-context synchronization: the standalone Driver path must not
        # depend on Torch being installed.
        self._libcuda.cuCtxSynchronize.argtypes = []
        self._libcuda.cuCtxSynchronize.restype = CUresult

        # Context management for multi-GPU
        self._libcuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(CUcontext), CUdevice]
        self._libcuda.cuDevicePrimaryCtxRetain.restype = CUresult
        getattr(self._libcuda, release_name).argtypes = [CUdevice]
        getattr(self._libcuda, release_name).restype = CUresult

        # Module unload paired with cuModuleLoadData.
        self._libcuda.cuModuleUnload.argtypes = [CUmodule]
        self._libcuda.cuModuleUnload.restype = CUresult

        # Current-context management: cuCtxSetCurrent is always error-checked;
        # cuCtxGetCurrent validates that a freshly created/retained context is
        # actually bound (mixed-version contexts can report success on create
        # yet fail to become current).
        self._libcuda.cuCtxSetCurrent.argtypes = [CUcontext]
        self._libcuda.cuCtxSetCurrent.restype = CUresult
        self._libcuda.cuCtxGetCurrent.argtypes = [ctypes.POINTER(CUcontext)]
        self._libcuda.cuCtxGetCurrent.restype = CUresult

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
            self._libcuda.cuOccupancyMaxActiveClusters.argtypes = [
                ctypes.POINTER(ctypes.c_int),
                CUfunction,
                ctypes.POINTER(CUlaunchConfig),
            ]
            self._libcuda.cuOccupancyMaxActiveClusters.restype = CUresult
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

    def _init_context(self, context: Optional[Any] = None, device: Optional[int] = None):
        """Initialize CUDA context with explicit ownership (RF04)."""
        # Explicitly borrowed context: never create or release one ourselves.
        if context is not None:
            handle = int(context.value if hasattr(context, "value") else context)
            self._device = int(device if device is not None else 0)
            err = self._libcuda.cuInit(0)
            self._check_error(err, "cuInit")
            managed = ManagedContext(
                handle,
                self._device,
                ContextOwnership.BORROWED,
                validate=self._validate_bound_context,
                label="explicitly borrowed context",
            )
            self._managed_context = managed
            self._ctx = CUcontext(managed.handle)
            return

        # Retain PyTorch's primary context when it owns CUDA. We retain so the
        # backing owner stays alive for the lifetime of this runtime, and
        # release the matching retain exactly once on close.
        if TORCH_AVAILABLE and torch.cuda.is_available():
            self._device = (int(device) if device is not None else torch.cuda.current_device())
            dev = CUdevice()
            err = self._libcuda.cuDeviceGet(ctypes.byref(dev), self._device)
            self._check_error(err, "cuDeviceGet")
            ctx = CUcontext()
            err = self._libcuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev)
            self._check_error(err, "cuDevicePrimaryCtxRetain")
            managed = ManagedContext(
                int(ctx.value or 0),
                self._device,
                ContextOwnership.PRIMARY_RETAINED,
                release=self._release_retained_primary,
                validate=self._validate_bound_context,
                label="torch primary context",
            )
            self._managed_context = managed
            self._ctx = CUcontext(managed.handle)
            return

        # Standalone path: create a *modern* (v2) context and validate that it
        # is really bound before any allocation/launch.
        err = self._libcuda.cuInit(0)
        self._check_error(err, "cuInit")

        self._device = int(device if device is not None else 0)
        dev = CUdevice()
        err = self._libcuda.cuDeviceGet(ctypes.byref(dev), self._device)
        self._check_error(err, "cuDeviceGet")

        ctx = CUcontext()
        err = self._libcuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev)
        self._check_error(err, "cuCtxCreate_v2")
        managed = ManagedContext(
            int(ctx.value or 0),
            self._device,
            ContextOwnership.CREATED,
            release=self._release_created_context,
            validate=self._validate_bound_context,
            label="standalone cuCtxCreate_v2 context",
        )
        self._managed_context = managed
        self._ctx = CUcontext(managed.handle)

    def _ensure_context(self):
        """Ensure this runtime's CUDA context is the current one."""
        if getattr(self, "_closed", False):
            raise ResourceClosedError("this CUDARuntime has already been closed")
        if self._managed_context is not None:
            if self._managed_context.closed:
                raise ResourceClosedError("this CUDARuntime context has already been closed")
            err = self._libcuda.cuCtxSetCurrent(CUcontext(self._managed_context.handle))
            self._check_error(err, "cuCtxSetCurrent")
        elif self._ctx:
            err = self._libcuda.cuCtxSetCurrent(self._ctx)
            self._check_error(err, "cuCtxSetCurrent")

    def _check_error(self, err: int, func_name: str):
        """Check CUDA error and raise exception if needed."""
        if err != CUDA_SUCCESS:
            error_str = ctypes.c_char_p()
            self._libcuda.cuGetErrorString(err, ctypes.byref(error_str))
            raise RuntimeError(f"CUDA error in {func_name}: {error_str.value.decode()}")

    def _unload_module_checked(self, handle: int) -> None:
        err = self._libcuda.cuModuleUnload(CUmodule(handle))
        self._check_error(err, "cuModuleUnload")

    def load_module(self, cubin: bytes, *, borrowed: bool = False):
        """Load CUBIN module into the correct GPU context.

        Returns the raw ``CUmodule`` handle (legacy-compatible). The runtime
        additionally records an :class:`ManagedModule` for it so that the
        matching ``cuModuleUnload`` is exactly-once and refuses to unload
        while live launchers still depend on it. ``borrowed=True`` registers
        a handle owned elsewhere: dependencies are tracked but it is never
        unloaded by this runtime.
        """
        self._ensure_context()
        module = CUmodule()
        err = self._libcuda.cuModuleLoadData(ctypes.byref(module), cubin)
        self._check_error(err, "cuModuleLoadData")
        handle = int(module.value or 0)
        managed = ManagedModule(
            handle,
            unload=None if borrowed else self._unload_module_checked,
            borrowed=borrowed,
            label=f"{len(cubin)}-byte cubin",
        )
        with self._modules_lock:
            self._managed_modules[handle] = managed
        return module

    def _managed_module(self, module: Any) -> Optional[ManagedModule]:
        handle = int(module.value if hasattr(module, "value") else module)
        with self._modules_lock:
            return self._managed_modules.get(handle)

    def register_module_user(self, module: Any) -> None:
        """Mark a live launcher as depending on ``module``."""
        managed = self._managed_module(module)
        if managed is None:
            # Module loaded outside this runtime: treat it as borrowed so we
            # can still track the dependency, but never unload it ourselves.
            handle = int(module.value if hasattr(module, "value") else module)
            managed = ManagedModule(handle, borrowed=True, label="external")
            with self._modules_lock:
                self._managed_modules.setdefault(handle, managed)
        managed.attach_user()

    def release_module_user(self, module: Any) -> None:
        """Release a dependency previously registered for ``module``."""
        managed = self._managed_module(module)
        if managed is not None:
            managed.detach_user()

    def unload_module(self, module: Any) -> bool:
        """Unload a loaded module exactly once (checked).

        Refuses while live launchers still depend on it. Returns ``True`` if
        this call performed the unload, ``False`` if it was already unloaded
        or the module is borrowed. A live-user refusal or a failed unload
        leaves ownership tracking intact so the call can be retried.
        """
        handle = int(module.value if hasattr(module, "value") else module)
        with self._modules_lock:
            managed = self._managed_modules.get(handle)
        if managed is None:
            # Unknown raw handle: perform the checked unload directly so the
            # public API stays useful for externally loaded modules.
            self._unload_module_checked(handle)
            return True
        released = managed.close()
        if released:
            with self._modules_lock:
                self._managed_modules.pop(handle, None)
        return released

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
                    cuobjdump_paths = []
                    for variable in ('CUDA_HOME', 'CUDA_PATH'):
                        if root := os.environ.get(variable):
                            cuobjdump_paths.append(os.path.join(root, 'bin', 'cuobjdump'))
                    if tool := shutil.which(os.environ.get('CUOBJDUMP', 'cuobjdump')):
                        cuobjdump_paths.insert(0, tool)
                    if nvcc := shutil.which('nvcc'):
                        cuobjdump_paths.append(os.path.join(os.path.dirname(nvcc), 'cuobjdump'))

                    cuobjdump = None
                    for path in cuobjdump_paths:
                        if os.path.isfile(path) and os.access(path, os.X_OK):
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

    def configure_function(self, func: CUfunction, shared_mem_bytes: int, cluster_shape: Optional[Tuple[int, int,
                                                                                                        int]] = None,
                           allow_non_portable_cluster: bool = False):
        """Configure function attributes for shared memory and cluster launch.
        
        Sets MaxDynamicSharedMemorySize and optionally NonPortableClusterSizeAllowed.
        """
        # Non-portable cluster sizes are opt-in; ordinary cluster launches do
        # not require this attribute.
        if allow_non_portable_cluster:
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

    def max_active_clusters(
        self,
        func: CUfunction,
        block: Tuple[int, int, int],
        shared_mem_bytes: int,
        cluster_shape: Tuple[int, int, int],
    ) -> int:
        """Return the maximum number of simultaneously resident clusters."""
        if not self._has_launch_kernel_ex:
            raise RuntimeError("cluster occupancy query requires cuLaunchKernelEx support")

        self.configure_function(
            func,
            shared_mem_bytes,
            cluster_shape=cluster_shape,
            allow_non_portable_cluster=True,
        )
        config = self.CUlaunchConfig()
        config.gridDimX, config.gridDimY, config.gridDimZ = cluster_shape
        config.blockDimX, config.blockDimY, config.blockDimZ = block
        config.sharedMemBytes = shared_mem_bytes
        config.hStream = ctypes.c_void_p(0)

        cluster_attr = self.CUlaunchAttribute()
        cluster_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        cluster_attr.value.clusterDim = (ctypes.c_uint * 3)(*cluster_shape)
        attrs = (self.CUlaunchAttribute * 1)(cluster_attr)
        config.attrs = ctypes.cast(attrs, ctypes.POINTER(self.CUlaunchAttribute))
        config.numAttrs = 1

        result = ctypes.c_int()
        err = self._libcuda.cuOccupancyMaxActiveClusters(
            ctypes.byref(result),
            func,
            ctypes.byref(config),
        )
        self._check_error(err, "cuOccupancyMaxActiveClusters")
        return int(result.value)

    @staticmethod
    def _adapt_pointer_value(value: Any) -> Any:
        """Adapt a framework buffer to a raw device address (POINTER param).

        Framework adaptation happens at the runtime boundary only; the ABI
        module marshals raw addresses and ctypes values.
        """
        if hasattr(value, "data_ptr"):  # PyTorch tensor
            return int(value.data_ptr())
        if isinstance(value, np.ndarray):
            if not value.flags.c_contiguous:
                value = np.ascontiguousarray(value)
            return int(value.ctypes.data)
        return value

    def _pack_by_signature(self, signature: "launch_abi.KernelSignature", args: List[Any]):
        """Marshal ``args`` against the known signature (RF01 integration).

        A known signature always wins: no value-based guessing or silent
        narrowing. Framework buffers become raw addresses only for
        parameters declared as pointers.
        """
        adapted = []
        for param, value in zip(signature.parameters, args):
            if param.kind == launch_abi.POINTER:
                adapted.append(self._adapt_pointer_value(value))
            else:
                adapted.append(value)
        return launch_abi.pack_arguments(signature, adapted)

    def _legacy_pack_arguments(self, args: List[Any]):
        """Explicit legacy value-based marshalling for untyped callers.

        Used only when no signature is supplied. A known ABI can never be
        overridden by this value-based guessing (RF01). Its known legacy
        limitations (integer width inferred from magnitude, every float
        forced to float32, positive ints usable as raw addresses) are
        intentionally preserved; typed callers must pass a
        ``KernelSignature``.
        """
        arg_ptrs = []
        arg_data = []

        for arg in args:
            # __grid_constant__ structures (e.g. TMA descriptors): pointer
            # to the structure bytes.
            if isinstance(arg, ctypes.Structure):
                arg_data.append(arg)
                arg_ptrs.append(ctypes.cast(ctypes.byref(arg), ctypes.c_void_p))
            elif isinstance(arg, (int, np.integer)):
                val = int(arg)
                if -2**31 <= val < 2**31:
                    c_val = ctypes.c_int32(val)
                else:
                    c_val = ctypes.c_int64(val)
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
            elif hasattr(arg, "data_ptr"):  # PyTorch tensor
                ptr_val = ctypes.c_void_p(arg.data_ptr())
                arg_data.append(ptr_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p))
            elif isinstance(arg, np.ndarray):
                if not arg.flags.c_contiguous:
                    arg = np.ascontiguousarray(arg)
                ptr_val = ctypes.c_void_p(arg.ctypes.data)
                arg_data.append(ptr_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(ptr_val), ctypes.c_void_p))
            elif isinstance(arg, ctypes.c_void_p):
                arg_data.append(arg)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(arg), ctypes.c_void_p))
            elif isinstance(arg, (float, np.floating)):
                # Legacy: every float is stored as float32.
                c_val = ctypes.c_float(float(arg))
                arg_data.append(c_val)
                arg_ptrs.append(ctypes.cast(ctypes.pointer(c_val), ctypes.c_void_p))
            else:
                raise TypeError(f"Unsupported argument type: {type(arg)}")

        return arg_ptrs, arg_data

    def launch_kernel(
        self,
        func: CUfunction,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        args: List[Any],
        shared_mem_bytes: int = 0,
        stream: Optional[CUstream] = None,
        cluster_shape: Optional[Tuple[int, int, int]] = None,
        signature: Optional["launch_abi.KernelSignature"] = None,
    ):
        """Launch a CUDA kernel.

        ``signature`` is a trailing optional
        :class:`~little_kernel.runtime.abi.KernelSignature`. When supplied,
        arguments are validated/marshalled by the typed ABI
        (``pack_arguments``) and value-based guessing can never override it;
        when omitted, the explicit legacy value-based path serves untyped
        raw-source callers.
        """
        if cluster_shape is not None:
            if len(cluster_shape) != 3 or any(dim <= 0 for dim in cluster_shape):
                raise ValueError("cluster_shape must contain three positive dimensions, "
                                 f"got {cluster_shape}")
            if any(grid_dim % cluster_dim for grid_dim, cluster_dim in zip(grid, cluster_shape)):
                raise ValueError(f"grid {grid} must be divisible by cluster_shape "
                                 f"{cluster_shape}")
            if cluster_shape[0] * cluster_shape[1] * cluster_shape[2] > 16:
                raise ValueError(f"cluster_shape={cluster_shape} exceeds the supported "
                                 "16-CTA cluster limit")
            if (any(dim > 1 for dim in cluster_shape) and not self._has_launch_kernel_ex):
                raise RuntimeError(f"cluster_shape={cluster_shape} requires "
                                   "cuLaunchKernelEx support")

        # Every marshalling object must stay alive until the Driver returns.
        arg_data: List[Any] = []
        if signature is not None:
            if not isinstance(signature, launch_abi.KernelSignature):
                raise TypeError("signature must be a little_kernel.runtime.abi."
                                f"KernelSignature, got {type(signature).__name__}")
            packed = self._pack_by_signature(signature, list(args))
            arg_data.append(packed)  # retain storage + pointers
            kernel_params = packed.kernel_params
        else:
            # Explicit legacy path for untyped callers only.
            arg_ptrs, legacy_data = self._legacy_pack_arguments(list(args))
            arg_data.extend(legacy_data)
            if len(arg_ptrs) > 0:
                kernel_params_array = (ctypes.c_void_p * len(arg_ptrs))(*arg_ptrs)
                arg_data.append(kernel_params_array)
                kernel_params = ctypes.cast(kernel_params_array, ctypes.POINTER(ctypes.c_void_p))
            else:
                kernel_params = None

        # Ensure correct GPU context is active before launch
        self._ensure_context()

        # Configure function attributes before launch
        self.configure_function(func, shared_mem_bytes, cluster_shape)

        # Launch kernel
        # Use cuLaunchKernelEx only for multi-block cluster (>1 in any dim)
        use_cluster_launch = (cluster_shape is not None and self._has_launch_kernel_ex
                              and any(dim > 1 for dim in cluster_shape))
        if use_cluster_launch:
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
            if cluster_shape is not None and any(dim > 1 for dim in cluster_shape):
                cluster_attr = self.CUlaunchAttribute()
                cluster_attr.id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                cluster_attr.value.clusterDim = (ctypes.c_uint * 3)(cluster_shape[0], cluster_shape[1],
                                                                    cluster_shape[2])
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
            err = self._libcuda.cuLaunchKernel(func, grid[0], grid[1], grid[2],  # gridDim
                                               block[0], block[1], block[2],  # blockDim
                                               shared_mem_bytes,  # sharedMemBytes
                                               stream or ctypes.c_void_p(0),  # stream
                                               kernel_params,  # kernelParams - pointer to array of void* pointers
                                               None,  # extra
                                               )
            self._check_error(err, "cuLaunchKernel")

    def synchronize(self, stream: Optional[CUstream] = None):
        """Synchronize the explicit stream or the current CUDA context.

        An explicit stream uses a checked ``cuStreamSynchronize``; with no
        stream the current context uses a checked ``cuCtxSynchronize`` so
        synchronization is real even on the standalone (no-Torch) Driver
        path rather than being a silent no-op.
        """
        stream_handle = self._stream_handle(stream)
        self._ensure_context()
        if stream_handle:
            err = self._libcuda.cuStreamSynchronize(CUstream(stream_handle))
            self._check_error(err, "cuStreamSynchronize")
            return
        err = self._libcuda.cuCtxSynchronize()
        self._check_error(err, "cuCtxSynchronize")

    @staticmethod
    def _stream_handle(stream: Optional[Any]) -> Optional[int]:
        if stream is None:
            return None
        if TORCH_AVAILABLE and isinstance(stream, torch.cuda.Stream):
            return int(stream.cuda_stream)
        if hasattr(stream, "value"):
            return int(stream.value or 0)
        return int(stream)


class KernelLauncher:
    """Helper class to launch kernels with PyTorch integration."""

    def __init__(self, runtime: CUDARuntime, module: CUmodule, kernel_name: str, shared_mem_bytes: int = 0,
                 cubin_data: Optional[bytes] = None, cluster_shape: Optional[Tuple[int, int, int]] = None,
                 allow_non_portable_cluster: bool = False, signature: Optional["launch_abi.KernelSignature"] = None):
        self.runtime = runtime
        self.module = module
        self.kernel_name = kernel_name
        self.shared_mem_bytes = shared_mem_bytes
        self.cubin_data = cubin_data  # Store cubin data for symbol enumeration
        self.cluster_shape = cluster_shape
        self.allow_non_portable_cluster = allow_non_portable_cluster
        # Trailing optional typed ABI (RF01 integration). A launcher built
        # with a signature always marshals from it and never falls back to
        # value-based guessing.
        if signature is not None and not isinstance(signature, launch_abi.KernelSignature):
            raise TypeError("signature must be a little_kernel.runtime.abi."
                            f"KernelSignature, got {type(signature).__name__}")
        self.signature = signature
        self._func = None
        self._closed = False
        # The loaded module must outlive this launcher's functions; register
        # a dependency so the module cannot be unloaded while we are live.
        self.runtime.register_module_user(self.module)

    def close(self) -> None:
        """Release this launcher's dependency on its module exactly once."""
        if self._closed:
            return
        self._closed = True
        self.runtime.release_module_user(self.module)

    def __enter__(self) -> "KernelLauncher":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.close()
        return False

    @property
    def func(self) -> CUfunction:
        """Lazy load function."""
        if self._closed:
            raise ResourceClosedError(f"launcher for {self.kernel_name!r} has been closed")
        if self._func is None:
            self._func = self.runtime.get_function(self.module, self.kernel_name, self.cubin_data)
            self.runtime.configure_function(
                self._func,
                self.shared_mem_bytes,
                cluster_shape=self.cluster_shape,
                allow_non_portable_cluster=self.allow_non_portable_cluster,
            )
        return self._func

    def max_active_clusters(self, block: Tuple[int, int, int]) -> int:
        if self.cluster_shape is None:
            raise RuntimeError("kernel has no cluster launch configuration")
        return self.runtime.max_active_clusters(
            self.func,
            block,
            self.shared_mem_bytes,
            self.cluster_shape,
        )

    def __call__(
        self,
        grid: Tuple[int, int, int],
        block: Tuple[int, int, int],
        *args,
        stream: Optional[Any] = None,
        cluster_shape: Optional[Tuple[int, int, int]] = None,
        signature: Optional["launch_abi.KernelSignature"] = None,
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
        cluster_shape : Optional[Tuple[int, int, int]]
            Cluster dimensions for SM90+ cluster launch.
        signature : Optional[KernelSignature]
            Trailing per-call typed ABI. When omitted the launcher's bound
            ``self.signature`` is used. A supplied signature always wins over
            value-based guessing and is validated as a KernelSignature.
        """
        if self._closed:
            raise ResourceClosedError(f"launcher for {self.kernel_name!r} has been closed")
        # Resolve effective signature: explicit per-call signature takes
        # precedence; otherwise the launcher's bound signature applies. A
        # known signature is never overridden by value-based guessing.
        effective_signature = signature if signature is not None else self.signature
        if (effective_signature is not None and not isinstance(effective_signature, launch_abi.KernelSignature)):
            raise TypeError("signature must be a little_kernel.runtime.abi."
                            f"KernelSignature, got {type(effective_signature).__name__}")

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
            cluster_shape=cluster_shape,
            signature=effective_signature,
        )
