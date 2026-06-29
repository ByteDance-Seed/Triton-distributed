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
import os
from pathlib import Path
from typing import Dict, Optional, TypeVar, Callable, Union, Iterable
import tempfile
import subprocess
import signal
import re
import warnings

import triton
from triton.runtime.errors import PTXASError
from triton.runtime.jit import JITFunction, KernelInterface
from triton import knobs
from triton_dist.utils import is_ascend, is_cuda, is_hip, is_maca, HIP_CHECK

T = TypeVar("T")

# Environment variable for CGA (Cooperative Grid Array) cluster size
# Format: "x,y,z" or "x" (defaults y=1, z=1)
# Example: TRITON_DIST_CGA_CLUSTER_SIZE="2,1,1" or TRITON_DIST_CGA_CLUSTER_SIZE="2"
_CGA_CLUSTER_SIZE_ENV = "TRITON_DIST_CGA_CLUSTER_SIZE"


def _is_power_of_two(n: int) -> bool:
    """Check if n is a power of 2 (including 1)."""
    return n > 0 and (n & (n - 1)) == 0


def _parse_cga_cluster_size() -> Optional[tuple]:
    """
    Parse TRITON_DIST_CGA_CLUSTER_SIZE environment variable.
    
    Returns:
        tuple: (cluster_x, cluster_y, cluster_z) if set and valid, None otherwise.
        
    Constraints:
        - Each dimension must be a power of 2 (1, 2, 4, 8, ...)
        - Total product (x * y * z) must not exceed 16
        
    Examples:
        "2" -> (2, 1, 1)
        "2,1,1" -> (2, 1, 1)
        "2,2,1" -> (2, 2, 1)
        "4,4,1" -> (4, 4, 1)  # total = 16, valid
        "4,4,2" -> None       # total = 32 > 16, invalid
    """
    env_value = os.environ.get(_CGA_CLUSTER_SIZE_ENV)
    if not env_value:
        return None

    try:
        parts = [int(x.strip()) for x in env_value.split(",")]
        if len(parts) == 1:
            cluster_dims = (parts[0], 1, 1)
        elif len(parts) == 2:
            cluster_dims = (parts[0], parts[1], 1)
        elif len(parts) == 3:
            cluster_dims = (parts[0], parts[1], parts[2])
        else:
            warnings.warn(f"Invalid {_CGA_CLUSTER_SIZE_ENV} format: '{env_value}'. "
                          f"Expected 'x', 'x,y', or 'x,y,z'. Ignoring.")
            return None

        # Validate each dimension is a power of 2
        for i, dim in enumerate(cluster_dims):
            if not _is_power_of_two(dim):
                warnings.warn(f"Invalid {_CGA_CLUSTER_SIZE_ENV} value: '{env_value}'. "
                              f"Dimension {i} ({dim}) is not a power of 2. Ignoring.")
                return None

        # Validate total product does not exceed 16
        total = cluster_dims[0] * cluster_dims[1] * cluster_dims[2]
        if total > 16:
            warnings.warn(f"Invalid {_CGA_CLUSTER_SIZE_ENV} value: '{env_value}'. "
                          f"Total cluster size ({total}) exceeds maximum of 16. Ignoring.")
            return None

        return cluster_dims

    except ValueError as e:
        warnings.warn(f"Invalid {_CGA_CLUSTER_SIZE_ENV} value: '{env_value}'. "
                      f"Expected integers. Error: {e}. Ignoring.")
        return None


def shmem_kernel_module_init_hook(*args, **kwargs) -> None:
    key = kwargs["key"]
    jit_function = kwargs["fn"].jit_function
    device = kwargs["compile"]["device"]
    kernel_cache = jit_function.device_caches[device][0]
    kernel = kernel_cache.get(key, None)
    assert kernel is not None, f"kernel is None for key = {key}"
    kernel._init_handles()
    kernel_module = kernel.module

    if is_cuda():
        from triton_dist.utils import is_shmem_initialized
        has_shmem = "nvshmem" in kernel.asm['ptx']
        if has_shmem and is_shmem_initialized():
            import nvshmem.bindings.nvshmem as pynvshmem
            pynvshmem.cumodule_init(kernel_module)
    elif is_hip():
        import torch
        from hip import hip
        from triton_dist.utils import get_shmem_backend

        backend = get_shmem_backend()

        if backend == 'rocshmem':
            import pyrocshmem
            res = hip.hipModuleGetGlobal(kernel_module, b"ROCSHMEM_CTX_DEFAULT")
            # dptr, bytes = res[1], res[2]
            if res[0] == hip.hipError_t.hipSuccess:
                """
                    typedef struct rocshmem_ctx{
                        void *ctx_opaque;
                        void *team_opaque;
                    } rocshmem_ctx_t;
                    pyrocshmem.rocshmem_get_device_ctx only return the `ctx_opaque`.
                    `ROCSHMEM_CTX_DEFAULT` is a `rocshmem_ctx_t` struct, but only the `ctx_opaque` field needs to be updated on the device side.
                    (equal to `libshmem_device.set_rocshmem_ctx(ctx)` in the kernel)
                """
                ctx_opaque_bytes = 8  # assuming 64-bit pointer
                # get the host address of the `ctx_opaque` pointer.
                ctx = pyrocshmem.rocshmem_get_device_ctx()
                ctx_tensor = torch.tensor([ctx], dtype=torch.int64)
                # update the device `ROCSHMEM_CTX_DEFAULT` struct's `ctx_opaque` field in the kernel module.
                cp_res = hip.hipMemcpy(res[1], ctx_tensor.data_ptr(), ctx_opaque_bytes,
                                       hip.hipMemcpyKind.hipMemcpyHostToDevice)
                HIP_CHECK(cp_res)
            else:
                hip.hipGetLastError()  # Discard the last error
        elif backend == 'mori_shmem':
            if "mori_shmem" in kernel.asm.get('llir', ''):
                import mori.shmem as mori_shmem
                mori_shmem.shmem_module_init(kernel_module)
    elif is_maca():
        if "mxshmem" in kernel.asm['ttir']:
            import triton.pymxshmem as pymxshmem
            pymxshmem.mxshmemx_mcmodule_init(kernel_module)
    elif is_ascend():
        pass
    else:
        raise ValueError("Unsupported device type for shmem kernel module init hook.")


def get_shmem_extern_lib() -> Dict[str, str]:
    if is_cuda():
        from triton_dist.nv_utils import NVSHMEMHelper
        use_wrapper = os.getenv("TRITON_DIST_SHMEM_WRAPPER") in ["1", "True", "true"]
        if use_wrapper:
            return {}
        if os.getenv("NVSHMEM_IBGDA_SUPPORT") in ["1", "True", "true"]:
            warnings.warn(
                "`NVSHMEM_IBGDA_SUPPORT` will be ignored when `TRITON_DIST_SHMEM_WRAPPER` is not set. Please set `TRITON_DIST_SHMEM_WRAPPER` to True to enable IBGDA."
            )
        nvshmem_home = Path(NVSHMEMHelper.get_nvshmem_home())
        nvshmem_device_lib = os.getenv("NVSHMEM_LIBDEVICE_PATH", None) or str(nvshmem_home / 'lib')
        nvshmem_device_lib = Path(nvshmem_device_lib)
        return {'libnvshmem_device': str(nvshmem_device_lib / 'libnvshmem_device.bc')}

    elif is_hip():
        import triton_dist
        from .utils import get_shmem_backend, _get_rocshmem_libdevice, _get_mori_shmem_libdevice

        libdevice_extra_lib = Path(triton_dist.__path__[0]) / "tools" / "compile" / "libdevice_extra.ll"
        backend = get_shmem_backend()

        if backend == 'rocshmem':
            rocshmem_lib = _get_rocshmem_libdevice()
            # func name need to contain the lib name
            extern_libs = {"rocshmem": str(rocshmem_lib), "extra": str(libdevice_extra_lib)}
        elif backend == 'mori_shmem':
            mori_shmem_lib = _get_mori_shmem_libdevice()
            # func name need to contain the lib name
            extern_libs = {"mori_shmem": str(mori_shmem_lib), "extra": str(libdevice_extra_lib)}
        else:
            raise ValueError(f"Unknown HIP SHMEM backend: {backend}")

        return extern_libs

    elif is_maca():
        from .utils import _get_mxshmem_libdevice
        mxshmem_lib = _get_mxshmem_libdevice()
        extern_libs = {"libshmem": str(mxshmem_lib)}
        return extern_libs

    else:
        raise NotImplementedError("Unsupported device type to get shmem bitcode lib path.")


class TritonDistJITFunction(KernelInterface[T]):
    __triton_builtin__ = True

    def __init__(self, fn: JITFunction[T]):
        self._triton_jit_fn = fn
        self._extern_libs = get_shmem_extern_lib()

    def __getattribute__(self, name: str):
        if name in ["_triton_jit_fn", "_extern_libs", "run", "warmup"]:
            return super().__getattribute__(name)
        return getattr(super().__getattribute__('_triton_jit_fn'), name)

    def __setattr__(self, name: str, value):
        if name in ["_triton_jit_fn", "_extern_libs", "run", "warmup"]:
            super().__setattr__(name, value)
        else:
            setattr(self._triton_jit_fn, name, value)

    def run(self, *args, **kwargs):
        kwargs.setdefault('extern_libs', self._extern_libs)

        # Apply CGA cluster size from environment variable if not explicitly set
        cga_cluster_size = _parse_cga_cluster_size()
        if cga_cluster_size is not None:
            # Only set if user hasn't explicitly provided cluster_dims
            if 'cluster_dims' not in kwargs:
                # num_ctas should be the product of cluster_dims for cluster launch
                expected_num_ctas = cga_cluster_size[0] * cga_cluster_size[1] * cga_cluster_size[2]

                if 'num_ctas' in kwargs:
                    # Check consistency between user-provided num_ctas and computed value
                    user_num_ctas = kwargs['num_ctas']
                    if user_num_ctas != expected_num_ctas:
                        warnings.warn(
                            f"num_ctas={user_num_ctas} does not match {_CGA_CLUSTER_SIZE_ENV}={cga_cluster_size} "
                            f"(expected num_ctas={expected_num_ctas}). "
                            f"Ignoring {_CGA_CLUSTER_SIZE_ENV} and using user-provided num_ctas.")
                        # Don't set cluster_dims since num_ctas is inconsistent
                    else:
                        # num_ctas matches, safe to set cluster_dims
                        kwargs['cluster_dims'] = cga_cluster_size
                else:
                    # No user-provided num_ctas, set both cluster_dims and num_ctas
                    kwargs['cluster_dims'] = cga_cluster_size
                    kwargs['num_ctas'] = expected_num_ctas

        return self._triton_jit_fn.run(*args, **kwargs)

    def warmup(self, *args, grid, **kwargs):
        from triton.runtime.jit import MockTensor
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)


def nvidia_stages_inspection_hook(self, stages, options, language, capability):
    from triton.backends.nvidia.compiler import sm_arch_from_capability, get_ptxas
    from triton_dist.nv_utils import NVSHMEMHelper, get_nvlink

    def make_cubin(self, src, metadata, opt, capability):
        ptxas = get_ptxas().path
        with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
            tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
            fsrc.write(src)
            fsrc.flush()
            fbin = fsrc.name + '.o'

            fbin_combined = fbin + ".combined.cubin"
            has_nvshmem_wrapper = bool(re.search(r'nvshmem\w*wrapper\b', src))
            compile_only_cmds = ["-c"] if has_nvshmem_wrapper else []
            line_info = ["-lineinfo", "-suppress-debug-info"] if knobs.compilation.disable_line_info else ["-lineinfo"]
            fmad = [] if opt.enable_fp_fusion else ['--fmad=false']
            arch = sm_arch_from_capability(capability)

            # Disable ptxas optimizations if requested
            disable_opt = ['--opt-level', '0'] if knobs.nvidia.disable_ptxas_opt else []

            # Accept more ptxas options if provided
            ptx_extra_options = opt.ptx_options.split(" ") if opt.ptx_options else []

            ptxas_cmd = [
                ptxas, *compile_only_cmds, *line_info, *fmad, '-v', *disable_opt, *ptx_extra_options,
                f'--gpu-name={arch}', fsrc.name, '-o', fbin
            ]
            try:
                subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
                if os.path.exists(fsrc.name):
                    os.remove(fsrc.name)
                if os.path.exists(flog.name):
                    os.remove(flog.name)
            except subprocess.CalledProcessError as e:
                with open(flog.name) as log_file:
                    log = log_file.read()
                if os.path.exists(flog.name):
                    os.remove(flog.name)

                if e.returncode == 255:
                    error = 'Internal Triton PTX codegen error'
                elif e.returncode == 128 + signal.SIGSEGV:
                    error = '`ptxas` raised SIGSEGV'
                else:
                    error = f'`ptxas` failed with error code {e.returncode}'

                raise PTXASError(f"{error}\n"
                                 f"`ptxas` stderr:\n{log}\n"
                                 f'Repro command: {" ".join(ptxas_cmd)}\n')

            if has_nvshmem_wrapper:
                # nvlink
                nvlink, _ = get_nvlink()
                nvlink_cmds = [
                    nvlink,
                    f"-arch={arch}",
                    f"-L{NVSHMEMHelper.get_nvshmem_lib()}",
                    "-lnvshmem_device",
                    fbin,
                    NVSHMEMHelper.get_nvshmem_cubin(src, capability, metadata).__str__(),
                    "-o",
                    fbin_combined,
                ]
                try:
                    subprocess.run(nvlink_cmds, check=True, close_fds=False, stderr=flog)
                except Exception as e:
                    import logging
                    logging.error(f"error runing nvlink: {nvlink_cmds}")
                    logging.exception(e)
            if has_nvshmem_wrapper:
                with open(fbin_combined, "rb") as f:
                    cubin = f.read()
            else:
                with open(fbin, "rb") as f:
                    cubin = f.read()
            if os.path.exists(fbin_combined):
                os.remove(fbin_combined)

            if os.path.exists(fbin):
                os.remove(fbin)
        return cubin

    stages["cubin"] = lambda src, metadata: make_cubin(self, src, metadata, options, self.target.arch)


def stages_inspection_hook(self, stages, options, language, capability):
    if is_cuda():
        nvidia_stages_inspection_hook(self, stages, options, language, capability)


def _install_triton_dist_hook():
    # shmem
    knobs.runtime.jit_post_compile_hook = shmem_kernel_module_init_hook

    # stages inspection
    knobs.runtime.add_stages_inspection_hook = stages_inspection_hook


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[TritonDistJITFunction[T], Callable[[T], TritonDistJITFunction[T]]]:
    """
        Triton-distributed JIT decorator(The signature is the same as triton.jit)

        This decorator wraps the standard Triton JIT decorator and adds support for
        SHMEM (NVSHMEM/ROCSHMEM). It automatically links the
        necessary SHMEM device bitcode and initializes the SHMEM runtime
        when kernels containing SHMEM operations are compiled and loaded.

        Compared to the original triton.jit:
        - Link SHMEM libraries during compilation
        - Provides a post-compilation hook to initialize SHMEM runtime state
        - Enables seamless use of SHMEM collective and one-sided communication
        primitives inside Triton kernels without extra user setup
    """

    _install_triton_dist_hook()

    def decorator(fn: T) -> TritonDistJITFunction[T]:
        triton_jit_fn = triton.jit(fn, version=version, repr=repr, launch_metadata=launch_metadata,
                                   do_not_specialize=do_not_specialize,
                                   do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                   noinline=noinline)
        assert callable(triton_jit_fn)
        return TritonDistJITFunction(triton_jit_fn)

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
