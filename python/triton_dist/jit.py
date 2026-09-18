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

    # This is a *global* post-compile hook (fires for every compiled kernel in the
    # process, including plain ``@triton.jit`` ones). Only load the module onto the
    # device (``_init_handles``) when there is actual SHMEM state to initialise --
    # otherwise leave handle init lazy (upstream behaviour) and stay a no-op.
    if is_cuda():
        from triton_dist.utils import is_shmem_initialized
        has_shmem = "nvshmem" in kernel.asm['ptx']
        if has_shmem and is_shmem_initialized():
            import nvshmem.bindings.nvshmem as pynvshmem
            kernel._init_handles()
            pynvshmem.cumodule_init(kernel.module)
    elif is_hip():
        import torch
        from hip import hip
        from triton_dist.utils import get_shmem_backend

        kernel._init_handles()
        kernel_module = kernel.module
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
            # Initialize mori_shmem device symbols in this kernel module -- but only
            # once SHMEM is actually up (a distributed run). Single-GPU kernels that
            # use no shmem must skip this: calling shmem_module_init before mori is
            # initialized aborts in mori's CheckStatusValid(). Mirrors the CUDA
            # (is_shmem_initialized) and rocshmem (module-has-ctx) guards above.
            from triton_dist.utils import is_shmem_initialized
            if "mori_shmem" in kernel.asm.get('llir', '') and is_shmem_initialized():
                import mori.shmem as mori_shmem
                mori_shmem.shmem_module_init(kernel_module)
    elif is_maca():
        if "mxshmem" in kernel.asm['ttir']:
            import triton.pymxshmem as pymxshmem
            kernel._init_handles()
            pymxshmem.mxshmemx_mcmodule_init(kernel.module)
    elif is_ascend():
        pass
    # Unknown backend: nothing to initialise -- do not break compilation.


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

    elif is_ascend():
        return {}

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


def _dist_make_llir(self, src, metadata, options, capability):
    """Faithful re-implementation of the upstream NVIDIA ``CUDABackend.make_llir``
    that injects the plugin pass ``convert_triton_distributed_to_llvm`` right
    after the standard TritonGPU->LLVM conversion (``add_to_llvmir``), matching
    the placement used by the legacy intrusive fork.

    Distributed/SIMT ops survive ``add_to_llvmir`` (they are not in TritonGPU)
    and are lowered here, after shared-memory allocation and the global_smem
    symbol have been materialised. This must track the pinned Triton 3.7.1
    ``make_llir`` body; it lives entirely in triton_dist (no Triton source patch)."""
    from triton._C.libtriton import ir, passes, llvm, nvidia
    from triton.backends.nvidia.compiler import (sm_arch_from_capability, get_features, get_ptx_version_from_options,
                                                 CUDABackend)

    ptx_version = get_ptx_version_from_options(options, self.target.arch)

    mod = src
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_allocate_warp_groups(pm)
    passes.convert.add_scf_to_cf(pm)
    passes.gluon.add_inliner(pm)
    nvidia.passes.ttgpuir.add_allocate_shared_memory_nv(pm, capability, ptx_version)
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
    nvidia.passes.ttnvgpuir.add_check_matmul_two_cta(pm)
    if "consan" in options.instrumentation_mode:
        passes.ttgpuir.add_concurrency_sanitizer(pm)
    passes.ttgpuir.add_allocate_global_scratch_memory(pm)
    nvidia.passes.ttnvgpuir.add_proxy_fence_insertion(pm, capability)
    if CUDABackend.instrumentation:
        CUDABackend.instrumentation.patch("ttgpuir_to_llvmir", pm, mod.context)
    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
    # TritonDistributed Extension: Distributed/SIMT Dialect -> LLVM (plugin pass).
    # Placed immediately after the standard TritonGPU->LLVM conversion so the
    # Distributed/SIMT ops (which survive add_to_llvmir) are lowered while the
    # global_smem symbol and shared-memory allocation are still materialised --
    # the same placement the legacy intrusive fork used.
    passes.plugin.convert_triton_distributed_to_llvm(pm, [str(capability), str(ptx_version)])
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
    nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.convert.add_nvvm_to_llvm(pm)

    if not knobs.compilation.disable_line_info and not knobs.compilation.dump_ir_extract_di_local_variables:
        passes.llvmir.add_di_scope(pm)

    if CUDABackend.instrumentation:
        CUDABackend.instrumentation.patch("llvmir_to_llvm", pm, mod.context)

    pm.run(mod, 'make_llir')

    if knobs.compilation.dump_ir_extract_di_local_variables:
        if not knobs.compilation.disable_line_info:
            pm = ir.pass_manager(mod.context)
            pm.enable_debug()
            passes.llvmir.add_di_scope(pm)
            pm.run(mod, 'make_llir.disable_line_info')
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.llvmir.add_di_local_variable(pm)
        pm.run(mod, 'make_llir.dump_ir_extract_di_local_variables')

    # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    llvm.init_targets()
    context = llvm.context()
    if knobs.compilation.enable_asan:
        raise RuntimeError("Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
    llvm_mod = llvm.to_module(mod, context)
    proc = sm_arch_from_capability(capability)
    features = get_features(options, self.target.arch)
    triple = 'nvptx64-nvidia-cuda'
    nvidia.set_short_ptr()
    llvm.attach_datalayout(llvm_mod, triple, proc, features)
    if options.enable_reflect_ftz:
        nvidia.set_nvvm_reflect_ftz(llvm_mod)

    # Link extern device bitcode. The SHMEM device library (e.g.
    # libnvshmem_device.bc) must be linked whenever the module references SHMEM
    # symbols -- including for plain ``@triton.jit`` kernels that use distributed
    # ops but do not go through ``TritonDistJITFunction`` (which would inject it via
    # extern_libs). The legacy fork linked it unconditionally for such kernels; we
    # add it on demand here so unresolved ``nvshmem_*`` symbols never reach ptxas.
    extern_libs = list(options.extern_libs or [])
    if nvidia.has_extern_deps(llvm_mod):
        if "nvshmem" in str(llvm_mod):
            have = {name for (name, _path) in extern_libs}
            for name, path in get_shmem_extern_lib().items():
                if name not in have:
                    extern_libs.append((name, path))
        if extern_libs:
            paths = [path for (_name, path) in extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    total_num_warps = src.get_int_attr("ttg.total-num-warps")
    if total_num_warps is not None:
        metadata["num_warps"] = total_num_warps
    metadata["shared"] = src.get_int_attr("ttg.shared")
    metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
    metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size") or 0
    metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment") or 1
    metadata["profile_scratch_size"] = src.get_int_attr("ttg.profile_scratch_memory_size") or 0
    metadata["profile_scratch_align"] = src.get_int_attr("ttg.profile_scratch_memory_alignment") or 1
    ret = str(llvm_mod)
    del llvm_mod
    del context
    return ret


def nvidia_stages_inspection_hook(self, stages, options, language, capability):
    from triton._C.libtriton import ir, passes
    from triton.backends.nvidia.compiler import sm_arch_from_capability, get_ptxas
    from triton_dist.nv_utils import NVSHMEMHelper, get_nvlink

    # --- TTIR stage: inject Distributed/SIMT -> TritonGPU conversion -----------
    # The plugin conversion is a drop-in superset of the standard
    # convert-triton-to-tritongpu, run right after make_ttir so the subsequent
    # standard make_ttgir optimisation pipeline operates on TTGIR (mirrors utlx).
    original_make_ttir = self.make_ttir

    def make_ttir_wrapper(mod, metadata, opt, cap):
        mod = original_make_ttir(mod, metadata, opt, cap)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.plugin.convert_triton_distributed_to_tritongpu(
            pm, [f"cuda:{cap}", str(opt.num_warps), '32', str(opt.num_ctas)])
        pm.run(mod, 'dist_ttir_conversion')
        return mod

    stages["ttir"] = lambda src, metadata: make_ttir_wrapper(src, metadata, options, capability)

    # --- LLIR stage: inject Distributed/SIMT -> LLVM after add_to_llvmir -------
    stages["llir"] = lambda src, metadata: _dist_make_llir(self, src, metadata, options, capability)

    def make_cubin(self, src, metadata, opt, capability):
        ptxas = get_ptxas(capability).path
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
                    subprocess.run(nvlink_cmds, check=True, close_fds=False, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
                except subprocess.CalledProcessError as e:
                    # Surface the real linker error instead of letting execution fall
                    # through to a misleading FileNotFoundError on the (never-produced)
                    # combined cubin. Clean up partial outputs first.
                    for _p in (fbin, fbin_combined):
                        if os.path.exists(_p):
                            os.remove(_p)
                    nvlink_log = e.stderr.decode("utf-8", "replace") if e.stderr else ""
                    raise PTXASError(f"`nvlink` failed with error code {e.returncode}\n"
                                     f"`nvlink` stderr:\n{nvlink_log}\n"
                                     f'Repro command: {" ".join(nvlink_cmds)}\n') from e
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


def _amd_make_llir(self, src, metadata, options):
    """AMD analog of ``_dist_make_llir``.

    Rather than copy ``HIPBackend.make_llir`` verbatim (it is long and drifts with
    the AMD pipeline), run the stock one but temporarily swap the two AMD pass
    calls it makes so the distributed lowering is spliced in at the right places:

      * ``add_to_llvmir`` -> the standard pass followed by
        ``convert_amd_distributed_to_llvm``, which lowers the remaining
        Distributed/SIMT ops (the standard pass lowers tt/ttgpu -- incl. tt.func
        and tt.extern_elementwise -- and leaves the distributed ops untouched;
        the distributed pass then legalizes those). This matches the NVIDIA path
        (``add_to_llvmir`` then ``convert_triton_distributed_to_llvm``).
      * ``add_builtin_func_to_llvmir`` -> the standard pass followed by
        ``convert_builtin_func_to_llvmir_ext``, which lowers the distributed
        ``__triton_hip_*`` extern calls (ld/st/atomic/syncthreads/v4_b32).

    This mirrors the fork's intrusive HIP ``compiler.py`` edits
    (``add_distributed_to_llvm`` / ``add_builtin_func_to_llvmir_ext`` -- see the
    removed ``python/src/passes.cc``) without a Triton source patch. The swap is
    scoped to this call and restored in ``finally``. For plain kernels (no
    distributed ops) the superset conversion is functionally identical to the
    standard one and the ``_ext`` pass is a no-op."""
    from triton._C.libtriton import amd as _amd, passes as _passes
    _ttgpuir = _amd.passes.ttgpuir
    _orig_to_llvmir = _ttgpuir.add_to_llvmir
    _orig_builtin = _ttgpuir.add_builtin_func_to_llvmir

    def _dist_to_llvmir(pm, arch, ftz):
        _orig_to_llvmir(pm, arch, ftz)
        _passes.plugin.convert_amd_distributed_to_llvm(pm, [arch, "1" if ftz else "0"])

    def _dist_builtin(pm, arch, ftz):
        _orig_builtin(pm, arch, ftz)
        _passes.plugin.convert_builtin_func_to_llvmir_ext(pm, ["1" if ftz else "0"])

    _ttgpuir.add_to_llvmir = _dist_to_llvmir
    _ttgpuir.add_builtin_func_to_llvmir = _dist_builtin
    try:
        return self.make_llir(src, metadata, options)
    finally:
        _ttgpuir.add_to_llvmir = _orig_to_llvmir
        _ttgpuir.add_builtin_func_to_llvmir = _orig_builtin


def amd_stages_inspection_hook(self, stages, options, language):
    """AMD counterpart of ``nvidia_stages_inspection_hook``.

    Splices the distributed lowering into the stock AMD pipeline by swapping the
    two conversion entry points ``make_ttgir`` / ``make_llir`` call:

      * ``passes.ttir.add_convert_to_ttgpuir`` -> the distributed superset
        ``convert_triton_distributed_to_tritongpu``. This *replaces* (rather than
        runs before) the standard TTIR->TTGIR conversion, so there is exactly ONE
        conversion. Running the standard convert a second time on already-TTGIR IR
        leaves dangling 0-operand ``unrealized_conversion_cast`` ops that the AMD
        AxisInfo passes (e.g. ``convert-buffer-ops``) crash on.

    The llir stage is handled by ``_amd_make_llir`` (distributed LLVM lowering).
    Only the Triton-language path is wrapped -- Gluon has no distributed frontend
    here. The swaps are scoped to each stage call and restored in ``finally``."""
    from triton._C.libtriton import passes as _passes
    from triton.backends.compiler import Language

    if language == Language.TRITON:

        def _amd_make_ttgir(src, metadata, opt):
            _ttir = _passes.ttir
            _orig_convert = _ttir.add_convert_to_ttgpuir

            def _dist_convert(pm, target, num_warps, warp_size, num_ctas):
                _passes.plugin.convert_triton_distributed_to_tritongpu(
                    pm, [target, str(num_warps), str(warp_size), str(num_ctas)])

            # AMD's block-level pointer/buffer-op passes (tritonamdgpu-canonicalize-
            # pointers + convert-buffer-ops, newer than the 3.4 fork) assume plain
            # triton pointers derived from kernel args, and mishandle the custom
            # distributed/SIMT ops:
            #   * simt.simt_exec_region: the 1:N pointer conversion rewrites pointers
            #     used inside the region (e.g. a distributed.extern_call) into
            #     degenerate 0-operand unrealized_conversion_casts, aborting
            #     ModuleAxisInfoAnalysis.
            #   * distributed.symm_at / consume_token: these thread a pointer through
            #     a remote/symmetric handle or a scheduling token, which breaks the
            #     1:N loop-carried pointer rewrite (scf.for iter_args vs yields
            #     mismatch) in canonicalize-pointers.
            # None of this per-thread / remote-pointer code benefits from these
            # block/tensor-pointer optimizations, and the 3.4 fork had no such pass,
            # so disable buffer ops whenever the kernel uses a distributed or SIMT op.
            # Pure-compute kernels have neither and keep buffer ops.
            _ir = src.str_nodebug()
            _uses_dist_ops = ("simt." in _ir) or ("distributed." in _ir)
            _saved_buffer_ops = knobs.amd.use_buffer_ops
            if _uses_dist_ops:
                knobs.amd.use_buffer_ops = False

            _ttir.add_convert_to_ttgpuir = _dist_convert
            try:
                return self.make_ttgir(src, metadata, opt)
            finally:
                _ttir.add_convert_to_ttgpuir = _orig_convert
                knobs.amd.use_buffer_ops = _saved_buffer_ops

        stages["ttgir"] = lambda src, metadata: _amd_make_ttgir(src, metadata, options)

    stages["llir"] = lambda src, metadata: _amd_make_llir(self, src, metadata, options)


_HOOK_KEY = None
_HOOK_HASH = None


def _get_hook_cache_dims():
    """Cache key/hash contributed by the distributed plugin so kernels compiled
    with vs without the hook never collide. Hashes this module's source plus the
    plugin .so mtime/path. Replaces the fork's CACHE_INVALIDATING_ENV_VARS."""
    global _HOOK_KEY, _HOOK_HASH
    if _HOOK_KEY is None:
        import hashlib
        from triton_dist import _plugin
        parts = [Path(__file__).read_text()]
        so = _plugin.find_plugin()
        if so:
            parts.append(f"{so}:{os.path.getmtime(so)}")
        _HOOK_KEY = "\n".join(parts)
        _HOOK_HASH = hashlib.sha256(_HOOK_KEY.encode("utf-8")).hexdigest()
    return _HOOK_KEY, _HOOK_HASH


def stages_inspection_hook(self=None, stages=None, options=None, language=None, capability=None):
    # No-arg invocation: contribute to the compilation cache key only.
    if all(a is None for a in (self, stages, options, language, capability)):
        return _get_hook_cache_dims()
    if is_cuda():
        nvidia_stages_inspection_hook(self, stages, options, language, capability)
    elif is_hip():
        amd_stages_inspection_hook(self, stages, options, language)
    return _get_hook_cache_dims()


_CLUSTER_DIMS_PATCHED = False


def _install_cluster_dims_support():
    """Let the NVIDIA backend accept a ``cluster_dims`` launch option.

    Upstream Triton 3.7.1's ``CUDAOptions`` only exposes ``num_ctas`` (the cluster
    is always launched as ``[num_ctas, 1, 1]``); the legacy fork carried an
    explicit 3D ``cluster_dims``. We restore it non-intrusively: ``parse_options``
    consumes ``cluster_dims`` (so the runtime's strict unknown-kwarg check in
    ``_pack_args`` accepts it), normalises it, and records it on the *frozen*
    options object so it is serialised into the compiled kernel's metadata
    (``metadata = {**options.__dict__, ...}``) -- which is what
    ``TRITON_DIST_CGA_CLUSTER_SIZE`` consumers introspect. The cluster launch
    already derives ``clusterDim.x = num_ctas`` from ``num_ctas``, so we keep
    ``cluster_dims`` consistent with it (defaulting to ``[num_ctas, 1, 1]``)."""
    global _CLUSTER_DIMS_PATCHED
    if _CLUSTER_DIMS_PATCHED or not is_cuda():
        return
    from triton.backends.nvidia.compiler import CUDABackend
    orig_parse_options = CUDABackend.parse_options
    if getattr(orig_parse_options, "_triton_dist_wrapped", False):
        _CLUSTER_DIMS_PATCHED = True
        return

    def parse_options(self, opts):
        cluster_dims = None
        if "cluster_dims" in opts:
            opts = dict(opts)
            cluster_dims = opts.pop("cluster_dims")
        options = orig_parse_options(self, opts)
        if cluster_dims is None:
            cluster_dims = (options.num_ctas, 1, 1)
        cluster_dims = tuple(int(x) for x in cluster_dims)
        # Frozen dataclass: bypass the frozen guard to attach the field so it is
        # serialised into metadata next to num_ctas.
        object.__setattr__(options, "cluster_dims", list(cluster_dims))
        return options

    parse_options._triton_dist_wrapped = True
    CUDABackend.parse_options = parse_options
    _CLUSTER_DIMS_PATCHED = True


# Triton release this frontend was validated against. `_dist_make_llir` mirrors
# this release's NVIDIA `CUDABackend.make_llir` pass pipeline verbatim (to splice
# in the distributed->LLVM pass); on a different Triton that body can silently
# drift and produce subtly-wrong codegen. Warn loudly so the mismatch is visible.
# See docs/refactor/MONKEYPATCH_TRACKING.md.
_VALIDATED_TRITON_VERSION = "3.7.1"
_VERSION_CHECKED = False


def _check_triton_version():
    global _VERSION_CHECKED
    if _VERSION_CHECKED:
        return
    _VERSION_CHECKED = True
    version = getattr(triton, "__version__", "") or ""
    if not version.startswith(_VALIDATED_TRITON_VERSION):
        warnings.warn(f"triton_dist was validated against Triton {_VALIDATED_TRITON_VERSION}, "
                      f"but the installed Triton is {version!r}. The distributed compile hook "
                      f"(_dist_make_llir) mirrors that release's CUDABackend.make_llir pass "
                      f"pipeline and may drift on other versions -- verify codegen if results "
                      f"look wrong.")


def _install_triton_dist_hook():
    # Warn if the installed Triton is not the validated pin (see above).
    _check_triton_version()

    # shmem
    knobs.runtime.jit_post_compile_hook = shmem_kernel_module_init_hook

    # stages inspection
    knobs.runtime.add_stages_inspection_hook = stages_inspection_hook

    # cluster_dims launch option (CGA cluster size)
    _install_cluster_dims_support()


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
