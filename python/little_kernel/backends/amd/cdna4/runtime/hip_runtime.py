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
"""HIP module API through ctypes: load a code object, look up a kernel, launch it.

Kernel arguments are packed into one kernarg buffer by the parameter list of the kernel in the
generated source (each argument at its natural alignment) and passed with
HIP_LAUNCH_PARAM_BUFFER_POINTER, so 64-bit parameters never depend on the Python value's size.
"""

import ctypes
import os
import re
import struct
from typing import Any, List, Optional, Sequence, Tuple

hipModule_t = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p
hipStream_t = ctypes.c_void_p

_HIP_LAUNCH_PARAM_BUFFER_POINTER = 0x01
_HIP_LAUNCH_PARAM_BUFFER_SIZE = 0x02
_HIP_LAUNCH_PARAM_END = 0x03

# C parameter type -> (struct format, size); pointers are "P"
_SCALAR_PARAMS = {
    "bool": ("?", 1),
    "int8_t": ("b", 1),
    "uint8_t": ("B", 1),
    "int16_t": ("h", 2),
    "uint16_t": ("H", 2),
    "int": ("i", 4),
    "int32_t": ("i", 4),
    "unsigned": ("I", 4),
    "uint32_t": ("I", 4),
    "int64_t": ("q", 8),
    "uint64_t": ("Q", 8),
    "float": ("f", 4),
    "double": ("d", 8),
}


def parse_kernel_params(code: str, kernel_name: str) -> List[str]:
    """Struct formats of the parameters of ``__global__ ... kernel_name(...)`` in ``code``."""
    match = re.search(r"__global__[^;{]*?\b" + re.escape(kernel_name) + r"\s*\(([^)]*)\)\s*\{", code)
    if match is None:
        raise RuntimeError(f"kernel {kernel_name} not found in the HIP source")
    params = [p.strip() for p in match.group(1).split(",") if p.strip()]
    formats = []
    for param in params:
        # Pointer-ness comes from the whole parameter text: multi-word pointer
        # types such as "long long *p" or "unsigned long *p" lose the "*" when
        # only the text before the final identifier is inspected, so the trimmed
        # type alone must never decide this -- any declarator containing "*" is
        # a pointer regardless of how the words around it are spelled.
        if "*" in param:
            formats.append("P")
            continue
        ctype = param.rsplit(None, 1)[0] if " " in param else param
        ctype = re.sub(r"\b(const|volatile)\b", "", ctype).strip()
        if ctype not in _SCALAR_PARAMS:
            raise RuntimeError(f"unsupported kernel parameter type '{ctype}' in {kernel_name}")
        formats.append(_SCALAR_PARAMS[ctype][0])
    return formats


def pack_kernel_args(formats: Sequence[str], args: Sequence[Any]) -> bytes:
    """Kernarg buffer: every argument at its natural alignment."""
    if len(formats) != len(args):
        raise TypeError(f"kernel takes {len(formats)} arguments, got {len(args)}")
    buf = bytearray()
    for fmt, arg in zip(formats, args):
        size = struct.calcsize(fmt) if fmt != "P" else 8
        buf.extend(b"\0" * (-len(buf) % size))
        if fmt == "P":
            if hasattr(arg, "data_ptr"):
                value = arg.data_ptr()
            elif isinstance(arg, ctypes.c_void_p):
                value = arg.value or 0
            else:
                value = int(arg)
            buf.extend(struct.pack("<Q", value))
        elif fmt in "fd":
            buf.extend(struct.pack("<" + fmt, float(arg)))
        elif fmt == "?":
            buf.extend(struct.pack("<?", bool(arg)))
        else:
            value = int(arg)
            if fmt.isupper():  # unsigned: accept values given as signed
                value &= (1 << (8 * size)) - 1
            buf.extend(struct.pack("<" + fmt, value))
    return bytes(buf)


def _stream_handle(stream) -> int:
    if stream is None:
        return 0
    if hasattr(stream, "cuda_stream"):  # torch.cuda.Stream (HIP stream on ROCm)
        return int(stream.cuda_stream)
    if isinstance(stream, ctypes.c_void_p):
        return stream.value or 0
    return int(stream)


class HIPRuntime:
    """The HIP runtime library (libamdhip64) loaded in this process."""

    def __init__(self, device=None):
        self._lib = self._load_library()
        self._setup()
        if device is not None:
            self._lib.hipSetDevice.argtypes = [ctypes.c_int]
            self._lib.hipSetDevice.restype = ctypes.c_int
            self.check(self._lib.hipSetDevice(int(device)), "hipSetDevice")

    @staticmethod
    def _load_library():
        candidates = [os.environ.get("LK_LIBAMDHIP64")]
        for root in (os.environ.get("ROCM_PATH"), os.environ.get("HIP_PATH")):
            if root:
                # ROCm installs the runtime under lib or lib64 depending on the distro layout.
                candidates.append(os.path.join(root, "lib", "libamdhip64.so"))
                candidates.append(os.path.join(root, "lib64", "libamdhip64.so"))
        candidates.append("libamdhip64.so")
        errors = []
        for path in candidates:
            if not path:
                continue
            try:
                # RTLD_GLOBAL: share the one HIP runtime with torch if it is already loaded
                return ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
            except OSError as exc:
                errors.append(f"{path}: {exc}")
        raise RuntimeError("Cannot load libamdhip64.so: " + "; ".join(errors))

    def _setup(self):
        lib = self._lib
        lib.hipModuleLoadData.argtypes = [ctypes.POINTER(hipModule_t), ctypes.c_void_p]
        lib.hipModuleGetFunction.argtypes = [ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.c_char_p]
        lib.hipModuleLaunchKernel.argtypes = [
            hipFunction_t, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # grid
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,  # block
            ctypes.c_uint,  # dynamic LDS bytes
            hipStream_t,
            ctypes.POINTER(ctypes.c_void_p),  # kernelParams
            ctypes.POINTER(ctypes.c_void_p),  # extra
        ]
        lib.hipModuleUnload.argtypes = [hipModule_t]
        lib.hipStreamSynchronize.argtypes = [hipStream_t]
        lib.hipDeviceSynchronize.argtypes = []
        lib.hipGetErrorString.argtypes = [ctypes.c_int]
        lib.hipGetErrorString.restype = ctypes.c_char_p
        for name in ("hipModuleLoadData", "hipModuleGetFunction", "hipModuleLaunchKernel", "hipModuleUnload",
                     "hipStreamSynchronize", "hipDeviceSynchronize"):
            getattr(lib, name).restype = ctypes.c_int

    def check(self, err: int, what: str):
        if err != 0:
            msg = self._lib.hipGetErrorString(err)
            raise RuntimeError(f"{what} failed: {msg.decode() if msg else err}")

    def load_module(self, code_object: bytes) -> hipModule_t:
        module = hipModule_t()
        image = ctypes.create_string_buffer(code_object, len(code_object))
        self.check(self._lib.hipModuleLoadData(ctypes.byref(module), image), "hipModuleLoadData")
        return module

    def get_function(self, module: hipModule_t, name: str) -> hipFunction_t:
        func = hipFunction_t()
        self.check(self._lib.hipModuleGetFunction(ctypes.byref(func), module, name.encode()), "hipModuleGetFunction")
        return func

    def unload_module(self, module):
        self.check(self._lib.hipModuleUnload(module), "hipModuleUnload")

    def launch(self, func: hipFunction_t, grid: Tuple[int, int, int], block: Tuple[int, int, int], kernargs: bytes,
               shared_mem_bytes: int = 0, stream=None):
        buf = ctypes.create_string_buffer(kernargs, max(len(kernargs), 1))
        size = ctypes.c_size_t(len(kernargs))
        extra = (ctypes.c_void_p * 5)(_HIP_LAUNCH_PARAM_BUFFER_POINTER, ctypes.addressof(buf),
                                      _HIP_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(size), _HIP_LAUNCH_PARAM_END)
        err = self._lib.hipModuleLaunchKernel(func, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                                              shared_mem_bytes, hipStream_t(_stream_handle(stream)), None,
                                              ctypes.cast(extra, ctypes.POINTER(ctypes.c_void_p)))
        self.check(err, "hipModuleLaunchKernel")

    def synchronize(self, stream=None):
        if stream is None:
            self.check(self._lib.hipDeviceSynchronize(), "hipDeviceSynchronize")
        else:
            self.check(self._lib.hipStreamSynchronize(hipStream_t(_stream_handle(stream))), "hipStreamSynchronize")


_RUNTIME: Optional[HIPRuntime] = None


def get_hip_runtime() -> HIPRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = HIPRuntime()
    return _RUNTIME
