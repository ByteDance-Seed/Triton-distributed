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
"""
SHMEM host-side runtime wrapper via ctypes.

Wraps libshmem.so for the host-side SHMEM lifecycle:
init, symmetric heap malloc/free, FFTS config, barrier, finalize.
"""

import ctypes
import os
from typing import Optional

SHMEM_SUCCESS = 0
SHMEM_STATUS_IS_INITIALIZED = 2


def _find_shmem_lib(explicit_path: Optional[str] = None) -> ctypes.CDLL:
    """Locate and load libshmem.so."""
    if explicit_path and os.path.exists(explicit_path):
        return ctypes.CDLL(explicit_path)

    search = ["libshmem.so"]
    shmem_home = os.environ.get("SHMEM_HOME_PATH")
    if shmem_home:
        search.append(os.path.join(shmem_home, "lib", "libshmem.so"))

    for path in search:
        try:
            return ctypes.CDLL(path)
        except OSError:
            continue

    raise RuntimeError("Failed to load libshmem.so. Set SHMEM_HOME_PATH or LD_LIBRARY_PATH.")


class ShmemRuntime:
    """Host-side SHMEM runtime wrapper."""

    def __init__(self, lib_path: Optional[str] = None):
        self._lib = _find_shmem_lib(lib_path)
        self._attr = None
        self._setup_signatures()

    def _setup_signatures(self):
        lib = self._lib

        lib.shmem_set_conf_store_tls.argtypes = [
            ctypes.c_bool,
            ctypes.c_char_p,
            ctypes.c_uint32,
        ]
        lib.shmem_set_conf_store_tls.restype = ctypes.c_int32

        lib.shmem_set_attr.argtypes = [
            ctypes.c_int,  # my_rank
            ctypes.c_int,  # n_ranks
            ctypes.c_uint64,  # local_mem_size
            ctypes.c_char_p,  # ip_port
            ctypes.POINTER(ctypes.c_void_p),  # **attributes
        ]
        lib.shmem_set_attr.restype = ctypes.c_int

        lib.shmem_init_attr.argtypes = [ctypes.c_void_p]
        lib.shmem_init_attr.restype = ctypes.c_int

        lib.shmem_init_status.argtypes = []
        lib.shmem_init_status.restype = ctypes.c_int

        lib.shmem_malloc.argtypes = [ctypes.c_size_t]
        lib.shmem_malloc.restype = ctypes.c_void_p

        lib.shmem_free.argtypes = [ctypes.c_void_p]
        lib.shmem_free.restype = None

        lib.shmemx_get_ffts_config.argtypes = []
        lib.shmemx_get_ffts_config.restype = ctypes.c_uint64

        lib.shmemx_barrier_all_on_stream.argtypes = [ctypes.c_void_p]
        lib.shmemx_barrier_all_on_stream.restype = None

        lib.shmem_finalize.argtypes = []
        lib.shmem_finalize.restype = ctypes.c_int

    def _check(self, ret: int, func_name: str):
        if ret != SHMEM_SUCCESS:
            raise RuntimeError(f"SHMEM error in {func_name}: {ret}")

    # -- lifecycle --

    def set_conf_store_tls(self, enable: bool = False):
        ret = self._lib.shmem_set_conf_store_tls(enable, None, 0)
        self._check(ret, "shmem_set_conf_store_tls")

    def set_attr(
        self,
        rank_id: int,
        n_ranks: int,
        local_mem_size: int,
        ip_port: str,
    ):
        attr_ptr = ctypes.c_void_p()
        ret = self._lib.shmem_set_attr(
            rank_id,
            n_ranks,
            local_mem_size,
            ip_port.encode("utf-8"),
            ctypes.byref(attr_ptr),
        )
        self._check(ret, "shmem_set_attr")
        self._attr = attr_ptr

    def init_attr(self):
        if self._attr is None:
            raise RuntimeError("Call set_attr() before init_attr().")
        ret = self._lib.shmem_init_attr(self._attr)
        self._check(ret, "shmem_init_attr")

    def init_status(self) -> int:
        return self._lib.shmem_init_status()

    def init(
        self,
        rank_id: int,
        n_ranks: int,
        local_mem_size: int = 1024 * 1024 * 1024,
        ip_port: str = "tcp://127.0.0.1:8666",
    ):
        """Convenience: set_conf_store_tls + set_attr + init_attr."""
        self.set_conf_store_tls(False)
        self.set_attr(rank_id, n_ranks, local_mem_size, ip_port)
        self.init_attr()

    # -- symmetric heap --

    def malloc(self, size: int) -> ctypes.c_void_p:
        """Allocate symmetric heap memory. Returns device pointer."""
        ptr = self._lib.shmem_malloc(ctypes.c_size_t(size))
        if ptr is None or ptr == 0:
            raise RuntimeError(f"shmem_malloc({size}) returned NULL")
        return ctypes.c_void_p(ptr)

    def free(self, ptr):
        if isinstance(ptr, int):
            ptr = ctypes.c_void_p(ptr)
        self._lib.shmem_free(ptr)

    # -- FFTS & sync --

    def get_ffts_config(self) -> int:
        return self._lib.shmemx_get_ffts_config()

    def barrier_all_on_stream(self, stream):
        self._lib.shmemx_barrier_all_on_stream(stream)

    # -- teardown --

    def finalize(self):
        ret = self._lib.shmem_finalize()
        self._check(ret, "shmem_finalize")
