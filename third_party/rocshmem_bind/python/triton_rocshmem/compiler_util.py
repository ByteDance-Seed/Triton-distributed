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
import functools
import os
import pathlib
import re
import subprocess
import sysconfig

## TODO: add compiler utils for rocshmem

ROCSHMEM_WRAPPER_FUNCS = [
    "rocshmem_ptr_wrapper",
    "rocshmem_my_pe_wrapper",
    "rocshmem_n_pes_wrapper",
    #"rocshmem_int_p_wrapper", # not found in rocshmem
]

ROCSHMEM_GLOBAL_VARIABLES = [
    # not found
    #"rocshmemi_device_state_d",
]

ROCSHMEM_SYMBOLS = ROCSHMEM_WRAPPER_FUNCS + ROCSHMEM_GLOBAL_VARIABLES

# TODO: implement gcn
ROCSHMEM_WRAPPER_EXTERN = {
    "rocshmem_ptr_wrapper": """.extern .func (.param .b64 func_retval0) rocshmem_ptr_wrapper
(
    .param .b64 rocshmem_ptr_wrapper_param_0,
    .param .b32 rocshmem_ptr_wrapper_param_1
);""",
    "rocshmem_my_pe_wrapper": """.extern.func (.param.b32 func_retval0) rocshmem_my_pe_wrapper();""",
    "rocshmem_n_pes_wrapper": """.extern.func (.param.b32 func_retval0) rocshmem_n_pes_wrapper();""",
}


@functools.lru_cache()
def _path_to_binary(binary: str):
    binary += sysconfig.get_config_var("EXE")
    paths = [
        os.environ.get(f"TRITON_{binary.upper()}_PATH", ""),
        os.path.join(os.path.dirname(__file__), "bin", binary),
    ]

    paths += [os.environ.get("ROCM_HOME", "/opt/rocm/") + "bin/"]

    for path in paths:
        if os.path.exists(path) and os.path.isfile(path):
            result = subprocess.check_output([path, "--version"], stderr=subprocess.STDOUT)
            if result is not None:
                version = re.search(
                    r".*release (\d+\.\d+).*",
                    result.decode("utf-8"),
                    flags=re.MULTILINE,
                )
                if version is not None:
                    return path, version.group(1)
    raise RuntimeError(f"Cannot find {binary}")


@functools.lru_cache()
def get_rocshmem_home():
    return pathlib.Path(
        os.environ.get(
            "ROCSHMEM_HOME",
            pathlib.Path(__file__).parent.parent.parent.parent / "rocshmem" / "build" / "install",
        ))


@functools.lru_cache()
def get_rocshmem_lib():
    return get_rocshmem_home() / "lib"


@functools.lru_cache()
def get_triton_rocshmem_runtime_lib():
    pass


@functools.lru_cache()
def get_rocshmem_cubin(capability):
    return (pathlib.Path(__file__).parent.parent.parent / "runtime" / f"rocshmem_wrapper.sm{capability}.cubin")


@functools.lru_cache()
def get_hiplink(arch: int):
    pass


def has_rocshmem_wrappers(ptx):
    return any([x in ptx for x in ROCSHMEM_SYMBOLS])


def get_rocshmem_wrappers(ptx):
    extern_symbols = []
    for symbol in ROCSHMEM_SYMBOLS:
        if symbol in ptx:
            extern_symbols.append(symbol)
    return extern_symbols


def patch_rocshmem_wrapper_externs(ptx):
    wrappers = get_rocshmem_wrappers(ptx)
    if len(wrappers) == 0:
        return ptx
    externs = []
    for wrapper in wrappers:
        externs.append(ROCSHMEM_WRAPPER_EXTERN[wrapper])
    externs = "\n".join(externs)

    MARKER = ".address_size 64"
    loc = [x.strip() for x in ptx.split("\n")].index(MARKER) + 1
    assert loc > 0

    lines = ptx.split("\n")
    ptx = "\n".join(lines[:loc]) + "\n" + externs + "\n" + "\n".join(lines[loc:])
    return ptx
