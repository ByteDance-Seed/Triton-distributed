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
import importlib
import shutil
import logging
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import triton
import triton.backends
import triton.language as tl
from packaging.version import Version
from triton.tools.link import HeaderParser, KernelLinkerMeta

if Version(triton.__version__) < Version("3.0.0"):
    raise RuntimeError("AOT compilation requires triton>=3.0.0")

from triton.tools.link import (
    gen_signature_with_full_args,
    make_algo_decls,
    make_default_algo_kernel,
    make_get_num_algos_decl,
    make_get_num_algos_def,
    make_global_decl,
    make_kernel_hints_dispatcher,
    make_kernel_load_def,
    make_kernel_meta_const_dispatcher,
)

# bypass triton cache
ALWAYS_GENERATE = True


def aot_compile_spaces(args):

    def decrator(fn):
        fn.__aot_compile_spaces__ = args
        return fn

    # check args format
    assert isinstance(args, dict)
    for kernel_name, kernel_args in args.items():
        assert isinstance(kernel_args, dict)
        assert "signature" in kernel_args
        assert "grid" in kernel_args
        assert isinstance(kernel_args["grid"], list)
        assert "triton_algo_infos" in kernel_args
        assert isinstance(kernel_args["triton_algo_infos"], list)
        assert len(kernel_args["triton_algo_infos"]) > 0
    return decrator


@aot_compile_spaces({
    "vector_add_fp32": {
        "signature":
        "*fp32, *fp32, *fp32, i32:1, i32:16, %BLOCK_SIZE",
        "grid": [
            "n_elements / %BLOCK_SIZE",
            "1",
            "1",
        ],
        "triton_algo_infos": [
            {
                "num_warps": 4,
                "num_stages": 3,
                "BLOCK_SIZE": 1024,
            },
            {
                "num_warps": 4,
                "num_stages": 3,
                "BLOCK_SIZE": 2048,
            },
        ],
    },
    "vector_add_fp16": {
        "signature": "*fp16, *fp16, *fp16, i32:1, i32:16, %BLOCK_SIZE",
        "grid": [
            "n_elements / %BLOCK_SIZE",
            "1",
            "1",
        ],
        "triton_algo_infos": [{
            "num_warps": 4,
            "num_stages": 3,
            "BLOCK_SIZE": 1024,
        }],
    },
})
@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               stride, n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE) * stride
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def _write_if_changed(path: Path, content: str):
    old_content = path.read_text() if path.exists() else ""
    if old_content != content:
        path.write_text(content)


def _copy_if_changed(dest: Path, src: Path):
    # copy file if destination is older, or does not exist
    if (not dest.exists()) or ((src.stat().st_mtime > dest.stat().st_mtime) and src.read_text() != dest.read_text()):
        shutil.copy2(src, dest)


def _check_signature_or_throw(kernel: triton.JITFunction, signature: str):
    """
    check all scalar arguments or constexpr arguments.
    no check pointer dtypes

    argument signature should be:  [*]dtype[:annotation]
      * means pointer type
      dtype is one of i1/i16/i32/i64/u64/fp32
      annotation is optional, which can be "" or ":16" or ":1", which used for alignment
    """
    SCALAR_DTYPES = ["i1", "i16", "i32", "i64", "u64", "fp32"]
    POINTER_DTYPES = list(triton.runtime.jit.type_canonicalisation_dict.values())

    def _is_valid_arg_sig(sig: str):
        if sig.endswith(":16"):
            sig = sig[:-3]
        if sig.endswith(":1"):
            sig = sig[:-2]
        if sig.startswith("*"):
            sig = sig[1:]
            if sig.startswith("k"):
                sig = sig[1:]
            return sig in POINTER_DTYPES
        return sig in SCALAR_DTYPES

    signature = signature.split(",")
    signature = [s.strip(" ") for s in signature]
    assert len(kernel.arg_names) == len(
        signature), f"{len(kernel.arg_names)} vs {len(signature)}: {kernel.arg_names} | {signature}"
    for index, (param, sig) in enumerate(zip(kernel.params, signature)):
        if param.is_constexpr:
            assert (f"%{param.name}" == sig
                    ), f"invalid constexpr signature at {index}-th: `%{param.name}` expected but `{sig}` got"
        else:
            assert _is_valid_arg_sig(sig), f"invalid none-constepxr signature at {index}-th: {sig}"


def make_kernel_algo_info_struct_name(c_kernel_name):
    return f"{c_kernel_name}__triton_algo_info_t"


def make_algo_info_decl(kernel_name, algo_info_schema):
    src = f"struct {make_kernel_algo_info_struct_name(kernel_name)} {{\n"
    for name, ctype in algo_info_schema:
        if ctype is int:
            src += f" int {name};\n"
        elif ctype is bool:
            src += f" bool {name};\n"
        elif ctype is float:
            src += f" float {name};\n"
        else:
            raise ValueError(f"constexpr `{name}` type {ctype} not supported")
    src += " int num_warps;\n"
    src += " int num_stages;\n"
    src += "};"
    return src


def _compile_kernel(
    workspace: Path,
    signature: str,
    func: triton.JITFunction,
    out_name: str,
    num_warps: int,
    num_stages: int,
    grid: List[str],
):
    from triton.distributed.tools.compile import (
        make_ast_source,
        materialize_c_params,
        dump_c_code,
    )

    src, hints = make_ast_source(func, signature)
    opts = {"num_warps": num_warps, "num_stages": num_stages}
    ccinfo = triton.compile(src, options=opts)  # this may trigger cache
    params, func_kernel_suffix = materialize_c_params(
        func,
        hints,
        ccinfo,
        func.__name__,
        out_name,
        num_warps,
        num_stages,
        grid,
    )
    # func_kernel_suffix = kernel_name_suffix(src, num_stages, num_warps)
    # first check the cache manager
    cm: triton.runtime.cache.CacheManager = triton.runtime.cache.get_cache_manager(ccinfo.hash)
    assert isinstance(cm, triton.runtime.cache.FileCacheManager)
    cache_dir = Path(cm.cache_dir)
    cache_file = f"{func.__name__}"

    if (not cm.has_file(f"{cache_file}.c") or not cm.has_file(f"{cache_file}.h")) or ALWAYS_GENERATE:
        # if you update the c file generate logic, clean c/h files manually
        logging.info(f"dump {func.__name__}.c/h file to {cache_dir}")
        dump_c_code(cache_dir / f"{func.__name__}", params)
    #
    for ext in ["c", "h"]:
        cfile = cache_dir / f"{func.__name__}.{ext}"
        shutil.copy2(cfile, workspace / f"{out_name}.{func_kernel_suffix}.{ext}")


def compile_kernel(func: triton.JITFunction, workspace: Path):

    def _to_schema(triton_algo_infos):
        return [(param.name, type(triton_algo_infos[param.name])) for param in func.params if param.is_constexpr]

    def _materialize_constexpr(
        signature: str,
        grid: List[str],
        kernel: triton.JITFunction,
        triton_algo_infos: List[Dict[str, Any]],
    ):
        # replace %{constexpr} with concrete values
        for param in kernel.params:
            if param.is_constexpr:
                assert (
                    param.name
                    in triton_algo_infos), f"constexpr {param.name} not found in triton_algo_infos: {triton_algo_infos}"
                signature = signature.replace(f"%{param.name}", str(triton_algo_infos[param.name]))
                grid = [x.replace(f"%{param.name}", str(triton_algo_infos[param.name])) for x in grid]
        return signature, grid

    assert func.__aot_compile_spaces__ is not None
    context = {}
    for c_kernel_name, kernel_args in func.__aot_compile_spaces__.items():
        for triton_algo_infos in kernel_args["triton_algo_infos"]:
            logging.info(f"Compiling {c_kernel_name} with triton_algo_infos {triton_algo_infos}")
            signature, grid = kernel_args["signature"], kernel_args["grid"]
            _check_signature_or_throw(kernel, signature)
            signature, grid = _materialize_constexpr(signature, grid, kernel, triton_algo_infos)
            _compile_kernel(
                workspace=workspace,
                signature=signature,
                func=func,
                out_name=c_kernel_name,
                num_warps=triton_algo_infos.get("num_warps", 4),
                num_stages=triton_algo_infos.get("num_stages", 4),
                grid=grid,
            )
            tt_kernel_name = func.__name__
            if tt_kernel_name not in context:
                context[tt_kernel_name] = {
                    "kernel_names": [],
                    "constexpr": _to_schema(triton_algo_infos),
                }
            if c_kernel_name not in context[tt_kernel_name]["kernel_names"]:
                context[tt_kernel_name]["kernel_names"].append(c_kernel_name)
    return context


def _make_triton_algo_info_with_schema(algo_info: str, schema: List[Tuple[str, type]]):

    def _to_value(value, ctype):
        assert ctype in [int, float, bool]
        if ctype is int:
            return int(value)
        if ctype is float:
            return float(value)
        if ctype is bool:
            assert value.lower() in ["0", "1"], f"value {value} not in [0, 1]"
            return value.lower() == "1"

    # 1024_warps4xstages3
    import re

    schema_pattern = re.compile("x".join([r"(\d+)" if type_t is int else r"(\d+(?:\.\d+)?)"
                                          for name, type_t in schema]) + r"_warps(\d+)xstages(\d+)")
    return {
        name: _to_value(v, ctype)
        for v, (name, ctype) in zip(
            schema_pattern.match(algo_info).groups(),
            schema + [("num_warps", int), ("num_stages", int)],
        )
    }


# same kernel should have the same algo_info schema
def _get_algo_info(kernel_name, orig_kernel_name):
    assert kernel_name.startswith(orig_kernel_name)
    return kernel_name[len(orig_kernel_name) + 1:]


def make_global_decl_with_algo_info(meta: KernelLinkerMeta, algo_info_struct: str) -> str:
    return f"""
CUresult {meta.orig_kernel_name}_ex(CUstream stream, {gen_signature_with_full_args(meta)}, struct {algo_info_struct} algo_info);
    """


def map_arg_to_pybind_arg(ty: str, arg: str):
    if ty == "CUdeviceptr":
        return f"static_cast<CUdeviceptr>({arg})"
    return arg


def map_ty_to_pybind_ty(ty: str):
    mapping = {"CUdeviceptr": "int64_t"}
    if ty in mapping:
        return mapping[ty]
    return ty


def gen_pybind_args_with_full_args(m):
    return ", ".join([map_arg_to_pybind_arg(ty, arg) for ty, arg in zip(m.arg_ctypes, m.arg_names)])


def gen_pybind_signature_with_full_args(m):
    return ", ".join([f"{map_ty_to_pybind_ty(ty)} {arg}" for ty, arg in zip(m.arg_ctypes, m.arg_names)])


def make_global_pybind_with_algo_info(meta: KernelLinkerMeta, algo_info_struct: str, algo_info_schema) -> str:
    bind_struct = ""
    for name, ctype in algo_info_schema:
        bind_struct += f".def_readwrite(\"{name}\", &{algo_info_struct}::{name})"
    bind_struct += f".def_readwrite(\"num_warps\", &{algo_info_struct}::num_warps)"
    bind_struct += f".def_readwrite(\"num_stages\", &{algo_info_struct}::num_stages)"
    return f"""
static int _register_{meta.orig_kernel_name}_ops [[maybe_unused]] = []() {{
    OpInitRegistry::instance().register_one("{meta.orig_kernel_name}", [](py::module &m) {{
      py::class_<{algo_info_struct}>(m, "{algo_info_struct}")
        .def(py::init<>()){bind_struct};
      m.def(
        "{meta.orig_kernel_name}",
        [](
            int64_t stream,
            {gen_pybind_signature_with_full_args(meta)},
            {algo_info_struct} algo_info
        ){{
            {meta.orig_kernel_name}_ex(
                reinterpret_cast<CUstream>(stream),
                {gen_pybind_args_with_full_args(meta)},
                algo_info
            );
        }},
        py::arg("stream"),
        {', '.join([f'py::arg("{arg}")' for arg in meta.arg_names])},
        py::arg("algo_info")
    );
      
    }});
    return 0;
  }}();
    """


def make_algo_info_condition(algo_info_variable_name, algo_info_value):
    """
    algo_info.num_warps == 4 && algo_info.num_stages == 3
    """

    def _to_c_value(x):
        if isinstance(x, bool):
            return "true" if x else "false"
        return str(x)

    return " && ".join([f"{algo_info_variable_name}.{k}=={_to_c_value(v)}" for k, v in algo_info_value.items()])


def make_kernel_with_algo_info_param(
    tt_kernel_name: str,
    c_kernel_name: str,
    c_kernel_name_with_algo_infos,
    context,
) -> str:
    """
    algo_info_spaces: ["1024_warps4xstages3", ...]
    """
    meta = _take_a_meta(c_kernel_name_with_algo_infos)
    assert c_kernel_name == meta.orig_kernel_name
    schema = context[tt_kernel_name]["constexpr"]

    algo_info_variable_name = "algo_info"  # TODO(houqi.1993) do something to force no duplicate
    src = f"CUresult {c_kernel_name}_ex(CUstream stream, {gen_signature_with_full_args(meta)}, struct {make_kernel_algo_info_struct_name(c_kernel_name)} {algo_info_variable_name}){{\n"
    for c_kernel_name_with_algo_info in c_kernel_name_with_algo_infos.keys():
        algo_info = _get_algo_info(c_kernel_name_with_algo_info, c_kernel_name)
        algo_info_value = _make_triton_algo_info_with_schema(algo_info, schema)
        src += f"  if ({make_algo_info_condition(algo_info_variable_name, algo_info_value)}) {{\n"
        src += f"    return {c_kernel_name_with_algo_info}(stream, {', '.join(meta.arg_names)});\n"
        src += "  }\n"
    # TODO(houqi.1993) not supported code here
    params = " ".join([f"{key} = %d" for (key, dtype) in schema])
    args = ", ".join([f"{algo_info_variable_name}.{key}" for key, dtype in schema])

    src += f'  fprintf(stderr, "Error: kernel `{c_kernel_name}` algo_info not supported: {params}\\n", {args});\n'
    src += "  return CUDA_ERROR_NOT_SUPPORTED;\n"
    src += "}\n"
    return src


def _take_a_meta(x):
    if isinstance(x, dict):
        for v in x.values():
            return _take_a_meta(v)
    else:
        assert isinstance(x, list)
        return x[0]


def make_func_pointers_impl(c_kernel_name, c_kernel_name_with_algo_infos):
    kernel_func_t = f"{c_kernel_name}_kernel_func_t"

    def make_func_pointers(names: str, meta: KernelLinkerMeta) -> str:
        # the table of hint dispatchers
        src = f"typedef CUresult (*{kernel_func_t})(CUstream stream, {gen_signature_with_full_args(meta)});\n"
        src += f"{kernel_func_t} {meta.orig_kernel_name}_kernels[] = {{\n"
        for name in names:
            src += f"  {name},\n"
        src += "};\n"
        return src

    names = c_kernel_name_with_algo_infos.keys()

    return make_func_pointers(names, _take_a_meta(c_kernel_name_with_algo_infos))


def make_kernel_load_def_impl(c_kernel_name, c_kernel_name_with_algo_infos):
    names = c_kernel_name_with_algo_infos.keys()

    return make_kernel_load_def(names, _take_a_meta(c_kernel_name_with_algo_infos))


def link_all(workspace: Path, libname: str, context: Dict[str, Dict]):
    """
    4 levels with corresponding names:
     1. tt_kernel_name. e.g. "vecadd"
     2. c_kernel_name, also noted as orig_kernel_name. `c_kernel_name` is `tt_kernel_name` with all dtypes specified.
        c_kernel_name is the interface in the C Library.
        samples: "vecadd_fp16" and "vecadd_fp32".
     3. c_kernel_name_with_algo_info. `c_kernel_name_with_algo_info` is `c_kernel_name` with all tl.constexpr and runtime arguments(num_warps/num_stages) specified.
        such as vecadd_fp16_1024_warps4xstages3.
    4. c_kernel_name_impl: c_kernel_name_with_algo_info with all alignment hint. this is hided from the user.

    context: "vecadd" : {
        "kernel_names" : ["vecadd_fp16", "vecadd_fp32"],
        "constexpr" : [("BLOCK_SIZE", int)] # all constexpr in order
    }
    """
    logging.info(f"linking all kernels in {workspace} to {libname}.")
    parser = HeaderParser()
    headers = [x for x in workspace.glob("*.h") if x.stem != libname]
    includes = []
    for header in headers:
        h_path = Path(header)
        h_str = h_path.read_text()
        includes.append(h_path.name)
        parser.extract_linker_meta(h_str)

    #  mapping all kernels to jit.Function kernel names
    c2tt_kernel_mapping = {
        c_kernel_name: tt_kernel_name
        for tt_kernel_name, tt_kernel_info in context.items()
        for c_kernel_name in tt_kernel_info["kernel_names"]
    }
    # reorganize kernels by the upper levels in comment =>
    #  {"vecadd": {
    #     "vecadd_fp16": {
    #       "vecadd_fp16_1024_warps4xstages3": [],
    #       "vecadd_fp16_2048_warps4xstages3": []}}}
    kernels = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for name, meta_list in parser.kernels.items():
        meta = meta_list[0]
        tt_kernel_name = c2tt_kernel_mapping[meta.orig_kernel_name]
        kernels[tt_kernel_name][meta.orig_kernel_name][name] = meta_list

    out_path = workspace / libname

    # generate headers
    # per-c_kernel_name_with_algo_info
    algo_decls = [
        make_algo_decls(c_kernel_name_with_algo_info, meta_list)
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
        for c_kernel_name_with_algo_info, meta_list in c_kernel_with_algo_infos.items()
    ]
    # per c_kernel_name
    get_num_algos_decls = [
        make_get_num_algos_decl(_take_a_meta(c_kernel_with_algo_infos))
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    global_decls = [
        make_global_decl(_take_a_meta(c_kernel_with_algo_infos))
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]

    # per c_kernel_name
    kernel_algo_infos = [
        make_algo_info_decl(c_kernel_name, context[tt_kernel_name]["constexpr"])
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    # per c_kernel_name
    global_decls_with_algo_info = [
        make_global_decl_with_algo_info(
            _take_a_meta(c_kernel_with_algo_infos),  # meta.orig_kernel_name used
            make_kernel_algo_info_struct_name(c_kernel_name),
        )
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]

    content = "#include <cuda.h>\n"
    content += "#include <stdint.h>\n"
    content += "#include <stdbool.h>\n"
    content += """
    #pragma once
    #ifdef __cplusplus
    extern "C" {
    #endif"""
    content += "\n".join(algo_decls)
    content += "\n"
    content += "\n".join(get_num_algos_decls)
    content += "\n"
    content += "\n".join(kernel_algo_infos)
    content += "\n"
    content += "\n".join(global_decls)
    content += "\n".join(global_decls_with_algo_info)
    content += """
    #ifdef __cplusplus
    }
    #endif"""
    _write_if_changed(out_path.with_suffix(".h"), content)

    # generate sources

    # orig_kernel_name -> algo_infos
    orig_kernel_to_algo_infos = {}
    for name, meta_list in parser.kernels.items():
        meta = meta_list[0]
        if meta.orig_kernel_name not in orig_kernel_to_algo_infos:
            orig_kernel_to_algo_infos[meta.orig_kernel_name] = []
        orig_kernel_to_algo_infos[meta.orig_kernel_name].append(_get_algo_info(name, meta.orig_kernel_name))

    # per c_kernel_name_impl
    defs = [
        make_kernel_hints_dispatcher(c_kernel_name_with_algo_info, meta_list)
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
        for c_kernel_name_with_algo_info, meta_list in c_kernel_with_algo_infos.items()
    ]
    # per c_kernel_name
    func_pointers_defs = [
        make_func_pointers_impl(c_kernel_name, c_kernel_with_algo_infos)
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    meta_const_defs = [
        make_kernel_meta_const_dispatcher(_take_a_meta(c_kernel_with_algo_infos))
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    load_unload_defs = [
        make_kernel_load_def_impl(c_kernel_name, c_kernel_with_algo_infos)
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    get_num_algos_defs = [
        make_get_num_algos_def(_take_a_meta(c_kernel_with_algo_infos))
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    default_algo_kernels = [
        make_default_algo_kernel(_take_a_meta(c_kernel_with_algo_infos))
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    # per c_kernel_name_with_algo_info
    kernels_with_algo_info_param = [
        make_kernel_with_algo_info_param(
            tt_kernel_name,
            c_kernel_name,
            c_kernel_with_algo_infos,
            context,
        )
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]

    content = ""
    content += "#include <cuda.h>\n"
    content += "#include <stdint.h>\n"
    content += "#include <assert.h>\n"
    content += "#include <stdio.h>\n"
    content += f'#include "{libname}.h"\n'
    content += "\n"
    content += "\n".join(defs)
    content += "\n"
    content += "\n".join(func_pointers_defs)
    content += "\n"
    content += "\n".join(get_num_algos_defs)
    content += "\n"
    content += "\n".join(meta_const_defs)
    content += "\n"
    content += "\n".join(load_unload_defs)
    content += "\n"
    content += "\n".join(default_algo_kernels)
    content += "\n"
    content += "\n".join(kernels_with_algo_info_param)
    _write_if_changed(out_path.with_suffix(".c"), content)

    os.makedirs(workspace / "pybind", exist_ok=True)
    out_path = workspace / "pybind" / (libname + "_pybind")

    #include "registry.h"
    #include <c10/cuda/CUDAStream.h>
    #include "triton_aot_generated/triton_distributed_kernel.h"
    content = ""
    content += "#include \"registry.h\"\n"
    content += "#include <c10/cuda/CUDAStream.h>\n"
    content += f"#include \"{workspace}/{libname}.h\"\n"
    content += """
namespace distributed {

namespace ops {
    """

    global_pybind_with_algo_info = [
        make_global_pybind_with_algo_info(_take_a_meta(c_kernel_with_algo_infos),  # meta.orig_kernel_name used
                                          make_kernel_algo_info_struct_name(c_kernel_name),
                                          context[tt_kernel_name]["constexpr"])
        for tt_kernel_name, tt_kernel_with_dtypes in kernels.items()
        for c_kernel_name, c_kernel_with_algo_infos in tt_kernel_with_dtypes.items()
    ]
    content += "\n".join(global_pybind_with_algo_info)

    content += """
}  // ops

}  // distributed
    """

    _write_if_changed(out_path.with_suffix(".cc"), content)


@functools.lru_cache()
def libcuda_dirs():
    env_libcuda_path = os.getenv("TRITON_LIBCUDA_PATH")
    if env_libcuda_path:
        return [env_libcuda_path]

    libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode()
    # each line looks like the following:
    # libcuda.so.1 (libc6,x86-64) => /lib/x86_64-linux-gnu/libcuda.so.1
    locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [dir for dir in env_ld_library_path.split(":") if os.path.exists(os.path.join(dir, "libcuda.so.1"))]
    msg = "libcuda.so cannot found!\n"
    if locs:
        msg += "Possible files are located at %s." % str(locs)
        msg += "Please create a symlink of libcuda.so to any of the files."
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += " (requires sudo) to refresh the linker cache."
    assert any(os.path.exists(os.path.join(path, "libcuda.so.1")) for path in dirs), msg
    return dirs


@functools.lru_cache()
def triton_cuda_home():
    import triton.backends.nvidia

    return Path(triton.backends.nvidia.__file__).parent


@functools.lru_cache()
def library_dirs():
    return [triton_cuda_home() / "lib", *libcuda_dirs()]


@functools.lru_cache()
def aot_runtime_path():
    return Path(__file__).parent / "runtime"


@functools.lru_cache()
def include_dirs(with_runtime: bool = False):
    dirs = [str(triton_cuda_home() / "include")]
    if with_runtime:
        dirs.append(aot_runtime_path())
    return dirs


CMAKE_TEMPLATE = """
# DO NOT EDIT THIS FILE !!!
# This file is generated by distributed triton-AoT.
CMAKE_MINIMUM_REQUIRED(VERSION 3.17 FATAL_ERROR)
# add no project to allow
# PROJECT(TritonDistributed LANGUAGES C CXX)
FILE(GLOB C_FILES *.c)
SET(TRITON_DIST_AOT_RUNTIME_FILES {TRITON_DIST_AOT_RUNTIME_FILE})
SET(LIB_FILES
  ${{C_FILES}}
  ${{TRITON_DIST_AOT_RUNTIME_FILES}})
SET(CUDA_HOME {CUDA_HOME})

add_distributed_library({LIBNAME} ${{LIB_FILES}})
target_link_libraries({LIBNAME} PUBLIC
  -L${{CUDA_HOME}}/lib -ldl)
target_compile_options({LIBNAME} PUBLIC
  -I${{CUDA_HOME}}/include
)
install(TARGETS {LIBNAME}
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib)
"""


def gen_cmakelists(workspace: Path, libname: str, with_runtime: bool = False):
    logging.info(f"generating CMakeLists.txt in {workspace} for library {libname}")
    if with_runtime:
        _copy_if_changed(workspace / "triton_aot_runtime.cc", aot_runtime_path() / "triton_aot_runtime.cc")
        _copy_if_changed(workspace / "triton_aot_runtime.h", aot_runtime_path() / "triton_aot_runtime.h")

    content = CMAKE_TEMPLATE.format(
        LIBNAME=libname,
        CUDA_HOME=triton_cuda_home(),
        TRITON_DIST_AOT_RUNTIME_FILE="triton_aot_runtime.cc" if with_runtime else "",
    )
    _write_if_changed(workspace / "CMakeLists.txt", content)


def gen_kernel_library(workspace: Path):
    logging.info(f"building triton AoT library in {workspace}")
    command = ["cmake", "-S", ".", "-B", "build"]
    logging.info(f"cmake with command: {command}")
    subprocess.run(command, check=True, cwd=workspace)
    command = ["make", "-C", "build", "-j", "8"]
    logging.info(f"make with command: {command}")
    subprocess.run(command, check=True, cwd=workspace)


if __name__ == "__main__":

    def _parse_args():
        import argparse

        parser = argparse.ArgumentParser(description="Compile Triton kernels")
        parser.add_argument("--workspace", type=str, default="workspace")
        parser.add_argument(
            "--from-scratch",
            action="store_true",
            default=False,
            help="build from scratch. clean workspace first",
        )
        parser.add_argument(
            "--kernels",
            nargs="+",
            default=[f"{str(Path(__file__))}:add_kernel"],
            help="triton kernel for compile, in format: path/to/triton_kernel.py:kernel_name",
        )
        parser.add_argument(
            "--library",
            type=str,
            default="triton_distributed_kernel",
            help="output path for compiled library",
        )
        parser.add_argument(
            "--patch-cuda-stub",
            action=argparse.BooleanOptionalAction,
            help="patch mc support for triton kernels",
            default=True,
        )
        parser.add_argument(
            "--build",
            default=False,
            action="store_true",
        )
        return parser.parse_args()

    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s",
    )
    workspace = Path(args.workspace)
    if workspace.exists():
        logging.warning("please don't keep any thing important int AoT workspace!!! maybe overwrite or delete")
        if args.from_scratch:
            logging.warning(f"workspace {args.workspace} already exists, delete any files from workspace...")
            shutil.rmtree(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            logging.warning(f"workspace {args.workspace} already exists, will be overwritten")
    else:
        workspace.mkdir(parents=True, exist_ok=True)
    context = {}
    for kernel_ in args.kernels:
        kernel_path, kernel_name = kernel_.split(":")
        logging.info(f"loading kernel `{kernel_name}` from {kernel_path}")
        kernel_path = Path(kernel_path)
        sys.path.insert(0, str(kernel_path.parent))
        spec = importlib.util.spec_from_file_location(kernel_path.stem, kernel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        kernel = getattr(mod, kernel_name)
        logging.info(f"compiling {kernel_name}...")
        tmp_context = compile_kernel(kernel, workspace)
        context.update(tmp_context)

    link_all(
        workspace,
        args.library,
        context=context,
    )
    gen_cmakelists(workspace, args.library, args.patch_cuda_stub)
    if args.build:
        gen_kernel_library(workspace)
