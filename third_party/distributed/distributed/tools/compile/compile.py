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
import binascii
import hashlib
from pathlib import Path
from typing import Dict, List

import triton
from packaging.version import Version

_TRITON_VER = Version(triton.__version__)
_IS_NEW_TRITON = _TRITON_VER.major >= 3 and _TRITON_VER.minor >= 2
"""
some API changes:
 triton.compiler.ASTSource
    - signature:
        Dict[int, anotation] for 3.0.0
        Dict[name, anotation] for 3.2.0
    - constants: Dict[int, Any]
        Dict[int, anotation] for 3.0.0
        Dict[name, anotation] for 3.2.0
    - attrs: triton.compiler.AttrsDescriptor
"""


def hash_signature(signature: List[str]):
    m = hashlib.sha256()
    m.update(" ".join(signature).encode())
    return m.hexdigest()[:8]


def _meta_sig(num_stages: int, num_warps: int) -> str:
    meta_sig = f"warps{num_warps}xstages{num_stages}"
    return meta_sig


def constexpr(s):
    try:
        ret = int(s)
        return ret
    except ValueError:
        pass
    try:
        if s.lower() in ["true", "false"]:
            return 1 if s.lower() == "true" else 0
    except ValueError:
        pass
    try:
        ret = float(s)
        return ret
    except ValueError:
        pass
    return None


def make_ast_source_legacy(kernel: triton.JITFunction, signature: str) -> triton.compiler.CompiledKernel:
    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), signature.split(",")))

    hints = {i: constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}
    constants = {i: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {i: s.split(":")[0] for i, s in enumerate(signature) if i not in constants}

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"
    divisible_by_16 = [i for i, h in hints.items() if h == 16]
    equal_to_1 = [i for i, h in hints.items() if h == 1]
    attrs = triton.compiler.AttrsDescriptor(divisible_by_16=divisible_by_16, equal_to_1=equal_to_1)
    for i in equal_to_1:
        constants.update({i: 1})
    src = triton.compiler.ASTSource(fn=kernel, constants=constants, signature=signature, attrs=attrs)
    return src, hints


def make_ast_source_new(
    kernel: triton.JITFunction,
    signature: str,
) -> triton.compiler.CompiledKernel:
    # validate and parse signature
    signature = list(map(lambda s: s.strip(" "), signature.split(",")))

    hints = {(i, ): constexpr(s.split(":")[1]) for i, s in enumerate(signature) if ":" in s}
    hints = {k: v for k, v in hints.items() if v is not None}

    constants = {kernel.arg_names[i]: constexpr(s) for i, s in enumerate(signature)}
    constants = {k: v for k, v in constants.items() if v is not None}
    signature = {kernel.arg_names[i]: s.split(":")[0] for i, s in enumerate(signature)}

    # compile ast into cubin
    for h in hints.values():
        assert h in [1, 16], f"Only 1 and 16 are valid hints, got {h}"

    # for key, value in hints.items():
    #     if value == 1:
    #         constants[kernel.arg_names[key[0]]] = value

    for key in constants:
        signature[key] = "constexpr"
    attrs = {k: [["tt.divisibility", 16]] for k, v in hints.items() if v == 16}
    attrs.update({k: [["tt.constancy", 1]] for k, v in hints.items() if v == 1})
    src = triton.compiler.ASTSource(fn=kernel, constexprs=constants, signature=signature, attrs=attrs)
    return src, hints


def make_ast_source(kernel: triton.JITFunction, signature: str) -> triton.compiler.CompiledKernel:
    if _IS_NEW_TRITON:
        return make_ast_source_new(kernel, signature)
    else:
        return make_ast_source_legacy(kernel, signature)


def kernel_name_suffix(
    signature,
    const_sig,
    hints,
    num_stages: int,
    num_warps: int,
):
    meta_sig = f"warps{num_warps}xstages{num_stages}"
    sig_hash = hash_signature(list(signature.values()) + [const_sig] + [meta_sig])
    suffix = ''
    for i, ty in enumerate(signature.values()):
        suffix += str(i)
        if hints.get((i, ), None) == 1:
            suffix += 'c'
        if hints.get((i, ), None) == 16:
            suffix += 'd'
    return f"{sig_hash}_{suffix}"


def _indexed_signature(src: triton.compiler.ASTSource) -> Dict[int, str]:
    if not _IS_NEW_TRITON:
        return src.signature
    signature = {}
    for i, params in enumerate(src.fn.params):
        signature[i] = src.signature[params.name]
    return signature


def _make_const_sig(src: triton.compiler.ASTSource) -> str:
    constants = []
    # indexed_constants = _indexed_constants(src)
    indexed_constants = src.constants
    for i, params in enumerate(src.fn.params):
        if params.is_constexpr:
            constants.append(indexed_constants[(i, )])
    return "x".join([str(v) for v in constants])


def materialize_c_params(
    kernel,
    hints,
    ccinfo,
    kernel_name,
    out_name,
    num_warps,
    num_stages,
    grid: List[int],
):
    from triton.backends.nvidia.driver import ty_to_cpp

    src = ccinfo.src
    constants = ccinfo.src.constants
    signature = _indexed_signature(src)
    const_sig = _make_const_sig(src)
    suffix = kernel_name_suffix(signature, const_sig, hints, num_stages, num_warps)
    func_name = f"{out_name}_{suffix}"

    doc_string = [f"{kernel.arg_names[i[0]]}={constants[i]}" for i in constants]
    doc_string += [f"num_warps={num_warps}", f"num_stages={num_stages}"]

    assert len(grid) == 3, f"{grid}"
    arg_names = []
    arg_types = []
    for i in signature.keys():
        if not src.fn.params[i].is_constexpr:
            arg_names += [kernel.arg_names[i]]
            arg_types += [signature[i]]

    # dump C stub code
    hex_ = str(binascii.hexlify(ccinfo.asm["cubin"]))[2:-1]
    params = {
        "kernel_name":
        func_name,
        "triton_kernel_name":
        kernel_name,
        "bin_size":
        len(hex_),
        "bin_data":
        ", ".join([f"0x{x}{y}" for x, y in zip(hex_[::2], hex_[1::2])]),
        "signature":
        ", ".join([f"{ty_to_cpp(ty)} {name}" for name, ty in zip(arg_names, arg_types)]),
        "full_signature":
        ", ".join([
            f"{ty_to_cpp(signature[i])} {kernel.arg_names[i]}" for i in signature.keys()
            if not src.fn.params[i].is_constexpr
        ]),
        "arg_pointers":
        ", ".join([f"&{arg}" for arg in arg_names]),
        "num_args":
        len(arg_names),
        "kernel_docstring":
        doc_string,
        "shared":
        ccinfo.metadata.shared,
        "num_warps":
        num_warps,
        "algo_info":
        "_".join([const_sig, _meta_sig(num_stages, num_warps)]),
        "gridX":
        grid[0],
        "gridY":
        grid[1],
        "gridZ":
        grid[2],
        "_placeholder":
        "",
    }
    return params, suffix


def dump_c_code(out_path: Path, params):
    for ext in ["h", "c"]:
        template_path = Path(__file__).parent / f"compile.{ext}"
        with out_path.with_suffix(f".{ext}").open("w") as fp:
            fp.write(Path(template_path).read_text().format(**params))
