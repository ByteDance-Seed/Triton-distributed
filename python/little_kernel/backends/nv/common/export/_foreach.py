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
"""Ragged foreach groups: per-element plans and single-launch rewrites.

Owns the fused (by-value table, grid.z element index) and packed
(block-prefix table, macro-virtualized) single-launch variants, including
their parameter-budget and reserved-identifier contracts, and the grouped
host binding for packed dispatch.
"""

from __future__ import annotations

import re

from little_kernel.backends.nv.common.export._frontend import _cuda_kernel_arg, _kernel_param_names, _launcher_arg_decl, _scalar_splits
from little_kernel.backends.nv.common.export._schema import _CONSTRUCTOR_CPP, _DTYPES, _SCALAR_COMPONENT, _SCALAR_RUNTIME_CPP

_FOREACH_FUSED_MAX = 32

_FOREACH_SCALAR_CPP = {"int32": "int32_t", "int64": "int64_t", "float32": "float", "float64": "double"}


def _foreach_fused_plan(spec: dict) -> dict:
    """Split kernel_args into per-element and launch-uniform parameters for
    the single-launch foreach ABI.

    Per-element slots: the foreach inputs themselves, every output that is
    freshly allocated per element or aliases a foreach input, and every
    scalar that is neither a runtime forward argument nor a constructor value
    (shape expressions may read element tensors). Everything else --
    non-foreach inputs, outputs aliasing non-foreach inputs, runtime scalars
    and constructor values -- is identical for every element and stays a
    normal kernel parameter.
    """
    foreach = set(spec["_foreach_inputs"])
    outputs = {item["name"]: item for item in spec["_outputs"]}
    runtime = {item["name"] for item in spec["scalars"] if item["_runtime"]}
    plan = {"elem_tensors": [], "elem_scalars": [], "uniform": []}
    for name in spec["_kernel_args"]:
        if name in spec["_descriptor_names"]:
            raise ValueError("single_launch foreach does not support TMA descriptors as kernel arguments")
        if name in spec["_tensor_types"]:
            alias = outputs.get(name, {}).get("alias_of")
            if name in foreach or (name in outputs and (alias is None or alias in foreach)):
                plan["elem_tensors"].append(name)
            else:
                plan["uniform"].append(name)
        elif name in spec["_constructor_types"] or name in runtime:
            plan["uniform"].append(name)
        else:
            dtype = spec["_scalar_types"][name]
            if dtype in _SCALAR_COMPONENT:
                raise ValueError("single_launch foreach does not support per-element complex scalars")
            if dtype not in _FOREACH_SCALAR_CPP:
                raise ValueError(f"single_launch foreach: unsupported scalar dtype {dtype}")
            plan["elem_scalars"].append(name)
    return plan


def _skip_literal(text: str, i: int) -> int:
    quote = text[i]
    i += 1
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == quote:
            return i + 1
        i += 1
    raise ValueError("unterminated literal in generated CUDA")


def _scan_past(text: str, i: int, open_ch: str, close_ch: str) -> int:
    """Return the index just past the closer that matches text[i].

    String/char literals and both comment forms are skipped, so raw CUDA text
    emitted by Builtin bodies cannot upset the delimiter balance.
    """
    depth = 0
    while i < len(text):
        char = text[i]
        if char in "\"'":
            i = _skip_literal(text, i)
            continue
        if text.startswith("//", i):
            i = text.index("\n", i)
            continue
        if text.startswith("/*", i):
            i = text.index("*/", i) + 2
            continue
        if char == open_ch:
            depth += 1
        elif char == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError("unbalanced delimiters in generated CUDA")


def _split_cxx_decls(params: str) -> list:
    """Split a C++ parameter list at top-level commas."""
    decls, start, depth = [], 0, 0
    for i, char in enumerate(params):
        if char in "(<[":
            depth += 1
        elif char in ")>]" and depth:
            depth -= 1
        elif char == "," and depth == 0:
            decls.append(params[start:i].strip())
            start = i + 1
    tail = params[start:].strip()
    if tail:
        decls.append(tail)
    return decls


_DECL_TAIL = re.compile(r"([A-Za-z_][A-Za-z_0-9]*)\s*$")


def _split_decl(decl: str) -> tuple:
    """Split 'const float* __restrict__ name' into (type text, name)."""
    match = _DECL_TAIL.search(decl)
    if match is None:
        raise ValueError(f"cannot parse generated parameter: {decl!r}")
    return decl[:match.start()].strip(), match.group(1)


def _entry_kernel_parts(source: str, renamed: str) -> dict:
    """Locate the entry __global__ kernel in generated CUDA and split it into
    signature prefix, parameter text, and body text."""
    match = re.search(r'extern\s+"C"\s+__global__[^{;]*\b' + re.escape(renamed) + r"\s*\(", source)
    if match is None:
        raise ValueError(f"generated CUDA does not define entry kernel {renamed}")
    open_paren = match.end() - 1
    close_paren = _scan_past(source, open_paren, "(", ")")
    brace = close_paren
    while source[brace] != "{":
        brace += 1
    body_end = _scan_past(source, brace, "{", "}")
    return {
        "prefix": source[match.start():open_paren], "params": source[open_paren + 1:close_paren - 1], "body":
        source[brace + 1:body_end - 1]
    }


_FUSED_RESERVED = ("tbl", "lk_e", "lk_i", "lk_count", "lk_gx", "lk_gy")

_FUSED_AXIS_Z = re.compile(r"\b(?:blockIdx|gridDim)\s*\.\s*z\b")


def _render_foreach_fused_cuda(spec: dict, source: str, renamed: str) -> str:
    """Emit the single-launch foreach variant of the entry kernel.

    The fused kernel carries a by-value table struct of per-element device
    pointers and scalars; element index rides grid.z, and per-element grid
    limits make elements with smaller grids return early instead of running
    out of bounds. The element kernel body is reused verbatim from this
    compiler's own codegen output, with the per-element parameters rebound as
    prologue locals (keeping any __restrict__ qualifier the entry parameter
    already carried). Launch-uniform parameters keep their original declared
    types, copied from the generated device signature.
    """
    plan = _foreach_fused_plan(spec)
    shared = spec["launch"]["_shared"]
    function = spec["_function"]
    max_group = spec.get("_foreach_max_group", _FOREACH_FUSED_MAX)
    # The fused launcher receives grid extents as runtime arguments but no
    # block channel: the single fused kernel must run with the entry's static
    # launch.block. A shape-dependent block expression would render host-side
    # names that are out of scope inside the launcher, so reject it here.
    block = spec["launch"]["_block_cpp"]
    if not all(re.fullmatch(r"[0-9]+", value) for value in block):
        raise ValueError("single_launch foreach requires constant launch.block dimensions; "
                         "got a shape-dependent block expression")
    parts = _entry_kernel_parts(source, renamed)
    split = [_split_decl(decl) for decl in _split_cxx_decls(parts["params"])]
    if _FUSED_AXIS_Z.search(parts["body"]):
        raise ValueError("single_launch foreach requires a frontend that does not read blockIdx.z/gridDim.z")
    by_name = {cname: ctype for ctype, cname in split}
    for reserved in _FUSED_RESERVED:
        if reserved in by_name:
            raise ValueError(f"single_launch foreach: kernel argument {reserved!r} is reserved")
    elem = []
    for name in plan["elem_tensors"]:
        if name not in by_name:
            raise ValueError(f"single_launch foreach: kernel argument {name} is not a parameter")
        ctype = by_name[name]
        restrict = "__restrict__" in ctype
        base = ctype.replace("__restrict__", " ").strip()
        if not base.endswith("*"):
            raise ValueError(f"single_launch foreach: {name} must lower to a pointer parameter")
        elem.append({"name": name, "field": base, "restrict": restrict, "kind": "tensor"})
    for name in plan["elem_scalars"]:
        if name not in by_name:
            raise ValueError(f"single_launch foreach: kernel argument {name} is not a parameter")
        expected = _FOREACH_SCALAR_CPP[spec["_scalar_types"][name]]
        if by_name[name] != expected:
            raise ValueError(f"single_launch foreach: scalar {name} lowers to {by_name[name]!r}, "
                             f"expected {expected!r} from its export dtype")
        elem.append({"name": name, "field": expected, "restrict": False, "kind": "scalar"})
    elem_names = set(plan["elem_tensors"] + plan["elem_scalars"])
    uniform_names = set(plan["uniform"])
    expected_uniform = []
    for name in plan["uniform"]:
        if _scalar_splits(spec, name):
            expected_uniform += [f"{name}_re", f"{name}_im"]
        else:
            expected_uniform.append(name)
    uniform_decls = []
    found = set()
    for ctype, cname in split:
        if cname in elem_names:
            continue
        base = cname[:-3] if cname.endswith(("_re", "_im")) else cname
        if cname in uniform_names or base in uniform_names:
            uniform_decls.append(f"{ctype} {cname}")
            found.add(cname)
    # The uniform kernel arguments must survive as declarations exactly.
    if found != set(expected_uniform):
        raise ValueError("single_launch foreach: uniform parameter classification mismatch")
    table_fields = [f"    {item['field']} {item['name']}[{max_group}];" for item in elem]
    table_fields += [f"    unsigned lk_gx[{max_group}];", f"    unsigned lk_gy[{max_group}];"]
    launch_bounds = re.search(r"__launch_bounds__\([^)]*\)", parts["prefix"])
    prologue = [
        "    const unsigned lk_e = blockIdx.z;",
        "    if (lk_e >= lk_count || blockIdx.x >= tbl.lk_gx[lk_e] || "
        "blockIdx.y >= tbl.lk_gy[lk_e]) return;",
    ]
    for item in elem:
        qualifier = " __restrict__" if item["restrict"] else ""
        prologue.append(f"    {item['field']}{qualifier} {item['name']} = tbl.{item['name']}[lk_e];")
    fused_decl = [f"const agent_lk_{function}_table tbl"] + uniform_decls + ["const unsigned lk_count"]
    fused_kernel = ("\nstruct agent_lk_" + function + "_table {\n" + "\n".join(table_fields) + "\n};\n\n" +
                    'extern "C" __global__ ' + (launch_bounds.group(0) + " " if launch_bounds else "") +
                    f"void {renamed}_fused(\n    " + ",\n    ".join(fused_decl) + "\n) {\n" + "\n".join(prologue) +
                    "\n" + parts["body"] + "}\n")
    array_params = [f"const void* const* lk_tbl_{item['name']}" for item in elem if item["kind"] == "tensor"]
    array_params += [f"const {item['field']}* lk_tbl_{item['name']}" for item in elem if item["kind"] == "scalar"]
    uniform_params = [_launcher_arg_decl(spec, name) for name in plan["uniform"]]
    uniform_kernel_args = [_cuda_kernel_arg(spec, name) for name in plan["uniform"]]
    loader = [
        f"        tbl.{item['name']}[lk_i] = "
        f"reinterpret_cast<{item['field']}>(const_cast<void*>(lk_tbl_{item['name']}[lk_i]));"
        if item["kind"] == "tensor" else f"        tbl.{item['name']}[lk_i] = lk_tbl_{item['name']}[lk_i];"
        for item in elem
    ]
    loader += ["        tbl.lk_gx[lk_i] = tbl_lk_gx[lk_i];", "        tbl.lk_gy[lk_i] = tbl_lk_gy[lk_i];"]
    attr_lines = []
    if shared > 48 * 1024:
        attr_lines = [
            f"    cudaError_t error = cudaFuncSetAttribute({renamed}_fused, "
            f"cudaFuncAttributeMaxDynamicSharedMemorySize, {shared});",
            "    if (error != cudaSuccess) return error;",
        ]
    launcher = "agent_lk_launch_" + function + "_fused"
    launcher_params = (array_params +
                       ["const unsigned* tbl_lk_gx", "const unsigned* tbl_lk_gy", "const unsigned lk_count"] +
                       uniform_params + ["const unsigned grid_x", "const unsigned grid_y", "cudaStream_t stream"])
    call_args = (["tbl"] + [item for arg in uniform_kernel_args for item in [arg]] + ["lk_count"])
    fused_launcher = (f'\nextern "C" cudaError_t {launcher}(\n    ' + ",\n    ".join(launcher_params) + "\n) {\n" +
                      ("\n".join(attr_lines) + "\n" if attr_lines else "") + f"    agent_lk_{function}_table tbl;\n" +
                      "    for (unsigned lk_i = 0; lk_i < lk_count; ++lk_i) {\n" + "\n".join(loader) + "\n    }\n" +
                      f"    {renamed}_fused<<<dim3(grid_x, grid_y, lk_count), " +
                      f"dim3({block[0]}, {block[1]}, {block[2]}), {shared}, stream>>>(\n        " +
                      ",\n        ".join(call_args) + ");\n" + "    return cudaGetLastError();\n}\n")
    return fused_kernel + fused_launcher


_PACK_PARAM_BUDGET = 3904

_CPP_SCALAR_SIZES = {
    "bool": 1,
    "char": 1,
    "signed char": 1,
    "unsigned char": 1,
    "short": 2,
    "unsigned short": 2,
    "int": 4,
    "unsigned": 4,
    "int32_t": 4,
    "uint32_t": 4,
    "float": 4,
    "long": 8,
    "unsigned long": 8,
    "long long": 8,
    "int64_t": 8,
    "uint64_t": 8,
    "double": 8,
}

_PACK_RESERVED = ("lk_pack", "lk_plan", "lk_vb", "lk_vg", "lk_bx", "lk_g", "lk_hi", "lk_mid")

_GRID_SYNC_TOKENS = ("this_grid", "grid.sync", "cudaGridDependency", "cudaTriggerProgrammaticLaunch")


def _renamed_kernel(spec: dict) -> str:
    original = spec["_kernel"]
    return original if original.startswith("agent_") else f"agent_{original}"


def _validate_foreach_single(data: dict) -> None:
    """Structural preconditions for packed dispatch (grouped stays available)."""
    names = list(data["_foreach_inputs"])
    used = []
    for item in data["_outputs"]:
        alias = item.get("alias_of")
        if alias not in names or alias in used:
            raise ValueError('foreach dispatch "single" requires every output to alias_of a '
                             "distinct foreach input (the packed binding returns the aliased "
                             "group tensors without allocating fresh outputs)")
        used.append(alias)
    if data.get("workspaces"):
        raise ValueError('foreach dispatch "single" does not support workspaces; use "grouped"')
    if (data.get("transforms") or data.get("pre_transforms") or data.get("post_transforms")):
        raise ValueError('foreach dispatch "single" does not support tensor transforms; use "grouped"')
    if data["_descriptor_names"]:
        raise ValueError('foreach dispatch "single" does not support TMA descriptors; use "grouped"')
    if data["launch"]["_cluster"]:
        raise ValueError('foreach dispatch "single" does not support cluster launches; use "grouped"')
    for name in _kernel_param_names(data):
        if name.startswith("lk_"):
            raise ValueError(f'foreach dispatch "single" reserves the "lk_" prefix for packed-'
                             f'dispatch prologues; rename kernel argument {name}')
    data["_kernel_param_check"] = True


def _sizeof_cpp_param(type_text: str) -> int:
    """Host-known sizeof for a generated kernel parameter type."""
    text = type_text.strip()
    if text == "CUtensorMap":
        return 128
    if "*" in text:
        return 8
    bare = text[6:].strip() if text.startswith("const ") else text
    if bare in _CPP_SCALAR_SIZES:
        return _CPP_SCALAR_SIZES[bare]
    raise ValueError(f'foreach dispatch "single": cannot size kernel parameter type "{type_text}"')


def _scan_delimited(text: str, start: int, open_ch: str, close_ch: str) -> int:
    """Index just past the close delimiter matching text[start] == open_ch.

    Skips // and /* */ comments and "..." / '...' literals so braces inside
    inline-PTX strings do not disturb the count.
    """
    assert text[start] == open_ch
    depth, i = 0, start
    line_comment = block_comment = in_string = in_char = escape = False
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if line_comment:
            if c == "\n":
                line_comment = False
        elif block_comment:
            if c == "*" and nxt == "/":
                block_comment = False
                i += 1
        elif in_string:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
        elif in_char:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == "'":
                in_char = False
        elif c == "/" and nxt == "/":
            line_comment = True
            i += 1
        elif c == "/" and nxt == "*":
            block_comment = True
            i += 1
        elif c == '"':
            in_string = True
        elif c == "'":
            in_char = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    raise ValueError(f"unbalanced {open_ch}{close_ch} in generated CUDA source")


def _blank_comments_and_strings(text: str) -> str:
    """Copy of text with comment/string contents replaced by spaces (for token scans)."""
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        if c == "/" and nxt == "/":
            while i < len(text) and text[i] != "\n":
                out.append(" ")
                i += 1
            continue
        if c == "/" and nxt == "*":
            out += [" ", " "]
            i += 2
            while i < len(text) and not (text[i] == "*" and i + 1 < len(text) and text[i + 1] == "/"):
                out.append("\n" if text[i] == "\n" else " ")
                i += 1
            if i < len(text):
                out += [" ", " "]
                i += 2
            continue
        if c in "\"'":
            quote = c
            out.append(" ")
            i += 1
            while i < len(text) and text[i] != quote:
                if text[i] == "\\" and i + 1 < len(text):
                    out.append(" ")
                    i += 1
                out.append("\n" if i < len(text) and text[i] == "\n" else " ")
                i += 1
            if i < len(text):
                out.append(" ")
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out)


def _extract_entry_kernel(source: str, renamed: str):
    """Locate the compiled entry kernel inside generated CUDA source.

    Returns (launch_bounds, params, body, span) where launch_bounds is the
    optional "__launch_bounds__(...)" clause (or ""), params a list of
    (type_text, name), body the text between the outer braces, and span the
    (start, end) offsets of the whole kernel definition.
    """
    match = re.search(rf"__global__\b([^{{;]*?)\bvoid\s+{re.escape(renamed)}\s*\(", source)
    if match is None:
        raise ValueError(f'foreach dispatch "single": cannot locate entry kernel {renamed}')
    bounds = match.group(1).strip()
    open_paren = match.end() - 1
    close_paren = _scan_delimited(source, open_paren, "(", ")")
    j = close_paren
    while j < len(source) and source[j].isspace():
        j += 1
    if j >= len(source) or source[j] != "{":
        raise ValueError(f'foreach dispatch "single": entry kernel {renamed} has no inline body')
    end = _scan_delimited(source, j, "{", "}")
    body = source[j + 1:end - 1]
    params = []
    for piece in _split_cxx_decls(source[open_paren + 1:close_paren - 1]):
        if not piece or piece == "void":
            continue
        name_match = re.search(r"([A-Za-z_]\w*)\s*$", piece)
        if name_match is None:
            raise ValueError(f'foreach dispatch "single": cannot parse kernel parameter "{piece}"')
        params.append((piece[:name_match.start()].strip(), name_match.group(1)))
    return bounds, params, body, (match.start(), end)


def _pack_classification(spec: dict, params: list):
    """Split parsed kernel params into group tables, group scalars and pass-throughs."""
    foreach_set = set(spec["_foreach_inputs"])
    group_scalars = set(spec["_foreach_group_scalars"])
    alias_of = {item["name"]: item["alias_of"] for item in spec["_outputs"] if item.get("alias_of")}
    tables, scalars, shared = [], [], []
    for type_text, name in params:
        if name in foreach_set or name in alias_of:
            if "*" not in type_text:
                raise ValueError(f'foreach dispatch "single": per-group kernel parameter {name} '
                                 f'must be a pointer, got "{type_text}"')
            source_name = name if name in foreach_set else alias_of[name]
            tables.append((type_text, name, source_name))
        elif name in group_scalars:
            scalars.append((type_text, name))
        else:
            shared.append((type_text, name))
    return tables, scalars, shared


def _pack_layout(spec: dict, tables: list, scalars: list, shared: list) -> int:
    """Packed-struct group capacity inside the kernel-parameter budget."""
    shared_bytes = sum(_sizeof_cpp_param(type_text) for type_text, _ in shared)
    per_group = 4 + 8 * len(tables) + sum(_sizeof_cpp_param(type_text) for type_text, _ in scalars)
    cap = (_PACK_PARAM_BUDGET - shared_bytes - 8) // per_group
    if cap < 1:
        raise ValueError('foreach dispatch "single": one group of tables exceeds the kernel-'
                         "parameter budget; use dispatch \"grouped\"")
    return min(cap, 4096)


def _pack_struct_text(spec: dict) -> str:
    renamed = _renamed_kernel(spec)
    cap = spec["_pack_cap"]
    lines = [f"struct LkPack_{renamed} {{", "    int32_t count;", f"    int32_t prefix[{cap} + 1];"]
    for type_text, name, _ in spec["_pack_tables"]:
        lines.append(f"    {type_text} {name}[{cap}];")
    for type_text, name in spec["_pack_scalars"]:
        lines.append(f"    {type_text} {name}[{cap}];")
    lines.append("};")
    return "\n".join(lines) + "\n"


def _render_foreach_single_cuda(spec: dict, source: str, renamed: str) -> str:
    """Packed struct + virtualized packed kernel + packed launcher (CUDA side)."""
    bounds, params, body, (start, end) = _extract_entry_kernel(source, renamed)
    rest = _blank_comments_and_strings(source[:start] + source[end:])
    if re.search(r"\b(blockIdx|gridDim)\b", rest):
        raise ValueError('foreach dispatch "single" cannot virtualize blockIdx/gridDim reads '
                         "inside __device__ helpers or extra kernels; keep block-index math in "
                         'the entry kernel body or use dispatch "grouped"')
    clean_body = _blank_comments_and_strings(body)
    if any(token in clean_body for token in _GRID_SYNC_TOKENS):
        raise ValueError('foreach dispatch "single" does not support cooperative grid '
                         'synchronization in the entry kernel; use dispatch "grouped"')
    if re.search(r"\b(" + "|".join(_PACK_RESERVED) + r")\b", clean_body):
        raise ValueError('foreach dispatch "single" reserves the identifiers ' + ", ".join(_PACK_RESERVED) +
                         " for the packed prologue; rename the local or use dispatch \"grouped\"")
    tables, scalars, shared = _pack_classification(spec, params)
    spec["_pack_tables"] = tables
    spec["_pack_scalars"] = scalars
    spec["_pack_shared"] = shared
    spec["_pack_cap"] = _pack_layout(spec, tables, scalars, shared)
    shared_kernel_args = [
        name for name in spec["_kernel_args"]
        if all(name != t[1] for t in tables) and name not in spec["_foreach_group_scalars"]
    ]
    spec["_pack_shared_kernel_args"] = shared_kernel_args
    bounds_clause = (bounds + " ") if bounds else ""
    shared_decls = ", ".join(f"{t} {n}" for t, n in shared)
    prologue = [
        "    int lk_g = 0;",
        "    int lk_hi = lk_pack.count - 1;",
        "    const int lk_bx = static_cast<int>(blockIdx.x);",
        "    while (lk_g < lk_hi) {",
        "        const int lk_mid = (lk_g + lk_hi + 1) >> 1;",
        "        if (lk_pack.prefix[lk_mid] <= lk_bx) { lk_g = lk_mid; }",
        "        else { lk_hi = lk_mid - 1; }",
        "    }",
        "    const dim3 lk_vb(static_cast<unsigned>(lk_bx - lk_pack.prefix[lk_g]),"
        " blockIdx.y, blockIdx.z);",
        "    const dim3 lk_vg(static_cast<unsigned>(lk_pack.prefix[lk_g + 1]"
        " - lk_pack.prefix[lk_g]), gridDim.y, gridDim.z);",
    ]
    for type_text, name, _ in tables:
        prologue.append(f"    {type_text} {name} = lk_pack.{name}[lk_g];")
    for type_text, name in scalars:
        prologue.append(f"    {type_text} {name} = lk_pack.{name}[lk_g];")
    packed_kernel = (f"__global__ {bounds_clause}void {renamed}_packed(LkPack_{renamed} lk_pack" +
                     (", " + shared_decls if shared_decls else "") + ") {\n" + "\n".join(prologue) + "\n"
                     "    {\n"
                     "#define blockIdx lk_vb\n"
                     "#define gridDim lk_vg\n" + body + "\n#undef blockIdx\n"
                     "#undef gridDim\n"
                     "    }\n"
                     "}\n")
    launcher = "agent_lk_launch_" + spec["_function"]
    launcher_decls = ([f"LkPack_{renamed} lk_pack"] + [_launcher_arg_decl(spec, name) for name in shared_kernel_args] +
                      [f"unsigned grid_{axis}" for axis in "xyz"] + ["cudaStream_t stream"])
    call_args = [_cuda_kernel_arg(spec, name) for name in shared_kernel_args]
    block = spec["launch"]["_block"]
    shared = spec["launch"]["_shared"]
    launch_lines = []
    if shared > 48 * 1024:
        launch_lines += [
            f"    cudaError_t error = cudaFuncSetAttribute({renamed}_packed, "
            f"cudaFuncAttributeMaxDynamicSharedMemorySize, {shared});",
            "    if (error != cudaSuccess) return error;",
        ]
    launch_lines += [
        f"    {renamed}_packed<<<dim3(grid_x, grid_y, grid_z), "
        f"dim3({block[0]}, {block[1]}, {block[2]}), {shared}, stream>>>(",
        "        lk_pack" + (",\n        " + ",\n        ".join(call_args) if call_args else "") + ");",
        "    return cudaGetLastError();",
    ]
    packed_launcher = (f'\nextern "C" cudaError_t {launcher}_packed(\n' + ",\n".join("    " + decl
                                                                                     for decl in launcher_decls) +
                       "\n) {\n" + "\n".join(launch_lines) + "\n}\n")
    return ("\n// Packed single-launch dispatch for the foreach group: one launch\n"
            "// covers all groups; each block recovers (group, local blockIdx.x)\n"
            "// by binary search over the block-prefix table.\n" + _pack_struct_text(spec) + "\n" + packed_kernel +
            packed_launcher)


def _render_foreach_single_binding(spec: dict, launcher: str) -> tuple[str, str]:
    """Host-side packed foreach binding (dispatch "single"): fill the packed
    struct on the host, issue ONE packed launch, and return the aliased
    per-group output lists.  The same host expressions (shape checks, scalar
    and grid evaluation) are evaluated per group exactly as dispatch
    "grouped" evaluates them inside its per-group calls, so the merged
    launch observes identical arguments; only the launch topology changes.
    """
    inputs = spec["inputs"]
    outputs = spec["_outputs"]
    foreach = list(spec["_foreach_inputs"])
    selected = set(foreach)
    function = "agent_" + spec["_function"]
    single = function + "_single"
    renamed = _renamed_kernel(spec)
    struct = "LkPack_" + renamed
    cap = spec["_pack_cap"]
    group_scalars = set(spec["_foreach_group_scalars"])
    first = foreach[0]
    first_is_foreach = inputs[0]["name"] in selected

    def input_checks(item, inside):
        name = item["name"]
        dtype = _DTYPES[item["dtype"]][1]
        lines = [f'TORCH_CHECK({name}.is_cuda(), "{name} must be on CUDA");']
        if item.get("contiguous", True):
            lines.append(f'TORCH_CHECK({name}.is_contiguous(), "{name} must be contiguous");')
        lines += [
            f'TORCH_CHECK({name}.dim() == {item["rank"]}, "{name} rank mismatch");',
            f'TORCH_CHECK({name}.scalar_type() == {dtype}, "{name} dtype mismatch");'
        ]
        if name != inputs[0]["name"]:
            anchor = inputs[0]["name"]
            if not inside and first_is_foreach:
                anchor = inputs[0]["name"] + "_list[0]"
            lines.append(f'TORCH_CHECK({name}.device() == {anchor}.device(), '
                         '"inputs must share a device");')
        return lines

    def scalar_eval_lines(item):
        name = item["name"]
        if item["_dtype"] == "float32":
            return [
                f"const float {name} = static_cast<float>({item['_expr_cpp']});",
                f'TORCH_CHECK(std::isfinite({name}), "{name} must be finite");'
            ]
        if item["_dtype"] == "float64":
            return [
                f"const double {name} = static_cast<double>({item['_expr_cpp']});",
                f'TORCH_CHECK(std::isfinite({name}), "{name} must be finite");'
            ]
        bound = "INT64_MAX" if item["_dtype"] == "int64" else "INT32_MAX"
        return [
            f"const int64_t {name} = {item['_expr_cpp']};",
            f'TORCH_CHECK({name} >= 0 && {name} <= {bound}, "{name} exceeds range");'
        ]

    declaration = [
        f"std::vector<torch::Tensor> {item['name']}_list"
        if item["name"] in selected else f"torch::Tensor {item['name']}" for item in inputs
    ]
    declaration += [
        _SCALAR_RUNTIME_CPP[item["_dtype"]] + " " + item["name"] for item in spec["scalars"] if item["_runtime"]
    ]
    declaration += [
        _CONSTRUCTOR_CPP[item["dtype"]] + " " + item["name"]
        for item in spec["constructor"]
        if item["dtype"] != "ignored"
    ]

    vector_type = "std::vector<torch::Tensor>"
    return_type = (vector_type if len(outputs) == 1 else "std::tuple<" + ", ".join(vector_type for _ in outputs) + ">")

    body = [
        f'    TORCH_CHECK({name}_list.size() == {first}_list.size(), '
        f'"group count mismatch: {name}");' for name in foreach[1:]
    ]
    body.append(f'    const int64_t lk_count = static_cast<int64_t>({first}_list.size());')
    body.append(f'    TORCH_CHECK(lk_count >= 1 && lk_count <= {cap}, '
                '"group count exceeds packed-dispatch table capacity");')
    for item in inputs:
        if item["name"] not in selected:
            body += ["    " + line for line in input_checks(item, inside=False)]
    # Shared (non-group) scalars are group-independent: evaluate them once,
    # outside the group loop, exactly like the grouped path evaluates the
    # same expression per call.
    for item in spec["scalars"]:
        if not item["_runtime"] and item["name"] not in group_scalars:
            body += ["    " + line for line in scalar_eval_lines(item)]
    body.append(f"    {struct} lk_pack{{}};")
    body.append("    lk_pack.count = static_cast<int32_t>(lk_count);")
    body.append("    int64_t lk_total = 0;")
    body.append("    int64_t lk_grid_y = 1;")
    body.append("    int64_t lk_grid_z = 1;")
    body.append("    for (int64_t lk_group = 0; lk_group < lk_count; ++lk_group) {")
    for item in inputs:
        if item["name"] in selected:
            body.append(f'        const torch::Tensor& {item["name"]} = '
                        f'{item["name"]}_list[lk_group];')
    for item in inputs:
        if item["name"] in selected:
            body += ["        " + line for line in input_checks(item, inside=True)]
    for left, right in spec["_equalities_cpp"]:
        body.append(f'        TORCH_CHECK({left} == {right}, '
                    f'"shape equality failed: {left} == {right}");')
    for item in spec["scalars"]:
        if not item["_runtime"] and item["name"] in group_scalars:
            body += ["        " + line for line in scalar_eval_lines(item)]
    grid_x, grid_y, grid_z = spec["launch"]["_grid_cpp"]
    body.append(f"        const int64_t lk_gx = {grid_x};")
    body.append('        TORCH_CHECK(lk_gx >= 0 && lk_total + lk_gx <= INT32_MAX, '
                '"packed grid.x exceeds int32 range");')
    body.append(f"        const int64_t lk_gy = {grid_y};")
    body.append(f"        const int64_t lk_gz = {grid_z};")
    body.append('        TORCH_CHECK(lk_gy > 0 && lk_gy <= UINT32_MAX && lk_gz > 0 '
                '&& lk_gz <= UINT32_MAX, "invalid grid y/z");')
    body.append("        if (lk_group == 0) {")
    body.append("            lk_grid_y = lk_gy;")
    body.append("            lk_grid_z = lk_gz;")
    body.append("        }")
    body.append('        TORCH_CHECK(lk_gy == lk_grid_y && lk_gz == lk_grid_z, '
                '"foreach dispatch \\"single\\" requires uniform grid y/z across groups");')
    body.append("        lk_pack.prefix[lk_group] = static_cast<int32_t>(lk_total);")
    body.append("        lk_total += lk_gx;")
    for type_text, name, source_name in spec["_pack_tables"]:
        base = re.sub(r"\s*__restrict__", "", type_text)
        body.append(f"        lk_pack.{name}[lk_group] = "
                    f"reinterpret_cast<{base}>({source_name}.data_ptr());")
    for type_text, name in spec["_pack_scalars"]:
        body.append(f"        lk_pack.{name}[lk_group] = {name};")
    body.append("    }")
    body.append("    lk_pack.prefix[lk_count] = static_cast<int32_t>(lk_total);")
    body.append("    if (lk_total > 0) {")
    body.append("        cudaStream_t lk_stream = c10::cuda::getCurrentCUDAStream().stream();")
    call_args = []
    for name in spec["_pack_shared_kernel_args"]:
        if name in spec["_list_param_types"]:
            call_args.append(name)
        elif name in spec["_tensor_types"]:
            call_args.append(f"{name}.data_ptr()")
        elif _scalar_splits(spec, name):
            call_args += [f"{name}.real()", f"{name}.imag()"]
        else:
            call_args.append(name)
    packed_call = "lk_pack" + (",\n            " + ",\n            ".join(call_args) if call_args else "")
    body.append(f"        cudaError_t error = {launcher}_packed(\n"
                f"            {packed_call},\n"
                "            static_cast<unsigned>(lk_total), "
                "static_cast<unsigned>(lk_grid_y), "
                "static_cast<unsigned>(lk_grid_z), lk_stream);")
    body.append("        TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));")
    body.append("    }")
    out_names = [f"lk_packed_out_{index}" for index in range(len(outputs))]
    for out_name, item in zip(out_names, outputs):
        body.append(f'    {vector_type} {out_name}({item["alias_of"]}_list);')
    if len(outputs) == 1:
        body.append(f"    return {out_names[0]};")
    else:
        body.append(f"    return {return_type}(" + ", ".join(f"std::move({name})" for name in out_names) + ");")

    shared_decls = ", ".join(_launcher_arg_decl(spec, name) for name in spec["_pack_shared_kernel_args"])
    extern_decl = (f'extern "C" cudaError_t {launcher}_packed(\n'
                   f"    {struct} lk_pack" +
                   (",\n    " + shared_decls.replace(", ", ",\n    ") if shared_decls else "") +
                   ",\n    unsigned grid_x,\n    unsigned grid_y,\n    unsigned grid_z,"
                   "\n    cudaStream_t stream);")
    source = ("// Packed single-launch foreach binding: the host fills the per-group\n"
              "// tables (block prefix, pointers, group scalars) with the same host\n"
              "// expressions the grouped binding evaluates per group, then issues\n"
              "// one packed launch. Output lists alias the foreach inputs exactly.\n" + _pack_struct_text(spec) +
              "\n" + extern_decl + "\n\n" + f"static {return_type} {single}(\n" +
              ",\n".join("    " + decl for decl in declaration) + ") {\n" + "\n".join(body) + "\n}\n\n")
    names = [f"{name}_list" if name in selected else name for name in [item["name"] for item in inputs]]
    names += [item["name"] for item in spec["scalars"] if item["_runtime"]]
    names += [item["name"] for item in spec["constructor"] if item["dtype"] != "ignored"]
    arg_list = ", ".join(f'pybind11::arg("{name}")' for name in names)
    register = f'    m.def("{single}", &{single}, {arg_list});\n'
    return source, register
