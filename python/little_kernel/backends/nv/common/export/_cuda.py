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
"""CUDA translation-unit emission.

Compiles the authored entry through the public CUDA pipeline, enforces the
entry-parameter and TMA transport ABI, applies the contract-driven alias
policy, computes the dynamic-shared-memory floor, fans derived constexpr
selectors out into baked variants, and emits the entry kernel(s), builtin
transform helpers, launcher and both single-launch foreach rewrites.
"""

from __future__ import annotations

import ast
import inspect
import itertools
import json
import re
import textwrap
from pathlib import Path

from little_kernel.backends.nv.common.codegen.emitter import ALIAS_CONTRACT_CTX_KEY, CUDA_LAUNCH_BOUNDS_MIN_BLOCKS_CTX_KEY, codegen_cuda
from little_kernel.core.passes import PASSES

from little_kernel.backends.nv.common.export._foreach import _render_foreach_fused_cuda, _render_foreach_single_cuda
from little_kernel.backends.nv.common.export._frontend import _cuda_kernel_arg, _declared_tensor_dtypes, _descriptor_param_kind, _kernel_param_names, _launcher_arg_decl
from little_kernel.backends.nv.common.export._schema import _DTYPES
from little_kernel.backends.nv.common.export._shapes import _shape_expr, _split_cuda_prelude

_TRANSPOSE_CUDA = r'''
template <typename T>
__global__ void agent_lk_transpose2d_kernel(const T* src, T* dst, int rows, int cols) {
    __shared__ T tile[32][33];
    const int x = blockIdx.x * 32 + threadIdx.x;
    const int y = blockIdx.y * 32 + threadIdx.y;
    for (int i = 0; i < 32; i += 8) {
        if (x < cols && y + i < rows) tile[threadIdx.y + i][threadIdx.x] = src[(y + i) * cols + x];
    }
    __syncthreads();
    const int out_x = blockIdx.y * 32 + threadIdx.x;
    const int out_y = blockIdx.x * 32 + threadIdx.y;
    for (int i = 0; i < 32; i += 8) {
        if (out_x < rows && out_y + i < cols)
            dst[(out_y + i) * rows + out_x] = tile[threadIdx.x][threadIdx.y + i];
    }
}
'''

_PREFIX_SUM_CUDA = r'''
__global__ void agent_lk_prefix_sum1d_kernel(const int32_t* src, int32_t* dst, int rows) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows) return;
    int32_t total = 0;
    for (int j = 0; j < i; ++j) total += src[j];
    dst[i] = total;
}
'''

_CAST_CUDA = r'''
__global__ void agent_lk_pad_cast2d_kernel(const float* src, __nv_bfloat16* dst,
                                            int rows, int cols, int padded_rows, int padded_cols) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(padded_rows) * padded_cols) return;
    const int row = int(index / padded_cols);
    const int col = int(index % padded_cols);
    const float value = row < rows && col < cols ? src[int64_t(row) * cols + col] : 0.0f;
    dst[index] = __float2bfloat16_rn(value);
}

__global__ void agent_lk_crop_cast2d_kernel(const __nv_bfloat16* src, float* dst,
                                             int rows, int cols, int padded_cols) {
    const int64_t index = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= int64_t(rows) * cols) return;
    const int row = int(index / cols);
    const int col = int(index % cols);
    dst[index] = __bfloat162float(src[int64_t(row) * padded_cols + col]);
}
'''


def _contract_may_alias_params(spec: dict) -> set:
    """Entry-kernel parameters the export contract itself can alias.

    Codegen marks entry tensor/pointer parameters __restrict__ (nvcc noalias),
    which is only justified when the wrapper hands every parameter its own
    device allocation. Two export constructs break that disjointness proof: an
    output with "alias_of" IS the named model input updated in place
    (`torch::Tensor out = x;` in the binding), and an "alias" transform is
    another name for its source tensor (`torch::Tensor xv = x;`). Names linked
    by either construct (transitively) form one aliasing group; every kernel
    argument in a non-singleton group must opt out of __restrict__ or the
    entry kernel would run under an unjustified noalias assumption (nvcc may
    keep a load cached across a store through the aliasing parameter).

    The opt-out is contract-driven, not frontend-driven: a draft that forgets
    to declare __LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__ still gets a legal ABI.
    Fresh outputs/workspaces (at::empty) and unrelated inputs stay __restrict__,
    so the load-batching optimization is preserved wherever disjointness holds.
    A frontend that hands two overlapping ranges of one allocation through
    distinct input names (offset views the export JSON cannot express) still
    opts out itself via __LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__.
    """
    parent = {}

    def find(name):
        parent.setdefault(name, name)
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    def union(a, b):
        parent[find(a)] = find(b)

    for item in spec["_outputs"]:
        if item.get("alias_of"):
            union(item["name"], item["alias_of"])
    for item in spec["transforms"]:
        if item["op"] == "alias":
            union(item["name"], item["source"])
    groups = {}
    for name in parent:
        groups.setdefault(find(name), set()).add(name)
    kernel_args = set(spec["_kernel_args"])
    return {name for group in groups.values() if len(group) > 1 for name in group if name in kernel_args}


def _render_cuda(spec: dict, kernel) -> str:
    # Resolve the frontend's declared tensor element dtypes once, from the
    # same py_func annotations codegen_cuda reads when emitting the entry
    # signature; the launcher's void* casts key on this map (falling back to
    # spec/storage dtypes for unresolved names) so cast and signature agree.
    spec["_declared_dtypes"] = _declared_tensor_dtypes(kernel)
    expected = spec["_kernel_args"]  # noqa: F841
    expected_params = _kernel_param_names(spec)
    actual = list(inspect.signature(kernel.py_func).parameters)
    if actual != expected_params:
        raise ValueError(f"kernel parameters {actual} do not match spec {expected_params}")
    # Descriptor transport ABI: a singleton entry binds to a by-value
    # grid_constant[TmaDescriptor] parameter; a per-batch bundle binds to a
    # device-array ptr[TmaDescriptor] parameter. A mismatch (e.g. a bundle
    # indexed dB[b] against a by-value map, or a grid_constant pointer) is a
    # hard launch-time type error, so reject it at lowering.
    descriptor_bundles = spec.get("_descriptor_bundles", {})
    annotations = kernel.py_func.__annotations__
    for desc_name in spec["_descriptor_names"]:
        kind = _descriptor_param_kind(annotations.get(desc_name))
        if desc_name in descriptor_bundles:
            if kind != "pointer":
                raise ValueError(f"TMA descriptor bundle {desc_name} requires a "
                                 "const[ptr[TmaDescriptor]] entry parameter (device map "
                                 "array indexed at runtime)")
        elif kind != "grid_constant":
            raise ValueError(f"TMA descriptor {desc_name} requires a "
                             "const[grid_constant[TmaDescriptor]] entry parameter")
    # A frontend-declared may-alias name that matches no kernel parameter is a
    # malformed alias declaration: it silently fails to protect the intended
    # parameter (which keeps __restrict__ under an ABI that can deliver a
    # shared allocation through it), so reject the typo instead of ignoring it.
    declared_alias = set(kernel.ctx.get("__LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__", ()))
    unknown_alias = declared_alias - set(expected_params)
    if unknown_alias:
        raise ValueError("__LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__ names unknown "
                         f"kernel parameters: {sorted(unknown_alias)}")
    may_alias = _contract_may_alias_params(spec)
    if may_alias:
        key = "__LITTLE_KERNEL_CUDA_MAY_ALIAS_PARAMS__"
        kernel.ctx[key] = tuple(sorted(set(kernel.ctx.get(key, ())) | may_alias))
    if spec["launch"]["_min_blocks"] is not None:
        # Generic occupancy hint: emit __launch_bounds__(num_threads, min_blocks)
        # on the entry kernel. An explicit spec field takes precedence over the
        # frontend's module-level min-blocks global.
        kernel.ctx[CUDA_LAUNCH_BOUNDS_MIN_BLOCKS_CTX_KEY] = spec["launch"]["_min_blocks"]
    # Constexpr constructor values are threaded through lower/compile so shape
    # expressions sized by a baked literal see it during the shared-memory
    # floor scan as well; an empty mapping leaves non-constexpr manifests on
    # the unchanged code path (no substitution), keeping output byte-identical.
    constexpr_args = spec["_constexpr_constructor"]
    # Derived constexpr selectors fan the export out into one baked kernel
    # per domain combination (bool selectors => 2^k variants). The shared
    # memory floor scan below must see every variant so the host-side floor
    # covers the largest specialization; a manifest without derived
    # selectors lowers exactly once, as before.
    derived_selectors = spec["_constexpr_derived"]
    if derived_selectors:
        variant_values = [
            dict(zip([item["name"]
                      for item in derived_selectors], bits))
            for bits in itertools.product((False, True), repeat=len(derived_selectors))
        ]
        lowered_variants = [
            kernel.lower(PASSES["cuda"], constexpr_args={**constexpr_args, **values}) for values in variant_values
        ]
    else:
        variant_values = [{}]
        lowered_variants = [kernel.lower(PASSES["cuda"], constexpr_args=constexpr_args)]
    allocations = []
    dyn_allocs = []
    for lowered in lowered_variants:
        for call in ast.walk(lowered):
            # Match both `alloc_dynamic_shared_memory(...)` (ast.Name) and
            # `ll.alloc_dynamic_shared_memory(...)` (ast.Attribute) call forms.
            if not (isinstance(call, ast.Call) and
                    ((isinstance(call.func, ast.Name) and call.func.id == "alloc_dynamic_shared_memory") or
                     (isinstance(call.func, ast.Attribute) and call.func.attr == "alloc_dynamic_shared_memory"))
                    and call.args and isinstance(call.args[0], ast.Constant) and type(call.args[0].value) is int):
                continue
            allocations.append(call.args[0].value)
            if len(call.args) >= 4 and isinstance(call.args[3], ast.Constant) and \
                    type(call.args[3].value) is str:
                try:
                    dyn_allocs.extend(json.loads(call.args[3].value))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"malformed runtime-sized shared allocation descriptor: {exc}") from None
    required = max(allocations, default=0)
    # Static dynamic-shared floor: any runtime shared_memory_bytes must
    # cover the kernel's compile-time allocations (checked host-side).
    spec["_shared_floor"] = ((required + 127) // 128) * 128
    # Runtime-sized (shape-expression) dynamic_shared allocations: host-
    # evaluated bytes per allocation plus its static max_bytes bound.
    spec["_shared_dyn"] = []
    for entry in dyn_allocs:
        if (not isinstance(entry, dict) or type(entry.get("elems_src")) is not str
                or type(entry.get("elem_size")) is not int or type(entry.get("max_bytes")) is not int
                or entry["elem_size"] <= 0 or entry["max_bytes"] <= 0):
            raise ValueError(f"malformed runtime-sized shared allocation entry: {entry!r}")
        bytes_src = f"({entry['elems_src']}) * {entry['elem_size']}"
        spec["_shared_dyn"].append({
            "max_bytes":
            entry["max_bytes"],
            "bytes_cpp":
            _shape_expr(bytes_src, spec["_ranks"], spec["_launch_scalars"], lists=spec["_host_lists"]),
        })
    if spec["_shared_dyn"] and spec["launch"]["_shared"] == "auto":
        raise ValueError("launch.shared_memory_bytes \"auto\" cannot size a runtime-sized "
                         "(shape-expression) dynamic_shared allocation; give the "
                         "reservation as a shape expression")
    if spec["launch"]["_shared"] == "auto":
        spec["launch"]["_shared"] = spec["_shared_floor"]
    # num_threads is the static ceiling computed at manifest validation
    # (exact product for static blocks; block_max_threads or the caps
    # product for dynamic ones).
    # Bridge the validated launch.min_blocks_per_sm field to the single ctx
    # key codegen_cuda consumes (MIN_BLOCKS_CTX_KEY, emitted as
    # __launch_bounds__(num_threads, min_blocks) only when the key is
    # present). When the field is absent (validated to None) no ctx key is
    # written, so output is byte-identical for manifests that never set it;
    # the legacy draft.py module global
    # __LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__ keeps working because it
    # lives in the same ctx (LLKernel copies py_func.__globals__). Both
    # request channels converge on one consumer.
    if spec["launch"].get("_min_blocks_per_sm") is not None:
        kernel.ctx["__LITTLE_KERNEL_CUDA_LAUNCH_BOUNDS_MIN_BLOCKS__"] = \
            spec["launch"]["_min_blocks_per_sm"]
    # no_alias (validated default False at spec load, data["_no_alias"]) is
    # forwarded to codegen_cuda's blanket noalias knob; when unset the
    # per-parameter alias contract governs and output is unchanged.
    original = spec["_kernel"]
    renamed = f"agent_{original}" if not original.startswith("agent_") else original

    def _compile_entry(values: dict) -> str:
        text = kernel.compile(PASSES["cuda"], codegen_cuda, num_threads=spec["launch"]["_threads"],
                              constexpr_args={**constexpr_args, **values},
                              disable_var_hoisting=spec["launch"].get("disable_var_hoisting",
                                                                      False), no_alias=spec.get("_no_alias", False))
        if not re.search(rf"\b__global__\b[^{{;]*\b{re.escape(original)}\s*\(", text):
            raise ValueError(f"generated CUDA does not define entry kernel {original}")
        return re.sub(rf"\b{re.escape(original)}\b", renamed, text)

    variant_sources = []
    # Qualified entry-kernel name used by the launcher for each fanout variant.
    variant_kernels = []
    variant_prelude = ""
    for index, values in enumerate(variant_values):
        text = _compile_entry(values)
        if derived_selectors:
            # Each specialization lives in its own namespace so device
            # helpers with identical names across variants never collide in
            # one TU; the host launcher reaches the kernel as lk_cx{i}::<entry>.
            # The codegen prelude (#include/comment/blank lines) must stay at
            # global scope, so pull it out of every variant and emit it once.
            prelude, body = _split_cuda_prelude(text)
            if index == 0:
                variant_prelude = prelude
            variant_kernels.append(f"lk_cx{index}::{renamed}")
            variant_sources.append(f"namespace lk_cx{index} {{\n{body}\n}} // namespace lk_cx{index}\n")
        else:
            variant_kernels.append(renamed)
            variant_sources.append(text)
    source = variant_sources[0]
    dynamic_block = spec["launch"].get("_block_dynamic", False)  # noqa: F841
    # Runtime ABI name list: constexpr constructor scalars are baked into the
    # kernel and leave the launcher/binding ABI (spec["_runtime_kernel_args"]
    # == kernel_args minus constexpr names; identical to `expected` when no
    # constructor entry declares constexpr, so non-constexpr manifests keep
    # the previous ABI byte-for-byte).
    args = [_launcher_arg_decl(spec, name) for name in spec["_runtime_kernel_args"]]
    # Uniform launcher ABI: grid/block dims and shared_bytes are always
    # function arguments; static manifests simply receive constants, so the
    # launcher signature is independent of which block dims are dynamic.
    args += ([f"unsigned grid_{axis}" for axis in "xyz"] + [f"unsigned block_{axis}" for axis in "xyz"] +
             ["unsigned shared_bytes", "cudaStream_t stream"])
    # Derived constexpr selectors ride the launcher ABI as trailing bool
    # flags (computed by the generated Model.__init__ from the constructor
    # values); the launcher switches between the baked variants on them.
    selector_flag_decls = [f"bool lk_cx_{item['name']}" for item in derived_selectors]
    args += selector_flag_decls
    kernel_args = [_cuda_kernel_arg(spec, name) for name in spec["_runtime_kernel_args"]]
    launcher = "agent_lk_launch_" + spec["_function"]
    helpers = ""
    if any(item["op"] == "transpose2d" for item in spec["transforms"]):
        helpers += _TRANSPOSE_CUDA
    if any(item["op"] == "prefix_sum1d" for item in spec["transforms"]):
        helpers += _PREFIX_SUM_CUDA
    if spec["pre_transforms"] or spec["post_transforms"]:
        helpers += _CAST_CUDA
    for item in spec["transforms"]:
        if item["op"] == "transpose2d":
            name, dtype = item["name"], spec["_tensor_types"][item["name"]]
            cpp_type = _DTYPES[dtype][0]
            helpers += (f'\nextern "C" cudaError_t agent_lk_transpose_{name}('
                        'const void* src, void* dst, int rows, int cols, cudaStream_t stream) {\n'
                        f'    agent_lk_transpose2d_kernel<{cpp_type}><<<'
                        'dim3((cols + 31) / 32, (rows + 31) / 32), dim3(32, 8), 0, stream>>>(\n'
                        f'        reinterpret_cast<const {cpp_type}*>(src), '
                        f'reinterpret_cast<{cpp_type}*>(dst), rows, cols);\n'
                        '    return cudaGetLastError();\n}\n')
        elif item["op"] == "prefix_sum1d":
            name = item["name"]
            helpers += (f'\nextern "C" cudaError_t agent_lk_prefix_sum_{name}('
                        'const void* src, void* dst, int rows, cudaStream_t stream) {\n'
                        '    agent_lk_prefix_sum1d_kernel<<<dim3((rows + 128) / 128), '
                        'dim3(128), 0, stream>>>(\n'
                        '        reinterpret_cast<const int32_t*>(src), '
                        'reinterpret_cast<int32_t*>(dst), rows);\n'
                        '    return cudaGetLastError();\n}\n')
    for item in spec["pre_transforms"]:
        target = item["target"]
        helpers += (f'\nextern "C" cudaError_t agent_lk_pad_cast_{target}('
                    'const void* src, void* dst, int rows, int cols, '
                    'int padded_rows, int padded_cols, cudaStream_t stream) {\n'
                    '    const int64_t count = int64_t(padded_rows) * padded_cols;\n'
                    '    agent_lk_pad_cast2d_kernel<<<dim3((count + 255) / 256), '
                    'dim3(256), 0, stream>>>(\n'
                    '        reinterpret_cast<const float*>(src), '
                    'reinterpret_cast<__nv_bfloat16*>(dst), '
                    'rows, cols, padded_rows, padded_cols);\n'
                    '    return cudaGetLastError();\n}\n')
    for item in spec["post_transforms"]:
        target = item["target"]
        helpers += (f'\nextern "C" cudaError_t agent_lk_crop_cast_{target}('
                    'const void* src, void* dst, int rows, int cols, '
                    'int padded_cols, cudaStream_t stream) {\n'
                    '    const int64_t count = int64_t(rows) * cols;\n'
                    '    agent_lk_crop_cast2d_kernel<<<dim3((count + 255) / 256), '
                    'dim3(256), 0, stream>>>(\n'
                    '        reinterpret_cast<const __nv_bfloat16*>(src), '
                    'reinterpret_cast<float*>(dst), rows, cols, padded_cols);\n'
                    '    return cudaGetLastError();\n}\n')
    launch_lines = [
        "    if (shared_bytes > 48 * 1024) {",
        f"        cudaError_t error = cudaFuncSetAttribute({renamed}, "
        "cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);",
        "        if (error != cudaSuccess) return error;",
        "    }",
    ]
    cluster = spec["launch"]["_cluster"]
    kernel_args_text = ",\n        ".join(kernel_args)
    if derived_selectors:
        # One fully-pruned kernel per selector combination; the flags (bound
        # at Model.__init__) pick the instantiation, so the untaken path of
        # every `if <selector>` is absent from the launched binary.
        index_terms = [
            f"((unsigned)(lk_cx_{item['name']} ? 1 : 0) << {pos})" for pos, item in enumerate(derived_selectors)
        ]
        launch_lines = [
            "    const unsigned lk_cx_index = " + " | ".join(index_terms) + ";", "    switch (lk_cx_index) {"
        ]
        for index, kernel_name in enumerate(variant_kernels):
            launch_lines += [f"    case {index}: {{"]
            launch_lines += [
                "        if (shared_bytes > 48 * 1024) {",
                f"            cudaError_t lk_cx_attr = cudaFuncSetAttribute({kernel_name}, "
                "cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_bytes);",
                "            if (lk_cx_attr != cudaSuccess) return lk_cx_attr;", "        }"
            ]
            if cluster:
                launch_lines += [
                    "        cudaLaunchAttribute attribute{};",
                    "        attribute.id = cudaLaunchAttributeClusterDimension;",
                    f"        attribute.val.clusterDim.x = {cluster[0]};",
                    f"        attribute.val.clusterDim.y = {cluster[1]};",
                    f"        attribute.val.clusterDim.z = {cluster[2]};",
                    "        cudaLaunchConfig_t config{};",
                    "        config.gridDim = dim3(grid_x, grid_y, grid_z);",
                    "        config.blockDim = dim3(block_x, block_y, block_z);",
                    "        config.dynamicSmemBytes = shared_bytes;",
                    "        config.stream = stream;",
                    "        config.attrs = &attribute;",
                    "        config.numAttrs = 1;",
                    f"        return cudaLaunchKernelEx(&config, {kernel_name}, "
                    f"{', '.join(kernel_args)});",
                ]
            else:
                launch_lines += [
                    f"        {kernel_name}<<<dim3(grid_x, grid_y, grid_z), "
                    "dim3(block_x, block_y, block_z), shared_bytes, stream>>>(",
                    "        " + kernel_args_text + ");",
                    "        return cudaGetLastError();",
                ]
            launch_lines += ["    }"]
        launch_lines += [
            "    default: return cudaErrorInvalidValue;",
            "    }",
        ]
    elif cluster:
        launch_lines += [
            "    cudaLaunchAttribute attribute{};",
            "    attribute.id = cudaLaunchAttributeClusterDimension;",
            f"    attribute.val.clusterDim.x = {cluster[0]};",
            f"    attribute.val.clusterDim.y = {cluster[1]};",
            f"    attribute.val.clusterDim.z = {cluster[2]};",
            "    cudaLaunchConfig_t config{};",
            "    config.gridDim = dim3(grid_x, grid_y, grid_z);",
            "    config.blockDim = dim3(block_x, block_y, block_z);",
            "    config.dynamicSmemBytes = shared_bytes;",
            "    config.stream = stream;",
            "    config.attrs = &attribute;",
            "    config.numAttrs = 1;",
            f"    return cudaLaunchKernelEx(&config, {renamed}, {', '.join(kernel_args)});",
        ]
    else:
        launch_lines += [
            f"    {renamed}<<<dim3(grid_x, grid_y, grid_z), "
            "dim3(block_x, block_y, block_z), shared_bytes, stream>>>(",
            "        " + ",\n        ".join(kernel_args) + ");",
            "    return cudaGetLastError();",
        ]
    packed_cuda = ""
    if spec["_foreach_dispatch"] == "single" and spec["_foreach_inputs"]:
        packed_cuda = _render_foreach_single_cuda(spec, source, renamed)
    frontend = inspect.getsourcefile(kernel.py_func)
    if frontend:
        module_doc = ast.get_docstring(ast.parse(Path(frontend).read_text()))
        if module_doc:
            comments = ["// LittleKernel program intent:"]
            for paragraph in module_doc.splitlines()[:80]:
                if not paragraph.strip():
                    comments.append("//")
                else:
                    comments.extend("// " + line
                                    for line in textwrap.wrap(paragraph.strip(), width=88, break_long_words=False))
            prefix = "\n".join(comments) + "\n\n"
            variant_sources = [prefix + text for text in variant_sources]
            source = variant_sources[0]
    declarations = ",\n".join("    " + arg for arg in args)
    fused = ""
    if spec.get("_foreach_single_launch"):
        fused = _render_foreach_fused_cuda(spec, source, renamed)
    return (variant_prelude + "".join(variant_sources) + helpers + fused + f'\nextern "C" cudaError_t {launcher}(\n' +
            declarations + "\n) {\n" + "\n".join(launch_lines) + "\n}\n" + packed_cuda)
