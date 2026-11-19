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
"""Host binding and model emission.

Renders the self-contained pybind11 C++ binding (input/rank/dtype checks,
optional-input channels, host-list staging transforms, TMA descriptor
encoding, shape/scalar/grid evaluation, foreach groups) and the drop-in
``model_new.py``, preserving every ABI spelling of the authored manifest.
"""

from __future__ import annotations

import json

from little_kernel.backends.nv.common.export._foreach import _FOREACH_FUSED_MAX, _FOREACH_SCALAR_CPP, _foreach_fused_plan, _render_foreach_single_binding
from little_kernel.backends.nv.common.export._frontend import _launcher_arg_decl, _scalar_splits
from little_kernel.backends.nv.common.export._schema import _CONSTRUCTOR_CPP, _DTYPES, _SCALAR_COMPONENT, _SCALAR_RUNTIME_CPP, _TMA_DTYPES


def _render_binding(spec: dict) -> str:
    inputs, outputs, scalars = spec["inputs"], spec["_outputs"], spec["scalars"]
    constexpr_values = spec["_constexpr_constructor"]
    derived_selectors = spec["_constexpr_derived"]
    constexpr_names = spec["_constexpr_names"]
    constructor = [
        item for item in spec["constructor"] if item["dtype"] != "ignored" and item["name"] not in constexpr_names
    ]
    function = "agent_" + spec["_function"]
    launcher = "agent_lk_launch_" + spec["_function"]
    dynamic_block = spec["launch"].get("_block_dynamic", False)  # noqa: F841
    args = [_launcher_arg_decl(spec, name) for name in spec["_runtime_kernel_args"]]
    # Uniform launcher ABI, matching _render_cuda's declaration.
    args += ([f"unsigned grid_{axis}" for axis in "xyz"] + [f"unsigned block_{axis}" for axis in "xyz"] +
             ["unsigned shared_bytes", "cudaStream_t stream"] +
             [f"bool lk_cx_{item['name']}" for item in derived_selectors])
    # Host-side alias for each constexpr constructor: the literal is baked into the
    # CUDA kernel, and the same literal backs shape/grid/equality expressions here.
    _CONSTEXPR_CPP = {"int32": "int64_t", "int64": "int64_t", "float32": "float", "bool": "bool"}
    constexpr_lines = [
        f"    const {_CONSTEXPR_CPP[item['dtype']]} {item['name']} = "
        f"{('true' if item['constexpr'] else 'false') if item['dtype'] == 'bool' else item['constexpr']};"
        for item in spec["constructor"]
        if item["name"] in constexpr_values
    ]
    optional_inputs = {item["name"] for item in inputs if item.get("optional", False)}
    present_inputs = set(spec.get("_present_inputs", ()))
    # An optional input lets the caller pass None for a feature-disabled
    # tensor. The presence channel is selected per input:
    # * present()-driven optional (a scalar declared present(name)): a
    #   std::optional value parameter `<name>_opt`; the tensor is
    #   materialized as zero-sized of the declared dtype/rank on another
    #   input's device when absent, so every check and data_ptr path stays
    #   valid; kernels observe absence only through the present() scalar and
    #   never dereference the substitute. present() renders
    #   `(<name>_opt.has_value() ? 1 : 0)` in _shape_expr.
    # * plain optional: a `const c10::optional<torch::Tensor>& lk_opt_<name>`
    #   parameter, rebound to a possibly-undefined local alias so downstream
    #   rendering is unchanged, checks run under `if (<name>.defined())`, and
    #   the launcher receives nullptr when absent.
    # Invariant: each optional input is observed through exactly one presence
    # channel (an explicit present() scalar, or nullptr), chosen at spec load
    # and used consistently by the signature, rebind, checks and launch call.
    items_by_name = {item["name"]: item for item in inputs}
    # TensorOptions come from a required tensor input only: a host int-list
    # (std::vector<int64_t> at the ABI) has no .options(). For manifests
    # whose first input is a required tensor this reduces to inputs[0]/the
    # historical spelling, keeping emitted bytes unchanged there.
    options_source = next(
        (x["name"] for x in inputs if x.get("_kind", "tensor") == "tensor" and x["name"] not in optional_inputs), None)
    options_spelling = (f"{options_source}.options()"
                        if options_source is not None else "at::TensorOptions().device(at::kCUDA)")
    # Trailing-ABI manifests bind plain optionals as trailing
    # `const c10::optional<torch::Tensor>& <name>` parameters used directly;
    # positional-ABI manifests rebind the `lk_opt_<name>` parameter to a
    # possibly-undefined local alias. Invariant: checks and launch arguments
    # observe one spelling per input, matching the emitted parameter.
    trailing = bool(spec.get("_optional_trailing"))
    binding_decls = []
    for name in sorted(optional_inputs):
        item = items_by_name[name]
        if name in present_inputs:
            dtype = _DTYPES[item["dtype"]][2]
            dims = ["0"] + ["1"] * (item["rank"] - 1)
            base = (f"{options_source}.options()"
                    if options_source is not None else "at::TensorOptions().device(at::kCUDA)")
            binding_decls.append(f"    const torch::Tensor {name} = {name}_opt.value_or("
                                 f"at::empty({{{', '.join(dims)}}}, {base}.dtype({dtype})));")
        elif not trailing:
            binding_decls.append(f"    torch::Tensor {name} = lk_opt_{name}.value_or(torch::Tensor());")
    checks = []
    for item in inputs:
        if item.get("_kind") == "int_list":
            # pybind11 already validates the host list transport
            # (std::vector<int64_t>, or std::optional when declared optional).
            continue
        name = item["name"]
        dtype = _DTYPES[item["dtype"]][1]
        optional = name in optional_inputs
        # Trailing-ABI plain optionals are c10::optional parameters: checks
        # run under .has_value() with arrow access; positional/present()
        # spellings resolve to a local torch::Tensor holding dot access.
        arrow = optional and trailing and name not in present_inputs
        member = "->" if arrow else "."
        prefix = "        " if optional else "    "
        if optional:
            checks.append(f"    if ({name}.{'has_value()' if arrow else 'defined()'}) {{")
        checks.append(f'{prefix}TORCH_CHECK({name}{member}is_cuda(), "{name} must be on CUDA");')
        if item.get("contiguous", True):
            checks.append(f'{prefix}TORCH_CHECK({name}{member}is_contiguous(), "{name} must be contiguous");')
        checks += [
            f'{prefix}TORCH_CHECK({name}{member}dim() == {item["rank"]}, "{name} rank mismatch");',
            f'{prefix}TORCH_CHECK({name}{member}scalar_type() == {dtype}, "{name} dtype mismatch");',
        ]
        if name != inputs[0]["name"]:
            checks.append(
                f'{prefix}TORCH_CHECK({name}{member}device() == {inputs[0]["name"]}.device(), "inputs must share a device");'
            )
        if optional:
            checks.append("    }")
    for item in spec["tensor_lists"]:
        name = item["name"]
        dtype = _DTYPES[item["dtype"]][1]
        checks.append(f'    TORCH_CHECK({name}.size() >= 1 && {name}.size() <= {item["max_len"]}, '
                      f'"{name} list length out of range");')
        checks.append(f'    for (int64_t lk_i_{name} = 0; lk_i_{name} < static_cast<int64_t>({name}.size()); '
                      f'++lk_i_{name}) {{\n'
                      f'        const torch::Tensor& lk_el_{name} = {name}[lk_i_{name}];\n'
                      f'        TORCH_CHECK(lk_el_{name}.is_cuda(), "{name} elements must be on CUDA");\n'
                      f'        TORCH_CHECK(lk_el_{name}.is_contiguous(), "{name} elements must be contiguous");\n'
                      f'        TORCH_CHECK(lk_el_{name}.dim() == {item["rank"]}, "{name} element rank mismatch");\n'
                      f'        TORCH_CHECK(lk_el_{name}.scalar_type() == {dtype}, "{name} element dtype mismatch");\n'
                      f'        TORCH_CHECK(lk_el_{name}.device() == {inputs[0]["name"]}.device(), '
                      f'"{name} elements must share the input device");\n'
                      '    }')
    equality_lines = [
        f'    TORCH_CHECK({left} == {right}, "shape equality failed: {left} == {right}");'
        for left, right in spec["_equalities_cpp"]
    ]
    scalar_lines = []
    for item in scalars:
        if item["_runtime"]:
            continue
        name = item["name"]
        if item["_dtype"] == "float32":
            scalar_lines += [
                f"    const float {name} = static_cast<float>({item['_expr_cpp']});",
                f'    TORCH_CHECK(std::isfinite({name}), "{name} must be finite");'
            ]
        elif item["_dtype"] == "float64":
            scalar_lines += [
                f"    const double {name} = static_cast<double>({item['_expr_cpp']});",
                f'    TORCH_CHECK(std::isfinite({name}), "{name} must be finite");'
            ]
        else:
            bound = "INT64_MAX" if item["_dtype"] == "int64" else "INT32_MAX"
            scalar_lines += [
                f"    const int64_t {name} = {item['_expr_cpp']};",
                f'    TORCH_CHECK({name} >= 0 && {name} <= {bound}, "{name} exceeds range");'
            ]
    grid_lines = []
    for axis, value in zip("xyz", spec["launch"]["_grid_cpp"]):
        grid_lines += [
            f"    const int64_t lk_grid_{axis} = {value};",
            f'    TORCH_CHECK(lk_grid_{axis} > 0 && lk_grid_{axis} <= UINT32_MAX, "invalid grid {axis}");'
        ]
    # Declare each block axis exactly once: expression dims are checked
    # against their declared cap (`_block[axis]`, already validated against
    # CUDA per-axis limits at spec load, so a checked value is always
    # launchable); static dims against the CUDA per-axis hardware limit. The
    # product bound `_threads` check below then covers the 1024-thread ceiling
    # for both static and dynamic manifests.
    for axis, value, is_expr, cap in zip("xyz", spec["launch"]["_block_cpp"], spec["launch"]["_block_expr"],
                                         spec["launch"]["_block"]):
        if is_expr:
            grid_lines += [
                f"    const int64_t lk_block_{axis} = {value};",
                f'    TORCH_CHECK(lk_block_{axis} >= 1 && lk_block_{axis} <= {cap}, '
                f'"invalid block {axis}");'
            ]
        else:
            grid_lines += [
                f"    const int64_t lk_block_{axis} = {value};",
                f'    TORCH_CHECK(lk_block_{axis} > 0 && lk_block_{axis} <= '
                f'{1024 if axis in "xy" else 64}, "invalid block dimension {axis}");'
            ]
    cluster = spec["launch"]["_cluster"]
    if cluster:
        for axis, count in zip("xyz", cluster):
            grid_lines.append(
                f'    TORCH_CHECK(lk_grid_{axis} % {count} == 0, "grid {axis} must be divisible by cluster dimension");'
            )
    grid_lines.append(f'    TORCH_CHECK(lk_block_x * lk_block_y * lk_block_z <= {spec["launch"]["_threads"]}, '
                      '"block exceeds the compiled block_max_threads launch bound");')
    dyn_lines = []
    dyn_terms = []
    for i, entry in enumerate(spec.get("_shared_dyn", [])):
        dyn_lines.append(f"    const int64_t lk_shdyn_{i} = {entry['bytes_cpp']};")
        dyn_lines.append(f'    TORCH_CHECK(lk_shdyn_{i} >= 0 && lk_shdyn_{i} <= {entry["max_bytes"]}, '
                         '"runtime-sized dynamic_shared allocation exceeds its max_elems legality bound");')
        dyn_terms.append(f"lk_shdyn_{i}")
    floor = spec.get("_shared_floor", 0)
    required_cpp = " + ".join(([str(floor)] if floor else []) + dyn_terms)
    if spec["launch"]["_shared_cpp"] is not None:
        grid_lines += [
            f"    const int64_t lk_shared = {spec['launch']['_shared_cpp']};",
            '    TORCH_CHECK(lk_shared >= 0, "shared_memory_bytes must be nonnegative");'
        ]
        grid_lines += dyn_lines
        if required_cpp:
            grid_lines.append(f'    TORCH_CHECK(lk_shared >= {required_cpp}, '
                              '"shared_memory_bytes below the kernel dynamic shared floor");')
        shared_arg = "static_cast<unsigned>(lk_shared)"
    else:
        grid_lines += dyn_lines
        if dyn_terms:
            grid_lines.append(f'    TORCH_CHECK({required_cpp} <= {spec["launch"]["_shared"]}, '
                              '"shared_memory_bytes below the kernel dynamic shared floor");')
        shared_arg = str(spec["launch"]["_shared"])
    input_names = [x["name"] for x in inputs]
    input_decls = []
    for x in inputs:
        if x.get("_kind") == "int_list":
            if spec["_host_lists"][x["name"]]:
                input_decls.append(f"std::optional<std::vector<int64_t>> {x['name']}")
            else:
                input_decls.append(f"std::vector<int64_t> {x['name']}")
        else:
            input_decls.append("torch::Tensor " + x["name"])
    constructor_names = [item["name"] for item in constructor]
    # Derived-selector flags bound by the generated Model.__init__; they sit
    # after the constructor parameters in the wrapper/launcher ABI and never
    # reach the (already specialized) device kernels.
    selector_names = [f"lk_cx_{item['name']}" for item in derived_selectors]
    selector_decls = [f"bool {name}" for name in selector_names]
    output_names = [x["name"] for x in outputs]
    return_type = ("torch::Tensor" if len(outputs) == 1 else "std::tuple<" +
                   ", ".join("torch::Tensor" for _ in outputs) + ">")
    return_value = (output_names[0] if len(outputs) == 1 else "std::make_tuple(" + ", ".join(output_names) + ")")
    # A self-aliased output (alias_of == name) IS the input parameter itself:
    # the kernel writes through the caller's buffer, and the binding returns
    # that same object. Emitting `torch::Tensor name = name;` would shadow the
    # function parameter (a compile error) and serves no purpose, so self-
    # aliases get NO declaration line at all; cross-input aliases still
    # rebind (`torch::Tensor out = x;`) to spell the renamed view.
    output_lines = [(f"    torch::Tensor {item['name']} = {item['alias_of']};" if item.get("alias_of")
                     and item["alias_of"] != item["name"] else f"    torch::Tensor {item['name']} = at::empty("
                     f"{{{', '.join(item['_shape_cpp'])}}}, "
                     f"{options_spelling}.dtype({_DTYPES[item['dtype']][2]}));")
                    for item in outputs
                    if item.get("alias_of") != item["name"]]
    empty_check = " && ".join(f"{name}.numel() == 0" for name in output_names)
    list_ptr_lines = []
    for item in spec["tensor_lists"]:
        name = item["name"]
        for index, param in enumerate(item["kernel_params"]):
            list_ptr_lines.append(f'    void* {param} = ({name}.size() > static_cast<size_t>({index}))'
                                  f' ? {name}[{index}].data_ptr() : nullptr;')
    # Outputs may request zeroing ("zero_init": true): the fresh at::empty
    # allocation is memset on the launch stream before the kernel runs, so
    # split-reduction / last-block-closes schedules can accumulate with
    # atomicAdd into a stage output without a separate memset launch. Emitted
    # next to workspace zeroing, i.e. after lk_stream is declared and before
    # the launcher call; the two sets share one ordered emission block.
    output_zero_lines = []
    for item in outputs:
        if not item.get("zero_init"):
            continue
        name = item["name"]
        output_zero_lines.append(f"    cudaError_t lk_zero_error_{name} = cudaMemsetAsync({name}.data_ptr(), 0, "
                                 f"{name}.numel() * {name}.element_size(), lk_stream);\n"
                                 f'    TORCH_CHECK(lk_zero_error_{name} == cudaSuccess, '
                                 f'cudaGetErrorString(lk_zero_error_{name}));')
    launch_args = []
    # Iterate the runtime launcher ABI (constexpr-folded args excluded),
    # including generic list-param and optional-input pass-through cases. An
    # injected presence scalar resolves through the spelling of its optional
    # source: arrow + .has_value() for trailing-ABI c10::optional
    # parameters, dot + .defined() for the positional-ABI rebound alias.
    presence_args = spec.get("_presence_args", {})
    for name in spec["_runtime_kernel_args"]:
        if name in spec["_list_param_types"]:
            launch_args.append(name)
        elif name in spec["_tensor_types"]:
            if name in optional_inputs:
                if trailing and name not in present_inputs:
                    launch_args.append(f"({name}.has_value() ? {name}->data_ptr() : nullptr)")
                else:
                    launch_args.append(f"{name}.defined() ? {name}.data_ptr() : nullptr")
            else:
                launch_args.append(f"{name}.data_ptr()")
        elif name in presence_args:
            source = presence_args[name]
            if trailing:
                launch_args.append(f"({source}.has_value() ? 1 : 0)")
            else:
                launch_args.append(f"{source}.defined() ? 1 : 0")
        elif _scalar_splits(spec, name):
            launch_args += [f"{name}.real()", f"{name}.imag()"]
        else:
            launch_args.append(name)
    launch_args += [f"static_cast<unsigned>(lk_grid_{x})" for x in "xyz"]
    # Uniform launcher ABI: every lk_block_{axis} binding is emitted exactly
    # once above (static dims as constants, expression dims cap-checked), and
    # shared_arg carries the static bytes or the checked lk_shared cast.
    launch_args += [f"static_cast<unsigned>(lk_block_{x})" for x in "xyz"]
    launch_args += [shared_arg, "lk_stream"]
    launch_args += selector_names
    temp_lines = []
    workspace_zero_lines = []
    for item in spec["workspaces"]:
        name = item["name"]
        shape = ", ".join(item["_shape_cpp"])
        temp_lines.append(f"    torch::Tensor {name} = at::empty({{{shape}}}, "
                          f"{options_spelling}.dtype({_DTYPES[item['dtype']][2]}));")
        if item.get("zero", False):
            workspace_zero_lines.append(f"    cudaError_t lk_zero_error_{name} = cudaMemsetAsync({name}.data_ptr(), 0, "
                                        f"{name}.numel() * {name}.element_size(), lk_stream);\n"
                                        f'    TORCH_CHECK(lk_zero_error_{name} == cudaSuccess, '
                                        f'cudaGetErrorString(lk_zero_error_{name}));')
    transform_lines = []
    transform_decls = []
    for item in spec["transforms"]:
        name, source = item["name"], item["source"]
        if item["op"] in ("host_cumsum1d", "host_upload1d"):
            # Stage a host int list into a fresh int32 device tensor with one
            # async H2D copy. host_cumsum1d folds the (nonnegative) values
            # into an exclusive prefix sum with a leading zero on the host —
            # the ragged-metadata resolution primitive. An absent optional
            # source produces a 0-element tensor and skips the copy.
            optional = spec["_host_lists"][source]
            cumsum = item["op"] == "host_cumsum1d"
            stage = f"lk_stage_{name}"
            if optional:
                size = f"static_cast<int64_t>({source}.has_value() ? {source}->size() + {1 if cumsum else 0} : 0)"
                opened, iterated, close = (f"if ({source}.has_value()) {{", f"*{source}", "}")
            else:
                size = f"static_cast<int64_t>({source}.size() + {1 if cumsum else 0})"
                opened, iterated, close = ("", source, "")
            temp_lines.append(f"    torch::Tensor {name} = at::empty({{{size}}}, "
                              f"{options_spelling}.dtype(at::kInt));")
            body = []
            if cumsum:
                body.append(f"    {stage}.push_back(0);")
                body.append(f"    int64_t lk_running_{name} = 0;")
            body += [
                f"    for (int64_t lk_value : {iterated}) {{",
                f'        TORCH_CHECK(lk_value >= INT32_MIN && lk_value <= INT32_MAX, '
                f'"{source} value exceeds int32 range");',
            ]
            if cumsum:
                body += [
                    f'        TORCH_CHECK(lk_value >= 0, "{source} values must be '
                    f'nonnegative for a prefix sum");',
                    f"        lk_running_{name} += lk_value;",
                    f'        TORCH_CHECK(lk_running_{name} <= INT32_MAX, '
                    f'"{source} prefix sum exceeds int32 range");',
                    f"        {stage}.push_back(static_cast<int32_t>(lk_running_{name}));",
                ]
            else:
                body.append(f"        {stage}.push_back(static_cast<int32_t>(lk_value));")
            body += [
                "    }",
                f"    cudaError_t lk_h2d_error_{name} = cudaMemcpyAsync({name}.data_ptr(), "
                f"{stage}.data(), {stage}.size() * sizeof(int32_t), "
                "cudaMemcpyHostToDevice, lk_stream);",
                f'    TORCH_CHECK(lk_h2d_error_{name} == cudaSuccess, '
                f'cudaGetErrorString(lk_h2d_error_{name}));',
            ]
            transform_lines.append(f"    std::vector<int32_t> {stage};")
            transform_lines.append(f"    {stage}.reserve({name}.numel());")
            if optional:
                transform_lines.append(f"    {opened}")
                transform_lines.extend("    " + line for line in body)
                transform_lines.append(f"    {close}")
            else:
                transform_lines.extend(body)
            continue
        if item["op"] == "transpose2d":
            temp_lines.append(f"    torch::Tensor {name} = at::empty("
                              f"{{{source}.size(1), {source}.size(0)}}, {source}.options());")
            transform_decls.append(f'extern "C" cudaError_t agent_lk_transpose_{name}('
                                   'const void*, void*, int, int, cudaStream_t);')
            transform_lines += [
                f'    TORCH_CHECK({source}.size(0) <= INT32_MAX && {source}.size(1) <= INT32_MAX, '
                f'"{source} exceeds transpose int32 range");',
                f"    cudaError_t lk_transpose_error_{name} = agent_lk_transpose_{name}("
                f"{source}.data_ptr(), {name}.data_ptr(), "
                f"static_cast<int>({source}.size(0)), static_cast<int>({source}.size(1)), lk_stream);",
                f'    TORCH_CHECK(lk_transpose_error_{name} == cudaSuccess, '
                f'cudaGetErrorString(lk_transpose_error_{name}));',
            ]
        elif item["op"] == "prefix_sum1d":
            temp_lines.append(f"    torch::Tensor {name} = at::empty("
                              f"{{{source}.size(0) + 1}}, {source}.options());")
            transform_decls.append(f'extern "C" cudaError_t agent_lk_prefix_sum_{name}('
                                   'const void*, void*, int, cudaStream_t);')
            transform_lines += [
                f'    TORCH_CHECK({source}.size(0) <= INT32_MAX - 1, '
                f'"{source} exceeds prefix sum int32 range");',
                f"    cudaError_t lk_prefix_error_{name} = agent_lk_prefix_sum_{name}("
                f"{source}.data_ptr(), {name}.data_ptr(), "
                f"static_cast<int>({source}.size(0)), lk_stream);",
                f'    TORCH_CHECK(lk_prefix_error_{name} == cudaSuccess, '
                f'cudaGetErrorString(lk_prefix_error_{name}));',
            ]
        else:
            temp_lines.append(f"    torch::Tensor {name} = {source};")
    pre_lines = []
    post_lines = []
    for item in spec["pre_transforms"]:
        source, target = item["source"], item["target"]
        transform_decls.append(f'extern "C" cudaError_t agent_lk_pad_cast_{target}('
                               'const void*, void*, int, int, int, int, cudaStream_t);')
        pre_lines += [
            f'    TORCH_CHECK({source}.size(0) <= INT32_MAX && {source}.size(1) <= INT32_MAX '
            f'&& {target}.size(0) <= INT32_MAX && {target}.size(1) <= INT32_MAX, '
            f'"{target} exceeds int32 range");',
            f'    TORCH_CHECK({target}.size(0) >= {source}.size(0) && '
            f'{target}.size(1) >= {source}.size(1), "{target} must cover {source}");',
            f"    cudaError_t lk_pad_error_{target} = agent_lk_pad_cast_{target}("
            f"{source}.data_ptr(), {target}.data_ptr(), "
            f"static_cast<int>({source}.size(0)), static_cast<int>({source}.size(1)), "
            f"static_cast<int>({target}.size(0)), static_cast<int>({target}.size(1)), lk_stream);",
            f'    TORCH_CHECK(lk_pad_error_{target} == cudaSuccess, cudaGetErrorString(lk_pad_error_{target}));',
        ]
    for item in spec["post_transforms"]:
        source, target = item["source"], item["target"]
        transform_decls.append(f'extern "C" cudaError_t agent_lk_crop_cast_{target}('
                               'const void*, void*, int, int, int, cudaStream_t);')
        post_lines += [
            f'    TORCH_CHECK({source}.size(0) >= {target}.size(0) && '
            f'{source}.size(1) >= {target}.size(1), "{source} must cover {target}");',
            f"    cudaError_t lk_crop_error_{target} = agent_lk_crop_cast_{target}("
            f"{source}.data_ptr(), {target}.data_ptr(), "
            f"static_cast<int>({target}.size(0)), static_cast<int>({target}.size(1)), "
            f"static_cast<int>({source}.size(1)), lk_stream);",
            f'    TORCH_CHECK(lk_crop_error_{target} == cudaSuccess, cudaGetErrorString(lk_crop_error_{target}));',
        ]
    descriptor_lines = []
    for item in spec["tma_descriptors"]:
        name, tensor = item["name"], item["tensor"]
        dtype, elem_size = _TMA_DTYPES[item["_dtype"]]
        fill = ("CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA"
                if item.get("oob_fill", False) else "CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE")

        def _make_call(map_arg: str, ptr_arg: str) -> str:
            if item["_rank"] == 2:
                return (f"agent_lk_make_tma_2d({map_arg}, {ptr_arg}, "
                        f"{item['_inner_cpp']}, {item['_outer_cpp']}, {item['_outer_stride_cpp']}, "
                        f"{item['_box'][0]}, {item['_box'][1]}, {dtype}, {elem_size}, "
                        f"{item['_swizzle']}, {fill})")
            if item["_rank"] == 3:
                return (f"agent_lk_make_tma_3d({map_arg}, {ptr_arg}, "
                        f"{item['_dim0_cpp']}, {item['_dim1_cpp']}, {item['_dim2_cpp']}, "
                        f"{item['_stride1_cpp']}, {item['_stride2_cpp']}, "
                        f"{item['_box'][0]}, {item['_box'][1]}, {item['_box'][2]}, "
                        f"{dtype}, {elem_size}, {item['_swizzle']}, {fill})")
            return (f"agent_lk_make_tma_4d({map_arg}, {ptr_arg}, "
                    f"{item['_dim0_cpp']}, {item['_dim1_cpp']}, {item['_dim2_cpp']}, "
                    f"{item['_dim3_cpp']}, "
                    f"{item['_stride1_cpp']}, {item['_stride2_cpp']}, {item['_stride3_cpp']}, "
                    f"{item['_box'][0]}, {item['_box'][1]}, {item['_box'][2]}, "
                    f"{item['_box'][3]}, "
                    f"{dtype}, {elem_size}, {item['_swizzle']}, {fill})")

        if not item.get("_bundle"):
            create = _make_call(f"&{name}", f"{tensor}.data_ptr()")
            descriptor_lines += [
                f"    CUtensorMap {name};", f"    CUresult lk_tma_error_{name} = {create};",
                f'    TORCH_CHECK(lk_tma_error_{name} == CUDA_SUCCESS, '
                f'"TMA descriptor {name} failed: ", static_cast<int>(lk_tma_error_{name}));'
            ]
            continue
        # Per-batch bundle: encode one identical map per leading batch slice
        # into a host vector, stage it to a device-resident CUtensorMap array
        # (the entry indexes it as dB[b]), and release it after launch.
        count = f"lk_desc_count_{name}"
        per_slice_ptr = (f"static_cast<char*>(({tensor}).data_ptr()) + "
                         f"lk_b * ({item['_batch_stride_cpp']}) * {elem_size}")
        create = _make_call(f"&{name}_host[lk_b]", per_slice_ptr)
        descriptor_lines += [
            f"    int64_t {count} = {item['_batch_count_cpp']};",
            f'    TORCH_CHECK({count} > 0, "TMA descriptor bundle {name} requires a positive batch count");',
            f"    std::vector<CUtensorMap> {name}_host(static_cast<size_t>({count}));",
            f"    for (int64_t lk_b = 0; lk_b < {count}; ++lk_b) {{",
            f"        CUresult lk_tma_error_{name} = {create};",
            f'        TORCH_CHECK(lk_tma_error_{name} == CUDA_SUCCESS, '
            f'"TMA descriptor {name} batch ", lk_b, " failed: ", '
            f'static_cast<int>(lk_tma_error_{name}));',
            "    }",
            f"    CUtensorMap* {name} = nullptr;",
            f'    TORCH_CHECK(cudaMalloc(reinterpret_cast<void**>(&{name}), '
            f'sizeof(CUtensorMap) * static_cast<size_t>({count})) == cudaSuccess, '
            f'"TMA descriptor bundle {name} device alloc failed");',
            f'    TORCH_CHECK(cudaMemcpy({name}, {name}_host.data(), '
            f'sizeof(CUtensorMap) * static_cast<size_t>({count}), cudaMemcpyHostToDevice) '
            f'== cudaSuccess, "TMA descriptor bundle {name} upload failed");',
        ]
        post_lines += [
            f'    TORCH_CHECK(cudaFree({name}) == cudaSuccess, '
            f'"TMA descriptor bundle {name} free failed");'
        ]
    tma_helper = ""
    if spec["tma_descriptors"]:
        # Resolve the driver entry point through the CUDA runtime API (already
        # linked) so extensions load without -lcuda on the plain build path.
        tma_helper = '''typedef CUresult (*agent_lk_cuTensorMapEncodeTiled_t)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*,
    const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);
static agent_lk_cuTensorMapEncodeTiled_t agent_lk_cuTensorMapEncodeTiled() {
    static agent_lk_cuTensorMapEncodeTiled_t fn = []() {
        void* sym = nullptr;
        cudaDriverEntryPointQueryResult query = cudaDriverEntryPointSymbolNotFound;
        if (cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &sym,
                cudaEnableDefault, &query) != cudaSuccess) sym = nullptr;
        return reinterpret_cast<agent_lk_cuTensorMapEncodeTiled_t>(sym);
    }();
    return fn;
}
static CUresult agent_lk_make_tma_2d(
    CUtensorMap* map, void* ptr, uint64_t inner, uint64_t outer, uint64_t stride,
    uint32_t box_inner, uint32_t box_outer, CUtensorMapDataType dtype,
    uint32_t element_size, CUtensorMapSwizzle swizzle, CUtensorMapFloatOOBfill fill) {
    agent_lk_cuTensorMapEncodeTiled_t encode = agent_lk_cuTensorMapEncodeTiled();
    if (encode == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    const uint64_t dims[2] = {inner, outer};
    const uint64_t strides[1] = {stride * element_size};
    const uint32_t box[2] = {box_inner, box_outer};
    const uint32_t element_strides[2] = {1, 1};
    return encode(map, dtype, 2, ptr, dims, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_256B, fill);
}
'''
        if any(item["_rank"] == 3 for item in spec["tma_descriptors"]):
            tma_helper += '''static CUresult agent_lk_make_tma_3d(
    CUtensorMap* map, void* ptr, uint64_t dim0, uint64_t dim1, uint64_t dim2,
    uint64_t stride1, uint64_t stride2, uint32_t box0, uint32_t box1, uint32_t box2,
    CUtensorMapDataType dtype, uint32_t element_size, CUtensorMapSwizzle swizzle,
    CUtensorMapFloatOOBfill fill) {
    const uint64_t dims[3] = {dim0, dim1, dim2};
    const uint64_t strides[2] = {stride1 * element_size, stride2 * element_size};
    const uint32_t box[3] = {box0, box1, box2};
    const uint32_t element_strides[3] = {1, 1, 1};
    agent_lk_cuTensorMapEncodeTiled_t encode = agent_lk_cuTensorMapEncodeTiled();
    if (encode == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    return encode(map, dtype, 3, ptr, dims, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);
}
'''
        if any(item["_rank"] == 4 for item in spec["tma_descriptors"]):
            tma_helper += '''static CUresult agent_lk_make_tma_4d(
    CUtensorMap* map, void* ptr, uint64_t dim0, uint64_t dim1, uint64_t dim2,
    uint64_t dim3, uint64_t stride1, uint64_t stride2, uint64_t stride3,
    uint32_t box0, uint32_t box1, uint32_t box2, uint32_t box3,
    CUtensorMapDataType dtype, uint32_t element_size, CUtensorMapSwizzle swizzle,
    CUtensorMapFloatOOBfill fill) {
    const uint64_t dims[4] = {dim0, dim1, dim2, dim3};
    const uint64_t strides[3] = {stride1 * element_size, stride2 * element_size,
        stride3 * element_size};
    const uint32_t box[4] = {box0, box1, box2, box3};
    const uint32_t element_strides[4] = {1, 1, 1, 1};
    agent_lk_cuTensorMapEncodeTiled_t encode = agent_lk_cuTensorMapEncodeTiled();
    if (encode == nullptr) return CUDA_ERROR_NOT_SUPPORTED;
    return encode(map, dtype, 4, ptr, dims, strides, box,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, fill);
}
'''
    runtime_scalars = [item["name"] for item in spec["scalars"] if item["_runtime"]]
    runtime_decls = [
        _SCALAR_RUNTIME_CPP[item["_dtype"]] + " " + item["name"] for item in spec["scalars"] if item["_runtime"]
    ]
    host_list_helpers = ""
    if spec["_host_lists"]:
        host_list_helpers = ("static int64_t lk_host_sum(const std::vector<int64_t>& values) {\n"
                             "    int64_t total = 0;\n"
                             "    for (int64_t value : values) { total += value; }\n"
                             "    return total;\n}\n"
                             "static int64_t lk_host_len(const std::vector<int64_t>& values) {\n"
                             "    return static_cast<int64_t>(values.size());\n}\n")
    empty_return = f"    if ({empty_check}) return {return_value};\n" if empty_check else ""
    complex_include = ("#include <pybind11/complex.h>\n#include <complex>\n" if any(
        item["_dtype"] in _SCALAR_COMPONENT for item in spec["scalars"]) else "")
    list_names = spec["_list_names"]
    # pybind11 named arguments must mirror the wrapper signature below: under
    # the trailing optional ABI the plain optionals are bound last, with
    # pybind11::none() defaults (defaults must be trailing in pybind11).
    if trailing:
        plain_optional = [x for x in input_names if x in optional_inputs and x not in present_inputs]
        pybind_names = [x for x in input_names if x not in plain_optional] \
            + list_names + runtime_scalars + constructor_names + selector_names
        pybind_args = ", ".join([f'pybind11::arg("{x}")' for x in pybind_names] +
                                [f'pybind11::arg("{x}") = pybind11::none()' for x in plain_optional])
    else:
        pybind_args = ", ".join(f'pybind11::arg("{x}")' for x in input_names + list_names + runtime_scalars +
                                constructor_names + selector_names)
    foreach_source = ""
    foreach_register = ""
    if spec["_foreach_inputs"]:
        selected = set(spec["_foreach_inputs"])
        names = [f"{name}_list" if name in selected else name for name in input_names]
        names += runtime_scalars + constructor_names
        declaration = [
            f"std::vector<torch::Tensor> {name}_list" if name in selected else f"torch::Tensor {name}"
            for name in input_names
        ]
        declaration += runtime_decls
        declaration += [_CONSTRUCTOR_CPP[item["dtype"]] + " " + item["name"] for item in constructor]
        first = spec["_foreach_inputs"][0]
        calls = [f"{name}_list[lk_group]" if name in selected else name for name in input_names]
        calls += runtime_scalars + constructor_names
        list_checks = [
            f'    TORCH_CHECK({name}_list.size() == {first}_list.size(), '
            f'"group count mismatch: {name}");' for name in spec["_foreach_inputs"][1:]
        ]
        grouped = function + "_grouped"
        n_outputs = len(spec["_outputs"])
        if spec.get("_foreach_single_launch"):
            # Single launch for the whole group: per-element values are
            # collected on the host into plain arrays, then handed to the
            # fused launcher, which packs the by-value table. Per-element
            # checks/output allocation/scalar and grid evaluation keep the
            # same text (and order) as the per-element path, so the element
            # tensors simply shadow the list names inside the loop.
            plan = _foreach_fused_plan(spec)
            max_group = spec.get("_foreach_max_group", _FOREACH_FUSED_MAX)
            indent = lambda lines: [("    " + line) if line.strip() else line for line in lines]
            collectors = []
            for name in plan["elem_tensors"]:
                collectors.append(f"    std::vector<const void*> lk_tbl_{name};")
            for name in plan["elem_scalars"]:
                ctype = _FOREACH_SCALAR_CPP[spec["_scalar_types"][name]]
                collectors.append(f"    std::vector<{ctype}> lk_tbl_{name};")
            collectors += ["    std::vector<unsigned> lk_tbl_gx;", "    std::vector<unsigned> lk_tbl_gy;"]
            results = "\n".join(f"    std::vector<torch::Tensor> lk_results_{j};\n"
                                f"    lk_results_{j}.reserve({first}_list.size());" for j in range(n_outputs))
            aliases = [f"        torch::Tensor {name} = {name}_list[lk_group];" for name in spec["_foreach_inputs"]]
            body = aliases
            body += indent(checks)
            body += indent(equality_lines)
            body += indent(output_lines)
            body += [f"        lk_results_{j}.push_back({output_names[j]});" for j in range(n_outputs)]
            body.append(f"        if ({empty_check}) {{")
            body += [f"            lk_tbl_{name}.push_back(nullptr);" for name in plan["elem_tensors"]]
            body += [f"            lk_tbl_{name}.push_back(0);" for name in plan["elem_scalars"]]
            body += [
                "            lk_tbl_gx.push_back(0);", "            lk_tbl_gy.push_back(0);", "            continue;",
                "        }"
            ]
            body += indent(scalar_lines)
            body += indent(grid_lines)
            body.append('        TORCH_CHECK(lk_grid_z == 1, "single_launch foreach '
                        'requires launch grid z == 1");')
            body += [f"        lk_tbl_{name}.push_back({name}.data_ptr());" for name in plan["elem_tensors"]]
            for name in plan["elem_scalars"]:
                ctype = _FOREACH_SCALAR_CPP[spec["_scalar_types"][name]]
                body.append(f"        lk_tbl_{name}.push_back(static_cast<{ctype}>({name}));")
            body += [
                "        lk_tbl_gx.push_back(static_cast<unsigned>(lk_grid_x));",
                "        lk_tbl_gy.push_back(static_cast<unsigned>(lk_grid_y));",
                "        if (lk_grid_x > static_cast<int64_t>(lk_gx_max)) "
                "lk_gx_max = static_cast<unsigned>(lk_grid_x);",
                "        if (lk_grid_y > static_cast<int64_t>(lk_gy_max)) "
                "lk_gy_max = static_cast<unsigned>(lk_grid_y);"
            ]
            ret_grouped = ("lk_results_0" if n_outputs == 1 else "std::make_tuple(" +
                           ", ".join(f"lk_results_{j}" for j in range(n_outputs)) + ")")
            grouped_ret = ("std::vector<torch::Tensor>" if n_outputs == 1 else "std::tuple<" +
                           ", ".join("std::vector<torch::Tensor>" for _ in range(n_outputs)) + ">")
            call_args = [f"lk_tbl_{name}.data()" for name in plan["elem_tensors"]]
            call_args += [f"lk_tbl_{name}.data()" for name in plan["elem_scalars"]]
            call_args += ["lk_tbl_gx.data()", "lk_tbl_gy.data()", f"static_cast<unsigned>({first}_list.size())"]
            for name in plan["uniform"]:
                if name in spec["_tensor_types"]:
                    call_args.append(f"{name}.data_ptr()")
                elif _scalar_splits(spec, name):
                    call_args += [f"{name}.real()", f"{name}.imag()"]
                else:
                    call_args.append(name)
            call_args += ["lk_gx_max", "lk_gy_max", "lk_stream"]
            foreach_source = (
                f"static {grouped_ret} {grouped}({', '.join(declaration)}) {{\n" + "\n".join(list_checks) + "\n" +
                f"    TORCH_CHECK({first}_list.size() >= 1 && {first}_list.size() <= " +
                f"{max_group}, \"single_launch foreach group count out of range\");\n" + "\n".join(collectors) + "\n" +
                results + "\n" + "    unsigned lk_gx_max = 0;\n    unsigned lk_gy_max = 0;\n" +
                f"    for (size_t lk_group = 0; lk_group < {first}_list.size(); ++lk_group) {{\n" + "\n".join(body) +
                "\n    }\n" + f"    if (lk_gx_max == 0 || lk_gy_max == 0) return {ret_grouped};\n" +
                "    cudaStream_t lk_stream = c10::cuda::getCurrentCUDAStream().stream();\n" +
                f"    cudaError_t lk_fused_error = {launcher}_fused(\n        " + ",\n        ".join(call_args) +
                ");\n" + "    TORCH_CHECK(lk_fused_error == cudaSuccess, " + "cudaGetErrorString(lk_fused_error));\n" +
                f"    return {ret_grouped};\n}}\n\n")
        elif spec["_foreach_dispatch"] == "single":
            foreach_source, foreach_register = _render_foreach_single_binding(spec, launcher)
        elif n_outputs == 1:
            foreach_source = (f"static std::vector<torch::Tensor> {grouped}({', '.join(declaration)}) {{\n" +
                              "\n".join(list_checks) + "\n"
                              "    std::vector<torch::Tensor> lk_results;\n"
                              f"    lk_results.reserve({first}_list.size());\n"
                              f"    for (size_t lk_group = 0; lk_group < {first}_list.size(); ++lk_group) {{\n"
                              f"        lk_results.push_back({function}({', '.join(calls)}));\n"
                              "    }\n    return lk_results;\n}\n\n")
        else:
            # One result list per output: aggregate the per-group output tuple
            # so pybind returns a Python tuple of K-lists (one per output),
            # preserving the declared aliased structure exactly.
            tuple_vec_type = ("std::tuple<" + ", ".join("std::vector<torch::Tensor>" for _ in outputs) + ">")
            init_lines = "\n".join(
                f"    std::get<{i}>(lk_results).reserve({first}_list.size());" for i in range(len(outputs)))
            push_lines = "\n".join(f"        std::get<{i}>(lk_results).push_back(std::move(std::get<{i}>(lk_out)));"
                                   for i in range(len(outputs)))
            foreach_source = (f"static {tuple_vec_type} {grouped}({', '.join(declaration)}) {{\n" +
                              "\n".join(list_checks) + "\n"
                              f"    {tuple_vec_type} lk_results;\n" + init_lines + "\n"
                              f"    for (size_t lk_group = 0; lk_group < {first}_list.size(); ++lk_group) {{\n"
                              f"        auto lk_out = {function}({', '.join(calls)});\n" + push_lines + "\n"
                              "    }\n    return lk_results;\n}\n\n")
        group_pybind_args = ", ".join(f'pybind11::arg("{name}")' for name in names)
        if spec["_foreach_dispatch"] != "single":
            # dispatch "single" emits its own pybind register inside
            # _render_foreach_single_binding; grouped/loop and single_launch
            # share the per-group wrapper registered here.
            foreach_register = f'    m.def("{grouped}", &{grouped}, {group_pybind_args});\n'
    fused_extern = ""
    if spec["_foreach_inputs"] and spec.get("_foreach_single_launch"):
        fused_params = [f"const void* const* lk_tbl_{name}" for name in plan["elem_tensors"]]
        for name in plan["elem_scalars"]:
            ctype = _FOREACH_SCALAR_CPP[spec["_scalar_types"][name]]
            fused_params.append(f"const {ctype}* lk_tbl_{name}")
        fused_params += ["const unsigned* tbl_lk_gx", "const unsigned* tbl_lk_gy", "unsigned lk_count"]
        fused_params += [_launcher_arg_decl(spec, name) for name in plan["uniform"]]
        fused_params += ["unsigned grid_x", "unsigned grid_y", "cudaStream_t stream"]
        fused_extern = (f'extern "C" cudaError_t {launcher}_fused({", ".join(fused_params)});\n')
    input_params = []
    trailing_params = []
    for x in input_names:
        if x in present_inputs:
            input_params.append(f"std::optional<torch::Tensor> {x}_opt")
        elif x in optional_inputs:
            if trailing:
                trailing_params.append(f"const c10::optional<torch::Tensor>& {x}")
            else:
                input_params.append(f"const c10::optional<torch::Tensor>& lk_opt_{x}")
        else:
            input_params.append("torch::Tensor " + x)
    return (
        "// Generated by LittleKernel for vibe-cuda-eval.\n"
        "#include <cstdint>\n#include <climits>\n#include <cmath>\n#include <cuda.h>\n#include <cuda_runtime.h>\n"
        "#include <c10/util/Optional.h>\n"
        "#include <torch/types.h>\n#include <torch/csrc/utils/pybind.h>\n#include <pybind11/stl.h>\n" +
        complex_include + "#include <vector>\n" + ("#include <optional>\n" if optional_inputs else "") +
        "#include <ATen/ops/empty.h>\n"
        "#include <c10/cuda/CUDAStream.h>\n#include \"binding_registry.h\"\n\n"
        "static int64_t lk_ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }\n"
        "static int64_t lk_min(int64_t a, int64_t b) { return a < b ? a : b; }\n"
        "static int64_t lk_max(int64_t a, int64_t b) { return a > b ? a : b; }\n"
        # Host-side device SM count for device_sms() launch expressions;
        # resolved at every launch on the stream's current device.
        "static int64_t lk_device_sms() {\n"
        "    int lk_device = 0;\n"
        "    TORCH_CHECK(cudaGetDevice(&lk_device) == cudaSuccess, \"device_sms(): cudaGetDevice failed\");\n"
        "    int lk_sms = 0;\n"
        "    TORCH_CHECK(cudaDeviceGetAttribute(&lk_sms, cudaDevAttrMultiProcessorCount, lk_device) == cudaSuccess,\n"
        "                \"device_sms(): multiprocessor count query failed\");\n"
        "    return static_cast<int64_t>(lk_sms);\n"
        "}\n" + host_list_helpers + tma_helper + "\n".join(transform_decls) + "\n"
        f"extern \"C\" cudaError_t {launcher}({', '.join(args)});\n" + fused_extern + "\n"
        # Wrapper signature: per input the optional parameter spelling
        # matches its binding mode — present()-driven inputs arrive as
        # std::optional values `<name>_opt` in input order; plain optionals
        # arrive as c10::optional const refs, either `lk_opt_<name>` in input
        # order (positional ABI) or `<name>` appended after the runtime scalar
        # and constructor parameters (trailing ABI, so pybind11 defaults stay
        # trailing). Non-optional inputs stay plain torch::Tensor. tensor_lists
        # are std::vector params appended after the named inputs.
        f"static {return_type} {function}({', '.join(input_params + ['std::vector<torch::Tensor> ' + x for x in list_names] + runtime_decls + [_CONSTRUCTOR_CPP[item['dtype']] + ' ' + item['name'] for item in constructor] + selector_decls + trailing_params)}) {{\n"
        + "\n".join(binding_decls + checks) + "\n"
        # constexpr-folded scalars (incoming side): emitted as C++ constexpr
        # after the checks, excluded from the runtime launcher ABI above.
        + ("\n".join(constexpr_lines) + "\n" if constexpr_lines else "") + "\n".join(output_lines) + "\n" +
        empty_return + "\n".join(temp_lines + equality_lines + scalar_lines + grid_lines) + "\n"
        "    cudaStream_t lk_stream = c10::cuda::getCurrentCUDAStream().stream();\n" +
        "\n".join(output_zero_lines + workspace_zero_lines + transform_lines + pre_lines + descriptor_lines +
                  list_ptr_lines) + "\n"
        f"    cudaError_t error = {launcher}({', '.join(launch_args)});\n"
        "    TORCH_CHECK(error == cudaSuccess, cudaGetErrorString(error));\n" + "\n".join(post_lines) + "\n"
        f"    return {return_value};\n"
        "}\n\n" + foreach_source + f"static void register_{function}(pybind11::module& m) {{\n"
        f"    m.def(\"{function}\", &{function}, {pybind_args});\n" + foreach_register + "}\n"
        f"REGISTER_BINDING({function}, register_{function});\n")


def _render_model(spec: dict) -> str:
    foreach = set(spec["_foreach_inputs"])
    optional = set(spec["_optional_inputs"])
    # Plain optionals under the trailing-ABI become Python-None-defaulted
    # trailing forward parameters passed to the extension by keyword, after
    # the constructor values; positional-ABI and present()-driven inputs stay
    # positional. Validation already rejects non-trailing optionals in
    # trailing-ABI manifests, so the =None defaults never precede a required
    # parameter.
    trailing = bool(spec.get("_optional_trailing"))
    present_inputs = set(spec.get("_present_inputs", ()))
    kw_optional = {name for name in optional if trailing and name not in present_inputs}
    # Container projections (union merge): building the binding-order call
    # list through the projection-aware loop below reduces exactly to the
    # plain `inputs + _list_names + runtime` list when no input projects out
    # of a container; with projections the call site spells cache["key"]
    # while forward_inputs keeps one parameter per container in first-use
    # declaration order.
    input_from = spec.get("_input_from", {})
    ignored = spec.get("_ignored_forward_args", [])
    call_inputs = []
    forward_inputs = []
    seen_containers = set()
    for x in spec["inputs"]:
        name = x["name"] + ("_list" if x["name"] in foreach else "")
        projection = input_from.get(x["name"])
        if projection is None:
            call_inputs.append(name)
            forward_inputs.append(name)
        else:
            container, key = projection
            call_inputs.append(f"{container}[{json.dumps(key)}]")
            if container not in seen_containers:
                seen_containers.add(container)
                forward_inputs.append(container)
    runtime = [item["name"] for item in spec["scalars"] if item["_runtime"]]
    inputs = call_inputs + spec["_list_names"] + runtime
    unpack_map = spec["_forward_unpack"]
    forward_args = spec.get("_forward_args")
    if forward_args is not None and foreach:
        raise ValueError("forward_args cannot be combined with a foreach group")
    if unpack_map and foreach:
        raise ValueError("forward_unpack cannot be combined with a foreach group")
    # forward_params follow the (possibly permuted) forward pool: inputs, then
    # tuple arguments, then uncovered runtime scalars. The extension is always
    # called in binding order (inputs, all runtime scalars, constructor).
    forward_params = (
        [name + ("_list" if name in foreach else "") for name in forward_args] if forward_args is not None else
        (forward_inputs + spec["_list_names"] + ignored + runtime if
         (input_from or ignored) else [name + ("_list" if name in foreach else "") for name in spec["_forward_pool"]]))
    forward_params = [param + "=None" if param in kw_optional else param for param in forward_params]
    unpack_lines = "".join(f"        {', '.join(members)}{',' if len(members) == 1 else ''}"
                           f" = {tuple_name}\n" for tuple_name, members in unpack_map.items())
    constructor = spec["constructor"]
    derived_selectors = spec["_constexpr_derived"]
    derived_names = {item["name"] for item in derived_selectors}
    # A derived selector is not itself a Model parameter: its value follows
    # from the ordinary constructor arguments it was declared from.
    init_args = [
        item["name"] + ("=" + repr(item["default"]) if "default" in item else "")
        for item in constructor
        if item["name"] not in derived_names
    ]
    stored = [item["name"] for item in constructor if item["dtype"] != "ignored" and "constexpr" not in item]
    # The literal is baked into the CUDA kernel; accept the constructor argument for
    # ABI compatibility but fail loudly on a mismatched runtime value.
    guards = "".join(f"        if {item['name']} != {item['constexpr']!r}:\n"
                     f"            raise ValueError(\"constexpr constructor '{item['name']}' "
                     f"must equal {item['constexpr']!r}, got \" + repr({item['name']}))\n"
                     for item in constructor
                     if "constexpr" in item and not item.get("_derived_constexpr"))
    # Derived constexpr selectors are evaluated at construction from the
    # actual constructor values (the expression is plain Python over the
    # constructor locals) and forwarded to pick the baked kernel variant.
    derived_lines = "".join(
        f"        self._lk_cx_{item['name']} = bool({item['expr']})\n" for item in derived_selectors)
    function = "agent_" + spec["_function"]
    if foreach:
        function += ("_single" if spec["_foreach_dispatch"] == "single" else "_grouped")
    # Positional arguments: every non-optional forward input (in declared
    # order, then the constructor values; a trailing-ABI optional is passed
    # by keyword after them so its pybind default stays trailing.
    kw_optional_order = [x["name"] for x in spec["inputs"] if x["name"] in kw_optional]
    call_args = ([name for name in inputs if name not in kw_optional] + ["self." + name for name in stored] +
                 [f"{name}={name}" for name in kw_optional_order] +
                 [f"lk_cx_{item['name']}=self._lk_cx_{item['name']}" for item in derived_selectors])
    call = f"cuda_extension.{function}({', '.join(call_args)})"
    outputs = spec["_outputs"]
    wrap_single = spec.get("_wrap_single_as_tuple", False)
    if foreach or not any(item.get("alias_of") for item in outputs):
        # wrap_single_as_tuple: preserve a reference contract of `(y,)` for a
        # single freshly computed output (validated to be non-foreach,
        # single-output in _load_spec).
        body = (f"        return ({call},)\n" if wrap_single else f"        return {call}\n")
    else:
        # Aliased outputs are the caller's own buffers updated in place; the
        # pybind layer only sees fresh wrappers around the same storage, so
        # return the original forward arguments to preserve object identity.
        result = [item["alias_of"] if item.get("alias_of") else None for item in outputs]
        if len(outputs) == 1:
            returned = result[0] if result[0] is not None else "lk_outputs"
            if wrap_single:
                returned = f"({returned},)"
            body = f"        lk_outputs = {call}\n        return {returned}\n"
        else:
            returned = ", ".join(name if name is not None else f"lk_outputs[{index}]"
                                 for index, name in enumerate(result))
            body = f"        lk_outputs = {call}\n        return ({returned})\n"
    return ("# Generated by LittleKernel for vibe-cuda-eval.\n"
            "import torch\nimport cuda_extension\n\n\n"
            "class Model(torch.nn.Module):\n"
            f"    def __init__(self{', ' if init_args else ''}{', '.join(init_args)}):\n"
            "        super().__init__()\n" + guards + derived_lines +
            "".join(f"        self.{name} = {name}\n" for name in stored) + "\n"
            f"    def forward(self, {', '.join(forward_params)}):\n" + unpack_lines + body)
