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
try:
    from torch.utils._ordered_set import OrderedSet
    from torch._inductor.utils import triton_version_uses_attrs_dict
    from torch._inductor.runtime.triton_heuristics import (
        GridExpr,
        config_to_dict,
        get_first_attr,
    )
except ImportError:
    # earlier version (e.g. torch 2.4) won't have these imports. The patch wont work.
    # To avoid import errors, we can just skip these imports.
    pass


def make_launcher(self):
    from triton import knobs
    """
    Launching triton kernels is performance sensitive, we compile
    a custom Python function get the grid() and reorder the args to
    the underlying wrapper.
    """
    cfg = self.config
    compile_meta = self.compile_meta
    binary = self.kernel
    fn = binary.src.fn
    binary._init_handles()
    """
    https://github.com/pytorch/pytorch/issues/115344

    self.fn.constexprs doesn't properly deal with None args, so when we filter out
    an arg in UserDefinedTritonKernel.codegen, we need to filter it here as well.
    We also don't want to modify self.fn.

    We know that we removed something from the signature if:
        1. It's in compile_meta["constants"]
        2. It isn't a constant we already know about
            Note: The value of interest has already been added to compile_meta['constants'],
                so we use self.fn.constexprs instead.
        3. It isn't in the compile_meta signature
    """
    known_constants = OrderedSet(arg for i, arg in enumerate(fn.arg_names) if i in fn.constexprs)
    none_args = OrderedSet(k for k, v in compile_meta["constants"].items() if v is None and k not in known_constants)
    none_args = none_args.difference(OrderedSet(compile_meta["signature"].keys()))

    if triton_version_uses_attrs_dict():
        call_args = fn.arg_names
        def_args = fn.arg_names
        if ("num_warps" in compile_meta["constants"] or "num_stages" in compile_meta["constants"]):
            # num_warps/num_stages are special implicit args that are not in the signature
            # see test_triton_kernel_special_params
            def_args = [arg for arg in def_args if arg not in ("num_warps", "num_stages")]
            repl = {k: str(compile_meta["constants"].get(k)) for k in ("num_warps", "num_stages")}
            call_args = [repl.get(arg, arg) for arg in call_args]
    else:
        call_args = [arg for i, arg in enumerate(fn.arg_names) if i not in fn.constexprs and arg not in none_args]
        cfg_dict = config_to_dict(cfg)
        def_args = [name for name in fn.arg_names if name not in cfg_dict and name not in none_args]

    binary_shared = (binary.shared if hasattr(binary, "shared") else binary.metadata.shared)

    scope = {
        "grid_meta":
        cfg.kwargs,
        "bin":
        binary,
        "launch_enter_hook":
        knobs.runtime.launch_enter_hook,
        "launch_exit_hook":
        knobs.runtime.launch_exit_hook,
        "metadata": (binary.packed_metadata if hasattr(binary, "packed_metadata") else binary.metadata),
        "shared":
        binary_shared,
        "num_warps": (binary.num_warps if hasattr(binary, "num_warps") else binary.metadata.num_warps),
        "cta_args": ((
            binary.num_ctas,
            *get_first_attr(binary, "cluster_dims", "clusterDims"),
        ) if hasattr(binary, "num_ctas") else
                     ((binary.metadata.num_ctas, *binary.metadata.cluster_dims) if hasattr(binary, "metadata") else
                      ())),
        "function":
        get_first_attr(binary, "function", "cu_function"),
        "runner":
        get_first_attr(binary, "run", "c_wrapper"),
    }

    if not hasattr(binary, "launch_metadata"):
        # launch args before CompiledKernel.launch_metadata is added.
        # TODO(jansel): delete this branch in mid-2025
        runner_args = [
            "grid_0",
            "grid_1",
            "grid_2",
            "num_warps",
            "*cta_args",
            "shared",
            "stream",
            "function",
            "launch_enter_hook",
            "launch_exit_hook",
            "metadata",
            *call_args,
        ]
    else:  # args after CompiledKernel.launch_metadata: https://github.com/openai/triton/pull/3492
        # Getting the kernel launch args is extremely perf-sensitive.  Evaluating
        # `bin.launch_metadata` is relatively expensive, and returns None unless a
        # `launch_enter_hook` is installed.  So if we don't have that hook installed,
        # we want to burn None in to the launch args with zero overhead.
        # See https://github.com/pytorch/pytorch/issues/123597
        if knobs.runtime.launch_enter_hook:
            launch_metadata = f"bin.launch_metadata((grid_0, grid_1, grid_2), stream, {', '.join(call_args)})"
        else:
            launch_metadata = "None"
        runner_args = [
            "grid_0",
            "grid_1",
            "grid_2",
            "stream",
            "function",
            "metadata",
            launch_metadata,
            "launch_enter_hook",
            "launch_exit_hook",
            *call_args,
        ]

    if "extra_launcher_args" in self.inductor_meta:
        def_args = [*def_args, *self.inductor_meta["extra_launcher_args"]]

    grid = GridExpr.from_meta(self.inductor_meta, cfg)
    # grid.prefix is usually empty, grid.x_grid is something like `-(xnumel//-1024)`
    lines = [
        f"def launcher({', '.join(def_args)}, stream):",
        *[f"    {line}" for line in grid.prefix],
        f"    grid_0 = {grid.x_grid}",
        f"    grid_1 = {grid.y_grid}",
        f"    grid_2 = {grid.z_grid}",
        f"    runner({', '.join(runner_args)})",
    ]
    exec("\n".join(lines), scope)

    launcher = scope["launcher"]
    launcher.config = cfg
    launcher.n_regs = getattr(binary, "n_regs", None)
    launcher.n_spills = getattr(binary, "n_spills", None)
    launcher.shared = binary_shared
    launcher.store_cubin = self.inductor_meta.get("store_cubin", False)
    # store this global variable to avoid the high overhead of reading it when calling run
    if launcher.store_cubin:
        launcher.fn = fn
        launcher.bin = binary
        if triton_version_uses_attrs_dict():
            # arg filtering wasn't done above
            cfg_dict = config_to_dict(cfg)
            def_args = [x for x in def_args if x not in cfg_dict]
            call_args = [
                x for x in call_args
                if compile_meta["signature"].get(x, "constexpr") != "constexpr" and x not in none_args
            ]
        launcher.def_args = def_args
        launcher.call_args = call_args
    return launcher


def apply_patch_inductor_triton_heuristics():

    from torch._inductor.runtime.triton_heuristics import TritonCompileResult

    TritonCompileResult.make_launcher = make_launcher


def apply_patch_inductor_triton_compat():
    from triton import knobs
    from torch._inductor.runtime import triton_compat

    # from torch._inductor.runtime import triton_heuristics
    # setattr(triton_heuristics, "knobs", knobs)
    setattr(triton_compat, "knobs", knobs)
    triton_compat.__all__.append("knobs")


def apply_triton340_inductor_patch():
    """
    Call this function Before your program to apply the necessary patches for
    compatibility with Triton 3.4.0 in the context of PyTorch 2.7.1 (`torch.compile`).

    >>> apply_triton340_inductor_patch()
    >>> ... Your Code
    """
    import torch
    import triton
    from packaging.version import Version

    torch_version = torch.__version__
    triton_version = triton.__version__

    require_patched_torch = (Version("2.7.0") <= Version(torch_version) < Version("2.8.0"))
    if require_patched_torch and triton_version in ["3.4.0"]:
        # 1. Patch the `triton_compat` symbol
        apply_patch_inductor_triton_compat()

        # 2. Patch the `make_launcher` method
        apply_patch_inductor_triton_heuristics()
