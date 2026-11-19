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
"""Reusable component implementations configured by individual backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from .interfaces import BackendTarget


@dataclass(frozen=True)
class ModuleExporter:
    """Lazy adapter for a backend's implemented export/schema integration."""
    module_name: str

    def __getattr__(self, name):
        from importlib import import_module
        return getattr(import_module(self.module_name), name)


@dataclass(frozen=True)
class LazyRuntimeProvider:
    factory_reference: str
    device_keyword: str = "device"

    def create(self, device=None, **options):
        from .registry import load_symbol
        if device is not None:
            options[self.device_keyword] = device
        return load_symbol(self.factory_reference)(**options)


@dataclass(frozen=True)
class ArchitecturePolicy:
    spec: object
    default_architecture: str
    resolver_reference: str | None = None
    detector_reference: str | None = None

    def resolve(self, architecture=None, device=None):
        arch = getattr(architecture, "arch", architecture) or self.default_architecture
        if arch not in self.spec.architectures:
            raise ValueError(f"{self.spec.backend_id}: unsupported architecture {arch!r}")
        if self.resolver_reference:
            from .registry import load_symbol
            arch = load_symbol(self.resolver_reference)(arch=arch, device=device).arch
        return BackendTarget(self.spec.backend_id, arch)

    def detect(self):
        if self.detector_reference is None:
            raise RuntimeError(f"{self.spec.backend_id} has no device detector")
        from .registry import load_symbol
        architecture = load_symbol(self.detector_reference)()
        if architecture is None:
            raise RuntimeError(f"No matching device for {self.spec.backend_id}")
        return self.resolve(architecture)


@dataclass(frozen=True)
class FunctionGenerator:
    generate_function: Callable

    def generate(self, tree, context, **options):
        return self.generate_function(tree, context, **options)


@dataclass(frozen=True)
class FunctionToolchain:
    source_suffix: str
    compile_function: Callable
    architecture_keyword: str = "arch"

    def compile(self, source: str, kernel_name: str, **options) -> bytes:
        if "architecture" in options:
            value = options.pop("architecture")
            options[self.architecture_keyword] = getattr(value, "arch", value)
        return self.compile_function(source, kernel_name, **options)


@dataclass(frozen=True)
class FunctionBuilder:
    build_function: Callable
    unsupported_options: tuple[str, ...] = ()

    def build(self, request):
        for name in self.unsupported_options:
            value = request.options.get(name, getattr(request, name, None))
            if value:
                raise ValueError(f"{request.kernel.backend} does not support build option {name}")
        return self.build_function(request)


@dataclass(frozen=True)
class StandardPassProvider:
    """Compose shared passes with explicitly selected optional components."""
    gpu_allocations: bool = False
    static_unroll: bool = False

    def register(self, backend_key):
        from little_kernel.core.passes.pass_config import (PASS_DEBUG, SIMPLE_MODE, dump_ast)
        from little_kernel.core.passes.pass_registry import register_pass, empty_pass, get_pipeline
        from little_kernel.core.passes.special_struct_materialize_pass import special_struct_materialize_pass
        from little_kernel.core.passes.method_to_intrin import method_to_intrin
        from little_kernel.core.passes.constfold import const_fold
        from little_kernel.core.passes.inline import inline

        register_pass(backend_key,
                      dump_ast("initial") if PASS_DEBUG else empty_pass, "initial", enable_debug=False,
                      pass_id="initial")
        previous = "initial"
        if self.static_unroll:
            from little_kernel.core.passes.compile_time_unroll import compile_time_unroll
            register_pass(backend_key, compile_time_unroll, "compile_time_unroll", simple_mode=SIMPLE_MODE,
                          pass_id="compile_time_unroll", requires=(previous, ))
            previous = "compile_time_unroll"
        stages = [("special_struct_materialize", special_struct_materialize_pass),
                  ("method_to_intrin", method_to_intrin), ("const_fold", const_fold), ("inline", inline)]
        if self.gpu_allocations:
            from little_kernel.core.passes.flatten_empty import flatten_empty
            from little_kernel.backends.common.gpu.passes.rebind_alloc_names import rebind_alloc_names
            from little_kernel.backends.common.gpu.passes.insert_mem_alloc import insert_mem_alloc
            stages.extend([("flatten_empty", flatten_empty), ("rebind_alloc_names", rebind_alloc_names),
                           ("insert_mem_alloc", insert_mem_alloc)])
        for pass_id, function in stages:
            register_pass(backend_key, function, pass_id, simple_mode=SIMPLE_MODE, pass_id=pass_id,
                          requires=(previous, ))
            previous = pass_id
        return get_pipeline(backend_key)
