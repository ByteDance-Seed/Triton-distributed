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
"""Shared contracts implemented by hardware-specific component factories."""
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, runtime_checkable


class BackendUnavailable(RuntimeError):
    """A requested backend is absent from this source distribution."""


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    factory: str
    aliases: tuple[str, ...] = ()
    architectures: tuple[str, ...] = ()
    language_exports: Mapping[str, str] = field(default_factory=dict)
    modules: tuple[str, ...] = ()
    resources: tuple[str, ...] = ()
    runtime_exports: Mapping[str, str] = field(default_factory=dict)
    export_module: str | None = None
    declaration_limits: Mapping[str, Any] = field(default_factory=dict)
    priority: int = 0

    def __post_init__(self):
        if len(self.backend_id.split("/")) != 2 or not all(self.backend_id.split("/")):
            raise ValueError(f"Backend id must have vendor/hardware form: {self.backend_id!r}")
        module, separator, attribute = self.factory.partition(":")
        if not separator or not attribute.isidentifier() or not all(part.isidentifier() for part in module.split(".")):
            raise ValueError("Backend factory must be module:attribute")
        object.__setattr__(self, "aliases", tuple(self.aliases))
        object.__setattr__(self, "architectures", tuple(self.architectures))
        object.__setattr__(self, "modules", tuple(self.modules))
        object.__setattr__(self, "resources", tuple(self.resources))
        object.__setattr__(self, "language_exports", MappingProxyType(dict(self.language_exports)))
        object.__setattr__(self, "runtime_exports", MappingProxyType(dict(self.runtime_exports)))
        limits = {
            k: tuple(tuple(x) if isinstance(x, list) else x
                     for x in v) if isinstance(v, list) else v
            for k, v in self.declaration_limits.items()
        }
        object.__setattr__(self, "declaration_limits", MappingProxyType(limits))


@runtime_checkable
class CodeGenerator(Protocol):

    def generate(self, tree, context: Mapping[str, Any], **options) -> str:
        ...


@runtime_checkable
class PassProvider(Protocol):

    def register(self, backend_key: str):
        ...


@runtime_checkable
class Toolchain(Protocol):
    source_suffix: str

    def compile(self, source: str, kernel_name: str, **options) -> bytes:
        ...


@runtime_checkable
class KernelBuilder(Protocol):

    def build(self, request: "BuildRequest"):
        ...


@runtime_checkable
class RuntimeProvider(Protocol):

    def create(self, device=None, **options):
        ...


@runtime_checkable
class ExecutableKernel(Protocol):

    def __call__(self, *arguments, stream=None, grid=None):
        ...

    def synchronize(self, stream=None):
        ...

    def close(self):
        ...


@runtime_checkable
class TargetPolicy(Protocol):

    def resolve(self, architecture=None, device=None):
        ...

    def detect(self):
        ...


@dataclass(frozen=True)
class BackendTarget:
    backend_id: str
    arch: str

    def __str__(self):
        return self.arch


@dataclass(frozen=True)
class BuildRequest:
    kernel: Any
    passes: Any
    codegen: Callable | None
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_mem_bytes: int = 0
    architecture: str | None = None
    cache_dir: str | None = None
    verbose: bool = False
    include_paths: Any = None
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "block", tuple(self.block))
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


@dataclass(frozen=True)
class BackendComponents:
    """One hardware instance composed from shared and backend-owned components."""
    spec: BackendSpec
    codegen: CodeGenerator
    passes: PassProvider
    toolchain: Toolchain
    builder: KernelBuilder
    runtime: RuntimeProvider
    target: TargetPolicy

    def __post_init__(self):
        for name, contract in (("codegen", CodeGenerator), ("passes", PassProvider), ("toolchain", Toolchain),
                               ("builder", KernelBuilder), ("runtime", RuntimeProvider), ("target", TargetPolicy)):
            if not isinstance(getattr(self, name), contract):
                raise TypeError(f"{self.spec.backend_id}: {name} does not implement {contract.__name__}")

    @property
    def backend_id(self):
        return self.spec.backend_id

    @property
    def language(self):
        from little_kernel.backends.registry import LanguageNamespace
        return LanguageNamespace(self.spec.backend_id)

    @property
    def exporter(self):
        if self.spec.export_module is None:
            raise BackendUnavailable(f"{self.backend_id} does not provide an export-bundle integration")
        from .shared import ModuleExporter
        return ModuleExporter(self.spec.export_module)
