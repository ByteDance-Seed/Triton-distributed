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
"""Immutable compilation artifact (RF01).

A :class:`CompilationArtifact` ties generated CUDA source to its
:class:`~little_kernel.runtime.abi.KernelSignature`, entry-point name,
resolved target and immutable build metadata.  It deliberately carries no
mutable module/launcher state: those belong to the runtime/cache layer that
consumes the artifact.

Artifacts compare by value but are intentionally **not hashable**: generated
source and free-form metadata are not suitable dict keys without a separate
declared build-identity value.  The metadata mapping is deep-frozen (nested
lists/dicts are copied and made read-only).

The existing ``little_kernel.core.compile`` entry points keep returning
source strings; a future ``compile_artifact()`` (integration worker) will
construct and return this object.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

__all__ = ["CompilationArtifact"]


def _freeze(value: Any) -> Any:
    """Recursively copy and freeze metadata containers.

    Mappings become read-only ``MappingProxyType`` views, lists/tuples/sets
    become tuples of frozen elements, and immutable builtins are copied by
    value.  Custom mutable objects are rejected so metadata can never change
    behind the artifact's back.
    """
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))})
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool, bytes)) or value is None:
        return value
    raise TypeError("CompilationArtifact metadata values must be immutable builtins or "
                    f"nested containers; got mutable {type(value).__name__}")


@dataclass(frozen=True, eq=True)
class CompilationArtifact:
    """Result of compiling one kernel specialization.

    Attributes
    ----------
    source:
        Generated CUDA/C++ source text.
    kernel_name:
        Device entry-point symbol.
    signature:
        Ordered launch ABI signature (may be ``None`` for an explicitly
        untyped/raw-source path; such a path can never override a known one).
    target:
        Resolved target descriptor/arch string the source was compiled for.
    metadata:
        Deep-frozen, read-only build metadata (options, source identity,
        dependencies, launch requirements).
    """

    source: str
    kernel_name: str
    signature: Optional[Any] = None
    target: Any = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=True)

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("CompilationArtifact.source must be a non-empty str")
        if not isinstance(self.kernel_name, str) or not self.kernel_name:
            raise ValueError("CompilationArtifact.kernel_name must be a non-empty str")
        if self.signature is not None:
            # Import lazily to keep core independent of runtime import order.
            from little_kernel.runtime.abi import KernelSignature

            if not isinstance(self.signature, KernelSignature):
                raise TypeError("CompilationArtifact.signature must be a KernelSignature "
                                "or None")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("CompilationArtifact.metadata must be a mapping")
        object.__setattr__(self, "metadata", _freeze(self.metadata))

    __hash__ = None  # free-form source/metadata do not define build identity

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        if self.signature is None:
            return ()
        return self.signature.names

    def has_typed_signature(self) -> bool:
        return self.signature is not None
