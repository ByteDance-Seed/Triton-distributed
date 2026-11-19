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
"""Generic, evaluator-independent rendering of an export bundle.

This module turns an authored LittleKernel entry plus a typed
:class:`~little_kernel.backends.nv.common.export.ResolvedSpec` into three self-contained strings
(CUDA translation unit, pybind binding, model file). It has no knowledge of
``task/``/``agent/`` evaluator directories, ``compile.sh`` orchestration or
file rollback; those policies live in the thin compatibility runner.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from little_kernel.backends.nv.common.codegen.emitter import ALIAS_CONTRACT_CTX_KEY

from little_kernel.backends.nv.common.export._bindings import _render_binding, _render_model
from little_kernel.backends.nv.common.export._cuda import _render_cuda
from little_kernel.backends.nv.common.export._frontend import _load_kernel
from little_kernel.backends.nv.common.export._records import ResolvedSpec
from little_kernel.backends.nv.common.export._validation import validate_spec


@dataclass(frozen=True)
class ExportBundle:
    """The three generated, self-contained artifacts as strings."""

    spec: ResolvedSpec
    cuda: str
    binding: str
    model: str

    @property
    def function(self) -> str:
        return self.spec.function

    @property
    def cuda_filename(self) -> str:
        return f"agent_lk_{self.function}.cu"

    @property
    def binding_filename(self) -> str:
        return f"agent_lk_{self.function}_binding.cpp"

    @property
    def model_filename(self) -> str:
        return "model_new.py"

    def filenames(self) -> tuple:
        return (self.cuda_filename, self.binding_filename, self.model_filename)

    def write(self, output_dir) -> list:
        """Write the bundle into a generic output directory and return paths.

        The directory is created if needed; a ``kernels/`` subdirectory holds
        the CUDA/binding pair and the model sits at the directory root.
        """
        root = Path(output_dir)
        kernels = root / "kernels"
        kernels.mkdir(parents=True, exist_ok=True)
        paths = [kernels / self.cuda_filename, kernels / self.binding_filename, root / self.model_filename]
        contents = (self.cuda, self.binding, self.model)
        for path, text in zip(paths, contents):
            path.write_text(text)
        return paths


def write_bundle(source: Any, spec: Any, output_dir, *, overwrite: bool = False,
                 enforce_alias_contract: bool = True) -> ExportBundle:
    """Render and write a bundle into a generic output directory.

    Unlike the compatibility workspace writer, this requires no ``task/`` or
    ``agent/`` evaluator layout and performs no ``compile.sh``/rollback policy.
    Returns the :class:`ExportBundle` (whose written paths are the files just
    created). When *overwrite* is false and a target exists, raises
    :class:`FileExistsError`.
    """
    bundle = render_bundle(source, spec, enforce_alias_contract=enforce_alias_contract)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    paths = [
        root / "kernels" / bundle.cuda_filename, root / "kernels" / bundle.binding_filename,
        root / bundle.model_filename
    ]
    if not overwrite:
        existing = [path for path in paths if path.exists()]
        if existing:
            raise FileExistsError("generated output exists; pass overwrite=True to replace it: " +
                                  ", ".join(str(path) for path in existing))
    paths = bundle.write(root)
    return bundle


def _coerce_kernel(source: Any, kernel_name: str):
    if isinstance(source, (str, Path)):
        return _load_kernel(Path(source), kernel_name)
    # An already-loaded LLKernel entry: accept it without re-importing.
    if hasattr(source, "compile") and hasattr(source, "py_func"):
        if source.py_func.__name__ != kernel_name:
            raise ValueError(f"loaded entry {source.py_func.__name__!r} does not match spec kernel {kernel_name!r}")
        # Export applies alias/occupancy options and lowers the entry. Give
        # each render an independent kernel context so those options never
        # leak into another render of the same authored LLKernel object.
        kernel = copy.copy(source)
        kernel.ctx = dict(source.ctx)
        return kernel
    raise ValueError("source must be a path to authored LittleKernel source or "
                     "a loaded LLKernel entry")


def render_bundle(source: Any, spec: Any, *, enforce_alias_contract: bool = True) -> ExportBundle:
    """Render an :class:`ExportBundle` from an authored entry and a spec.

    ``source`` is a frontend source path or a loaded LLKernel entry. ``spec``
    is a path/mapping/:class:`RawSpec`, or an already-validated
    :class:`ResolvedSpec` (in which case validation is idempotent and reuses
    it). Rendering always reads a fresh :meth:`ResolvedSpec.legacy_mapping`
    working copy, so repeated renders share no mutable state.
    """
    resolved: ResolvedSpec = validate_spec(spec)
    kernel = _coerce_kernel(source, resolved.kernel)
    view = resolved.legacy_mapping()
    if enforce_alias_contract:
        # The generic export path applies the contract-driven entry-alias
        # policy: pointer parameters the contract proves disjoint stay
        # __restrict__; aliased groups opt out. A direct emitter caller can
        # disable it to render without touching frontend context.
        kernel.ctx[ALIAS_CONTRACT_CTX_KEY] = True
    cuda = _render_cuda(view, kernel)
    binding = _render_binding(view)
    model = _render_model(view)
    return ExportBundle(resolved, cuda, binding, model)
