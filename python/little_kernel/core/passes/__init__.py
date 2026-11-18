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

# Import pass_registry first (no dependencies)
from little_kernel.core.passes.pass_registry import PassSpec, Pipeline, PassRegistry, get_pass_registry, get_pipeline, register_pass, get_passes, empty_pass, debug_wrapper


class _DefaultPassesView:
    """Mapping view whose values are fresh default-pass lists per lookup.

    Legacy callers write ``passes = PASSES["cuda"]``.  Each lookup now hands
    out an independent list of pass callables, so no consumer can mutate or
    hold a stale reference to the global default pipeline (RF07).

    Every accessor resolves the requested name (or alias) to the canonical
    backend id first, so ``PASSES["cuda"]`` and ``PASSES["nv/sm90"]`` share
    one registry entry and pipeline instead of registering duplicate,
    alias-keyed stage lists.
    """

    @staticmethod
    def _canonical_id(backend: str) -> str:
        from little_kernel.backends.registry import resolve_backend_spec
        return resolve_backend_spec(backend).backend_id

    def __getitem__(self, backend: str):
        from little_kernel.core.passes.pass_config import get_passes_for_backend
        return get_passes_for_backend(self._canonical_id(backend))

    def get(self, backend: str, default=None):
        if backend not in self:
            return default
        return self[backend]

    def pipeline(self, backend: str) -> "Pipeline":
        from .pass_config import get_default_pipeline
        return get_default_pipeline(self._canonical_id(backend))

    def __contains__(self, backend: str) -> bool:
        from little_kernel.backends import resolve_backend_spec, BackendUnavailable
        try:
            resolve_backend_spec(backend)
            return True
        except BackendUnavailable:
            return False

    def keys(self):
        from little_kernel.backends.registry import backend_specs
        return tuple(dict.fromkeys(name for spec in backend_specs() for name in (spec.backend_id, *spec.aliases)))

    def __iter__(self):
        return iter(self.keys())

    def __repr__(self):
        return "PASSES(fresh default-pipeline view)"


PASSES = _DefaultPassesView()

# Import passes  # noqa: E402
from little_kernel.core.passes.pass_base import *
from little_kernel.core.passes.constfold import *
from little_kernel.core.passes.inline import *
from little_kernel.core.passes.utils.add_parent_reference import *
from little_kernel.core.passes.special_struct_materialize_pass import special_struct_materialize_pass as special_struct_materialize_pass
from little_kernel.core.passes.method_to_intrin import method_to_intrin as method_to_intrin
from little_kernel.core.passes.flatten_empty import flatten_empty as flatten_empty
from little_kernel.core.passes.dump import dump_ast as dump_ast

# Import pass_config after passes
from little_kernel.core.passes.pass_config import DEFAULT_PASS_IDS, register_default_passes as register_default_passes, get_default_pipeline, get_passes_for_backend, PASS_DEBUG as PASS_DEBUG, PASS_DEBUG_ALL as PASS_DEBUG_ALL, SIMPLE_MODE as SIMPLE_MODE

__all__ = [
    "PassSpec",
    "Pipeline",
    "PassRegistry",
    "get_pass_registry",
    "get_pipeline",
    "get_default_pipeline",
    "register_pass",
    "get_passes",
    "get_passes_for_backend",
    "register_default_passes",
    "empty_pass",
    "debug_wrapper",
    "PASSES",
    "DEFAULT_PASS_IDS",
]
