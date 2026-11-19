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
"""
Default pipeline configuration (RF07).

Default pipelines are declared once with stable pass ids and prerequisites.
Registration is idempotent (calling :func:`register_default_passes` twice does
not append stages), and accessors hand back fresh lists/immutable pipelines so
consumers cannot mutate the global defaults.
"""

import os
from typing import Callable, List

from little_kernel.core.passes.pass_registry import Pipeline, register_pass, get_passes, get_pipeline, empty_pass
from little_kernel.core.passes.dump import dump_ast

# Environment variables
PASS_DEBUG = os.getenv("PASS_DEBUG", "").split(",")
if "" in PASS_DEBUG:
    PASS_DEBUG.remove("")
PASS_DEBUG_ALL = "all" in PASS_DEBUG
SIMPLE_MODE = os.getenv("SIMPLE_MODE", "true") == "true"

# Stable default pipeline stage ids, in execution order.  ``insert_mem_alloc``
# must follow ``flatten_empty``/``rebind_alloc_names`` so the memory analyzer
# it embeds sees one allocation site per duplicated name.
DEFAULT_PASS_IDS = (
    "initial",
    "special_struct_materialize",
    "method_to_intrin",
    "const_fold",
    "inline",
    "flatten_empty",
    "rebind_alloc_names",
    "insert_mem_alloc",
)


def _canonical_backend_id(backend: str) -> str:
    """Resolve a backend name or alias to its canonical backend id.

    The pass registry keys stage lists by backend id, so registering under
    both an alias (``"cuda"``) and its canonical id (``"nv/sm90"``) would
    otherwise create two divergent pipelines for one backend.
    """
    from little_kernel.backends.registry import resolve_backend_spec
    return resolve_backend_spec(backend).backend_id


def register_default_passes(backend: str = "cuda") -> Pipeline:
    from little_kernel.backends import get_backend
    backend = _canonical_backend_id(backend)
    return get_backend(backend).passes.register(backend)


def get_passes_for_backend(backend: str = "cuda") -> List[Callable]:
    """
    Get a fresh list of pass callables for a backend.

    The returned list is owned by the caller; reordering or extending it
    never changes global defaults.

    Args:
        backend: Backend name or alias

    Returns:
        Fresh list of pass functions
    """
    # Register *and* read under the canonical id: registering under the
    # canonical key but reading the raw alias returns an empty stage list.
    canonical = _canonical_backend_id(backend)
    register_default_passes(canonical)
    return get_passes(canonical)


def get_default_pipeline(backend: str = "cuda") -> Pipeline:
    """Return the immutable default :class:`Pipeline` for a backend."""
    return register_default_passes(backend)


# Auto-register default pipelines for installed backends (idempotent).
