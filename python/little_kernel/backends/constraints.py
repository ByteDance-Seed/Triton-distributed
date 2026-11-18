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
"""Validate normalized declaration options against backend-owned capability data."""
from .registry import resolve_backend_spec


def validate_declaration(backend, options):
    spec = resolve_backend_spec(backend)
    limits = spec.declaration_limits
    for name in limits.get("unsupported_options", ()):
        if options.get(name) not in (None, False, (), frozenset()):
            raise ValueError(f"{spec.backend_id} does not support declaration option {name}")
    cluster = options.get("cluster_shape")
    max_cluster = limits.get("max_cluster_ctas")
    if cluster is not None and max_cluster is not None:
        size = cluster[0] * cluster[1] * cluster[2]
        if size > max_cluster:
            raise ValueError(f"cluster_shape={cluster} exceeds the supported {max_cluster}-CTA cluster limit")
    group = options.get("cta_group_shape")
    shapes = limits.get("cta_group_shapes")
    if group is not None and shapes is not None and group not in shapes:
        raise ValueError(f"cta_group_shape currently supports only {shapes}, got {group}")
    bounds = options.get("launch_bounds")
    max_threads = limits.get("max_threads")
    if bounds is not None and max_threads is not None and bounds[0] > max_threads:
        raise ValueError(f"launch_bounds max_threads must be in [1, {max_threads}], got {bounds[0]}")
