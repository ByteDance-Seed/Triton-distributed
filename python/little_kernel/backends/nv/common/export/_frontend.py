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
"""Frontend loading and entry-parameter ABI records.

Resolves the authored LittleKernel entry, reads the tensor element dtypes the
frontend declared (the single source of truth for void* transport casts), and
declares launcher/entry argument types from the resolved spec view.
"""

from __future__ import annotations

import importlib.util
import inspect
import typing
from pathlib import Path

from little_kernel.backends.nv.common.export._schema import _CONSTRUCTOR_CPP, _DTYPES, _SCALAR_COMPONENT


def _load_kernel(source: Path, name: str):
    module_spec = importlib.util.spec_from_file_location("_lk_eval_draft", source)
    if module_spec is None or module_spec.loader is None:
        raise ValueError(f"cannot load LittleKernel source: {source}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, name, None)
    if kernel is None or not hasattr(kernel, "compile") or not hasattr(kernel, "py_func"):
        raise ValueError(f"{name} is not a LittleKernel entry in {source}")
    defined_in = inspect.getsourcefile(kernel.py_func)
    if defined_in is None or Path(defined_in).resolve() != source:
        raise ValueError(f"{name} must be defined in the agent's frontend source, not imported")
    return kernel


def _declared_tensor_dtypes(kernel) -> dict:
    """Frontend-declared tensor element dtypes, keyed by kernel parameter name.

    The launcher's void* transport casts in _cuda_kernel_arg must equal the
    generated kernel signature's parameter element type: C++ has no implicit
    conversion between distinct one-byte pointer types (bool* vs uint8_t*),
    so keying the cast on the spec/storage dtype alone breaks the legacy
    byte-view pattern (spec dtype "bool", draft declaring
    ll.Tensor[ll.uint8]) with a translation unit that nvcc rejects. The
    frontend annotation is the same source the code generator used to emit
    the signature, so resolving it here keeps both sides in agreement by
    construction. Parameters whose annotations do not resolve to a Tensor
    element type are omitted, and _cuda_kernel_arg falls back to the
    spec/storage dtype; every program whose declared and spec dtypes already
    agree (the common case, including complex tensors whose storage element
    is float/double on both sides) renders byte-identical output.
    """
    from little_kernel.core.type_system import TensorType
    element_map = {}
    try:
        import little_kernel.language as ll
        for attr, dtype in (("float16", "float16"), ("bfloat16", "bfloat16"), ("float32", "float32"),
                            ("float64", "float64"), ("int32", "int32"), ("int64", "int64"), ("int8", "int8"),
                            ("uint8", "uint8"), ("int16", "int16"), ("uint16", "uint16"), ("bool_", "bool")):
            lltype = getattr(ll, attr, None)
            if lltype is not None:
                element_map[lltype] = dtype
    except Exception:
        return {}
    declared = {}
    try:
        hints = typing.get_type_hints(kernel.py_func)
    except Exception:
        hints = {}
    for name, param in inspect.signature(kernel.py_func).parameters.items():
        annotation = hints.get(name, param.annotation)
        if isinstance(annotation, str):
            try:
                annotation = eval(annotation, dict(kernel.py_func.__globals__))
            except Exception:
                continue
        if isinstance(annotation, TensorType):
            dtype = element_map.get(annotation.element_type)
            if dtype is not None:
                declared[name] = dtype
    return declared


def _scalar_splits(spec: dict, name: str) -> bool:
    """A complex scalar name expands to two component kernel parameters."""
    return (name in spec["_scalar_types"] and name not in spec["_constructor_types"]
            and spec["_scalar_types"][name] in _SCALAR_COMPONENT)


def _launcher_arg_decl(spec: dict, name: str) -> str:
    # Injected presence scalars are neither declared spec scalars nor
    # constructor values, so the dtype-keyed paths below cannot resolve them;
    # they ride the ABI as the frontend's trailing ll.int32 parameter.
    if name in spec.get("_presence_args", {}):
        return f"int32_t {name}"
    if name in spec["_descriptor_names"]:
        # A per-batch bundle transports a device-resident map array by
        # pointer; a singleton transports one host-encoded map by value.
        if name in spec.get("_descriptor_bundles", {}):
            return f"CUtensorMap* {name}"
        return f"CUtensorMap {name}"
    if name in spec["_list_param_types"]:
        return f"void* {name}"
    if name in spec["_tensor_types"]:
        return f"void* {name}"
    if name in spec["_constructor_types"]:
        return f"{_CONSTRUCTOR_CPP[spec['_constructor_types'][name]]} {name}"
    dtype = spec["_scalar_types"][name]
    if dtype in _SCALAR_COMPONENT:
        component = _SCALAR_COMPONENT[dtype]
        return f"{component} {name}_re, {component} {name}_im"
    cpp = {"float32": "float", "float64": "double"}.get(dtype, "int64_t")
    return f"{cpp} {name}"


def _descriptor_param_kind(annotation):
    """Classify a typed-IR TMA descriptor parameter annotation.

    A bare ``TmaDescriptor`` entry-parameter annotation is the documented
    shorthand for the by-value single-descriptor transport (the entry
    signature emitter resolves it to ``grid_constant[TmaDescriptor]`` via
    :func:`normalize_entry_grid_constant_param`, matching the runtime ABI);
    the explicit ``grid_constant[TmaDescriptor]`` spelling (optionally
    wrapped in ``const[...]``) is equally accepted. ``ptr[TmaDescriptor]``
    is the device-array bundle transport. ``None`` means the annotation is
    not a legal descriptor parameter, so the manifest/IR ABI check can
    reject it.
    """
    from little_kernel.core.type_system import Const, GridConstant, Pointer
    from little_kernel.backends.nv.common.core.types import TmaDescriptorType
    t = annotation
    while isinstance(t, Const) and not isinstance(t, GridConstant):
        t = t.inner_type
    if isinstance(t, TmaDescriptorType):
        return "grid_constant"
    if isinstance(t, GridConstant) and isinstance(t.inner_type, TmaDescriptorType):
        return "grid_constant"
    if isinstance(t, Pointer) and isinstance(t.inner_type, TmaDescriptorType):
        return "pointer"
    return None


def _cuda_kernel_arg(spec: dict, name: str) -> str:
    # Injected presence scalars transport as int32_t into the entry kernel;
    # they are absent from _scalar_types and _constructor_types, so the
    # dtype-keyed cast below cannot see them.
    if name in spec.get("_presence_args", {}):
        return f"static_cast<int32_t>({name})"
    if name in spec["_descriptor_names"]:
        return name
    # The void* transport cast is keyed by the frontend's declared tensor
    # element type (spec["_declared_dtypes"], resolved in _render_cuda from
    # the same annotations the code generator used for the kernel signature),
    # falling back to the spec/storage dtype when no declared type resolves.
    # Invariant: the cast type must equal the entry signature's parameter
    # element type; C++ has no implicit conversion between distinct one-byte
    # pointer types (bool* vs uint8_t*), so the storage table alone cannot
    # serve both ll.Tensor[ll.bool_] and the legacy ll.Tensor[ll.uint8]
    # byte-view pattern backed by at::kBool storage.
    declared = spec.get("_declared_dtypes") or {}
    if name in spec["_list_param_types"]:
        dtype = declared.get(name, spec["_list_param_types"][name])
        return f"reinterpret_cast<{_DTYPES[dtype][0]}*>({name})"
    if name in spec["_tensor_types"]:
        dtype = declared.get(name, spec["_tensor_types"][name])
        return f"reinterpret_cast<{_DTYPES[dtype][0]}*>({name})"
    dtype = spec["_constructor_types"].get(name, spec["_scalar_types"].get(name))
    if dtype in _SCALAR_COMPONENT:
        return f"{name}_re, {name}_im"
    casts = {"float32": "float", "float64": "double", "bool": "bool", "int64": "int64_t", "int32": "int32_t"}
    if dtype not in casts:
        raise ValueError(f"unsupported scalar dtype for kernel argument {name}: {dtype}")
    return f"static_cast<{casts[dtype]}>({name})"


def _kernel_param_names(spec: dict) -> list:
    """Frontend function parameter names expected for spec['_kernel_args']."""
    names = []
    for name in spec["_kernel_args"]:
        if _scalar_splits(spec, name):
            names += [f"{name}_re", f"{name}_im"]
        else:
            names.append(name)
    return names
