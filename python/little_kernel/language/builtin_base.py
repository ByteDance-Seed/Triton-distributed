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
from dataclasses import dataclass
from typing import List, Optional, Union, Callable, Dict
from little_kernel.core.type_system import LLType

# Constants for builtin/intrin function attributes (for easier reference)
# These should match the strings in INTERNAL_ATTR
BUILTIN_ATTR = "__builtin__"
EVAL_RETURN_TYPE_ATTR = "__eval_return_type__"
EVAL_ARG_TYPE_ATTR = "__eval_arg_type__"
CONST_FUNC_ATTR = "__const_func__"
CONST_TYPE_FUNC_ATTR = "__const_type_func__"
INLINE_FUNC_ATTR = "__inline_func__"
CODEGEN_FUNC_ATTR = "__codegen_func__"
ORIGINAL_PY_FUNC_ATTR = "__original_py_func__"

# Global registry for intrin functions
# Maps function name -> function object
_INTRIN_REGISTRY: Dict[str, Callable] = {}


def get_intrin_ctx() -> Dict[str, Callable]:
    """
    Get a context dictionary containing all registered intrin functions.
    
    Returns:
        A dictionary mapping intrin function names to function objects.
    """
    return dict(_INTRIN_REGISTRY)


# Unified list of all internal attributes that should be preserved
# This includes both function object attributes and AST node attributes
INTERNAL_ATTR = [
    # Function object attributes (for builtin decorators)
    BUILTIN_ATTR,
    EVAL_RETURN_TYPE_ATTR,
    EVAL_ARG_TYPE_ATTR,
    CONST_FUNC_ATTR,
    CONST_TYPE_FUNC_ATTR,
    INLINE_FUNC_ATTR,
    CODEGEN_FUNC_ATTR,
    ORIGINAL_PY_FUNC_ATTR,
    # Add more special attributes here as needed
]

# AST-specific special attributes (subset of INTERNAL_ATTR, only AST node attributes)
# These are attributes that should be preserved when copying/transforming AST nodes
AST_SPECIAL_ATTRS = [
    "_is_special_struct_decl",
    "_special_struct_name",
    "_special_struct_info",
    "_is_special_struct_method",
    "_special_struct_var",
    "_method_name",
    "_is_special_struct",
    # Add more AST-specific attributes here as needed
]


def builtin(eval_return_type: Union[LLType, Callable], codegen_func: Callable,
            eval_arg_type: Optional[Callable] = None):

    def _builtin(func):

        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__name__ = func.__name__
        wrapper.__doc__ = func.__doc__
        wrapper.__module__ = func.__module__
        for attr in INTERNAL_ATTR:
            if hasattr(func, attr):
                setattr(wrapper, attr, getattr(func, attr))
        if not hasattr(wrapper, ORIGINAL_PY_FUNC_ATTR):
            setattr(wrapper, ORIGINAL_PY_FUNC_ATTR, func)
        setattr(wrapper, BUILTIN_ATTR, True)
        if isinstance(eval_return_type, LLType):
            eval_return_type_func = lambda *_, **kwargs: eval_return_type
        else:
            eval_return_type_func = eval_return_type
        if eval_arg_type is not None:
            setattr(wrapper, EVAL_ARG_TYPE_ATTR, eval_arg_type)
        setattr(wrapper, CODEGEN_FUNC_ATTR, codegen_func)
        assert isinstance(eval_return_type_func, Callable)
        setattr(wrapper, EVAL_RETURN_TYPE_ATTR, eval_return_type_func)

        # Register intrin function to global registry if it's from intrin module
        if "intrin" in wrapper.__module__:
            _INTRIN_REGISTRY[wrapper.__name__] = wrapper

        return wrapper

    return _builtin


def const_func(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    for attr in INTERNAL_ATTR:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    if not hasattr(wrapper, ORIGINAL_PY_FUNC_ATTR):
        setattr(wrapper, ORIGINAL_PY_FUNC_ATTR, func)
    setattr(wrapper, CONST_FUNC_ATTR, True)
    return wrapper


def const_type_func(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    for attr in INTERNAL_ATTR:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    if not hasattr(wrapper, ORIGINAL_PY_FUNC_ATTR):
        setattr(wrapper, ORIGINAL_PY_FUNC_ATTR, func)
    setattr(wrapper, CONST_TYPE_FUNC_ATTR, True)
    return wrapper


def inline_func(func):

    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    wrapper.__module__ = func.__module__
    for attr in INTERNAL_ATTR:
        if hasattr(func, attr):
            setattr(wrapper, attr, getattr(func, attr))
    if not hasattr(wrapper, ORIGINAL_PY_FUNC_ATTR):
        setattr(wrapper, ORIGINAL_PY_FUNC_ATTR, func)
    setattr(wrapper, INLINE_FUNC_ATTR, True)
    return wrapper


def builtin_class(cls):
    setattr(cls, BUILTIN_ATTR, True)
    return cls


@dataclass
class Builtin:
    body: str
    includes: List[str]
    return_val: str


# Export all constants for easy import
__all__ = [
    "INTERNAL_ATTR",
    "AST_SPECIAL_ATTRS",
    "BUILTIN_ATTR",
    "EVAL_RETURN_TYPE_ATTR",
    "EVAL_ARG_TYPE_ATTR",
    "CONST_FUNC_ATTR",
    "CONST_TYPE_FUNC_ATTR",
    "INLINE_FUNC_ATTR",
    "CODEGEN_FUNC_ATTR",
    "ORIGINAL_PY_FUNC_ATTR",
    "builtin",
    "const_func",
    "const_type_func",
    "inline_func",
    "builtin_class",
    "Builtin",
    "get_intrin_ctx",
]
