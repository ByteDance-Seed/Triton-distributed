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

# Opt-in marker: when this attribute is truthy on a builtin wrapper, the
# emitter additionally passes the element LLType of each positional argument
# as the `_arg_dtypes` keyword to its codegen function.  This lets a single
# canonical op (e.g. fma) lower to a dtype-spelled target intrinsic
# (fmaf/fma/...) instead of one hardcoded spelling, without task-name or
# string matching.  Dtype validation itself belongs in eval_return_type.
CODEGEN_NEEDS_TYPES_ATTR = "__codegen_needs_types__"

# Opt-in marker: a builtin whose return value is a Tensor[element] array may
# be written directly in an assignment/annotation whose target supplies the
# buffer name and whose scalar annotation declares the array's *element*
# type: ``v: ll.float32 = ll.alloc_local_memory(ll.float32, N)`` declares an
# N-element ``float32 v[N]``.  Without this marker a scalar annotation
# mismatching the call's Tensor[...] return type stays a type error.  Type
# inference normalizes the direct spelling into the builtin's internal
# (var_name, dtype, ...) form by inserting the target name as the first
# argument, so the builtin's own eval_arg_type/codegen keep exactly one
# contract.
ELEMENT_ANNOTATION_ALLOC_ATTR = "__element_annotation_alloc__"
ORIGINAL_PY_FUNC_ATTR = "__original_py_func__"
MEMORY_EFFECTS_ATTR = "__memory_effects__"
CTA_GROUP_ATTR = "__cta_group__"

# Global registry for intrin functions
# Maps function name -> function object
_INTRIN_REGISTRY: Dict[str, Callable] = {}

# Explicit metadata keyed by object identity, never by an operation's spelling.
_BY_VALUE_ARGUMENTS = []


def register_by_value_arguments(function, parameters):
    """Declare arguments consumed by value during a builtin's evaluation.

    Re-registration of the same function with an identical parameter set is a
    no-op; re-registering it with a *different* parameter set indicates stale
    metadata and raises ValueError instead of silently keeping first wins.
    """
    if not getattr(function, BUILTIN_ATTR, False):
        raise TypeError("By-value argument metadata requires a builtin")
    names = frozenset(parameters)
    if not all(isinstance(name, str) and name for name in names):
        raise TypeError("By-value argument names must be nonempty strings")
    for candidate, existing_names in _BY_VALUE_ARGUMENTS:
        if candidate is function:
            if existing_names != names:
                raise ValueError("By-value argument re-registration disagrees: "
                                 f"already registered as {sorted(existing_names)!r}, "
                                 f"new request is {sorted(names)!r}")
            return
    _BY_VALUE_ARGUMENTS.append((function, names))


def by_value_argument_primitives():
    return tuple(_BY_VALUE_ARGUMENTS)


def register_element_annotation_alloc(function):
    """Mark a declaration-alloc builtin for the element-annotation spelling.

    See ELEMENT_ANNOTATION_ALLOC_ATTR.  The builtin keeps its internal
    (var_name, dtype, ...) signature; type inference inserts the assignment
    target's name as the first argument of a direct spelling before
    validating or codegen-ing it.
    """
    if not getattr(function, BUILTIN_ATTR, False):
        raise TypeError("Element-annotation alloc metadata requires a builtin")
    setattr(function, ELEMENT_ANNOTATION_ALLOC_ATTR, True)


def get_intrin_ctx() -> Dict[str, Callable]:
    """
    Get a context dictionary containing all registered intrin functions.
    
    Returns:
        A dictionary mapping intrin function names to function objects.
    """
    return dict(_INTRIN_REGISTRY)


# Unified list of internal attributes copied verbatim from a wrapped function
# to its wrapper (only when already present; nothing here is ever fabricated).
# BUILTIN_ATTR is in this list so an already-@builtin source keeps its marker
# through const_func/const_type_func/inline_func — it is *not* a promise that
# wrapping a plain function makes it a builtin (see the decorators below).
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
    CODEGEN_NEEDS_TYPES_ATTR,
    ORIGINAL_PY_FUNC_ATTR,
    MEMORY_EFFECTS_ATTR,
    CTA_GROUP_ATTR,
    ELEMENT_ANNOTATION_ALLOC_ATTR,
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
    # Inline-callsite provenance stamped by FunctionInliner and preserved
    # through downstream rewrites so diagnostics can name the originating
    # inlined copy.  See passes/utils/inline_provenance.py.
    "_lk_inline_provenance",
    # Add more AST-specific attributes here as needed
]


def builtin(eval_return_type: Union[LLType, Callable, type] = None, codegen_func: Callable = None,
            eval_arg_type: Optional[Callable] = None, effects=(), cta_group: int = 1, namespace: Optional[str] = None):
    """
    Decorator for defining builtin functions/operations.
    
    Can be used in two ways:
    
    1. Function decorator with explicit codegen:
        @builtin(eval_return_type=float32, codegen_func=my_codegen)
        def my_func(a, b):
            pass
    
    2. Class decorator for BuiltinOp subclasses:
        @builtin
        class MyOp(BuiltinOp):
            @staticmethod
            def return_type(a, b):
                return float32
            
            @staticmethod
            def codegen(a, b):
                return Builtin(body="", includes=[], return_val=f"a + b")
    """
    if namespace is not None and (not isinstance(namespace, str) or not namespace
                                  or not all(part.isidentifier() for part in namespace.split("."))):
        raise ValueError("builtin namespace must be a dotted identifier or None")

    # Check if @builtin is used directly on a class (no parentheses)
    if isinstance(eval_return_type, type) and issubclass(eval_return_type, BuiltinOp):
        # @builtin applied directly to a BuiltinOp subclass
        return _make_builtin_from_class(eval_return_type, namespace=namespace)

    # Check if @builtin() is used on a class (with parentheses but no args)
    if eval_return_type is None and codegen_func is None:

        def _class_or_func_decorator(target):
            if isinstance(target, type) and issubclass(target, BuiltinOp):
                return _make_builtin_from_class(target, namespace=namespace)
            else:
                raise ValueError("builtin() without arguments can only decorate BuiltinOp subclasses. "
                                 "For functions, use @builtin(eval_return_type=..., codegen_func=...)")

        return _class_or_func_decorator

    # Original function decorator logic
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

            def eval_return_type_func(*_, **kwargs):
                return eval_return_type

        else:
            eval_return_type_func = eval_return_type
        if eval_arg_type is not None:
            setattr(wrapper, EVAL_ARG_TYPE_ATTR, eval_arg_type)
        if effects:
            setattr(wrapper, MEMORY_EFFECTS_ATTR, tuple(effects))
        if cta_group != 1:
            setattr(wrapper, CTA_GROUP_ATTR, cta_group)
        setattr(wrapper, CODEGEN_FUNC_ATTR, codegen_func)
        assert isinstance(eval_return_type_func, Callable)
        setattr(wrapper, EVAL_RETURN_TYPE_ATTR, eval_return_type_func)

        # Register intrin function to global registry if it's from intrin module
        if namespace is not None or "intrin" in wrapper.__module__:
            _INTRIN_REGISTRY[f"{namespace}.{wrapper.__name__}" if namespace else wrapper.__name__] = wrapper

        return wrapper

    return _builtin


# ``const_func``/``const_type_func``/``inline_func`` wrap a Python helper that
# has no target-side codegen: ``constfold`` always folds a CONST/CONST_TYPE call
# to an ``ast.Constant`` before emission, and the ``inline`` pass rewrites an
# INLINE call to the callee body.  None of these wrappers therefore fabricates
# ``BUILTIN_ATTR``.  The shared ``INTERNAL_ATTR`` copy loop below only ever
# *propagates an existing* marker, so wrapping an already-@builtin callable
# (e.g. ``const_func(sizeof)``) keeps ``BUILTIN_ATTR=True`` while a plain helper
# stays a non-builtin.  Pipeline stages test ``getattr(f, BUILTIN_ATTR, False)``
# to decide between builtin-codegen and call handling (see
# ``call_codegen._handle_builtin_call``); a codegen-less wrapper must not be
# flagged as a builtin or it would be routed there and trip the
# ``CODEGEN_FUNC_ATTR`` assert.  Keeping the marker absent on plain helpers is
# the contract; the helpers below intentionally never set it.
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


class BuiltinOp:
    """
    Base class for defining complex builtin operations via class inheritance.
    
    Users subclass this and implement:
    - return_type(*args, **kwargs) -> LLType: Return the result type
    - codegen(*args, **kwargs) -> Builtin: Generate the code
    
    Example:
        @builtin
        class MyComplexOp(BuiltinOp):
            @staticmethod
            def return_type(a, b):
                return float32
            
            @staticmethod
            def codegen(a, b):
                body = '''
        __device__ __forceinline__ float my_func(float a, float b) {
            return a * b + 1.0f;
        }
        '''
                return Builtin(body=body, includes=[], return_val=f"my_func({a}, {b})")
        
        # Usage in kernel:
        result = MyComplexOp(x, y)
    """

    @staticmethod
    def return_type(*args, **kwargs) -> LLType:
        """Override to return the result type based on arguments."""
        raise NotImplementedError("Subclass must implement return_type()")

    @staticmethod
    def codegen(*args, **kwargs) -> Builtin:
        """Override to generate code. Returns a Builtin dataclass."""
        raise NotImplementedError("Subclass must implement codegen()")


def _make_builtin_from_class(cls, namespace=None):
    """
    Convert a BuiltinOp subclass into a callable builtin function.
    
    This creates a wrapper function that:
    1. Has the class name as its __name__
    2. Has BUILTIN_ATTR set to True
    3. Has CODEGEN_FUNC_ATTR pointing to cls.codegen
    4. Has EVAL_RETURN_TYPE_ATTR pointing to cls.return_type
    5. If cls has py_impl staticmethod, the wrapper calls it when invoked
    
    Usage with py_impl:
        @builtin
        class sizeof(BuiltinOp):
            @staticmethod
            def py_impl(dtype):
                # Python implementation for const_fold
                return dtype.bits // 8
            
            @staticmethod
            def return_type(dtype):
                return int32
            
            @staticmethod
            def codegen(dtype):
                return Builtin(body="", includes=[], return_val=f"sizeof({dtype})")
    """
    # Check if class has py_impl for Python-callable behavior
    has_py_impl = hasattr(cls, 'py_impl') and callable(getattr(cls, 'py_impl'))

    if has_py_impl:

        def wrapper(*args, **kwargs):
            return cls.py_impl(*args, **kwargs)
    else:

        def wrapper(*args, **kwargs):
            # This function is never actually called during compilation
            # It's just a placeholder that carries the builtin attributes
            raise RuntimeError(f"{cls.__name__} should not be called directly during compilation")

    wrapper.__name__ = cls.__name__
    wrapper.__doc__ = cls.__doc__
    wrapper.__module__ = cls.__module__

    setattr(wrapper, BUILTIN_ATTR, True)
    setattr(wrapper, CODEGEN_FUNC_ATTR, cls.codegen)
    setattr(wrapper, EVAL_RETURN_TYPE_ATTR, cls.return_type)
    setattr(wrapper, ORIGINAL_PY_FUNC_ATTR, cls)
    # A BuiltinOp class may opt into typed codegen exactly like function
    # builtins: its codegen() then receives the per-argument LLType list via
    # the ``_arg_dtypes`` keyword (see call_codegen).  Declare the opt-in by
    # setting ``__codegen_needs_types__ = True`` on the class.
    if getattr(cls, CODEGEN_NEEDS_TYPES_ATTR, False):
        setattr(wrapper, CODEGEN_NEEDS_TYPES_ATTR, True)
    if hasattr(cls, "effects"):
        setattr(wrapper, MEMORY_EFFECTS_ATTR, tuple(getattr(cls, "effects")))

    # Register to intrin registry if from intrin module
    if namespace is not None or "intrin" in cls.__module__:
        _INTRIN_REGISTRY[f"{namespace}.{cls.__name__}" if namespace else cls.__name__] = wrapper

    return wrapper


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
    "CODEGEN_NEEDS_TYPES_ATTR",
    "ORIGINAL_PY_FUNC_ATTR",
    "MEMORY_EFFECTS_ATTR",
    "builtin",
    "const_func",
    "const_type_func",
    "inline_func",
    "builtin_class",
    "Builtin",
    "BuiltinOp",
    "get_intrin_ctx",
]
