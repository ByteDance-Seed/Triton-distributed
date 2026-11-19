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
"""RF08 regressions: immutable interned LLType values.

Covers:
* attempted field mutation fails for every type family;
* equality/hash stability and dict/set behavior after re-construction;
* tuple/list/iterator/generator input equivalence for composite types;
* the one-element iterator construction no longer consumes the input twice
  and cannot corrupt an existing interned TupleType;
* invalid construction never inserts/pollutes the intern pool;
* cached instances are reused and never reinitialized.
"""

import copy

import ast

import pytest

from little_kernel.core.type_system import LLType, AnnotateType, IntType, FloatType, VoidType, StringType, VectorType, Pointer, Const, GridConstant, Template, Constexpr, TensorType, StructType, TupleType, SpecialStructType, int32, uint32, int64, float32, bool_, binary, void, str_, u32x2, u32x4, Tensor, Struct, Tuple, ptr, const, const_annotation, grid_constant, template, constexpr
from little_kernel.backends.nv.common.core.types import TmaDescriptorType, TmaDescriptor
from little_kernel.backends.nv.sm90.core.types import WgmmaType, Wgmma

# -----------------------------------------------------------------------------
# Copy / deepcopy share the immutable value (baseline compatibility)
# -----------------------------------------------------------------------------
COPY_CASES = [
    void,
    str_,
    TmaDescriptor,
    Wgmma,
    Tensor,
    Struct,
    Tuple,
    const_annotation,
    LLType(),
    int32,
    float32,
    bool_,
    binary,
    ptr[int32],
    const[int32],
    grid_constant[TmaDescriptor],
    template[int32],
    constexpr[int32],
    Tensor[float32],
    Struct[(("a", int32), )],
    Tuple[int32, float32],
    VectorType(uint32, 4),
    SpecialStructType("Scheduler"),
    AnnotateType(int32),
]


@pytest.mark.parametrize("obj", COPY_CASES)
def test_copy_returns_same_immutable_instance(obj):
    assert copy.copy(obj) is obj
    assert copy.deepcopy(obj) is obj


def test_deepcopy_registers_memo_and_preserves_identity():
    obj = Tuple[int32, float32]
    memo = {}
    result = copy.deepcopy(obj, memo)
    assert result is obj
    assert memo[id(obj)] is obj


def test_ast_deepcopy_preserves_type_constant_identity():
    # AST constants can carry an LLType value; deepcopy must not try to
    # restore a frozen _type_key on a fresh, empty object.
    node = ast.Constant(value=int32)
    clone = copy.deepcopy(node)
    assert clone.value is int32

    container = ast.Tuple(elts=[ast.Constant(value=void), ast.Constant(value=Tuple[int32])])
    cloned = copy.deepcopy(container)
    assert cloned.elts[0].value is void
    assert cloned.elts[1].value is Tuple[int32]


# -----------------------------------------------------------------------------
# AnnotateType public constructor / external subclasses
# -----------------------------------------------------------------------------
def test_annotate_type_direct_constructor():
    wrapped = AnnotateType(int32)
    assert isinstance(wrapped, LLType)
    assert wrapped.inner_type is int32
    assert wrapped == AnnotateType(int32)
    assert hash(wrapped) == hash(AnnotateType(int32))
    assert copy.deepcopy(wrapped) is wrapped
    with pytest.raises(AttributeError):
        wrapped.inner_type = float32


def test_external_wrapper_subclass_super_init():

    class MyAnnotation(AnnotateType):

        def __init__(self, inner_type):
            super().__init__(inner_type)

    obj = MyAnnotation(int64)
    assert isinstance(obj, AnnotateType)
    assert obj.inner_type is int64
    assert MyAnnotation(int64) == obj
    assert hash(MyAnnotation(int64)) == hash(obj)
    with pytest.raises(AttributeError):
        obj.inner_type = int32
    # distinct from the base AnnotateType wrapper and built-in Const
    assert obj != AnnotateType(int64)
    assert obj != Const(int64)


def test_builtin_wrapper_identities_unchanged():
    assert Pointer(int32) is ptr[int32]
    assert Const(int32) is const[int32]
    assert GridConstant(TmaDescriptor) is grid_constant[TmaDescriptor]
    assert Template(int32) is template[int32]
    assert Constexpr(int32) is constexpr[int32]
    assert ptr[int32].bits == 64


# -----------------------------------------------------------------------------
# Immutability
# -----------------------------------------------------------------------------
MUTATION_CASES = [
    (int32, "bits"),
    (int32, "signed"),
    (int32, "special"),
    (float32, "fmt"),
    (float32, "bits"),
    (TupleType((int32, float32)), "element_types"),
    (StructType((("a", int32), )), "type_tuple"),
    (TensorType(int32), "element_type"),
    (VectorType(int32, 2), "lanes"),
    (VectorType(int32, 2), "element_type"),
    (Pointer(int32), "inner_type"),
    (Pointer(int32), "bits"),
    (Const(int32), "inner_type"),
    (GridConstant(TmaDescriptor), "inner_type"),
    (Template(int32), "inner_type"),
    (Constexpr(int32), "inner_type"),
    (SpecialStructType("Scheduler"), "struct_name"),
]


@pytest.mark.parametrize("obj,attr", MUTATION_CASES)
def test_fields_are_read_only(obj, attr):
    with pytest.raises(AttributeError):
        setattr(obj, attr, "mutated")
    with pytest.raises(AttributeError):
        delattr(obj, attr)


def test_cannot_override_frozen_value_via_dict():
    # There is no instance __dict__, so the frozen value cannot be shadowed
    # behind the descriptor guard.
    assert not hasattr(int32, "__dict__")
    assert int32.bits == 32


def test_values_remain_unchanged_after_failed_mutation():
    t = TupleType((int32, float32))
    with pytest.raises(AttributeError):
        t.element_types = ()
    assert t.element_types == (int32, float32)
    assert len(t) == 2
    assert t[0] is int32


# -----------------------------------------------------------------------------
# Equality / hash stability and mappings
# -----------------------------------------------------------------------------
def test_scalar_singletons_identity_and_special_bool_binary():
    assert IntType(32) is int32
    assert IntType(32, True) is int32
    assert IntType(64) is int64
    assert FloatType("float32") is float32
    assert bool_ is not binary
    assert bool_ != binary
    assert bool_ == IntType(1, False, special="bool")
    assert binary == IntType(1, True, special="binary")
    assert VoidType() is void
    assert StringType() is str_
    assert TmaDescriptorType() is TmaDescriptor
    assert WgmmaType() is Wgmma


def test_hash_matches_equality_for_all_families():
    pairs = [
        (IntType(32), int32),
        (FloatType("float32"), float32),
        (Pointer(int32), ptr[int32]),
        (Const(int32), const[int32]),
        (TensorType(int32), Tensor[int32]),
        (TupleType((int32, float32)), Tuple[int32, float32]),
        (StructType((("a", int32), )), Struct[(("a", int32), )]),
        (VectorType(uint32, 4), u32x4),
        (SpecialStructType("S"), SpecialStructType("S")),
    ]
    for a, b in pairs:
        assert a == b
        assert hash(a) == hash(b)
        assert not (a != b)


def test_distinct_types_are_unequal():
    assert int32 != uint32
    assert int32 != float32
    assert Pointer(int32) != Pointer(int64)
    assert TupleType((int32, )) != TupleType((int64, ))
    assert StructType((("a", int32), )) != StructType((("b", int32), ))
    assert TensorType(int32) != TensorType(float32)
    assert SpecialStructType("A") != SpecialStructType("B")


def test_dict_and_set_lookups_remain_stable():
    t = TupleType((int32, float32))
    h_before = hash(t)
    mapping = {int32: "i", float32: "f", t: "pair", ptr[int32]: "p"}

    # Repeated construction returns the same interned value.
    assert TupleType((int32, float32)) is t
    assert mapping[TupleType(tuple(t.element_types))] == "pair"
    assert mapping[IntType(32)] == "i"

    # set semantics
    s = {t, TupleType(list(t.element_types)), IntType(32), int32}
    assert s == {t, int32}

    assert hash(t) == h_before


def test_non_lltype_comparison_does_not_error():
    assert (int32 == 32) is False
    assert (int32 == "int32") is False
    assert int32 != None  # noqa: E711


def test_const_annotation_compatibility_equality_and_hash():
    assert const == const_annotation
    assert hash(const) == hash(const_annotation)
    assert {const, const_annotation} == {const_annotation}


# -----------------------------------------------------------------------------
# Composite normalization exactly once
# -----------------------------------------------------------------------------
def test_tuple_list_iterator_generator_equivalence():
    expected = TupleType((int32, float32))
    variants = [
        TupleType((int32, float32)),
        TupleType([int32, float32]),
        TupleType(iter([int32, float32])),
        TupleType(t for t in (int32, float32)),
        Tuple[int32, float32],
    ]
    for v in variants:
        assert v is expected
        assert isinstance(v.element_types, tuple)
        assert v.element_types == (int32, float32)


def test_single_element_iterator_does_not_consume_input_twice():
    # Regression: TupleType(iter((int32,))) used to normalize in both
    # __new__ and __init__; the second tuple() on the exhausted iterator
    # produced () and overwrote the cached one-element instance.
    existing = TupleType((int32, ))
    assert len(existing) == 1

    rebuilt = TupleType(iter((int32, )))
    assert rebuilt is existing
    assert len(existing) == 1
    assert existing.element_types == (int32, )

    # The canonical empty tuple remains distinct and intact.
    empty = TupleType(())
    assert len(empty) == 0
    assert existing != empty
    assert len(existing) == 1


def test_exhausting_iterator_does_not_corrupt_other_instance():
    one = TupleType((float32, ))
    TupleType(iter([int32]))
    assert one.element_types == (float32, )
    assert TupleType((float32, )) is one


def test_composite_members_are_stored_as_tuples_and_copied_from_lists():
    members = [int32, float32]
    t = TupleType(members)
    assert isinstance(t.element_types, tuple)
    members.append(int64)
    assert t.element_types == (int32, float32)

    fields = [("a", int32)]
    s = StructType(fields)
    assert isinstance(s.type_tuple, tuple)
    fields.append(("b", float32))
    assert s.type_tuple == (("a", int32), )


def test_struct_list_tuple_equivalence():
    a = StructType((("a", int32), ("b", float32)))
    b = StructType([("a", int32), ("b", float32)])
    assert a is b


def test_helper_single_element_tuple():
    assert Tuple[int32] is TupleType((int32, ))


# -----------------------------------------------------------------------------
# Validation before intern insertion / invalid pool entries
# -----------------------------------------------------------------------------
def test_invalid_tuple_elements_raise_and_leave_pool_clean():
    before = dict(TupleType._instance)
    with pytest.raises(TypeError):
        TupleType((int32, "not a type"))
    with pytest.raises(TypeError):
        TupleType([7])
    with pytest.raises(TypeError):
        TupleType("int32")  # strings are not iterables-of-types
    assert TupleType._instance == before


def test_invalid_struct_fields_raise_and_leave_pool_clean():
    before = dict(StructType._instance)
    with pytest.raises(TypeError):
        StructType([("a", )])  # bad pair length
    with pytest.raises(TypeError):
        StructType([(3, int32)])  # bad name
    with pytest.raises(TypeError):
        StructType([("a", 7)])  # bad field type
    with pytest.raises(TypeError):
        StructType("ab")  # not a list/tuple
    assert StructType._instance == before


def test_invalid_scalar_construction_does_not_cache():
    before = dict(IntType._cache)
    with pytest.raises(ValueError):
        IntType(3)
    with pytest.raises(ValueError):
        IntType(1, special="nope")
    with pytest.raises(ValueError):
        IntType(8, special="bool")
    assert IntType._cache == before

    before_f = dict(FloatType._cache)
    with pytest.raises(ValueError):
        FloatType("float7")
    assert FloatType._cache == before_f


def test_invalid_wrapper_vector_tensor_raise():
    # Annotation wrappers historically accept any hashable marker payload
    # (e.g. plain classes), but an unhashable payload can never intern and
    # must fail without polluting the pool.
    with pytest.raises(TypeError):
        Pointer(["unhashable"])
    with pytest.raises(TypeError):
        Const({"a": 1})
    with pytest.raises(TypeError):
        TensorType(int)
    with pytest.raises(TypeError):
        VectorType("int32", 4)
    with pytest.raises(ValueError):
        VectorType(int32, 1)
    with pytest.raises(ValueError):
        VectorType(int32, 17)
    with pytest.raises(TypeError):
        SpecialStructType("")


def test_wrapper_accepts_arbitrary_hashable_marker():

    class Marker:
        pass

    wrapped = Template(Marker)
    assert Template(Marker) is wrapped
    assert wrapped.inner_type is Marker


# -----------------------------------------------------------------------------
# Cached reuse / no reinitialization
# -----------------------------------------------------------------------------
def test_cached_objects_are_reused_and_not_reinitialized():
    first = TupleType((int32, int64))
    second = TupleType((int32, int64))
    assert first is second

    ptr_a = Pointer(float32)
    ptr_b = Pointer(float32)
    assert ptr_a is ptr_b
    assert ptr_b.bits == 64

    vec = VectorType(uint32, 2)
    assert vec is u32x2


def test_subclass_specific_cache_does_not_alias():
    # GridConstant subclasses Const; their pools must not cross-resolve.
    c = Const(TmaDescriptor)
    g = GridConstant(TmaDescriptor)
    assert c is not g
    assert type(c) is Const
    assert type(g) is GridConstant
    assert g.is_grid_constant()
    assert c.is_const() and not c.is_grid_constant()


def test_decorator_style_annotation_helpers_round_trip():
    assert grid_constant[TmaDescriptor] is GridConstant(TmaDescriptor)
    assert template[int32] is Template(int32)
    assert constexpr[int32] is Constexpr(int32)
    assert ptr[uint32] is Pointer(uint32)
    assert Tensor[float32] is TensorType(float32)


def test_base_type_marker_stable():
    assert LLType() == LLType()
    assert hash(LLType()) == hash(LLType())
