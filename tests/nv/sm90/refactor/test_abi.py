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
"""RF01 typed launch ABI and compilation artifact regressions.

These are deterministic CPU tests: they inspect exact ctypes argument buffer
sizes and bytes (the layout ``cuLaunchKernel`` receives), annotation
resolution, retention and immutable artifact/specification contracts.
"""

import ctypes
import gc
import struct as _struct

import numpy as np
import pytest

import little_kernel.language as ll
from little_kernel.core.artifact import CompilationArtifact
from little_kernel.runtime.abi import POINTER, SCALAR, STRUCTURE, KernelParameter, KernelSignature, PackedArguments, bfloat16_bits, ctypes_type_for_scalar, half_bits, pack_arguments, register_struct_layout, signature_from_annotations


class _Point(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_int32),
        ("y", ctypes.c_int32),
        ("tag", ctypes.c_uint8),
    ]


def _pack(signature, values):
    packed = pack_arguments(signature, values)
    return packed, packed.argument_bytes()


# ---------------------------------------------------------------------------
# Scalar sizes/sign are driven by the signature, not by value magnitude
# ---------------------------------------------------------------------------


def test_small_int64_values_use_eight_bytes():
    sig = KernelSignature([
        KernelParameter.scalar("neg", ll.int64),
        KernelParameter.scalar("pos", ll.int64),
    ])
    packed, chunks = _pack(sig, [-1, 1])

    assert sig.sizes == (8, 8)
    assert chunks[0] == b"\xff" * 8
    assert chunks[1] == b"\x01" + b"\x00" * 7
    # Every Driver slot points at the start of its retained storage.
    for ptr, obj in zip(packed.pointers, packed.storage):
        assert ptr == ctypes.addressof(obj)


def test_uint64_high_bits_preserved_and_int64_overflow_rejected():
    value = (1 << 64) - 1
    sig = KernelSignature([KernelParameter.scalar("big", ll.uint64)])
    _, chunks = _pack(sig, [value])
    assert chunks[0] == b"\xff" * 8

    # A numpy uint64 scalar carries the same raw bits.
    _, chunks = _pack(sig, [np.uint64(1 << 63)])
    assert chunks[0] == b"\x00" * 7 + b"\x80"

    signed_sig = KernelSignature([KernelParameter.scalar("v", ll.int64)])
    with pytest.raises(OverflowError):
        pack_arguments(signed_sig, [1 << 63])


@pytest.mark.parametrize(
    "lltype,expected_ctype",
    [
        (ll.int8, ctypes.c_int8),
        (ll.uint16, ctypes.c_uint16),
        (ll.int32, ctypes.c_int32),
        (ll.int64, ctypes.c_int64),
        (ll.uint32, ctypes.c_uint32),
        (ll.uint64, ctypes.c_uint64),
        (ll.float32, ctypes.c_float),
        (ll.float64, ctypes.c_double),
        (ll.bool_, ctypes.c_bool),
    ],
)
def test_scalar_type_to_ctype(lltype, expected_ctype):
    assert ctypes_type_for_scalar(lltype) is expected_ctype
    assert KernelParameter.scalar("p", lltype).c_type is expected_ctype


def test_float64_is_not_demoted_to_float32():
    val = 1.0000000001
    sig64 = KernelSignature([KernelParameter.scalar("x", ll.float64)])
    _, (raw64, ) = _pack(sig64, [val])
    assert raw64 == _struct.pack("<d", val)
    assert raw64 != _struct.pack("<f", val)

    sig32 = KernelSignature([KernelParameter.scalar("x", ll.float32)])
    _, (raw32, ) = _pack(sig32, [val])
    assert raw32 == _struct.pack("<f", val)
    assert len(raw64) == 8 and len(raw32) == 4

    with pytest.raises(TypeError):
        pack_arguments(sig64, [1])  # int never implicitly becomes float


def test_bool_is_one_distinct_byte():
    sig = KernelSignature([
        KernelParameter.scalar("t", ll.bool_),
        KernelParameter.scalar("f", ll.bool_),
    ])
    _, chunks = _pack(sig, [True, False])
    assert chunks == (b"\x01", b"\x00")

    with pytest.raises(TypeError):
        pack_arguments(KernelSignature([KernelParameter.scalar("b", ll.bool_)]), [1])


def test_explicit_ctypes_scalars_must_match_declared_storage():
    sig = KernelSignature([KernelParameter.scalar("x", ctypes.c_double)])
    _, (raw, ) = _pack(sig, [ctypes.c_double(2.5)])
    assert raw == _struct.pack("<d", 2.5)

    # Explicit c_float can never be passed where c_double is declared.
    with pytest.raises(TypeError):
        pack_arguments(sig, [ctypes.c_float(2.5)])


def test_sixteen_bit_float_types_have_real_abi():
    # __half and __nv_bfloat16 are real 16-bit by-value CUDA scalars.
    half_t = ctypes_type_for_scalar(ll.float16)
    bf16_t = ctypes_type_for_scalar(ll.bfloat16)
    assert ctypes.sizeof(half_t) == 2 and ctypes.sizeof(bf16_t) == 2

    sig = KernelSignature([
        KernelParameter.scalar("h", ll.float16),
        KernelParameter.scalar("b", ll.bfloat16),
    ])
    packed, chunks = _pack(sig, [1.0, 1.0])
    assert chunks == (b"\x00\x3c", b"\x80\x3f")  # half(1)=0x3c00, bf16(1)=0x3f80

    # Explicit raw-bit storage is accepted verbatim.
    packed, chunks = _pack(sig, [half_t(0x0001), bf16_t(0x8000)])
    assert chunks == (b"\x01\x00", b"\x00\x80")

    # Device-only widths without a host scalar ABI stay explicit errors.
    with pytest.raises(ValueError):
        ctypes_type_for_scalar("int128")
    with pytest.raises(ValueError):
        ctypes_type_for_scalar(ll.fp8_e4m3)


# ---------------------------------------------------------------------------
# Pointers
# ---------------------------------------------------------------------------


def test_pointer_arguments_are_eight_byte_addresses():
    sig = KernelSignature([
        KernelParameter.pointer("a", ll.float32, const=True),
        KernelParameter.pointer("b", ll.int64),
    ])
    assert [p.kind for p in sig] == [POINTER, POINTER]
    assert sig.sizes == (8, 8)

    address = (1 << 64) - 4096
    packed, chunks = _pack(sig, [address, ctypes.c_void_p(0xDEADBEEF)])
    assert chunks[0] == _struct.pack("<Q", address)
    assert chunks[1] == _struct.pack("<Q", 0xDEADBEEF)
    assert packed.storage[0].value == address

    with pytest.raises(OverflowError):
        pack_arguments(sig, [1 << 64, 0])
    with pytest.raises(TypeError):
        pack_arguments(sig, [True, 0])


def test_pointer_to_ctypes_array():
    arr = (ctypes.c_int32 * 4)(1, 2, 3, 4)
    sig = KernelSignature([KernelParameter.pointer("p", ll.int32)])
    packed = pack_arguments(sig, [arr])
    assert packed.argument_bytes()[0] == _struct.pack("<Q", ctypes.addressof(arr))


# ---------------------------------------------------------------------------
# Structures and strong retention
# ---------------------------------------------------------------------------


def test_structure_argument_bytes_and_pointer_slots():
    sig = KernelSignature([
        KernelParameter.structure("pt", _Point),
        KernelParameter.scalar("n", ll.int32),
    ])
    point = _Point(x=-7, y=42, tag=255)
    packed = pack_arguments(sig, [point, 3])

    struct_raw, n_raw = packed.argument_bytes()
    assert struct_raw == bytes(_Point(x=-7, y=42, tag=255))
    assert n_raw == _struct.pack("<i", 3)
    # The Driver gets pointers to each retained object, in declared order.
    assert packed.pointers[0] == ctypes.addressof(packed.storage[0])
    assert packed.pointers[1] == ctypes.addressof(packed.storage[1])
    assert packed.kernel_params[0] == packed.pointers[0]


def test_packed_storage_survives_gc_and_caller_mutation():
    sig = KernelSignature([KernelParameter.structure("pt", _Point)])
    point = _Point(x=1, y=2, tag=3)
    packed = pack_arguments(sig, [point])
    expected = packed.argument_bytes()[0]

    # Packing takes ownership of a copy: later caller mutation is irrelevant.
    point.x = -999
    gc.collect()
    assert ctypes.string_at(packed.pointers[0], ctypes.sizeof(_Point)) == expected


def test_structure_type_mismatch_rejected():
    sig = KernelSignature([KernelParameter.structure("pt", _Point)])
    with pytest.raises(TypeError):
        pack_arguments(sig, [ctypes.c_int64(1)])
    with pytest.raises(TypeError):
        pack_arguments(sig, [_TwoByte()])


# ---------------------------------------------------------------------------
# Signature validation, ordering and keyword packing
# ---------------------------------------------------------------------------


def test_signature_retains_order_and_is_immutable():
    p_a = KernelParameter.scalar("a", ll.int32)
    p_b = KernelParameter.pointer("b", ll.int32)
    sig = KernelSignature(parameters=[p_a, p_b])

    assert sig.names == ("a", "b")
    assert len(sig) == 2 and sig[1] is p_b
    with pytest.raises((AttributeError, Exception)):
        sig.parameters = ()  # type: ignore[misc]
    with pytest.raises(Exception):
        p_a.name = "other"  # type: ignore[misc]

    with pytest.raises(ValueError):
        KernelSignature([p_a, KernelParameter.scalar("a", ll.int64)])


def test_wrong_argument_count_and_unknown_keyword_rejected():
    sig = KernelSignature([KernelParameter.scalar("a", ll.int32)])
    with pytest.raises(TypeError):
        pack_arguments(sig, [1, 2])
    with pytest.raises(TypeError):
        pack_arguments(sig, {"b": 1})
    packed = pack_arguments(sig, {"a": 5})
    assert packed.argument_bytes()[0] == _struct.pack("<i", 5)


# ---------------------------------------------------------------------------
# signature_from_annotations
# ---------------------------------------------------------------------------


def test_signature_from_object_annotations():

    def kernel(
        x: ll.Tensor[ll.float32],
        n: ll.int64,
        ratio: ll.float64,
        flag: ll.bool_,
        raw: ctypes.c_uint8,
    ) -> ll.void:
        pass

    sig = signature_from_annotations(kernel)
    assert sig.names == ("x", "n", "ratio", "flag", "raw")
    x, n, ratio, flag, raw = sig
    assert x.kind == POINTER and x.element_type is ll.float32
    assert n.c_type is ctypes.c_int64
    assert ratio.c_type is ctypes.c_double
    assert flag.c_type is ctypes.c_bool
    assert raw.c_type is ctypes.c_uint8


def test_signature_from_string_and_modifier_annotations():

    def kernel(
        a: "ll.Tensor[ll.int64]",
        b: "ll.ptr[ll.uint32]",
        c: "ll.const[ll.int32]",
    ) -> ll.void:
        pass

    sig = signature_from_annotations(kernel)
    a, b, c = sig
    assert a.kind == POINTER and a.element_type is ll.int64
    assert b.kind == POINTER and b.element_type is ll.uint32
    assert c.kind == SCALAR and c.const is True and c.c_type is ctypes.c_int32


def test_constexpr_parameters_are_removed_by_name_and_annotation():

    def kernel(
        x: ll.Tensor[ll.int32],
        BLOCK: ll.constexpr[ll.int32],
        n: ll.int32,
    ) -> ll.void:
        pass

    sig = signature_from_annotations(kernel, constexpr_names=("n", ))
    # BLOCK removed via constexpr annotation; n removed via explicit name list.
    assert sig.names == ("x", )

    def untyped(x, n: ll.int32):
        pass

    with pytest.raises(TypeError):
        signature_from_annotations(untyped)


def test_signature_context_resolves_string_annotations():

    def kernel(pt: "Point", n: "ll.int32") -> ll.void:  # noqa: F821
        pass

    sig = signature_from_annotations(kernel, context={"Point": _Point})
    pt, n = sig
    assert pt.kind == STRUCTURE and pt.c_type is _Point
    assert n.c_type is ctypes.c_int32


def test_pointer_and_const_annotations_round_trip_pack():

    def kernel(out: ll.Tensor[ll.uint8], scale: ll.float64, count: ll.int32):
        pass

    sig = signature_from_annotations(kernel)
    packed = pack_arguments(sig, [0x100000000, 3.25, np.int32(-12)])
    chunks = packed.argument_bytes()
    assert chunks[0] == _struct.pack("<Q", 0x100000000)
    assert chunks[1] == _struct.pack("<d", 3.25)
    assert chunks[2] == _struct.pack("<i", -12)


# ---------------------------------------------------------------------------
# CompilationArtifact
# ---------------------------------------------------------------------------


def test_compilation_artifact_immutable():
    sig = KernelSignature([KernelParameter.scalar("n", ll.int64)])
    artifact = CompilationArtifact(
        source="__global__ void k() {}",
        kernel_name="k",
        signature=sig,
        target="sm_90a",
        metadata={"options": ("-O3", ), "shared_mem_bytes": 0},
    )

    assert artifact.source.startswith("__global__")
    assert artifact.parameter_names == ("n", )
    assert artifact.has_typed_signature()
    with pytest.raises(Exception):
        artifact.kernel_name = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        CompilationArtifact(source="x", kernel_name="k", signature="not-a-signature")
    with pytest.raises(ValueError):
        CompilationArtifact(source="", kernel_name="k")


def test_untyped_artifact_is_explicit():
    artifact = CompilationArtifact(source="x", kernel_name="k")
    assert artifact.signature is None
    assert artifact.parameter_names == ()
    assert not artifact.has_typed_signature()


# ---------------------------------------------------------------------------
# Concrete frontend struct layouts (follow-up 1)
# ---------------------------------------------------------------------------


class _TwoByte(ctypes.Structure):
    _fields_ = [("v", ctypes.c_uint16)]


def test_grid_constant_tma_descriptor_has_128_byte_layout():
    from little_kernel.backends.nv.common.runtime.tma_descriptor import CUtensorMap

    def kernel(desc: "ll.grid_constant[ll.TmaDescriptor]") -> ll.void:
        pass

    sig = signature_from_annotations(kernel)
    (desc, ) = sig
    assert desc.kind == STRUCTURE
    assert desc.c_type is CUtensorMap
    assert desc.size == 128 and desc.grid_constant

    packed = pack_arguments(sig, [CUtensorMap()])
    assert len(packed.argument_bytes()[0]) == 128

    # A 2-byte unrelated struct can never satisfy the known TMA ABI.
    with pytest.raises(TypeError):
        pack_arguments(sig, [_TwoByte()])


def test_unknown_frontend_struct_requires_explicit_adapter():

    class _CustomWgmma(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 32)]

    # ll.Wgmma has no built-in host layout until an adapter is registered.
    def kernel(w: ll.Wgmma) -> ll.void:
        pass

    with pytest.raises(TypeError, match="register_struct_layout"):
        signature_from_annotations(kernel)

    register_struct_layout(ll.Wgmma, _CustomWgmma)
    sig = signature_from_annotations(kernel)
    (w, ) = sig
    assert w.c_type is _CustomWgmma and w.size == 32
    packed = pack_arguments(sig, [_CustomWgmma()])
    assert len(packed.argument_bytes()[0]) == 32
    with pytest.raises(TypeError):
        pack_arguments(sig, [_TwoByte()])


# ---------------------------------------------------------------------------
# Emittable vector layouts (follow-up 2)
# ---------------------------------------------------------------------------


def test_vector_arguments_use_matching_c_layout():

    def kernel(v2: ll.u32x2, v4: ll.u32x4) -> ll.void:
        pass

    sig = signature_from_annotations(kernel)
    v2_param, v4_param = sig
    assert v2_param.kind == STRUCTURE and v2_param.size == 8
    assert v4_param.kind == STRUCTURE and v4_param.size == 16

    V2 = v2_param.c_type
    V4 = v4_param.c_type
    packed = pack_arguments(sig, [V2(1, 2), V4(4, 5, 6, 7)])
    assert packed.argument_bytes()[0] == b"\x01\x00\x00\x00\x02\x00\x00\x00"
    assert packed.argument_bytes()[1] == (b"\x04\x00\x00\x00\x05\x00\x00\x00"
                                          b"\x06\x00\x00\x00\x07\x00\x00\x00")

    # A previously valid caller passing a same-size ABI-compatible ctypes
    # structure remains valid; a different-sized one is rejected.
    class CompatibleU32x2(ctypes.Structure):
        _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32)]

    compatible = pack_arguments(sig, [CompatibleU32x2(1, 2), V4(0, 0, 0, 0)])
    assert compatible.argument_bytes()[0] == b"\x01\x00\x00\x00\x02\x00\x00\x00"

    class WrongU32x3(ctypes.Structure):
        _fields_ = [("x", ctypes.c_uint32), ("y", ctypes.c_uint32), ("z", ctypes.c_uint32)]

    with pytest.raises(TypeError):
        pack_arguments(sig, [WrongU32x3(1, 2, 3), V4(0, 0, 0, 0)])


def test_half_and_bfloat16_raw_bit_helpers():
    import math
    import struct as _struct

    assert half_bits(1.0) == 0x3C00
    assert bfloat16_bits(1.0) == 0x3F80
    # Just below/above the BF16 midpoint round down/up respectively.
    down = _struct.unpack("<f", _struct.pack("<I", 0x3F807FFF))[0]
    up = _struct.unpack("<f", _struct.pack("<I", 0x3F808001))[0]
    assert bfloat16_bits(down) == 0x3F80
    assert bfloat16_bits(up) == 0x3F81


def test_bfloat16_preserves_nan_inf_and_signs():
    import struct as _struct

    def f32(payload):
        return _struct.unpack("<f", _struct.pack("<I", payload))[0]

    # P2 payloads that previously collapsed to -0/0.
    assert bfloat16_bits(f32(0x7FFFFFFF)) == 0x7FFF  # positive quiet NaN
    assert bfloat16_bits(f32(0xFFFFFFFF)) == 0xFFFF  # negative quiet NaN
    assert bfloat16_bits(f32(0xFF800001)) == 0xFFC0  # signed NaN keeps sign
    assert bfloat16_bits(float("inf")) == 0x7F80
    assert bfloat16_bits(float("-inf")) == 0xFF80
    assert bfloat16_bits(-0.0) == 0x8000  # genuine signed zero stays -0

    # A NaN whose payload sits entirely in the discarded low bits still
    # emerges as a quiet NaN, not as infinity.
    nan_bits = bfloat16_bits(f32(0x7F800001))
    assert (nan_bits & 0x7F80) == 0x7F80 and (nan_bits & 0x7F) != 0


def test_exact_struct_registration_does_not_alias_other_instances():
    # P1: two distinct Struct parameterizations must not collide by class name.
    small = ll.Struct[[("x", ll.int32)]]
    big = ll.Struct[[("y", ll.int64)]]

    class _FourByte(ctypes.Structure):
        _fields_ = [("x", ctypes.c_int32)]

    class _EightByte(ctypes.Structure):
        _fields_ = [("v", ctypes.c_int64)]

    register_struct_layout(small, _FourByte)

    def kernel(a: "small", b: "big"):
        pass

    sig = signature_from_annotations(kernel, context={"small": small, "big": big})
    a_param, b_param = sig
    # The exact registration applies only to A; B synthesizes its own 8 bytes.
    assert a_param.c_type is _FourByte and a_param.size == 4
    assert b_param.c_type is not _FourByte and b_param.size == 8

    pack_arguments(sig, [_FourByte(1), _EightByte(2)])
    # The emitted 8-byte B cannot accept the 4-byte A adapter.
    with pytest.raises(TypeError):
        pack_arguments(sig, [_FourByte(1), _FourByte(2)])


def test_distinct_vector_and_special_struct_layouts_do_not_collide():
    v2 = ll.VectorType(ll.uint32, 2)
    v4 = ll.VectorType(ll.uint32, 4)
    special_small = ll.SpecialStructType("small_layout")

    class _Eight(ctypes.Structure):
        _fields_ = [("v", ctypes.c_uint64)]

    register_struct_layout(v2, _Eight)
    register_struct_layout(special_small, _Eight)

    def kernel(a: "v2", b: "v4", c: "special_small"):
        pass

    sig = signature_from_annotations(
        kernel,
        context={"v2": v2, "v4": v4, "special_small": special_small},
    )
    a_param, b_param, c_param = sig
    assert a_param.c_type is _Eight and a_param.size == 8
    assert b_param.c_type is not _Eight and b_param.size == 16
    assert c_param.c_type is _Eight
    # Native uint4 top-level storage is 16-byte aligned.
    assert b_param.c_type._align_ == 16

    # A string-keyed registration IS a deliberate class-wide alias.
    class _OtherSpecial(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 64)]

    register_struct_layout("SpecialStructType", _OtherSpecial)
    other = ll.SpecialStructType("unregistered_name")

    def kernel2(d: "other"):
        pass

    sig2 = signature_from_annotations(kernel2, context={"other": other})
    assert sig2[0].c_type is _OtherSpecial and sig2[0].size == 64


def test_nested_vector_struct_field_requires_explicit_adapter():
    nested = ll.Struct[[("v", ll.u32x2)]]

    def kernel(s: "nested"):
        pass

    with pytest.raises(TypeError, match="register_struct_layout"):
        signature_from_annotations(kernel, context={"nested": nested})


# ---------------------------------------------------------------------------
# Deep-frozen, non-hashable artifact metadata (follow-up 3)
# ---------------------------------------------------------------------------


def test_artifact_metadata_is_deep_frozen_and_unhashable():
    nested = {"options": ["-O3", "-DMODE=1"], "dep": {"header": ["a.cuh"]}}
    artifact = CompilationArtifact(source="x", kernel_name="k", metadata=nested)

    # The caller's own object can never reach the artifact.
    nested["options"].append("-DMUTATED")
    nested["dep"]["header"].append("b.cuh")
    assert artifact.metadata["options"] == ("-O3", "-DMODE=1")
    assert artifact.metadata["dep"]["header"] == ("a.cuh", )

    with pytest.raises(TypeError):
        artifact.metadata["options"] = ()
    with pytest.raises(TypeError):
        hash(artifact)

    # Equality remains value based for two equal artifacts.
    other = CompilationArtifact(
        source="x",
        kernel_name="k",
        metadata={"dep": {"header": ["a.cuh"]}, "options": ["-O3", "-DMODE=1"]},
    )
    assert artifact == other

    with pytest.raises(TypeError):
        CompilationArtifact(source="x", kernel_name="k", metadata={"bad": object()})
