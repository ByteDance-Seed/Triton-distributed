################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################
"""
Libdevice (`tl.extra.libdevice`) function"
==============================
Triton can invoke a custom function from an external library.

"""

import re

import triton
import torch
import triton.language as tl
import triton_dist.language.extra.language_extra as language_extra
from triton_dist.language.extra.language_extra import tid
import os

SCOPES = ["workgroup", "agent", "system"]
LD_SEMANTICS = ["acquire", "monotonic"]
ST_SEMANTICS = ["release", "monotonic"]
ALL_SEMANTICS = ["acquire", "release", "acq_rel", "monotonic"]


@triton.jit(do_not_specialize=["rank"])
def test_load_kernel(x_ptr, out_ptr, semantic: tl.constexpr = "monotonic", scope: tl.constexpr = "agent"):
    if tid(0) == 0:
        x = language_extra.ld(x_ptr, semantic=semantic, scope=scope)
        tl.store(out_ptr, x)


@triton.jit
def test_store_kernel(x_ptr, val, semantic: tl.constexpr = "monotonic", scope: tl.constexpr = "agent"):
    if tid(0) == 0:
        language_extra.st(x_ptr, val, semantic=semantic, scope=scope)


@triton.jit(do_not_specialize=["val"])
def test_atomic_add_kernel(
    ptr,
    out_ptr,
    val,
    semantic: tl.constexpr = "monotonic",
    scope: tl.constexpr = "agent",
):
    if tid(0) == 0:
        x = language_extra.atomic_add(ptr, val, semantic=semantic, scope=scope)
        tl.store(out_ptr, x)


@triton.jit(do_not_specialize=["val", "target_val"])
def test_atomic_cas_kernel(
    x_ptr,
    old_val_ptr,
    val,
    target_val,
    semantic: tl.constexpr = "monotonic",
    scope: tl.constexpr = "agent",
):
    if tid(0) == 0:
        x = language_extra.atomic_cas(x_ptr, val, target_val, semantic=semantic, scope=scope)
        # tl.device_print("x", x)
        tl.store(old_val_ptr, x)


def test_load(dtype: torch.dtype = torch.int32, semantic="monotonic", scope="agent"):
    x = torch.ones((1, ), dtype=dtype, device="cuda")
    y = torch.zeros((1, ), dtype=dtype, device="cuda")
    y.fill_(0)
    compiled = test_load_kernel[(1, )](x, y, scope=scope, semantic=semantic)
    torch.cuda.synchronize()
    # %6 = load atomic i32, ptr addrspace(1) %0 syncscope("workgroup-one-as") monotonic, align 4, !dbg !11
    if scope == "system":
        pattern = re.compile(rf"load atomic .*? {semantic}.*$", re.MULTILINE)
    else:
        pattern = re.compile(
            rf'load atomic .*?syncscope\("{scope}-one-as"\).*? {semantic}.*$',
            re.MULTILINE,
        )

    res = re.search(pattern, compiled.asm["llir"])
    print(y)
    assert res
    assert y == 1
    print(f"✅ test load scope={scope} semantic={semantic} dtype={dtype} passed")


def test_store(dtype: torch.dtype = torch.int32, semantic="monotonic", scope="agent"):
    x = torch.ones((1, ), dtype=dtype, device="cuda")
    x.zero_()
    compiled = test_store_kernel[(1, )](x, 1, scope=scope, semantic=semantic)
    torch.cuda.synchronize()

    # store atomic i32 1, ptr addrspace(1) %0 monotonic, align 4, !dbg !11
    if scope == "system":
        pattern = re.compile(rf"store atomic .*? {semantic}.*$", re.MULTILINE)
    else:
        pattern = re.compile(
            rf'store atomic .*?syncscope\("{scope}-one-as"\).*? {semantic}.*$',
            re.MULTILINE,
        )

    res = re.search(pattern, compiled.asm["llir"])
    assert res
    assert int(x) == 1
    print(f"✅ test store scope={scope} semantic={semantic} dtype={dtype} passed")


def test_atomic_add(dtype: torch.dtype = torch.int32, semantic="monotonic", scope="agent"):
    x = torch.ones((1, ), dtype=dtype, device="cuda")
    y = torch.ones((1, ), dtype=dtype, device="cuda")
    x.zero_()
    y.fill_(100)
    compiled = test_atomic_add_kernel[(1, )](x, y, 1, scope=scope, semantic=semantic)
    torch.cuda.synchronize()

    # %8 = atomicrmw add ptr addrspace(1) %0, i32 %2 syncscope("workgroup-one-as") release, align 4, !dbg !11
    if scope == "system":
        pattern = re.compile(rf"atomicrmw add .*? {semantic}.*$", re.MULTILINE)
    else:
        pattern = re.compile(
            rf'atomicrmw add .*?syncscope\("{scope}-one-as"\).*? {semantic}.*$',
            re.MULTILINE,
        )

    res = re.search(pattern, compiled.asm["llir"])
    assert res
    assert int(x) == 1
    assert int(y) == 0
    print(f"✅ test atomic_add scope={scope} semantic={semantic} dtype={dtype} passed")


def test_atomic_cas(dtype: torch.dtype = torch.int32, semantic="monotonic", scope="agent"):
    x = torch.ones((1, ), dtype=dtype, device="cuda")
    y = torch.ones((1, ), dtype=dtype, device="cuda")
    x.zero_()
    y.fill_(100)  # old val
    # if x == 0, then set to 1: compare success
    compiled = test_atomic_cas_kernel[(1, )](x, y, 1, 2, scope=scope, semantic=semantic)
    torch.cuda.synchronize()
    assert int(x) == 0
    assert int(y) == 0

    y.fill_(100)  # old val
    # compare failed
    compiled = test_atomic_cas_kernel[(1, )](x, y, 0, 3, scope=scope, semantic=semantic)
    torch.cuda.synchronize()
    assert int(x) == 3
    assert int(y) == 0

    # %9 = cmpxchg ptr addrspace(1) %0, i32 %2, i32 %3 syncscope("workgroup-one-as") release monotonic, align 4, !dbg !11
    if scope == "system":
        pattern = re.compile(rf"cmpxchg .*? {semantic}.*$", re.MULTILINE)
    else:
        pattern = re.compile(rf'cmpxchg .*?syncscope\("{scope}-one-as"\).*? {semantic}.*$', re.MULTILINE)

    res = re.search(pattern, compiled.asm["llir"])
    assert res
    print(f"✅ test atomic_cas scope={scope} semantic={semantic} dtype={dtype} passed")


os.environ["TRITON_DEBUG"] = "0"
DEVICE = triton.runtime.driver.active.get_active_torch_device()

for dtype in [torch.uint32, torch.uint64, torch.int32, torch.int64]:
    for scope in SCOPES:
        for semantic in LD_SEMANTICS:
            test_load(dtype, semantic=semantic, scope=scope)

for dtype in [torch.uint32, torch.uint64, torch.int32, torch.int64]:
    for scope in SCOPES:
        for semantic in ST_SEMANTICS:
            test_store(dtype, semantic=semantic, scope=scope)

for dtype in [torch.int32, torch.int64]:
    for scope in SCOPES:
        for semantic in ALL_SEMANTICS:
            test_atomic_add(dtype, semantic=semantic, scope=scope)
            test_atomic_cas(dtype, semantic=semantic, scope=scope)
