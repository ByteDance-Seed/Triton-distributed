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
"""RF04 + RF01-integration regressions for Driver ownership, sync and launch ABI.

All tests are CPU/mock-only except the clearly marked real-Driver acceptance
test at the bottom, which skips when a CUDA device is unavailable.
"""

import ctypes
import threading

import numpy as np
import pytest

import little_kernel.backends.nv.common.runtime.cuda_runtime as cr
from little_kernel.runtime import lifecycle
from little_kernel.runtime.abi import KernelParameter, KernelSignature

# ---------------------------------------------------------------------------
# Fake libcuda
# ---------------------------------------------------------------------------


class FakeDriver:
    """Deterministic stand-in for libcuda with resource counters."""

    def __init__(self, current_handle=None, fail=None):
        self.calls = {}
        self.fail = dict(fail or {})
        self._next_ctx = 0xAA00
        self._next_module = 0xBB00
        self.created_contexts = []
        self.destroyed = []
        self.retained = []
        self.released = []
        self.unloaded = []
        self.set_current_args = []
        self._forced_current = current_handle
        self.launched = []  # recorded cuLaunchKernel argument pointers
        # Optional list of per-slot byte sizes; when set, the fake snapshots
        # each argument slot during the (synchronous) launch, mirroring the
        # Driver copying parameters before launch returns.
        self.snapshot_sizes = None

    @staticmethod
    def _v(value):
        """Extract an integer from an int or ctypes scalar/handle."""
        return int(value.value) if hasattr(value, "value") else int(value)

    def _count(self, name):
        self.calls[name] = self.calls.get(name, 0) + 1

    def _err(self, name):
        return self.fail.get(name, 0)

    # -- driver/device/context ------------------------------------------------

    def cuInit(self, flags):
        self._count("cuInit")
        return self._err("cuInit")

    def cuDeviceGet(self, dev_ptr, ordinal):
        self._count("cuDeviceGet")
        ctypes.cast(dev_ptr, ctypes.POINTER(ctypes.c_int))[0] = 7
        return self._err("cuDeviceGet")

    def cuCtxCreate_v2(self, ctx_ptr, flags, device):
        self._count("cuCtxCreate_v2")
        handle = self._next_ctx
        self._next_ctx += 0x10
        self.created_contexts.append(handle)
        if self._forced_current is None:
            self._forced_current = handle
        ctypes.cast(ctx_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return self._err("cuCtxCreate_v2")

    def cuCtxDestroy_v2(self, ctx):
        self._count("cuCtxDestroy_v2")
        err = self._err("cuCtxDestroy_v2")
        if err == 0:
            self.destroyed.append(self._v(ctx))
        return err

    def cuDevicePrimaryCtxRetain(self, ctx_ptr, device):
        self._count("cuDevicePrimaryCtxRetain")
        handle = self._next_ctx
        self._next_ctx += 0x10
        self.retained.append(handle)
        if self._forced_current is None:
            self._forced_current = handle
        ctypes.cast(ctx_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return self._err("cuDevicePrimaryCtxRetain")

    def cuDevicePrimaryCtxRelease_v2(self, device):
        self._count("cuDevicePrimaryCtxRelease_v2")
        self.released.append(self._v(device))
        return self._err("cuDevicePrimaryCtxRelease_v2")

    def cuCtxSetCurrent(self, ctx):
        self._count("cuCtxSetCurrent")
        self.set_current_args.append(self._v(ctx))
        return self._err("cuCtxSetCurrent")

    def cuCtxGetCurrent(self, ctx_ptr):
        self._count("cuCtxGetCurrent")
        ctypes.cast(ctx_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = (self._forced_current or 0)
        return self._err("cuCtxGetCurrent")

    def cuCtxSynchronize(self):
        self._count("cuCtxSynchronize")
        return self._err("cuCtxSynchronize")

    def cuStreamSynchronize(self, stream):
        self._count("cuStreamSynchronize")
        return self._err("cuStreamSynchronize")

    # -- module/function -------------------------------------------------------

    def cuModuleLoadData(self, mod_ptr, cubin):
        self._count("cuModuleLoadData")
        handle = self._next_module
        self._next_module += 0x10
        ctypes.cast(mod_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = handle
        return self._err("cuModuleLoadData")

    def cuModuleUnload(self, module):
        self._count("cuModuleUnload")
        err = self._err("cuModuleUnload")
        if err == 0:
            self.unloaded.append(self._v(module))
        return err

    def cuModuleGetFunction(self, func_ptr, module, name):
        self._count("cuModuleGetFunction")
        ctypes.cast(func_ptr, ctypes.POINTER(ctypes.c_void_p))[0] = 0xF00D
        return self._err("cuModuleGetFunction")

    def cuGetErrorString(self, err, str_ptr):
        ctypes.cast(str_ptr, ctypes.POINTER(ctypes.c_char_p))[0] = b"fake error"
        return 0

    # -- launch ----------------------------------------------------------------

    def cuLaunchKernel(self, func, gx, gy, gz, bx, by, bz, shared, stream, kparams, extra):
        self._count("cuLaunchKernel")
        snapshot = None
        if kparams and self.snapshot_sizes is not None:
            arr = ctypes.cast(kparams, ctypes.POINTER(ctypes.c_void_p))
            snapshot = [ctypes.string_at(arr[i], size) for i, size in enumerate(self.snapshot_sizes)]
        self.launched.append(
            dict(func=self._v(func), grid=(gx, gy, gz), block=(bx, by, bz), shared=self._v(shared),
                 stream=self._v(stream or 0), kparams=kparams, arg_bytes=snapshot))
        return self._err("cuLaunchKernel")


def _make_runtime(fake, **kw):
    """Construct a CUDARuntime bound to a FakeDriver without touching disk."""
    rt = object.__new__(cr.CUDARuntime)
    rt._libcuda = fake
    rt._ctx = None
    rt._device = None
    rt._managed_context = None
    rt._managed_modules = {}
    rt._modules_lock = threading.RLock()
    rt._closed = False
    rt._has_launch_kernel_ex = False
    rt._has_enumeration_api = False
    rt._primary_ctx_release_name = "cuDevicePrimaryCtxRelease_v2"
    rt._init_context(**kw)
    return rt


@pytest.fixture
def no_torch(monkeypatch):
    """Force the standalone (no-Torch) context path."""
    monkeypatch.setattr(cr, "TORCH_AVAILABLE", False)


@pytest.fixture
def standalone(no_torch):
    fake = FakeDriver()
    rt = _make_runtime(fake)
    yield fake, rt
    rt.close()


# ---------------------------------------------------------------------------
# RF04: modern context-create API and actual setup validation
# ---------------------------------------------------------------------------


def test_standalone_uses_modern_ctx_create_v2(standalone):
    fake, rt = standalone
    assert fake.calls.get("cuCtxCreate_v2") == 1
    # The legacy unversioned API must never be touched.
    assert not hasattr(rt._libcuda, "cuCtxCreate") or not fake.calls.get("cuCtxCreate")
    assert rt.managed_context.ownership is lifecycle.ContextOwnership.CREATED
    # Validation made the context current and read it back.
    assert fake.calls.get("cuCtxSetCurrent") >= 1
    assert fake.calls.get("cuCtxGetCurrent") == 1


def test_context_setup_mismatch_is_rejected_at_construction(no_torch):
    # cuCtxGetCurrent reports a *different* context than the one created:
    # this is exactly the H800 mixed-version failure shape.
    fake = FakeDriver(current_handle=0xDEADBEEF)
    with pytest.raises(RuntimeError, match="did not become current"):
        _make_runtime(fake)
    # The unusable context must be released rather than leaked.
    assert fake.destroyed == [fake.created_contexts[0]]


def test_retained_primary_setup_failure_is_released(monkeypatch):
    """Owned PRIMARY_RETAINED setup failure releases the matching retain."""
    monkeypatch.setattr(cr, "TORCH_AVAILABLE", True)
    fake_torch = type("T", (), {})()
    fake_torch.cuda = type(
        "C",
        (),
        {
            "is_available": staticmethod(lambda: True),
            "current_device": staticmethod(lambda: 5),
        },
    )()
    monkeypatch.setattr(cr, "torch", fake_torch, raising=False)

    fake = FakeDriver(current_handle=0xBADBAD)
    with pytest.raises(RuntimeError, match="did not become current"):
        _make_runtime(fake)
    # Retained once, then released once on setup failure; never destroyed.
    assert len(fake.retained) == 1
    assert fake.released == [5]
    assert fake.destroyed == []


def test_ctx_set_current_error_is_checked(no_torch):
    fake = FakeDriver(fail={"cuCtxSetCurrent": 201})
    with pytest.raises(RuntimeError, match="cuCtxSetCurrent"):
        _make_runtime(fake)


def test_borrowed_context_is_validated_but_never_released():
    fake = FakeDriver(current_handle=0x5151)
    rt = _make_runtime(fake, context=0x5151, device=3)
    try:
        assert rt.managed_context.borrowed
        assert rt.managed_context.ownership is lifecycle.ContextOwnership.BORROWED
        assert fake.destroyed == []
    finally:
        rt.close()
    # Close must not destroy or release a borrowed owner.
    assert fake.destroyed == []
    assert fake.released == []


def test_created_context_destroyed_exactly_once(standalone):
    fake, rt = standalone
    handle = rt.managed_context.handle
    assert rt.close() is None
    assert fake.destroyed == [handle]
    # Idempotent.
    rt.close()
    assert fake.destroyed == [handle]


def test_context_manager_closes_resources(no_torch):
    fake = FakeDriver()
    with _make_runtime(fake) as rt:
        handle = rt.managed_context.handle
    assert fake.destroyed == [handle]


def test_closed_context_cannot_be_used(standalone):
    fake, rt = standalone
    rt.close()
    with pytest.raises(lifecycle.ResourceClosedError):
        rt._ensure_context()


# ---------------------------------------------------------------------------
# RF04: real synchronization through the Driver
# ---------------------------------------------------------------------------


def test_synchronize_no_stream_uses_checked_ctx_synchronize(standalone):
    fake, rt = standalone
    before = fake.calls.get("cuCtxSynchronize", 0)
    rt.synchronize()
    assert fake.calls.get("cuCtxSynchronize") == before + 1
    assert fake.calls.get("cuStreamSynchronize", 0) == 0


def test_synchronize_explicit_stream_uses_stream_sync(standalone):
    fake, rt = standalone
    rt.synchronize(stream=ctypes.c_void_p(0x1234))
    assert fake.calls.get("cuStreamSynchronize") == 1
    assert fake.calls.get("cuCtxSynchronize", 0) == 0


def test_synchronize_failures_raise(standalone):
    fake, rt = standalone
    fake.fail["cuCtxSynchronize"] = 700
    with pytest.raises(RuntimeError, match="cuCtxSynchronize"):
        rt.synchronize()
    fake.fail.clear()
    fake.fail["cuStreamSynchronize"] = 701
    with pytest.raises(RuntimeError, match="cuStreamSynchronize"):
        rt.synchronize(stream=ctypes.c_void_p(0x9))


def test_ensure_context_set_current_failure_is_checked(standalone):
    fake, rt = standalone
    fake.fail["cuCtxSetCurrent"] = 215
    with pytest.raises(RuntimeError, match="cuCtxSetCurrent"):
        rt._ensure_context()


# ---------------------------------------------------------------------------
# RF04: module ownership and live-launcher protection
# ---------------------------------------------------------------------------


def test_module_unload_idempotent_and_balanced(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"\x00cubin")
    assert fake.calls.get("cuModuleLoadData") == 1
    assert rt.unload_module(mod) is True
    assert fake.unloaded == [int(mod.value)]
    # Unknown-again raw handle is not double-tracked; second managed close is
    # idempotent through the registry.
    managed = rt._managed_modules
    assert not managed  # popped on first unload


def test_module_cannot_unload_while_live_launcher_depends(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"cubin")
    rt.register_module_user(mod)
    with pytest.raises(RuntimeError, match="live launcher"):
        rt.unload_module(mod)
    assert fake.unloaded == []
    rt.release_module_user(mod)
    assert rt.unload_module(mod) is True
    assert len(fake.unloaded) == 1


def test_close_unloads_owned_modules_then_context(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"cubin")
    mod_handle = int(mod.value)
    ctx_handle = rt.managed_context.handle
    rt.close()
    assert fake.unloaded == [mod_handle]
    assert fake.destroyed == [ctx_handle]


def test_close_refusal_leaves_runtime_usable_and_context_alive(standalone):
    """RF04 review defect: live launchers must block close up front.

    A refused close must not destroy the owning context, unload the module,
    or drop ownership tracking; the runtime stays usable, and after the
    launcher detaches a single successful close releases everything once.
    """
    fake, rt = standalone
    mod = rt.load_module(b"cubin")
    mod_handle = int(mod.value)
    ctx_handle = rt.managed_context.handle
    rt.register_module_user(mod)

    with pytest.raises(RuntimeError, match="live launchers"):
        rt.close()

    # Nothing was released and tracking is intact.
    assert fake.unloaded == []
    assert fake.destroyed == []
    assert not rt._closed
    assert rt.managed_context is not None
    assert not rt.managed_context.closed
    assert mod_handle in rt._managed_modules
    assert rt._managed_modules[mod_handle].live_users == 1

    # Runtime/context are still usable (make-current still works).
    rt._ensure_context()
    rt.synchronize()
    assert fake.calls.get("cuCtxSynchronize") == 1

    # Detach the live launcher; retry close now releases everything once.
    rt.release_module_user(mod)
    rt.close()
    assert fake.unloaded == [mod_handle]
    assert fake.destroyed == [ctx_handle]
    # Idempotent: second close releases nothing extra.
    rt.close()
    assert fake.unloaded == [mod_handle]
    assert fake.destroyed == [ctx_handle]


def test_failed_module_unload_preserves_tracking_and_context(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"cubin")
    mod_handle = int(mod.value)
    ctx_handle = rt.managed_context.handle
    fake.fail["cuModuleUnload"] = 400

    with pytest.raises(RuntimeError, match="cuModuleUnload"):
        rt.close()

    # Context is not destroyed while module ownership remains, and tracking
    # is preserved for a retry.
    assert fake.destroyed == []
    assert not rt._closed
    assert mod_handle in rt._managed_modules
    assert rt.managed_context is not None

    fake.fail.clear()
    rt.close()
    assert fake.unloaded == [mod_handle]
    assert fake.destroyed == [ctx_handle]


def test_failed_context_destroy_is_retriable(standalone):
    fake, rt = standalone
    ctx_handle = rt.managed_context.handle
    fake.fail["cuCtxDestroy_v2"] = 401
    with pytest.raises(RuntimeError, match="cuCtxDestroy_v2"):
        rt.close()
    # Owned context is not discarded; module was already unloaded.
    assert not rt._closed
    assert rt.managed_context is not None
    assert not rt.managed_context.closed
    fake.fail.clear()
    rt.close()
    assert fake.destroyed == [ctx_handle]
    assert rt._closed


def test_managed_context_failed_release_is_retriable_and_borrowed_safe():
    # Owned release failure must not be mistaken for a successful close.
    released = []

    def release(managed):
        if not released:
            released.append("attempt")
            raise RuntimeError("driver busy")
        released.append(managed.handle)

    ctx = lifecycle.ManagedContext(
        0x900,
        0,
        lifecycle.ContextOwnership.CREATED,
        release=release,
    )
    assert not ctx.closed
    with pytest.raises(RuntimeError, match="driver busy"):
        ctx.close()
    assert not ctx.closed
    assert ctx.close() is True
    assert released == ["attempt", 0x900]

    # A borrowed context is marked closed locally but never released.
    borrowed_release = []
    bctx = lifecycle.ManagedContext(
        0x901,
        1,
        lifecycle.ContextOwnership.BORROWED,
    )
    assert bctx.close() is False
    assert bctx.closed
    assert borrowed_release == []


def test_setup_cleanup_failure_preserves_original_error():
    calls = []

    def validate(managed):
        calls.append("validate")
        raise RuntimeError("did not become current")

    def release(managed):
        calls.append("release")
        raise RuntimeError("cleanup failed")

    with pytest.raises(RuntimeError, match="did not become current") as exc:
        lifecycle.ManagedContext(
            0x902,
            0,
            lifecycle.ContextOwnership.CREATED,
            release=release,
            validate=validate,
        )
    # Owned release was attempted during setup cleanup...
    assert calls == ["validate", "release"]
    # ...but the original setup error is the raised one, cleanup chained.
    assert isinstance(exc.value.__cause__, RuntimeError)
    assert "cleanup failed" in str(exc.value.__cause__)

    # Borrowed setup failure must never invoke any release (and a borrowed
    # context may not even carry a release callable).
    borrowed_calls = []

    def bvalidate(managed):
        borrowed_calls.append("validate")
        raise RuntimeError("bad borrowed ctx")

    with pytest.raises(RuntimeError, match="bad borrowed ctx"):
        lifecycle.ManagedContext(
            0x903,
            0,
            lifecycle.ContextOwnership.BORROWED,
            validate=bvalidate,
        )
    assert borrowed_calls == ["validate"]


def test_borrowed_module_is_never_unloaded(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"cubin", borrowed=True)
    managed = rt._managed_module(mod)
    assert managed.borrowed
    assert rt.unload_module(mod) is False
    assert fake.unloaded == []


def test_managed_module_context_manager_and_lifecycle_unit():
    unloaded = []
    mod = lifecycle.ManagedModule(0x77, unload=lambda h: unloaded.append(h))
    with mod:
        mod.attach_user()
        with pytest.raises(RuntimeError, match="live"):
            mod.close()
        mod.detach_user()
    assert unloaded == [0x77]
    assert mod.closed
    assert mod.close() is False


# ---------------------------------------------------------------------------
# RF01 integration: typed ABI in Driver launch; legacy path stays explicit
# ---------------------------------------------------------------------------


class _Pair(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int64), ("y", ctypes.c_void_p)]


def _full_signature():
    return KernelSignature([
        KernelParameter.scalar("small_int64", "int64"),
        KernelParameter.scalar("big_uint64", "uint64"),
        KernelParameter.scalar("f64", "float64"),
        KernelParameter.scalar("flag", "bool"),
        KernelParameter.pointer("out"),
        KernelParameter.structure("pair", _Pair),
    ])


def _read_launched_bytes(fake, signature):
    assert fake.launched
    record = fake.launched[-1]
    if record["arg_bytes"] is not None:
        return record["arg_bytes"]
    kparams = ctypes.cast(record["kparams"], ctypes.POINTER(ctypes.c_void_p))
    chunks = []
    for param in signature.parameters:
        ptr = kparams[len(chunks)]
        size = param.size if param.size is not None else ctypes.sizeof(_Pair)
        chunks.append(ctypes.string_at(ptr, size))
    return chunks


def test_launch_with_signature_marshals_exact_abi_bytes(standalone):
    fake, rt = standalone
    sig = _full_signature()
    fake.snapshot_sizes = [p.size if p.size is not None else ctypes.sizeof(_Pair) for p in sig.parameters]
    pair = _Pair(0x1122334455667788, ctypes.c_void_p(0xABCDEF))
    args = [
        3,  # small value but declared int64 -> 8 bytes
        1 << 63,  # uint64 high-bit set
        1.0000000001,  # declared float64, must NOT narrow to float32
        True,
        0xDEADBEEF,  # raw device address
        pair,
    ]
    rt.launch_kernel(
        ctypes.c_void_p(0xF00D),
        (1, 1, 1),
        (32, 1, 1),
        args,
        signature=sig,
    )
    chunks = _read_launched_bytes(fake, sig)
    assert chunks[0] == ctypes.c_int64(3).value.to_bytes(8, "little", signed=True)
    assert len(chunks[0]) == 8
    assert chunks[1] == (1 << 63).to_bytes(8, "little")
    assert chunks[2] == ctypes.string_at(ctypes.byref(ctypes.c_double(1.0000000001)), 8)
    # float32 would have collapsed this to 1.0; prove it did not.
    assert chunks[2] != ctypes.string_at(ctypes.byref(ctypes.c_float(1.0000000001)), 4)
    assert chunks[3] == b"\x01"
    assert chunks[4] == (0xDEADBEEF).to_bytes(8, "little")
    assert chunks[5] == ctypes.string_at(ctypes.byref(pair), ctypes.sizeof(pair))


def test_known_signature_is_not_overridden_by_value_guessing(standalone):
    fake, rt = standalone
    sig = KernelSignature([
        KernelParameter.scalar("i64", "int64"),
        KernelParameter.scalar("f64", "float64"),
    ])
    # A float cannot be silently stored into a declared int64 slot.
    with pytest.raises(TypeError):
        rt.launch_kernel(
            ctypes.c_void_p(1),
            (1, 1, 1),
            (1, 1, 1),
            [2.0, 1.0],
            signature=sig,
        )
    assert not fake.launched
    # Explicit c_float is rejected where c_double is declared.
    with pytest.raises(TypeError):
        rt.launch_kernel(
            ctypes.c_void_p(1),
            (1, 1, 1),
            (1, 1, 1),
            [1, ctypes.c_float(2.0)],
            signature=sig,
        )


def test_pointer_signature_accepts_numpy_array_address(standalone):
    fake, rt = standalone
    sig = KernelSignature([
        KernelParameter.scalar("n", "int32"),
        KernelParameter.pointer("x"),
    ])
    fake.snapshot_sizes = [4, 8]
    arr = np.arange(16, dtype=np.float32)
    rt.launch_kernel(
        ctypes.c_void_p(1),
        (1, 1, 1),
        (1, 1, 1),
        [arr.shape[0], arr],
        signature=sig,
    )
    chunks = _read_launched_bytes(fake, sig)
    assert chunks[0] == (16).to_bytes(4, "little", signed=True)
    assert int.from_bytes(chunks[1], "little") == int(arr.ctypes.data)


def test_legacy_untyped_path_preserved_for_legacy_callers(standalone):
    fake, rt = standalone
    # No signature -> legacy value-based behavior is retained explicitly.
    fake.snapshot_sizes = [4, 4]
    rt.launch_kernel(
        ctypes.c_void_p(1),
        (1, 1, 1),
        (1, 1, 1),
        [3, 1.0000000001],
    )
    record = fake.launched[-1]
    chunks = record["arg_bytes"]
    # Small int narrowed to int32 (4 bytes), float narrowed to float32.
    assert chunks[0] == (3).to_bytes(4, "little", signed=True)
    assert chunks[1] == ctypes.string_at(ctypes.byref(ctypes.c_float(1.0)), 4)


def test_invalid_signature_type_rejected(standalone):
    fake, rt = standalone
    with pytest.raises(TypeError, match="KernelSignature"):
        rt.launch_kernel(
            ctypes.c_void_p(1),
            (1, 1, 1),
            (1, 1, 1),
            [1],
            signature="nope",
        )


# ---------------------------------------------------------------------------
# RF01/RF04: KernelLauncher trailing signature and ownership
# ---------------------------------------------------------------------------


def _launcher(rt, fake, signature=None):
    mod = rt.load_module(b"cubin")
    return cr.KernelLauncher(
        rt,
        mod,
        "k",
        shared_mem_bytes=0,
        cubin_data=b"cubin",
        signature=signature,
    ), mod


def test_kernel_launcher_carries_trailing_signature(standalone):
    fake, rt = standalone
    sig = KernelSignature([KernelParameter.scalar("a", "float64")])
    launcher, mod = _launcher(rt, fake, signature=sig)
    fake.snapshot_sizes = [8]
    try:
        assert rt._managed_module(mod).live_users == 1
        launcher((1, 1, 1), (1, 1, 1), 1.5)
        chunks = _read_launched_bytes(fake, sig)
        # Bound signature yields a true float64.
        assert chunks[0] == ctypes.string_at(ctypes.byref(ctypes.c_double(1.5)), 8)
        # A per-call override that contradicts the declared type is rejected.
        with pytest.raises(TypeError):
            launcher((1, 1, 1), (1, 1, 1), "bad")
    finally:
        launcher.close()
    assert rt._managed_module(mod).live_users == 0


def test_kernel_launcher_blocks_module_unload_until_closed(standalone):
    fake, rt = standalone
    launcher, mod = _launcher(rt, fake)
    with pytest.raises(RuntimeError, match="live launcher"):
        rt.unload_module(mod)
    launcher.close()
    assert rt.unload_module(mod) is True
    # Idempotent launcher close.
    launcher.close()


def test_kernel_launcher_rejects_bad_signature(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"cubin")
    with pytest.raises(TypeError, match="KernelSignature"):
        cr.KernelLauncher(rt, mod, "k", signature=("float64", ))


# ---------------------------------------------------------------------------
# Resource balance: failure paths release exactly what was acquired
# ---------------------------------------------------------------------------


def test_primary_context_retain_release_balanced_with_torch(monkeypatch):
    monkeypatch.setattr(cr, "TORCH_AVAILABLE", True)

    class _Stream:  # torch.cuda.Stream stand-in is not needed here
        pass

    fake_torch = type("T", (), {})()
    fake_torch.cuda = type(
        "C",
        (),
        {
            "is_available": staticmethod(lambda: True),
            "current_device": staticmethod(lambda: 2),
        },
    )()
    monkeypatch.setattr(cr, "torch", fake_torch, raising=False)

    fake = FakeDriver()
    rt = _make_runtime(fake)
    try:
        assert rt.managed_context.ownership is (lifecycle.ContextOwnership.PRIMARY_RETAINED)
        assert fake.retained and not fake.released
    finally:
        rt.close()
    assert fake.released == [2]
    assert fake.destroyed == []
    assert len(fake.retained) == 1


def test_module_unload_failure_is_checked(standalone):
    fake, rt = standalone
    mod = rt.load_module(b"cubin")
    fake.fail["cuModuleUnload"] = 400
    with pytest.raises(RuntimeError, match="cuModuleUnload"):
        rt.unload_module(mod)
    # Tracking is preserved for retry; clear the injected fault so teardown
    # can complete cleanly.
    assert int(mod.value) in rt._managed_modules
    fake.fail.clear()


# ---------------------------------------------------------------------------
# Real-Driver acceptance path (skipped without hardware)
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_real_driver_modern_context_alloc_and_sync():
    """A modern context must support allocation and synchronization.

    This is the regression for the H800 probe where the old path mixed
    unversioned cuCtxCreate with cuMemAlloc_v2 and got
    CUDA_ERROR_INVALID_CONTEXT (201).
    """
    try:
        lib = ctypes.CDLL("libcuda.so.1")
    except OSError:
        pytest.skip("CUDA Driver not available")

    def _bind(name, argtypes, restype=ctypes.c_int):
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        return fn

    _bind("cuInit", [ctypes.c_uint])
    if lib.cuInit(0) != 0:
        pytest.skip("CUDA Driver initialization unavailable")
    count = ctypes.c_int()
    _bind("cuDeviceGetCount", [ctypes.POINTER(ctypes.c_int)])
    if lib.cuDeviceGetCount(ctypes.byref(count)) != 0 or not count.value:
        pytest.skip("No CUDA devices")

    # Hardware is available: setup, allocation, synchronization and cleanup
    # failures are regressions and must fail the test rather than become skips.
    with cr.CUDARuntime() as rt:
        _bind("cuMemAlloc_v2", [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t])
        _bind("cuMemFree_v2", [ctypes.c_void_p])
        addr = ctypes.c_void_p()
        try:
            assert lib.cuMemAlloc_v2(ctypes.byref(addr), 4096) == 0
            assert addr.value
            rt.synchronize()
        finally:
            if addr.value:
                assert lib.cuMemFree_v2(addr) == 0
