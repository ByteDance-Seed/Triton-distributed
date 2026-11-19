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
Barrier and synchronization intrinsics.
All implementations use standalone PTX (no CuTe/CUTLASS dependency).

Declared-effect contract (the synchronization axis of the scan/indexing
coverage audit): each boundary op below carries a synchronization-boundary
effect via ``sync_effect`` (see little_kernel/core/memory_effects.py).  The
effect records the ORDERING boundary itself — the distinguished operand ""
names no parameter — with the conservative RW access on the ordered space
that tells effect-driven motion/merge analyses to keep every access of that
space on its own side of the op.  Issuer sets select who synchronizes:
WARP for warp-level barriers, THREADS for CTA/named barriers (arrived at by
individual threads) and for device/system fences.
"""

from dataclasses import dataclass

from little_kernel.core.memory_effects import RW_GMEM_DEVICE_FENCE, RW_RMEM_WARP_BARRIER, RW_SMEM_CTA_BARRIER, sync_effect
from little_kernel.language.builtin_base import builtin, Builtin, BuiltinOp
from little_kernel.core.type_system import void, uint32, uint64, Pointer, bool_ as ll_bool

# ==============================================================================
# Canonical Hopper producer/consumer mbarrier rings
# ==============================================================================


@dataclass(frozen=True)
class MBarrierRingSlot:
    """A ring slot and the parity waited on by an mbarrier instruction."""

    slot: int
    phase: int


@dataclass(frozen=True)
class MBarrierFullEmptyRing:
    """Schedule arithmetic for a canonical multi-stage full/empty mbarrier ring.

    The producer leader initializes one full barrier per stage with one
    arrival and one empty barrier per stage with ``consumer_arrivals``.
    Before reuse it waits on the empty barrier; after TMA issues it arrives
    with ``transaction_bytes`` on the full barrier.  Consumers wait on the
    full barrier, consume the stage, release the preceding empty slot after
    the prior WGMMA group has retired, and release the final slot after the
    last group drains.  This class computes only that generic slot/parity
    schedule; it emits no task-specific CUDA text.
    """

    stages: int
    transaction_bytes: int
    consumer_arrivals: int = 1

    def __post_init__(self):
        for name in ("stages", "transaction_bytes", "consumer_arrivals"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {value!r}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if not 1 <= self.stages <= 8:
            raise ValueError(f"stages must be in [1, 8], got {self.stages}")
        if self.transaction_bytes % 16 != 0:
            raise ValueError("transaction_bytes must be a 16-byte multiple, got "
                             f"{self.transaction_bytes}")

    def slot(self, iteration: int) -> int:
        self._check_iteration(iteration)
        return iteration % self.stages

    def full_phase(self, iteration: int) -> int:
        self._check_iteration(iteration)
        return (iteration // self.stages) & 1

    def consumer_full_wait(self, iteration: int) -> MBarrierRingSlot:
        """Full barrier slot/parity consumed before computing ``iteration``."""
        return MBarrierRingSlot(self.slot(iteration), self.full_phase(iteration))

    def producer_empty_wait(self, iteration: int):
        """Empty barrier wait before refilling a wrapped slot, else ``None``."""
        self._check_iteration(iteration)
        if iteration < self.stages:
            return None
        return MBarrierRingSlot(
            iteration % self.stages,
            ((iteration // self.stages) - 1) & 1,
        )

    def consumer_release_previous(self, current_iteration: int):
        """Release the slot made reusable after starting ``current_iteration``."""
        self._check_iteration(current_iteration)
        if current_iteration == 0:
            return None
        return (current_iteration - 1) % self.stages

    def consumer_release_final(self, last_iteration: int) -> int:
        """Release the final slot after all asynchronous compute drains."""
        return self.slot(last_iteration)

    @staticmethod
    def _check_iteration(iteration: int):
        if not isinstance(iteration, int) or isinstance(iteration, bool):
            raise TypeError(f"iteration must be an integer, got {iteration!r}")
        if iteration < 0:
            raise ValueError(f"iteration must be nonnegative, got {iteration}")


# ==============================================================================
# Basic synchronization
# ==============================================================================


@builtin
class __syncwarp(BuiltinOp):
    """Sync all threads in the warp.

    Declared effect: ``sync_effect(RW_RMEM_WARP_BARRIER)`` — a warp-issued
    ordering boundary over the cross-lane register view (RMEM); shuffle/scan
    effects are protected from motion across it.
    """

    effects = (sync_effect(RW_RMEM_WARP_BARRIER), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        return Builtin(body="", includes=[], return_val="__syncwarp()")


@builtin
class __syncthreads(BuiltinOp):
    """CUDA __syncthreads() barrier synchronization.

    Declared effect: ``sync_effect(RW_SMEM_CTA_BARRIER)`` — a CTA-wide
    ordering boundary over shared memory (the channel for all inter-thread
    visibility within a CTA).
    """

    effects = (sync_effect(RW_SMEM_CTA_BARRIER), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        return Builtin(body="", includes=[], return_val="__syncthreads()")


# ==============================================================================
# Cluster synchronization
# ==============================================================================


@builtin
class cluster_arrive(BuiltinOp):
    """Arrive at cluster barrier.

    Declared effect: ``sync_effect(RW_SMEM_CTA_BARRIER)`` — the release half
    of the cluster ordering boundary over shared memory.  The synchronized
    set (the cluster) is a superset of the CTA, so the CTA-barrier kind is
    the sound conservative claim while the effect model carries no
    cluster-issuer vocabulary.
    """

    effects = (sync_effect(RW_SMEM_CTA_BARRIER), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ void cluster_arrive_fn() {
    asm volatile("barrier.cluster.arrive;\\n" ::: "memory");
}
"""
        return Builtin(body=body, includes=[], return_val="cluster_arrive_fn()")


@builtin
class cluster_wait(BuiltinOp):
    """Wait on cluster barrier.

    Declared effect: ``sync_effect(RW_SMEM_CTA_BARRIER)`` — the acquire half
    of the cluster ordering boundary over shared memory (same conservative
    kind as cluster_arrive).
    """

    effects = (sync_effect(RW_SMEM_CTA_BARRIER), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ void cluster_wait_fn() {
    asm volatile("barrier.cluster.wait;\\n" ::: "memory");
}
"""
        return Builtin(body=body, includes=[], return_val="cluster_wait_fn()")


@builtin
class cluster_sync(BuiltinOp):
    """Full cluster barrier sync.

    Declared effect: ``sync_effect(RW_SMEM_CTA_BARRIER)`` — the full cluster
    ordering boundary over shared memory (same conservative kind as
    cluster_arrive/cluster_wait).
    """

    effects = (sync_effect(RW_SMEM_CTA_BARRIER), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ void cluster_sync_fn() {
    asm volatile("barrier.cluster.arrive;\\nbarrier.cluster.wait;\\n" ::: "memory");
}
"""
        return Builtin(body=body, includes=[], return_val="cluster_sync_fn()")


@builtin
class cluster_rank(BuiltinOp):
    """Get cluster CTA rank."""

    @staticmethod
    def return_type():
        return uint32

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ uint32_t cluster_rank_fn() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}
"""
        return Builtin(body=body, includes=[], return_val="cluster_rank_fn()")


# ==============================================================================
# MBarrier operations (standalone PTX, uses __cvta_generic_to_shared)
#
# Two typed barrier-slot idioms are accepted by every CTA-scoped mbarrier op:
#
#   1. Shared tensor slot (legacy): pass an indexable shared allocation, e.g.
#        bar = ll.empty([1], dtype=ll.uint64, scope="shared")
#        ll.alloc_shared_memory("s_bar", ll.uint64, 1, 8)  # ... then s_bar
#      The emitted helper takes ``uint64_t*`` and derives the shared-window
#      address itself with __cvta_generic_to_shared(&bar[0]).
#
#   2. Byte address (canonical PTX form): pass a uint32 shared-window byte
#      address obtained from ll.cvta_generic_to_shared, e.g.
#        bar_addr: ll.uint32 = ll.cvta_generic_to_shared(s_bar)
#        ll.init_smem_barrier(bar_addr, 1)
#      The emitted helper takes ``uint32_t`` and uses it directly in [%0];
#      no second cvta is applied (a uint32 value is already a .shared state
#      space address, and re-running cvta on it is invalid PTX).
#
# Type-driven selection happens in codegen through ``_arg_dtypes``; anything
# that is neither a Tensor/pointer slot nor a uint32 address is rejected at
# type inference (a scalar uint64 value, e.g. reading slot contents, is a
# load, not the barrier address — matching dsm_store_f32's uint32-address
# contract).
# ==============================================================================


def _mbarrier_arg_is_u32_address(_arg_dtypes, index=0):
    """True when argument ``index`` carries a uint32 shared byte address."""
    if not _arg_dtypes or len(_arg_dtypes) <= index:
        return False
    return _arg_dtypes[index] == uint32


def _check_mbarrier_slot_type(intrin_name, bar_t):
    """Legality of an mbarrier slot argument at type inference.

    ``bar_t`` is the inferred LLType of the first argument.  Accept a shared
    tensor/pointer slot and a uint32 cvta address; reject scalar values
    (most often an accidental load of the 64-bit barrier object) with a
    diagnosis that names both canonical idioms.
    """
    if bar_t is None:
        return
    if bar_t == uint32:
        return
    # Indexing a shared barrier ring by pointer arithmetic decays its Tensor
    # type to uint64*. It still names a slot; it does not load slot contents.
    if isinstance(bar_t, Pointer) and bar_t.inner_type == uint64:
        return
    is_tensor = getattr(bar_t, "is_tensor", None)
    if callable(is_tensor) and is_tensor():
        return
    from little_kernel.core.passes.utils.type_inference.type_inference_core import TypeInferenceError
    raise TypeInferenceError(f"{intrin_name}: barrier slot argument has type {bar_t}, but it must "
                             f"be either a shared-memory tensor slot "
                             f"(bar = ll.empty([1], ll.uint64, scope='shared'), or "
                             f'll.alloc_shared_memory("s_bar", ll.uint64, 1, 8) and pass s_bar) '
                             f"or a uint32 shared-window byte address "
                             f"(addr: ll.uint32 = ll.cvta_generic_to_shared(bar)). A scalar value "
                             f"loads the barrier slot contents instead of naming its address.")


@builtin
class init_smem_barrier(BuiltinOp):
    """Initialize mbarrier at a shared memory slot.

    Accepts a shared tensor pointer or a uint32 shared-window byte address
    from ll.cvta_generic_to_shared (see module mbarrier note).
    """

    __codegen_needs_types__ = True

    @staticmethod
    def return_type(smem_bar_ptr, arrive_cnt):
        _check_mbarrier_slot_type("init_smem_barrier", smem_bar_ptr)
        return void

    @staticmethod
    def codegen(smem_bar_ptr, arrive_cnt, _arg_dtypes=None):
        if _mbarrier_arg_is_u32_address(_arg_dtypes):
            body = """
__device__ __forceinline__ void init_smem_barrier_addr_fn(uint32_t bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"(bar), "r"(count));
}
"""
            return Builtin(body=body, includes=[],
                           return_val=f"init_smem_barrier_addr_fn({smem_bar_ptr}, {arrive_cnt})")
        body = """
__device__ __forceinline__ void init_smem_barrier_fn(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(count));
}
"""
        return Builtin(body=body, includes=[], return_val=f"init_smem_barrier_fn({smem_bar_ptr}, {arrive_cnt})")


@builtin
class fence_smem_barrier_init(BuiltinOp):
    """Fence for mbarrier initialization."""

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ void fence_smem_barrier_init_fn() {
    asm volatile("fence.mbarrier_init.release.cluster;\\n" ::: "memory");
}
"""
        return Builtin(body=body, includes=[], return_val="fence_smem_barrier_init_fn()")


@builtin
class mbarrier_arrive(BuiltinOp):
    """Arrive on a mbarrier.

    Accepts a shared tensor pointer or a uint32 shared-window byte address
    from ll.cvta_generic_to_shared (see module mbarrier note).
    """

    __codegen_needs_types__ = True

    @staticmethod
    def return_type(smem):
        _check_mbarrier_slot_type("mbarrier_arrive", smem)
        return void

    @staticmethod
    def codegen(smem, _arg_dtypes=None):
        if _mbarrier_arg_is_u32_address(_arg_dtypes):
            body = """
__device__ __forceinline__ void mbarrier_arrive_addr_fn(uint32_t bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"(bar) : "memory");
}
"""
            return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_addr_fn({smem})")
        body = """
__device__ __forceinline__ void mbarrier_arrive_fn(uint64_t* bar) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])) : "memory");
}
"""
        return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_fn({smem})")


@builtin
class mbarrier_arrive_and_expect_tx(BuiltinOp):
    """Arrive and expect transaction bytes on a mbarrier.

    Accepts a shared tensor pointer or a uint32 shared-window byte address
    from ll.cvta_generic_to_shared (see module mbarrier note).
    """

    __codegen_needs_types__ = True

    @staticmethod
    def return_type(smem, transaction_bytes):
        _check_mbarrier_slot_type("mbarrier_arrive_and_expect_tx", smem)
        return void

    @staticmethod
    def codegen(smem, transaction_bytes, _arg_dtypes=None):
        if _mbarrier_arg_is_u32_address(_arg_dtypes):
            body = """
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_addr_fn(uint32_t bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"(bar), "r"(tx_bytes) : "memory");
}
"""
            return Builtin(body=body, includes=[],
                           return_val=f"mbarrier_arrive_and_expect_tx_addr_fn({smem}, {transaction_bytes})")
        body = """
__device__ __forceinline__ void mbarrier_arrive_and_expect_tx_fn(uint64_t* bar, uint32_t tx_bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;"
        :: "r"((uint32_t)__cvta_generic_to_shared(&bar[0])), "r"(tx_bytes) : "memory");
}
"""
        return Builtin(body=body, includes=[],
                       return_val=f"mbarrier_arrive_and_expect_tx_fn({smem}, {transaction_bytes})")


@builtin
class mbarrier_wait(BuiltinOp):
    """Wait on a mbarrier parity.

    Accepts a shared tensor pointer or a uint32 shared-window byte address
    from ll.cvta_generic_to_shared (see module mbarrier note).
    """

    __codegen_needs_types__ = True

    @staticmethod
    def return_type(smem, phase):
        _check_mbarrier_slot_type("mbarrier_wait", smem)
        return void

    @staticmethod
    def codegen(smem, phase, _arg_dtypes=None):
        if _mbarrier_arg_is_u32_address(_arg_dtypes):
            body = """
__device__ __forceinline__ void mbarrier_wait_addr_fn(uint32_t smem_addr, uint32_t phase) {
    uint32_t ticks = 0x989680;  // Large timeout value (matches FA3 CUDA)
    asm volatile(
        "{\\n"
        ".reg .pred P1;\\n"
        "LAB_WAIT_%=:\\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"
        "@P1 bra.uni DONE_%=;\\n"
        "bra.uni LAB_WAIT_%=;\\n"
        "DONE_%=:\\n"
        "}\\n"
        :: "r"(smem_addr), "r"(phase), "r"(ticks));
}
"""
            return Builtin(body=body, includes=[], return_val=f"mbarrier_wait_addr_fn({smem}, {phase})")
        body = """
__device__ __forceinline__ void mbarrier_wait_fn(uint64_t* bar, uint32_t phase) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t ticks = 0x989680;  // Large timeout value (matches FA3 CUDA)
    asm volatile(
        "{\\n"
        ".reg .pred P1;\\n"
        "LAB_WAIT_%=:\\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"
        "@P1 bra.uni DONE_%=;\\n"
        "bra.uni LAB_WAIT_%=;\\n"
        "DONE_%=:\\n"
        "}\\n"
        :: "r"(smem_addr), "r"(phase), "r"(ticks));
}
"""
        return Builtin(body=body, includes=[], return_val=f"mbarrier_wait_fn({smem}, {phase})")


@builtin
class mbarrier_try_wait_parity(BuiltinOp):
    """Non-blocking try_wait on mbarrier parity. Returns true if barrier is ready.

    Accepts a shared tensor pointer or a uint32 shared-window byte address
    from ll.cvta_generic_to_shared (see module mbarrier note).
    """

    __codegen_needs_types__ = True

    @staticmethod
    def return_type(smem, phase):
        _check_mbarrier_slot_type("mbarrier_try_wait_parity", smem)
        return ll_bool

    @staticmethod
    def codegen(smem, phase, _arg_dtypes=None):
        if _mbarrier_arg_is_u32_address(_arg_dtypes):
            body = """
__device__ __forceinline__ bool mbarrier_try_wait_parity_addr_fn(uint32_t smem_addr, uint32_t phase) {
    uint32_t ready;
    asm volatile(
        "{\\n"
        ".reg .pred P1;\\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\\n"
        "selp.b32 %0, 1, 0, P1;\\n"
        "}\\n"
        : "=r"(ready)
        : "r"(smem_addr), "r"(phase)
    );
    return ready != 0;
}
"""
            return Builtin(body=body, includes=[], return_val=f"mbarrier_try_wait_parity_addr_fn({smem}, {phase})")
        body = """
__device__ __forceinline__ bool mbarrier_try_wait_parity_fn(uint64_t* bar, uint32_t phase) {
    uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t ready;
    asm volatile(
        "{\\n"
        ".reg .pred P1;\\n"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\\n"
        "selp.b32 %0, 1, 0, P1;\\n"
        "}\\n"
        : "=r"(ready)
        : "r"(smem_addr), "r"(phase)
    );
    return ready != 0;
}
"""
        return Builtin(body=body, includes=[], return_val=f"mbarrier_try_wait_parity_fn({smem}, {phase})")


@builtin
class mbarrier_wait_if_needed(BuiltinOp):
    """Conditional wait on mbarrier - only waits if not already ready.

    Accepts a shared tensor pointer or a uint32 shared-window byte address
    from ll.cvta_generic_to_shared (see module mbarrier note).
    """

    __codegen_needs_types__ = True

    @staticmethod
    def return_type(smem, phase, already_ready):
        _check_mbarrier_slot_type("mbarrier_wait_if_needed", smem)
        return void

    @staticmethod
    def codegen(smem, phase, already_ready, _arg_dtypes=None):
        if _mbarrier_arg_is_u32_address(_arg_dtypes):
            body = """
__device__ __forceinline__ void mbarrier_wait_if_needed_addr_fn(uint32_t smem_addr, uint32_t phase, bool already_ready) {
    if (!already_ready) {
        uint32_t ticks = 0x989680;  // Large timeout value (matches FA3 CUDA)
        asm volatile(
            "{\\n"
            ".reg .pred P1;\\n"
            "LAB_WAIT_%=:\\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"
            "@P1 bra.uni DONE_%=;\\n"
            "bra.uni LAB_WAIT_%=;\\n"
            "DONE_%=:\\n"
            "}\\n"
            :: "r"(smem_addr), "r"(phase), "r"(ticks));
    }
}
"""
            return Builtin(body=body, includes=[],
                           return_val=f"mbarrier_wait_if_needed_addr_fn({smem}, {phase}, {already_ready})")
        body = """
__device__ __forceinline__ void mbarrier_wait_if_needed_fn(uint64_t* bar, uint32_t phase, bool already_ready) {
    if (!already_ready) {
        uint32_t smem_addr = (uint32_t)__cvta_generic_to_shared(&bar[0]);
        uint32_t ticks = 0x989680;  // Large timeout value (matches FA3 CUDA)
        asm volatile(
            "{\\n"
            ".reg .pred P1;\\n"
            "LAB_WAIT_%=:\\n"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\\n"
            "@P1 bra.uni DONE_%=;\\n"
            "bra.uni LAB_WAIT_%=;\\n"
            "DONE_%=:\\n"
            "}\\n"
            :: "r"(smem_addr), "r"(phase), "r"(ticks));
    }
}
"""
        return Builtin(body=body, includes=[],
                       return_val=f"mbarrier_wait_if_needed_fn({smem}, {phase}, {already_ready})")


@builtin
class mbarrier_arrive_remote(BuiltinOp):
    """Arrive on mbarrier at a remote CTA in cluster."""

    @staticmethod
    def return_type(bar_ptr, target_cta):
        return void

    @staticmethod
    def codegen(bar_ptr, target_cta):
        body = """
__device__ __forceinline__ void mbarrier_arrive_remote_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t smem_addr = __cvta_generic_to_shared(&bar[0]);
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(smem_addr), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" :: "r"(remote_addr) : "memory");
}
"""
        return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_remote_fn({bar_ptr}, {target_cta})")


@builtin
class dsm_store_f32(BuiltinOp):
    """Store a float32 into a peer CTA's shared window via DSM.

    Contract:
      * ``smem_base_u32`` -- raw uint32 shared-window address in the calling
        CTA (e.g. from ``cvta_generic_to_shared`` on a shared tensor slot);
        it is remapped to ``target_rank``'s window with
        ``mapa.shared::cluster.u32``.
      * ``target_rank``   -- destination CTA rank within the cluster.
      * ``byte_off``      -- byte offset added to the remapped base.
      * ``value``         -- the float32 to store.

    Legality: requires a cluster launch (``launch.cluster_shape``); the
    (base, byte_off) pair must address a 4-byte-aligned float32 slot. The
    store is only visible to the peer after a subsequent cluster_sync or
    barrier release/acquire chain.
    """

    @staticmethod
    def return_type(smem_base_u32, target_rank, byte_off, value):
        return void

    @staticmethod
    def codegen(smem_base_u32, target_rank, byte_off, value):
        body = """
__device__ __forceinline__ void dsm_store_f32_fn(uint32_t smem_base_u32, uint32_t target_rank, uint32_t byte_off, float value) {
    uint32_t remote_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote_addr) : "r"(smem_base_u32), "r"(target_rank));
    asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(remote_addr + byte_off), "f"(value) : "memory");
}
"""
        return Builtin(body=body, includes=["<cstdint>"],
                       return_val=f"dsm_store_f32_fn({smem_base_u32}, {target_rank}, {byte_off}, {value})")


# ==============================================================================
# Memory fences
# ==============================================================================


@builtin
class threadfence(BuiltinOp):
    """Device-scope memory fence (__threadfence).

    Declared effect: ``sync_effect(RW_GMEM_DEVICE_FENCE)`` — a thread-issued
    ordering boundary over global memory at device scope.  The effect model
    carries no fence-scope vocabulary, so the system-scope variant shares
    this same conservative kind.
    """

    effects = (sync_effect(RW_GMEM_DEVICE_FENCE), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        return Builtin(body="", includes=[], return_val="__threadfence()")


@builtin
class threadfence_system(BuiltinOp):
    """System-scope memory fence (__threadfence_system).

    Declared effect: ``sync_effect(RW_GMEM_DEVICE_FENCE)`` — same kind as
    the device-scope fence (no fence-scope vocabulary in the effect model;
    the wider scope is a superset, so the shared claim is sound).
    """

    effects = (sync_effect(RW_GMEM_DEVICE_FENCE), )

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        return Builtin(body="", includes=[], return_val="__threadfence_system()")


@builtin
class fence_proxy_async(BuiltinOp):
    """Generic async proxy fence (fence.proxy.async, without shared::cta qualifier).
    Required after manual shared memory writes before WGMMA consumption."""

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ void fence_proxy_async_fn() {
    asm volatile("fence.proxy.async;\\n" ::: "memory");
}
"""
        return Builtin(body=body, includes=[], return_val="fence_proxy_async_fn()")


@builtin
class fence_async_shared(BuiltinOp):
    """Async proxy fence on shared memory (fence.proxy.async.shared::cta)."""

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
__device__ __forceinline__ void fence_async_shared_fn() {
    asm volatile("fence.proxy.async.shared::cta;\\n" ::: "memory");
}
"""
        return Builtin(body=body, includes=[], return_val="fence_async_shared_fn()")


# ==============================================================================
# Named barrier
# ==============================================================================


def _named_barrier_literal(text):
    """Parse a visited codegen operand as an int literal, else None."""
    try:
        return int(str(text).strip(), 0)
    except (TypeError, ValueError):
        return None


def _check_named_barrier(bar_id, count, name):
    """Compile-time legality checks for named/partial barriers.

    Checked when the operands are integer literals (the common case);
    non-literal operands defer to the documented caller-side contract on the
    builtins (same style as the 16B-alignment precondition on the vector
    GMEM intrinsics). Raises ValueError on a statically-known violation.
    """
    bar_id_lit = _named_barrier_literal(bar_id)
    if bar_id_lit is not None and not 0 <= bar_id_lit <= 15:
        raise ValueError(f"{name}: bar_id must be in [0, 15] (PTX barrier resource id), "
                         f"got {bar_id_lit}")
    count_lit = _named_barrier_literal(count)
    if count_lit is not None and (count_lit <= 0 or count_lit % 32 != 0):
        raise ValueError(f"{name}: count must be a positive multiple of 32 (warp size), "
                         f"got {count_lit}")


_NAMED_BARRIER_CONTRACT = ("Legality contract (caller-side, like the 16B-alignment precondition on "
                           "the vector GMEM intrinsics): bar_id is a PTX barrier resource in "
                           "[0, 15]; count is a positive multiple of 32 equal to the TOTAL number "
                           "of threads that arrive at this barrier id; every participating warp "
                           "executes the same sequence of bar.* operations on this id (uniformity "
                           "across the participating warps is required — divergent arrival is "
                           "undefined); barriers with distinct ids are independent, so independent "
                           "sub-groups of one block may sync without coupling. bar_id=0 with "
                           "count == blockDim is equivalent to __syncthreads().")


@builtin
class named_barrier_sync(BuiltinOp):
    """Named/partial barrier synchronization (PTX ``bar.sync bar_id, count``).

    Syncs only the warps that arrive at the same barrier resource id,
    decoupling independent thread sub-groups inside one block (e.g. two
    problem replicas packed into a 256-thread block, or producer/consumer
    warp specialization). """ + _NAMED_BARRIER_CONTRACT + """

    Values are never altered (pure CUDA primitive passthrough); compile-time
    legality checks apply when bar_id/count are integer literals.

    Declared effect: ``sync_effect(RW_SMEM_CTA_BARRIER)`` — an ordering
    boundary over shared memory for the arriving CTA thread sub-set.  The
    kind is shared with __syncthreads because the effect model carries no
    per-barrier-id vocabulary; the synchronized set is a subset of the CTA,
    never a superset, so a CTA-wide boundary is the sound conservative
    claim.
    """

    effects = (sync_effect(RW_SMEM_CTA_BARRIER), )

    @staticmethod
    def return_type(bar_id, count):
        _check_named_barrier(bar_id, count, "named_barrier_sync")
        return void

    @staticmethod
    def codegen(bar_id, count):
        _check_named_barrier(bar_id, count, "named_barrier_sync")
        body = """
__device__ __forceinline__ void named_barrier_sync_fn(int bar_id, int count) {
    asm volatile("bar.sync %0, %1;" :: "r"(bar_id), "r"(count) : "memory");
}
"""
        return Builtin(body=body, includes=[], return_val=f"named_barrier_sync_fn({bar_id}, {count})")


@builtin
class named_barrier_arrive(BuiltinOp):
    """Named/partial barrier arrive without wait (PTX ``bar.arrive bar_id, count``).

    Producer half of a named barrier split; the consumer executes
    named_barrier_sync on the same id. """ + _NAMED_BARRIER_CONTRACT + """

    Compile-time legality checks apply when bar_id/count are integer literals.

    Declared effect: ``sync_effect(RW_SMEM_CTA_BARRIER)`` — arrival is the
    release half of the named barrier's ordering boundary over shared
    memory (same conservative kind as named_barrier_sync).
    """

    effects = (sync_effect(RW_SMEM_CTA_BARRIER), )

    @staticmethod
    def return_type(bar_id, count):
        _check_named_barrier(bar_id, count, "named_barrier_arrive")
        return void

    @staticmethod
    def codegen(bar_id, count):
        _check_named_barrier(bar_id, count, "named_barrier_arrive")
        body = """
__device__ __forceinline__ void named_barrier_arrive_fn(int bar_id, int count) {
    asm volatile("bar.arrive %0, %1;" :: "r"(bar_id), "r"(count));
}
"""
        return Builtin(body=body, includes=[], return_val=f"named_barrier_arrive_fn({bar_id}, {count})")


# ==============================================================================
# Grid-level synchronization (atomic-based, no cooperative_groups needed)
# ==============================================================================


@builtin
class grid_sync(BuiltinOp):
    """Grid-level synchronization via atomic barrier (no cooperative_groups)."""

    @staticmethod
    def return_type():
        return void

    @staticmethod
    def codegen():
        body = """
// Grid-level barrier using atomic operations (no cooperative_groups needed)
__device__ unsigned int __grid_sync_count = 0;
__device__ volatile int __grid_sync_sense = 0;

__device__ __forceinline__ void grid_sync_fn() {
    __syncthreads();
    __threadfence();
    // Only (0,0,0) thread of each block participates in the grid barrier
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        unsigned int num_blocks = gridDim.x * gridDim.y * gridDim.z;
        // Atomically increment the arrival counter
        unsigned int arrived = atomicAdd(&__grid_sync_count, 1);
        if (arrived == num_blocks - 1) {
            // Last block: reset counter and flip sense
            __grid_sync_count = 0;
            __threadfence();
            __grid_sync_sense ^= 1;
        } else {
            // Wait for the sense to flip
            int expected = __grid_sync_sense ^ 1;
            while (__grid_sync_sense != expected) {
                // Spin-wait
            }
        }
    }
    __syncthreads();
}
"""
        return Builtin(body=body, includes=[], return_val="grid_sync_fn()")


# ==============================================================================
# SM100 Cluster Barrier Operations
# ==============================================================================

# Cluster-scoped barrier operations (for SM100 2SM coordination)
# ==============================================================================


def codegen_mbarrier_arrive_expect_tx_cluster(bar_ptr, tx_bytes, target_cta):
    body = """
__device__ __forceinline__ void mbarrier_arrive_expect_tx_cluster_fn(uint64_t* bar, uint32_t tx, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.expect_tx.shared::cluster.b64 _, [%0], %1;"
                 :: "r"(remote_a), "r"(tx));
}
"""
    return Builtin(body=body, includes=[],
                   return_val=f"mbarrier_arrive_expect_tx_cluster_fn({bar_ptr}, {tx_bytes}, {target_cta})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive_expect_tx_cluster)
def mbarrier_arrive_expect_tx_cluster(bar_ptr, tx_bytes, target_cta):
    """Cluster-scoped arrive_expect_tx: signals a barrier in target_cta's shared memory."""
    raise RuntimeError("should not call mbarrier_arrive_expect_tx_cluster in compilation")


def codegen_mbarrier_arrive_cluster(bar_ptr, target_cta):
    body = """
__device__ __forceinline__ void mbarrier_arrive_cluster_fn(uint64_t* bar, uint32_t target_cta) {
    uint32_t a = (uint32_t)__cvta_generic_to_shared(&bar[0]);
    uint32_t remote_a;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote_a) : "r"(a), "r"(target_cta));
    asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                 :: "r"(remote_a));
}
"""
    return Builtin(body=body, includes=[], return_val=f"mbarrier_arrive_cluster_fn({bar_ptr}, {target_cta})")


@builtin(eval_return_type=void, codegen_func=codegen_mbarrier_arrive_cluster)
def mbarrier_arrive_cluster(bar_ptr, target_cta):
    """Arrive at a barrier on a remote CTA in the cluster."""
    raise RuntimeError("should not call mbarrier_arrive_cluster in compilation")
