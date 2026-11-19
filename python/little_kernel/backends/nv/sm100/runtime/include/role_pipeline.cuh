/*
 * Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

// RF05/RF15: this is the canonical, packaged role-pipeline backend header.
//
// It references the CUDA Driver type CUtensorMap (TMA descriptors). NVCC 13.0
// standalone compilation does not pre-force-include the Driver header, so the
// dependency is stated explicitly here rather than relying on a translation
// unit to have included <cuda.h> first.
#include <cuda.h>

#include <stdint.h>

#ifndef LK_ROLE_PIPELINE_DEVICE
#define LK_ROLE_PIPELINE_DEVICE __device__ __forceinline__
#endif

namespace little_kernel {
namespace role_pipeline {

LK_ROLE_PIPELINE_DEVICE uint32_t smem_addr(uint64_t *bar) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(&bar[0]));
}

LK_ROLE_PIPELINE_DEVICE uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}

template <int CtaGroup> LK_ROLE_PIPELINE_DEVICE uint32_t cta_group_rank() {
  static_assert(CtaGroup == 1 or CtaGroup == 2,
                "CTA group size must be 1 or 2");
  return cluster_rank() % CtaGroup;
}

template <int CtaGroup>
LK_ROLE_PIPELINE_DEVICE uint32_t cta_group_leader_rank() {
  return cluster_rank() - cta_group_rank<CtaGroup>();
}

template <int CtaGroup> LK_ROLE_PIPELINE_DEVICE bool cta_group_is_leader() {
  return cta_group_rank<CtaGroup>() == 0;
}

template <int ClusterX, int ClusterY, int ClusterZ, int Axis>
LK_ROLE_PIPELINE_DEVICE uint32_t cluster_coord() {
  static_assert(ClusterX > 0 and ClusterY > 0 and ClusterZ > 0);
  static_assert(Axis >= 0 and Axis < 3);
  uint32_t rank = cluster_rank();
  if constexpr (Axis == 0) {
    return rank % ClusterX;
  } else if constexpr (Axis == 1) {
    return (rank / ClusterX) % ClusterY;
  } else {
    return rank / (ClusterX * ClusterY);
  }
}

template <int ClusterX, int ClusterY, int ClusterZ, int GroupX, int GroupY,
          int GroupZ, int Axis>
LK_ROLE_PIPELINE_DEVICE uint32_t group_coord() {
  static_assert(ClusterX % GroupX == 0 and ClusterY % GroupY == 0 and
                ClusterZ % GroupZ == 0);
  constexpr int GroupExtent =
      Axis == 0 ? GroupX : (Axis == 1 ? GroupY : GroupZ);
  return cluster_coord<ClusterX, ClusterY, ClusterZ, Axis>() / GroupExtent;
}

template <int ClusterX, int ClusterY, int ClusterZ, int GroupX, int GroupY,
          int GroupZ, int Axis>
LK_ROLE_PIPELINE_DEVICE uint16_t tma_multicast_mask() {
  static_assert(ClusterX * ClusterY * ClusterZ <= 16);
  constexpr int ClusterExtent =
      Axis == 0 ? ClusterX : (Axis == 1 ? ClusterY : ClusterZ);
  constexpr int GroupExtent =
      Axis == 0 ? GroupX : (Axis == 1 ? GroupY : GroupZ);
  constexpr int AxisStride =
      Axis == 0 ? 1 : (Axis == 1 ? ClusterX : ClusterX * ClusterY);
  constexpr int GroupStride = AxisStride * GroupExtent;
  constexpr int GroupCount = ClusterExtent / GroupExtent;
  constexpr uint32_t Pattern =
      ((1u << (GroupCount * GroupStride)) - 1u) / ((1u << GroupStride) - 1u);
  uint32_t group =
      group_coord<ClusterX, ClusterY, ClusterZ, GroupX, GroupY, GroupZ, Axis>();
  return static_cast<uint16_t>(Pattern
                               << (cluster_rank() - group * GroupStride));
}

template <int ClusterX, int ClusterY, int ClusterZ, int GroupX, int GroupY,
          int GroupZ, int Axis>
LK_ROLE_PIPELINE_DEVICE bool tma_multicast_is_issuer() {
  return group_coord<ClusterX, ClusterY, ClusterZ, GroupX, GroupY, GroupZ,
                     Axis>() == 0;
}

template <int ClusterX, int ClusterY, int ClusterZ, int GroupX, int GroupY,
          int GroupZ>
LK_ROLE_PIPELINE_DEVICE uint32_t rank_in_group() {
  uint32_t local_x = cluster_coord<ClusterX, ClusterY, ClusterZ, 0>() % GroupX;
  uint32_t local_y = cluster_coord<ClusterX, ClusterY, ClusterZ, 1>() % GroupY;
  uint32_t local_z = cluster_coord<ClusterX, ClusterY, ClusterZ, 2>() % GroupZ;
  return local_x + GroupX * (local_y + GroupY * local_z);
}

template <int ClusterX, int ClusterY, int ClusterZ, int GroupX, int GroupY,
          int GroupZ>
LK_ROLE_PIPELINE_DEVICE uint32_t group_index() {
  constexpr int GridX = ClusterX / GroupX;
  constexpr int GridY = ClusterY / GroupY;
  uint32_t x =
      group_coord<ClusterX, ClusterY, ClusterZ, GroupX, GroupY, GroupZ, 0>();
  uint32_t y =
      group_coord<ClusterX, ClusterY, ClusterZ, GroupX, GroupY, GroupZ, 1>();
  uint32_t z =
      group_coord<ClusterX, ClusterY, ClusterZ, GroupX, GroupY, GroupZ, 2>();
  return x + GridX * (y + GridY * z);
}

template <int CtaGroup>
LK_ROLE_PIPELINE_DEVICE uint16_t cta_group_multicast_mask() {
  uint32_t base = cta_group_leader_rank<CtaGroup>();
  return static_cast<uint16_t>(((1u << CtaGroup) - 1u) << base);
}

LK_ROLE_PIPELINE_DEVICE void init_mbarrier(uint64_t *bar, uint32_t count) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" ::"r"(smem_addr(bar)),
               "r"(count));
}

LK_ROLE_PIPELINE_DEVICE void pipeline_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}

LK_ROLE_PIPELINE_DEVICE uint32_t pipeline_elect_one() {
  uint32_t pred;
  asm volatile("{\n.reg .pred p;\n"
               "elect.sync _|p, 0xFFFFFFFF;\n"
               "selp.b32 %0, 1, 0, p;\n}\n"
               : "=r"(pred));
  return pred;
}

template <typename... Pipelines>
LK_ROLE_PIPELINE_DEVICE void init_pipelines(int warp_idx, int init_warp,
                                            Pipelines &...pipelines) {
  if (warp_idx == init_warp && pipeline_elect_one()) {
    (pipelines.init(), ...);
    pipeline_init_fence();
  }
}

LK_ROLE_PIPELINE_DEVICE void prefetch_tma_descriptor(const CUtensorMap *d) {
  asm volatile("prefetch.tensormap [%0];" ::"l"(d) : "memory");
}

template <int CtaGroup>
LK_ROLE_PIPELINE_DEVICE void tmem_alloc(uint32_t *dst_smem, int ncols) {
  static_assert(CtaGroup == 1 || CtaGroup == 2,
                "TMEM allocation requires CTA group 1 or 2");
  uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
  if constexpr (CtaGroup == 1) {
    asm volatile(
        "tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(a),
        "r"(ncols));
  } else {
    asm volatile(
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::
            "r"(a),
        "r"(ncols));
  }
}

template <int CtaGroup>
LK_ROLE_PIPELINE_DEVICE void tmem_relinquish_alloc_permit() {
  static_assert(CtaGroup == 1 || CtaGroup == 2,
                "TMEM allocation requires CTA group 1 or 2");
  if constexpr (CtaGroup == 1) {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;" ::
                     : "memory");
  } else {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::
                     : "memory");
  }
}

template <int CtaGroup>
LK_ROLE_PIPELINE_DEVICE void tmem_dealloc(uint32_t addr, int ncols) {
  static_assert(CtaGroup == 1 || CtaGroup == 2,
                "TMEM allocation requires CTA group 1 or 2");
  if constexpr (CtaGroup == 1) {
    asm volatile(
        "tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" ::"r"(addr),
        "r"(ncols));
  } else {
    asm volatile(
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(addr),
        "r"(ncols));
  }
}

template <int CtaGroup> LK_ROLE_PIPELINE_DEVICE void cluster_sync() {
  static_assert(CtaGroup == 1 or CtaGroup == 2,
                "cluster synchronization requires CTA group 1 or 2");
  if constexpr (CtaGroup == 2) {
    asm volatile("barrier.cluster.arrive;\nbarrier.cluster.wait;\n" ::
                     : "memory");
  } else {
    __syncthreads();
  }
}

LK_ROLE_PIPELINE_DEVICE void wait_mbarrier(uint64_t *bar, uint32_t phase) {
  uint32_t ticks = 0x989680;
  asm volatile("{\n"
               ".reg .pred P1;\n"
               "LAB_WAIT_%=:\n"
               "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2;\n"
               "@P1 bra.uni DONE_%=;\n"
               "bra.uni LAB_WAIT_%=;\n"
               "DONE_%=:\n"
               "}\n" ::"r"(smem_addr(bar)),
               "r"(phase), "r"(ticks));
}

LK_ROLE_PIPELINE_DEVICE void arrive_expect_tx(uint64_t *bar,
                                              uint32_t tx_bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(
                   smem_addr(bar)),
               "r"(tx_bytes)
               : "memory");
}

LK_ROLE_PIPELINE_DEVICE void arrive_mbarrier(uint64_t *bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(smem_addr(bar))
               : "memory");
}

LK_ROLE_PIPELINE_DEVICE void arrive_cluster(uint64_t *bar,
                                            uint32_t target_cta) {
  uint32_t remote_addr;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
               : "=r"(remote_addr)
               : "r"(smem_addr(bar)), "r"(target_cta));
  asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];" ::"r"(remote_addr)
               : "memory");
}

LK_ROLE_PIPELINE_DEVICE void arrive_mbarrier_multicast(uint64_t *bar,
                                                       uint16_t mask) {
  uint32_t self = cluster_rank();
#pragma unroll
  for (uint32_t cta = 0; cta < 16; ++cta) {
    if ((mask >> cta) & 1u) {
      if (cta == self) {
        arrive_mbarrier(bar);
      } else {
        arrive_cluster(bar, cta);
      }
    }
  }
}

template <int CtaGroup>
LK_ROLE_PIPELINE_DEVICE void umma_commit(uint64_t *bar, bool issue_pred,
                                         uint16_t multicast_mask = 0) {
  uint32_t elected = pipeline_elect_one();
  if (!issue_pred or !elected) {
    return;
  }
  if constexpr (CtaGroup == 2) {
    uint16_t mask = multicast_mask != 0 ? multicast_mask
                                        : cta_group_multicast_mask<CtaGroup>();
    asm volatile("tcgen05.commit.cta_group::2"
                 ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
                 " [%0], %1;" ::"r"(smem_addr(bar)),
                 "h"(mask)
                 : "memory");
  } else {
    if (multicast_mask != 0) {
      asm volatile(
          "tcgen05.commit.cta_group::1"
          ".mbarrier::arrive::one.shared::cluster.multicast::cluster.b64"
          " [%0], %1;" ::"r"(smem_addr(bar)),
          "h"(multicast_mask)
          : "memory");
    } else {
      asm volatile("tcgen05.commit.cta_group::1"
                   ".mbarrier::arrive::one.shared::cluster.b64"
                   " [%0];" ::"r"(smem_addr(bar))
                   : "memory");
    }
  }
}

LK_ROLE_PIPELINE_DEVICE void tcgen05_after_thread_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
}

LK_ROLE_PIPELINE_DEVICE void tcgen05_before_thread_sync() {
  asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
}

LK_ROLE_PIPELINE_DEVICE void async_proxy_fence() {
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}

LK_ROLE_PIPELINE_DEVICE void tma_store_fence() { async_proxy_fence(); }

LK_ROLE_PIPELINE_DEVICE void tma_store_commit() {
  asm volatile("cp.async.bulk.commit_group;" ::: "memory");
}

template <int Pending> LK_ROLE_PIPELINE_DEVICE void tma_store_wait_group() {
  static_assert(Pending >= 0 && Pending <= 7,
                "cp.async.bulk.wait_group supports 0..7");
  if constexpr (Pending == 0) {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
  } else if constexpr (Pending == 1) {
    asm volatile("cp.async.bulk.wait_group 1;\n" ::: "memory");
  } else if constexpr (Pending == 2) {
    asm volatile("cp.async.bulk.wait_group 2;\n" ::: "memory");
  } else if constexpr (Pending == 3) {
    asm volatile("cp.async.bulk.wait_group 3;\n" ::: "memory");
  } else if constexpr (Pending == 4) {
    asm volatile("cp.async.bulk.wait_group 4;\n" ::: "memory");
  } else if constexpr (Pending == 5) {
    asm volatile("cp.async.bulk.wait_group 5;\n" ::: "memory");
  } else if constexpr (Pending == 6) {
    asm volatile("cp.async.bulk.wait_group 6;\n" ::: "memory");
  } else {
    asm volatile("cp.async.bulk.wait_group 7;\n" ::: "memory");
  }
}

template <int Pending>
LK_ROLE_PIPELINE_DEVICE void tma_store_wait_group_read() {
  static_assert(Pending >= 0 && Pending <= 7,
                "cp.async.bulk.wait_group supports 0..7");
  if constexpr (Pending == 0) {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
  } else if constexpr (Pending == 1) {
    asm volatile("cp.async.bulk.wait_group.read 1;\n" ::: "memory");
  } else if constexpr (Pending == 2) {
    asm volatile("cp.async.bulk.wait_group.read 2;\n" ::: "memory");
  } else if constexpr (Pending == 3) {
    asm volatile("cp.async.bulk.wait_group.read 3;\n" ::: "memory");
  } else if constexpr (Pending == 4) {
    asm volatile("cp.async.bulk.wait_group.read 4;\n" ::: "memory");
  } else if constexpr (Pending == 5) {
    asm volatile("cp.async.bulk.wait_group.read 5;\n" ::: "memory");
  } else if constexpr (Pending == 6) {
    asm volatile("cp.async.bulk.wait_group.read 6;\n" ::: "memory");
  } else {
    asm volatile("cp.async.bulk.wait_group.read 7;\n" ::: "memory");
  }
}

LK_ROLE_PIPELINE_DEVICE void tmem_store_wait() {
  asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
}

struct TmaStoreStream {
  int pending_ = 0;

  LK_ROLE_PIPELINE_DEVICE int pending() const { return pending_; }

  LK_ROLE_PIPELINE_DEVICE void commit() {
    tma_store_commit();
    ++pending_;
  }

  template <int Keep> LK_ROLE_PIPELINE_DEVICE void wait() {
    tma_store_wait_group_read<Keep>();
    __syncwarp();
    pending_ = Keep;
  }

  LK_ROLE_PIPELINE_DEVICE void tail() {
    if (pending_ != 0) {
      tma_store_wait_group<0>();
      pending_ = 0;
    }
  }
};

template <int Id, int Count> LK_ROLE_PIPELINE_DEVICE void named_barrier_sync() {
  asm volatile("bar.sync %0, %1;" ::"n"(Id), "n"(Count) : "memory");
}

// Two roles, one ring of Stages slots. Each role has its own cursor; the
// barrier arrays are named by who arrives, and indexed by the waiter's slot.
//
//   producer_stage_                              consumer_stage_
//         |                                              |
//         v                                              v
//   +-----------+    +-----------+    +-----------+    +-----------+
//   | slot 0    |    | slot 1    |    | slot 2    |    | ...       |
//   | producer_ |    | producer_ |    | producer_ |    | producer_ |  data
//   ready | consumer_ |    | consumer_ |    | consumer_ |    | consumer_ | slot
//   free
//   +-----------+    +-----------+    +-----------+    +-----------+
//
//   producer_acquire  wait  consumer_[producer_stage_]  // this slot is free
//   producer_commit   arrive producer_[producer_stage_] // data in this slot is
//   ready consumer_wait     wait  producer_[consumer_stage_]  // data in this
//   slot is ready consumer_release  arrive consumer_[consumer_stage_] // this
//   slot is free
template <int Stages> struct PipelineState {
  static_assert(Stages > 0, "pipeline stages must be positive");

  int producer_stage_ = 0;
  int consumer_stage_ = 0;
  uint32_t producer_phase_ = 1;
  uint32_t consumer_phase_ = 0;

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const { return producer_stage_; }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const { return consumer_stage_; }

  LK_ROLE_PIPELINE_DEVICE void advance_producer() {
    if (++producer_stage_ == Stages) {
      producer_stage_ = 0;
      producer_phase_ ^= 1;
    }
  }

  LK_ROLE_PIPELINE_DEVICE void advance_consumer() {
    if (++consumer_stage_ == Stages) {
      consumer_stage_ = 0;
      consumer_phase_ ^= 1;
    }
  }
};

struct PipelineBarrierArray {
  uint64_t *base_;

  LK_ROLE_PIPELINE_DEVICE explicit PipelineBarrierArray(uint64_t *base)
      : base_(base) {}

  LK_ROLE_PIPELINE_DEVICE uint64_t *operator[](int stage) const {
    return base_ + stage;
  }
};

template <int Stages, uint32_t TxBytes, uint32_t ProducerCount = 1,
          uint32_t ConsumerCount = 1, int CtaGroup = 1,
          bool ClusterMulticast = false, uint16_t ConsumerMask = 0>
struct PipelineTmaUmma {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;
  uint16_t consumer_mask_;
  bool producer_prepared_ = false;

  LK_ROLE_PIPELINE_DEVICE PipelineTmaUmma(uint64_t *producer,
                                          uint64_t *consumer,
                                          uint16_t consumer_mask = ConsumerMask)
      : producer_(producer), consumer_(consumer),
        consumer_mask_(consumer_mask) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_prepare_transaction() {
    if (!producer_prepared_) {
      if constexpr (CtaGroup == 2) {
        // cta_group::2 TMA clears the peer bit on every participant's local
        // barrier address, so complete_tx always targets the group leader.
        if (cta_group_is_leader<CtaGroup>()) {
          arrive_expect_tx(producer_[state_.producer_stage_], TxBytes);
        } else if constexpr (ProducerCount == 2) {
          // Plain cg2 (or mixed plain+multicast) needs the nonleader arrive.
          // Pure multicast uses ProducerCount=1 and skips this path.
          arrive_cluster(producer_[state_.producer_stage_],
                         cta_group_leader_rank<CtaGroup>());
        }
      } else {
        arrive_expect_tx(producer_[state_.producer_stage_], TxBytes);
      }
      producer_prepared_ = true;
    }
  }

  LK_ROLE_PIPELINE_DEVICE uint64_t *producer_get_barrier() const {
    return producer_[state_.producer_stage_];
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
    producer_prepared_ = false;
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    __syncwarp();
    uint32_t elected = pipeline_elect_one();
    if (elected && !producer_prepared_) {
      if constexpr (CtaGroup == 2) {
        if (cta_group_is_leader<CtaGroup>()) {
          arrive_mbarrier(producer_[state_.producer_stage_]);
        } else if constexpr (ProducerCount == 2) {
          arrive_cluster(producer_[state_.producer_stage_],
                         cta_group_leader_rank<CtaGroup>());
        }
      } else {
        arrive_mbarrier(producer_[state_.producer_stage_]);
      }
    }
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    umma_commit<CtaGroup>(consumer_[state_.consumer_stage_], true,
                          ClusterMulticast ? consumer_mask_ : uint16_t(0));
    state_.advance_consumer();
  }
};

template <int Stages, uint32_t TxBytes, uint32_t ProducerCount = 1,
          uint32_t ConsumerCount = 1, int CtaGroup = 1,
          bool ClusterMulticast = false, uint16_t ConsumerMask = 0,
          int ReleaseTid = 0>
struct PipelineTmaGeneric {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;
  uint16_t consumer_mask_;
  bool producer_prepared_ = false;

  LK_ROLE_PIPELINE_DEVICE
  PipelineTmaGeneric(uint64_t *producer, uint64_t *consumer,
                     uint16_t consumer_mask = ConsumerMask)
      : producer_(producer), consumer_(consumer),
        consumer_mask_(consumer_mask) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    static_assert(ProducerCount > 0 && ConsumerCount > 0,
                  "mbarrier arrival counts must be positive");
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_prepare_transaction() {
    if (!producer_prepared_) {
      if constexpr (CtaGroup == 2) {
        if (cta_group_is_leader<CtaGroup>()) {
          arrive_expect_tx(producer_[state_.producer_stage_], TxBytes);
        } else if constexpr (ProducerCount == 2) {
          arrive_cluster(producer_[state_.producer_stage_],
                         cta_group_leader_rank<CtaGroup>());
        }
      } else {
        arrive_expect_tx(producer_[state_.producer_stage_], TxBytes);
      }
      producer_prepared_ = true;
    }
  }

  LK_ROLE_PIPELINE_DEVICE uint64_t *producer_get_barrier() const {
    return producer_[state_.producer_stage_];
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
    producer_prepared_ = false;
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    __syncwarp();
    uint32_t elected = pipeline_elect_one();
    if (elected && !producer_prepared_) {
      if constexpr (CtaGroup == 2) {
        if (cta_group_is_leader<CtaGroup>()) {
          arrive_mbarrier(producer_[state_.producer_stage_]);
        } else if constexpr (ProducerCount == 2) {
          arrive_cluster(producer_[state_.producer_stage_],
                         cta_group_leader_rank<CtaGroup>());
        }
      } else {
        arrive_mbarrier(producer_[state_.producer_stage_]);
      }
    }
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    if constexpr (CtaGroup == 2) {
      // complete_tx lands on the pair leader. Wake the peer's local producer_
      // so every destination CTA can wait locally.
      if (cta_group_is_leader<CtaGroup>() && threadIdx.x == ReleaseTid) {
        arrive_cluster(producer_[state_.consumer_stage_],
                       cta_group_leader_rank<CtaGroup>() + 1);
      }
    }
    async_proxy_fence();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    if constexpr (ClusterMulticast || CtaGroup == 2) {
      uint16_t mask = ClusterMulticast ? consumer_mask_
                                       : cta_group_multicast_mask<CtaGroup>();
      arrive_mbarrier_multicast(consumer_[state_.consumer_stage_], mask);
    } else {
      arrive_mbarrier(consumer_[state_.consumer_stage_]);
    }
    state_.advance_consumer();
  }
};

template <int Stages, uint32_t TxBytes, uint32_t ProducerCount = 1>
struct PipelineTmaTmaStoreSameRole {
  static_assert(Stages > 0 && Stages <= 8,
                "TMA store pipeline stages must be in [1, 8]");

  struct State {
    int stage = 0;
    uint32_t phase = 0;
  } state_;
  PipelineBarrierArray producer_;
  bool producer_prepared_ = false;

  LK_ROLE_PIPELINE_DEVICE explicit PipelineTmaTmaStoreSameRole(
      uint64_t *producer)
      : producer_(producer) {}

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const { return state_.stage; }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const { return state_.stage; }

  LK_ROLE_PIPELINE_DEVICE void producer_prepare_transaction() {
    if (!producer_prepared_) {
      arrive_expect_tx(producer_[state_.stage], TxBytes);
      producer_prepared_ = true;
    }
  }

  LK_ROLE_PIPELINE_DEVICE uint64_t *producer_get_barrier() const {
    return producer_[state_.stage];
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    tma_store_wait_group_read<Stages - 1>();
    __syncwarp();
    producer_prepared_ = false;
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    __syncwarp();
    if (pipeline_elect_one() && !producer_prepared_) {
      arrive_mbarrier(producer_[state_.stage]);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.stage], state_.phase);
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    tma_store_commit();
    if (++state_.stage == Stages) {
      state_.stage = 0;
      state_.phase ^= 1;
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_tail() { tma_store_wait_group<0>(); }
};

template <int Stages, uint32_t TxBytes, uint32_t ProducerCount = 1,
          uint32_t ConsumerCount = 1>
struct PipelineTmaTmaStore {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;
  bool producer_prepared_ = false;
  int inflight_ = 0;

  LK_ROLE_PIPELINE_DEVICE PipelineTmaTmaStore(uint64_t *producer,
                                              uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_prepare_transaction() {
    if (!producer_prepared_) {
      arrive_expect_tx(producer_[state_.producer_stage_], TxBytes);
      producer_prepared_ = true;
    }
  }

  LK_ROLE_PIPELINE_DEVICE uint64_t *producer_get_barrier() const {
    return producer_[state_.producer_stage_];
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
    producer_prepared_ = false;
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    __syncwarp();
    if (pipeline_elect_one() && !producer_prepared_) {
      arrive_mbarrier(producer_[state_.producer_stage_]);
    }
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    // Both sides use the async proxy, so the TMA-load completion barrier is
    // sufficient before the TMA store reads this stage.
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE bool consumer_full() const {
    return inflight_ == Stages;
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    ++inflight_;
    state_.advance_consumer();
  }

  LK_ROLE_PIPELINE_DEVICE void retire_completed() {
    int completion_stage = state_.consumer_stage_ - inflight_;
    if (completion_stage < 0) {
      completion_stage += Stages;
    }
    if (pipeline_elect_one()) {
      arrive_mbarrier(consumer_[completion_stage]);
    }
    --inflight_;
  }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1,
          int CtaGroup = 2>
struct PipelineUmmaTmemLoad {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;

  LK_ROLE_PIPELINE_DEVICE PipelineUmmaTmemLoad(uint64_t *producer,
                                               uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    umma_commit<CtaGroup>(producer_[state_.producer_stage_], true);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    tcgen05_before_thread_sync();
    if constexpr (CtaGroup == 2) {
      arrive_cluster(consumer_[state_.consumer_stage_],
                     cta_group_leader_rank<CtaGroup>());
    } else {
      arrive_mbarrier(consumer_[state_.consumer_stage_]);
    }
    __syncwarp();
    state_.advance_consumer();
  }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1,
          int CtaGroup = 1>
struct PipelineUmmaUmma {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;

  LK_ROLE_PIPELINE_DEVICE PipelineUmmaUmma(uint64_t *producer,
                                           uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    umma_commit<CtaGroup>(producer_[state_.producer_stage_], true);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    umma_commit<CtaGroup>(consumer_[state_.consumer_stage_], true);
    state_.advance_consumer();
  }
};

template <int Stages, int Threads = 128, int BarrierId = 1>
struct PipelineTmaStoreSameRole {
  static_assert(Stages > 0, "pipeline stages must be positive");

  struct State {
    int stage = 0;
    int inflight = 0;
  } state_;

  LK_ROLE_PIPELINE_DEVICE void init() {}
  LK_ROLE_PIPELINE_DEVICE int producer_stage() const { return state_.stage; }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const { return state_.stage; }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    // Join the role after the store warp releases the reusable stage.
    named_barrier_sync<BarrierId, Threads>();
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    // Publish local SMEM writes to the TMA async proxy, then wait for all
    // writers.
    tma_store_fence();
    named_barrier_sync<BarrierId, Threads>();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    // Same role reaches tma_st after producer_commit in program order.
  }

  LK_ROLE_PIPELINE_DEVICE bool consumer_full() const {
    return state_.inflight == Stages;
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    ++state_.inflight;
    if (++state_.stage == Stages) {
      state_.stage = 0;
    }
  }

  LK_ROLE_PIPELINE_DEVICE void retire_completed() { --state_.inflight; }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1>
struct PipelineTmaStoreCrossRole {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;
  int inflight_ = 0;

  LK_ROLE_PIPELINE_DEVICE PipelineTmaStoreCrossRole(uint64_t *producer,
                                                    uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    // Producer barrier means SMEM writes are visible to the async proxy and the
    // stage is ready.
    tma_store_fence();
    arrive_mbarrier(producer_[state_.producer_stage_]);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE bool consumer_full() const {
    return inflight_ == Stages;
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    ++inflight_;
    state_.advance_consumer();
  }

  LK_ROLE_PIPELINE_DEVICE void retire_completed() {
    int completion_stage = state_.consumer_stage_ - inflight_;
    if (completion_stage < 0) {
      completion_stage += Stages;
    }
    if (pipeline_elect_one()) {
      arrive_mbarrier(consumer_[completion_stage]);
    }
    --inflight_;
  }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1,
          int CtaGroup = 1>
struct PipelineTmemStoreUmma {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;

  LK_ROLE_PIPELINE_DEVICE PipelineTmemStoreUmma(uint64_t *producer,
                                                uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    tmem_store_wait();
    arrive_mbarrier(producer_[state_.producer_stage_]);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    umma_commit<CtaGroup>(consumer_[state_.consumer_stage_], true);
    state_.advance_consumer();
  }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1,
          int CtaGroup = 1>
struct PipelineGenericUmma {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;

  LK_ROLE_PIPELINE_DEVICE PipelineGenericUmma(uint64_t *producer,
                                              uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    async_proxy_fence();
    arrive_mbarrier(producer_[state_.producer_stage_]);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    umma_commit<CtaGroup>(consumer_[state_.consumer_stage_], true);
    state_.advance_consumer();
  }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1>
struct PipelineTmemStoreTmemLoad {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;

  LK_ROLE_PIPELINE_DEVICE PipelineTmemStoreTmemLoad(uint64_t *producer,
                                                    uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    tmem_store_wait();
    arrive_mbarrier(producer_[state_.producer_stage_]);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
    tcgen05_after_thread_sync();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    tcgen05_before_thread_sync();
    arrive_mbarrier(consumer_[state_.consumer_stage_]);
    __syncwarp();
    state_.advance_consumer();
  }
};

template <int Stages, uint32_t ProducerCount = 1, uint32_t ConsumerCount = 1>
struct PipelineGenericGeneric {
  PipelineState<Stages> state_;
  PipelineBarrierArray producer_;
  PipelineBarrierArray consumer_;

  LK_ROLE_PIPELINE_DEVICE PipelineGenericGeneric(uint64_t *producer,
                                                 uint64_t *consumer)
      : producer_(producer), consumer_(consumer) {}

  LK_ROLE_PIPELINE_DEVICE int producer_stage() const {
    return state_.producer_stage();
  }
  LK_ROLE_PIPELINE_DEVICE int consumer_stage() const {
    return state_.consumer_stage();
  }

  LK_ROLE_PIPELINE_DEVICE void init() {
    for (int s = 0; s < Stages; ++s) {
      init_mbarrier(producer_[s], ProducerCount);
      init_mbarrier(consumer_[s], ConsumerCount);
    }
  }

  LK_ROLE_PIPELINE_DEVICE void producer_acquire() {
    wait_mbarrier(consumer_[state_.producer_stage_], state_.producer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE void producer_commit() {
    arrive_mbarrier(producer_[state_.producer_stage_]);
    state_.advance_producer();
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_wait() {
    wait_mbarrier(producer_[state_.consumer_stage_], state_.consumer_phase_);
  }

  LK_ROLE_PIPELINE_DEVICE void consumer_release() {
    arrive_mbarrier(consumer_[state_.consumer_stage_]);
    state_.advance_consumer();
  }
};

} // namespace role_pipeline
} // namespace little_kernel
