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
Scheduler atom for block scheduling in GEMM kernels.

This module defines the Scheduler class and its methods, which will be
translated by special_struct_materialize_pass into LittleKernel IR and embedded into
the generated CUDA code.
"""

from enum import Enum

import little_kernel.language as ll
from little_kernel.codegen.registries.special_struct_registry import register_special_struct
from little_kernel.codegen.registries.enum_registry import register_enum
from little_kernel.language.intrin.arith import cdiv
from little_kernel.language.intrin.simt import blockIdx_x


@register_enum
class IndexType(Enum):
    """Index type for scheduler get_global_idx method."""
    MN = 0
    K = 1
    SF_K = 2


@register_enum
class GemmType(Enum):
    """GEMM type for scheduler."""
    Normal = 0
    MGroupedContiguous = 1
    MGroupedMasked = 2
    KGroupedContiguous = 3
    Batched = 4


@register_special_struct()
class Scheduler:
    """
    Scheduler for block scheduling in GEMM kernels.
    
    This class defines the scheduler interface and methods that will be
    translated by special_struct_materialize_pass into LittleKernel IR.
    """

    def __init__(self, gemm_type: ll.template[GemmType], block_m: ll.template[ll.int32], block_n: ll.template[ll.int32],
                 num_groups: ll.template[ll.int32], num_multicast: ll.template[ll.int32],
                 is_multicast_on_a: ll.template[ll.bool_], num_sms: ll.template[ll.int32], shape_m, shape_n, shape_k,
                 grouped_layout: ll.ptr[ll.int32] = None, sf_k_alignment: ll.template[ll.int32] = 512):
        """
        Initialize scheduler.
        
        Args:
            gemm_type: Type of GEMM (Normal, MGroupedContiguous, etc.)
            block_m: Block size in M dimension
            block_n: Block size in N dimension
            num_groups: Number of groups
            num_multicast: Number of TMA multicast CTAs
            is_multicast_on_a: Whether multicast is on A tensor
            num_sms: Number of SMs
            shape_m: Shape in M dimension
            shape_n: Shape in N dimension
            shape_k: Shape in K dimension
            grouped_layout: Grouped layout tensor (optional)
            sf_k_alignment: SF_K alignment (default 512)
        """
        # Template parameters (compile-time constants)
        self.gemm_type = gemm_type
        self.block_m = block_m
        self.block_n = block_n
        self.num_groups = num_groups
        self.num_multicast = num_multicast
        self.is_multicast_on_a = is_multicast_on_a
        self.num_sms = num_sms
        self.sf_k_alignment = sf_k_alignment

        # Calculate num_1d_blocks_per_group
        self.num_1d_blocks_per_group = self._get_num_1d_blocks_per_group(self.is_multicast_on_a, self.block_m,
                                                                         self.block_n, self.num_sms)

        # Initialize struct members (these will be translated to C++ struct members)
        self.current_iter = -1
        self.num_blocks = 0
        self.num_m_blocks = cdiv(shape_m, block_m)
        self.num_n_blocks = cdiv(shape_n, block_n)
        self.num_blocks_in_group = 0
        self.is_peer_cta_alive = True
        self.grouped_layout = grouped_layout
        self.current_group_idx = 0
        self.current_m_cumsum = 0
        self.current_shape_k = shape_k
        self.current_num_valid_groups = 0
        self.current_k_cumsum = 0
        self.current_sf_k_cumsum = 0
        self.next_group_idx = 0
        self.next_shape_k = 0

        # Initialize based on gemm_type
        if ll.const(gemm_type == GemmType.Normal or gemm_type == GemmType.Batched):
            self.num_blocks = self.num_m_blocks * self.num_n_blocks
        elif ll.const(gemm_type == GemmType.MGroupedContiguous):
            self.num_blocks = self.num_m_blocks * self.num_n_blocks
        elif ll.const(gemm_type == GemmType.KGroupedContiguous):
            self._get_next_k_group(self.current_group_idx, self.current_shape_k)
            self.next_group_idx = self.current_group_idx + 1
            self._get_next_k_group(self.next_group_idx, self.next_shape_k)

    @staticmethod
    def _get_num_1d_blocks_per_group(is_multicast_on_a: ll.bool_, block_m: ll.int32, block_n: ll.int32,
                                     num_sms: ll.int32) -> ll.int32:
        """Calculate optimal number of 1D blocks per group."""
        num_best_blocks = 0
        min_usage: ll.uint32 = ll.uint32(0xFFFFFFFF)  # max uint32

        for candidate in [8, 16]:
            if ll.const(is_multicast_on_a):
                usage: ll.uint32 = candidate * block_n + cdiv(num_sms, candidate) * block_m
            else:
                usage: ll.uint32 = candidate * block_m + cdiv(num_sms, candidate) * block_n

            if usage < min_usage:
                min_usage = usage
                num_best_blocks = candidate

        return num_best_blocks

    def _get_next_k_group(self, group_idx, shape_k) -> None:
        """
        Get next valid K group.
        """
        while group_idx < self.num_groups:
            shape_k = self.grouped_layout[group_idx]  # __ldg will be added by codegen
            if shape_k > 0:
                break
            group_idx = group_idx + 1

    def get_swizzled_block_idx(self, block_idx):
        """
        Get swizzled block indices from block index.
        
        Returns:
            Tuple of (m_block_idx, n_block_idx)
        
        Note: In C++, this will be translated to a function that takes
        m_block_idx and n_block_idx as reference parameters and updates them in-place.
        """
        # DG_STATIC_ASSERT(kNum1DBlocksPerGroup % kNumMulticast == 0, "Invalid group size");
        # This will be checked at compile time

        # Swizzle for better L2 usages
        if self.is_multicast_on_a:
            primary_num_blocks = self.num_n_blocks
            secondary_num_blocks = self.num_m_blocks
        else:
            primary_num_blocks = self.num_m_blocks
            secondary_num_blocks = self.num_n_blocks

        num_blocks_per_group = secondary_num_blocks * self.num_1d_blocks_per_group
        group_idx = block_idx // num_blocks_per_group
        first_block_idx = group_idx * self.num_1d_blocks_per_group
        in_group_idx = block_idx % num_blocks_per_group

        # num_blocks_in_group = min(kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx)
        if self.num_1d_blocks_per_group < (primary_num_blocks - first_block_idx):
            self.num_blocks_in_group = self.num_1d_blocks_per_group
        else:
            self.num_blocks_in_group = primary_num_blocks - first_block_idx

        # Fix unaligned TMA multicast (for SM90 only)
        # This will be wrapped in #if __CUDA_ARCH__ < 1000 in codegen
        if self.num_multicast > 1 and self.num_blocks_in_group % 2 != 0:
            if in_group_idx < (self.num_blocks_in_group ^ 1) * secondary_num_blocks:
                self.num_blocks_in_group = self.num_blocks_in_group ^ 1
            else:
                in_group_idx = in_group_idx - (self.num_blocks_in_group ^ 1) * secondary_num_blocks
                first_block_idx = first_block_idx + (self.num_blocks_in_group ^ 1)
                self.num_blocks_in_group = 1

        # Convert to final M/N block indices
        if ll.const(self.is_multicast_on_a):
            m_block_idx = in_group_idx // self.num_blocks_in_group
            n_block_idx = first_block_idx + (in_group_idx % self.num_blocks_in_group)
        else:
            m_block_idx = first_block_idx + (in_group_idx % self.num_blocks_in_group)
            n_block_idx = in_group_idx // self.num_blocks_in_group
        return (m_block_idx, n_block_idx)

    def get_global_idx(self, with_group_offset: bool, shape_dim, block_size, block_idx, m_block_idx=0,
                       index_type: IndexType = IndexType.MN) -> ll.int32:
        """
        Get global index for a block.
        
        Translated from C++ template method get_global_idx.
        """
        if ll.const(self.gemm_type == GemmType.Normal):
            return block_idx * block_size
        elif ll.const(self.gemm_type == GemmType.MGroupedContiguous):
            if with_group_offset:
                # Use max(0, ...) - this will be translated to cute::max(0, ...) in codegen
                layout_val = self.grouped_layout[m_block_idx * self.block_m]  # __ldg will be added
                offset = layout_val if layout_val > 0 else 0
            else:
                offset = 0
            return offset * shape_dim + block_idx * block_size
        elif ll.const(self.gemm_type == GemmType.MGroupedMasked):
            if with_group_offset:
                offset = self.current_group_idx
            else:
                offset = 0
            return offset * shape_dim + block_idx * block_size
        elif ll.const(self.gemm_type == GemmType.KGroupedContiguous):
            offset = 0
            if with_group_offset:
                if index_type == IndexType.MN:
                    offset = self.current_group_idx * shape_dim
                elif index_type == IndexType.K:
                    offset = self.current_k_cumsum
                elif index_type == IndexType.SF_K:
                    offset = self.current_sf_k_cumsum
            return offset + block_idx * block_size
        elif ll.const(self.gemm_type == GemmType.Batched):
            # Ignore kWithGroupOffset, and apply offset for IndexType::SF_K
            if index_type == IndexType.SF_K:
                offset = self.current_group_idx
            else:
                offset = 0
            return offset * shape_dim + block_idx * block_size
        return 0

    def get_next_block(self):
        """
        Get next block to process.
        
        Translated from C++ get_next_block method.
        """
        # const auto next_block_idx = (++ current_iter) * kNumSMs + blockIdx.x;
        self.current_iter = self.current_iter + 1
        # Import blockIdx_x at runtime to avoid circular import
        next_block_idx = self.current_iter * self.num_sms + blockIdx_x()

        if ll.const(self.gemm_type == GemmType.MGroupedMasked):
            while True:
                # End of the task
                if self.current_group_idx == self.num_groups:
                    return (False, -1, -1)

                # Within current group
                from little_kernel.language.intrin.arith import cdiv
                self.num_m_blocks = cdiv(self.grouped_layout[self.current_group_idx],
                                         self.block_m)  # __ldg will be added
                current_m_block_cumsum = self.current_m_cumsum + self.num_m_blocks
                if next_block_idx < current_m_block_cumsum * self.num_n_blocks:
                    break

                # Move to check the next group
                self.current_group_idx = self.current_group_idx + 1
                self.current_m_cumsum = current_m_block_cumsum

            m_block_idx, n_block_idx = self.get_swizzled_block_idx(next_block_idx -
                                                                   self.current_m_cumsum * self.num_n_blocks)
        elif ll.const(self.gemm_type == GemmType.KGroupedContiguous):
            while True:
                # End of the task
                if self.current_group_idx == self.num_groups:
                    return (False, -1, -1)

                # Within current group
                if next_block_idx < (self.current_num_valid_groups + 1) * self.num_m_blocks * self.num_n_blocks:
                    break

                # Move to check the next group
                from little_kernel.language.intrin.arith import cdiv
                self.current_k_cumsum = self.current_k_cumsum + self.current_shape_k
                self.current_sf_k_cumsum = self.current_sf_k_cumsum + cdiv(self.current_shape_k, self.sf_k_alignment)
                self.current_num_valid_groups = self.current_num_valid_groups + 1

                self.current_group_idx = self.next_group_idx
                self.next_group_idx = self.next_group_idx + 1
                self.current_shape_k = self.next_shape_k
                self._get_next_k_group(self.next_group_idx, self.next_shape_k)

            m_block_idx, n_block_idx = self.get_swizzled_block_idx(next_block_idx - self.current_num_valid_groups *
                                                                   self.num_m_blocks * self.num_n_blocks)
        elif ll.const(self.gemm_type == GemmType.Batched):
            if next_block_idx >= self.num_blocks * self.num_groups:
                return (False, -1, -1)

            self.current_group_idx = next_block_idx // self.num_blocks
            block_idx = next_block_idx - self.current_group_idx * self.num_blocks
            if self.is_multicast_on_a:
                m_block_idx = block_idx // self.num_n_blocks
                n_block_idx = block_idx % self.num_n_blocks
            else:
                m_block_idx = block_idx % self.num_m_blocks
                n_block_idx = block_idx // self.num_m_blocks
        else:
            # Normal or MGroupedContiguous
            if next_block_idx >= self.num_blocks:
                return (False, -1, -1)

            # For SM90 only
            # NOTES: we don't have to set `is_peer_cta_alive` for masked grouped GEMM, as it must be aligned
            self.is_peer_cta_alive = ((self.num_n_blocks % self.num_multicast == 0)
                                      or (self.num_m_blocks % self.num_multicast == 0)
                                      or ((next_block_idx ^ 1) < self.num_blocks))
            m_block_idx, n_block_idx = self.get_swizzled_block_idx(next_block_idx)

        return (True, m_block_idx, n_block_idx)

    def is_tma_multicast_valid(self, m_block_idx) -> ll.bool_:
        """
        Check if TMA multicast is valid for given M block index.
        
        Translated from C++ is_tma_multicast_valid method.
        """
        if self.num_blocks_in_group == 1:
            return False

        if ll.const(self.gemm_type == GemmType.Normal or self.gemm_type == GemmType.MGroupedMasked
                    or self.gemm_type == GemmType.KGroupedContiguous or self.gemm_type == GemmType.Batched):
            return True
        else:
            # MGroupedContiguous
            if ll.const(self.is_multicast_on_a):
                return True
            else:
                group_idx = self.grouped_layout[m_block_idx * self.block_m]  # __ldg will be added
                peer_group_idx = self.grouped_layout[(m_block_idx ^ 1) * self.block_m]  # __ldg will be added
                return group_idx == peer_group_idx

    def is_computation_valid(self, m_block_idx, m_offset) -> ll.bool_:
        """
        Check if computation is valid for given M block index and offset.
        
        Translated from C++ is_computation_valid method.
        """
        if ll.const(self.gemm_type == GemmType.Normal or self.gemm_type == GemmType.Batched):
            return True
        elif ll.const(self.gemm_type == GemmType.MGroupedContiguous):
            return self.grouped_layout[m_offset + m_block_idx * self.block_m] >= 0  # __ldg will be added
        elif ll.const(self.gemm_type == GemmType.MGroupedMasked):
            return (m_offset +
                    m_block_idx * self.block_m) < self.grouped_layout[self.current_group_idx]  # __ldg will be added
        return False
