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
from cutlass.cutlass_dsl import (
    Int32,
    dsl_user_op,
    extract_mlir_values,
    new_from_mlir_values,
)
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.utils.grouped_gemm_persistent_tile_scheduler import (
    GroupSearchResult,
    GroupedGemmGroupSearchState,
    GroupedWorkTileInfo,
    StaticPersistentGroupTileScheduler,
)


class MContiguousGroupTileScheduler(StaticPersistentGroupTileScheduler):
    """Grouped-GEMM tile scheduler for the M-contiguous (shared N/K) case.

    Differences from ``StaticPersistentGroupTileScheduler``:
      * Per-group problem is a 1D ``problem_sizes_m`` (G,) Int32 tensor of
        real (un-padded) token counts; N/K/L are compile-time constants.
      * ``_get_problem_for_group`` is a single Int32 load (vs vec-4 in base).
      * ``_get_cluster_tile_count_mn`` folds N into a constexpr.
      * ``delinearize_z`` recovers ``cluster_count_m`` from search state and
        keeps N/K constexpr (no runtime ceil_div on constants).

    Host invariant: ``padded_m`` is a multiple of ``Cm`` so
    ``ceil_div(real_m, Cm) == ceil_div(padded_m, Cm)``.
    """

    def __init__(
        self,
        params: utils.PersistentTileSchedulerParams,
        num_persistent_clusters: Int32,
        current_work_linear_idx: Int32,
        cta_id_in_cluster: cute.Coord,
        num_tiles_executed: Int32,
        cluster_tile_shape_mnk: tuple,
        search_state: GroupedGemmGroupSearchState,
        group_count: int,
        problem_sizes_m: cute.Tensor,
        problem_shape_n: int,
        problem_shape_k: int,
    ):
        StaticPersistentGroupTileScheduler.__init__(
            self,
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            cluster_tile_shape_mnk,
            search_state,
            group_count,
            problem_sizes_m,
        )
        self.problem_sizes_m = problem_sizes_m
        self.problem_shape_n = problem_shape_n
        self.problem_shape_k = problem_shape_k

        self.ncluster_n = (problem_shape_n + cluster_tile_shape_mnk[1] - 1) // cluster_tile_shape_mnk[1]
        self.ncluster_k = (problem_shape_k + cluster_tile_shape_mnk[2] - 1) // cluster_tile_shape_mnk[2]

    def __extract_mlir_values__(self) -> list:
        values = extract_mlir_values(self.num_persistent_clusters)
        values.extend(extract_mlir_values(self._current_work_linear_idx))
        values.extend(extract_mlir_values(self.cta_id_in_cluster))
        values.extend(extract_mlir_values(self._num_tiles_executed))
        values.extend(extract_mlir_values(self.search_state))
        values.extend(extract_mlir_values(self.problem_sizes_m))
        values.extend(extract_mlir_values(self.params))
        return values

    def __new_from_mlir_values__(self, values: list) -> "MContiguousGroupTileScheduler":
        if len(values) < 11:
            raise ValueError("Length of mlir values extracted is incorrect.")
        new_num_persistent_clusters = new_from_mlir_values(self.num_persistent_clusters, [values[0]])
        new_current_work_linear_idx = new_from_mlir_values(self._current_work_linear_idx, [values[1]])
        new_cta_id_in_cluster = new_from_mlir_values(self.cta_id_in_cluster, values[2:5])
        new_num_tiles_executed = new_from_mlir_values(self._num_tiles_executed, [values[5]])
        search_state = new_from_mlir_values(self.search_state, values[6:10])
        problem_sizes_m = new_from_mlir_values(self.problem_sizes_m, [values[10]])
        params = new_from_mlir_values(self.params, values[11:])

        return MContiguousGroupTileScheduler(
            params,
            new_num_persistent_clusters,
            new_current_work_linear_idx,
            new_cta_id_in_cluster,
            new_num_tiles_executed,
            self.cluster_tile_shape_mnk,
            search_state,
            self.group_count,
            problem_sizes_m,
            self.problem_shape_n,
            self.problem_shape_k,
        )

    @staticmethod
    @dsl_user_op
    def create(
        params: utils.PersistentTileSchedulerParams,
        block_idx: tuple,
        grid_dim: tuple,
        cluster_tile_shape_mnk: tuple,
        initial_search_state: GroupedGemmGroupSearchState,
        group_count: int,
        problem_sizes_m: cute.Tensor,
        problem_shape_n: int,
        problem_shape_k: int,
        *,
        loc=None,
        ip=None,
    ) -> "MContiguousGroupTileScheduler":
        """Build an M-contiguous scheduler with 1D ``problem_sizes_m`` and
        constexpr ``(N, K)``. See the base class for shared arguments."""
        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(params.cluster_shape_mn, loc=loc,
                                                                                   ip=ip)

        bidx, bidy, bidz = block_idx
        current_work_linear_idx = Int32(bidz)
        cta_id_in_cluster = (
            Int32(bidx % params.cluster_shape_mn[0]),
            Int32(bidy % params.cluster_shape_mn[1]),
            Int32(0),
        )
        num_tiles_executed = Int32(0)
        return MContiguousGroupTileScheduler(
            params,
            num_persistent_clusters,
            current_work_linear_idx,
            cta_id_in_cluster,
            num_tiles_executed,
            cluster_tile_shape_mnk,
            initial_search_state,
            group_count,
            problem_sizes_m,
            problem_shape_n,
            problem_shape_k,
        )

    # Overloaded helpers: the base class's ``_group_search`` / ``_prefix_sum``
    # call into these two methods, so overriding them redirects the scheduler
    # to the 1D ``problem_sizes_m`` tensor.

    @dsl_user_op
    def _get_problem_for_group(self, problem_shape_mnkl: cute.Tensor, group_idx: Int32, *, loc=None, ip=None) -> Int32:
        """Single Int32 load of the real M for this group."""
        return Int32(problem_shape_mnkl[group_idx])

    @dsl_user_op
    def _get_cluster_tile_count_mn(self, problem_shape: Int32, *, loc=None, ip=None) -> Int32:
        """Cluster-tile count from real m; N factor is constexpr."""
        cur_ntile_m = (problem_shape + self.cluster_tile_shape_mnk[0] - 1) // self.cluster_tile_shape_mnk[0]
        return cur_ntile_m * Int32(self.ncluster_n)

    @dsl_user_op
    def delinearize_z(
        self,
        cta_tile_coord: tuple,
        *,
        loc=None,
        ip=None,
    ) -> GroupedWorkTileInfo:
        linear_idx = self._current_work_linear_idx

        self.search_state = self._group_search(
            linear_idx,
            self.problem_sizes_m,
            self.search_state.start_group_idx,
            self.search_state.tile_count_prev_group,
            loc=loc,
            ip=ip,
        )

        found = self.search_state.found
        is_valid = found
        group_idx = self.search_state.start_group_idx

        tile_count_prev_group = self.search_state.tile_count_prev_group
        cur_ntile_mn = self.search_state.tile_count_searched - tile_count_prev_group
        cluster_count_m = cur_ntile_mn // Int32(self.ncluster_n)
        cluster_count_n = Int32(self.ncluster_n)

        cluster_tile_idx_in_current_group = linear_idx - tile_count_prev_group
        cta_tile_idx_m, cta_tile_idx_n = self._compute_cta_tile_coord(
            cluster_tile_idx_in_current_group,
            cta_tile_coord,
            cluster_count_m,
            cluster_count_n,
            loc=loc,
            ip=ip,
        )

        group_search_result = GroupSearchResult(
            group_idx,
            cta_tile_idx_m,
            cta_tile_idx_n,
            cur_ntile_mn,
            Int32(self.problem_shape_n),
            Int32(self.problem_shape_k),
            Int32(self.ncluster_k),
        )
        return GroupedWorkTileInfo(cta_tile_coord, is_valid, group_search_result)
