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
"""Host-side unit tests for the NV GEMM scheduler atoms.

These tests exercise the Python scheduling state machine that the special
struct codegen translates to C++ device code. The ``Scheduler`` methods under
test here (``_get_next_k_group`` and the ``GemmType.KGroupedContiguous``
branches of ``__init__``/``get_next_block``) are pure integer dataflow and run
host-side without a GPU; swizzle intrinsics and TMA machinery are not covered.
"""

import ast
import pathlib
import unittest

from little_kernel.backends.nv.common.atom import scheduler as scheduler_module
from little_kernel.backends.nv.common.atom.scheduler import (
    GemmType,
    Scheduler,
    get_tile_coord_along_m,
    get_tile_coord_along_n,
    get_tile_coord_grouped_m,
)


class TestTileCoordinateMappings(unittest.TestCase):

    def test_tile_coordinate_mappings(self) -> None:
        assert [get_tile_coord_along_m(i, 3) for i in range(6)] == [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1)]
        assert [get_tile_coord_along_n(i, 2) for i in range(6)] == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
        assert [get_tile_coord_grouped_m(i, 5, 2, 2) for i in range(10)] == [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (2, 0),
            (3, 0),
            (2, 1),
            (3, 1),
            (4, 0),
            (4, 1),
        ]


def _make_kgroup_stub(grouped_layout, shape_k=1024):
    """Build a bare Scheduler instance with just the K-group state fields.

    ``Scheduler.__new__`` is used (per the TMADescriptorHelper test precedent)
    because the full ``__init__`` template path requires compile-time builtins
    that are deliberately not callable outside compilation.
    """
    sched = Scheduler.__new__(Scheduler)
    sched.gemm_type = GemmType.KGroupedContiguous
    sched.num_groups = len(grouped_layout)
    sched.grouped_layout = grouped_layout
    sched.current_group_idx = 0
    sched.current_shape_k = shape_k
    sched.next_group_idx = 0
    sched.next_shape_k = 0
    sched.current_k_cumsum = 0
    sched.current_sf_k_cumsum = 0
    sched.current_num_valid_groups = 0
    return sched


def _kgrouped_branch_body(function_name, keep):
    """Compile the statements of the ``GemmType.KGroupedContiguous`` branch of
    a Scheduler method into an executable function of ``(self, **extra)``.

    The branch is located structurally in the real module source so the test
    executes the same statements that codegen translates; only the trailing
    statements listed by ``keep`` filtering are excluded.
    """
    module_path = pathlib.Path(scheduler_module.__file__).resolve()
    tree = ast.parse(module_path.read_text())
    method = None
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            method = node
            break
    if method is None:
        raise AssertionError(f"{function_name} not found in {module_path}")
    for stmt in ast.walk(method):
        if not isinstance(stmt, ast.If):
            continue
        # Match the elif/if chain entry guarding on KGroupedContiguous.
        if "KGroupedContiguous" not in ast.dump(stmt.test):
            continue
        body = [s for s in stmt.body if keep(s)] or [ast.Pass()]
        func = ast.FunctionDef(
            name="_run_kgrouped_branch",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg="self"), ast.arg(arg="next_block_idx")],
                kwonlyargs=[],
                kw_defaults=[],
                vararg=None,
                kwarg=None,
                defaults=[],
            ),
            body=body,
            decorator_list=[],
        )
        module = ast.fix_missing_locations(ast.Module(body=[func], type_ignores=[]))
        ns = {}
        exec(compile(module, filename=f"<scheduler.{function_name}.KGroupedContiguous>", mode="exec"), ns)
        return ns["_run_kgrouped_branch"]
    raise AssertionError(f"KGroupedContiguous branch not found in Scheduler.{function_name}")


class TestGetNextKGroup(unittest.TestCase):
    """Direct out-parameter contract of Scheduler._get_next_k_group."""

    def test_first_valid_group_returned_when_no_skip_needed(self):
        sched = _make_kgroup_stub([64, 128])
        group_idx, shape_k = sched._get_next_k_group(0, 0)
        self.assertEqual(group_idx, 0)
        self.assertEqual(shape_k, 64)

    def test_skips_zero_k_groups(self):
        sched = _make_kgroup_stub([0, 64, 0, 128])
        group_idx, shape_k = sched._get_next_k_group(0, 0)
        self.assertEqual(group_idx, 1)
        self.assertEqual(shape_k, 64)

    def test_start_index_is_respected(self):
        sched = _make_kgroup_stub([64, 128])
        group_idx, shape_k = sched._get_next_k_group(1, 64)
        self.assertEqual(group_idx, 1)
        self.assertEqual(shape_k, 128)

    def test_exhaustion_keeps_group_idx_at_num_groups(self):
        # With no positive-K group at or after the start index, group_idx
        # saturates at num_groups (C++ reference-loop termination semantics).
        sched = _make_kgroup_stub([0, 0])
        group_idx, _ = sched._get_next_k_group(0, 0)
        self.assertEqual(group_idx, 2)

    def test_result_must_be_assigned_by_caller(self):
        # Regression guard for the original defect: the method mutates only its
        # locals, so the caller-visible state must be wired through the return
        # value. The returned tuple carries the updated state.
        sched = _make_kgroup_stub([64, 128])
        result = sched._get_next_k_group(0, 0)
        self.assertIsNotNone(result)
        self.assertEqual(result, (0, 64))


class TestKGroupedContiguousInit(unittest.TestCase):
    """The __init__ KGroupedContiguous branch must initialize the lookahead
    state from grouped_layout through the _get_next_k_group contract."""

    def _run_init_branch(self, grouped_layout, shape_k=1024):
        sched = _make_kgroup_stub(grouped_layout, shape_k)
        run = _kgrouped_branch_body("__init__", keep=lambda s: True)
        run(sched, 0)
        return sched

    def test_init_assigns_current_and_next_k_group(self):
        sched = self._run_init_branch([64, 128])
        self.assertEqual(sched.current_group_idx, 0)
        self.assertEqual(sched.current_shape_k, 64)
        self.assertEqual(sched.next_group_idx, 1)
        self.assertEqual(sched.next_shape_k, 128)

    def test_init_skips_zero_sized_groups(self):
        sched = self._run_init_branch([0, 0, 64, 0, 128])
        self.assertEqual(sched.current_group_idx, 2)
        self.assertEqual(sched.current_shape_k, 64)
        self.assertEqual(sched.next_group_idx, 4)
        self.assertEqual(sched.next_shape_k, 128)


class TestKGroupedContiguousAdvance(unittest.TestCase):
    """Stepping past a K group's blocks must advance K cumsums and shape_k
    consistently, which is what get_global_idx(IndexType.K/SF_K) consumes."""

    # Drops the trailing get_swizzled_block_idx statements: the K/SF_K state
    # under test has already been updated when the loop breaks.
    @staticmethod
    def _is_advance_statement(stmt):
        if isinstance(stmt, ast.Assign):
            text = ast.dump(stmt.value)
            return "get_swizzled_block_idx" not in text
        return True

    def _run_advance(self, grouped_layout, next_block_idx, num_m_blocks=2, num_n_blocks=3):
        sched = _make_kgroup_stub(grouped_layout)
        sched.num_m_blocks = num_m_blocks
        sched.num_n_blocks = num_n_blocks
        sched.sf_k_alignment = 512
        # Prime the lookahead as __init__ does.
        init_run = _kgrouped_branch_body("__init__", keep=lambda s: True)
        init_run(sched, 0)
        run = _kgrouped_branch_body("get_next_block", keep=self._is_advance_statement)
        try:
            run(sched, next_block_idx)
        except StopIteration:  # pragma: no cover - defensive
            self.fail("advance loop did not break")
        return sched

    def test_step_past_first_group_advances_k_state(self):
        # [64, 128] with 2*3 = 6 blocks per group: block index 6 is the first
        # block of the second group.
        sched = self._run_advance([64, 128], next_block_idx=6)
        self.assertEqual(sched.current_k_cumsum, 64)
        self.assertEqual(sched.current_sf_k_cumsum, 1)  # cdiv(64, 512)
        self.assertEqual(sched.current_group_idx, 1)
        self.assertEqual(sched.current_shape_k, 128)
        self.assertEqual(sched.current_num_valid_groups, 1)

    def test_block_within_first_group_does_not_advance(self):
        sched = self._run_advance([64, 128], next_block_idx=5)
        self.assertEqual(sched.current_k_cumsum, 0)
        self.assertEqual(sched.current_shape_k, 64)
        self.assertEqual(sched.current_num_valid_groups, 0)

    def test_step_past_both_groups_advances_twice(self):
        # Block index 12 is past both 6-block groups; second group's K (128)
        # adds two 512-aligned SF chunks (cdiv(128, 512) = 1 each advance).
        sched = self._run_advance([64, 128], next_block_idx=12)
        self.assertEqual(sched.current_k_cumsum, 64 + 128)
        self.assertEqual(sched.current_sf_k_cumsum, 1 + 1)
        self.assertEqual(sched.current_num_valid_groups, 2)

    def test_zero_k_group_in_the_middle_is_skipped(self):
        # Group 1 has K == 0 and contributes no blocks; block 6 is the first
        # block of group 2.
        sched = self._run_advance([64, 0, 128], next_block_idx=6)
        self.assertEqual(sched.current_k_cumsum, 64)
        self.assertEqual(sched.current_group_idx, 2)
        self.assertEqual(sched.current_shape_k, 128)


if __name__ == "__main__":
    unittest.main()
