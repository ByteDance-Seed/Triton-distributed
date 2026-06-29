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

import os
import json
import glob
import warnings
import torch
import triton
import triton.language as tl

# Import directly from triton_dist.jit to avoid language module issues
from triton_dist.jit import jit, _parse_cga_cluster_size


def test_parse_cga_cluster_size():
    """Test the _parse_cga_cluster_size function directly."""
    print("[test_parse_cga_cluster_size] start...")

    # Valid cases: power of 2 dimensions, total <= 16
    valid_cases = [("1", (1, 1, 1)), ("2", (2, 1, 1)), ("4", (4, 1, 1)), ("8", (8, 1, 1)), ("16", (16, 1, 1)),
                   ("2,1,1", (2, 1, 1)), ("4,2,1", (4, 2, 1)), ("2,2", (2, 2, 1)), ("4,4,1", (4, 4, 1)),  # total = 16
                   ("2,2,2", (2, 2, 2)),  # total = 8
                   ("4,2,2", (4, 2, 2)),  # total = 16
                   ("2,4,2", (2, 4, 2)),  # total = 16
                   ]

    original_env = os.environ.get("TRITON_DIST_CGA_CLUSTER_SIZE")

    for env_value, expected in valid_cases:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = env_value
        result = _parse_cga_cluster_size()
        assert result == expected, f"Failed for '{env_value}': got {result}, expected {expected}"

    # Invalid cases: format errors
    format_invalid_cases = ["", "abc", "1,2,3,4"]
    for env_value in format_invalid_cases:
        if env_value == "":
            if "TRITON_DIST_CGA_CLUSTER_SIZE" in os.environ:
                del os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"]
        else:
            os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = env_value
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = _parse_cga_cluster_size()
        assert result is None, f"Failed for invalid '{env_value}': got {result}, expected None"

    # Invalid cases: not power of 2
    not_power_of_2_cases = ["3", "5", "6", "7", "3,1,1", "2,3,1", "2,2,3"]
    for env_value in not_power_of_2_cases:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = env_value
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _parse_cga_cluster_size()
        assert result is None, f"Should reject non-power-of-2 '{env_value}': got {result}"
        assert any("not a power of 2" in str(warning.message) for warning in w), \
            f"Should warn about non-power-of-2 for '{env_value}'"

    # Invalid cases: total exceeds 16
    exceeds_16_cases = ["32", "4,4,2", "8,4,1", "2,2,8", "4,8,1"]
    for env_value in exceeds_16_cases:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = env_value
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = _parse_cga_cluster_size()
        assert result is None, f"Should reject total>16 '{env_value}': got {result}"
        assert any("exceeds maximum of 16" in str(warning.message) for warning in w), \
            f"Should warn about exceeding 16 for '{env_value}'"

    # Restore original env
    if original_env is not None:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = original_env
    elif "TRITON_DIST_CGA_CLUSTER_SIZE" in os.environ:
        del os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"]

    print("✅ [test_parse_cga_cluster_size] done...")


def test_cga_cluster_size_env():
    """Test that TRITON_DIST_CGA_CLUSTER_SIZE sets cluster_dims correctly."""
    print("[test_cga_cluster_size_env] start...")

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    if capability[0] < 9:
        print("⚠️  [test_cga_cluster_size_env] skipped (requires SM90+)")
        return

    # Save original env
    original_env = os.environ.get("TRITON_DIST_CGA_CLUSTER_SIZE")

    # Set env variable
    os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = "2,1,1"

    @jit
    def simple_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, pid, mask=mask)

    n_elements = 10240
    BLOCK_SIZE = 128
    output = torch.zeros(n_elements, dtype=torch.int32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Launch kernel without explicit cluster_dims - should use env
    simple_kernel[grid](output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Verify from cache
    cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
    json_files = glob.glob(os.path.join(cache_dir, '**/simple_kernel.json'), recursive=True)

    assert json_files, "No kernel metadata found"

    json_files.sort(key=os.path.getmtime, reverse=True)
    with open(json_files[0], 'r') as f:
        metadata = json.load(f)

    assert metadata.get('cluster_dims') == [2, 1, 1], \
        f"cluster_dims mismatch: {metadata.get('cluster_dims')}"
    assert metadata.get('num_ctas') == 2, \
        f"num_ctas mismatch: {metadata.get('num_ctas')}"

    # Restore original env
    if original_env is not None:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = original_env
    elif "TRITON_DIST_CGA_CLUSTER_SIZE" in os.environ:
        del os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"]

    print("✅ [test_cga_cluster_size_env] done...")


def test_explicit_cluster_dims_override():
    """Test that explicit cluster_dims overrides environment variable."""
    print("[test_explicit_cluster_dims_override] start...")

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    if capability[0] < 9:
        print("⚠️  [test_explicit_cluster_dims_override] skipped (requires SM90+)")
        return

    # Save original env
    original_env = os.environ.get("TRITON_DIST_CGA_CLUSTER_SIZE")

    # Set env variable to (2,1,1)
    os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = "2,1,1"

    @jit
    def override_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, pid, mask=mask)

    n_elements = 512
    BLOCK_SIZE = 64
    output = torch.zeros(n_elements, dtype=torch.int32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Explicitly set different cluster_dims
    override_kernel[grid](
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        cluster_dims=(1, 1, 1),
        num_ctas=1,
    )

    # Verify from cache
    cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
    json_files = glob.glob(os.path.join(cache_dir, '**/override_kernel.json'), recursive=True)

    assert json_files, "No kernel metadata found"

    json_files.sort(key=os.path.getmtime, reverse=True)
    with open(json_files[0], 'r') as f:
        metadata = json.load(f)

    # Should use explicit values, not env
    assert metadata.get('cluster_dims') == [1, 1, 1], \
        f"cluster_dims should be [1,1,1], got {metadata.get('cluster_dims')}"
    assert metadata.get('num_ctas') == 1, \
        f"num_ctas should be 1, got {metadata.get('num_ctas')}"

    # Restore original env
    if original_env is not None:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = original_env
    elif "TRITON_DIST_CGA_CLUSTER_SIZE" in os.environ:
        del os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"]

    print("✅ [test_explicit_cluster_dims_override] done...")


def test_inconsistent_num_ctas_warning():
    """Test that inconsistent num_ctas triggers warning and ignores env cluster_dims."""
    print("[test_inconsistent_num_ctas_warning] start...")

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    if capability[0] < 9:
        print("⚠️  [test_inconsistent_num_ctas_warning] skipped (requires SM90+)")
        return

    # Save original env
    original_env = os.environ.get("TRITON_DIST_CGA_CLUSTER_SIZE")

    # Set env variable to (2,1,1) which implies num_ctas=2
    os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = "2,1,1"

    @jit
    def inconsistent_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, pid, mask=mask)

    n_elements = 512
    BLOCK_SIZE = 64
    output = torch.zeros(n_elements, dtype=torch.int32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # User provides num_ctas=4, inconsistent with env (2,1,1)
    warning_raised = False
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        inconsistent_kernel[grid](
            output,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
            num_ctas=4,
        )
        for warning in w:
            if "does not match" in str(warning.message):
                warning_raised = True
                break

    assert warning_raised, "Warning should be raised for inconsistent num_ctas"

    # Verify from cache - cluster_dims should NOT be from env
    cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
    json_files = glob.glob(os.path.join(cache_dir, '**/inconsistent_kernel.json'), recursive=True)

    assert json_files, "No kernel metadata found"

    json_files.sort(key=os.path.getmtime, reverse=True)
    with open(json_files[0], 'r') as f:
        metadata = json.load(f)

    # cluster_dims should NOT be [2,1,1] from env (triton defaults to [num_ctas,1,1])
    assert metadata.get('cluster_dims') != [2, 1, 1], \
        f"cluster_dims should NOT be [2,1,1] from env, got {metadata.get('cluster_dims')}"
    assert metadata.get('num_ctas') == 4, \
        f"num_ctas should be 4, got {metadata.get('num_ctas')}"

    # Restore original env
    if original_env is not None:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = original_env
    elif "TRITON_DIST_CGA_CLUSTER_SIZE" in os.environ:
        del os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"]

    print("✅ [test_inconsistent_num_ctas_warning] done...")


def test_consistent_num_ctas():
    """Test that consistent num_ctas allows env cluster_dims to be set."""
    print("[test_consistent_num_ctas] start...")

    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device)

    if capability[0] < 9:
        print("⚠️  [test_consistent_num_ctas] skipped (requires SM90+)")
        return

    # Save original env
    original_env = os.environ.get("TRITON_DIST_CGA_CLUSTER_SIZE")

    # Set env variable to (2,1,1) which implies num_ctas=2
    os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = "2,1,1"

    @jit
    def consistent_kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        tl.store(output_ptr + offsets, pid, mask=mask)

    n_elements = 512
    BLOCK_SIZE = 64
    output = torch.zeros(n_elements, dtype=torch.int32, device='cuda')
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # User provides num_ctas=2, consistent with env (2,1,1)
    consistent_kernel[grid](
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_ctas=2,
    )

    # Verify from cache
    cache_dir = os.environ.get('TRITON_CACHE_DIR', os.path.expanduser('~/.triton/cache'))
    json_files = glob.glob(os.path.join(cache_dir, '**/consistent_kernel.json'), recursive=True)

    assert json_files, "No kernel metadata found"

    json_files.sort(key=os.path.getmtime, reverse=True)
    with open(json_files[0], 'r') as f:
        metadata = json.load(f)

    # cluster_dims should be from env since num_ctas is consistent
    assert metadata.get('cluster_dims') == [2, 1, 1], \
        f"cluster_dims should be [2,1,1], got {metadata.get('cluster_dims')}"
    assert metadata.get('num_ctas') == 2, \
        f"num_ctas should be 2, got {metadata.get('num_ctas')}"

    # Restore original env
    if original_env is not None:
        os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"] = original_env
    elif "TRITON_DIST_CGA_CLUSTER_SIZE" in os.environ:
        del os.environ["TRITON_DIST_CGA_CLUSTER_SIZE"]

    print("✅ [test_consistent_num_ctas] done...")


if __name__ == "__main__":
    test_parse_cga_cluster_size()
    test_cga_cluster_size_env()
    test_explicit_cluster_dims_override()
    test_inconsistent_num_ctas_warning()
    test_consistent_num_ctas()
    print("\n✅ All CGA cluster size tests passed!")
