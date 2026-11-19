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
"""Byte-buffer transport for host TMA descriptors
(semantic key ir.export.tma_descriptors.byte_buffer).

The export path moves host-encoded ``CUtensorMap`` descriptors between
Python, generated bindings and staging code as an opaque byte buffer: one
descriptor is exactly 128 Driver-owned bytes, and a per-batch bundle is the N
concatenated payloads (``N * 128`` bytes, the layout generated bindings upload
to a device ``CUtensorMap`` array).  These CPU regressions pin that transport
on the shared NV surface without touching libcuda or a GPU:

  * ``tensor_map_to_bytes`` snapshots exactly 128 bytes, independent of the
    source descriptor's lifetime and later mutation;
  * ``tensor_maps_to_bytes`` lays out a bundle with descriptor ``i`` at byte
    offset ``i * 128`` and rejects empty / non-descriptor inputs;
  * ``tensor_map_from_bytes`` rebuilds a 64-byte-aligned descriptor whose
    storage stays owned (Driver requirement) and round-trips byte-for-byte;
  * invalid types and payload lengths are rejected before any Driver call.
"""

import ctypes
import gc
import unittest

from little_kernel.backends.nv.common.runtime._tensor_map import CUtensorMap
import little_kernel.backends.nv.common.runtime.tma_descriptor as tma


def _toy_descriptor(fill: int) -> CUtensorMap:
    """An aligned CUtensorMap with a recognizable opaque pattern, no Driver."""
    tensor_map = tma._new_aligned_tensor_map()
    ctypes.memset(ctypes.byref(tensor_map), fill, ctypes.sizeof(CUtensorMap))
    return tensor_map


class TestTensorMapToBytes(unittest.TestCase):

    def test_snapshots_exactly_128_opaque_bytes(self):
        tensor_map = _toy_descriptor(0xAB)
        payload = tma.tensor_map_to_bytes(tensor_map)
        self.assertIsInstance(payload, bytes)
        self.assertEqual(len(payload), ctypes.sizeof(CUtensorMap))
        self.assertEqual(payload, b"\xab" * 128)

    def test_snapshot_is_independent_of_source_mutation_and_lifetime(self):
        tensor_map = _toy_descriptor(0x01)
        payload = tma.tensor_map_to_bytes(tensor_map)
        ctypes.memset(ctypes.byref(tensor_map), 0xFF, ctypes.sizeof(CUtensorMap))
        self.assertEqual(payload, b"\x01" * 128)
        # Freeing the source storage must not corrupt the snapshot.
        del tensor_map
        gc.collect()
        self.assertEqual(payload, b"\x01" * 128)

    def test_rejects_non_descriptor(self):
        for bad in (None, b"\x00" * 128, bytearray(128), 0, object()):
            with self.assertRaises(TypeError):
                tma.tensor_map_to_bytes(bad)


class TestTensorMapsToBytes(unittest.TestCase):

    def test_bundle_layout_is_contiguous_128_byte_slots(self):
        descriptors = [_toy_descriptor(fill) for fill in (0x10, 0x20, 0x30)]
        buffer = tma.tensor_maps_to_bytes(descriptors)
        self.assertEqual(len(buffer), 3 * 128)
        for index, fill in enumerate((0x10, 0x20, 0x30)):
            slot = buffer[index * 128:(index + 1) * 128]
            self.assertEqual(slot, bytes([fill]) * 128)
            # Each slot rebuilds the original descriptor byte-for-byte.
            rebuilt = tma.tensor_map_from_bytes(slot)
            self.assertEqual(tma.tensor_map_to_bytes(rebuilt), tma.tensor_map_to_bytes(descriptors[index]))

    def test_accepts_any_sequence_including_generator(self):
        buffer = tma.tensor_maps_to_bytes(_toy_descriptor(0x55) for _ in range(2))
        self.assertEqual(buffer, b"\x55" * 256)

    def test_rejects_empty_bundle(self):
        with self.assertRaises(ValueError):
            tma.tensor_maps_to_bytes([])

    def test_rejects_non_sequence_and_single_descriptor(self):
        for bad in (_toy_descriptor(0x00), b"\x00" * 256, bytearray(256), "descriptor", None, 7):
            with self.assertRaises(TypeError):
                tma.tensor_maps_to_bytes(bad)

    def test_rejects_mixed_content(self):
        with self.assertRaisesRegex(TypeError, r"tensor_maps\[1\]"):
            tma.tensor_maps_to_bytes([_toy_descriptor(0x00), None])


class TestTensorMapFromBytes(unittest.TestCase):

    def test_round_trip_preserves_bytes(self):
        pattern = bytes(range(128))
        tensor_map = tma.tensor_map_from_bytes(pattern)
        self.assertIsInstance(tensor_map, CUtensorMap)
        self.assertEqual(tma.tensor_map_to_bytes(tensor_map), pattern)

    def test_result_is_64_byte_aligned_with_owned_storage(self):
        tensor_map = tma.tensor_map_from_bytes(b"\x7f" * 128)
        address = ctypes.addressof(tensor_map)
        self.assertEqual(address % 64, 0)
        # The rebuilt descriptor owns its storage: dropping every other
        # reference and collecting must not release or corrupt it.
        del tensor_map
        survivors = [tma.tensor_map_from_bytes(b"\x7f" * 128) for _ in range(16)]
        gc.collect()
        addresses = [ctypes.addressof(item) for item in survivors]
        self.assertEqual(len(set(addresses)), len(survivors))
        self.assertTrue(all(item % 64 == 0 for item in addresses))
        self.assertTrue(all(tma.tensor_map_to_bytes(item) == b"\x7f" * 128 for item in survivors))

    def test_accepts_bytes_like_payloads(self):
        base = b"\x33" * 128
        for payload in (base, bytearray(base), memoryview(base)):
            tensor_map = tma.tensor_map_from_bytes(payload)
            self.assertEqual(tma.tensor_map_to_bytes(tensor_map), base)

    def test_rejects_non_bytes_like_payload(self):
        for bad in (None, 128, "x" * 128, _toy_descriptor(0x00), object()):
            with self.assertRaises(TypeError):
                tma.tensor_map_from_bytes(bad)

    def test_rejects_wrong_length(self):
        for size in (0, 1, 127, 129, 256):
            with self.assertRaisesRegex(ValueError, "128"):
                tma.tensor_map_from_bytes(b"\x00" * size)

    def test_bytearray_source_is_copied_not_aliased(self):
        payload = bytearray(b"\x44" * 128)
        tensor_map = tma.tensor_map_from_bytes(payload)
        payload[:] = b"\x99" * 128
        self.assertEqual(tma.tensor_map_to_bytes(tensor_map), b"\x44" * 128)


if __name__ == "__main__":
    unittest.main()
