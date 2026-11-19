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
"""Regressions: missing quoted includes are keyed per including directory.

A quoted ``#include`` first searches the *including file's* own directory, so
two includers in different directories that miss the same spelling must not
collapse into one undifferentiated ``missing:<spelling>`` entry: creating the
header next to one includer later changes exactly that includer's resolution
and must be visible in the dependency identity, while the other includer has
to remain recorded as missing.

Backend-agnostic: this exercises the shared
``little_kernel.runtime.cache.scan_user_dependencies`` used by every backend's
build-identity computation, so it runs on CPU without a toolchain or device.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

from little_kernel.runtime.cache import scan_user_dependencies


def _missing_key(includer_dir, spelling):
    dir_key = os.fspath(includer_dir) if includer_dir is not None else None
    return f"missing:{json.dumps([dir_key, spelling])}"


class MissingQuotedIncludeKeyTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name).resolve()
        self.dir_a = self.root / "a"
        self.dir_b = self.root / "b"
        self.dir_a.mkdir()
        self.dir_b.mkdir()
        # Two includers in different directories, same quoted spelling.
        (self.dir_a / "hdr.h").write_text('#include "cfg.h"\n')
        (self.dir_b / "hdr.h").write_text('#include "cfg.h"\n')
        self.source = '#include "a/hdr.h"\n#include "b/hdr.h"\n'

    def _scan_names(self):
        return dict(scan_user_dependencies(self.source, [self.root]))

    def test_missing_spelling_is_keyed_per_includer_directory(self):
        names = self._scan_names()
        # Both includers miss cfg.h: the entries record each directory,
        # they must not collapse into one spelling-only marker.
        assert _missing_key(self.dir_a, "cfg.h") in names, names
        assert _missing_key(self.dir_b, "cfg.h") in names, names
        # The old spelling-only form must be gone.
        assert "missing:cfg.h" not in names, names
        self.assertEqual(names[_missing_key(self.dir_a, "cfg.h")], "missing")
        self.assertEqual(names[_missing_key(self.dir_b, "cfg.h")], "missing")

    def test_creating_header_next_to_one_includer_changes_only_that_entry(self):
        before = scan_user_dependencies(self.source, [self.root])
        (self.dir_a / "cfg.h").write_text("#define CFG 1\n")
        after = scan_user_dependencies(self.source, [self.root])

        # The combined identity changes once a/cfg.h exists.
        self.assertNotEqual(before, after)
        names = dict(after)
        # The includer in dir_a now resolves to real content.
        resolved_a = os.fspath((self.dir_a / "cfg.h").resolve())
        assert resolved_a in names, names
        assert _missing_key(self.dir_a, "cfg.h") not in names, names
        # The includer in dir_b is still missing, and is still attributed
        # to dir_b (not silently satisfied by dir_a's new header).
        assert _missing_key(self.dir_b, "cfg.h") in names, names

        # Creating it for the second includer removes its marker too.
        (self.dir_b / "cfg.h").write_text("#define CFG 2\n")
        final = scan_user_dependencies(self.source, [self.root])
        final_names = dict(final)
        assert not any(k.startswith("missing:") for k in final_names), final_names
        assert os.fspath((self.dir_b / "cfg.h").resolve()) in final_names

    def test_header_in_unsearched_directory_does_not_clear_missing(self):
        # A cfg.h in a directory that is neither the includer's own nor a
        # searched include path must not affect the identity.
        other = self.root / "elsewhere"
        other.mkdir()
        (other / "cfg.h").write_text("#define CFG 1\n")
        names = self._scan_names()
        assert _missing_key(self.dir_a, "cfg.h") in names, names
        assert _missing_key(self.dir_b, "cfg.h") in names, names

    def test_same_directory_includers_share_one_missing_entry(self):
        # Resolution for both includers is identical, so dedup to one entry
        # (keyed by that directory) is correct.
        (self.dir_a / "hdr2.h").write_text('#include "cfg.h"\n')
        source = '#include "a/hdr.h"\n#include "a/hdr2.h"\n'
        names = dict(scan_user_dependencies(source, [self.root]))
        missing = [k for k in names if k.startswith("missing:")]
        self.assertEqual(missing, [_missing_key(self.dir_a, "cfg.h")])

    def test_top_level_missing_include_keys_with_null_directory(self):
        names = dict(scan_user_dependencies('#include "nope.h"\n', [self.root]))
        assert _missing_key(None, "nope.h") in names, names

    def test_source_must_be_str(self):
        with self.assertRaises(TypeError):
            scan_user_dependencies(b'#include "cfg.h"\n', [self.root])


if __name__ == "__main__":
    unittest.main()
