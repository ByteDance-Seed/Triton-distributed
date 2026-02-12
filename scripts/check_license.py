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

import argparse
import re
import os
import subprocess
from pathlib import Path

PYTHON_LICENSES = [
    """################################################################################
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
################################################################################""",
    """################################################################################
# Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
################################################################################"""
]

TD_LICENSES = [
    """//
// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//""", """//
// Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
//"""
]

CPP_LICENSES = [
    """/*
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
 */""", """//
// Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//
""", """
/*
 * Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
 */
""", """//
// Modification Copyright 2025 ByteDance Ltd. and/or its affiliates.
//
"""
]

PYTHON_EXTENSIONS = ['.py', '.pyi']
TD_EXTENSIONS = ['.td']
CPP_EXTENSIONS = ['.cc', '.cpp', '.c', '.h', '.hpp', '.cu', '.cuh']
WHITELIST_PATTERNS = [
    r'setup.py', r'^.*patches\/triton\/.*$', r"\.codebase\/.*"
]


def get_modified_files():
    try:
        result1 = subprocess.run(['git', 'diff', '--name-only'],
                                 capture_output=True,
                                 text=True)
        result2 = subprocess.run(['git', 'diff', '--name-only', '--cached'],
                                 capture_output=True,
                                 text=True)
        modified_files = set(result1.stdout.strip().split('\n') +
                             result2.stdout.strip().split('\n'))
        return [file for file in modified_files if file]
    except Exception as e:
        print(f"Error getting modified files: {e}")
        return []


def get_git_tracked_files():
    try:
        result = subprocess.run(['git', 'ls-files'],
                                capture_output=True,
                                text=True,
                                cwd=Path(__file__).parent.parent)
        tracked_files = set(result.stdout.strip().split('\n'))
        return [file for file in tracked_files if file]
    except Exception as e:
        print(f"Error getting tracked files: {e}")
        return []


def is_whitelisted(file_path):
    for pattern in WHITELIST_PATTERNS:
        if re.match(pattern, file_path):
            return True
    return False


def check_license(file_path):
    if is_whitelisted(file_path):
        return True

    _, ext = os.path.splitext(file_path)
    if ext in PYTHON_EXTENSIONS:
        licenses = PYTHON_LICENSES
    elif ext in TD_EXTENSIONS:
        licenses = TD_LICENSES
    elif ext in CPP_EXTENSIONS:
        licenses = CPP_LICENSES
    else:
        return True

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            max_license_lines = max(
                len(license.splitlines()) for license in licenses)
            num_lines = max_license_lines + 5
            first_lines = ''.join(lines[:num_lines])
            for license_text in licenses:
                if license_text.strip() in first_lines.strip():
                    return True
            return False
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return False


def add_license(file_path):
    """Add license header to a file that is missing it."""
    _, ext = os.path.splitext(file_path)
    if ext in PYTHON_EXTENSIONS:
        license_text = PYTHON_LICENSES[0] + "\n\n"
    elif ext in TD_EXTENSIONS:
        license_text = TD_LICENSES[0] + "\n\n"
    elif ext in CPP_EXTENSIONS:
        license_text = CPP_LICENSES[0] + "\n\n"
    else:
        return False

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    if ext in PYTHON_EXTENSIONS:
        prefix = ""
        rest = content
        lines = content.split("\n")
        if lines and lines[0].startswith("#!"):
            prefix = lines[0] + "\n"
            rest = "\n".join(lines[1:]).lstrip("\n")
            if rest:
                rest = "\n" + rest
        elif lines and "coding" in lines[0]:
            prefix = lines[0] + "\n"
            rest = "\n".join(lines[1:]).lstrip("\n")
            if rest:
                rest = "\n" + rest
        new_content = prefix + license_text + rest
    else:
        new_content = license_text + content

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check license for modified files.")
    parser.add_argument("--modify-only", default=False, action="store_true")
    parser.add_argument("--add",
                        default=False,
                        action="store_true",
                        help="Add license to files that are missing it.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.modify_only:
        to_check_files = get_modified_files()
    else:
        to_check_files = get_git_tracked_files()

    missing_license_files = []

    for file in to_check_files:
        if Path(file).is_file():
            if not check_license(file):
                missing_license_files.append(file)

    if missing_license_files:
        if args.add:
            for file in missing_license_files:
                add_license(file)
                print(f"Added license to: {file}")
            print(f"[✅] Added license to {len(missing_license_files)} files")
        else:
            print("The following files are missing the required license:")
            for file in missing_license_files:
                print(file)
            assert not missing_license_files, "Some files are missing the required license."
    else:
        print("[✅] check license passed")


if __name__ == "__main__":
    main()
