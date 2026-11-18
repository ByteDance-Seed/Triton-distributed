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
"""Check implementation dependency direction across hardware source boundaries."""
from __future__ import annotations
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "python/little_kernel"


def owner(module):
    parts = module.split(".")
    if parts[:2] != ["little_kernel", "backends"] or len(parts) < 4:
        return None
    if parts[2] == "common":
        return ("shared", parts[3])
    return tuple(parts[2:4])


def check(root=PACKAGE):
    errors = []
    for path in root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        parts = path.relative_to(root.parent).with_suffix("").parts
        module = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
        source = owner(module)
        package = module.split(".") if path.name == "__init__.py" else module.split(".")[:-1]
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                target = node.module or ""
                if node.level:
                    target = ".".join(package[:len(package) - node.level + 1] + target.split("."))
                imports = [target]
            elif isinstance(node, ast.Import):
                imports = [name.name for name in node.names]
            else:
                continue
            for target in imports:
                destination = owner(target)
                if destination is None:
                    continue
                allowed = destination[0] == "shared" or destination == source
                if source is not None and source[0] == destination[0] and destination[1] == "common":
                    allowed = True
                if source is None:
                    allowed = False
                    # This component is optional, shared across GPU backends, and
                    # selected by the configured pass provider; it is not hardware-private.
                    if module == "little_kernel.backends.shared" and destination[0] == "shared":
                        allowed = True
                if not allowed:
                    errors.append(f"{path.relative_to(root)}:{node.lineno}: {module} imports {target}")
    return errors


def main():
    errors = check()
    for error in errors:
        print(error)
    print(f"backend dependency check: {len(errors)} violation(s)")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
