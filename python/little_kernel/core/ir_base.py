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

from enum import Enum, auto

# basic IR


class LLIR:

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            return False
        return self.__dict__ == other.__dict__

    def __hash__(self) -> int:
        print(self)
        return hash((type(self), tuple(sorted(self.__dict__.items()))))

    def __str__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.__dict__})"


class Function(LLIR):

    def __init__(self):
        super().__init__()


class Struct(LLIR):

    def __init__(self):
        super().__init__()


class Module(LLIR):

    def __init__(self):
        super().__init__()


class MemorySpace(Enum):
    """Memory spaces for allocation."""
    GLOBAL = auto()  # Global memory (__device__)
    SHARED = auto()  # Shared memory (__shared__)
    DYNAMIC_SHARED = auto()  # Dynamic shared memory (__shared__ with dynamic size)
    LOCAL = auto()  # Thread-local memory
    CONSTANT = auto()  # Constant memory (__constant__)


class AllocateMode(Enum):
    """Initialization mode for tensor allocation (mimics PyTorch's behavior)."""
    EMPTY = auto()  # Uninitialized (like torch.empty)
    ZEROS = auto()  # Initialized to zeros (like torch.zeros)
    ONES = auto()  # Initialized to ones (like torch.ones)
