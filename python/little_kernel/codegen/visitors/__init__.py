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
Codegen visitor mixins.

This package contains mixin classes for organizing codegen visitor methods:
- ExpressionCodegenMixin: Expression nodes (return C++ code strings)
- StatementCodegenMixin: Statement nodes (write to buffers)
- ControlFlowCodegenMixin: Control flow statements (if, while, for, etc.)
- CallCodegenMixin: Function call handling
"""

from .expression_codegen import ExpressionCodegenMixin
from .statement_codegen import StatementCodegenMixin
from .control_flow_codegen import ControlFlowCodegenMixin
from .call_codegen import CallCodegenMixin

__all__ = [
    'ExpressionCodegenMixin',
    'StatementCodegenMixin',
    'ControlFlowCodegenMixin',
    'CallCodegenMixin',
]
