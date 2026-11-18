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
"""Shared role/dataflow compilation-context keys and constants."""

from __future__ import annotations

from little_kernel.backends.nv.sm100.language.dataflow.declarations import TMATransferPlan

DATAFLOW_CTX_KEY = "__LK_DATAFLOW_GRAPHS__"
CLUSTER_SHAPE_CTX_KEY = "__LK_CLUSTER_SHAPE__"
CTA_GROUP_SHAPE_CTX_KEY = "__LK_CTA_GROUP_SHAPE__"

#: The no-multicast/no-sharding default TMA transfer plan.
DEFAULT_TMA_TRANSFER_PLAN = TMATransferPlan()
