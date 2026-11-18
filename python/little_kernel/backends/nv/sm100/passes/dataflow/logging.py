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
"""Small logging helpers for dataflow analysis passes."""

from __future__ import annotations

import logging
import os

_LOG_FORMAT = "%(asctime)s %(levelname)s %(filename)s] %(message)s"
_DATE_FORMAT = "%m-%d %H:%M:%S"
_ENV_LOG = "LK_DATAFLOW_LOG"
_ENV_LEVEL = "LK_DATAFLOW_LOG_LEVEL"


def get_logger(name: str | None = None) -> logging.Logger:
    base = "little_kernel.dataflow"
    return logging.getLogger(f"{base}.{name}" if name else base)


def configure_logging(level: int | str = logging.INFO) -> None:
    logging.basicConfig(level=level, format=_LOG_FORMAT, datefmt=_DATE_FORMAT)


def configure_logging_from_env() -> None:
    value = os.environ.get(_ENV_LOG)
    if not value or value.lower() in {"0", "false", "no", "off"}:
        return
    level = os.environ.get(_ENV_LEVEL)
    if level is None:
        level = "DEBUG" if value.lower() in {"1", "true", "yes", "on"} else value
    configure_logging(level.upper() if isinstance(level, str) else level)
