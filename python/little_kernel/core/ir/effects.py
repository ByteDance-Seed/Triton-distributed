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
"""Declared effects for lowered IR operations.

These effect categories are deliberately coarser than the intrinsic-level
:mod:`little_kernel.core.memory_effects` model: a lowered escape-hatch op
declares *what class of effect* it has (pure read, write, barrier, opaque
backend escape) so phase verifiers and code motion can reason about it without
parsing the emitted target string. Concrete intrinsic operands keep using the
richer ``memory_effects`` module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class IROpEffect(str, Enum):
    """The effect category of a lowered operation."""

    #: Reads/returns a value with no observable side effect; movable and
    #: cacheable.
    PURE = "pure"
    #: Reads named resources (may observe state written by other ops).
    READ = "read"
    #: Writes named resources.
    WRITE = "write"
    #: Both reads and writes named resources.
    READWRITE = "readwrite"
    #: Synchronization/ordering boundary (e.g. ``__syncthreads``). Acts as a
    #: barrier to reordering; never pure.
    BARRIER = "barrier"
    #: Opaque backend escape hatch whose effects the producer declares but the
    #: compiler cannot inspect. Treated conservatively (never reordered).
    ESCAPE = "escape"


PURE_EFFECT = IROpEffect.PURE
READ_EFFECT = IROpEffect.READ
WRITE_EFFECT = IROpEffect.WRITE
READWRITE_EFFECT = IROpEffect.READWRITE
BARRIER_EFFECT = IROpEffect.BARRIER
ESCAPE_EFFECT = IROpEffect.ESCAPE

#: Effect categories that observe or mutate resource state.
EFFECT_CATEGORIES: Tuple[str, ...] = tuple(e.value for e in IROpEffect)

#: Effects that are never safe to reorder across or fold away.
_SIDE_EFFECTING = frozenset({
    IROpEffect.WRITE,
    IROpEffect.READWRITE,
    IROpEffect.BARRIER,
    IROpEffect.ESCAPE,
})


@dataclass(frozen=True)
class EffectDeclaration:
    """An effect category paired with the named resources it touches.

    ``resources`` names lowered resources (memory regions/pipelines). An empty
    resource tuple is valid for a barrier (ordering-only) or an opaque escape
    hatch whose resources are not individually named.
    """

    category: IROpEffect
    resources: Tuple[str, ...] = ()

    @property
    def is_side_effecting(self) -> bool:
        return self.category in _SIDE_EFFECTING


def coerce_effect(effect) -> IROpEffect:
    """Normalize an effect argument to an :class:`IROpEffect`.

    Accepts an :class:`IROpEffect` member or its string value; anything else
    raises ``TypeError`` rather than silently defaulting to pure.
    """
    if isinstance(effect, IROpEffect):
        return effect
    if isinstance(effect, str):
        try:
            return IROpEffect(effect)
        except ValueError:
            pass
    raise TypeError(f"invalid IR effect {effect!r}; expected one of {EFFECT_CATEGORIES}")
