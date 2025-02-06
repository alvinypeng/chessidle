from __future__ import annotations

import functools

import chessidle.nnue as nnue
from chessidle.position import Position


MATE = 30_000
NO_SCORE = 32_000
MAX_EVAL = 10_000


def clamp_score(score: int, bound: int = MAX_EVAL) -> int:
    return max(-bound, min(score, bound))


def evaluate(position: Position) -> int:
    score = nnue.evaluate(position)

    return clamp_score(score)
