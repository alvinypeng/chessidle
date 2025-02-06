from __future__ import annotations

import functools
from collections import defaultdict

from chessidle.move import Move
from chessidle.piece import (
    Color,
    WHITE,
    BLACK,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    Piece,
)
from chessidle.square import Square


def _stats(bound: int, fill: int = 0, key: callable | None = None) -> type:

    class Stat:
        value: int = fill

        def update(self, bonus: int) -> None:
            bonus = max(-bound, min(bonus, bound))
            self.value += bonus - int(self.value * abs(bonus) / bound)

    # Correction history needs a key to get stat.
    if key:
        @functools.lru_cache(4096)
        def get(x: object) -> Stat:
            return Stat()
        
        class Stats:

            def __init__(self) -> None:
                get.cache_clear()

            def __getitem__(self, args: object | tuple[object, ...]) -> Stat:
                return get(key(*args) if isinstance(args, tuple) else key(args))

    # History heuristic used for move-sorting.
    else:
        class Stats(defaultdict):

            def __init__(self) -> None:
                super().__init__(Stat)

    return Stats


MAX_CORRECTION = 1000

NonPawnCorrectionHistory = _stats(
    bound=MAX_CORRECTION,
    key=lambda position, color: (
        position.turn,
        position.pieces(color, KNIGHT),
        position.pieces(color, BISHOP),
        position.pieces(color, ROOK),
        position.pieces(color, QUEEN),
        position.pieces(color, KING),
    )   
)

PawnCorrectionHistory = _stats(
    bound=MAX_CORRECTION,
    key=lambda position: (
        position.turn,
        position.pieces(WHITE, PAWN), position.pieces(BLACK, PAWN),
    )
)

InCheck = bool
IsCapture = bool
Captured = Piece

History: defaultdict[tuple[Color, Move], object] = _stats(10000)
CaptureHistory: defaultdict[tuple[Captured, Move], object] = _stats(12000)
ContinuationHistory: defaultdict[tuple[Piece, Square], object] = _stats(20000)
ContinuationHistories = defaultdict[tuple[InCheck, IsCapture, Piece, Move], ContinuationHistory]
