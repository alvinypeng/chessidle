from __future__ import annotations

import gzip
import itertools
import pickle
from functools import cached_property
from typing import TYPE_CHECKING

from chessidle.bitboard import loop_through, popcount
from chessidle.nnue_weights import compressed
from chessidle.piece import (
    Piece,
    Color,
    WHITE,
    BLACK,
    COLORS,
    PIECES,
    PIECE_TYPES,
)
from chessidle.square import (
    Square,
    file_of,
    flip_horizontal,
    flip_vertical,
    SQUARES,
    FILE_D,
)

if TYPE_CHECKING:
    from chessidle.position import Position


# Load neural network weights.
_decompressed = gzip.decompress(compressed)
HIDDEN_WEIGHTS, HIDDEN_BIASES, OUTPUTS_WEIGHTS, OUTPUTS_BIASES = pickle.loads(_decompressed)


try:
    import numpy as np
except ImportError:
    np = None

if np:
    HIDDEN_WEIGHTS = np.array(HIDDEN_WEIGHTS)
    HIDDEN_BIASES = np.array(HIDDEN_BIASES)
    OUTPUTS_WEIGHTS = np.array(OUTPUTS_WEIGHTS)
    OUTPUTS_BIASES = np.array(OUTPUTS_BIASES)


KING_BUCKETS = (
    0,  1,  2,  3,  3,  2,  1,  0,
    4,  5,  6,  7,  7,  6,  5,  4,
    8,  9,  10, 11, 11, 10, 9,  8,
    8,  9,  10, 11, 11, 10, 9,  8,
    12, 12, 13, 13, 13, 13, 12, 12,
    12, 12, 13, 13, 13, 13, 12, 12,
    14, 14, 15, 15, 15, 15, 14, 14,
    14, 14, 15, 15, 15, 15, 14, 14,
)


def _feature_index(view: Color, king: Square, piece: Piece, square: Square) -> int:
    if view == BLACK:
        king, square = flip_vertical(king), flip_vertical(square)

    if file_of(king) > FILE_D:
        king, square = flip_horizontal(king), flip_horizontal(square)

    return (
        square
        + len(SQUARES) * piece.type_
        + len(SQUARES) * len(PIECE_TYPES) * (view != piece.color)
        + len(SQUARES) * len(PIECE_TYPES) * len(COLORS) * KING_BUCKETS[king]
    )


def _bucket_index(position: Position) -> int:
    return (popcount(position.occupied) - 1) // 4


FEATURE_WEIGHTS = {
    (view, king, piece, square):
    HIDDEN_WEIGHTS[_feature_index(view, king, piece, square)]
    for view in COLORS
    for king in SQUARES
    for piece in PIECES
    for square in SQUARES
}


Accumulation = np.ndarray if np else list[int]

class Accumulator:

    def __init__(
        self,
        position: Position,
        previous: Position | None = None,
        removals: list[tuple[Piece, Square]] | None = None,
        additions: list[tuple[Piece, Square]] | None = None,
    ) -> None:
        # Defer the process of initializing/updating accumulations.
        # Save all necessary data for that process right now. 
        self._position = position
        self._previous = previous
        self._removals = removals
        self._additions = additions
    
    @cached_property
    def _accumulations(self) -> tuple[Accumulation, Accumulation]:
        accumulate_method = self._updated if self._removals else self._initialized
        return accumulate_method(WHITE), accumulate_method(BLACK)

    if np:

        def _initialized(self, view: Color) -> Accumulation:
            king = self._position.king(view)

            accumulation = HIDDEN_BIASES.copy()

            for square, piece in enumerate(self._position.board):
                if piece:
                    accumulation += FEATURE_WEIGHTS[view, king, piece, square]

            return accumulation

        def _updated(self, view: Color) -> Accumulation:
            king = self._position.king(view)

            # Don't incrementally update if the king square changed.
            if king != self._previous.king(view):
                return self._initialized(view)

            accumulation = self._previous.accumulator[view].copy()

            for piece, square in self._removals:
                accumulation -= FEATURE_WEIGHTS[view, king, piece, square]

            for piece, square in self._additions:
                accumulation += FEATURE_WEIGHTS[view, king, piece, square]

            return accumulation

    else:
    
        def _initialized(self, view: Color) -> Accumulation:
            king = self._position.king(view)

            vectors_to_add = (
                FEATURE_WEIGHTS[view, king, piece, square]
                for square, piece in enumerate(self._position.board) if piece
            )

            return [sum(weights) for weights in zip(HIDDEN_BIASES, *vectors_to_add)]

        def _updated(self, view: Color) -> Accumulation:
            king = self._position.king(view)

            # Don't incrementally update if the king square changed.
            if king != self._previous.king(view):
                return self._initialized(view)

            vectors_to_sub = (
                FEATURE_WEIGHTS[view, king, piece, square]
                for piece, square in self._removals
            )

            vectors_to_add = (
                FEATURE_WEIGHTS[view, king, piece, square]
                for piece, square in self._additions
            )

            zipped = zip(self._previous.accumulator[view], *vectors_to_sub, *vectors_to_add)

            # Non-capture, non-castle move.
            if len(self._removals) == 1 and len(self._additions) == 1:
                return [w0 - w1 + w2 for w0, w1, w2 in zipped]
            # Capture move.
            elif len(self._removals) == 2 and len(self._additions) == 1:
                return [w0 - w1 - w2 + w3 for w0, w1, w2, w3 in zipped]
            # Castle.
            else:
                return [w0 - w1 - w2 + w3 + w4 for w0, w1, w2, w3, w4 in zipped]

    def __getitem__(self, view: Color) -> Accumulation:
        return self._accumulations[view]


if np:
    
    def evaluate(position: Position) -> int:
        # Concatenate accumulations for both views.
        x = np.concatenate([
            position.accumulator[position.turn], position.accumulator[not position.turn]
        ])

        index = _bucket_index(position)

        # ReLU -> Dot-Product.
        x = np.maximum(x, 0) @ OUTPUTS_WEIGHTS[index]

        return round(x + OUTPUTS_BIASES[index])

else:
    
    def evaluate(position: Position) -> int:
        # Concatenate accumulations for both views.
        x = itertools.chain(
            position.accumulator[position.turn], position.accumulator[not position.turn],
        )

        index = _bucket_index(position)
        
        # ReLU -> Dot-Product.
        x = sum(w * out for w, out in zip(OUTPUTS_WEIGHTS[index], x) if out > 0)

        return round(x + OUTPUTS_BIASES[index])
