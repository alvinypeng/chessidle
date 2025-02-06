from __future__ import annotations

import dataclasses


Color = bool
COLORS = WHITE, BLACK = False, True


PieceType = int
PIECE_TYPES = PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING = range(NO_PIECE_TYPE := 6)
PIECE_TYPE_CHARS = 'pnbrqk.'


PIECE_TYPE_VALUES = (
    PAWN_VALUE := 100,
    KNIGHT_VALUE := 370,
    BISHOP_VALUE := 390,
    ROOK_VALUE := 610,
    QUEEN_VALUE := 1210,
    KING_VALUE := 0,
    NO_PIECE_TYPE_VALUE := 0,
)


@dataclasses.dataclass(frozen=True)
class Piece:
    color: Color
    type_: PieceType

    @classmethod
    def none(cls) -> Piece:
        return cls(False, NO_PIECE_TYPE)

    def __str__(self) -> str:
        char = PIECE_TYPE_CHARS[self.type_]
        return char.upper() if self.color == WHITE else char.lower()

    def __bool__(self) -> bool:
        return self.type_ != NO_PIECE_TYPE
    

PIECES = [
    Piece(color, piece_type)
    for color in COLORS
    for piece_type in PIECE_TYPES
]
