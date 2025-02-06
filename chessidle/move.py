from __future__ import annotations

from dataclasses import dataclass

from chessidle.piece import (
    Piece,
    PieceType,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    PIECE_TYPE_CHARS,
)
from chessidle.square import (
    Square,
    A1,
    B1,
    C1,
    D1,
    F1,
    G1,
    H1,
    C8,
    D8,
    F8,
    G8,
    SQUARE_NAMES,
)


_MOVE_TYPE_NORMAL = 0
_MOVE_TYPE_CASTLE = QUEEN + 1
_MOVE_TYPE_EN_PASSANT = QUEEN + 2


@dataclass(frozen=True, order=True)
class Move:
    from_: Square
    to: Square
    type_: int | PieceType

    @classmethod
    def none(cls) -> Move:
        return cls(A1, A1, _MOVE_TYPE_NORMAL)

    @classmethod
    def null(cls) -> Move:
        return cls(B1, B1, _MOVE_TYPE_NORMAL) 

    @classmethod
    def normal(cls, from_: Square, to: Square) -> Move:
        return cls(from_, to, _MOVE_TYPE_NORMAL)

    @classmethod
    def castle(cls, king: Square, rook: Square) -> Move:
        return cls(king, rook, _MOVE_TYPE_CASTLE)

    @classmethod
    def en_passant(cls, from_: Square, to: Square) -> Move:
        return cls(from_, to, _MOVE_TYPE_EN_PASSANT)

    @classmethod
    def promotion(cls, from_: Square, to: Square, promote_type: PieceType) -> Move:
        return cls(from_, to, promote_type)

    @property
    def is_normal(self) -> bool:
        return self.type_ == _MOVE_TYPE_NORMAL

    @property
    def is_castle(self) -> bool:
        return self.type_ == _MOVE_TYPE_CASTLE

    @property
    def is_en_passant(self) -> bool:
        return self.type_ == _MOVE_TYPE_EN_PASSANT

    @property
    def is_promotion(self) -> bool:
        return self.type_ in (KNIGHT, BISHOP, ROOK, QUEEN)

    @property
    def is_underpromotion(self) -> bool:
        return self.type_ in (KNIGHT, BISHOP, ROOK)

    @property
    def is_queen_promotion(self) -> bool:
        return self.type_ == QUEEN

    @property
    def promote_type(self) -> PieceType:
        return self.type_

    @property
    def rook_to(self) -> Square:
        return ((D1, F1), (D8, F8))[self.to > H1][self.from_ < self.to]

    @property
    def king_to(self) -> Square:
        return ((C1, G1), (C8, G8))[self.to > H1][self.from_ < self.to]

    @classmethod
    def decode_int(cls, i: int) -> Move:
        # Decode a 16-bit integer.
        return cls(i & 0b111_111, (i >> 6) & 0b111_111, i >> 12)
        
    def __int__(self) -> int:
        # Encode the move into a 16-bit integer.
        return self.from_ | self.to << 6 | self.type_ << 12

    def __bool__(self) -> bool:
        # Returns false if move is Move.none() or Move.null().
        return self.from_ != self.to

    def uci(self, chess960: bool = False) -> str:
        return (
            ''
            + SQUARE_NAMES[self.from_]
            + SQUARE_NAMES[self.to if chess960 or not self.is_castle else self.king_to]
            + self.is_promotion * PIECE_TYPE_CHARS[self.promote_type]
        )
