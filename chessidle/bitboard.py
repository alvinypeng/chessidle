from __future__ import annotations

import itertools
from collections.abc import Generator

from chessidle.piece import Color, COLORS
from chessidle.square import (
    Square,
    file_of,
    rank_of,
    SQUARES,
    FILES,
    RANKS,
    NORTH,
    SOUTH,
    EAST,
    WEST,
    DISTANCE
)


Bitboard = int
ALL_MASK = 0xFFFF_FFFF_FFFF_FFFF

FILE_MASKS = (
    FILE_A_MASK,
    FILE_B_MASK,
    FILE_C_MASK,
    FILE_D_MASK,
    FILE_E_MASK,
    FILE_F_MASK,
    FILE_G_MASK,
    FILE_H_MASK,
) = [0x0101_0101_0101_0101 << file for file in FILES]

RANK_MASKS = [
    RANK_1_MASK,
    RANK_2_MASK,
    RANK_3_MASK,
    RANK_4_MASK,
    RANK_5_MASK,
    RANK_6_MASK,
    RANK_7_MASK,
    RANK_8_MASK,
] = [0xFF << (8 * rank) for rank in RANKS]


# For python <= 3.9 versions.
popcount = getattr(int, 'bit_count', lambda bitboard: bin(bitboard).count('1'))


def msb(bitboard: Bitboard) -> int:
    return bitboard.bit_length() - 1


def loop_through(bitboard: Bitboard) -> Generator:
    while bitboard:
        yield (square := bitboard.bit_length() - 1)
        bitboard ^= 1 << square


def shifted(bitboard: Bitboard, step: int) -> Bitboard:
    # Find the step horizontal component.
    x = step % NORTH
    # Ensure horizontal shifts don't wrap.
    if x:
        bitboard &= ~(FILE_H_MASK if x == EAST else FILE_A_MASK)
    # Shifts cannot be negative.  
    return (bitboard << step) & ALL_MASK if step > 0 else bitboard >> -step


PAWN_STEPS = [NORTH + EAST, NORTH + WEST], [SOUTH + EAST, SOUTH + WEST]
KNIGHT_STEPS = [
    NORTH + NORTH + EAST, NORTH + EAST + EAST,
    NORTH + NORTH + WEST, NORTH + WEST + WEST,
    SOUTH + SOUTH + EAST, SOUTH + EAST + EAST,
    SOUTH + SOUTH + WEST, SOUTH + WEST + WEST,
]
BISHOP_STEPS = [NORTH + EAST, NORTH + WEST, SOUTH + EAST, SOUTH + WEST]
ROOK_STEPS = [NORTH, SOUTH, EAST, WEST]
KING_STEPS = BISHOP_STEPS + ROOK_STEPS


def _slider_attacks(square: Square, occupied: Bitboard, steps: list[int]) -> Bitboard:
    attacks = 0

    for step in steps:
        to = square

        while True:
            to += step

            if to not in SQUARES or DISTANCE[to - step][to] > 2:
                break

            attacks |= 1 << to

            if occupied & (1 << to):
                break

    return attacks    


def _step_attack_tables(steps: list[int]) -> list[Bitboard]:
    return [_slider_attacks(square, ALL_MASK, steps) for square in SQUARES]


def _occupied_powerset(mask: Bitboard):
    squares = [square for square in loop_through(mask)]

    for count in range(len(squares) + 1):
        for combination in itertools.combinations(squares, count):
            yield sum(1 << square for square in combination)


def _slider_attack_tables(steps: list[int]) -> tuple[list[Bitboard], list[dict[Bitboard, Bitboard]]]:
    masks = []
    attacks = []

    for square in SQUARES:
        square_attacks = {}

        file_edges = (FILE_A_MASK | FILE_H_MASK) & ~FILE_MASKS[file_of(square)]
        rank_edges = (RANK_1_MASK | RANK_8_MASK) & ~RANK_MASKS[rank_of(square)]
        mask = _slider_attacks(square, 0, steps) & ~(file_edges | rank_edges)

        for subset in _occupied_powerset(mask):
            square_attacks[subset] = _slider_attacks(square, subset, steps)

        masks.append(mask)
        attacks.append(square_attacks)

    return masks, attacks


PAWN_ATTACKS = [_step_attack_tables(PAWN_STEPS[color]) for color in COLORS]
KING_ATTACKS = _step_attack_tables(KING_STEPS)
KNIGHT_ATTACKS = _step_attack_tables(KNIGHT_STEPS)
BISHOP_MASKS, BISHOP_ATTACKS = _slider_attack_tables(BISHOP_STEPS)
ROOK_MASKS, ROOK_ATTACKS = _slider_attack_tables(ROOK_STEPS)


def pawn_attacks(color: Color, square: Square) -> Bitboard:
    return PAWN_ATTACKS[color][square]


def king_attacks(square: Square) -> Bitboard:
    return KING_ATTACKS[square]


def knight_attacks(square: Square) -> Bitboard:
    return KNIGHT_ATTACKS[square]


def bishop_attacks(square: Square, occupied: Bitboard) -> Bitboard:
    return BISHOP_ATTACKS[square][occupied & BISHOP_MASKS[square]]


def rook_attacks(square: Square, occupied: Bitboard) -> Bitboard:
    return ROOK_ATTACKS[square][occupied & ROOK_MASKS[square]]


def queen_attacks(square: Square, occupied: Bitboard) -> Bitboard:
    return bishop_attacks(square, occupied) | rook_attacks(square, occupied)


PIECE_ATTACKS = [
    None,
    lambda square, _: knight_attacks(square),
    bishop_attacks,
    rook_attacks,
    queen_attacks,
    lambda square, _: king_attacks(square),
]


def _line(from_: Square, to: Square) -> Bitboard:
    for attacks in bishop_attacks, rook_attacks:
        if attacks(from_, 0) & (1 << to):
            return (attacks(from_, 0) & attacks(to, 0)) | (1 << from_) | (1 << to)

    return 0

def _between(from_: Square, to: Square) -> Bitboard:
    bitboard = 1 << to

    for attacks in bishop_attacks, rook_attacks:
        if attacks(from_, 0) & (1 << to):
            bitboard |= attacks(from_, 1 << to) & attacks(to, 1 << from_)
            
    return bitboard


LINE = [[_line(from_, to) for to in SQUARES] for from_ in SQUARES]
BETWEEN = [[_between(from_, to) for to in SQUARES] for from_ in SQUARES]


def print_bitboard(bitboard: Bitboard) -> None:
    from chessidle.square import make_square, FILE_A
    
    string = ''
    
    for rank in reversed(RANKS):
        for file in FILES:
            if file == FILE_A:
                string += f' {rank + 1} '
            string += f' {1 if bitboard & (1 << make_square(file, rank)) else 0}'
        string += '\n'
        
    string += f'\n    a b c d e f g h\n'
    string += f' Bitboard: 0x{hex(bitboard)[2:].upper()}\n'
    print(string)   
