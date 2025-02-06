from __future__ import annotations


File = int
FILES = (
    FILE_A,
    FILE_B,
    FILE_C,
    FILE_D,
    FILE_E,
    FILE_F,
    FILE_G,
    FILE_H,
) = range(8)
FILE_NAMES = 'abcdefgh'


Rank = int
RANKS = (
    RANK_1,
    RANK_2,
    RANK_3,
    RANK_4,
    RANK_5,
    RANK_6,
    RANK_7, 
    RANK_8,
) = range(8)
RANK_NAMES = '12345678'


Square = int
SQUARES = (
    A1, B1, C1, D1, E1, F1, G1, H1,
    A2, B2, C2, D2, E2, F2, G2, H2,
    A3, B3, C3, D3, E3, F3, G3, H3,
    A4, B4, C4, D4, E4, F4, G4, H4,
    A5, B5, C5, D5, E5, F5, G5, H5,
    A6, B6, C6, D6, E6, F6, G6, H6,
    A7, B7, C7, D7, E7, F7, G7, H7,
    A8, B8, C8, D8, E8, F8, G8, H8,
) = range(NO_SQUARE := 64)
SQUARE_NAMES = [file + rank for rank in RANK_NAMES for file in FILE_NAMES]


NORTH = A2 - A1
SOUTH = -NORTH
EAST = B1 - A1
WEST = -EAST
UP = NORTH, SOUTH


def make_square(file: File, rank: Rank) -> Square:
    return 8 * rank + file

def flip_horizontal(square: Square) -> Square:
    '''Flip the square horizontally, e.g. A1 -> H1.'''
    return square ^ 7

def flip_vertical(square: Square) -> Square:
    '''Flip the square vertically e.g. A1 -> A8.'''
    return square ^ 56

def file_of(square: Square) -> File:
    return square % 8

def rank_of(square: Square) -> Rank:
    return square // 8


def _distance(from_: Square, to: Square) -> int:
    file_distance = abs(file_of(from_) - file_of(to))
    rank_distance = abs(rank_of(from_) - rank_of(to))
    return max(file_distance, rank_distance)

DISTANCE = [[_distance(from_, to) for from_ in SQUARES] for to in SQUARES]
