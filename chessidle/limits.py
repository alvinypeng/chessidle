from __future__ import annotations

import ctypes

from chessidle.position import Position
from chessidle.move import Move


MAX_FEN_LENGTH = 100
MAX_HALF_MOVES = 100
MAX_MOVES = 255
MAX_DEPTH = 127


class Limits(ctypes.Structure):
    _fields_ = [
        ('stopped', ctypes.c_bool),
        ('depth', ctypes.c_uint),
        ('nodes', ctypes.c_uint64),
        ('movetime', ctypes.c_uint64),
        ('wtime', ctypes.c_uint64),
        ('btime', ctypes.c_uint64),
        ('winc', ctypes.c_uint64),
        ('binc', ctypes.c_uint64),
        ('movestogo', ctypes.c_uint),
        ('_fen', ctypes.c_wchar * MAX_FEN_LENGTH),
        ('_past_keys', ctypes.c_uint64 * MAX_HALF_MOVES),
        ('_searchable', ctypes.c_uint16 * MAX_MOVES)
    ]

    def set_position(self, position: Position) -> None:
        self._fen = position.fen(chess960=True)
        self._past_keys[:] = [0] * MAX_HALF_MOVES
        for i, key in enumerate(position.past_keys):
            self._past_keys[i] = key

    @property
    def position(self) -> Position:
        position = Position(self._fen)
        position.past_keys = [key for key in self._past_keys if key != 0]
        return position

    @property
    def searchable(self) -> set[Move]:
        return set(Move.decode_int(move) for move in self._searchable if move)

    @searchable.setter
    def searchable(self, searchable: set[Move]) -> None:
        self._searchable[:] = [0] * MAX_MOVES
        for i, move in enumerate(searchable):
            self._searchable[i] = int(move)


class Tally(ctypes.Structure):
    _fields_ = [
        ('nodes', ctypes.c_uint64),
        ('tbhits', ctypes.c_uint64),
    ]


class SearchStopped(Exception):
    pass
