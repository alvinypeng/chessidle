from __future__ import annotations

import ctypes
import multiprocessing as mp

from chessidle.move import Move
from chessidle.evaluation import MATE, NO_SCORE, MAX_EVAL


BOUND_NONE = 0
BOUND_LOWER = 1
BOUND_UPPER = 2
BOUND_EXACT = 3


class TranspositionEntry(ctypes.Structure):
    _fields_ = [
        ('key', ctypes.c_uint64),
        ('_move', ctypes.c_uint16),
        ('_score', ctypes.c_int16),
        ('raw_eval', ctypes.c_int16),
        ('depth', ctypes.c_int8),
        ('_pv_bound', ctypes.c_uint8),
    ]

    @property
    def is_pv(self) -> bool:
        return self._pv_bound >> 2

    @property
    def bound(self) -> int:
        return self._pv_bound & 0b11

    @property
    def move(self) -> Move:
        return Move.decode_int(self._move)

    def score(self, ply: int) -> int:
        if self._score == NO_SCORE:
            return NO_SCORE

        if self._score > MAX_EVAL:
            return self._score - ply

        if self._score < -MAX_EVAL:
            return self._score + ply

        return self._score
    
    def save(
        self,
        key: int,
        move: Move,
        score: int,
        raw_eval: int,
        depth: int,
        is_pv: bool,
        bound: int,
        ply: int,
    ) -> None:
        if score > MAX_EVAL:
            score = score + ply
        elif score < -MAX_EVAL:
            score = score - ply

        if move or (key != self.key):
            self._move = int(move)
    
        if bound == BOUND_EXACT or key != self.key or depth + 4 > self.depth:
            self.key = key
            self._score = score
            self.raw_eval = raw_eval
            self.depth = depth
            self._pv_bound = (is_pv << 2) | bound


class TranspositionTable:

    def __init__(self, mb: int) -> None:
        self.set_size(mb)

    def set_size(self, mb: int) -> None:
        self._length = mb * 1_000_000 // ctypes.sizeof(TranspositionEntry)
        self._entries = mp.Value(TranspositionEntry * self._length, lock=False)

    def clear(self) -> None:
        ctypes.memset(ctypes.addressof(self._entries), 0, ctypes.sizeof(self._entries))

    def __getitem__(self, key: int) -> TranspositionEntry:
        return self._entries[key % self._length]
