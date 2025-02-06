from __future__ import annotations

import time
from collections.abc import Iterator
from typing import TYPE_CHECKING

import chessidle.nnue as nnue
from chessidle.bitboard import (
    Bitboard,
    popcount,
    msb,
    shifted,
    loop_through,
    pawn_attacks,
    rook_attacks,
    king_attacks,
    PIECE_ATTACKS,
    BETWEEN,
    LINE,
    FILE_MASKS,
    RANK_MASKS,
)
from chessidle.move import Move
from chessidle.piece import Piece, PAWN, KNIGHT, BISHOP, ROOK, QUEEN
from chessidle.square import (
    NO_SQUARE,
    file_of,
    RANK_1,
    RANK_4,
    RANK_5,
    RANK_8,
    EAST,
    WEST,
    UP,
)

if TYPE_CHECKING:
    from chessidle.position import Position


def generate(position: Position, *, captures_and_promotions: bool) -> Iterator[Move]:
    targets = position.colors[not position.turn] if captures_and_promotions else ~position.occupied
    king = position.king(position.turn)

    # Generate king moves.
    yield from (Move.normal(king, to) for to in loop_through(king_attacks(king) & targets))

    if position.checkers:
        # Only king moves are allowed in double check.
        if popcount(position.checkers) > 1:
            return
        # Moves must block check or capture the checker.
        targets &= BETWEEN[king][msb(position.checkers)]

    elif not captures_and_promotions:
        # Castling.
        for rook in position.castle_rooks[position.turn]:
            move = Move.castle(king, rook)

            # Make sure the castling path is not blocked.
            if not position.occupied & ~(1 << king) & ~(1 << rook) & (
                BETWEEN[king][move.king_to] | BETWEEN[rook][move.rook_to]
            ):
                yield move
                        
    up = UP[position.turn]
    rank4 = RANK_MASKS[RANK_5 if position.turn else RANK_4]
    rank8 = RANK_MASKS[RANK_1 if position.turn else RANK_8]
    pawns = position.pieces(position.turn, PAWN)
    push1 = shifted(pawns, up) & ~position.occupied
    push2 = shifted(push1, up) & ~position.occupied & rank4

    if captures_and_promotions:
        push1 &= rank8

        if position.checkers:
            push1 &= BETWEEN[king][msb(position.checkers)]

        if (to := position.en_passant) != NO_SQUARE and position.checkers in (0, 1 << (to - up)):
            for from_ in loop_through(pawn_attacks(not position.turn, to) & pawns):
                yield Move.en_passant(from_, to)

        def generate(targets: Bitboard, step: int) -> Iterator[Move]:
            # Non-promotions.
            for to in loop_through(targets & ~rank8):
                yield Move.normal(to - step, to)
            # Promotions. 
            for to in loop_through(targets & rank8):
                yield Move.promotion(to - step, to, KNIGHT)
                yield Move.promotion(to - step, to, BISHOP)
                yield Move.promotion(to - step, to, ROOK)
                yield Move.promotion(to - step, to, QUEEN)
                
        yield from generate(shifted(pawns, up + EAST) & targets, up + EAST)
        yield from generate(shifted(pawns, up + WEST) & targets, up + WEST)
        yield from generate(push1, up)

    else:
        push1 &= ~rank8

        if position.checkers:
            push1 &= targets
            push2 &= targets

        yield from (Move.normal(to - up, to) for to in loop_through(push1))
        yield from (Move.normal(to - up - up, to) for to in loop_through(push2))

    for piece_type in KNIGHT, BISHOP, ROOK, QUEEN:
        # Loop through piece bitboard.
        for from_ in loop_through(position.pieces(position.turn, piece_type)):
            attacks = PIECE_ATTACKS[piece_type](from_, position.occupied)
            # Generate moves for a piece.
            yield from (Move.normal(from_, to) for to in loop_through(attacks & targets))


def perft(position: Position, depth: int, *, debug: bool = False) -> int:
    nodes = 0

    if depth == 0:
        return 1

    if depth == 1:
        return len(position.legal_moves)

    if debug:
        assert position.key == position._hash()

    for move in position.legal_moves:
        nodes += perft(
            position.do(move, gives_check := position.gives_check(move)),
            depth - 1,
            debug=debug,
        )
        # Test checks.
        if debug and gives_check:
            assert position.do(move).checkers

        # Catch false negatives for position.is_pseudolegal.
        if debug:
            assert position.is_pseudolegal(move)

    return nodes


def perft_divide(position: Position, depth: int, chess960: bool) -> None:
    print(f'\nRunning performance test to depth {depth}\n')

    total_nodes = 0
    start = time.perf_counter()

    for move in position.legal_moves:
        nodes = perft(position.do(move), depth - 1)
        print(move.uci(chess960), ':', nodes)
        total_nodes += nodes

    total_time = time.perf_counter() - start

    print()
    print(f'Nodes: {total_nodes}')
    print(f'Time: {int(total_time * 1000)}ms')
    print(f'NPS: {int(total_nodes / total_time)}')
    print()
    
