from __future__ import annotations

from typing import Iterator

from chessidle.bitboard import (
    shifted,
    loop_through,
    pawn_attacks,
    knight_attacks,
    bishop_attacks,
    rook_attacks,
)
from chessidle.evaluation import NO_SCORE
from chessidle.history import History, CaptureHistory, ContinuationHistory
from chessidle.limits import MAX_DEPTH
from chessidle.move import Move
from chessidle.piece import (
    Color,
    PieceType,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    Piece,
    PIECE_TYPE_VALUES,
)
from chessidle.position import Position
from chessidle.see import see
from chessidle.square import Square, UP, EAST, WEST


class Ply(int):
    STACK_OFFSET = 7
    STACK_SIZE = MAX_DEPTH + STACK_OFFSET
    
    _stack: tuple[Self]
    _index: int

    position: Position
    line: tuple[Move, ...]

    cutoffs: int = 0
    move_count: int = 0
    static_eval: int = NO_SCORE
    skip: Move = Move.none()
    move: Move = Move.none()
    killer: Move = Move.none()
    continuation_history: ContinuationHistory = ContinuationHistory()
    
    @classmethod
    def new_stack(cls) -> Ply:
        stack = []
        for i in range(cls.STACK_SIZE):
            ply = cls(i - cls.STACK_OFFSET)
            ply._index = i
            ply._stack = stack
            stack.append(ply)
        return stack[cls.STACK_OFFSET]

    def __add__(self, offset: int) -> Ply:
        return self._stack[self._index + offset]

    def __sub__(self, offset: int) -> Ply:
        return self._stack[self._index - offset]


class MovePicker:

    def __init__(
        self, ply: Ply, tt_move: Move, depth: int, history: History, capture_history: CaptureHistory
    ) -> None:
        self.ply = ply
        self.tt_move = tt_move
        self.depth = depth
        self.history = history
        self.capture_history = capture_history
        self.move_count_prune = False
        
    def __iter__(self) -> Iterator[Move]:
        ply = self.ply
        position = ply.position
        
        if self.tt_move and position.is_pseudolegal(self.tt_move):
            yield self.tt_move

        # Evasion moves when in check.
        if position.checkers:
            
            def evasion_score(move: Move) -> int:
                # Captures have highest precedence.
                if captured := position.captured(move):
                    return 1e7 + PIECE_TYPE_VALUES[captured.type_] - position.piece(move).type_

                elif move.is_promotion:
                    return self.capture_history[captured, move].value
                
                return (
                    + self.history[position.turn, move].value
                    + (ply - 1).continuation_history[position.piece(move), move.to].value
                )

            for move in sorted(position.pseudolegal_moves, key=evasion_score, reverse=True):
                if move != self.tt_move:
                    yield move
            return

        scored = []
        # Score captures and promotions.
        for move in position.captures_promotions:
            score = (
                + 12 * PIECE_TYPE_VALUES[position.captured(move).type_]
                + self.capture_history[position.captured(move), move].value
            )
            scored.append((score, move))
        # Sort captures and promotions from highest to lowest score.
        scored.sort(reverse=True)

        # Quiescence search.
        if self.depth <= 0:
            # Play captures and queen promotions.
            for _, move in scored:
                if move != self.tt_move and not move.is_underpromotion:
                    yield move
            return
            
        bad_captures_promotions = []
        # Play captures that pass SEE and queen promotions.
        for score, move in scored:
            if move == self.tt_move:
                continue
            if not move.is_underpromotion and see(position, move, -score / 24):
                yield move
            else:
                bad_captures_promotions.insert(0, move)

        scored = []
        bad_non_captures_promotions = []
        
        if not self.move_count_prune:
            us = position.turn

            up = UP[us]
            pawns = position.pieces(not us, PAWN)
            threatened_minor_area = shifted(pawns, EAST - up) | shifted(pawns, WEST - up)

            threatened_rook_area = threatened_minor_area

            for square in loop_through(position.pieces(not us, KNIGHT)):
                threatened_rook_area |= knight_attacks(square)

            for square in loop_through(position.pieces(not us, BISHOP)):
                threatened_rook_area |= bishop_attacks(square, position.occupied)

            threatened_queen_area = threatened_rook_area

            for square in loop_through(position.pieces(not us, ROOK)):
                threatened_queen_area |= rook_attacks(square, position.occupied)

            # Score moves that are not captures or promotions.
            for move in position.non_captures_promotions:
                piece = position.piece(move)
                from_ = move.from_
                to = move.to
                
                score = (
                    + self.history[position.turn, move].value
                    + (ply - 1).continuation_history[piece, to].value
                    + (ply - 2).continuation_history[piece, to].value
                    + (ply - 4).continuation_history[piece, to].value
                    + (move == ply.killer) * 1e7
                )

                threatened_minor_bonus = 7000
                threatened_rook_bonus = 14000
                threatened_queen_bonus = 28000

                if piece.type_ in (KNIGHT, BISHOP):                    
                    score += threatened_minor_bonus * bool(threatened_minor_area & (1 << from_))
                    score -= threatened_minor_bonus * bool(threatened_minor_area & (1 << to))

                elif piece.type_ == ROOK:
                    score += threatened_rook_bonus * bool(threatened_rook_area & (1 << from_))
                    score -= threatened_rook_bonus * bool(threatened_rook_area & (1 << to))

                elif piece.type_ == QUEEN:
                    score += threatened_queen_bonus * bool(threatened_queen_area & (1 << from_))
                    score -= threatened_queen_bonus * bool(threatened_queen_area & (1 << to))

                if position.check_squares[piece.type_] & (1 << to):
                    score += 6000 
                
                scored.append((score, move))
            
            # Play good moves that are not captures or promotions.
            for score, move in sorted(scored, reverse=True):
                if self.move_count_prune:
                    break
                if move == self.tt_move:
                    continue
                if score > -1e7:
                    yield move
                else:
                    bad_non_captures_promotions.insert(0, move)
        
        yield from (move for move in bad_captures_promotions)
        yield from (move for move in bad_non_captures_promotions if not self.move_count_prune)
