from __future__ import annotations

from functools import cached_property

import chessidle.nnue as nnue
import chessidle.polyglot as polyglot
from chessidle.bitboard import (
    Bitboard,
    popcount,
    msb,
    shifted,
    loop_through,
    pawn_attacks,
    knight_attacks,
    bishop_attacks,
    rook_attacks,
    king_attacks,
    PIECE_ATTACKS,
    BETWEEN,
    LINE,
    FILE_MASKS,
    RANK_MASKS,
)
from chessidle.move import Move
from chessidle.move_generation import generate
from chessidle.piece import (
    Piece,
    Color,
    PieceType,
    WHITE,
    BLACK,
    COLORS,
    PAWN,
    KNIGHT,
    BISHOP,
    ROOK,
    QUEEN,
    KING,
    PIECE_TYPES,
    PIECE_TYPE_CHARS,
)
from chessidle.square import (
    Square,
    make_square,
    file_of,
    rank_of,
    NO_SQUARE,
    SQUARES,
    SQUARE_NAMES,
    FILES,
    FILE_NAMES,
    RANKS,
    FILE_A,
    FILE_H,
    RANK_1,
    RANK_2,
    RANK_3,
    RANK_6,
    RANK_7,
    RANK_8,
    A1,
    E1,
    H1,
    A8,
    E8,
    H8,
    UP,
)


START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'


class Position:
    turn: Color
    colors: list[Bitboard]
    piece_types: list[Bitboard]
    castle_rooks: list[list[Square]]
    en_passant: Square
    half_move: int
    full_move: int
    
    key: int
    board: list[Piece]
    checkers: Bitboard
    check_squares: list[Bitboard]
    king_blockers: list[Bitboard]
    pinners: list[Bitboard]
    past_keys: list[int]

    accumulator: nnue.Accumulator

    def __init__(self, fen: str) -> None:
        self._parse_fen(fen)

    def _parse_fen(self, fen: str) -> None:
        '''Set up the position from a FEN string.'''
        if not isinstance(fen, str):
            raise TypeError(f'expected type {str} but instead of got type {type(fen)}')

        if fen == '':
            raise ValueError('fen cannot be empty')
        
        fen_parts = fen.strip().split()
        board_part = fen_parts[0]
        turn_part = fen_parts[1] if len(fen_parts) > 1 else 'w'
        castle_part = fen_parts[2] if len(fen_parts) > 2 else '-'
        en_passant_part = fen_parts[3] if len(fen_parts) > 3 else '-'
        half_move_part = fen_parts[4] if len(fen_parts) > 4 else '0'
        full_move_part = fen_parts[5] if len(fen_parts) > 5 else '1'

        if not set(board_part).issubset('12345678pnbrqkPNBRQK/'):
            raise ValueError('invalid character(s) in board part of fen: ' + fen)

        if len(board_part.split('/')) != len(RANKS):
            raise ValueError('invalid number of ranks in board part of fen: ' + fen)

        self.board = [Piece.none()] * len(SQUARES)
        self.colors = [0] * len(COLORS)
        self.piece_types = [0] * len(PIECE_TYPES)
        self.castle_rooks = [[], []]
        self.en_passant = NO_SQUARE
        self.key = 0

        file = FILE_A
        rank = RANK_8
        was_digit = False

        for char in board_part:
            # Start a new rank.
            if char == '/':
                if file != FILE_H + 1:
                    raise ValueError('expected 8 files in board part of fen: ' + fen)
                file = FILE_A
                rank -= 1
            # Empty squares.
            if char.isnumeric():
                if was_digit:
                    raise ValueError('two subsequent digits in board part of fen: ' + fen)
                file += int(char)
                was_digit = True
            else:
                was_digit = False
            # Piece on a square.
            if char.isalpha():
                color = WHITE if char.isupper() else BLACK
                piece_type = PIECE_TYPE_CHARS.index(char.lower())
                self._place(Piece(color, piece_type), make_square(file, rank))
                file += 1
            
        if (file, rank) != (FILE_H + 1, RANK_1):
            raise ValueError('expected 64 squares in board part of fen: ' + fen)

        if turn_part not in set('wb'):
            raise ValueError('invalid turn part in fen: ' + fen)

        self.turn = WHITE if turn_part == 'w' else BLACK

        if castle_part != '-':
            for char in castle_part:
                king = self.king(color := WHITE if char.isupper() else BLACK)
                rank = RANK_1 if color == WHITE else RANK_8

                # Standard FEN.
                if char in 'KQkq':
                    file = FILE_A if char in 'Qq' else FILE_H
                    if king not in (E1, E8):
                        raise ValueError('invalid castling part in fen: ' + fen)
                # Shredder FEN.
                else:
                    file = FILE_NAMES.index(char.lower())

                self.castle_rooks[color].append(make_square(file, rank))
                    
        if en_passant_part != '-':
            if en_passant_part not in SQUARE_NAMES:
                raise ValueError('invalid en passant part in fen: ' + fen)
            
            square = SQUARE_NAMES.index(en_passant_part)
            
            if pawn_attacks(not self.turn, square) & self.pieces(self.turn, PAWN):
                self.en_passant = square

        if not self._is_ok():
            raise ValueError('invalid fen: ' + fen)

        self.half_move = int(half_move_part)
        self.full_move = int(full_move_part)
        self.key = self._hash()
        self.checkers = self._find_checkers()
        self.past_keys = []
        
        self._set_check_squares()
        self._set_king_blockers()
        
        self.accumulator = nnue.Accumulator(self)

    def _is_ok(self) -> bool:
        '''
        There are some FENs that are in the correct FEN format, but
        can't be reached from a real game. This method is intended to
        prevent such FENs from being parsed. It is not perfect and does
        not cover all edge cases, such as:

        "4k3/8/8/8/8/P7/PP6/4K3 w - - 0 1"
        
        or
        
        "4k3/8/8/8/8/8/PP6/b3K3 w - - 0 1"

        However, the assumption is that users will only use well behaved
        FENs and that this method is overkill.
        '''
        # There is a capturable king (or if kings next to each other).
        if self.attackers_to(self.king(not self.turn)) & self.colors[self.turn]:
            return False
        
        # Pawns are on invalid ranks.
        if self.pawns & (RANK_1 | RANK_8):
            return False
        
        for color in COLORS:
            # There are an invalid number of pieces.
            if (
                popcount(self.colors[color]) > 16
                or popcount(self.pieces(color, PAWN)) > 8
                or popcount(self.pieces(color, KNIGHT)) > 10
                or popcount(self.pieces(color, ROOK)) > 10
                or popcount(self.pieces(color, QUEEN)) > 9
                or popcount(self.pieces(color, KING)) != 1
            ):
                return False

            if castle_rooks := sorted(self.castle_rooks[color]):
                # The king can't start in a corner (for chess960).
                if (king := self.king(color)) in (A1, H1, A8, H8):
                    return False
                # The king isn't on the back rank.
                if rank_of(king) != (RANK_1 if color == WHITE else RANK_8):
                    return False

            if len(castle_rooks) > 2:
                return False

            # King isn't between the castling rooks.
            if len(castle_rooks) == 2 and not (castle_rooks[0] < king < castle_rooks[1]):
                return False

            for rook in castle_rooks:
                # No rook on a castling rook square.
                if self.board[rook] != Piece(color, ROOK):
                    return False

        if self.en_passant != NO_SQUARE:
            # En passant square is on the wrong rank.
            if not (1 << self.en_passant) & RANK_MASKS[RANK_6 if self.turn == WHITE else RANK_3]:
                return False
            # There is no capturable en passant pawn.
            if not (1 << self.en_passant - UP[self.turn]) & self.pieces(not self.turn, PAWN):
                return False
        
        return True

    @property
    def occupied(self) -> Bitboard:
        return self.colors[WHITE] | self.colors[BLACK]

    @property
    def pawns(self) -> Bitboard:
        return self.piece_types[PAWN]

    @property
    def knights(self) -> Bitboard:
        return self.piece_types[KNIGHT]

    @property
    def bishops(self) -> Bitboard:
        return self.piece_types[BISHOP]

    @property
    def rooks(self) -> Bitboard:
        return self.piece_types[ROOK]

    @property
    def queens(self) -> Bitboard:
        return self.piece_types[QUEEN]

    @property
    def kings(self) -> Bitboard:
        return self.piece_types[KING]

    def is_draw(self, ply: int) -> bool:
        # Twofold/threefold repetition draw, depending on ply.
        if self.past_keys.count(self.key) >= (1 if ply > 1 else 2):
            return True

        # Fifty-move rule:
        if self.half_move >= 100:
            return True

        if self.pawns or self.rooks or self.queens:
            return False

        white_count, black_count = map(popcount, self.colors)

        if white_count + black_count > 4:
            return False

        if white_count <= 2 and black_count <= 2:
            return True

        # Two minor pieces (except for 2 knights) is usually a win.
        if popcount(self.knights) != 2:
            return False

        # Two knights is pretty much almost always a draw.
        return self.king(WHITE if white_count < black_count else BLACK) not in (A1, A8, H1, H8)
    
    def pieces(self, color: Color, piece_type: PieceType) -> Bitboard:
        return self.colors[color] & self.piece_types[piece_type]

    def king(self, color: Color) -> Square:
        return msb(self.colors[color] & self.piece_types[KING])
    
    def attackers_to(self, to: Square, occupied: Bitboard = None) -> Bitboard:
        occupied = occupied or self.occupied
        return (
            (king_attacks(to) & self.kings)
            | (knight_attacks(to) & self.knights)
            | (pawn_attacks(BLACK, to) & self.pieces(WHITE, PAWN))
            | (pawn_attacks(WHITE, to) & self.pieces(BLACK, PAWN))
            | (bishop_attacks(to, occupied) & (self.bishops | self.queens))
            | (rook_attacks(to, occupied) & (self.rooks | self.queens))
        )

    def captured(self, move: Move) -> Piece:
        if move.is_normal or move.is_promotion:
            return self.board[move.to]
        if move.is_en_passant:
            return Piece(not self.turn, PAWN)
        return Piece.none()

    def piece(self, move: Move) -> Piece:
        return self.board[move.from_]

    def has_non_pawn(self, color: Color) -> bool:
        return bool(self.colors[color] ^ self.pieces(color, KING) ^ self.pieces(color, PAWN))

    def _place(self, piece: Piece, square: Square) -> None:
        self.board[square] = piece
        self.colors[piece.color] |= 1 << square
        self.piece_types[piece.type_] |= 1 << square
        self.key ^= polyglot.PIECE_KEYS[piece][square]

    def _remove(self, piece: Piece, square: Square) -> None:
        self.board[square] = Piece.none()
        self.colors[piece.color] &= ~(1 << square)
        self.piece_types[piece.type_] &= ~(1 << square)
        self.key ^= polyglot.PIECE_KEYS[piece][square]

    def _hash_en_passant(self) -> int:
        if self.en_passant == NO_SQUARE:
            return 0
        return polyglot.EN_PASSANT_KEYS[file_of(self.en_passant)]

    def _hash_castle(self) -> int:
        key = 0
        for color in COLORS:
            king = self.king(color)
            for rook in self.castle_rooks[color]:
                key ^= polyglot.CASTLE_KEYS[color][rook < king]
        return key

    def _hash(self) -> int:
        '''Calculate the position's polyglot key from scratch.'''
        key = 0 if self.turn == BLACK else polyglot.TURN_KEY
        for square in loop_through(self.occupied):
            key ^= polyglot.PIECE_KEYS[self.board[square]][square]
        return key ^ self._hash_en_passant() ^ self._hash_castle()

    def _find_checkers(self) -> Bitboard:
        return self.attackers_to(self.king(self.turn)) & self.colors[not self.turn]

    def _set_check_squares(self) -> None:
        king = self.king(not self.turn)
        self.check_squares = [
            pawn_attacks(not self.turn, king),
            knight_attacks(king),
            bishop_check_squares := bishop_attacks(king, self.occupied),
            rook_check_squares := rook_attacks(king, self.occupied),
            bishop_check_squares | rook_check_squares,
            0,
        ]

    def _set_king_blockers(self) -> None:
        diagonal = self.bishops | self.queens
        straight = self.rooks | self.queens
        occupied = self.occupied
        self.king_blockers = [0, 0]
        self.pinners = [0, 0]

        for color in COLORS:
            king = self.king(color)
            self.pinners[not color] = pinners = self.colors[not color] & ~self.checkers & (
                bishop_attacks(king, occupied & ~bishop_attacks(king, occupied)) & diagonal
                | rook_attacks(king, occupied & ~rook_attacks(king, occupied)) & straight
            )
            self.king_blockers[color] = sum(
                BETWEEN[king][pinner] & ~pinners & occupied for pinner in loop_through(pinners)
            )

    def _next_position(self) -> Position:
        position = object.__new__(type(self))
        position.turn = not self.turn
        position.board = [*self.board]
        position.colors = [*self.colors]
        position.piece_types = [*self.piece_types]
        position.key = self.key ^ polyglot.TURN_KEY ^ self._hash_en_passant() 
        position.en_passant = NO_SQUARE
        return position

    def do(self, move: Move, find_checkers: bool = True) -> Position:
        position = self._next_position()
        
        # Castling is encoded as "king captures rook".
        captured_or_castle_rook = self.board[
            (square := move.to - (move.is_en_passant) * (up := UP[self.turn]))
        ]

        removals = [
            (moving := self.board[move.from_], move.from_)
        ] + (captured_or_castle_rook != Piece.none()) * [
            (captured_or_castle_rook, square)
        ]
        additions = [
            (Piece(self.turn, ROOK), move.rook_to),
            (Piece(self.turn, KING), move.king_to),            
        ] if move.is_castle else [
            (Piece(self.turn, move.promote_type) if move.is_promotion else moving, move.to)
        ]

        for piece, square in removals: position._remove(piece, square)
        for piece, square in additions: position._place(piece, square)

        position.castle_rooks = [
            [rook for rook in self.castle_rooks[WHITE] if rook not in (move.from_, move.to)],
            [rook for rook in self.castle_rooks[BLACK] if rook not in (move.from_, move.to)],
        ]

        # Double pawn push
        if (
            moving.type_ == PAWN
            and move.from_ + up + up == move.to
            and pawn_attacks(self.turn, move.from_ + up) & self.pieces(not self.turn, PAWN)
        ):
            position.en_passant = move.from_ + up
            position.key ^= position._hash_en_passant()

        elif moving.type_ == KING:
            position.castle_rooks[self.turn] = []

        irreversible = moving.type_ == PAWN or self.captured(move)
        
        position.half_move = 0 if irreversible else (self.half_move + 1)
        position.full_move = self.full_move + (self.turn == BLACK)
        position.key ^= self._hash_castle() ^ position._hash_castle()
        position.checkers = position._find_checkers() if find_checkers else 0
        position.past_keys = [] if irreversible else [*self.past_keys, self.key][-100:]
            
        position._set_check_squares()
        position._set_king_blockers()

        position.accumulator = nnue.Accumulator(position, self, removals, additions)
            
        return position

    def do_null(self) -> Position:
        position = self._next_position()
        position.castle_rooks = [[*self.castle_rooks[WHITE]], [*self.castle_rooks[BLACK]]]
        position.half_move = 0
        position.full_move = 0
        position.checkers = 0
        position.past_keys = []

        position._set_check_squares()
        position._set_king_blockers()

        position.accumulator = self.accumulator

        return position

    def is_pseudolegal(self, move: Move) -> bool:
        '''Test if a move can be generated by the move generator.'''
        if not move.is_normal:
            return move in self.pseudolegal_moves

        piece = self.piece(move)
        turn = self.turn
        from_ = move.from_
        to = move.to

        if (
            piece == Piece.none()
            or piece.color != turn
            or self.colors[turn] & (1 << move.to)
        ):
            return False

        if piece.type_ == PAWN:

            # Already handled promotions.
            if rank_of(to) in (RANK_1, RANK_8):
                return False

            up = UP[turn]

            # Valid pawn capture.
            if pawn_attacks(turn, from_) & (1 << to) & self.colors[not turn]:
                pass

            # Single pawn push.
            elif from_ + up == to:

                if self.occupied & (1 << to):
                    return False

            # Double pawn push.
            elif from_ + up + up == to:

                if self.occupied & (1 << to) or self.occupied & (1 << to - up):
                    return False

                if rank_of(from_) != (RANK_2 if turn == WHITE else RANK_7):
                    return False

            else:
                return False
            
        elif not PIECE_ATTACKS[piece.type_](from_, self.occupied) & (1 << to):
            return False

        if self.checkers and piece.type_ != KING:

            if popcount(self.checkers) > 1:
                return False

            if not BETWEEN[self.king(turn)][msb(self.checkers)] & (1 << to):
                return False      
        
        return True
        
    def is_legal(self, move: Move) -> bool:
        '''Test if a pseudolegal move is legal.'''
        king = self.king(self.turn)
        
        # Handle the tricky horizontally pinned en passant capture.
        if move.is_en_passant and rank_of(move.from_) == rank_of(king):
            # There is a pin if the king is attacked.
            if self.colors[not self.turn] & self.attackers_to(
                king, self.occupied & ~(1 << move.from_) & ~(1 << move.to - UP[self.turn])
            ):
                return False

        if self.board[move.from_].type_ != KING:
            # The piece is pinned.
            if (1 << move.from_) & self.king_blockers[self.turn]:
                # Pinned pieces must stay aligned with the king.
                return bool((1 << move.to) & LINE[king][move.from_])
            # The piece is not pinned.
            return True

        if move.is_castle:
            # Castling through check is not allowed.
            for square in loop_through(BETWEEN[king][move.king_to]):
                if self.attackers_to(square) & self.colors[not self.turn]:
                    return False
            # Make sure the castling rook isn't pinned (for chess960).
            return not (1 << move.to) & self.king_blockers[self.turn]

        # Make sure the king "to" square is not attacked by enemies.
        return not self.colors[not self.turn] & self.attackers_to(
            move.to, self.occupied & ~(1 << king)
        )

    def gives_check(self, move: Move) -> bool:
        piece = self.board[move.from_]
        
        # Test for direct check.
        if piece.type_ != KING and self.check_squares[piece.type_] & (1 << move.to):
            return True

        # Normal discover check.
        if (
            not LINE[(king := self.king(not self.turn))][move.from_] & (1 << move.to)
            and self.king_blockers[not self.turn] & (1 << move.from_)
        ):
            return True

        # Exit early if move is a normal move.
        if move.is_normal:
            return False

        # Promotion check.
        if move.is_promotion:
            return bool((1 << king) & PIECE_ATTACKS[move.promote_type](
                move.to, self.occupied & ~(1 << move.from_)
            ))            

        # Castle check.
        if move.is_castle:
            return bool(
                self.check_squares[ROOK] & (1 << move.rook_to)
                or self.king_blockers[not self.turn] & (1 << self.king(self.turn))
            )

        # En passant discover check.
        if move.is_en_passant:
            occupied = self.occupied | (1 << self.en_passant)
            occupied &= ~(1 << move.from_)
            occupied &= ~(1 << self.en_passant - UP[self.turn])
            
            return bool(self.colors[self.turn] & (
                bishop_attacks(king, occupied) & (self.bishops | self.queens)
                | rook_attacks(king, occupied) & (self.rooks | self.queens)
            ))

        assert False

    @cached_property
    def captures_promotions(self) -> list[Move]:
        return list(generate(self, captures_and_promotions=True))

    @cached_property
    def non_captures_promotions(self) -> list[Move]:
        return list(generate(self, captures_and_promotions=False))

    @property
    def pseudolegal_moves(self) -> list[Move]:
        return self.captures_promotions + self.non_captures_promotions

    @property
    def legal_moves(self) -> list[Move]:
        return [move for move in self.pseudolegal_moves if self.is_legal(move)]

    def fen(self, *, chess960: bool) -> str:
        fen = ''

        for rank in reversed(RANKS):
            empty = 0
            
            for file in FILES:
                if (piece := self.board[make_square(file, rank)]) != Piece.none():
                    fen += '' if empty == 0 else str(empty)
                    fen += str(piece)
                    empty = 0
                else:
                    empty += 1

            fen += '' if empty == 0 else str(empty)
            fen += '' if rank == RANK_1 else '/'

        fen += ' ' + ('w' if self.turn == WHITE else 'b')

        castle_part = ''
        for color in COLORS:
            king = self.king(color)
            # Sort the rooks so kingside comes before queenside.
            for rook in sorted(self.castle_rooks[color], reverse=True):
                char = FILE_NAMES[file_of(rook)] if chess960 else 'kq'[rook < king]
                castle_part += char.upper() if color == WHITE else char.lower()

        if self.en_passant == NO_SQUARE:
            en_passant_part = '-'
        else:
            en_passant_part = SQUARE_NAMES[self.en_passant].lower()

        fen += ' ' + (castle_part or '-')
        fen += ' ' + (en_passant_part)
        fen += ' ' + str(self.half_move)
        fen += ' ' + str(self.full_move)
        return fen

    def pretty_print(self, chess960: bool) -> None:
        string = ''
        
        for rank in reversed(RANKS):
            for file in FILES:
                if file == FILE_A:
                    string += f' {rank + 1} '
                if piece := self.board[make_square(file, rank)]:
                    string += f' {str(piece)}'
                if piece == Piece.none():
                    string += f' .'
                if file == FILE_H:
                    string += f' \n'

        string += f'\n    a b c d e f g h\n'
        string += f'\n Key: 0x{self._hash():X}'
        string += f'\n FEN: {self.fen(chess960=chess960)}\n'
        print(string)

    def copy(self) -> Position:
        position = type(self)(self.fen(chess960=True))
        position.past_keys = self.past_keys.copy()
        return position
