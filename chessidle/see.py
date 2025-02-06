from __future__ import annotations

from chessidle.bitboard import msb, bishop_attacks, rook_attacks
from chessidle.move import Move
from chessidle.piece import (
    PIECE_TYPE_VALUES,
    PAWN_VALUE,
    KNIGHT_VALUE,
    BISHOP_VALUE,
    ROOK_VALUE,
    QUEEN_VALUE,
)
from chessidle.position import Position


def see(position: Position, move: Move, threshold: float) -> bool:
    if not move.is_normal:
        return 0 >= threshold
    
    if (value := PIECE_TYPE_VALUES[position.board[(to := move.to)].type_] - threshold) < 0:
        return False
    
    if (value := PIECE_TYPE_VALUES[position.board[(from_ := move.from_)].type_] - value) <= 0:
        return True
    
    turn = position.turn
    occupied = position.occupied ^ (1 << from_) ^ (1 << to)
    attackers = position.attackers_to(to, occupied)
    diagonal = position.bishops | position.queens
    straight = position.rooks | position.queens
    result = True
    
    while True:
        turn = not turn
        attackers &= occupied
        
        if not (attackers_for_turn := attackers & position.colors[turn]):
            break

        # SF idea: pinned pieces don't attack if pinners didn't move.
        if position.pinners[not turn] & occupied:
            attackers_for_turn &= ~position.king_blockers[turn]
            
            if not attackers_for_turn:
                break
        
        result = not result
        
        if smallest_attackers := attackers_for_turn & position.pawns:
            if (value := PAWN_VALUE - value) < result:
                break
            occupied ^= 1 << msb(smallest_attackers)
            attackers |= bishop_attacks(to, occupied) & diagonal
            
        elif smallest_attackers := attackers_for_turn & position.knights:
            if (value := KNIGHT_VALUE - value) < result:
                break
            occupied ^= 1 << msb(smallest_attackers)
            
        elif smallest_attackers := attackers_for_turn & position.bishops:
            if (value := BISHOP_VALUE - value) < result:
                break
            occupied ^= 1 << msb(smallest_attackers)
            attackers |= bishop_attacks(to, occupied) & diagonal
            
        elif smallest_attackers := attackers_for_turn & position.rooks:
            if (value := ROOK_VALUE - value) < result:
                break
            occupied ^= 1 << msb(smallest_attackers)
            attackers |= rook_attacks(to, occupied) & straight
            
        elif smallest_attackers := attackers_for_turn & position.queens:
            if (value := QUEEN_VALUE - value) < result:
                break
            occupied ^= 1 << msb(smallest_attackers)
            attackers |= bishop_attacks(to, occupied) & diagonal
            attackers |= rook_attacks(to, occupied) & straight
            
        else:
            return (not result) if (attackers & ~position.colors[turn]) else result
        
    return result
    
