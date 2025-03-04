from __future__ import annotations

import ctypes
import math
import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import chessidle.nnue as nnue
from chessidle.evaluation import (
    MATE,
    NO_SCORE,
    MAX_EVAL,
    evaluate,
    clamp_score,
)
from chessidle.history import (
    History,
    CaptureHistory,
    ContinuationHistory,
    ContinuationHistories,
    NonPawnCorrectionHistory,
    PawnCorrectionHistory,
)
from chessidle.limits import Limits, Tally, SearchStopped, MAX_DEPTH
from chessidle.move import Move
from chessidle.move_pick import MovePicker, Ply
from chessidle.piece import WHITE, BLACK
from chessidle.position import Position
from chessidle.see import see
from chessidle.square import Square, NO_SQUARE
from chessidle.transposition import (
    BOUND_NONE,
    BOUND_LOWER,
    BOUND_UPPER,
    BOUND_EXACT,
    TranspositionTable,
)

if TYPE_CHECKING:
    from chessidle.options import Options


@dataclass
class RootMove:
    score: int = -MATE
    previous_score: int = -MATE
    mean_score: int = -MATE
    seldepth: int = 0
    nodes: int = 0
    line: tuple[Move, ...] = ()

    @property
    def move(self) -> Move:
        return self.line[0]

    def __lt__(self, other: RootMove) -> bool:
        return (self.score, self.previous_score) < (other.score, other.previous_score)
    

def standby(
    worker_id: int,
    limits: Limits,
    options: Options,
    exit_flag: ctypes.c_bool,
    barrier: mp.synchronize.Barrier | threading.Barrier,
    tallies: list[Tally],
    tt: TranspositionTable,
) -> None:
    MAIN_WORKER = worker_id == 0
    CHESS960 = options['UCI_Chess960']

    tally = tallies[worker_id]

    history = History()
    capture_history = CaptureHistory()
    continuation_histories = ContinuationHistories(ContinuationHistory)

    non_pawn_correction_history = NonPawnCorrectionHistory()
    pawn_correction_history = PawnCorrectionHistory()

    def stat_bonus(depth: int) -> int:
        return min(200 * depth - 100, 1600)
    
    def stat_malus(depth: int) -> int:
        return min(640 * depth - 200, 2000)

    def reduction(depth: int, move_count: int, improving: int, delta: int) -> float:
        r = math.log(depth) * math.log(move_count) / 2.5
        return r + 1.0 * (r > 1.0 and not improving) - 1.0 * (delta / root_delta) + 1.0
        
    def draw_score() -> int:
        return 1 - (tally.nodes & 2)

    def time_elapsed() -> float:
        return time.perf_counter() - start

    def total_nodes() -> int:
        return sum(tally.nodes for tally in tallies)

    def exit_if_limits_exceeded() -> None:
        if MAIN_WORKER:
            if root_depth == 1:
                return
            if time_elapsed() >= time_allocated or total_nodes() >= max_nodes:
                limits.stopped = True

        if limits.stopped:
            raise SearchStopped

    def print_info(alpha: int, beta: int) -> None:
        elapsed = time_elapsed()
        nodes = total_nodes()
        
        for i, root_move in enumerate(root_moves[:options['MultiPV']], start=1):
            string = f'info depth {root_depth} seldepth {root_move.seldepth} multipv {i} '

            if (score := root_move.score) == -MATE:
                score = root_move.previous_score
            else:
                score = max(alpha, min(score, beta))

            if abs(score) <= MAX_EVAL:
                string += f'score cp {score} '
            else:
                string += f'score mate ' + '-' * (score < 0) + f'{(MATE - abs(score) + 1) // 2} '

            if abs(score) == MATE:
                pass
            elif score <= alpha:
                string += 'upperbound '
            elif score >= beta:
                string += 'lowerbound '
                
            string += (
                f'nodes {nodes} '
                f'nps {int(nodes / elapsed)} '
                f'time {int(elapsed * 1000)} '
                f'pv ' + ' '.join(move.uci(CHESS960) for move in root_move.line)
            )
            
            print(string, flush=True)

    def search(ply: Ply, depth: int, alpha: int, beta: int, cut_node: bool) -> int:
        exit_if_limits_exceeded()

        if depth <= 0:
            return qsearch(ply, 0, alpha, beta)

        position = ply.position
        in_check = bool(position.checkers)
        is_pv = beta - alpha != 1
        skip = ply.skip

        ply.move_count = move_count = 0
        ply.line = ()

        if is_pv:
            nonlocal seldepth
            seldepth = max(seldepth, int(ply) + 1)

        if not (is_root := int(ply) == 0):
            tally.nodes += 1

            if position.is_draw(int(ply)):
                return draw_score()

            if int(ply) >= MAX_DEPTH - 1:
                return 0 if in_check else evaluate(position)

            # Mate distance pruning.
            alpha = max(alpha, -MATE + int(ply))
            beta = min(beta, MATE - int(ply) - 1)
            if alpha >= beta:
                return alpha

        tte = tt[(key := position.key)]
        tt_hit = tte.key == key
        tt_move = root_moves[root_index].move if is_root else (tte.move if tt_hit else Move.none())
        tt_score = tte.score(int(ply)) if tt_hit else NO_SCORE
        was_pv = tte.is_pv if tt_hit else False

        if previous_move := (ply - 1).move:
            was_capture = bool(previous_captured := (ply - 1).position.captured(previous_move))
        else:
            was_capture = False

        if (
            not is_pv
            and not skip
            and tt_score != NO_SCORE
            and tte.depth >= depth
            and tte.bound & (BOUND_LOWER if tt_score >= beta else BOUND_UPPER)
        ):
            if tt_move and tt_score >= beta:

                # Bonus for tt_move that is not a capture/promotion.
                if not (position.captured(tt_move) or tt_move.is_promotion):
                    update_non_capture_non_promotion_histories(ply, tt_move, stat_bonus(depth))

                # Penalty for early previous quiet move.
                if (
                    previous_move
                    and not was_capture
                    and not previous_move.is_promotion
                    and (ply - 1).move_count <= 2
                ):
                    update_continuation_histories(ply - 1, previous_move, -stat_malus(depth + 1))
            
            # No transposition cutoffs for high half move counts.
            if position.half_move < 90:
                return tt_score

        if in_check:
            raw_eval = ply.static_eval = eval_ = NO_SCORE
            correction = 0
            improving = False
            
        else:
            correction = get_correction(position, ply)

            if skip:
                # We already have the evaluation in singular search.
                eval_ = ply.static_eval
                
            elif tt_hit:
                raw_eval = tte.raw_eval
                
                # Evaluate if there is no evaluation.
                if raw_eval == NO_SCORE:
                    raw_eval = evaluate(position)

                ply.static_eval = eval_ = clamp_score(raw_eval + correction)
                    
                # Use transposition table value as a better evaluation.
                if (
                    tt_score != NO_SCORE
                    and tte.bound & (BOUND_LOWER if tt_score > eval_ else BOUND_UPPER)
                ):
                    eval_ = tt_score
                    
            else:
                raw_eval = evaluate(position)
                ply.static_eval = eval_ = clamp_score(raw_eval + correction)

                tte.save(key, Move.none(), NO_SCORE, raw_eval, -1, is_pv or was_pv, BOUND_NONE, 0)

            # Use static eval difference to improve move ordering.
            if (
                previous_move
                and not was_capture
                and not previous_move.is_promotion
                and (ply - 1).static_eval != NO_SCORE
            ):
                bonus = 400 - 10 * clamp_score(ply.static_eval + (ply - 1).static_eval, 200)
                history[not position.turn, previous_move].update(bonus)                
                
            improving = ply.static_eval > (ply - 2).static_eval

        (ply + 1).killer = Move.none()

        if not in_check:
            
            # Reverse futility pruning.
            if (
                not is_pv
                and depth < 8
                and MAX_EVAL >= eval_ >= beta
                and eval_ - 90 * depth + 90 * improving >= beta
            ):
                return int(eval_ / 2 + beta / 2)

            # Null move pruning.
            if (
                cut_node
                and not skip
                and eval_ >= beta >= -MAX_EVAL
                and position.has_non_pawn(position.turn)
                and (ply - 1).move != Move.null()
            ):
                r = 4.0 + depth / 4.0 + min(4.0, (eval_ - beta) / 200)

                (ply).move = Move.null()
                (ply).continuation_history = ContinuationHistory()
                (ply + 1).position = position.do_null()

                score = -search(ply + 1, depth - int(r), -beta, -beta + 1, False)

                if MAX_EVAL >= score >= beta:
                    return score

            if is_pv and not tt_move:
                depth -= 1

            if depth <= 0:
                return qsearch(ply, 0, alpha, beta)

        (ply + 1).cutoffs = 0

        best_score = -MATE
        best_move = Move.none()

        captures_promotions = []
        non_captures_promotions = []
        picker = MovePicker(ply, tt_move, depth, history, capture_history)

        for move in picker:
            if move == skip:
                continue
            if is_root and move not in searchable:
                continue
            if not position.is_legal(move):
                continue

            ply.move_count = move_count = move_count + 1

            if MAIN_WORKER and is_root and time_elapsed() > 3.0:
                string = 'info depth {} currmove {} currmovenumber {}'
                print(string.format(root_depth, move.uci(CHESS960), move_count), flush=True)

            piece = position.piece(move)
            gives_check = position.gives_check(move)
            is_capture = bool(captured := position.captured(move))

            new_depth = depth - 1

            r = reduction(depth, move_count, improving, beta - alpha)

            if (
                not is_root
                and best_score >= -MAX_EVAL
                and position.has_non_pawn(position.turn)
            ):
                picker.move_count_prune = move_count >= (5 + depth * depth) // (2 - improving)

                if is_capture or gives_check or move.is_promotion:

                    if not see(position, move, -90 * depth):
                        continue
                    
                else:
                    reduced_depth = max(new_depth - int(r), 0)

                    if (
                        + (ply - 1).continuation_history[piece, move.to].value
                        + (ply - 2).continuation_history[piece, move.to].value < -4000 * depth
                    ):
                        continue

                    if (
                        not in_check
                        and reduced_depth < 7
                        and eval_ + 90 + 50 * reduced_depth <= alpha
                    ):
                        continue

                    if not see(position, move, -20 * reduced_depth * reduced_depth):
                        continue
        
            extension = 0           

            # Singular extension.
            if (
                move == tt_move
                and not skip
                and not is_root
                and depth >= 5
                and tte.depth >= depth - 3
                and tte.bound & BOUND_LOWER
                and int(ply) < 2 * root_depth
                and abs(tt_score) <= MAX_EVAL
            ):
                singular_beta = tt_score - 1 * depth

                ply.skip = move
                score = search(ply, depth // 2, singular_beta - 1, singular_beta, cut_node)
                ply.skip = Move.none()
                
                if score < singular_beta:
                    extension = 1

                elif singular_beta >= beta:
                    return singular_beta

                elif tt_score >= beta:
                    extension = -2

                elif cut_node:
                    extension = -1

            new_depth += extension

            (ply).move = move
            (ply).continuation_history = continuation_histories[in_check, is_capture, piece, move]
            (ply + 1).position = position.do(move, gives_check)

            starting_node_count = tally.nodes

            if cut_node:
                r += 2.0

            if is_pv or was_pv:
                r -= 1.0

            if tt_hit:
                r -= 1.0 * (tte.depth >= depth)

            if (ply + 1).cutoffs > 4:
                r += 1.0

            elif move == tt_move:
                r -= 2.0

            if is_capture or move.is_promotion:
                captures_promotions.append(move)

                h = capture_history[captured, move].value
                r -= (h - 4000) / 15000

            else:
                non_captures_promotions.append(move)

                h = (
                    + history[position.turn, move].value
                    + (ply - 1).continuation_history[piece, move.to].value
                    + (ply - 2).continuation_history[piece, move.to].value
                )
                r -= (h - 4000) / 15000

                r += 1.0 * (tt_move in captures_promotions)

            # Late move reduction.
            if depth >= 2 and move_count > 1:
                # Allow for a +1 extension if r <= -1.
                reduced_depth = max(new_depth - max(int(r), -1), 1)
                
                score = -search(ply + 1, reduced_depth, -alpha - 1, -alpha, True)

                if score > alpha and new_depth > reduced_depth:
                    new_depth += (score > best_score + 70)
                    new_depth -= (score < best_score + 1 * new_depth)

                    if new_depth > reduced_depth:
                        score = -search(ply + 1, new_depth, -alpha - 1, -alpha, not cut_node)

                    if score >= beta:
                        update_continuation_histories(ply, move, stat_bonus(new_depth))

            elif not is_pv or move_count > 1:
                r += 2.0 * (not tt_move)
                
                score = -search(ply + 1, new_depth - (r > 3.0), -alpha - 1, -alpha, not cut_node)

            if is_pv and (move_count == 1 or score > alpha):
                score = -search(ply + 1, new_depth, -beta, -alpha, False)

            if is_root:
                for root_move in root_moves:
                    if root_move.move == move:
                        break

                root_move.nodes += tally.nodes - starting_node_count

                # Track a weighted mean score for each root move.
                root_move.mean_score = int(
                    score if root_move.mean_score == -MATE else (root_move.mean_score + score) / 2)
        
                if move_count == 1 or score > alpha:
                    root_move.seldepth = seldepth
                    root_move.score = score
                    root_move.line = ply.line = (move,) + (ply + 1).line
                else:
                    root_move.score = -MATE

            if score > best_score:
                best_score = score

                if score > alpha:
                    best_move = move

                    if is_pv:
                        ply.line = (move,) + (ply + 1).line

                    if score >= beta:
                        if not skip: ply.cutoffs += 1
                        break

                    alpha = score

        if move_count == 0:
            return alpha if skip else (-MATE + int(ply)) if in_check else 0

        bound = (
            BOUND_LOWER if best_score >= beta else
            BOUND_EXACT if best_move and is_pv else
            BOUND_UPPER
        )

        if bound == BOUND_LOWER:
            bonus = stat_bonus(depth)
            malus = stat_malus(depth)

            if best_move in captures_promotions:
                capture_history[position.captured(best_move), best_move].update(bonus)

            else:
                ply.killer = best_move
                update_non_capture_non_promotion_histories(ply, best_move, bonus)
                
                for move in non_captures_promotions:
                    if move != best_move:
                        update_non_capture_non_promotion_histories(ply, move, -malus)

            for move in captures_promotions:
                if move != best_move:
                    capture_history[position.captured(move), move].update(-malus)

        elif not best_move and previous_move:
            bonus = stat_bonus(depth)

            if was_capture:
                capture_history[previous_captured, previous_move].update(bonus)

            elif not previous_move.is_promotion:
                update_non_capture_non_promotion_histories(ply - 1, previous_move, bonus)
            
        if not skip and not (is_root and root_index == 0):
            tte.save(key, best_move, best_score, raw_eval, depth, is_pv or was_pv, bound, int(ply))

        if (
            not in_check
            and best_move not in captures_promotions
            and (eval_error := ply.static_eval - best_score)
            and (best_move if (eval_error < 0) else (best_score < beta))
        ):
            bonus = clamp_score(-eval_error * depth / 8, 250)

            non_pawn_correction_history[position, WHITE].update(int(bonus))
            non_pawn_correction_history[position, BLACK].update(int(bonus))
            pawn_correction_history[position].update(int(bonus))

        return best_score

    def qsearch(ply: Ply, depth: int, alpha: int, beta: int) -> int:
        exit_if_limits_exceeded()
        
        position = ply.position
        in_check = bool(position.checkers)
        is_pv = beta - alpha != 1

        ply.move_count = move_count = 0
        ply.line = ()
        
        tally.nodes += 1

        if position.is_draw(int(ply)):
            return draw_score()

        if int(ply) >= MAX_DEPTH - 1:
            return 0 if in_check else evaluate(position)

        tte = tt[(key := position.key)]
        tt_hit = tte.key == key
        tt_move = tte.move if tt_hit else Move.none()
        tt_score = tte.score(int(ply)) if tt_hit else NO_SCORE
        was_pv = tte.is_pv if tt_hit else False

        if (
            not is_pv
            and tt_score != NO_SCORE
            and tte.bound & (BOUND_LOWER if tt_score >= beta else BOUND_UPPER)
        ):
            return tt_score

        if in_check:
            raw_eval = ply.static_eval = eval_ = NO_SCORE
            best_score = futility = -MATE
            
        else:
            correction = get_correction(position, ply)
            
            if tt_hit:
                raw_eval = tte.raw_eval
                
                # Evaluate if there is no evaluation.
                if raw_eval == NO_SCORE:
                    raw_eval = evaluate(position)

                ply.static_eval = eval_ = clamp_score(raw_eval + correction)
                    
                # Use transposition table value as a better evaluation.
                if (
                    tt_score != NO_SCORE
                    and tte.bound & (BOUND_LOWER if tt_score > eval_ else BOUND_UPPER)
                ):
                    eval_ = tt_score
                    
            else:
                raw_eval = evaluate(position)
                ply.static_eval = eval_ = clamp_score(raw_eval + correction)

            if eval_ >= beta:

                if not tt_hit:
                    tte.save(key, Move.none(), eval_, raw_eval, -1, False, BOUND_LOWER, int(ply))
                
                return eval_
            
            if eval_ > alpha:
                alpha = eval_

            best_score = eval_
            futility = ply.static_eval + 150

        previous_square = (ply - 1).move.to if (ply - 1).move else NO_SQUARE

        best_move = Move.none()

        picker = MovePicker(ply, tt_move, depth, history, capture_history)

        for move in picker:
            if not position.is_legal(move):
                continue

            ply.move_count = move_count = move_count + 1

            piece = position.piece(move)
            gives_check = position.gives_check(move)
            is_capture = bool(captured := position.captured(move))

            if best_score >= -MAX_EVAL:

                if (
                    futility >= -MAX_EVAL
                    and not gives_check
                    and not move.is_promotion
                    and move.to != previous_square
                ):
                    if move_count > 2:
                        continue
                    
                    if futility <= alpha and not see(position, move, 1):
                        best_score = max(best_score, futility)
                        continue

                if not see(position, move, -40):
                    continue

            (ply).move = move
            (ply).continuation_history = continuation_histories[in_check, is_capture, piece, move]
            (ply + 1).position = position.do(move, gives_check)
            
            score = -qsearch(ply + 1, depth - 1, -beta, -alpha)

            if score > best_score: 
                best_score = score

                if score > alpha:
                    best_move = move

                    if is_pv:
                        ply.line = (move,) + (ply + 1).line

                    if score >= beta:
                        break

                    alpha = score

        if move_count == 0 and in_check:
            return -MATE + int(ply)

        if best_score >= beta and abs(best_score) <= MAX_EVAL:
            best_score = int(best_score / 2 + beta / 2)

        bound = BOUND_LOWER if best_score >= beta else BOUND_UPPER
        tte.save(key, best_move, best_score, raw_eval, 0, was_pv, bound, int(ply))

        return best_score

    def update_continuation_histories(ply: Ply, move: Move, bonus: int) -> None:
        position = ply.position
        piece, to = position.piece(move), move.to
        
        for i in 1, 2, 4:
            if i > 2 and position.checkers:
                break
            if (ply - i).move:
                (ply - i).continuation_history[piece, to].update(bonus)

    def update_non_capture_non_promotion_histories(ply: Ply, move: Move, bonus: int) -> None:
        history[ply.position.turn, move].update(bonus)
        update_continuation_histories(ply, move, bonus)

    def get_correction(position: Position, ply: Ply) -> int:
        a = non_pawn_correction_history[position, WHITE].value
        b = non_pawn_correction_history[position, BLACK].value
        c = pawn_correction_history[position].value

        return int((a + b) / 20 + c / 20)
            
    while True:
        barrier.wait()
        
        start = time.perf_counter()

        if exit_flag.value == True:
            return

        tally.nodes = tally.tbhits = 0
        
        ply = Ply.new_stack()
        ply.position = position = limits.position.copy()
                
        root_moves = [RootMove(line=(move,)) for move in limits.searchable]

        if not root_moves:
            if not position.legal_moves:
                print('info depth 0 score', 'mate' if position.checkers else 'cp', 0, flush=True)
            print('bestmove (none)', flush=True)
            continue

        increment = getattr(limits, 'winc' if position.turn == WHITE else 'binc')
        total_time = getattr(limits, 'wtime' if position.turn == WHITE else 'btime') or float('inf')
        max_nodes = getattr(limits, 'nodes') or float('inf')
        move_time = getattr(limits, 'movetime') or float('inf')
        moves_left = getattr(limits, 'movestogo') or float('inf')
        max_depth = getattr(limits, 'depth') or float('inf')
            
        moves_left = min(moves_left, 30)
        max_depth = min(max_depth, MAX_DEPTH)
            
        # Convert milliseconds to seconds.
        increment /= 1000
        total_time /= 1000
        move_time /= 1000

        if MAIN_WORKER:
            time_left = min(total_time, move_time) - (options['MoveOverhead'] / 1000)
        
            if use_time_management := total_time != float('inf'):
                time_max = min(time_left, (time_left + increment) / 2)
                time_allocated = time_base = min(time_max, max(time_left / moves_left, increment))

                # Don't spend effort for only-moves.
                if len(root_moves) == 1:
                    max_depth = 1

                move_stability = 0
                
            else:
                time_allocated = time_left

            print('info string NNUE numpy is', 'ENABLED' if nnue.np else 'DISABLED', flush=True)
            
        for root_depth in range(1, max_depth + 1):
            searchable = limits.searchable.copy()

            for root_move in root_moves:
                root_move.previous_score = root_move.score
                
            for root_index, root_move in enumerate(root_moves[:options['MultiPV']]):
                depth = root_depth
                seldepth = 0
                alpha = -MATE
                beta = MATE

                if root_depth >= 5:
                    mean_score = root_move.mean_score
                    delta = 11
                    alpha = max(mean_score - delta, -MATE)
                    beta = min(mean_score + delta, MATE)
                
                while not limits.stopped:
                    root_delta = beta - alpha
                    
                    try:
                        score = search(ply, depth, alpha, beta, False)
                    except SearchStopped:
                        break
                    finally:
                        root_moves.sort(reverse=True)

                    if (
                        MAIN_WORKER
                        and options['MultiPV'] == 1
                        and time_elapsed() > 3.0
                        and (score <= alpha or score >= beta)
                    ):
                        print_info(alpha, beta)

                    if score <= alpha:
                        depth = root_depth
                        beta = int(alpha / 2 + beta / 2)
                        alpha = max(alpha - delta, -MATE)

                        if MAIN_WORKER and use_time_management and root_index == 0:
                            time_allocated = time_max

                    elif score >= beta:
                        depth = max(depth - 1, 1)
                        beta = min(beta + delta, MATE)
                        
                    else:
                        # Don't search the same move again.
                        searchable.remove(root_move.move)
                        break

                    delta += delta // 3

            best_move = root_moves[0].move
            best_score = root_moves[0].score

            # Calculate how much time to spend next search iteration.
            if MAIN_WORKER and use_time_management and root_depth > 1:
                time_allocated = time_base

                score_drop = max(previous_best_score - best_score, 0)
                time_allocated *= 1.00 + min(2.00, score_drop / 50)

                move_stability = (move_stability + 1) if best_move == previous_best_move else 0
                time_allocated *= 1.00 - 0.050 * min(move_stability, 10)
                
                effort_not_on_best_move = 1 - (root_moves[0].nodes / tally.nodes)
                time_allocated *= 1.00 + 2.00 * effort_not_on_best_move

                time_allocated = min(time_allocated, time_max)

            previous_best_move = best_move
            previous_best_score = best_score

            if MAIN_WORKER and (not limits.stopped or options['MultiPV'] != 1):
                print_info(-MATE, MATE)

            if limits.stopped:
                break

        if not MAIN_WORKER:
            continue

        # Print info again if node limit was reached.
        if total_nodes() >= max_nodes:
            print_info(-MATE, MATE)

        print(f'bestmove {best_move.uci(CHESS960)}', flush=True)
