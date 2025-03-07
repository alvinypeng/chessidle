from __future__ import annotations

import time

from chessidle.engine import Engine
from chessidle.position import Position

# Stockfish bench defaults. 
BENCH_FENS = [
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 10',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 11',
    '4rrk1/pp1n3p/3q2pQ/2p1pb2/2PP4/2P3N1/P2B2PP/4RRK1 b - - 7 19',
    'rq3rk1/ppp2ppp/1bnpb3/3N2B1/3NP3/7P/PPPQ1PP1/2KR3R w - - 7 14',
    'r1bq1r1k/1pp1n1pp/1p1p4/4p2Q/4Pp2/1BNP4/PPP2PPP/3R1RK1 w - - 2 14',
    'r3r1k1/2p2ppp/p1p1bn2/8/1q2P3/2NPQN2/PPP3PP/R4RK1 b - - 2 15',
    'r1bbk1nr/pp3p1p/2n5/1N4p1/2Np1B2/8/PPP2PPP/2KR1B1R w kq - 0 13',
    'r1bq1rk1/ppp1nppp/4n3/3p3Q/3P4/1BP1B3/PP1N2PP/R4RK1 w - - 1 16',
    '4r1k1/r1q2ppp/ppp2n2/4P3/5Rb1/1N1BQ3/PPP3PP/R5K1 w - - 1 17',
    '2rqkb1r/ppp2p2/2npb1p1/1N1Nn2p/2P1PP2/8/PP2B1PP/R1BQK2R b KQ - 0 11',
    'r1bq1r1k/b1p1npp1/p2p3p/1p6/3PP3/1B2NN2/PP3PPP/R2Q1RK1 w - - 1 16',
    '3r1rk1/p5pp/bpp1pp2/8/q1PP1P2/b3P3/P2NQRPP/1R2B1K1 b - - 6 22',
    'r1q2rk1/2p1bppp/2Pp4/p6b/Q1PNp3/4B3/PP1R1PPP/2K4R w - - 2 18',
    '4k2r/1pb2ppp/1p2p3/1R1p4/3P4/2r1PN2/P4PPP/1R4K1 b - - 3 22',
    '3q2k1/pb3p1p/4pbp1/2r5/PpN2N2/1P2P2P/5PP1/Q2R2K1 b - - 4 26',
]


def run_benchmark(engine: Engine, fens: list[str]) -> None:
    total_nodes = 0
    total_time = 1e-9
    
    for i, fen in enumerate(fens):
        print(f'\nPosition: {i + 1}/{len(fens)}', f'({fen})')
            
        engine.clear_hash()
        engine.set_position(Position(fen))

        start = time.perf_counter()
        engine.start_searching(depth=12)
        engine.busy_wait_until_ready_to_search(interval=1.0)

        total_nodes += engine.get_node_count()
        total_time += time.perf_counter() - start

    print('Total time (ms) :', int(total_time * 1000))
    print(total_nodes, 'nodes', int(total_nodes / total_time), 'nps')
