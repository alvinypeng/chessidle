import pytest

from chessidle.position import Position
from chessidle.move_generation import perft


@pytest.mark.parametrize('fen, depth, expected_nodes', [
    ('4k3/8/8/8/8/8/8/r1R1KR1r w FC - 0 1', 4, 64786),
    # https://www.chessprogramming.org/Perft_Results
    ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 5, 4865609),
    ('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', 4, 4085603),
    ('8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', 5, 674624),
    ('r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1', 4, 422333),
    ('r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1', 4, 422333),
    ('rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8', 4, 2103487),
    ('r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10', 4, 3894594),
    # https://github.com/AndyGrant/Ethereal/blob/master/src/perft/fischer.epd
    ('bqnb1rkr/pp3ppp/3ppn2/2p5/5P2/P2P4/NPP1P1PP/BQ1BNRKR w HFhf - 2 9', 4, 326672),
    ('2nnrbkr/p1qppppp/8/1ppb4/6PP/3PP3/PPP2P2/BQNNRBKR w HEhe - 1 9', 4, 667366),
    ('b1q1rrkb/pppppppp/3nn3/8/P7/1PPP4/4PPPP/BQNNRKRB w GE - 1 9', 4, 273318),
    ('qbbnnrkr/2pp2pp/p7/1p2pp2/8/P3PP2/1PPP1KPP/QBBNNR1R w hf - 0 9', 4, 382958),
])
def test_move_generation(fen: str, depth: int, expected_nodes: int) -> None:
    assert perft(Position(fen), depth, debug=True) == expected_nodes
