import pytest

from chessidle.position import Position


@pytest.mark.parametrize('fen, key', [
    ('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 0x463B96181691FC9C),
    ('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1', 0x823C9B50FD114196),
    ('rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2', 0x756B94461C50FB0),
    ('rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2', 0x662FAFB965DB29D4),
    ('rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3', 0x22A48B5A8E47FF78),
    ('rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR b kq - 0 3', 0x652A607CA3F242C1),
    ('rnbq1bnr/ppp1pkpp/8/3pPp2/8/8/PPPPKPPP/RNBQ1BNR w - - 0 4', 0xFDD303C946BDD9),
    ('rnbqkbnr/p1pppppp/8/8/PpP4P/8/1P1PPPP1/RNBQKBNR b KQkq c3 0 3', 0x3C8123EA7B067637),
    ('rnbqkbnr/p1pppppp/8/8/P6P/R1p5/1P1PPPP1/1NBQKBNR b Kkq - 0 4', 0x5C3F9B829B279560),
])
def test_polyglot(fen: str, key: int) -> None:
    assert Position(fen).key == key
