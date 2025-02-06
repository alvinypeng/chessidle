from __future__ import annotations

import ctypes
import multiprocessing as mp
import time

import chessidle
import chessidle.tune
import chessidle.bench
from chessidle.engine import Engine
from chessidle.move import Move
from chessidle.move_generation import perft_divide
from chessidle.options import OPTION_RANGES
from chessidle.position import Position, START_FEN


def uci_loop(engine: Engine) -> None:
    position = Position(START_FEN)

    art = r'''
           +
          _^_       o
         \   /    \^^^/      o
          )_(      )_(      (/)    _-=^_     _ _ _ 
         <--->    <--->     /_\    )    \   ||-|-||    (_) 
          | |      | |      ) (   [_/)  /    |---|     < >
         /___\    /___\    /___\    /___\    |___|    _/_\_
        (_____)  (_____)  (_____)  (_____)  (_____)  (_____)  Chessidle
    '''
    print(chessidle.tune.CONFIG_STRING if engine.tune else art)

    while True:
        string = input()

        chess960 = engine.options['UCI_Chess960']

        def tokens_after(token: str) -> list[str]:
            return string.split(token)[1].split()

        def get_move(token: str) -> Move:
            for move in position.legal_moves:
                if move.uci(chess960) == token:
                    return move
            raise ValueError('no move:', token)

        if string == 'uci':
            print('id name Chessidle', chessidle.__version__)
            print('id author', chessidle.__author__)
            engine.print_options()
            print('uciok')

        elif string == 'ucinewgame':
            position = Position(START_FEN)
            engine.set_position(position)
            engine.clear_hash()

        elif string == 'isready':
            print('readyok')

        elif string.startswith('setoption'):
            name = tokens_after('name')[0]
            value = tokens_after('value')[0]
            type_ = engine.option_ranges[name].type_
            value = type_(value) if type_ != bool else {'true': True, 'false': False}[value]
            engine.change_option(name, value)

        elif string.startswith('position'):
            # All tokens after "fen" but before "moves".
            fen = string.split('fen ')[1].split(' moves')[0] if 'fen' in string else START_FEN
            position = Position(fen)

            if 'moves' in string:
                for token in tokens_after('moves'):
                    position = position.do(get_move(token))

            engine.set_position(position)

        elif string.startswith('go perft'):
            perft_divide(position, int(tokens_after('go perft')[0]), chess960)

        elif string.startswith('go'):
            args = {}
            
            # Time, depth, node limits. Set to zero if not given.
            for name in 'depth', 'nodes', 'movetime', 'wtime', 'btime', 'winc', 'binc', 'movestogo':
                args[name] = int(tokens_after(name)[0]) if name in string else None

            # Sometimes, a user wants only certain moves searched.
            if 'searchmoves' in string:
                args['moves'] = {get_move(token) for token in tokens_after('searchmoves')}

            engine.start_searching(**args)

        elif string == 'stop':
            engine.force_stop()

        elif string == 'quit':
            break

        # Nonstandard UCI commands

        elif string == 'bench':
            chessidle.bench.run_benchmark(engine, chessidle.bench.BENCH_FENS)

        elif string == 'd':
            position.pretty_print(chess960)

        elif string == 'eval':
            ...
