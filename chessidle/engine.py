from __future__ import annotations

import ctypes
import multiprocessing as mp
import threading
import time

import chessidle.tune
from chessidle.limits import Limits, Tally
from chessidle.options import Options, OPTION_RANGES
from chessidle.position import Position, START_FEN
from chessidle.search import standby
from chessidle.transposition import TranspositionTable


# Needed for pyinstaller.
mp.freeze_support()


class Engine:

    def __init__(self, options: Options, tune: bool = False) -> None:
        self.limits = mp.Value(Limits, lock=False)
        self.options = options
        self.tune = tune
        self.create_worker_pool()

        self.option_ranges = OPTION_RANGES.copy()

        if self.tune:
            self.option_ranges |= chessidle.tune.TUNE_OPTION_RANGES.copy()

        self.set_position(Position(START_FEN))

    @property
    def worker_count(self) -> int:
        return self.options['Threads']

    def change_option(self, name: str, value: int | bool | str) -> None:
        self.delete_worker_pool()
        self.options[name] = value
        self.create_worker_pool()

    def set_position(self, position: Position) -> None:
        self.limits.set_position(position)

    def create_worker_pool(self) -> None:
        if self.tune:
            assert self.worker_count == 1, 'Threads must be 1 for tuning'
            exec(chessidle.tune.modified_code(self.options), globals())

        if self.worker_count == 1:
            Worker = threading.Thread
            Barrier = threading.Barrier
        else:
            Worker = mp.Process
            Barrier = mp.Barrier

        assert not hasattr(self, 'workers')
        assert not hasattr(self, 'exit_flag')
        assert not hasattr(self, 'barrier')
        assert not hasattr(self, 'tallies')
        assert not hasattr(self, 'transposition_table')

        self.workers = []
        self.exit_flag = mp.Value(ctypes.c_bool, False, lock=False)
        self.barrier = Barrier(self.worker_count + 1)
        self.tallies = [mp.Value(Tally, lock=False) for _ in range(self.worker_count)]
        self.transposition_table = TranspositionTable(self.options['Hash'])

        for worker_id in range(self.worker_count):
            worker = Worker(
                target=standby,
                args=[
                    worker_id,
                    self.limits,
                    self.options,
                    self.exit_flag,
                    self.barrier,
                    self.tallies,
                    self.transposition_table,
                ],
            )
            worker.start()
            self.workers.append(worker)

        # Wait for workers to finish setting up.
        self.busy_wait_until_ready_to_search()

    def delete_worker_pool(self) -> None:
        assert self.barrier.n_waiting == self.worker_count
        self.exit_flag.value = True
        self.barrier.wait()

        del self.workers
        del self.exit_flag
        del self.barrier
        del self.tallies
        del self.transposition_table

    def clear_hash(self) -> None:
        # Hacky way to "clear hash".
        self.delete_worker_pool()
        self.create_worker_pool()

    def start_searching(
        self,
        depth: int | None = None,
        nodes: int | None = None,
        movetime: int | None = None,
        wtime: int | None = None,
        btime: int | None = None,
        winc: int | None = None,
        binc: int | None = None,
        movestogo: int | None = None,
        moves: set[Move] | None = None
    ) -> None:
        self.limits.depth = depth or 0
        self.limits.nodes = nodes or 0
        self.limits.movetime = movetime or 0
        self.limits.wtime = wtime or 0
        self.limits.btime = btime or 0
        self.limits.winc = winc or 0
        self.limits.binc = binc or 0
        self.limits.movestogo = movestogo or 0
        self.limits.searchable = moves or set(self.limits.position.legal_moves)

        self.limits.stopped = False

        # Allow workers to pass the barrier and start searching.
        assert self.barrier.n_waiting == self.worker_count
        self.barrier.wait()

    def busy_wait_until_ready_to_search(self, interval: float = 0.0) -> None:
        while self.barrier.n_waiting != self.worker_count:
            time.sleep(interval)

    def force_stop(self) -> None:
        self.limits.stopped = True
        self.busy_wait_until_ready_to_search()

    def get_node_count(self) -> int | None:
        assert hasattr(self, 'tallies')
        return sum(tally.nodes for tally in self.tallies)

    def print_options(self) -> None:
        string = '\n'.join(
            str(option_range).format(name) for name, option_range in self.option_ranges.items()
        )
        print(string)
