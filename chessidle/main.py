from __future__ import annotations

import argparse
from collections.abc import Sequence

from chessidle.bench import BENCH_FENS, run_benchmark
from chessidle.engine import Engine
from chessidle.options import OPTION_DEFAULTS
from chessidle.tune import TUNE_OPTION_DEFAULTS
from chessidle.uci import uci_loop


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('-tune', action='store_true', help='Enable tuning mode')
    subparsers = parser.add_subparsers(dest='command', required=False)
    subparsers.add_parser('bench', help='Run benchmark')
    args = parser.parse_args(argv)

    tune = args.tune
    options = OPTION_DEFAULTS | (TUNE_OPTION_DEFAULTS if tune else {})
    engine = Engine(options, tune)
    run_benchmark(engine, BENCH_FENS) if args.command == 'bench' else uci_loop(engine)
    engine.delete_worker_pool()

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
