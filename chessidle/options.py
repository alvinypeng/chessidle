from __future__ import annotations

from typing import NamedTuple


Options = dict[str, object]


class OptionRange(NamedTuple):
    type_: type
    default: int | bool | str
    minimum: int | None = None
    maximum: int | None = None

    def __str__(self) -> str:
        t = {int: 'spin', bool: 'check', str: 'string'}[self.type_]
        min_and_max = f'min {self.minimum} max {self.maximum}' if t == 'spin' else ''
        return 'option name {} ' + f'type {t} default {self.default} '.lower() + min_and_max


OPTION_RANGES = {
    'Hash': OptionRange(int, 16, 1, 262144),
    'Threads': OptionRange(int, 1, 1, 256),
    'MultiPV': OptionRange(int, 1, 1, 256),
    'UCI_Chess960': OptionRange(bool, False),
    'MoveOverhead': OptionRange(int, 300, 0, 30000),
}

OPTION_DEFAULTS = {key: value.default for key, value in OPTION_RANGES.items()}
