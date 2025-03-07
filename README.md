<div align="center">

<img src="https://github.com/user-attachments/assets/b125b54f-cd3f-4b50-9a63-95c918bd74cb" height="200">
<h2>Chessidle</h2>

</div>

Chessidle is a **Python chess engine** that can analyze standard and chess960 (Fischer random) chess positions.

## Installation

***Requires Python 3.9+***

Download the latest release from pypi:
```
pip install chessidle
```

or download the most recent development version from source:
```
git clone https://github.com/alvinypeng/chessidle
cd chessidle
pip install .
```

## Usage

### Run UCI

```
chessidle
```

Run as a module if running as a script does not work:
```
python -m chessidle
```

### Use with python-chess

It is possible to use Chessidle with [python-chess][python-chess-link]. Here is a small example script to play against Chessidle in the terminal.

```python
import chess
import chess.engine

board = chess.Board()
engine = chess.engine.SimpleEngine.popen_uci('chessidle')

while not board.is_game_over():
    print(board, '\n')

    s = input('Your move:') if board.turn else engine.play(board, chess.engine.Limit(time=3)).move.uci()
    
    for push in (board.push_uci, board.push_san):
        try:
            push(s)
            print(s, 'played \n')
            break
        except ValueError:
            pass
    else:
        print('Invalid move \n')

print(board, '\n\n', 'Game is over')
engine.quit()
```

## Supported UCI commands

### Options

```Hash``` ```Threads``` ```MultiPV``` ```UCI_Chess960``` ```MoveOverhead```

### Go

```wtime``` ```btime``` ```winc``` ```binc``` ```movestogo``` ```depth``` ```nodes``` ```movetime```


## Acknowledgements
Neural networks for this project are trained on [data generated by Leela Chess Zero][leela-data-link].

[python-chess-link]: https://github.com/niklasf/python-chess
[leela-data-link]: https://storage.lczero.org/files/training_data/
