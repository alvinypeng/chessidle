EXE = chessidle

# system python interpreter. used only to create virtual environment
PY = python3
VENV = venv
BIN = $(VENV)/bin

RM = rm -rf
DEL = rm -f

EXT =

# make it work on windows too
ifeq ($(OS), Windows_NT)
    BIN = $(VENV)/Scripts
    PY = python

    RM = rmdir /s /q
    DEL = del /f /q

    EXT = .exe
endif

# Use numpy (yes/no)
numpy ?= yes

all: $(VENV)
	$(BIN)/pip install --upgrade pyinstaller
ifeq ($(numpy), yes)
	$(BIN)/pip install --upgrade numpy
endif
	$(BIN)/pyinstaller --onefile --distpath "." -n $(EXE) main.py

$(VENV):
	$(PY) -m venv $(VENV)

clean:
	-$(RM) $(VENV) build __pycache__
	-$(DEL) $(EXE).spec $(EXE)$(EXT)