PYTHON ?= python3.11
VENV := .venv
VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip

# Windows venvs put executables in Scripts/, not bin/.
ifeq ($(OS),Windows_NT)
VENV_PY := $(VENV)/Scripts/python.exe
VENV_PIP := $(VENV)/Scripts/pip.exe
endif

.PHONY: install test lint run-sweep figures

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV_PIP) install --upgrade pip
	$(VENV_PIP) install -e ".[dev]"
	$(VENV_PIP) freeze --exclude-editable > requirements.lock
	$(VENV_PY) scripts/record_environment.py

test:
	$(VENV_PY) -m pytest

lint:
	$(VENV_PY) -m ruff check src tests scripts

run-sweep:
	$(VENV_PY) scripts/run_sweep.py

figures: run-sweep
	$(VENV_PY) scripts/make_figures.py
