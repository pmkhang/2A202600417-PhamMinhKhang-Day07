VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv

.PHONY: install test run clean

install:
	$(UV) venv $(VENV)
	$(UV) pip install -r requirements.txt --python $(PYTHON)

test:
	$(PYTHON) -m pytest tests/ -v

run:
	$(PYTHON) main.py

clean:
	rm -rf $(VENV) __pycache__ .pytest_cache
