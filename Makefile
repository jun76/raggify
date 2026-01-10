VENV := .venv

UV := uv
TOOL_DIR := $(shell $(UV) tool dir)

ifeq ($(RUNNER_OS),Windows)
PY := $(VENV)/Scripts/python.exe
TOOL_PY := $(TOOL_DIR)/raggify/Scripts/python.exe
else
PY := $(VENV)/bin/python
TOOL_PY := $(TOOL_DIR)/raggify/bin/python
endif

TOOL_PIP := $(UV) pip install --python $(TOOL_PY)
RAGGIFY_ALL_EXTRA := './raggify[all]'

.PHONY: venv min all tools test clean

venv:
	$(UV) sync --all-packages

min: venv
	$(MAKE) tools

all: venv
	$(UV) sync --all-packages --extra all
	$(MAKE) tools
	$(TOOL_PIP) -e $(RAGGIFY_ALL_EXTRA)

tools:
	-$(UV) tool uninstall raggify >/dev/null 2>&1
	$(UV) tool install -e ./raggify
	-$(UV) tool uninstall raggify-client >/dev/null 2>&1
	$(UV) tool install -e ./raggify-client

upgrade-all:
	$(UV) lock --upgrade
	$(MAKE) all

test:
	$(PY) -m pytest --maxfail=1 \
		--cov=raggify \
		--cov=raggify_client \
		--cov-report=term-missing \
		--cov-report=xml

clean:
	rm -rf $(VENV)
	rm -f uv.lock

.DEFAULT_GOAL := all
