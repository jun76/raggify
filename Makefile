VENV := .venv
ifeq ($(RUNNER_OS),Windows)
PY := $(VENV)/Scripts/python.exe
else
PY := $(VENV)/bin/python
endif
UV := uv
CLIP_PKG := "clip@git+https://github.com/openai/CLIP.git"
TOOL_DIR := $(shell $(UV) tool dir)
TOOL_PY := $(TOOL_DIR)/raggify/bin/python

.PHONY: venv min api all tools test clean

venv:
	$(UV) sync --all-packages

min: venv
	$(MAKE) tools

api: venv
	$(UV) sync --all-packages \
		--extra text \
		--extra image \
		--extra audio \
		--extra video \
		--extra rerank \
		--extra postgres \
		--extra redis \
		--extra exam
	$(MAKE) tools

all: venv
	$(UV) sync --all-packages --extra all
	$(MAKE) tools
	$(UV) pip install $(CLIP_PKG)
	$(UV) pip install --python $(TOOL_PY) $(CLIP_PKG)

tools:
	$(UV) tool install -e ./raggify
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

.DEFAULT_GOAL := all
