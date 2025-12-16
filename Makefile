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
TOOL_PIP := $(UV) pip install --python $(TOOL_PY)
RAGGIFY_ALL_EXTRA := './raggify[all]'
RAGGIFY_API_EXTRA := './raggify[text,image,audio,video,rerank,postgres,redis,exam,dev]'

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
		--extra exam \
		--extra dev
	$(MAKE) tools
	$(TOOL_PIP) -e $(RAGGIFY_API_EXTRA)

all: venv
	$(UV) sync --all-packages --extra all
	$(MAKE) tools
	$(UV) pip install $(CLIP_PKG)
	$(TOOL_PIP) -e $(RAGGIFY_ALL_EXTRA)
	$(TOOL_PIP) $(CLIP_PKG)

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

.DEFAULT_GOAL := all
