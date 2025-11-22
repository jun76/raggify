VENV := .venv
PY := $(VENV)/bin/python
PIP := $(PY) -m pip
CLIP_PKG := "clip@git+https://github.com/openai/CLIP.git"
WHISPER_PKG := "openai-whisper@git+https://github.com/openai/whisper.git"
TOOL_PY := $(shell uv tool dir)/raggify/bin/python

venv:
	@test -x $(PY) || uv venv $(VENV)
	uv pip install --python $(PY) --upgrade pip

install: venv
	$(PIP) install -e raggify[exam]
	$(PIP) install -e raggify-client
	$(PIP) install $(CLIP_PKG)
	$(PIP) install $(WHISPER_PKG)

tools:
	uv tool install --reinstall -e ./raggify
	uv tool install --reinstall -e ./raggify-client
	uv pip install $(CLIP_PKG)
	uv pip install $(WHISPER_PKG)
	uv pip install --python $(TOOL_PY) $(CLIP_PKG)
	uv pip install --python $(TOOL_PY) $(WHISPER_PKG)

all: install tools
