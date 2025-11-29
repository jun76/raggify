VENV := .venv
OS := $(shell uname 2>/dev/null || echo Windows)

ifeq ($(OS),Windows)
PY := $(VENV)/Scripts/python.exe
else
PY := $(VENV)/bin/python
endif

PIP := $(PY) -m pip
CLIP_PKG := "clip@git+https://github.com/openai/CLIP.git"
WHISPER_PKG := "openai-whisper@git+https://github.com/openai/whisper.git"
TOOL_PY := $(shell uv tool dir)/raggify/bin/python

venv:
	uv venv $(VENV)
ifeq ($(OS),Windows)
	uv pip install --python $(VENV)/Scripts/python.exe --upgrade pip
else
	uv pip install --python $(VENV)/bin/python --upgrade pip
endif

install: venv
	$(PIP) install -e raggify[exam,dev]
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

test: venv
	env \
	  OPENAI_API_KEY=dummy \
	  COHERE_API_KEY=dummy \
	  VOYAGE_API_KEY=dummy \
	  AWS_ACCESS_KEY_ID=dummy \
	  AWS_SECRET_ACCESS_KEY=dummy \
	  AWS_REGION=us-east-1 \
	  $(PY) -m pytest --maxfail=1 --cov=raggify --cov=raggify_client --cov-report=term-missing --cov-report=xml
