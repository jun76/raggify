VENV := .venv

ifeq ($(RUNNER_OS),Windows)
	PY := $(VENV)/Scripts/python.exe
else
	PY := $(VENV)/bin/python
endif

PIP := $(PY) -m pip
CLIP_PKG := "clip@git+https://github.com/openai/CLIP.git"
WHISPER_PKG := "openai-whisper@git+https://github.com/openai/whisper.git"
TOOL_PY := $(shell uv tool dir)/raggify/bin/python

install:
	uv venv $(VENV)
	uv pip install --python $(PY) --upgrade pip
	$(PIP) install -e raggify[exam,dev]
	$(PIP) install -e raggify-client

	uv tool install --reinstall -e ./raggify
	uv tool install --reinstall -e ./raggify-client

all: install
	$(PIP) install -e raggify[all]
	$(PIP) install $(CLIP_PKG)
	$(PIP) install $(WHISPER_PKG)
	
	uv pip install $(CLIP_PKG)
	uv pip install $(WHISPER_PKG)
	uv pip install --python $(TOOL_PY) $(CLIP_PKG)
	uv pip install --python $(TOOL_PY) $(WHISPER_PKG)


.DEFAULT_GOAL := install

test:
	uv pip install --python $(PY) --upgrade pip
	env \
	  OPENAI_API_KEY=dummy \
	  COHERE_API_KEY=dummy \
	  VOYAGE_API_KEY=dummy \
	  AWS_ACCESS_KEY_ID=dummy \
	  AWS_SECRET_ACCESS_KEY=dummy \
	  AWS_REGION=us-east-1 \
	  RG_CONFIG_PATH=tests/config.yaml \
	  $(PY) -m pytest --maxfail=1 --cov=raggify --cov=raggify_client --cov-report=term-missing --cov-report=xml
