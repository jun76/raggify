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

venv:
	@if [ ! -d "$(VENV)" ]; then uv venv $(VENV) --prompt raggify-dev; fi
	uv pip install --python $(PY) --upgrade pip

min: venv
	$(PIP) install -e raggify
	$(PIP) install -e raggify-client
	
	uv tool install --reinstall -e ./raggify
	uv tool install --reinstall -e ./raggify-client

api: venv
	$(PIP) install -e raggify[text,image,audio,video,rerank,postgres,redis,exam,dev]
	$(PIP) install -e raggify-client
	
	uv tool install --reinstall -e ./raggify
	uv tool install --reinstall -e ./raggify-client

all: venv
	$(PIP) install -e raggify[all]
	$(PIP) install -e raggify-client
	$(PIP) install $(CLIP_PKG)
	$(PIP) install $(WHISPER_PKG)
	
	uv tool install --reinstall -e ./raggify
	uv tool install --reinstall -e ./raggify-client
	uv pip install $(CLIP_PKG)
	uv pip install $(WHISPER_PKG)
	uv pip install --python $(TOOL_PY) $(CLIP_PKG)
	uv pip install --python $(TOOL_PY) $(WHISPER_PKG)

test:
	env \
	  OPENAI_API_KEY=dummy \
	  COHERE_API_KEY=dummy \
	  VOYAGE_API_KEY=dummy \
	  AWS_ACCESS_KEY_ID=dummy \
	  AWS_SECRET_ACCESS_KEY=dummy \
	  AWS_REGION=us-east-1 \
	  RG_CONFIG_PATH=tests/config.yaml \
	  $(PY) -m pytest --maxfail=1 --cov=raggify --cov=raggify_client --cov-report=term-missing --cov-report=xml

clean:
	rm -rf $(VENV)

.DEFAULT_GOAL := all
