install:
	uv sync --all-extras
	uv tool install --reinstall -e '.[exam]'

	uv pip install clip@git+https://github.com/openai/CLIP.git
	uv pip install openai-whisper@git+https://github.com/openai/whisper.git
	#uv pip install llama-index-embeddings-cohere@git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-integrations/embeddings/llama-index-embeddings-cohere

	uv pip install --python $(shell uv tool dir)/raggify/bin/python "clip@git+https://github.com/openai/CLIP.git"
	uv pip install --python $(shell uv tool dir)/raggify/bin/python "openai-whisper@git+https://github.com/openai/whisper.git"
	#uv pip install --python $(shell uv tool dir)/raggify/bin/python "llama-index-embeddings-cohere@git+https://github.com/run-llama/llama_index.git#subdirectory=llama-index-integrations/embeddings/llama-index-embeddings-cohere"
