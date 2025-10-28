install:
	uv sync
	uv tool install -e '.[local, exam]'
