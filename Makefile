install:
	uv sync --all-extras
	uv tool install --reinstall -e '.[local, exam]'
