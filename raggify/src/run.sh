#!/bin/sh

set -e

RAGGIFY_HOST=0.0.0.0
RAGGIFY_PORT=8000

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")
cd "$PROJECT_ROOT"

uvicorn raggify.server.fastapi:app --host "$RAGGIFY_HOST" --port "$RAGGIFY_PORT"
