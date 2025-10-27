#!/bin/sh

set -e

RAGCLIENT_PORT=8002

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

streamlit run main.py --server.port $RAGCLIENT_PORT
