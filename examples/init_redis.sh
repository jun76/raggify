#!/usr/bin/env bash

set -euo pipefail

REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"
REDIS_USERNAME="${REDIS_USERNAME:-}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"

usage() {
  cat <<USAGE
Usage:
  ./init_redis.sh          Show basic info of Redis target.
  ./init_redis.sh --reset  Flush Redis DB (dangerous).
USAGE
}

if ! command -v redis-cli >/dev/null 2>&1; then
  echo "[error] redis-cli is not installed" >&2
  exit 1
fi

redis_cmd=(redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -n "$REDIS_DB")
if [[ -n "$REDIS_USERNAME" ]]; then
  redis_cmd+=(--user "$REDIS_USERNAME")
fi
if [[ -n "$REDIS_PASSWORD" ]]; then
  redis_cmd+=(-a "$REDIS_PASSWORD")
fi

if [[ ${1:-} == "-h" || ${1:-} == "--help" ]]; then
  usage
  exit 0
fi

if [[ ${1:-} == "--reset" ]]; then
  "${redis_cmd[@]}" PING >/dev/null
  echo "[reset] FLUSHDB redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
  "${redis_cmd[@]}" FLUSHDB
  echo "[reset] Done."
  exit 0
fi

info=$("${redis_cmd[@]}" INFO server | head -n 5)
echo "[info] Redis info:"$'\n'$info
