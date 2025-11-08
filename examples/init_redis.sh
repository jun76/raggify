#!/usr/bin/env bash

# set -e: stop on error / -u: treat unset vars as errors / -o pipefail: detect failures in pipelines
set -euo pipefail

# =========================
# Configuration (adjust via environment variables)
# =========================
REDIS_HOST="${REDIS_HOST:-localhost}"
REDIS_PORT="${REDIS_PORT:-6379}"
REDIS_DB="${REDIS_DB:-0}"
REDIS_USERNAME="${REDIS_USERNAME:-}"
REDIS_PASSWORD="${REDIS_PASSWORD:-}"
KEY_PREFIX="${KEY_PREFIX:-raggify}"

# =========================
# Helper functions
# =========================
usage() {
  cat <<EOF
Usage:
  ./init_redis.sh          Check connectivity and show a summary of existing raggify keys/indexes
  ./init_redis.sh --reset  Drop RediSearch indexes and delete Redis keys that start with KEY_PREFIX (${KEY_PREFIX})

Environment overrides:
  REDIS_HOST     (default: ${REDIS_HOST})
  REDIS_PORT     (default: ${REDIS_PORT})
  REDIS_DB       (default: ${REDIS_DB})
  REDIS_USERNAME (default: unset)
  REDIS_PASSWORD (default: unset)
  KEY_PREFIX     (default: ${KEY_PREFIX})
EOF
}

trim_ws() {
  local s="$1"
  s="${s#"${s%%[![:space:]]*}"}"
  s="${s%"${s##*[![:space:]]}"}"
  printf "%s" "$s"
}

require_redis_cli() {
  if ! command -v redis-cli >/dev/null 2>&1; then
    echo "[error] redis-cli is not installed or not in PATH." >&2
    exit 1
  fi
}

build_redis_cmd() {
  local -n ref=$1
  ref=(redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" -n "${REDIS_DB}")
  if [[ -n "${REDIS_USERNAME}" ]]; then
    ref+=(--user "${REDIS_USERNAME}")
  fi
  if [[ -n "${REDIS_PASSWORD}" ]]; then
    ref+=(-a "${REDIS_PASSWORD}")
  fi
}

redis_cmd=()
build_redis_cmd redis_cmd

redis_exec() {
  "${redis_cmd[@]}" "$@"
}

redis_scan_pattern() {
  local pattern="$1"
  "${redis_cmd[@]}" --scan --pattern "${pattern}"
}

ping_redis() {
  redis_exec PING >/dev/null
  echo "[info] Connected to redis://${REDIS_HOST}:${REDIS_PORT}/${REDIS_DB}"
}

summarize_keys() {
  local count
  count=$(redis_scan_pattern "${KEY_PREFIX}*" | wc -l | tr -d '[:space:]' || true)
  echo "[info] Keys matching prefix '${KEY_PREFIX}': ${count}"
}

summarize_indexes() {
  local raw
  if ! raw=$(redis_exec FT._LIST 2>/dev/null); then
    echo "[info] FT._LIST is unavailable (RediSearch not loaded?); skipping index summary"
    return
  fi

  raw=$(printf "%s" "${raw}" | tr -d '\r"')
  if [[ -z "${raw}" ]]; then
    echo "[info] No RediSearch indexes found"
    return
  fi

  local matches=()
  mapfile -t entries <<<"${raw}"
  for idx in "${entries[@]}"; do
    [[ -z "${idx}" ]] && continue
    idx=$(trim_ws "${idx}")
    [[ -z "${idx}" ]] && continue
    if [[ "${idx}" == "${KEY_PREFIX}"* ]]; then
      matches+=("${idx}")
    fi
  done

  if ((${#matches[@]} == 0)); then
    echo "[info] No RediSearch indexes matching prefix '${KEY_PREFIX}'"
    return
  fi

  echo "[info] RediSearch indexes matching prefix '${KEY_PREFIX}':"
  for idx in "${matches[@]}"; do
    echo "  - ${idx}"
  done
}

drop_indexes() {
  local raw
  if ! raw=$(redis_exec FT._LIST 2>/dev/null); then
    echo "[reset] FT._LIST failed; RediSearch module may be missing. Skipping index removal."
    return
  fi

  raw=$(printf "%s" "${raw}" | tr -d '\r"')
  if [[ -z "${raw}" ]]; then
    echo "[reset] No RediSearch indexes found."
    return
  fi

  local dropped=0
  mapfile -t entries <<<"${raw}"
  for idx in "${entries[@]}"; do
    [[ -z "${idx}" ]] && continue
    idx=$(trim_ws "${idx}")
    [[ -z "${idx}" ]] && continue
    if [[ "${idx}" == "${KEY_PREFIX}"* ]]; then
      if redis_exec FT.DROPINDEX "${idx}" DD >/dev/null 2>&1 || redis_exec FT.DROPINDEX "${idx}" >/dev/null 2>&1; then
        echo "[reset] Dropped index ${idx}"
        ((dropped++))
      fi
    fi
  done

  if ((dropped == 0)); then
    echo "[reset] No RediSearch indexes matched the prefix '${KEY_PREFIX}'."
  fi
}

delete_prefixed_keys() {
  local pattern="${KEY_PREFIX}*"
  local -a batch=()
  local deleted=0

  while IFS= read -r key; do
    [[ -z "${key}" ]] && continue
    batch+=("${key}")
    if ((${#batch[@]} >= 128)); then
      redis_exec DEL "${batch[@]}" >/dev/null || true
      deleted=$((deleted + ${#batch[@]}))
      batch=()
    fi
  done < <(redis_scan_pattern "${pattern}" || true)

  if ((${#batch[@]})); then
    redis_exec DEL "${batch[@]}" >/dev/null || true
    deleted=$((deleted + ${#batch[@]}))
  fi

  echo "[reset] Deleted ${deleted} keys (pattern: ${pattern})"
}

# =========================
# Main routine
# =========================
ARG="${1:-}"

if [[ "${ARG}" == "-h" || "${ARG}" == "--help" ]]; then
  usage
  exit 0
fi

require_redis_cli
ping_redis

if [[ "${ARG}" == "--reset" ]]; then
  echo "[reset] Removing RediSearch indexes (if any)..."
  drop_indexes
  echo "[reset] Removing prefixed keys..."
  delete_prefixed_keys
  echo "[reset] Done."
  exit 0
fi

summarize_keys
summarize_indexes
echo "[info] Use ./init_redis.sh --reset to remove existing data (KEY_PREFIX='${KEY_PREFIX}')."
