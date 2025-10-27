#!/usr/bin/env bash

# set -e: stop on error / -u: treat unset vars as errors / -o pipefail: detect failures in pipelines
set -euo pipefail

# =========================
# Configuration (adjust as needed)
# =========================
PG_SUPERUSER="postgres"      # OS user with privileges (toggle via sudo -u)
DB_NAME="raggify"            # Application database name
APP_USER="raggify"           # Application DB user
APP_PASSWORD="raggify"       # Password for application DB user

# =========================
# Helper functions
# =========================
run_psql() {
  # Execute a psql command as superuser without selecting a DB
  local sql="$1"
  sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 -c "$sql"
}

run_psql_db() {
  # Execute a psql command as superuser against a specific DB
  local db="$1"
  local sql="$2"
  sudo -u "$PG_SUPERUSER" psql -v ON_ERROR_STOP=1 -d "$db" -c "$sql"
}

usage() {
  cat <<EOF
Usage:
  ./init_pgdb.sh          Run initial setup for PostgreSQL so raggify can use it
  ./init_pgdb.sh --reset  Drop and recreate the DB, then run setup again

Current configuration:
  PG_SUPERUSER=${PG_SUPERUSER}
  DB_NAME=${DB_NAME}
  APP_USER=${APP_USER}
EOF
}

# =========================
# Main routine
# =========================
ARG="${1:-}"

if [[ "$ARG" == "-h" || "$ARG" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$ARG" == "--reset" ]]; then
  echo "[reset] Dropping and recreating existing DB (${DB_NAME})..."
  # Terminate active sessions before dropping
  run_psql "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname = '${DB_NAME}' AND pid <> pg_backend_pid();" || true
  run_psql "DROP DATABASE IF EXISTS \"${DB_NAME}\";"
  run_psql "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${APP_USER}') THEN CREATE ROLE \"${APP_USER}\" LOGIN PASSWORD '${APP_PASSWORD}'; END IF; END \$\$;"
  run_psql "CREATE DATABASE \"${DB_NAME}\" OWNER \"${APP_USER}\";"
  run_psql_db "$DB_NAME" "CREATE EXTENSION IF NOT EXISTS vector;"
  run_psql_db "$DB_NAME" "GRANT CREATE, USAGE ON SCHEMA public TO \"${APP_USER}\";"
  echo "[reset] Done"
  exit 0
fi

echo "[init] Preparing user/database/privileges for raggify..."
# Create application user if missing
run_psql "DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${APP_USER}') THEN
    CREATE ROLE \"${APP_USER}\" LOGIN PASSWORD '${APP_PASSWORD}';
  END IF;
END \$\$;"

# Create database if missing
if ! sudo -u "$PG_SUPERUSER" psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  run_psql "CREATE DATABASE \"${DB_NAME}\" OWNER \"${APP_USER}\";"
fi

# Ensure pgvector extension
run_psql_db "$DB_NAME" "CREATE EXTENSION IF NOT EXISTS vector;"

# Grant schema privileges
run_psql_db "$DB_NAME" "GRANT CREATE, USAGE ON SCHEMA public TO \"${APP_USER}\";"

echo "[init] Done"
