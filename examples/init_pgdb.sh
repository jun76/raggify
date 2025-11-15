#!/usr/bin/env bash
set -euo pipefail

# =========================
# Configuration (env override possible)
# =========================
PG_SUPERUSER="${PG_SUPERUSER:-postgres}"  # superuser name
DB_NAME="${DB_NAME:-raggify}"             # Application database name
APP_USER="${APP_USER:-raggify}"           # Application DB user
APP_PASSWORD="${APP_PASSWORD:-raggify}"   # Password for application DB user
PGHOST="${PGHOST:-localhost}"             # PostgreSQL host (例: pgvector, localhost)
PGPORT="${PGPORT:-5432}"                  # PostgreSQL port

# psql password (matches docker's POSTGRES_PASSWORD)
PGPASSWORD="${PGPASSWORD:-raggify}"
export PGPASSWORD

# =========================
# Helper functions
# =========================
notice() {
  cat >&2 <<EOF
[error] Failed to connect to or execute on PostgreSQL.
This is often caused by the superuser (${PG_SUPERUSER}) password not matching the value of the PGPASSWORD environment variable.
If you are using a local PostgreSQL instance, set the password using the following command and then run this script again:

  sudo -u postgres psql -c "ALTER USER ${PG_SUPERUSER} WITH PASSWORD '${PGPASSWORD}';"

Translated with DeepL.com (free version)

EOF
}

run_psql() {
  # Execute a psql command as superuser without selecting a DB
  local sql="$1"
  if ! psql \
    -h "$PGHOST" \
    -p "$PGPORT" \
    -U "$PG_SUPERUSER" \
    -v ON_ERROR_STOP=1 \
    -c "$sql"
  then
    notice
    exit 1
  fi
}

run_psql_db() {
  # Execute a psql command as superuser against a specific DB
  local db="$1"
  local sql="$2"
  if ! psql \
    -h "$PGHOST" \
    -p "$PGPORT" \
    -U "$PG_SUPERUSER" \
    -d "$db" \
    -v ON_ERROR_STOP=1 \
    -c "$sql"
  then
    notice
    exit 1
  fi
}

run_psql_query() {
  # Run a query that returns a scalar value (used for existence checks)
  local sql="$1"
  if ! psql \
    -h "$PGHOST" \
    -p "$PGPORT" \
    -U "$PG_SUPERUSER" \
    -tAc "$sql"
  then
    notice
    exit 1
  fi
}

usage() {
  cat <<EOF
Usage:
  ./init_pgdb.sh          Run initial setup for PostgreSQL so raggify can use it
  ./init_pgdb.sh --reset  Drop and recreate the DB, then run setup again

Current configuration:
  PG_SUPERUSER  (${PG_SUPERUSER})
  DB_NAME       (${DB_NAME})
  APP_USER      (${APP_USER})
  APP_PASSWORD  (****)
  PGHOST        (${PGHOST})
  PGPORT        (${PGPORT})
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
  echo "[reset] Dropping and recreating existing DB (${DB_NAME}) on ${PGHOST}:${PGPORT}..."
  # Terminate active sessions before dropping
  run_psql "SELECT pg_terminate_backend(pid)
            FROM pg_stat_activity
            WHERE datname = '${DB_NAME}'
              AND pid <> pg_backend_pid();" || true

  run_psql "DROP DATABASE IF EXISTS \"${DB_NAME}\";"

  # Ensure app user exists
  run_psql "DO \$\$ BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${APP_USER}') THEN
      CREATE ROLE \"${APP_USER}\" LOGIN PASSWORD '${APP_PASSWORD}';
    END IF;
  END \$\$;"

  # Recreate DB owned by app user
  run_psql "CREATE DATABASE \"${DB_NAME}\" OWNER \"${APP_USER}\";"

  # Ensure pgvector extension and privileges
  run_psql_db "$DB_NAME" "CREATE EXTENSION IF NOT EXISTS vector;"
  run_psql_db "$DB_NAME" "GRANT CREATE, USAGE ON SCHEMA public TO \"${APP_USER}\";"

  echo "[reset] Done"
  exit 0
fi

echo "[init] Preparing user/database/privileges for raggify on ${PGHOST}:${PGPORT}..."

# Create application user if missing
run_psql "DO \$\$ BEGIN
  IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname='${APP_USER}') THEN
    CREATE ROLE \"${APP_USER}\" LOGIN PASSWORD '${APP_PASSWORD}';
  END IF;
END \$\$;"

# Create database if missing
if ! run_psql_query "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  run_psql "CREATE DATABASE \"${DB_NAME}\" OWNER \"${APP_USER}\";"
fi

# Ensure pgvector extension
run_psql_db "$DB_NAME" "CREATE EXTENSION IF NOT EXISTS vector;"

# Grant schema privileges
run_psql_db "$DB_NAME" "GRANT CREATE, USAGE ON SCHEMA public TO \"${APP_USER}\";"

echo "[init] Done"
