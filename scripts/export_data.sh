#!/usr/bin/env bash
set -euo pipefail

# Export Postgres data to a host-mounted backup directory using the postgres
# service from docker-compose.
#
# Usage (from repo root):
#   bash scripts/export_data.sh
#
# Environment overrides (optional):
#   BACKUP_DIR   - host directory for backups (default: ./backups)
#   DB_NAME      - database name (default: weather)
#   DB_USER      - database user (default: weather)
#   SERVICE_NAME - docker-compose service name for Postgres (default: postgres)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

BACKUP_DIR="${BACKUP_DIR:-$ROOT_DIR/backups}"
DB_NAME="${DB_NAME:-weather}"
DB_USER="${DB_USER:-weather}"
SERVICE_NAME="${SERVICE_NAME:-postgres}"

mkdir -p "$BACKUP_DIR"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DUMP="$BACKUP_DIR/postgres_${TS}.dump"

echo "[export_data] Writing logical backup to $OUT_DUMP"

# Use pg_dump inside the postgres service container.
docker compose exec -T "$SERVICE_NAME" pg_dump -U "$DB_USER" -d "$DB_NAME" -Fc > "$OUT_DUMP"

echo "[export_data] Done."

