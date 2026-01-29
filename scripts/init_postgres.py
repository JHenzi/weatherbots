import os
from pathlib import Path

import psycopg2


def _get_dsn() -> str:
    url = os.getenv("PGDATABASE_URL") or os.getenv("DATABASE_URL")
    if url:
        return str(url).strip()

    # Try to determine if we are inside docker or on host.
    # 'postgres' is the service name in docker-compose.
    import socket

    host = "postgres"
    port = 5432
    try:
        socket.gethostbyname(host)
    except socket.gaierror:
        # Fallback for host-based execution
        host = "localhost"
        port = 5433  # Mapped port in docker-compose.yml

    return f"postgresql://weather:weather@{host}:{port}/weather"


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    schema_path = root / "db" / "schema.sql"
    if not schema_path.exists():
        raise SystemExit(f"Missing schema file: {schema_path}")

    dsn = _get_dsn()
    print(f"[init_postgres] Applying schema from {schema_path} to {dsn!r}")
    sql = schema_path.read_text()

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()
    finally:
        conn.close()
    print("[init_postgres] Schema applied successfully.")


if __name__ == "__main__":
    main()

