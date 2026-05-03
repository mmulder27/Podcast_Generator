import json
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor


DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql:///kb_local")
OUTPUT_DIR = Path("exported_jsonl")
BATCH_SIZE = 10_000


def get_tables(conn):
    query = """
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_type = 'BASE TABLE'
          AND table_schema NOT IN ('pg_catalog', 'information_schema')
        ORDER BY table_schema, table_name;
    """
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()


def export_table(conn, schema, table):
    safe_name = f"{schema}.{table}".replace("/", "_")
    output_path = OUTPUT_DIR / f"{safe_name}.jsonl"

    query = f'SELECT * FROM "{schema}"."{table}";'

    with conn.cursor(name=f"cursor_{schema}_{table}", cursor_factory=RealDictCursor) as cur:
        cur.itersize = BATCH_SIZE
        cur.execute(query)

        with output_path.open("w", encoding="utf-8") as f:
            for row in cur:
                f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")

    print(f"Exported {schema}.{table} -> {output_path}")


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    conn = psycopg2.connect(DATABASE_URL)

    try:
        tables = get_tables(conn)
        for schema, table in tables:
            export_table(conn, schema, table)
    finally:
        conn.close()


if __name__ == "__main__":
    main()