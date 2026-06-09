# store/migrate.py
"""
One-time migration script.
Creates tables in Supabase PostgreSQL and seeds movies from CSV.

Run once:
    python3 -m store.migrate

Safe to re-run — uses IF NOT EXISTS and INSERT ON CONFLICT DO NOTHING.
"""
import pandas as pd
from store.supabase_client import get_pg_connection, get_pg_cursor, ping_supabase
from config import DATA_PATH


# ── Table definitions ──────────────────────────────────
CREATE_MOVIES_TABLE = """
CREATE TABLE IF NOT EXISTS movies (
    id          INTEGER PRIMARY KEY,
    title       TEXT NOT NULL,
    year        INTEGER,
    genre       TEXT,
    description TEXT
);
"""

CREATE_USERS_TABLE = """
CREATE TABLE IF NOT EXISTS users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

CREATE_RATINGS_TABLE = """
CREATE TABLE IF NOT EXISTS ratings (
    id          SERIAL PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id    INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    rating      FLOAT CHECK (rating >= 1.0 AND rating <= 5.0),
    watched_pct INTEGER CHECK (watched_pct >= 0 AND watched_pct <= 100),
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(user_id, movie_id)   -- one rating per user per movie
);
"""

CREATE_WATCH_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS watch_history (
    id          SERIAL PRIMARY KEY,
    user_id     UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id    INTEGER NOT NULL REFERENCES movies(id) ON DELETE CASCADE,
    watched_pct INTEGER CHECK (watched_pct >= 0 AND watched_pct <= 100),
    watched_at  TIMESTAMPTZ DEFAULT NOW()
);
"""

# ── Indexes for query performance ──────────────────────
CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_movies_genre ON movies(genre);",
    "CREATE INDEX IF NOT EXISTS idx_movies_year ON movies(year);",
    "CREATE INDEX IF NOT EXISTS idx_ratings_user ON ratings(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_ratings_movie ON ratings(movie_id);",
    "CREATE INDEX IF NOT EXISTS idx_watch_history_user ON watch_history(user_id);",
    "CREATE INDEX IF NOT EXISTS idx_watch_history_movie ON watch_history(movie_id);",
]


def create_tables(cursor, conn):
    """Creates all tables if they don't exist."""
    print("[migrate] Creating tables...")

    for sql, name in [
        (CREATE_MOVIES_TABLE,       "movies"),
        (CREATE_USERS_TABLE,        "users"),
        (CREATE_RATINGS_TABLE,      "ratings"),
        (CREATE_WATCH_HISTORY_TABLE,"watch_history"),
    ]:
        cursor.execute(sql)
        print(f"  ✓ {name}")

    print("[migrate] Creating indexes...")
    for sql in CREATE_INDEXES:
        cursor.execute(sql)

    conn.commit()
    print("[migrate] ✓ All tables and indexes created")


def seed_movies(cursor, conn):
    """Seeds movies from CSV into Supabase. Skips duplicates."""
    print(f"[migrate] Seeding movies from {DATA_PATH}...")

    df = pd.read_csv(DATA_PATH)
    inserted = 0
    skipped  = 0

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO movies (id, title, year, genre, description)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
        """, (
            int(row["id"]),
            str(row["title"]),
            int(row["year"]) if pd.notna(row["year"]) else None,
            str(row["genre"]) if pd.notna(row["genre"]) else None,
            str(row["description"]) if pd.notna(row["description"]) else None,
        ))
        if cursor.rowcount == 1:
            inserted += 1
        else:
            skipped += 1

    conn.commit()
    print(f"[migrate] ✓ Movies seeded — {inserted} inserted, {skipped} skipped")


def verify(cursor):
    """Quick sanity check after migration."""
    print("[migrate] Verifying...")

    cursor.execute("SELECT COUNT(*) FROM movies")
    movie_count = cursor.fetchone()["count"]

    cursor.execute("""
        SELECT table_name FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name
    """)
    tables = [row["table_name"] for row in cursor.fetchall()]

    print(f"  ✓ Movies in DB : {movie_count}")
    print(f"  ✓ Tables found : {tables}")


def run_migration():
    """Full migration — create tables, seed data, verify."""
    print("\n=== Supabase Migration ===\n")

    # Check connectivity first
    if not ping_supabase():
        print("[migrate] ✗ Cannot reach Supabase — check your .env file")
        return

    conn   = get_pg_connection()
    cursor = get_pg_cursor(conn)

    try:
        create_tables(cursor, conn)
        seed_movies(cursor, conn)
        verify(cursor)
        print("\n[migrate] ✓ Migration complete\n")
    except Exception as e:
        conn.rollback()
        print(f"\n[migrate] ✗ Migration failed: {e}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    run_migration()