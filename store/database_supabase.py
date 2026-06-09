# store/database_supabase.py
"""
PostgreSQL backend for Supabase.
Clean implementation — no SQLite code, no if/else.
All functions return the same shapes as database_sqlite.py
so the rest of the codebase works unchanged.
"""
from store.supabase_client import get_pg_connection, get_pg_cursor


# ── Internal helpers ───────────────────────────────────
def _conn_cursor():
    """Returns (conn, cursor) pair. Caller must close conn."""
    conn   = get_pg_connection()
    cursor = get_pg_cursor(conn)
    return conn, cursor


# ── Core movie queries ─────────────────────────────────
def get_movie_count() -> int:
    """Returns total number of movies in the database."""
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM movies")
        return cursor.fetchone()["count"]
    finally:
        conn.close()


def get_all_movies() -> list[dict]:
    """Returns all movies as a list of dicts."""
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("SELECT * FROM movies ORDER BY id")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_movie_by_title(title: str, year: int = None) -> list[dict]:
    """
    Case-insensitive title search.
    If year is provided, returns exact match.
    Always returns a list — caller handles multiple matches.
    """
    conn, cursor = _conn_cursor()
    try:
        if year:
            cursor.execute("""
                SELECT * FROM movies
                WHERE LOWER(title) = LOWER(%s)
                AND year = %s
            """, (title, year))
        else:
            cursor.execute("""
                SELECT * FROM movies
                WHERE LOWER(title) = LOWER(%s)
            """, (title,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_movies_by_genre(genre: str) -> list[dict]:
    """Returns all movies of a given genre."""
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            SELECT * FROM movies
            WHERE LOWER(genre) = LOWER(%s)
            ORDER BY id
        """, (genre,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_movies_by_ids(ids: list[int]) -> list[dict]:
    """
    Fetch multiple movies by their FAISS index positions.
    Uses PostgreSQL ANY() — cleaner than building IN (?,?,?).
    Returns in same order as ids (critical for FAISS result ordering).
    """
    if not ids:
        return []

    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            SELECT * FROM movies
            WHERE id = ANY(%s)
        """, (ids,))
        rows = {row["id"]: dict(row) for row in cursor.fetchall()}
        return [rows[i] for i in ids if i in rows]
    finally:
        conn.close()


def search_by_description_text(query: str, limit: int = 5) -> list[dict]:
    """
    Basic text search on descriptions.
    Fallback — FAISS semantic search is preferred.
    """
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            SELECT * FROM movies
            WHERE LOWER(description) LIKE LOWER(%s)
            LIMIT %s
        """, (f"%{query}%", limit))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_duplicate_titles() -> list[dict]:
    """Find movies with the same title — handles duplicate title edge case."""
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            SELECT title, COUNT(*) as count
            FROM movies
            GROUP BY title
            HAVING COUNT(*) > 1
        """)
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


# ── Ratings (Phase 2 — new) ────────────────────────────
def upsert_rating(user_id: str, movie_id: int,
                  rating: float, watched_pct: int) -> dict:
    """
    Insert or update a user's rating for a movie.
    ON CONFLICT → updates existing rating in place.
    """
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            INSERT INTO ratings (user_id, movie_id, rating, watched_pct)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id, movie_id)
            DO UPDATE SET
                rating      = EXCLUDED.rating,
                watched_pct = EXCLUDED.watched_pct,
                created_at  = NOW()
            RETURNING *
        """, (user_id, movie_id, rating, watched_pct))
        conn.commit()
        return dict(cursor.fetchone())
    finally:
        conn.close()


def get_user_ratings(user_id: str) -> list[dict]:
    """Returns all ratings for a user, joined with movie title and genre."""
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            SELECT r.*, m.title, m.genre
            FROM ratings r
            JOIN movies m ON r.movie_id = m.id
            WHERE r.user_id = %s
            ORDER BY r.created_at DESC
        """, (user_id,))
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def log_watch_event(user_id: str, movie_id: int, watched_pct: int) -> dict:
    """
    Records a watch event in watch_history.
    Feeds into Feast feature pipeline:
        watch_count_7d, watch_count_30d, avg_completion_pct
    Every play is a new row — full history preserved.
    """
    conn, cursor = _conn_cursor()
    try:
        cursor.execute("""
            INSERT INTO watch_history (user_id, movie_id, watched_pct)
            VALUES (%s, %s, %s)
            RETURNING *
        """, (user_id, movie_id, watched_pct))
        conn.commit()
        return dict(cursor.fetchone())
    finally:
        conn.close()

def init_db(): pass

# ── Main — smoke test ──────────────────────────────────
if __name__ == "__main__":
    print(f"[DB] Backend: supabase")
    print(f"[DB] Movie count: {get_movie_count()}")

    print("\n[Test] Search by title:")
    m = get_movie_by_title("inception")
    print(f"  Found: {m}")

    print("\n[Test] Search by genre:")
    scifi = get_movies_by_genre("Sci-Fi")
    print(f"  Sci-Fi movies: {[m['title'] for m in scifi]}")

    print("\n[Test] Search by description keyword:")
    results = search_by_description_text("astronaut")
    print(f"  Matches: {[m['title'] for m in results]}")

    print("\n[Test] Fetch by ids:")
    by_ids = get_movies_by_ids([1, 2, 3])
    print(f"  Movies: {[m['title'] for m in by_ids]}")

    print("\n[Test] Duplicate titles:")
    dupes = get_duplicate_titles()
    print(f"  Duplicates: {dupes if dupes else 'None found'}")