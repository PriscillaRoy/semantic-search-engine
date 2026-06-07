# database.py
import sqlite3
import pandas as pd
from pathlib import Path
from config import DB_PATH, DATA_PATH


# ── Create table + load data ───────────────────────────
def init_db():
    """
    Creates the movies table and loads data from CSV.
    Run once — replaces movies_meta.pkl.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS movies (
            id          INTEGER PRIMARY KEY,
            title       TEXT NOT NULL,
            year        INTEGER,
            genre       TEXT,
            description TEXT
        )
    """)

    # Load from CSV and insert
    df = pd.read_csv(DATA_PATH)
    for _, row in df.iterrows():
        cursor.execute("""
            INSERT OR IGNORE INTO movies 
            (id, title, year, genre, description)
            VALUES (?, ?, ?, ?, ?)
        """, (row["id"], row["title"], row["year"],
              row["genre"], row["description"]))

    conn.commit()
    conn.close()
    print(f"[DB] Initialized with {len(df)} movies → {DB_PATH}")

def get_movie_count() -> int:
    """Returns total number of movies in the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM movies")
    count = cursor.fetchone()[0]
    conn.close()
    return count

# ── Query functions ────────────────────────────────────
def get_all_movies():
    """Returns all movies as a list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM movies")
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_movie_by_title(title: str, year: int = None):
    """
    Case-insensitive title search.
    If year is provided, returns exact match.
    If multiple matches found, returns all of them
    so the caller can decide how to handle it.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if year:
        cursor.execute("""
            SELECT * FROM movies
            WHERE LOWER(title) = LOWER(?)
            AND year = ?
        """, (title, year))
    else:
        cursor.execute("""
            SELECT * FROM movies
            WHERE LOWER(title) = LOWER(?)
        """, (title,))

    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows   # always returns a list now


def get_movies_by_genre(genre: str):
    """Returns all movies of a given genre."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM movies 
        WHERE LOWER(genre) = LOWER(?)
    """, (genre,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_movies_by_ids(ids: list[int]):
    """
    Fetch multiple movies by their FAISS index positions.
    Used after FAISS search returns indices.
    """
    placeholders = ",".join("?" * len(ids))
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(f"""
        SELECT * FROM movies 
        WHERE id IN ({placeholders})
    """, ids)
    rows = {row["id"]: dict(row) for row in cursor.fetchall()}
    conn.close()
    # return in same order as ids
    return [rows[i] for i in ids if i in rows]


def search_by_description_text(query: str, limit: int = 5):
    """
    Basic text search on descriptions.
    This is a fallback — FAISS semantic search is better.
    But useful for exact keyword matches.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM movies
        WHERE LOWER(description) LIKE LOWER(?)
        LIMIT ?
    """, (f"%{query}%", limit))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def get_duplicate_titles():
    """
    Find movies with the same title.
    Used to handle the duplicate title edge case.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT title, COUNT(*) as count 
        FROM movies 
        GROUP BY LOWER(title) 
        HAVING count > 1
    """)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


# ── Main — run to initialize ───────────────────────────
if __name__ == "__main__":
    init_db()

    print("\n[Test] All movies:")
    movies = get_all_movies()
    for m in movies:
        print(f"  {m['id']:2}. {m['title']} ({m['year']}) [{m['genre']}]")

    print("\n[Test] Search by title:")
    m = get_movie_by_title("inception")
    print(f"  Found: {m}")

    print("\n[Test] Search by genre:")
    scifi = get_movies_by_genre("Sci-Fi")
    print(f"  Sci-Fi movies: {[m['title'] for m in scifi]}")

    print("\n[Test] Search by description keyword:")
    results = search_by_description_text("astronaut")
    print(f"  Matches: {[m['title'] for m in results]}")

    print("\n[Test] Duplicate titles:")
    dupes = get_duplicate_titles()
    print(f"  Duplicates: {dupes if dupes else 'None found'}")