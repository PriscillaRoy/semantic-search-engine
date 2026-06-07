# feature_store.py
"""
Feature store for the Semantic Search Engine.

Two-tier architecture:
  Offline store → SQLite (all features, batch updated)
  Online store  → Redis  (hot features, real-time served)

Mirrors production patterns from Uber/Airbnb/Netflix.
Features are computed once and reused everywhere —
eliminates training-serving skew.
"""
import json
import sqlite3
import numpy as np
import faiss
from functools import lru_cache
from pathlib import Path
from config import (DB_PATH, EMB_PATH, REDIS_HOST,
                    REDIS_PORT, REDIS_TTL)
from store.database import get_all_movies


# ── Feature definitions ────────────────────────────────
# What features we compute per movie
# Adding a new feature = add it here + in compute_features()
FEATURE_SCHEMA = {
    "embedding":       "vector",    # 384-dim float array
    "genre_encoded":   "int",       # label encoded genre
    "decade":          "int",       # release decade
    "description_len": "int",       # word count
    "title_len":       "int",       # title word count
    "similar_ids":     "list",      # precomputed top-5 FAISS neighbors
}

# Genre label encoding — consistent across runs
GENRE_ENCODING = {
    "Sci-Fi":    0,
    "Horror":    1,
    "Thriller":  2,
    "Crime":     3,
    "Drama":     4,
    "Action":    5,
    "Animation": 6,
    "Romance":   7,
    "Mystery":   8,
    "Comedy":    9,
}


# ── Feature computation ────────────────────────────────
def compute_features(movie: dict, embedding: np.ndarray,
                     index: faiss.Index,
                     all_movies: list) -> dict:
    """
    Compute all features for a single movie.
    This is the single source of truth for feature engineering.
    Same logic used for training AND serving — no skew.
    """
    # normalize embedding for cosine similarity
    vec = embedding.copy().reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)

    # precompute top-5 similar movie ids
    distances, indices = index.search(vec, 6)
    similar_ids = [
        int(all_movies[idx]["id"])
        for dist, idx in zip(distances[0], indices[0])
        if all_movies[idx]["id"] != movie["id"]
    ][:5]

    return {
        "movie_id":        int(movie["id"]),
        "embedding":       vec[0].tolist(),
        "genre_encoded":   GENRE_ENCODING.get(movie["genre"], -1),
        "decade":          (int(movie["year"]) // 10) * 10,
        "description_len": len(movie["description"].split()),
        "title_len":       len(movie["title"].split()),
        "similar_ids":     similar_ids,
    }


# ── Offline store (SQLite) ─────────────────────────────
def init_feature_store():
    """
    Creates the features table in SQLite.
    Separate from movies table — features can be
    recomputed without touching raw data.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS features (
            movie_id        INTEGER PRIMARY KEY,
            embedding       TEXT NOT NULL,
            genre_encoded   INTEGER,
            decade          INTEGER,
            description_len INTEGER,
            title_len       INTEGER,
            similar_ids     TEXT,
            computed_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()
    print("[FeatureStore] Offline store initialized")


def build_offline_store():
    """
    Compute and store features for all movies.
    Run this after embeddings.py — uses the same FAISS index.
    """
    init_feature_store()

    # load precomputed embeddings and index
    embeddings = np.load(str(EMB_PATH))
    from config import INDEX_PATH
    index      = faiss.read_index(str(INDEX_PATH))
    all_movies = get_all_movies()

    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for i, movie in enumerate(all_movies):
        features = compute_features(
            movie, embeddings[i], index, all_movies
        )
        cursor.execute("""
            INSERT OR REPLACE INTO features
            (movie_id, embedding, genre_encoded, decade,
             description_len, title_len, similar_ids)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            features["movie_id"],
            json.dumps(features["embedding"]),
            features["genre_encoded"],
            features["decade"],
            features["description_len"],
            features["title_len"],
            json.dumps(features["similar_ids"]),
        ))

    conn.commit()
    conn.close()
    print(f"[FeatureStore] Offline store built — "
          f"{len(all_movies)} movies")


def get_offline_features(movie_id: int) -> dict:
    """
    Retrieve features from SQLite offline store.
    Returns None if movie not found.
    """
    conn   = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT * FROM features WHERE movie_id = ?
    """, (movie_id,))
    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "movie_id":        row["movie_id"],
        "embedding":       json.loads(row["embedding"]),
        "genre_encoded":   row["genre_encoded"],
        "decade":          row["decade"],
        "description_len": row["description_len"],
        "title_len":       row["title_len"],
        "similar_ids":     json.loads(row["similar_ids"]),
        "computed_at":     row["computed_at"],
    }


# ── Online store (Redis) ───────────────────────────────
@lru_cache(maxsize=1)
def get_redis():
    """Singleton Redis client for feature store."""
    import redis
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2
        )
        client.ping()
        return client
    except Exception:
        return None


def _online_key(movie_id: int) -> str:
    """Redis key for a movie's features."""
    return f"features:{movie_id}"


def get_online_features(movie_id: int) -> dict:
    """
    Retrieve features from Redis online store.
    Falls back to offline store on miss.
    Populates online store on fallback — warm-up pattern.
    """
    client = get_redis()

    # try online store first
    if client:
        cached = client.get(_online_key(movie_id))
        if cached:
            client.expire(_online_key(movie_id), REDIS_TTL)
            return json.loads(cached)

    # fall back to offline store
    features = get_offline_features(movie_id)
    if not features:
        return None

    # populate online store for next time
    if client:
        client.setex(
            _online_key(movie_id),
            REDIS_TTL,
            json.dumps(features)
        )

    return features


def set_online_features(movie_id: int, features: dict):
    """Push features to online store explicitly."""
    client = get_redis()
    if client:
        client.setex(
            _online_key(movie_id),
            REDIS_TTL,
            json.dumps(features)
        )


def warm_online_store(top_n: int = 20):
    """
    Pre-load top N movies into Redis online store.
    Call at server startup to pre-warm the cache.
    Mimics Netflix pre-warming popular content features.
    """
    all_movies = get_all_movies()
    warmed     = 0

    for movie in all_movies[:top_n]:
        features = get_offline_features(movie["id"])
        if features:
            set_online_features(movie["id"], features)
            warmed += 1

    print(f"[FeatureStore] Online store warmed — "
          f"{warmed} movies pre-loaded")


# ── Feature-based similarity ───────────────────────────
def get_precomputed_similar(movie_id: int,
                             top_k: int = 4) -> list:
    """
    Return precomputed similar movie IDs from feature store.
    Zero latency — no FAISS search needed at query time.
    This is the key serving optimization.
    """
    features = get_online_features(movie_id)
    if not features:
        return []
    return features["similar_ids"][:top_k]


# ── Stats ──────────────────────────────────────────────
def feature_store_stats() -> dict:
    """Returns stats about both stores."""
    # offline stats
    conn   = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM features")
    offline_count = cursor.fetchone()[0]
    cursor.execute("""
        SELECT MIN(computed_at), MAX(computed_at)
        FROM features
    """)
    row = cursor.fetchone()
    conn.close()

    # online stats
    client = get_redis()
    online_count = 0
    if client:
        online_count = len(client.keys("features:*"))

    return {
        "offline_store": {
            "count":        offline_count,
            "oldest":       row[0],
            "newest":       row[1],
        },
        "online_store": {
            "count":        online_count,
            "backend":      "redis",
        }
    }


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Building feature store...")
    build_offline_store()

    print("\nWarming online store...")
    warm_online_store(top_n=20)

    print("\nStats:")
    print(json.dumps(feature_store_stats(), indent=2))

    print("\nTest — get features for Inception (id=0):")
    f = get_online_features(0)
    print(f"  genre_encoded:   {f['genre_encoded']}")
    print(f"  decade:          {f['decade']}")
    print(f"  description_len: {f['description_len']}")
    print(f"  similar_ids:     {f['similar_ids']}")
    print(f"  embedding[:5]:   {f['embedding'][:5]}")

    print("\nTest — precomputed similar movies for Inception:")
    ids = get_precomputed_similar(0, top_k=4)
    all_movies = get_all_movies()
    movie_map  = {m["id"]: m for m in all_movies}
    for mid in ids:
        m = movie_map.get(mid)
        if m:
            print(f"  {m['title']} ({m['year']}) [{m['genre']}]")