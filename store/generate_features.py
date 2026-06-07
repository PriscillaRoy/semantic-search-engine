# store/generate_features.py
"""
Generates parquet files for Feast offline store.

Produces:
  movie_features.parquet  → item-level features
  user_features.parquet   → simulated user behavior

Run after embeddings.py — uses precomputed FAISS index.
In production this would be a nightly batch job.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
import numpy as np
import faiss
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from config import DB_PATH, EMB_PATH, INDEX_PATH
from store.database import get_all_movies

# ── Config ─────────────────────────────────────────────
OUTPUT_DIR   = Path("store/feature_repo/data")
N_FAKE_USERS = 50      # simulate 50 users
GENRE_ENCODING = {
    "Sci-Fi": 0, "Horror": 1, "Thriller": 2,
    "Crime": 3, "Drama": 4, "Action": 5,
    "Animation": 6, "Romance": 7, "Mystery": 8,
    "Comedy": 9
}

def generate_movie_features():
    """
    Compute item-level features for all movies.
    popularity_score and avg_completion are simulated
    but realistic — in production these come from
    actual user event logs.
    """
    movies     = get_all_movies()
    embeddings = np.load(str(EMB_PATH))
    index      = faiss.read_index(str(INDEX_PATH))
    now        = datetime.utcnow()

    rows = []
    for i, movie in enumerate(movies):
        # simulate popularity — older classics get higher scores
        # newer movies build popularity over time
        year       = int(movie["year"])
        age_factor = min(1.0, (2026 - year) / 30)
        popularity = round(random.uniform(0.3, 0.7) +
                           age_factor * 0.3, 3)

        # simulate avg completion — good movies get watched fully
        avg_completion = round(random.uniform(0.65, 0.98), 3)

        # simulate avg rating — correlated with completion
        avg_rating = round(avg_completion * 5 *
                           random.uniform(0.85, 1.0), 2)

        # simulate request count — popular genres get more requests
        genre_popularity = {
            "Sci-Fi": 150, "Horror": 120, "Thriller": 130,
            "Crime": 100, "Drama": 90, "Action": 160,
            "Animation": 110, "Romance": 80,
            "Mystery": 70, "Comedy": 95
        }
        base_requests  = genre_popularity.get(movie["genre"], 100)
        request_count  = int(base_requests *
                             random.uniform(0.5, 1.5) *
                             popularity)

        rows.append({
            "movie_id":        int(movie["id"]),
            "event_timestamp": datetime.now(timezone.utc) - timedelta(minutes=random.randint(0, 30)),
            "genre_encoded":   GENRE_ENCODING.get(
                                   movie["genre"], -1),
            "decade":          (year // 10) * 10,
            "description_len": len(movie["description"].split()),
            "title_len":       len(movie["title"].split()),
            "popularity_score": popularity,
            "avg_completion":  avg_completion,
            "avg_rating":      avg_rating,
            "request_count":   request_count,
        })

    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "movie_features.parquet"
    df.to_parquet(path, index=False)
    print(f"[Features] Movie features → {path} "
          f"({len(df)} rows)")
    return df


def generate_user_features():
    """
    Simulate user behavior features.
    Mimics what you'd compute from a real event stream:
      watch_events → aggregate → user feature vector

    In production:
      Kafka events → Spark/Flink job → parquet → Feast
    """
    movies = get_all_movies()
    genres = list(GENRE_ENCODING.keys())
    now    = datetime.utcnow()

    rows = []
    for i in range(N_FAKE_USERS):
        user_id = f"user_{i:03d}"

        # each user has a dominant genre preference
        preferred_genre     = random.choice(genres)
        preferred_genre_enc = GENRE_ENCODING[preferred_genre]

        # simulate watch history
        watch_count_7d  = random.randint(1, 12)
        watch_count_30d = watch_count_7d + random.randint(5, 30)

        # power users finish more movies
        is_power_user      = watch_count_7d > 8
        avg_completion_pct = round(
            random.uniform(0.75, 0.98) if is_power_user
            else random.uniform(0.45, 0.80), 3
        )

        # ratings given — correlated with completion
        avg_rating_given = round(
            avg_completion_pct * 5 * random.uniform(0.8, 1.0),
            2
        )

        # last watched genre — may differ from preferred
        last_watched_genre = random.choice(
            [preferred_genre] * 3 + genres  # biased toward preferred
        )

        # activity score — normalized engagement signal
        activity_score = round(
            (watch_count_7d / 12) * 0.5 +
            avg_completion_pct * 0.3 +
            (avg_rating_given / 5) * 0.2,
            3
        )

        rows.append({
            "user_id":             user_id,
            "event_timestamp":     now - timedelta(
                                       minutes=random.randint(0, 60)),
            "preferred_genre":     preferred_genre,
            "preferred_genre_enc": preferred_genre_enc,
            "watch_count_7d":      watch_count_7d,
            "watch_count_30d":     watch_count_30d,
            "avg_completion_pct":  avg_completion_pct,
            "avg_rating_given":    avg_rating_given,
            "last_watched_genre":  last_watched_genre,
            "activity_score":      activity_score,
        })

    df = pd.DataFrame(rows)
    path = OUTPUT_DIR / "user_features.parquet"
    df.to_parquet(path, index=False)
    print(f"[Features] User features → {path} "
          f"({len(df)} rows)")
    return df


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating offline feature store...")
    movie_df = generate_movie_features()
    user_df  = generate_user_features()

    print("\nSample movie features:")
    print(movie_df[["movie_id", "genre_encoded",
                     "popularity_score", "avg_rating",
                     "request_count"]].head(5).to_string(index=False))

    print("\nSample user features:")
    print(user_df[["user_id", "preferred_genre",
                    "watch_count_7d",
                    "avg_completion_pct",
                    "activity_score"]].head(5).to_string(index=False))

    print("\nApplying Feast materialize...")
    os.system(
        "cd store/feature_repo && "
        "feast materialize-incremental "
        f"$(date -u +%Y-%m-%dT%H:%M:%S)"
    )
    print("\nFeature store ready!")