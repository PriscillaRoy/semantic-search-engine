# store/feature_repo/features.py
"""
Feast feature definitions for Semantic Search Engine.

Two entity types:
  movie  → item-level features (content signals)
  user   → user-level features (behavior signals, simulated)

Two feature views:
  movie_features → genre, decade, popularity, embeddings
  user_features  → watch history, preferences, activity
"""
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, ValueType
from feast.types import Float32, Int64, String

# ── Entities ───────────────────────────────────────────
movie = Entity(
    name="movie_id",
    value_type=ValueType.INT64,
    description="Unique movie identifier"
)

user = Entity(
    name="user_id",
    value_type=ValueType.STRING,
    description="Unique user identifier"
)

# ── Data sources ───────────────────────────────────────
movie_source = FileSource(
    path="data/movie_features.parquet",
    timestamp_field="event_timestamp",
    description="Movie item features — updated nightly"
)

user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
    description="User behavior features — updated hourly"
)

# ── Movie feature view ─────────────────────────────────
movie_features = FeatureView(
    name="movie_features",
    entities=[movie],
    ttl=timedelta(days=7),
    source=movie_source,
    schema=[
        Field(name="genre_encoded",    dtype=Int64),
        Field(name="decade",           dtype=Int64),
        Field(name="description_len",  dtype=Int64),
        Field(name="title_len",        dtype=Int64),
        Field(name="popularity_score", dtype=Float32),
        Field(name="avg_completion",   dtype=Float32),
        Field(name="avg_rating",       dtype=Float32),
        Field(name="request_count",    dtype=Int64),
    ],
)

# ── User feature view ──────────────────────────────────
user_features = FeatureView(
    name="user_features",
    entities=[user],
    ttl=timedelta(hours=1),
    source=user_source,
    schema=[
        Field(name="preferred_genre",      dtype=String),
        Field(name="preferred_genre_enc",  dtype=Int64),
        Field(name="watch_count_7d",       dtype=Int64),
        Field(name="watch_count_30d",      dtype=Int64),
        Field(name="avg_completion_pct",   dtype=Float32),
        Field(name="avg_rating_given",     dtype=Float32),
        Field(name="last_watched_genre",   dtype=String),
        Field(name="activity_score",       dtype=Float32),
    ],
)