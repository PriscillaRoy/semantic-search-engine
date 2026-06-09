# api/routers/features.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, field_validator
from functools import lru_cache
from feast import FeatureStore

from config import DEFAULT_TOP_K, MIN_TOP_K
from store.database import get_all_movies, get_movie_by_title, get_movie_count
from core.graph import combined_recommend_silent
from api.dependencies import AppResources, get_resources
from store.cache import make_cache_key, get_cached, set_cached

router = APIRouter(tags=["features"])


# ── Feast singleton ─────────────────────────────────────
@lru_cache(maxsize=1)
def get_feature_store() -> FeatureStore:
    """Singleton Feast feature store — loaded once per process."""
    return FeatureStore(repo_path="store/feature_repo")


# ── Models ─────────────────────────────────────────────
class PersonalizedRequest(BaseModel):
    title:   str
    user_id: str = "user_000"
    top_k:   int = DEFAULT_TOP_K

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_valid(cls, v):
        max_allowed = get_movie_count() - 1
        if v < MIN_TOP_K or v > max_allowed:
            raise ValueError(f"top_k must be between {MIN_TOP_K} and {max_allowed}")
        return v


# ── GET /features/{movie_id} ────────────────────────────
@router.get("/features/{movie_id}")
def get_movie_features(movie_id: int):
    """
    Retrieve precomputed features for a movie from
    the Feast online store (Redis).
    Sub-millisecond lookup — no recomputation needed.
    """
    cache_key = make_cache_key("features", {"movie_id": movie_id})
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    try:
        store    = get_feature_store()
        features = store.get_online_features(
            features=[
                "movie_features:popularity_score",
                "movie_features:avg_rating",
                "movie_features:avg_completion",
                "movie_features:genre_encoded",
                "movie_features:request_count",
                "movie_features:decade",
            ],
            entity_rows=[{"movie_id": movie_id}]
        ).to_dict()

        result = {
            "movie_id":         movie_id,
            "popularity_score": features["popularity_score"][0],
            "avg_rating":       features["avg_rating"][0],
            "avg_completion":   features["avg_completion"][0],
            "genre_encoded":    features["genre_encoded"][0],
            "request_count":    features["request_count"][0],
            "decade":           features["decade"][0],
            "cached":           False
        }
        set_cached(cache_key, result)
        return result

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Features not found for movie_id={movie_id}: {str(e)}"
        )


# ── POST /recommend/personalized ───────────────────────
@router.post("/recommend/personalized")
def recommend_personalized(
    request: PersonalizedRequest,
    resources: AppResources = Depends(get_resources)
):
    cache_key = make_cache_key("recommend_personalized", {
        "title":   request.title,
        "user_id": request.user_id,
        "top_k":   request.top_k
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    matches = get_movie_by_title(request.title)
    if not matches:
        raise HTTPException(status_code=404,
                            detail=f"'{request.title}' not found")

    ranked     = combined_recommend_silent(request.title,
                                           top_k=request.top_k * 2)
    all_movies = get_all_movies()
    movie_map  = {m["title"]: m for m in all_movies}
    id_map     = {m["title"]: m["id"] for m in all_movies}
    store      = get_feature_store()

    # fetch user features
    try:
        user_features = store.get_online_features(
            features=[
                "user_features:preferred_genre_enc",
                "user_features:activity_score",
                "user_features:avg_completion_pct",
            ],
            entity_rows=[{"user_id": request.user_id}]
        ).to_dict()
        user_genre_enc  = user_features["preferred_genre_enc"][0] or 0
        user_activity   = user_features["activity_score"][0] or 0.5
        user_completion = user_features["avg_completion_pct"][0] or 0.7
    except Exception:
        user_genre_enc  = 0
        user_activity   = 0.5
        user_completion = 0.7

    candidates = []
    for title, base_score in ranked:
        movie_id = id_map.get(title)
        if movie_id is None:
            continue

        try:
            features = store.get_online_features(
                features=[
                    "movie_features:popularity_score",
                    "movie_features:avg_rating",
                    "movie_features:genre_encoded",
                ],
                entity_rows=[{"movie_id": movie_id}]
            ).to_dict()
            popularity = features["popularity_score"][0] or 0.5
            avg_rating = features["avg_rating"][0] or 3.0
            genre_enc  = features["genre_encoded"][0] or 0
        except Exception:
            popularity = 0.5
            avg_rating = 3.0
            genre_enc  = 0

        genre_match = 0.1 if genre_enc == user_genre_enc else 0.0

        final_score = (
            base_score           * 0.5  +
            popularity           * 0.15 +
            (avg_rating / 5.0)   * 0.15 +
            genre_match          * 0.1  +
            user_completion      * 0.05 +
            user_activity        * 0.05
        )

        m = movie_map.get(title)
        if m:
            candidates.append({
                "title":       m["title"],
                "year":        m["year"],
                "genre":       m["genre"],
                "base_score":  round(base_score, 4),
                "popularity":  round(popularity, 4),
                "avg_rating":  round(avg_rating, 4),
                "genre_match": genre_match > 0,
                "final_score": round(final_score, 4),
            })

    candidates = sorted(candidates,
                        key=lambda x: x["final_score"],
                        reverse=True)[:request.top_k]

    response = {
        "query":   request.title,
        "user_id": request.user_id,
        "results": candidates,
        "cached":  False
    }
    set_cached(cache_key, response)
    return response