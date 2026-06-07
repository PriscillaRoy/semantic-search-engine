import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, field_validator

from config import (INDEX_PATH, META_PATH,
                    DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K, SEARCH_BACKEND)
from store.database import get_all_movies, get_movie_by_title, get_movie_count
from core.graph import combined_recommend_silent
from core.rag import retrieve, build_prompt, generate
from api.dependencies import AppResources, get_resources
from core.embeddings import search_by_description, prepare_text
from store.cache import make_cache_key, get_cached, set_cached, get_cache_stats
from core.milvus_store import milvus_search, upsert_movie
from feast import FeatureStore
from functools import lru_cache

# ── App ────────────────────────────────────────────────
app = FastAPI(
    title="Content Discovery Engine",
    description="Movie recommendations using FAISS + Graph + RAG",
    version="1.0.0"
)

# ── Request models ─────────────────────────────────────

class RecommendRequest(BaseModel):
    title: str
    top_k: int = DEFAULT_TOP_K

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_valid(cls, v):
        max_allowed = get_movie_count() - 1
        if v < MIN_TOP_K or v > max_allowed:
            raise ValueError(f"top_k must be between {MIN_TOP_K} and {max_allowed}")
        return v


class ExplainRequest(BaseModel):
    title: str
    top_k: int = DEFAULT_TOP_K

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_valid(cls, v):
        max_allowed = get_movie_count() - 1
        if v < MIN_TOP_K or v > max_allowed:
            raise ValueError(f"top_k must be between {MIN_TOP_K} and {max_allowed}")
        return v

class SearchRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_valid(cls, v):
        max_allowed = get_movie_count() - 1
        if v < MIN_TOP_K or v > max_allowed:
            raise ValueError(f"top_k must be between {MIN_TOP_K} and {max_allowed}")
        return v

# ── Endpoint 1: GET /similar/{title} ───────────────────
@app.get("/similar/{title}")
def get_similar(
    title: str,
    top_k: int = DEFAULT_TOP_K,
    year: int = None,
    return_all: bool = False,
    resources: AppResources = Depends(get_resources)
):
    # ── Cache check ────────────────────────────────────
    cache_key = make_cache_key("similar", {
        "title": title, "top_k": top_k,
        "year": year, "return_all": return_all
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    # ── Normal flow ────────────────────────────────────
    matches = get_movie_by_title(title, year=year)
    if not matches:
        raise HTTPException(status_code=404,
                            detail=f"'{title}' not found")

    if len(matches) > 1 and year is None and not return_all:
        return {
            "error":   "multiple_matches",
            "message": f"Multiple matches for '{title}'. "
                       f"Add more details or use return_all=true.",
            "matches": [{"title": m["title"], "year": m["year"],
                         "genre": m["genre"]} for m in matches]
        }

    q    = matches[0]
    qvec = resources.encode(q["description"])
    distances, indices = resources.search(qvec, top_k + 1)

    all_movies = get_all_movies()
    results    = []
    for dist, idx in zip(distances[0], indices[0]):
        m = all_movies[idx]
        if m["title"].lower() == title.lower():
            continue
        results.append({
            "title":      m["title"],
            "year":       m["year"],
            "genre":      m["genre"],
            "similarity": round(float(dist), 4)
        })
        if len(results) == top_k:
            break

    response = {"query": f"{title} ({q['year']})",
                "results": results, "cached": False}
    set_cached(cache_key, response)
    return response



# ── Endpoint 2: POST /recommend ─────────────────────────
@app.post("/recommend")
def recommend(
    request: RecommendRequest,
    resources: AppResources = Depends(get_resources)
):
    # ── Cache check ────────────────────────────────────
    cache_key = make_cache_key("recommend", {
        "title": request.title, "top_k": request.top_k
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    # ── Normal flow ────────────────────────────────────
    matches = get_movie_by_title(request.title)
    if not matches:
        raise HTTPException(status_code=404,
                            detail=f"'{request.title}' not found")

    ranked     = combined_recommend_silent(request.title,
                                           top_k=request.top_k)
    all_movies = get_all_movies()
    movie_map  = {m["title"]: m for m in all_movies}

    results = []
    for title, score in ranked:
        m = movie_map.get(title)
        if m:
            results.append({
                "title": m["title"],
                "year":  m["year"],
                "genre": m["genre"],
                "score": round(score, 4)
            })

    response = {"query": request.title,
                "results": results, "cached": False}
    set_cached(cache_key, response)
    return response

# ── Endpoint 3: POST /explain ───────────────────────────
@app.post("/explain")
def explain(
    request: ExplainRequest,
    resources: AppResources = Depends(get_resources)
):
    query_movie, retrieved = retrieve(request.title,
                                      top_k=request.top_k)
    if not retrieved:
        raise HTTPException(status_code=404,
                            detail=f"'{request.title}' not found")

    prompt      = build_prompt(query_movie, retrieved)
    explanation = generate(prompt)

    return {
        "query":           request.title,
        "recommendations": [m["title"] for m in retrieved],
        "explanation":     explanation
    }


# ── Endpoint 4: GET /evaluate ───────────────────────────
@app.get("/evaluate")
def evaluate(
    top_k: int = DEFAULT_TOP_K,
    resources: AppResources = Depends(get_resources)
):
    all_movies = get_all_movies()
    scores = []

    for query in all_movies:
        query_title = query["title"]
        query_genre = query["genre"]

        qvec = resources.encode(prepare_text(query))
        distances, indices = resources.search(qvec, top_k + 1)

        relevant = 0
        shown    = 0
        for dist, idx in zip(distances[0], indices[0]):
            m = all_movies[idx]
            if m["title"].lower() == query_title.lower():
                continue
            if m["genre"] == query_genre:
                relevant += 1
            shown += 1
            if shown == top_k:
                break

        scores.append({
            "title":     query_title,
            "genre":     query_genre,
            "precision": round(relevant / top_k, 4)
        })

    avg = round(sum(s["precision"] for s in scores) / len(scores), 4)

    return {
        "metric":        "precision@k",
        "k":             top_k,
        "avg_precision": avg,
        "per_movie":     sorted(scores,
                                key=lambda x: x["precision"],
                                reverse=True)
    }

# ── Endpoint 5: POST /search ────────────────────────────
@app.post("/search")
def search(
    request: SearchRequest,
    resources: AppResources = Depends(get_resources)
):
    """
    Search by description text.
    Backend controlled by SEARCH_BACKEND in config.py:
      "faiss"  → local FAISS index
      "milvus" → Milvus vector database
    """
    cache_key = make_cache_key("search", {
        "query":   request.query,
        "top_k":   request.top_k,
        "backend": SEARCH_BACKEND    # ← cache is backend-aware
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    if SEARCH_BACKEND == "milvus":
        results = milvus_search(request.query, top_k=request.top_k)
    else:
        results = search_by_description(
            query_text=request.query,
            top_k=request.top_k,
            resources=resources
        )

    response = {
        "query":   request.query,
        "backend": SEARCH_BACKEND,
        "results": results,
        "cached":  False
    }
    set_cached(cache_key, response)
    return response

class UpsertRequest(BaseModel):
    id:          int
    title:       str
    year:        int
    genre:       str
    description: str

# ── Endpoint 6: POST /milvus/upsert ────────────────────
@app.post("/milvus/upsert")
def upsert(request: UpsertRequest):
    """
    Add or update a movie in Milvus in real-time.
    No index rebuild needed — available for search instantly.
    This is the key advantage over FAISS.
    """
    upsert_movie(request.dict())

    # invalidate cache so stale results aren't served
    from store.cache import invalidate_cache
    invalidate_cache("cde:search_milvus:*")

    return {
        "status":  "upserted",
        "title":   request.title,
        "message": "Available for search immediately"
    }


# ── Endpoint 7: GET /features/{movie_id} ─────────────────
@lru_cache(maxsize=1)
def get_feature_store() -> FeatureStore:
    """Singleton Feast feature store — loaded once."""
    return FeatureStore(repo_path="store/feature_repo")

@app.get("/features/{movie_id}")
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
            "movie_id":        movie_id,
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
        raise HTTPException(status_code=404,
                            detail=f"Features not found for movie_id={movie_id}: {str(e)}")


# ── Endpoint 8: POST /recommend/personalized ──────────────
class PersonalizedRequest(BaseModel):
    title:   str
    user_id: str = "user_000"   # default user
    top_k:   int = DEFAULT_TOP_K

    @field_validator("top_k")
    @classmethod
    def top_k_must_be_valid(cls, v):
        max_allowed = get_movie_count() - 1
        if v < MIN_TOP_K or v > max_allowed:
            raise ValueError(f"top_k must be between {MIN_TOP_K} and {max_allowed}")
        return v
    
@app.post("/recommend/personalized")
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
        user_genre_enc   = user_features["preferred_genre_enc"][0] or 0
        user_activity    = user_features["activity_score"][0] or 0.5
        user_completion  = user_features["avg_completion_pct"][0] or 0.7
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
            popularity   = features["popularity_score"][0] or 0.5
            avg_rating   = features["avg_rating"][0] or 3.0
            genre_enc    = features["genre_encoded"][0] or 0
        except Exception:
            popularity   = 0.5
            avg_rating   = 3.0
            genre_enc    = 0

        # genre match boost — reward movies matching user preference
        genre_match = 0.1 if genre_enc == user_genre_enc else 0.0

        # final score combines all signals
        final_score = (
            base_score             * 0.5 +
            popularity             * 0.15 +
            (avg_rating / 5.0)     * 0.15 +
            genre_match            * 0.1  +
            user_completion        * 0.05 +
            user_activity          * 0.05
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

# ── Health check ────────────────────────────────────────
@app.get("/health")
def health(resources: AppResources = Depends(get_resources)):
    return {
        "status":     "ok",
        "model":      "llama3.2",
        "index_size": resources.index.ntotal,
        "search_backend": SEARCH_BACKEND
    }
# ── Cache stats ─────────────────────────────────────────
@app.get("/cache/stats")
def cache_stats():
    return get_cache_stats()

# ── Cache invalidation ──────────────────────────────────
@app.delete("/cache/clear")
def clear_cache():
    from store.cache import invalidate_cache
    count = invalidate_cache()
    return {"message": f"Cleared {count} cached keys"}