import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, field_validator

from config import (INDEX_PATH, META_PATH,
                    DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K)
from database import get_all_movies, get_movie_by_title, get_movie_count
from graph import combined_recommend_silent
from rag import retrieve, build_prompt, generate
from dependencies import AppResources, get_resources
from embeddings import search_by_description, prepare_text
from cache import make_cache_key, get_cached, set_cached, get_cache_stats

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
    # ── Cache check ────────────────────────────────────
    cache_key = make_cache_key("search", {
        "query": request.query, "top_k": request.top_k
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    # ── Normal flow ────────────────────────────────────
    results  = search_by_description(
        query_text=request.query,
        top_k=request.top_k,
        resources=resources
    )
    response = {"query": request.query,
                "results": results, "cached": False}
    set_cached(cache_key, response)
    return response

# ── Health check ────────────────────────────────────────
@app.get("/health")
def health(resources: AppResources = Depends(get_resources)):
    return {
        "status":     "ok",
        "model":      "llama3.2",
        "index_size": resources.index.ntotal
    }
# ── Cache stats ─────────────────────────────────────────
@app.get("/cache/stats")
def cache_stats():
    return get_cache_stats()

# ── Cache invalidation ──────────────────────────────────
@app.delete("/cache/clear")
def clear_cache():
    from cache import invalidate_cache
    count = invalidate_cache()
    return {"message": f"Cleared {count} cached keys"}