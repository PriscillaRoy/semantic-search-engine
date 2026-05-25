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
    resources: AppResources = Depends(get_resources)
):
    matches = get_movie_by_title(title, year=year)
    if not matches:
        raise HTTPException(status_code=404,
                            detail=f"'{title}' not found")

    # Handle duplicates
    if len(matches) > 1 and year is None:
        return {
            "error": "multiple_matches",
            "message": f"Multiple movies found for '{title}'. Specify year.",
            "matches": [{"title": m["title"], "year": m["year"],
                         "genre": m["genre"]} for m in matches]
        }

    q = matches[0]
    qvec = resources.encode(q["description"])
    distances, indices = resources.search(qvec, top_k + 1)

    all_movies = get_all_movies()
    results = []
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

    return {"query": f"{title} ({q['year']})", "results": results}


# ── Endpoint 2: POST /recommend ─────────────────────────
@app.post("/recommend")
def recommend(
    request: RecommendRequest,
    resources: AppResources = Depends(get_resources)
):
    q = get_movie_by_title(request.title)
    if not q:
        raise HTTPException(status_code=404,
                            detail=f"'{request.title}' not found")

    ranked    = combined_recommend_silent(request.title,
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

    return {"query": request.title, "results": results}


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

        qvec = resources.encode(query["description"])
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
    Search by raw description text.
    No title needed — works for movies not in the catalog.
    """
    qvec = resources.encode(request.query)
    distances, indices = resources.search(qvec, request.top_k)

    all_movies = get_all_movies()
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        m = all_movies[idx]
        results.append({
            "title":      m["title"],
            "year":       m["year"],
            "genre":      m["genre"],
            "similarity": round(float(dist), 4)
        })

    return {"query": request.query, "results": results}

# ── Health check ────────────────────────────────────────
@app.get("/health")
def health(resources: AppResources = Depends(get_resources)):
    return {
        "status":     "ok",
        "model":      "llama3.2",
        "index_size": resources.index.ntotal
    }