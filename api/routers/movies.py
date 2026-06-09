# api/routers/movies.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, field_validator

from config import DEFAULT_TOP_K, MAX_TOP_K, MIN_TOP_K, SEARCH_BACKEND
from store.database import get_all_movies, get_movie_by_title, get_movie_count
from core.graph import combined_recommend_silent
from core.rag import retrieve, build_prompt, generate
from api.dependencies import AppResources, get_resources
from core.embeddings import search_by_description, prepare_text
from store.cache import make_cache_key, get_cached, set_cached
from core.milvus_store import milvus_search

router = APIRouter(tags=["movies"])


# ── Models ─────────────────────────────────────────────
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


# ── GET /similar/{title} ────────────────────────────────
@router.get("/similar/{title}")
def get_similar(
    title: str,
    top_k: int = DEFAULT_TOP_K,
    year: int = None,
    return_all: bool = False,
    resources: AppResources = Depends(get_resources)
):
    cache_key = make_cache_key("similar", {
        "title": title, "top_k": top_k,
        "year": year, "return_all": return_all
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    matches = get_movie_by_title(title, year=year)
    if not matches:
        raise HTTPException(status_code=404, detail=f"'{title}' not found")

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


# ── POST /recommend ─────────────────────────────────────
@router.post("/recommend")
def recommend(
    request: RecommendRequest,
    resources: AppResources = Depends(get_resources)
):
    cache_key = make_cache_key("recommend", {
        "title": request.title, "top_k": request.top_k
    })
    cached = get_cached(cache_key)
    if cached:
        cached["cached"] = True
        return cached

    matches = get_movie_by_title(request.title)
    if not matches:
        raise HTTPException(status_code=404,
                            detail=f"'{request.title}' not found")

    ranked     = combined_recommend_silent(request.title, top_k=request.top_k)
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

    response = {"query": request.title, "results": results, "cached": False}
    set_cached(cache_key, response)
    return response


# ── POST /explain ───────────────────────────────────────
@router.post("/explain")
def explain(
    request: ExplainRequest,
    resources: AppResources = Depends(get_resources)
):
    query_movie, retrieved = retrieve(request.title, top_k=request.top_k)
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


# ── POST /search ────────────────────────────────────────
@router.post("/search")
def search(
    request: SearchRequest,
    resources: AppResources = Depends(get_resources)
):
    """
    Unified search endpoint.
    Backend controlled by SEARCH_BACKEND in config:
        faiss  → local FAISS index
        milvus → Milvus vector database
    """
    cache_key = make_cache_key("search", {
        "query":   request.query,
        "top_k":   request.top_k,
        "backend": SEARCH_BACKEND
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