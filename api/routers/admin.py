# api/routers/admin.py
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from config import SEARCH_BACKEND
from store.database import get_all_movies
from api.dependencies import AppResources, get_resources
from core.embeddings import prepare_text
from store.cache import make_cache_key, get_cached, get_cache_stats
from core.milvus_store import upsert_movie

router = APIRouter(tags=["admin"])


# ── Models ─────────────────────────────────────────────
class UpsertRequest(BaseModel):
    id:          int
    title:       str
    year:        int
    genre:       str
    description: str


# ── GET /health ─────────────────────────────────────────
@router.get("/health")
def health(resources: AppResources = Depends(get_resources)):
    return {
        "status":         "ok",
        "model":          "llama3.2",
        "index_size":     resources.index.ntotal,
        "search_backend": SEARCH_BACKEND
    }


# ── GET /evaluate ───────────────────────────────────────
@router.get("/evaluate")
def evaluate(
    top_k: int = 4,
    resources: AppResources = Depends(get_resources)
):
    from config import DEFAULT_TOP_K
    all_movies = get_all_movies()
    scores     = []

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


# ── GET /cache/stats ────────────────────────────────────
@router.get("/cache/stats")
def cache_stats():
    return get_cache_stats()


# ── DELETE /cache/clear ─────────────────────────────────
@router.delete("/cache/clear")
def clear_cache():
    from store.cache import invalidate_cache
    count = invalidate_cache()
    return {"message": f"Cleared {count} cached keys"}


# ── POST /milvus/upsert ─────────────────────────────────
@router.post("/milvus/upsert")
def upsert(request: UpsertRequest):
    """
    Add or update a movie in Milvus in real-time.
    No index rebuild needed — available for search instantly.
    Key advantage over FAISS.
    """
    upsert_movie(request.dict())

    from store.cache import invalidate_cache
    invalidate_cache("cde:search_milvus:*")

    return {
        "status":  "upserted",
        "title":   request.title,
        "message": "Available for search immediately"
    }