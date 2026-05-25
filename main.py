import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from graph import combined_recommend_silent
from rag import retrieve, build_prompt, generate

# ── Paths ──────────────────────────────────────────────
INDEX_PATH = Path("indexes/movies.faiss")
META_PATH  = Path("indexes/movies_meta.pkl")

# ── App ────────────────────────────────────────────────
app = FastAPI(
    title="Content Discovery Engine",
    description="Movie recommendations using FAISS + Graph + RAG",
    version="1.0.0"
)

# ── Load index + model once at startup ─────────────────
@app.on_event("startup")
def load_resources():
    global index, meta, model
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    meta  = payload["meta"]
    model = payload["model"]
    print("[Startup] FAISS index and model loaded")


# ── Request/Response models ────────────────────────────
class RecommendRequest(BaseModel):
    title: str
    top_k: int = 4

class ExplainRequest(BaseModel):
    title: str
    top_k: int = 3


# ── Endpoint 1: GET /similar/{title} ───────────────────
@app.get("/similar/{title}")
def get_similar(title: str, top_k: int = 4):
    """
    Pure FAISS similarity search.
    Fast, no graph, no LLM.
    """
    match = [m for m in meta if m["title"].lower() == title.lower()]
    if not match:
        raise HTTPException(status_code=404, detail=f"'{title}' not found")

    q = match[0]
    qvec = model.encode([q["description"]]).astype(np.float32)
    faiss.normalize_L2(qvec)
    distances, indices = index.search(qvec, top_k + 1)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        m = meta[idx]
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

    return {"query": title, "results": results}


# ── Endpoint 2: POST /recommend ─────────────────────────
@app.post("/recommend")
def recommend(request: RecommendRequest):
    """
    Combined FAISS + Graph recommendations.
    Better than FAISS alone — uses genre relationships.
    """
    match = [m for m in meta if m["title"].lower() == request.title.lower()]
    if not match:
        raise HTTPException(status_code=404, detail=f"'{request.title}' not found")

    ranked = combined_recommend_silent(request.title, top_k=request.top_k)

    results = []
    for title, score in ranked:
        movie = next((m for m in meta if m["title"] == title), None)
        if movie:
            results.append({
                "title": movie["title"],
                "year":  movie["year"],
                "genre": movie["genre"],
                "score": round(score, 4)
            })

    return {"query": request.title, "results": results}


# ── Endpoint 3: POST /explain ───────────────────────────
@app.post("/explain")
def explain(request: ExplainRequest):
    """
    Full RAG pipeline — FAISS + Graph retrieval
    with LLM-generated natural language explanation.
    Slowest endpoint — calls Ollama locally.
    """
    query_movie, retrieved = retrieve(request.title, top_k=request.top_k)
    if not retrieved:
        raise HTTPException(status_code=404, detail=f"'{request.title}' not found")

    prompt      = build_prompt(query_movie, retrieved)
    explanation = generate(prompt)

    return {
        "query":           request.title,
        "recommendations": [m["title"] for m in retrieved],
        "explanation":     explanation
    }

# ── Endpoint 4: GET /evaluate ───────────────────────────
@app.get("/evaluate")
def evaluate(top_k: int = 4):
    """
    Precision@k evaluation.
    Measures what fraction of top-k recommendations
    share the same genre as the query movie.
    Genre match is a proxy for relevance.
    """
    scores = []

    for query in meta:
        query_title = query["title"]
        query_genre = query["genre"]

        # Get FAISS recommendations
        qvec = model.encode([query["description"]]).astype(np.float32)
        faiss.normalize_L2(qvec)
        distances, indices = index.search(qvec, top_k + 1)

        relevant = 0
        shown    = 0
        for dist, idx in zip(distances[0], indices[0]):
            m = meta[idx]
            if m["title"].lower() == query_title.lower():
                continue
            if m["genre"] == query_genre:
                relevant += 1
            shown += 1
            if shown == top_k:
                break

        precision = relevant / top_k
        scores.append({
            "title":      query_title,
            "genre":      query_genre,
            "precision":  round(precision, 4)
        })

    avg_precision = round(sum(s["precision"] for s in scores) / len(scores), 4)

    return {
        "metric":        "precision@k",
        "k":             top_k,
        "avg_precision": avg_precision,
        "per_movie":     sorted(scores, key=lambda x: x["precision"], reverse=True)
    }

# ── Health check ────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "model": "llama3.2", "index_size": index.ntotal}