import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from store.database import init_db, get_all_movies, get_movie_by_title

# ── Paths ──────────────────────────────────────────────
from config import (DATA_PATH, INDEX_PATH, META_PATH, EMB_PATH, EMBEDDING_MODEL,
                    FAISS_INDEX_TYPE, FAISS_METRIC, IVF_NLIST, IVF_NPROBE, PQ_M,
                    EMBEDDING_DIM)
# ── Step 1: Load data ──────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[Step 1] Loaded {len(df)} movies")
    return df

# ── Step 2: Embed descriptions ─────────────────────────
def prepare_text(movie: dict) -> str:
    """
    Combine multiple fields into richer embedding text.
    Genre prefix anchors the vector in the right semantic space.
    Mirrors the prepare_text_for_embedding pattern from match-ai.
    """
    return (
        f"{movie['title']}. "
        f"Genre: {movie['genre']}. "
        f"Year: {movie['year']}. "
        f"{movie['description']}"
    )

def embed_descriptions(df):
    print("[Step 2] Loading sentence-transformer model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Use richer text instead of raw description only
    texts = [prepare_text(row) for _, row in df.iterrows()]
    print(f"         Embedding {len(texts)} descriptions...")
    print(f"         Sample text: {texts[0][:100]}...")

    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    print(f"         Shape: {embeddings.shape}")
    return embeddings, model

# ── Step 3: Build FAISS index ──────────────────────────
def build_index(embeddings):
    dim       = embeddings.shape[1]
    n_vectors = embeddings.shape[0]

    # ── Metric registry ────────────────────────────────
    # Maps config string → (faiss metric constant, flat index class)
    # Add new metrics here — no changes needed elsewhere
    METRIC_REGISTRY = {
        "ip": (faiss.METRIC_INNER_PRODUCT, faiss.IndexFlatIP),
        "l2": (faiss.METRIC_L2,            faiss.IndexFlatL2),
    }

    if FAISS_METRIC not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown FAISS_METRIC: '{FAISS_METRIC}'. "
            f"Choose from: {list(METRIC_REGISTRY.keys())}"
        )

    metric_constant, FlatIndexClass = METRIC_REGISTRY[FAISS_METRIC]
    quantizer  = FlatIndexClass(dim)
    flat_index = FlatIndexClass(dim)

    print(f"[Step 3] Building index — "
          f"type={FAISS_INDEX_TYPE}, "
          f"metric={FAISS_METRIC}, "
          f"vectors={n_vectors}")

    # ── Safety check ───────────────────────────────────
    if FAISS_INDEX_TYPE in ("ivf", "ivfpq") and n_vectors < IVF_NLIST:
        print(f"[Warning] Stale index or data mismatch. "
              f"Vectors={n_vectors} < nlist={IVF_NLIST}. "
              f"Falling back to flat. Re-run embeddings.py.")
        flat_index.add(embeddings)
        return flat_index

    # ── Index registry ─────────────────────────────────
    # Each entry is a lambda that builds the right index
    # given (quantizer, dim, nlist, metric, pq_m)
    def build_flat(q, d, nlist, metric, pq_m):
        idx = FlatIndexClass(d)
        idx.add(embeddings)
        print(f"[Step 3] Flat — {idx.ntotal} vectors")
        return idx

    def build_ivf(q, d, nlist, metric, pq_m):
        idx = faiss.IndexIVFFlat(q, d, nlist, metric)
        print(f"         Training IVF nlist={nlist}...")
        idx.train(embeddings)
        idx.add(embeddings)
        idx.nprobe = IVF_NPROBE
        print(f"[Step 3] IVF — {idx.ntotal} vectors, "
              f"nlist={nlist}, nprobe={IVF_NPROBE}")
        return idx

    def build_ivfpq(q, d, nlist, metric, pq_m):
        if d % pq_m != 0:
            print(f"[Warning] dim={d} not divisible by "
                  f"PQ_M={pq_m}. Falling back to IVF.")
            return build_ivf(q, d, nlist, metric, pq_m)
        idx = faiss.IndexIVFPQ(q, d, nlist, pq_m, 8)
        print(f"         Training IVF_PQ "
              f"nlist={nlist}, m={pq_m}...")
        idx.train(embeddings)
        idx.add(embeddings)
        idx.nprobe = IVF_NPROBE
        print(f"[Step 3] IVF_PQ — {idx.ntotal} vectors, "
              f"nlist={nlist}, nprobe={IVF_NPROBE}, m={pq_m}")
        return idx

    INDEX_REGISTRY = {
        "flat":  build_flat,
        "ivf":   build_ivf,
        "ivfpq": build_ivfpq,
    }

    if FAISS_INDEX_TYPE not in INDEX_REGISTRY:
        raise ValueError(
            f"Unknown FAISS_INDEX_TYPE: '{FAISS_INDEX_TYPE}'. "
            f"Choose from: {list(INDEX_REGISTRY.keys())}"
        )

    # ── Build ──────────────────────────────────────────
    builder = INDEX_REGISTRY[FAISS_INDEX_TYPE]
    return builder(quantizer, dim, IVF_NLIST,
                   metric_constant, PQ_M)

# ── Save index + metadata ──────────────────────────────
def save(index, model, df, embeddings):
    faiss.write_index(index, str(INDEX_PATH))
    
    # Save embeddings separately
    np.save(str(EMB_PATH), embeddings)
    
    # Save model separately (still pickle — it's a Python object)
    with open(META_PATH, "wb") as f:
        pickle.dump({"model": model}, f)

    # Initialize SQLite with movie data
    init_db()
    
    print(f"[Save] Index, embeddings, model and database saved\n")

# ── Step 4: Query similar titles ───────────────────────
def find_similar(query_title, top_k=4, year=None):
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    model      = payload["model"]
    all_movies = get_all_movies()

    # Handle duplicates
    matches = get_movie_by_title(query_title, year=year)
    if not matches:
        print(f"'{query_title}' not found.")
        return
    if len(matches) > 1 and year is None:
        print(f"Multiple matches for '{query_title}':")
        for m in matches:
            print(f"  → {m['title']} ({m['year']}) [{m['genre']}]")
        print(f"Re-run with year: find_similar('{query_title}', year=1986)")
        return

    q = matches[0]
    print(f"Query: {q['title']} ({q['year']}) [{q['genre']}]")

    qvec = model.encode([q["description"]]).astype(np.float32)
    faiss.normalize_L2(qvec)
    distances, indices = index.search(qvec, top_k + 1)

    print(f"Top {top_k} similar movies:")
    shown = 0
    for dist, idx in zip(distances[0], indices[0]):
        r = all_movies[idx]
        if r["title"].lower() == query_title.lower():
            continue
        print(f"  {shown+1}. {r['title']} ({r['year']}) [{r['genre']}]  sim={dist:.4f}")
        shown += 1
        if shown == top_k:
            break
    print()

def search_by_description(query_text: str, top_k: int = 4, resources=None):
    """
    Search by raw description text.
    Works for ANY query — movie doesn't need to be in catalog.
    Accepts optional resources for dependency injection when called from API.
    Falls back to loading from disk when called as standalone script.
    """
    all_movies = get_all_movies()

    if resources:
        # called from API — use injected resources, no disk reads
        qvec = resources.encode(query_text)
        distances, indices = resources.search(qvec, top_k)
    else:
        # called as standalone script — load from disk
        index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            payload = pickle.load(f)
        model = payload["model"]
        qvec  = model.encode([query_text]).astype(np.float32)
        faiss.normalize_L2(qvec)
        distances, indices = index.search(qvec, top_k)

    print(f"\nSearch results for: '{query_text}'")
    print("─" * 50)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        m = all_movies[idx]
        print(f"  {m['title']} ({m['year']}) [{m['genre']}]  sim={dist:.4f}")
        results.append({
            "title":      m["title"],
            "year":       m["year"],
            "genre":      m["genre"],
            "similarity": round(float(dist), 4)
        })
    print()
    return results

# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    embeddings, model = embed_descriptions(df)
    index = build_index(embeddings)
    save(index, model, df, embeddings)

    print("=" * 50)
    print("SIMILARITY SEARCH")
    print("=" * 50)
    for title in ["Inception", "Hereditary", "The Martian"]:
        find_similar(title)

    print("=" * 50)
    print("DESCRIPTION SEARCH — movies not in catalog")
    print("=" * 50)
    search_by_description("astronaut alone survival space")
    search_by_description("family haunted dark secrets supernatural")
    search_by_description("hacker simulation virtual reality")