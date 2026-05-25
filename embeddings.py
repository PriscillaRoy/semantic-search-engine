import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
from database import init_db, get_all_movies, get_movie_by_title

# ── Paths ──────────────────────────────────────────────
from config import (DATA_PATH, INDEX_PATH, META_PATH, 
                    EMB_PATH, EMBEDDING_MODEL)

# ── Step 1: Load data ──────────────────────────────────
def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"[Step 1] Loaded {len(df)} movies")
    return df

# ── Step 2: Embed descriptions ─────────────────────────
def embed_descriptions(df):
    print("[Step 2] Loading sentence-transformer model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    texts = df["description"].tolist()
    print(f"         Embedding {len(texts)} descriptions...")
    embeddings = model.encode(texts, show_progress_bar=True)

    # Normalize to unit vectors so dot product = cosine similarity
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    print(f"         Shape: {embeddings.shape}")
    return embeddings, model

# ── Step 3: Build FAISS index ──────────────────────────
def build_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # IP = inner product (cosine sim)
    index.add(embeddings)
    print(f"[Step 3] FAISS index built — {index.ntotal} vectors, dim={dim}")
    return index

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

def search_by_description(query_text: str, top_k: int = 4):
    """
    Search by raw description text.
    Works for ANY query — movie doesn't need to be in catalog.
    This is the core of semantic search.
    """
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    model      = payload["model"]
    all_movies = get_all_movies()

    # Encode the raw query text directly
    qvec = model.encode([query_text]).astype(np.float32)
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