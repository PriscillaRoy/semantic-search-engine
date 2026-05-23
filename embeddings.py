import pandas as pd
import numpy as np
import faiss
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────
DATA_PATH  = Path("data/movies.csv")
INDEX_PATH = Path("indexes/movies.faiss")
META_PATH  = Path("indexes/movies_meta.pkl")

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
def save(index, model, df):
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "wb") as f:
        pickle.dump({
            "meta": df.to_dict(orient="records"),
            "model": model
        }, f)
    print(f"[Save] Index and metadata saved\n")

# ── Step 4: Query similar titles ───────────────────────
def find_similar(query_title, top_k=4):
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    meta  = payload["meta"]
    model = payload["model"]

    match = [m for m in meta if m["title"].lower() == query_title.lower()]
    if not match:
        print(f"'{query_title}' not found.")
        return

    q = match[0]
    print(f"Query: {q['title']} ({q['year']}) [{q['genre']}]")

    qvec = model.encode([q["description"]]).astype(np.float32)
    faiss.normalize_L2(qvec)

    distances, indices = index.search(qvec, top_k + 1)

    print(f"Top {top_k} similar movies:")
    shown = 0
    for dist, idx in zip(distances[0], indices[0]):
        r = meta[idx]
        if r["title"].lower() == query_title.lower():
            continue
        print(f"  {shown+1}. {r['title']} ({r['year']}) [{r['genre']}]  sim={dist:.4f}")
        shown += 1
        if shown == top_k:
            break
    print()

# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    embeddings, model = embed_descriptions(df)
    index = build_index(embeddings)
    save(index, model, df)

    print("=" * 50)
    for title in ["Inception", "Hereditary", "The Martian"]:
        find_similar(title)