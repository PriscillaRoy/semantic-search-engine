# milvus_store.py
"""
Milvus vector store — replaces FAISS for streaming/real-time updates.

Key advantages over FAISS:
  - Real-time upserts — no index rebuild needed
  - Persistent storage — survives restarts
  - Production-ready — used by your work project at scale
  - IVF_FLAT index — same algorithm, server-managed

Mirrors patterns from match-ai milvus_client.py but uses
the modern MilvusClient API instead of ORM-style connections.
"""
import numpy as np
import faiss
from pymilvus import MilvusClient, DataType
from sentence_transformers import SentenceTransformer
from database import get_all_movies, get_movie_by_title
from config import (MILVUS_URI, MILVUS_COLLECTION,
                    EMBEDDING_DIM, EMBEDDING_MODEL)


# ── Collection schema ──────────────────────────────────
def get_schema(client: MilvusClient):
    """
    Define the collection schema.
    Mirrors match-ai field structure but for movies.
    """
    schema = client.create_schema(
        auto_id=False,
        enable_dynamic_field=True   # allows adding fields later
    )

    schema.add_field(
        field_name="id",
        datatype=DataType.INT64,
        is_primary=True
    )
    schema.add_field(
        field_name="title",
        datatype=DataType.VARCHAR,
        max_length=500
    )
    schema.add_field(
        field_name="genre",
        datatype=DataType.VARCHAR,
        max_length=100
    )
    schema.add_field(
        field_name="year",
        datatype=DataType.INT64
    )
    schema.add_field(
        field_name="embeddings",
        datatype=DataType.FLOAT_VECTOR,
        dim=EMBEDDING_DIM
    )
    return schema


def get_index_params(client: MilvusClient):
    """
    IVF_FLAT index with Inner Product metric.
    Same algorithm as our FAISS index — consistent results.
    nlist=16 for our ~100 movie dataset.
    Your work project uses nlist=1024 for millions of programs.
    """
    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embeddings",
        index_type="IVF_FLAT",
        metric_type="IP",           # Inner Product = cosine sim
        params={"nlist": 16}
    )
    return index_params


# ── Connect ────────────────────────────────────────────
def get_client() -> MilvusClient:
    """Returns a connected MilvusClient."""
    return MilvusClient(uri=MILVUS_URI)


# ── Create collection ──────────────────────────────────
def create_collection(drop_existing: bool = False):
    """
    Create the movies collection with schema and index.
    Safe to call multiple times — skips if already exists.
    """
    client = get_client()

    if client.has_collection(MILVUS_COLLECTION):
        if drop_existing:
            client.drop_collection(MILVUS_COLLECTION)
            print(f"[Milvus] Dropped existing collection: {MILVUS_COLLECTION}")
        else:
            print(f"[Milvus] Collection already exists: {MILVUS_COLLECTION}")
            return client

    schema       = get_schema(client)
    index_params = get_index_params(client)

    client.create_collection(
        collection_name=MILVUS_COLLECTION,
        schema=schema,
        index_params=index_params
    )
    print(f"[Milvus] Created collection: {MILVUS_COLLECTION}")
    return client


# ── Load all movies into Milvus ────────────────────────
def build_milvus_index():
    """
    Embeds all movies and loads them into Milvus.
    One-time setup — after this use upsert() for updates.
    """
    client = create_collection(drop_existing=True)
    model  = SentenceTransformer(EMBEDDING_MODEL)

    movies = get_all_movies()
    print(f"[Milvus] Embedding {len(movies)} movies...")

    # prepare text — same as embeddings.py prepare_text()
    from embeddings import prepare_text
    texts      = [prepare_text(m) for m in movies]
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)

    # batch insert — mirrors milvus_loader.py pattern
    data = []
    for i, movie in enumerate(movies):
        data.append({
            "id":         movie["id"],
            "title":      movie["title"],
            "genre":      movie["genre"],
            "year":       int(movie["year"]),
            "embeddings": embeddings[i].tolist()
        })

    client.insert(
        collection_name=MILVUS_COLLECTION,
        data=data
    )
    client.flush(MILVUS_COLLECTION)

    stats = client.get_collection_stats(MILVUS_COLLECTION)
    print(f"[Milvus] Indexed {stats['row_count']} movies")
    return client


# ── Real-time upsert — the key advantage over FAISS ───
def upsert_movie(movie: dict):
    """
    Add or update a single movie in real-time.
    No index rebuild needed — available for search instantly.
    This is the streaming update FAISS can't do.
    """
    client = get_client()
    model  = SentenceTransformer(EMBEDDING_MODEL)

    from embeddings import prepare_text
    text      = prepare_text(movie)
    embedding = model.encode([text]).astype(np.float32)
    faiss.normalize_L2(embedding)

    client.upsert(
        collection_name=MILVUS_COLLECTION,
        data=[{
            "id":         movie["id"],
            "title":      movie["title"],
            "genre":      movie["genre"],
            "year":       int(movie["year"]),
            "embeddings": embedding[0].tolist()
        }]
    )
    print(f"[Milvus] Upserted: {movie['title']} ({movie['year']})")


# ── Search ─────────────────────────────────────────────
def milvus_search(query_text: str, top_k: int = 4) -> list:
    """
    Semantic search via Milvus.
    Drop-in replacement for FAISS search in embeddings.py.
    """
    client = get_client()
    model  = SentenceTransformer(EMBEDDING_MODEL)

    qvec = model.encode([query_text]).astype(np.float32)
    faiss.normalize_L2(qvec)

    results = client.search(
        collection_name=MILVUS_COLLECTION,
        data=qvec.tolist(),
        limit=top_k,
        output_fields=["title", "genre", "year"],
        search_params={"metric_type": "IP", "params": {"nprobe": 8}}
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "title":      hit["entity"]["title"],
            "genre":      hit["entity"]["genre"],
            "year":       hit["entity"]["year"],
            "similarity": round(hit["distance"], 4)
        })
    return hits


# ── Collection info ────────────────────────────────────
def collection_info() -> dict:
    """Returns collection stats — mirrors /milvus/info endpoint."""
    client = get_client()
    if not client.has_collection(MILVUS_COLLECTION):
        return {"status": "collection not found"}
    stats = client.get_collection_stats(MILVUS_COLLECTION)
    return {
        "collection":  MILVUS_COLLECTION,
        "row_count":   stats["row_count"],
        "index_type":  "IVF_FLAT",
        "metric_type": "IP",
        "dim":         EMBEDDING_DIM
    }


# ── Main ───────────────────────────────────────────────
if __name__ == "__main__":
    print("Building Milvus index...")
    build_milvus_index()

    print("\nCollection info:")
    print(collection_info())

    print("\nTest search: 'astronaut survival space'")
    results = milvus_search("astronaut survival space", top_k=4)
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['title']} ({r['year']}) "
              f"[{r['genre']}] sim={r['similarity']}")

    print("\nTest real-time upsert:")
    upsert_movie({
        "id":          200,
        "title":       "Avatar",
        "year":        2009,
        "genre":       "Sci-Fi",
        "description": "A paraplegic marine dispatched to the moon "
                       "Pandora on a mission becomes torn between "
                       "following his orders and protecting the world "
                       "he feels is his home."
    })
    print("\nSearch after upsert:")
    results = milvus_search("alien world nature humans", top_k=4)
    for i, r in enumerate(results):
        print(f"  {i+1}. {r['title']} ({r['year']}) "
              f"[{r['genre']}] sim={r['similarity']}")