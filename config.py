# config.py
from pathlib import Path
from dotenv import load_dotenv
import math
import os
import psycopg2
import sqlite3

load_dotenv()  # loads .env file from project root

# ── Paths ──────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_PATH  = BASE_DIR / "data/movies.csv"
INDEX_PATH = BASE_DIR / "indexes/movies.faiss"
META_PATH  = BASE_DIR / "indexes/movies_meta.pkl"
GRAPH_PATH = BASE_DIR / "indexes/movies_graph.pkl"
DB_PATH    = BASE_DIR / "indexes/movies.db"
EMB_PATH   = BASE_DIR / "indexes/embeddings.npy"

# ── Embedding model ────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── FAISS ──────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.25
# ── FAISS index type ───────────────────────────────────
# "flat"   → IndexFlatIP, exact brute force
# "ivf"    → IndexIVFFlat, approximate cluster search
# "ivfpq"  → IndexIVFPQ, compressed cluster search
FAISS_INDEX_TYPE = "ivf"
# ── FAISS metric ───────────────────────────────────────
# "ip" → Inner Product (cosine sim on normalized vectors)
# "l2" → Euclidean distance
FAISS_METRIC = "ip"

# ── IVF parameters ────────────────────────────────────
def get_optimal_nlist() -> int:
    """
    Dynamically compute nlist based on current dataset size.
    FAISS requires at least nlist * 39 training points.
    So nlist <= n // 39. Also minimum 4.
    Falls back to SQLite if Supabase is not configured.
    """
    try:
        # Try Supabase/PostgreSQL first
        db_url = os.getenv("SUPABASE_DB_URL")
        if db_url:
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM movies")
            n = cursor.fetchone()[0]
            conn.close()
        else:
            # Fall back to local SQLite
            conn = sqlite3.connect(str(DB_PATH))
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM movies")
            n = cursor.fetchone()[0]
            conn.close()

        max_safe = max(4, n // 39)           # FAISS constraint
        natural  = max(4, int(math.sqrt(n))) # ideal clustering
        return min(max_safe, natural)
    except:
        return 4

IVF_NLIST  = get_optimal_nlist()
IVF_NPROBE = max(1, IVF_NLIST // 2)   # search 50% of clusters — more accurate
# ── PQ parameters (only used when FAISS_INDEX_TYPE = "ivfpq") ──
PQ_M = 8

# ── API ────────────────────────────────────────────────
DEFAULT_TOP_K = 4
MAX_TOP_K     = 20
MIN_TOP_K     = 1

# ── Ollama ─────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2"

# ── Redis ──────────────────────────────────────────────
# Upstash in production (REDIS_URL set as env var)
# Falls back to local Redis if REDIS_URL not set
REDIS_URL      = os.getenv("REDIS_URL")          # Upstash URL (production)
REDIS_HOST     = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT     = int(os.getenv("REDIS_PORT", 6379))
REDIS_TTL      = 3600   # cache expiry in seconds (1 hour)
REDIS_DB_CACHE = 0      # app cache (ignored when using Upstash URL)
REDIS_DB_FEAST = 1      # Feast online store (local only)

# ── Milvus ─────────────────────────────────────────────
# Zilliz Cloud in production (MILVUS_URI set as env var)
# Falls back to local Docker Milvus if not set
MILVUS_URI        = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_TOKEN      = os.getenv("MILVUS_TOKEN", "")   # Zilliz API token
MILVUS_COLLECTION = "semantic_search_movies"

# ── Search backend ─────────────────────────────────────
# "faiss"  → use local FAISS index (default)
# "milvus" → use Milvus / Zilliz for real-time updates
SEARCH_BACKEND = os.getenv("SEARCH_BACKEND", "faiss")

# ── Supabase ───────────────────────────────────────────
SUPABASE_URL     = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
SUPABASE_DB_URL  = os.getenv("SUPABASE_DB_URL")

# ── Database backend ───────────────────────────────────
# "sqlite"   → local SQLite (default, no credentials needed)
# "supabase" → PostgreSQL on Supabase (production)
DB_BACKEND = os.getenv("DB_BACKEND", "sqlite")