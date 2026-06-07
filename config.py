# config.py
from pathlib import Path
import math
import sqlite3

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
    Rule: nlist = max(4, sqrt(n))
    Minimum 4 — IVF needs at least nlist vectors to train.
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM movies")
        n = cursor.fetchone()[0]
        conn.close()
        return max(4, int(math.sqrt(n)))
    except:
        return 4

IVF_NLIST  = get_optimal_nlist()
IVF_NPROBE = max(1, IVF_NLIST // 2)   # search 50% of clusters — more accurate
# ── PQ parameters (only used when FAISS_INDEX_TYPE = "ivfpq") ──
# m = number of subvectors to split each vector into
# dim must be divisible by m
# more m = better accuracy, more memory
# common values: 8, 16, 32
PQ_M = 8

# ── API ────────────────────────────────────────────────
DEFAULT_TOP_K = 4
MAX_TOP_K     = 20
MIN_TOP_K     = 1

# ── Ollama ─────────────────────────────────────────────
OLLAMA_MODEL = "llama3.2"

# ── Redis ──────────────────────────────────────────────
REDIS_HOST  = "localhost"
REDIS_PORT  = 6379
REDIS_TTL   = 3600   # cache expiry in seconds (1 hour)

# ── Milvus ─────────────────────────────────────────────
MILVUS_URI        = "http://localhost:19530"
MILVUS_COLLECTION = "semantic_search_movies"
# ── Search backend ─────────────────────────────────────
# "faiss"  → use local FAISS index (default)
# "milvus" → use Milvus for real-time updates
SEARCH_BACKEND = "faiss"   # "milvus" / "faiss"