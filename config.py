# config.py
from pathlib import Path

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
FAISS_INDEX_TYPE     = "flat"   # "flat" | "ivf"
IVF_NLIST            = 4        # number of clusters for IVF

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