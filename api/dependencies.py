# api/dependencies.py
import numpy as np
from functools import lru_cache
from config import INDEX_PATH, META_PATH, SEARCH_BACKEND


class AppResources:
    """
    Holds all shared resources — index, model.
    Loaded once at startup, injected into endpoints.

    When SEARCH_BACKEND=milvus: skips FAISS index load,
    loads embedding model only (still needed for encoding queries).

    When SEARCH_BACKEND=faiss: loads both index and model.
    """
    def __init__(self):
        if SEARCH_BACKEND == "milvus":
            self._load_model_only()
        else:
            self._load_faiss_and_model()

    def _load_model_only(self):
        """Milvus mode — load embedding model only, no FAISS index."""
        import pickle
        print("[Resources] Milvus backend — loading model only...")
        with open(META_PATH, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        self.index = None   # not used in milvus mode
        print("[Resources] Ready — model loaded, FAISS skipped")

    def _load_faiss_and_model(self):
        """FAISS mode — load both index and model."""
        import faiss
        import pickle
        print("[Resources] FAISS backend — loading index + model...")
        self.index = faiss.read_index(str(INDEX_PATH))
        with open(META_PATH, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]
        print(f"[Resources] Ready — {self.index.ntotal} vectors loaded")

    def encode(self, text: str) -> np.ndarray:
        """Encode and normalize a single text query."""
        import faiss
        vec = self.model.encode([text]).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def search(self, vec: np.ndarray, top_k: int):
        """
        Search FAISS index.
        Only called in FAISS mode — Milvus endpoints call milvus_search() directly.
        """
        if self.index is None:
            raise RuntimeError(
                "FAISS index not loaded — switch SEARCH_BACKEND to 'faiss'"
            )
        return self.index.search(vec, top_k)


@lru_cache(maxsize=1)
def get_resources() -> AppResources:
    """
    Returns singleton AppResources instance.
    lru_cache ensures it's only created once
    no matter how many times this is called.
    """
    return AppResources()