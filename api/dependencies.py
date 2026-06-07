# dependencies.py
import faiss
import pickle
import numpy as np
from functools import lru_cache
from config import INDEX_PATH, META_PATH


class AppResources:
    """
    Holds all shared resources — index, model.
    Loaded once at startup, injected into endpoints.
    Single source of truth, no globals.
    """
    def __init__(self):
        print("[Resources] Loading FAISS index...")
        self.index = faiss.read_index(str(INDEX_PATH))

        print("[Resources] Loading model...")
        with open(META_PATH, "rb") as f:
            payload = pickle.load(f)
        self.model = payload["model"]

        print(f"[Resources] Ready — {self.index.ntotal} vectors loaded")

    def encode(self, text: str) -> np.ndarray:
        """Encode and normalize a single text query."""
        vec = self.model.encode([text]).astype(np.float32)
        faiss.normalize_L2(vec)
        return vec

    def search(self, vec: np.ndarray, top_k: int):
        """Search FAISS index."""
        return self.index.search(vec, top_k)


@lru_cache(maxsize=1)
def get_resources() -> AppResources:
    """
    Returns singleton AppResources instance.
    lru_cache ensures it's only created once
    no matter how many times this is called.
    """
    return AppResources()