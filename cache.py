# cache.py
from venv import logger

import redis
import json
import hashlib
from config import REDIS_HOST, REDIS_PORT, REDIS_TTL

def get_redis_client():
    """
    Returns a Redis client.
    Returns None if Redis is unavailable —
    app continues without caching rather than crashing.
    """
    try:
        client = redis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            decode_responses=True,
            socket_connect_timeout=2
        )
        client.ping()
        return client
    except Exception:
        return None


def make_cache_key(endpoint: str, params: dict) -> str:
    """
    Creates a deterministic cache key from endpoint + params.
    Uses MD5 hash so keys stay short regardless of param size.

    Example:
      endpoint="similar", params={"title": "Inception", "top_k": 4}
      → "cde:similar:a3f8c2..."
    """
    param_str = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"cde:{endpoint}:{param_hash}"


def get_cached(key: str):
    """
    Retrieve cached result.
    Returns parsed dict or None if miss/unavailable.
    """
    client = get_redis_client()
    if not client:
        return None
    try:
        value = client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception:
        return None


def set_cached(key: str, value: dict, ttl: int = REDIS_TTL):
    """
    Store result in cache with TTL expiry.
    Silently fails if Redis is unavailable.
    """
    client = get_redis_client()
    if not client:
        return
    try:
        client.setex(key, ttl, json.dumps(value))
    except Exception:
        pass


def invalidate_cache(pattern: str = "cde:*"):
    """
    Delete all cache keys matching pattern.
    Call this when data changes — e.g. new movies added.
    """
    client = get_redis_client()
    if not client:
        return 0
    try:
        keys = client.keys(pattern)
        if keys:
            client.delete(*keys)
        return len(keys)
    except Exception:
        return 0


def get_cache_stats() -> dict:
    """
    Returns cache hit/miss stats and memory usage.
    Useful for monitoring and tuning TTL.
    """
    client = get_redis_client()
    if not client:
        return {"status": "unavailable"}
    try:
        info = client.info()
        keys = client.keys("cde:*")
        return {
            "status":        "connected",
            "cached_keys":   len(keys),
            "hits":          info.get("keyspace_hits", 0),
            "misses":        info.get("keyspace_misses", 0),
            "memory_used":   info.get("used_memory_human", "unknown"),
            "ttl_seconds":   REDIS_TTL,
        }
    except Exception:
        return {"status": "error"}