import redis
import json
import hashlib
from functools import lru_cache
from config import REDIS_HOST, REDIS_PORT, REDIS_TTL, REDIS_DB_CACHE, REDIS_URL


@lru_cache(maxsize=1)
def get_redis_client():
    """
    Singleton Redis client — created once, reused forever.

    Connection priority:
        1. REDIS_URL set → Upstash (production)
        2. REDIS_URL not set → local Redis via host/port (dev)

    Returns None if Redis unavailable — graceful degradation.
    App works without cache, just slower.
    """
    try:
        if REDIS_URL:
            # Upstash — TLS connection via URL
            # decode_responses=True so we get strings not bytes
            client = redis.from_url(
                REDIS_URL,
                decode_responses=True,
                socket_connect_timeout=2
            )
            client.ping()
            print("[Cache] Upstash Redis connected")
        else:
            # Local Redis — host/port connection
            client = redis.Redis(
                host=REDIS_HOST,
                port=REDIS_PORT,
                db=REDIS_DB_CACHE,
                decode_responses=True,
                socket_connect_timeout=2
            )
            client.ping()
            print("[Cache] Local Redis connected")
        return client
    except Exception:
        print("[Cache] Redis unavailable — caching disabled")
        return None


def make_cache_key(endpoint: str, params: dict) -> str:
    """
    Creates a deterministic cache key from endpoint + params.
    sort_keys=True ensures param order doesn't affect the key.
    MD5 keeps keys short regardless of param complexity.
    """
    param_str  = json.dumps(params, sort_keys=True)
    param_hash = hashlib.md5(param_str.encode()).hexdigest()
    return f"cde:{endpoint}:{param_hash}"


def get_cached(key: str):
    """
    Retrieve cached result.
    Refreshes TTL on every hit — popular keys never expire.
    Returns None on miss or if Redis unavailable.
    """
    client = get_redis_client()
    if not client:
        return None
    try:
        value = client.get(key)
        if value:
            # refresh TTL every time this key is accessed
            # popular keys stay alive, cold keys naturally expire
            client.expire(key, REDIS_TTL)
            return json.loads(value)
        return None
    except Exception:
        return None


def set_cached(key: str, value: dict, ttl: int = REDIS_TTL):
    """
    Store result in cache with TTL expiry.
    Silently fails if Redis unavailable.
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
    Call when data changes — e.g. new movies added.
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
            "status":      "connected",
            "backend":     "upstash" if REDIS_URL else "local",
            "cached_keys": len(keys),
            "hits":        info.get("keyspace_hits", 0),
            "misses":      info.get("keyspace_misses", 0),
            "hit_ratio":   round(
                info.get("keyspace_hits", 0) /
                max(1, info.get("keyspace_hits", 0) +
                       info.get("keyspace_misses", 0)),
                3
            ),
            "memory_used": info.get("used_memory_human", "unknown"),
            "ttl_seconds": REDIS_TTL,
        }
    except Exception:
        return {"status": "error"}