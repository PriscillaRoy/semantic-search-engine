# store/supabase_client.py
import psycopg2
import psycopg2.extras
from supabase import create_client, Client
from config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_DB_URL


# ── Supabase auth client (singleton) ──────────────────
_supabase_client: Client | None = None

def get_supabase_client() -> Client:
    """
    Returns a singleton Supabase client for auth operations.
    Used for: sign up, sign in, JWT verification.
    """
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in .env"
            )
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
    return _supabase_client


# ── PostgreSQL direct connection ───────────────────────
def get_pg_connection():
    """
    Returns a raw psycopg2 connection to Supabase PostgreSQL.
    Used for: table creation, bulk inserts, complex queries.
    Caller is responsible for closing the connection.

    Usage:
        conn = get_pg_connection()
        try:
            ...
        finally:
            conn.close()
    """
    if not SUPABASE_DB_URL:
        raise RuntimeError("SUPABASE_DB_URL must be set in .env")
    conn = psycopg2.connect(SUPABASE_DB_URL)
    conn.autocommit = False  # explicit commits required
    return conn


def get_pg_cursor(conn):
    """
    Returns a DictCursor so rows come back as dicts,
    matching the sqlite3.Row behavior in database.py.
    """
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ── Connection health check ────────────────────────────
def ping_supabase() -> bool:
    """
    Quick connectivity check.
    Returns True if Supabase PostgreSQL is reachable.
    """
    try:
        conn = get_pg_connection()
        cursor = get_pg_cursor(conn)
        cursor.execute("SELECT 1")
        conn.close()
        print("[Supabase] ✓ Connection healthy")
        return True
    except Exception as e:
        print(f"[Supabase] ✗ Connection failed: {e}")
        return False


# ── Auth helpers ───────────────────────────────────────
def sign_up(email: str, password: str) -> dict:
    """
    Creates a new user in Supabase Auth.
    Returns user data dict on success.
    """
    client = get_supabase_client()
    response = client.auth.sign_up({"email": email, "password": password})
    if response.user is None:
        raise RuntimeError(f"Sign up failed: {response}")
    return {
        "user_id": response.user.id,
        "email":   response.user.email,
    }


def sign_in(email: str, password: str) -> dict:
    """
    Signs in an existing user.
    Returns user data + JWT access token.
    """
    client = get_supabase_client()
    response = client.auth.sign_in_with_password(
        {"email": email, "password": password}
    )
    if response.user is None:
        raise RuntimeError(f"Sign in failed: {response}")
    return {
        "user_id":      response.user.id,
        "email":        response.user.email,
        "access_token": response.session.access_token,
    }


def sign_out(jwt: str) -> bool:
    """Signs out the current user by invalidating their JWT."""
    client = get_supabase_client()
    client.auth.sign_out()
    return True


def get_user_from_jwt(jwt: str) -> dict | None:
    """
    Validates a JWT and returns the user payload.
    Returns None if token is invalid or expired.
    Used as a dependency in protected FastAPI endpoints.
    """
    try:
        client = get_supabase_client()
        response = client.auth.get_user(jwt)
        if response.user is None:
            return None
        return {
            "user_id": response.user.id,
            "email":   response.user.email,
        }
    except Exception:
        return None


# ── Main — connectivity test ───────────────────────────
if __name__ == "__main__":
    ping_supabase()