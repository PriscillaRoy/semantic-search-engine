# api/routers/auth.py
from fastapi import APIRouter, HTTPException, Depends, Header
from pydantic import BaseModel
from store.supabase_client import sign_up, sign_in, sign_out, get_user_from_jwt

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Models ─────────────────────────────────────────────
class AuthRequest(BaseModel):
    email:    str
    password: str


# ── Auth dependency ─────────────────────────────────────
def get_current_user(authorization: str = Header(None)) -> dict:
    """
    Reusable dependency for protected endpoints.
    Extracts JWT from Authorization header, validates it,
    returns user dict or raises 401.

    Usage in any endpoint:
        user: dict = Depends(get_current_user)
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid Authorization header. "
                   "Expected: 'Bearer <token>'"
        )
    token = authorization.split(" ")[1]
    user  = get_user_from_jwt(token)
    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token"
        )
    return user


# ── POST /auth/signup ───────────────────────────────────
@router.post("/signup", status_code=201)
def signup(request: AuthRequest):
    """
    Creates a new user in Supabase Auth.
    Supabase sends a confirmation email automatically.
    """
    try:
        user = sign_up(request.email, request.password)
        return {
            "message": "Account created. Check your email to confirm.",
            "user_id": user["user_id"],
            "email":   user["email"],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ── POST /auth/signin ───────────────────────────────────
@router.post("/signin")
def signin(request: AuthRequest):
    """
    Signs in an existing user.
    Returns a JWT access token — include in the Authorization
    header for protected endpoints:
        Authorization: Bearer <access_token>
    """
    try:
        result = sign_in(request.email, request.password)
        return {
            "user_id":      result["user_id"],
            "email":        result["email"],
            "access_token": result["access_token"],
            "token_type":   "bearer",
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


# ── POST /auth/signout ──────────────────────────────────
@router.post("/signout")
def signout(authorization: str = Header(None)):
    """
    Signs out the current user by invalidating their JWT.
    Frontend should also delete the token from local storage.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="No token provided")
    token = authorization.split(" ")[1]
    sign_out(token)
    return {"message": "Signed out successfully"}


# ── GET /auth/me ────────────────────────────────────────
@router.get("/me")
def me(user: dict = Depends(get_current_user)):
    """
    Returns the current authenticated user.
    Protected — requires valid JWT in Authorization header.
    Use this to verify a token is still valid.
    """
    return {
        "user_id": user["user_id"],
        "email":   user["email"],
    }