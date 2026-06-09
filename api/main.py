import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
from fastapi import FastAPI
from api.routers import auth, movies, features, admin
 
# ── App ────────────────────────────────────────────────
app = FastAPI(
    title="Content Discovery Engine",
    description="Movie recommendations using FAISS + Graph + RAG",
    version="1.0.0"
)
 
# ── Routers ────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(movies.router)
app.include_router(features.router)
app.include_router(admin.router)