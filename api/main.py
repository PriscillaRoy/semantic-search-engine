import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
from fastapi import FastAPI
from api.routers import auth, movies, features, admin
from fastapi.middleware.cors import CORSMiddleware

# ── App ────────────────────────────────────────────────
app = FastAPI(
    title="Semantic Search Engine",
    description="Movie recommendations using FAISS + Graph + RAG",
    version="1.0.0"
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
 
# ── Routers ────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(movies.router)
app.include_router(features.router)
app.include_router(admin.router)