# store/database.py
"""
Database router — picks backend based on DB_BACKEND in .env:
    DB_BACKEND=sqlite    → store/database_sqlite.py (default)
    DB_BACKEND=supabase  → store/database_supabase.py (production)

All other modules import from here — they never need to know which
backend is active.

Usage:
    from store.database import get_all_movies, get_movie_by_title
"""
from config import DB_BACKEND

if DB_BACKEND == "supabase":
    from store.database_supabase import (
        init_db,
        get_movie_count,
        get_all_movies,
        get_movie_by_title,
        get_movies_by_genre,
        get_movies_by_ids,
        search_by_description_text,
        get_duplicate_titles,
        upsert_rating,
        get_user_ratings,
        log_watch_event,
    )
else:
    from store.database_sqlite import (
        init_db,
        get_movie_count,
        get_all_movies,
        get_movie_by_title,
        get_movies_by_genre,
        get_movies_by_ids,
        search_by_description_text,
        get_duplicate_titles,
    )