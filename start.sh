#!/bin/bash
# start.sh — Production server startup
# Usage: ./start.sh [dev|prod]

MODE=${1:-dev}

echo "Starting Semantic Search Engine in $MODE mode..."

if [ "$MODE" = "local" ]; then
    # Production — Gunicorn with multiple workers
    gunicorn api.main:app --config gunicorn_config.py
else
    # Development — single Uvicorn worker with auto-reload
    uvicorn api.main:app \
        --host 0.0.0.0 \
        --port 8000 \
        --reload
fi