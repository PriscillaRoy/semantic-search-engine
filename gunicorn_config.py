# gunicorn_config.py
import multiprocessing
import os

multiprocessing.set_start_method("spawn", force=True)
# add preload_app to load model before forking
preload_app = True

# ── Server socket ──────────────────────────────────────
bind             = "0.0.0.0:8000"
backlog          = 2048

# ── Workers ────────────────────────────────────────────
# (2 × CPU cores) + 1 is the standard formula
# Your M2 Pro: 10 cores → 21 workers
# For local simulation cap at 4 like your work project
workers = int(os.getenv("GUNICORN_WORKERS",
              # multiprocessing.cpu_count() * 2 + 1  # production
              2                                       # local testing
              ))
worker_class     = "uvicorn.workers.UvicornWorker"
worker_connections = 1000

# timeout = 0 for long-running jobs (like bulk imports)
# timeout = 120 for standard request/response APIs
timeout          = int(os.getenv("GUNICORN_TIMEOUT", 120))
keepalive        = 5

# ── Logging ────────────────────────────────────────────
# "-" = stdout/stderr — cloud-native, works with Docker
# Override with file paths for local debugging
loglevel         = os.getenv("LOG_LEVEL", "info")
accesslog        = os.getenv("ACCESS_LOG", "-")
errorlog         = os.getenv("ERROR_LOG", "-")
access_log_format = '%(h)s "%(r)s" %(s)s %(b)s %(D)sμs'

# ── Process naming ─────────────────────────────────────
proc_name        = "Semantic Search"

# ── Lifecycle hooks ────────────────────────────────────
def on_starting(server):
    print(f"[Gunicorn] Starting — "
          f"workers={workers}, "
          f"worker_class={worker_class}, "
          f"timeout={timeout}")

def on_exit(server):
    print("[Gunicorn] Shutdown complete")

def worker_exit(server, worker):
    print(f"[Gunicorn] Worker {worker.pid} exited")