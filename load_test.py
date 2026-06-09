# load_test.py
"""
Load test for Semantic Search Engine.
Simulates real user behavior patterns.

Run:
  locust -f load_test.py --host=http://localhost:8000

Then open http://localhost:8089 for the Locust UI.
Or headless:
  locust -f load_test.py --host=http://localhost:8000 \
    --headless -u 50 -r 5 --run-time 60s
"""
from locust import HttpUser, task, between
import random

# realistic query distribution
# popular movies get queried more often
# mirrors real-world Zipf distribution
POPULAR_MOVIES = [
    "Inception", "The Dark Knight", "Interstellar",
    "The Matrix", "Pulp Fiction", "The Godfather",
    "Hereditary", "The Martian", "Gravity"
]

LONG_TAIL_MOVIES = [
    "Suspiria", "Annihilation", "Moon",
    "Chinatown", "Vertigo", "Oldboy",
    "Whiplash", "Fargo", "Nightcrawler"
]

SEARCH_QUERIES = [
    "astronaut survival space",
    "family haunted dark secrets",
    "detective murder mystery city",
    "heist crime organized mob",
    "psychological thriller mind",
    "animated adventure friendship",
    "romance love lost found",
    "dystopian future survival",
]

USER_IDS = [f"user_{i:03d}" for i in range(50)]


class RecommendationUser(HttpUser):
    """
    Simulates a typical user browsing recommendations.
    Most requests hit popular content (cache hits).
    Some requests hit long-tail content (cache misses).
    """
    wait_time = between(0.5, 2.0)   # realistic think time

    @task(40)
    def get_similar_popular(self):
        """Most common — similar movies for popular titles."""
        title = random.choice(POPULAR_MOVIES)
        self.client.get(
            f"/similar/{title}",
            name="/similar/[popular]"
        )

    @task(20)
    def get_recommend_popular(self):
        """Combined recommendations for popular titles."""
        title = random.choice(POPULAR_MOVIES)
        self.client.post(
            "/recommend",
            json={"title": title, "top_k": 4},
            name="/recommend/[popular]"
        )

    @task(15)
    def search_by_description(self):
        """Description-based search."""
        query = random.choice(SEARCH_QUERIES)
        self.client.post(
            "/search",
            json={"query": query, "top_k": 4},
            name="/search"
        )

    @task(10)
    def personalized_recommend(self):
        """Personalized recommendations with user context."""
        title   = random.choice(POPULAR_MOVIES)
        user_id = random.choice(USER_IDS)
        self.client.post(
            "/recommend/personalized",
            json={"title": title, "user_id": user_id, "top_k": 4},
            name="/recommend/personalized"
        )

    @task(8)
    def get_similar_long_tail(self):
        """Less common — long tail content."""
        title = random.choice(LONG_TAIL_MOVIES)
        self.client.get(
            f"/similar/{title}",
            name="/similar/[long_tail]"
        )

    @task(5)
    def get_features(self):
        """Feature store lookup."""
        movie_id = random.randint(0, 100)
        self.client.get(
            f"/features/{movie_id}",
            name="/features/[id]"
        )

    @task(2)
    def health_check(self):
        """Background health monitoring."""
        self.client.get("/health", name="/health")


class HeavyUser(HttpUser):
    """
    Power user — hits the RAG explain endpoint.
    Much slower, fewer of these in the mix.
    """
    wait_time = between(5.0, 15.0)   # slower, expensive

    @task(1)
    def explain(self):
        """Full RAG pipeline — most expensive endpoint."""
        title = random.choice(POPULAR_MOVIES)
        self.client.post(
            "/explain",
            json={"title": title, "top_k": 3},
            name="/explain",
            timeout=60
        )