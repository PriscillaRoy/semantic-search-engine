import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import faiss
import pickle
import numpy as np
import pandas as pd
import ollama
from pathlib import Path
from graph import combined_recommend_silent


DATA_PATH  = Path("data/movies.csv")
INDEX_PATH = Path("indexes/movies.faiss")
META_PATH  = Path("indexes/movies_meta.pkl")
GRAPH_PATH = Path("indexes/movies_graph.pkl")


# ── Step 1: Retrieve — FAISS + Graph ──────────────────
def retrieve(query_title, top_k=3):
    """
    Now uses FAISS + Graph combined scoring
    instead of FAISS alone.
    """
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    meta  = payload["meta"]
    model = payload["model"]

    match = [m for m in meta if m["title"].lower() == query_title.lower()]
    if not match:
        print(f"'{query_title}' not found.")
        return None, []

    q = match[0]

    # Use combined FAISS + Graph scoring
    from graph import combined_recommend_silent
    ranked = combined_recommend_silent(query_title, top_k=top_k)

    results = []
    for title, score in ranked:
        movie = next((m for m in meta if m["title"] == title), None)
        if movie:
            results.append({
                "title":       movie["title"],
                "year":        movie["year"],
                "genre":       movie["genre"],
                "description": movie["description"],
                "similarity":  round(score, 4)
            })

    return q, results
# ── Step 2: Augment — Build the prompt ────────────────
def build_prompt(query_movie, retrieved_movies):
    """
    Augment the prompt with retrieved context.
    This is the A in RAG — stuffing context into the prompt.
    The LLM knows NOTHING except what we put here.
    """
    context = ""
    for i, m in enumerate(retrieved_movies):
        context += f"""
Movie {i+1}: {m['title']} ({m['year']}) [{m['genre']}]
Description: {m['description']}
Similarity score to {query_movie['title']}: {m['similarity']}
"""

    prompt = f"""You are a movie recommendation assistant.

A user enjoyed: "{query_movie['title']}" ({query_movie['year']}) [{query_movie['genre']}]
Description: {query_movie['description']}

Based on similarity search, here are recommended movies with their details:
{context}
For each recommended movie, write 1-2 sentences explaining specifically why 
a fan of "{query_movie['title']}" would enjoy it. Be specific — reference 
actual plot elements, themes, or mood from the descriptions above.

Format your response as:
1. [Movie Title] — [explanation]
2. [Movie Title] — [explanation]
3. [Movie Title] — [explanation]
"""
    return prompt


# ── Step 3: Generate — LLM explains the recommendations
def generate(prompt):
    """
    Generate natural language explanation using Ollama.
    This is the G in RAG — generation grounded in context.
    """
    response = ollama.chat(
        model="llama3.2",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]


# ── Full RAG pipeline ──────────────────────────────────
def rag_recommend(query_title):
    print(f"\n{'═' * 55}")
    print(f"RAG Recommendations for: '{query_title}'")
    print(f"{'═' * 55}")

    # Step 1: Retrieve
    print("[1/3] Retrieving similar movies via FAISS...")
    query_movie, retrieved = retrieve(query_title)
    if not retrieved:
        return

    print(f"      Found: {[m['title'] for m in retrieved]}")

    # Step 2: Augment
    print("[2/3] Building prompt with retrieved context...")
    prompt = build_prompt(query_movie, retrieved)

    # Step 3: Generate
    print("[3/3] Generating explanation with LLaMA 3.2...")
    explanation = generate(prompt)

    print(f"\n{explanation}")
    return explanation


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    queries = ["The Martian", "Inception", "Hereditary"]
    for title in queries:
        rag_recommend(title)