import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import ollama
from pathlib import Path
from core.graph import combined_recommend_silent
from store.database import get_movie_by_title, get_all_movies
from config import (DATA_PATH, INDEX_PATH, META_PATH,
                    GRAPH_PATH, OLLAMA_MODEL)


def retrieve(query_title, top_k=3):
    matches = get_movie_by_title(query_title)
    if not matches:
        print(f"'{query_title}' not found.")
        return None, []
    q = matches[0]    # ← fix: get first match from list

    ranked    = combined_recommend_silent(query_title, top_k=top_k)
    all_movies = get_all_movies()
    movie_map  = {m["title"]: m for m in all_movies}

    results = []
    for title, score in ranked:
        m = movie_map.get(title)
        if m:
            results.append({
                "title":       m["title"],
                "year":        m["year"],
                "genre":       m["genre"],
                "description": m["description"],
                "similarity":  round(score, 4)
            })

    return q, results


def build_prompt(query_movie, retrieved_movies):
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


def generate(prompt):
    # response = ollama.chat(
    #     model=OLLAMA_MODEL,
    #     messages=[{"role": "user", "content": prompt}]
    # )
    from groq import Groq
    import os
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content


def rag_recommend(query_title):
    print(f"\n{'═' * 55}")
    print(f"RAG Recommendations for: '{query_title}'")
    print(f"{'═' * 55}")

    print("[1/3] Retrieving similar movies via FAISS...")
    query_movie, retrieved = retrieve(query_title)
    if not retrieved:
        return

    print(f"      Found: {[m['title'] for m in retrieved]}")
    print("[2/3] Building prompt with retrieved context...")
    prompt = build_prompt(query_movie, retrieved)

    print("[3/3] Generating explanation with LLaMA 3.2...")
    explanation = generate(prompt)

    print(f"\n{explanation}")
    return explanation


if __name__ == "__main__":
    queries = ["The Martian", "Inception", "Hereditary"]
    for title in queries:
        rag_recommend(title)