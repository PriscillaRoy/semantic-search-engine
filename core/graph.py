import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import networkx as nx
import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from store.database import get_all_movies, get_movie_by_title
from config import (DATA_PATH, INDEX_PATH, META_PATH,
                    GRAPH_PATH, EMB_PATH,
                    SIMILARITY_THRESHOLD, OLLAMA_MODEL)


def build_graph(df):
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_node(row["title"], type="movie",
                   genre=row["genre"], year=row["year"])
    for _, row in df.iterrows():
        genre = row["genre"]
        if not G.has_node(genre):
            G.add_node(genre, type="genre")
        G.add_edge(row["title"], genre, relation="HAS_GENRE")
    print(f"[Graph] Nodes: {G.number_of_nodes()}")
    print(f"[Graph] Edges: {G.number_of_edges()}")
    return G


def add_similarity_edges(G, df, threshold=SIMILARITY_THRESHOLD):
    import faiss  # lazy — only needed when building graph locally
    index      = faiss.read_index(str(INDEX_PATH))
    embeddings = np.load(str(EMB_PATH))
    for i, row in df.iterrows():
        distances, indices = index.search(embeddings[i:i+1], 6)
        for dist, idx in zip(distances[0], indices[0]):
            if idx == i:
                continue
            if dist >= threshold:
                other_title = df.iloc[idx]["title"]
                G.add_edge(row["title"], other_title,
                           relation="IS_SIMILAR_TO",
                           weight=float(dist))
    similar_edges = [(u, v) for u, v, d in G.edges(data=True)
                     if d.get("relation") == "IS_SIMILAR_TO"]
    print(f"[Graph] Similarity edges added: {len(similar_edges)}")
    return G


def save_graph(G):
    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"[Graph] Saved to {GRAPH_PATH}")


def load_graph():
    with open(GRAPH_PATH, "rb") as f:
        return pickle.load(f)


def graph_recommend(query_title, top_k=4, verbose=True):
    G = load_graph()
    if query_title not in G:
        print(f"'{query_title}' not found in graph.")
        return []

    scores = {}
    for neighbor in G.neighbors(query_title):
        node_type = G.nodes[neighbor]["type"]
        edge_data = G[query_title][neighbor]
        relation  = edge_data.get("relation")

        if node_type == "genre":
            for movie_node in G.neighbors(neighbor):
                if movie_node == query_title:
                    continue
                if G.nodes[movie_node]["type"] != "movie":
                    continue
                scores[movie_node] = scores.get(movie_node, 0) + 1

        elif node_type == "movie" and relation == "IS_SIMILAR_TO":
            weight = edge_data.get("weight", 0)
            scores[neighbor] = scores.get(neighbor, 0) + weight

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    if verbose:
        print(f"\nGraph recommendations for '{query_title}':")
        print("─" * 50)
        for i, (title, score) in enumerate(ranked[:top_k]):
            genre = G.nodes[title]["genre"]
            year  = G.nodes[title]["year"]
            print(f"  {i+1}. {title} ({year}) [{genre}]  score={score:.4f}")
        print()

    return ranked[:top_k]


def combined_recommend(query_title, top_k=4):
    import faiss  # lazy — only needed in FAISS mode
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    model      = payload["model"]
    all_movies = get_all_movies()

    matches = get_movie_by_title(query_title)
    if not matches:
        print(f"'{query_title}' not found.")
        return
    q = matches[0]    # ← fix: get first match from list

    qvec = model.encode([q["description"]]).astype(np.float32)
    faiss.normalize_L2(qvec)
    distances, indices = index.search(qvec, top_k + 1)

    faiss_scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        title = all_movies[idx]["title"]
        if title.lower() != query_title.lower():
            faiss_scores[title] = float(dist)

    graph_results = graph_recommend(query_title, top_k=top_k, verbose=False)
    graph_scores  = {title: score for title, score in graph_results}

    all_titles = set(faiss_scores) | set(graph_scores)
    combined = {}
    for title in all_titles:
        f_score = faiss_scores.get(title, 0)
        g_score = graph_scores.get(title, 0)
        g_normalized = min(g_score / 2.0, 1.0)
        combined[title] = (f_score + g_normalized) / 2

    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    print(f"\nCombined recommendations for '{query_title}':")
    print("─" * 50)
    for i, (title, score) in enumerate(ranked[:top_k]):
        print(f"  {i+1}. {title}  combined_score={score:.4f}")
    print()


def combined_recommend_silent(query_title, top_k=4):
    """
    Called by API endpoints — uses graph only in Milvus mode,
    falls back to FAISS+graph in FAISS mode.
    """
    from config import SEARCH_BACKEND

    if SEARCH_BACKEND == "milvus":
        # graph-only recommendations — no faiss needed
        graph_results = graph_recommend(query_title, top_k=top_k, verbose=False)
        return graph_results

    # FAISS mode — combined FAISS + graph
    import faiss  # lazy — only imported in FAISS mode
    index = faiss.read_index(str(INDEX_PATH))
    with open(META_PATH, "rb") as f:
        payload = pickle.load(f)
    model      = payload["model"]
    all_movies = get_all_movies()

    matches = get_movie_by_title(query_title)
    if not matches:
        return []
    q = matches[0]    # ← fix: get first match from list

    qvec = model.encode([q["description"]]).astype(np.float32)
    faiss.normalize_L2(qvec)
    distances, indices = index.search(qvec, top_k + 1)

    faiss_scores = {}
    for dist, idx in zip(distances[0], indices[0]):
        title = all_movies[idx]["title"]
        if title.lower() != query_title.lower():
            faiss_scores[title] = float(dist)

    graph_results = graph_recommend(query_title, top_k=top_k, verbose=False)
    graph_scores  = {title: score for title, score in graph_results}

    all_titles = set(faiss_scores) | set(graph_scores)
    combined = {}
    for title in all_titles:
        f_score = faiss_scores.get(title, 0)
        g_score = graph_scores.get(title, 0)
        g_normalized = min(g_score / 2.0, 1.0)
        combined[title] = (f_score + g_normalized) / 2

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]


def inspect_graph(G):
    print("\n── NODES ──")
    for node, attrs in G.nodes(data=True):
        print(f"  {node}: {attrs}")
    print("\n── EDGES ──")
    for u, v, attrs in G.edges(data=True):
        print(f"  {u} ──{attrs}──► {v}")


if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    G = build_graph(df)
    G = add_similarity_edges(G, df)
    save_graph(G)
    print("=" * 50)
    print("GRAPH READY — Running recommendations")
    print("=" * 50)
    for title in ["The Martian", "Inception", "Hereditary"]:
        combined_recommend(title)