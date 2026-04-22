"""
src/data/data_prep/build_graph.py

Offline script — run once to build the image knowledge graph.

For each image, finds its top-K nearest neighbors using the
pre-computed field embeddings (mood + color_palette + visual_description).
Saves a lightweight adjacency structure to disk.

The graph is used at query time by GraphRAGAgent to boost candidates
that form coherent clusters.

Usage:
    python src/data/data_prep/build_graph.py
    python src/data/data_prep/build_graph.py --top_k 15 --threshold 0.75

Output:
    src/data/graph/image_graph.pkl
    src/data/graph/graph_meta.json   (stats)

Requirements:
    pip install faiss-cpu  (or faiss-gpu)
"""

import os
import json
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

BASE_DIR      = Path(__file__).resolve().parents[3]
EMBEDDING_NPZ = BASE_DIR / "src/data/embeddings/field_embeddings.npz"
DATASET_JSON  = BASE_DIR / "src/data/processed/description_grounding_outputs.json"
GRAPH_DIR     = BASE_DIR / "src/data/graph"

# Edge weights per field — mood and color are most important for coherence
FIELD_WEIGHTS = {
    "mood":               0.35,
    "color_palette":      0.20,
    "visual_description": 0.45,
}


def build_graph(top_k: int = 10, threshold: float = 0.72) -> dict:
    """
    Build adjacency list graph over all 25k images.

    For each image, finds top_k nearest neighbors per field using FAISS,
    combines into a weighted edge score, keeps edges above threshold.

    Args:
        top_k     : neighbors to retrieve per image per field
        threshold : minimum combined edge weight to create an edge (0-1)

    Returns:
        adjacency dict: {photo_id: [(neighbor_id, edge_weight), ...]}
    """
    print(f"[Graph] Loading field embeddings from {EMBEDDING_NPZ.name}...")
    npz = np.load(EMBEDDING_NPZ)

    print(f"[Graph] Loading dataset from {DATASET_JSON.name}...")
    with open(DATASET_JSON, encoding="utf-8") as f:
        dataset = json.load(f)

    photo_ids = [str(row.get("photo_id", f"idx_{i}")) for i, row in enumerate(dataset)]
    N = len(photo_ids)
    print(f"[Graph] {N} images.")

    try:
        import faiss
        use_faiss = True
        print("[Graph] Using FAISS for fast neighbor search.")
    except ImportError:
        use_faiss = False
        print("[Graph] FAISS not found — using numpy (slower). Run: pip install faiss-cpu")

    # Accumulate edge weights across fields
    # edge_weights[i][j] = combined similarity between image i and image j
    edge_accum = defaultdict(lambda: defaultdict(float))

    for field, field_weight in FIELD_WEIGHTS.items():
        if field not in npz:
            print(f"[Graph]   Field '{field}' not in embeddings — skipping.")
            continue

        emb = npz[field].astype(np.float32)  # (N, D)
        print(f"[Graph]   Field '{field}': shape={emb.shape}, weight={field_weight}")

        if use_faiss:
            # L2-normalize so inner product = cosine similarity
            faiss.normalize_L2(emb)
            index = faiss.IndexFlatIP(emb.shape[1])
            index.add(emb)
            # Search top_k+1 to exclude self (first result)
            similarities, indices = index.search(emb, top_k + 1)
        else:
            # Numpy fallback — memory-intensive but works
            norms = np.linalg.norm(emb, axis=1, keepdims=True)
            norms[norms == 0] = 1
            emb_norm = emb / norms
            sim_matrix = emb_norm @ emb_norm.T  # (N, N)
            # Get top_k+1 per row (includes self at index 0)
            indices   = np.argsort(-sim_matrix, axis=1)[:, :top_k + 1]
            similarities = sim_matrix[np.arange(N)[:, None], indices]

        # Accumulate weighted similarities
        for i in range(N):
            for rank in range(similarities.shape[1]):
                j   = int(indices[i, rank])
                sim = float(similarities[i, rank])
                if j == i or sim <= 0:
                    continue
                edge_accum[i][j] += field_weight * sim

    # Convert to adjacency list, keeping only edges above threshold
    print(f"[Graph] Building adjacency list (threshold={threshold})...")
    adjacency = {}
    n_edges   = 0

    for i, pid in enumerate(photo_ids):
        neighbors = []
        for j, weight in edge_accum[i].items():
            if weight >= threshold:
                neighbors.append((photo_ids[j], round(float(weight), 4)))
        # Sort by weight descending, keep top_k
        neighbors.sort(key=lambda x: x[1], reverse=True)
        adjacency[pid] = neighbors[:top_k]
        n_edges += len(adjacency[pid])

    print(f"[Graph] {n_edges} edges across {N} nodes.")
    print(f"[Graph] Average degree: {n_edges / N:.1f}")

    return adjacency


def main(top_k: int, threshold: float):
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    graph_path = GRAPH_DIR / "image_graph.pkl"
    meta_path  = GRAPH_DIR / "graph_meta.json"

    adjacency = build_graph(top_k=top_k, threshold=threshold)

    print(f"\n[Graph] Saving to {graph_path}...")
    with open(graph_path, "wb") as f:
        pickle.dump(adjacency, f)

    n_edges = sum(len(v) for v in adjacency.values())
    meta = {
        "n_nodes":    len(adjacency),
        "n_edges":    n_edges,
        "avg_degree": round(n_edges / len(adjacency), 2),
        "top_k":      top_k,
        "threshold":  threshold,
        "fields":     list(FIELD_WEIGHTS.keys()),
        "field_weights": FIELD_WEIGHTS,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[Graph] Done.")
    print(f"         nodes    : {meta['n_nodes']}")
    print(f"         edges    : {meta['n_edges']}")
    print(f"         avg deg  : {meta['avg_degree']}")
    print(f"         saved to : {graph_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k",     type=int,   default=10,
                        help="Neighbors per image (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.72,
                        help="Min edge weight to keep (default: 0.72)")
    args = parser.parse_args()
    main(args.top_k, args.threshold)
