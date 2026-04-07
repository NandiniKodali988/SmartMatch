"""
src/data/data_prep/generate_field_embeddings.py

Embeds visual_description / mood / color_palette fields separately,
then computes per-field cosine similarity between a query grounding output
and each record in the dataset grounding output.

final_score = weighted sum of per-field similarity scores (equal weights by default)

Inputs:
  data/processed/description_grounding_outputs.json  -- dataset expanded grounding outputs
  data/processed/sample_grounding_outputs.json       -- query input (or any grounding output JSON)

Outputs:
  data/embeddings/field_embeddings.npz  -- per-field embedding matrices (cached)
  (top-K results printed to stdout)

Usage:
  python src/data/data_prep/generate_field_embeddings.py
  python src/data/data_prep/generate_field_embeddings.py --query path/to/query.json --top_k 5
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[3]
DATASET_JSON    = BASE_DIR / "src/data/processed/description_grounding_outputs.json"
QUERY_JSON      = BASE_DIR / "src/data/processed/sample_grounding_outputs.json"
EMBEDDING_NPZ   = BASE_DIR / "src/data/embeddings/field_embeddings.npz"

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
BATCH_SIZE      = int(os.getenv("BATCH_SIZE", "100"))

# Fields to compare and their weights (must sum to 1.0)
FIELDS = {
    "visual_description": 1/3,
    "mood":               1/3,
    "color_palette":      1/3,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    # Support both a single dict and a list of dicts
    if isinstance(data, dict):
        data = [data]
    return data


def extract_field(grounding: dict, field: str) -> str:
    """Return the string value of a field from a grounding dict, or '' if missing."""
    val = grounding.get(field, "") or ""
    return str(val).strip()


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize each row so dot product equals cosine similarity."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return matrix / norms


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_texts(texts: list[str], client: OpenAI) -> np.ndarray:
    """
    Embed a list of strings in batches using OpenAI.
    Returns an L2-normalized (N, 3072) float32 matrix.
    Empty strings are replaced with a placeholder to avoid API errors.
    """
    cleaned = [t if t.strip() else "no description" for t in texts]
    all_embeddings = []

    for i in range(0, len(cleaned), BATCH_SIZE):
        batch = cleaned[i : i + BATCH_SIZE]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in resp.data])

    matrix = np.array(all_embeddings, dtype=np.float32)
    return normalize(matrix)


# ── Build / load dataset field embeddings ─────────────────────────────────────

def build_dataset_embeddings(dataset: list, client: OpenAI) -> dict[str, np.ndarray]:
    """
    Embed each field separately for all dataset records.

    Returns:
        {
            "visual_description": ndarray (N, 3072),
            "mood":               ndarray (N, 3072),
            "color_palette":      ndarray (N, 3072),
        }
    """
    field_embeddings = {}

    for field in FIELDS:
        texts = [extract_field(row["grounding_output"], field) for row in dataset]
        print(f"[Embed] Field '{field}': {len(texts)} texts...")
        field_embeddings[field] = embed_texts(texts, client)

    return field_embeddings


def load_or_build_dataset_embeddings(dataset: list, client: OpenAI) -> dict[str, np.ndarray]:
    """
    Load cached field embeddings from .npz if available, otherwise build and cache them.
    Caching avoids redundant API calls on subsequent runs.
    """
    if EMBEDDING_NPZ.exists():
        print(f"[Cache] Loading field embeddings from {EMBEDDING_NPZ.name}...")
        npz = np.load(EMBEDDING_NPZ)
        return {field: npz[field] for field in FIELDS}

    print("[Cache] No cache found — generating field embeddings...")
    field_embeddings = build_dataset_embeddings(dataset, client)

    EMBEDDING_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(EMBEDDING_NPZ, **field_embeddings)
    print(f"[Cache] Saved to {EMBEDDING_NPZ.name}")

    return field_embeddings


# ── Query & Score ─────────────────────────────────────────────────────────────

def score_query(
    query_grounding: dict,
    dataset_field_embeddings: dict[str, np.ndarray],
    client: OpenAI,
) -> np.ndarray:
    """
    Compute a weighted similarity score between a single query grounding output
    and every record in the dataset.

    For each field:
      - Embed the query field text
      - Compute cosine similarity against all dataset records for that field
      - Multiply by the field's weight and accumulate

    Fields missing from the query are skipped without penalizing the score.

    Returns:
        scores (N,) -- final weighted score for each dataset record
    """
    N = next(iter(dataset_field_embeddings.values())).shape[0]
    final_scores = np.zeros(N, dtype=np.float32)

    for field, weight in FIELDS.items():
        query_text = extract_field(query_grounding, field)

        # Skip fields absent from the query rather than penalizing
        if not query_text:
            print(f"  [Score] Field '{field}': missing in query — skipped.")
            continue

        # Embed the single query field text -> shape (3072,)
        q_emb = embed_texts([query_text], client)[0]

        # Cosine similarity = dot product (both sides already L2-normalized)
        field_scores = dataset_field_embeddings[field] @ q_emb  # (N,)
        final_scores += weight * field_scores

        print(f"  [Score] Field '{field}' (weight={weight:.2f}): mean={float(field_scores.mean()):.4f}")

    return final_scores


def retrieve(
    query_grounding: dict,
    dataset: list,
    dataset_field_embeddings: dict[str, np.ndarray],
    client: OpenAI,
    top_k: int = 5,
) -> list[dict]:
    """
    Return the top-k most similar dataset records for a given query grounding output.

    Each result dict contains:
        photo_id    -- identifier from the dataset record
        final_score -- weighted sum of per-field cosine similarities
        grounding   -- the matched record's grounding_output dict
    """
    scores = score_query(query_grounding, dataset_field_embeddings, client)
    ranked = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in ranked:
        row = dataset[idx]
        results.append({
            "photo_id":    row.get("photo_id", f"idx_{idx}"),
            "final_score": float(scores[idx]),
            "grounding":   row["grounding_output"],
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main(query_path: Path, top_k: int):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Load dataset
    print(f"[Load] Dataset: {DATASET_JSON.name}")
    dataset = load_json(DATASET_JSON)
    print(f"[Load] {len(dataset)} records.")

    # Load or build dataset field embeddings (cached after first run)
    dataset_field_embeddings = load_or_build_dataset_embeddings(dataset, client)

    # Load queries
    print(f"\n[Load] Query file: {query_path.name}")
    queries = load_json(query_path)
    print(f"[Load] {len(queries)} query/queries.")

    # Run retrieval for each query
    for i, q in enumerate(queries):
        input_text      = q.get("input_text", f"query_{i}")
        # Accept both {"grounding_output": {...}} and a bare grounding dict
        query_grounding = q.get("grounding_output", q)

        print(f"\n{'='*60}")
        print(f"Query {i+1}: \"{input_text}\"")
        print(f"{'='*60}")

        results = retrieve(
            query_grounding=query_grounding,
            dataset=dataset,
            dataset_field_embeddings=dataset_field_embeddings,
            client=client,
            top_k=top_k,
        )

        print(f"\nTop-{top_k} Results:")
        for rank, r in enumerate(results, 1):
            print(f"\n  {rank}. photo_id:    {r['photo_id']}")
            print(f"     final_score: {r['final_score']:.4f}")
            print(f"     visual:      {r['grounding']['visual_description'][:80]}...")
            print(f"     mood:        {r['grounding'].get('mood', '')}")
            print(f"     color:       {r['grounding'].get('color_palette', '')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Field-level embedding retrieval")
    parser.add_argument(
        "--query",
        type=str,
        default=str(QUERY_JSON),
        help="Path to query grounding output JSON (default: sample_grounding_outputs.json)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top results to return per query (default: 5)",
    )
    args = parser.parse_args()

    main(Path(args.query), args.top_k)
