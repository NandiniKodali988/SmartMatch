"""
src/tools/score_diagnostic.py
 
Diagnostic tool: given a user query, shows side-by-side results for
SigLIP-only, text-only, and hybrid retrieval at different weight combinations.
 
All three methods score the FULL dataset before ranking, so results are
directly comparable.
 
Usage:
    python src/tools/score_diagnostic.py --query "feeling burnt out after a long week"
    python src/tools/score_diagnostic.py --query "fresh blueberries" --top_k 5
"""
 
import os
import sys
import argparse
import numpy as np
 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
 
from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
 
 
def print_divider(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
 
 
def print_results(results: list, label: str):
    print(f"\n--- {label} ---")
    for i, r in enumerate(results, 1):
        photo_id = r.get("photo_id", "?")
        score    = r.get("score", 0.0)
        caption  = r.get("caption", "")
 
        print(f"\n  {i}. photo_id : {photo_id}")
        print(f"     score    : {score:.4f}", end="")
 
        if "siglip_score" in r and "text_score" in r:
            print(f"  (siglip={r['siglip_score']:.4f}, text={r['text_score']:.4f})", end="")
        if "field_scores" in r:
            fs = r["field_scores"]
            print(f"\n     fields   : vis={fs.get('visual_description',0):.3f}  "
                  f"mood={fs.get('mood',0):.3f}  "
                  f"color={fs.get('color_palette',0):.3f}", end="")
        print()
        print(f"     caption  : {str(caption)[:90]}...")
        url = r.get("image_url", "")
        if url:
            print(f"     url      : {str(url)[:70]}...")
 
 
def hybrid_full(
    siglip_scores: np.ndarray,
    text_scores: np.ndarray,
    dataset: list,
    url_lookup: dict,
    siglip_w: float,
    text_w: float,
    top_k: int,
) -> list:
    """Merge full-dataset siglip and text scores at given weights."""
    final_scores = siglip_w * siglip_scores + text_w * text_scores
    ranked = np.argsort(final_scores)[::-1][:top_k]
 
    results = []
    for idx in ranked:
        row      = dataset[idx]
        photo_id = str(row.get("photo_id", f"idx_{idx}"))
        grounding = row.get("grounding_output", {})
        results.append({
            "photo_id":     photo_id,
            "image_url":    url_lookup.get(photo_id, ""),
            "caption":      grounding.get("visual_description", ""),
            "score":        float(final_scores[idx]),
            "siglip_score": float(siglip_scores[idx]),
            "text_score":   float(text_scores[idx]),
        })
    return results
 
 
def main(query: str, top_k: int):
    # ── Step 1: Visual Grounding ──────────────────────────────────────────────
    print_divider(f'Query: "{query}"')
    print("\n[1/3] Running visual grounding...")
    grounding = QwenVisualGroundingAgent().run(query)
 
    print("\nGrounding output:")
    for k, v in grounding.items():
        print(f"  {k}: {v}")
 
    # ── Step 2: Init agents & score full dataset ──────────────────────────────
    print("\n[2/3] Scoring full dataset with SigLIP...")
    siglip_agent  = SiglipImageRetrievalAgent(top_k=top_k)
    siglip_scores = siglip_agent.retrieve_all_scores(grounding)   # (N,)
 
    print("\n[3/3] Scoring full dataset with field text embeddings...")
    text_agent               = FieldTextRetrievalAgent(top_k=top_k)
    text_scores, field_matrix = text_agent._score_all(grounding)  # (N,)
 
    dataset    = text_agent.dataset
    url_lookup = text_agent.url_lookup
 
    # ── Step 3: SigLIP-only top-k ─────────────────────────────────────────────
    siglip_ranked = np.argsort(siglip_scores)[::-1][:top_k]
    siglip_results = []
    for idx in siglip_ranked:
        row      = dataset[idx]
        photo_id = str(row.get("photo_id", f"idx_{idx}"))
        grounding_row = row.get("grounding_output", {})
        siglip_results.append({
            "photo_id":    photo_id,
            "image_url":   url_lookup.get(photo_id, ""),
            "caption":     grounding_row.get("visual_description", ""),
            "score":       float(siglip_scores[idx]),
            "field_scores": {f: float(field_matrix[f][idx]) for f in field_matrix},
        })
 
    print_divider("SigLIP-only Top Results  (image-text similarity)")
    print_results(siglip_results, f"Top-{top_k}")
 
    # ── Step 4: Text-only top-k ───────────────────────────────────────────────
    text_ranked  = np.argsort(text_scores)[::-1][:top_k]
    text_results = []
    for idx in text_ranked:
        row      = dataset[idx]
        photo_id = str(row.get("photo_id", f"idx_{idx}"))
        grounding_row = row.get("grounding_output", {})
        text_results.append({
            "photo_id":    photo_id,
            "image_url":   url_lookup.get(photo_id, ""),
            "caption":     grounding_row.get("visual_description", ""),
            "score":       float(text_scores[idx]),
            "field_scores": {f: float(field_matrix[f][idx]) for f in field_matrix},
        })
 
    print_divider("Text-only Top Results  (field embedding similarity)")
    print_results(text_results, f"Top-{top_k}")
 
    # ── Step 5: Hybrid at different weights ───────────────────────────────────
    weight_combos = [
        (1.0, 0.0),
        (0.7, 0.3),
        (0.5, 0.5),
        (0.3, 0.7),
        (0.0, 1.0),
    ]
 
    print_divider("Hybrid Results at Different Weight Combinations")
    for siglip_w, text_w in weight_combos:
        merged = hybrid_full(
            siglip_scores=siglip_scores,
            text_scores=text_scores,
            dataset=dataset,
            url_lookup=url_lookup,
            siglip_w=siglip_w,
            text_w=text_w,
            top_k=top_k,
        )
        print_results(merged, f"siglip={siglip_w:.1f}  text={text_w:.1f}")
 
    # ── Summary ───────────────────────────────────────────────────────────────
    print_divider("Score Summary")
    print(f"\n  SigLIP — max={siglip_scores.max():.4f}  "
          f"min={siglip_scores.min():.4f}  "
          f"mean={siglip_scores.mean():.4f}  "
          f"std={siglip_scores.std():.4f}")
    print(f"  Text   — max={text_scores.max():.4f}  "
          f"min={text_scores.min():.4f}  "
          f"mean={text_scores.mean():.4f}  "
          f"std={text_scores.std():.4f}")
    print()
    print("  Decision guide:")
    print("    - Low SigLIP std → SigLIP has poor discrimination → lower siglip weight")
    print("    - text=1.0 top results look most relevant → use siglip=0.2 / text=0.8")
    print("    - Both look good → try siglip=0.4 / text=0.6 as a baseline")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score diagnostic tool")
    parser.add_argument("--query", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    main(args.query, args.top_k)