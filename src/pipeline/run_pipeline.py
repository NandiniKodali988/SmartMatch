"""
src/pipeline/run_pipeline.py

Full pipeline:

  User Text + [optional uploaded images]
    │
    ├─ Has uploaded images
    │    → Visual Grounding
    │    → GenerationAgent (Claude picks editing mode from full grounding output)
    │    → Justification → return
    │
    └─ No uploaded images
         → Visual Grounding
         → Hybrid Retrieval: SigLIP (0.7) + FieldText (0.3) over full 25k dataset
         → top-1 hybrid score >= HYBRID_THRESHOLD?
             ├─ YES → Justification → return retrieval results
             └─ NO  → GenerationAgent (DALL·E 3, using full grounding output)
                    → Justification → return

Environment variables (.env):
    HYBRID_THRESHOLD   — score cutoff for retrieval vs generation (default 0.5)
    SIGLIP_WEIGHT      — weight for SigLIP image score (default 0.7)
    TEXT_WEIGHT        — weight for field text score (default 0.3)
    TOP_K              — number of results to return (default 3)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
from agents.generation.agent import GenerationAgent

# ── Config ────────────────────────────────────────────────────────────────────
HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
SIGLIP_WEIGHT    = float(os.getenv("SIGLIP_WEIGHT",    "0.7"))
TEXT_WEIGHT      = float(os.getenv("TEXT_WEIGHT",      "0.3"))
TOP_K            = int(os.getenv("TOP_K", "3"))


def run_pipeline(
    user_text: str,
    uploaded_image_paths: list[str] | None = None,
    style_ref_path: str | None = None,
) -> list[dict]:
    """
    Args:
        user_text             : raw text input from the user
        uploaded_image_paths  : local image file paths uploaded by the user (optional)
        style_ref_path        : style reference image for composite editing mode (optional)

    Returns:
        list of result dicts (up to TOP_K), each containing:
            photo_id, image_url, caption, score, source, justification
    """
    print("\n" + "=" * 60)
    print("[Pipeline] START")
    print(f"[Pipeline] user_text       : {user_text}")
    print(f"[Pipeline] uploaded_images : {uploaded_image_paths}")
    print(f"[Pipeline] style_ref_path  : {style_ref_path}")
    print("=" * 60)

    # ── Init agents ───────────────────────────────────────────────────────────
    grounding_agent     = QwenVisualGroundingAgent()
    justification_agent = QwenJustificationAgent()
    siglip_agent        = SiglipImageRetrievalAgent(top_k=TOP_K)
    text_agent          = FieldTextRetrievalAgent(top_k=TOP_K)
    generation_agent    = GenerationAgent()

    # ── Step 1: Visual Grounding ──────────────────────────────────────────────
    # Converts abstract user text into structured visual concepts.
    # Full grounding output (including intent, lighting, color_palette) is
    # passed downstream to all agents that need it.
    print("\n[Step 1] Visual Grounding...")
    grounding = grounding_agent.run(user_text)
    print(f"         visual_description : {grounding.get('visual_description', '')[:80]}...")
    print(f"         intent             : {grounding.get('intent', '')}")
    print(f"         mood               : {grounding.get('mood', '')}")
    print(f"         color_palette      : {grounding.get('color_palette', '')}")

    # ── Branch A: user uploaded images → image editing ────────────────────────
    if uploaded_image_paths:
        print("\n[Pipeline] Uploaded images detected — skipping retrieval.")
        print("\n[Step 2] GenerationAgent (image editing)...")
        images = generation_agent.run(
            grounding_output=grounding,
            user_text=user_text,
            uploaded_image_paths=uploaded_image_paths,
            style_ref_path=style_ref_path,
            siglip_agent=siglip_agent,
        )
        print(f"[Step 2] {len(images)} image(s) returned.")

        print(f"\n[Step 3] Justification...")
        results = justification_agent.run(user_text, images)

        _log_done(results)
        return results

    # ── Branch B: no uploads → hybrid retrieval, fallback to generation ───────
    print("\n[Pipeline] No uploads — running hybrid retrieval.")

    # Step 2: Hybrid retrieval over full 25k dataset
    # SigLIP scores visual similarity (image embeddings vs text query)
    # FieldText scores semantic similarity (field embeddings vs grounding fields)
    # final_score = SIGLIP_WEIGHT * siglip_score + TEXT_WEIGHT * text_score
    print(f"\n[Step 2] Hybrid Retrieval (siglip={SIGLIP_WEIGHT}, text={TEXT_WEIGHT})...")
    candidates = text_agent.merge_with_siglip(
        grounding_output=grounding,
        siglip_agent=siglip_agent,
    )

    top_score = candidates[0]["score"] if candidates else 0.0
    print(f"[Step 2] Top hybrid score: {top_score:.4f}  (threshold={HYBRID_THRESHOLD})")
    for i, c in enumerate(candidates[:TOP_K]):
        print(
            f"         [{i+1}] {c['photo_id']}  "
            f"hybrid={c['score']:.4f}  "
            f"siglip={c.get('siglip_score', 0):.4f}  "
            f"text={c.get('text_score', 0):.4f}"
        )

    # Step 3: Threshold check — retrieval or generation
    if top_score >= HYBRID_THRESHOLD:
        print(f"\n[Step 3] Score sufficient — using retrieval results.")
        images = candidates[:TOP_K]
    else:
        print(f"\n[Step 3] Score below threshold — falling back to GenerationAgent (DALL·E 3).")
        images = generation_agent.run(
            grounding_output=grounding,   # full grounding including intent, lighting, etc.
            n=TOP_K,
            user_text=user_text,
            uploaded_image_paths=None,
            style_ref_path=None,
            siglip_agent=siglip_agent,
        )
        print(f"[Step 3] {len(images)} image(s) generated.")
        for i, img in enumerate(images):
            print(
                f"         [{i+1}] {img.get('photo_id')}  "
                f"source={img.get('source')}  "
                f"score={img.get('score', 0):.4f}"
            )

    # Step 4: Justification
    print(f"\n[Step 4] Justification ({len(images)} image(s))...")
    results = justification_agent.run(user_text, images)

    _log_done(results)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_done(results: list):
    print(f"\n[Pipeline] DONE — {len(results)} result(s)")
    if results:
        print(f"           Sample justification: {results[0].get('justification', '')[:100]}...")
    print("=" * 60 + "\n")


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartMatch pipeline")
    parser.add_argument(
        "--query",
        type=str,
        default="Walking alone through a rainy city street at night",
        help="User text query",
    )
    parser.add_argument(
        "--images",
        type=str,
        nargs="*",
        help="Uploaded image paths (optional)",
    )
    parser.add_argument(
        "--style",
        type=str,
        default=None,
        help="Style reference image path for composite mode (optional)",
    )
    args = parser.parse_args()

    results = run_pipeline(
        user_text=args.query,
        uploaded_image_paths=args.images,
        style_ref_path=args.style,
    )

    print("\n=== Final Results ===")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.get('source', '?')}] {r.get('photo_id')}")
        print(f"   score         : {r.get('score', 0):.4f}")
        print(f"   url           : {r.get('image_url', '')[:70]}...")
        print(f"   justification : {r.get('justification', '')[:120]}...")
