"""
src/pipeline/run_pipeline.py  (updated with Fallback Generation)

Full pipeline:
    User Text
      → Visual Grounding (Claude)
      → Content Router
      → SigLIP Retrieval (FAISS)
      → [score check] → Fallback Generation (DALL·E 3) if needed
      → Justification (Claude)
      → Return top results
"""

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.content_router.agent import ContentRouterAgent
from agents.fallback_generation.agent import FallbackGenerationAgent


def run_pipeline(user_text: str) -> list:

    # Initialise agents
    qwen_agent          = QwenVisualGroundingAgent()
    justification_agent = QwenJustificationAgent()
    router              = ContentRouterAgent()
    siglip_agent        = SiglipImageRetrievalAgent()
    fallback_agent      = FallbackGenerationAgent()

    # ── Step 1: Visual Grounding ─────────────────────────────────────────────
    # Convert abstract user text → structured visual description
    grounding = qwen_agent.run(user_text)
    print(f"\n[Pipeline] Grounding output: {grounding}\n")

    # ── Step 2: Content Routing ──────────────────────────────────────────────
    route = router.route(grounding)
    print(f"[Pipeline] Route selected: {route}")

    if route != "unsplash":
        # other routes (local_library, ai_compositing, ai_generation) not yet implemented
        print(f"[Pipeline] Route '{route}' not yet implemented. Falling back to unsplash path.")

    # ── Step 3: SigLIP Retrieval ─────────────────────────────────────────────
    retrieved_images = siglip_agent.retrieve(grounding)
    print(f"[Pipeline] Retrieved {len(retrieved_images)} image(s). "
          f"Top score: {retrieved_images[0]['score']:.4f if retrieved_images else 'N/A'}")

    # ── Step 4: Fallback Generation (DALL·E 3) ───────────────────────────────
    # Merge retrieved + generated candidates, then re-rank by SigLIP score
    if fallback_agent.needs_fallback(retrieved_images):
        generated_images = fallback_agent.run(grounding)

        # Re-score generated images using SigLIP so they are comparable
        generated_images = _rescore_with_siglip(siglip_agent, grounding, generated_images)

        # Combine and re-rank all candidates by score descending
        all_candidates = retrieved_images + generated_images
        all_candidates.sort(key=lambda x: x["score"], reverse=True)
        print(f"[Pipeline] After merge & re-rank: {len(all_candidates)} candidate(s).")
    else:
        all_candidates = retrieved_images

    # Take top-3 for justification
    top_candidates = all_candidates[:3]

    # ── Step 5: Justification ────────────────────────────────────────────────
    results = justification_agent.run(user_text, top_candidates)

    return results


def _rescore_with_siglip(
    siglip_agent: SiglipImageRetrievalAgent,
    grounding: dict,
    images: list,
) -> list:
    """
    Re-scores generated images using SigLIP text–image similarity.

    For generated images we only have a text prompt (stored in 'caption'),
    so we compare the grounding query embedding against the caption embedding
    as a proxy — this is a text-text similarity, not true image similarity.

    TODO: Once DALL·E URLs are fetched and images are downloaded,
          replace with actual image embedding comparison.
    """
    query = siglip_agent.build_query(grounding)
    query_embedding = siglip_agent.embed_text(query)

    import numpy as np

    rescored = []
    for img in images:
        caption = img.get("caption", "")
        try:
            caption_embedding = siglip_agent.embed_text(caption)
            score = float(np.dot(query_embedding, caption_embedding))
        except Exception as e:
            print(f"[Pipeline] Warning: re-scoring failed for {img['photo_id']}: {e}")
            score = 0.0
        rescored.append({**img, "score": score})

    return rescored


if __name__ == "__main__":
    query = "Walking alone through a rainy city street at night"
    print(f"Running pipeline for: '{query}'\n")

    results = run_pipeline(query)

    print("\n=== Final Results ===")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.get('source', 'unsplash')}] {r['photo_id']}")
        print(f"   Score:         {r['score']:.4f}")
        print(f"   Caption:       {r['caption'][:80]}...")
        print(f"   Justification: {r.get('justification', '')[:120]}...")
        print(f"   URL:           {r['image_url'][:80]}...")
