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
         → Hybrid Retrieval: SigLIP + FieldText over full 25k dataset (CANDIDATE_K results)
         → top-1 hybrid score >= HYBRID_THRESHOLD?
             ├─ YES → Multimodal Verification (Claude Vision checks each image)
             │          → enough verified (>= MIN_VERIFIED)?
             │              ├─ YES → Coherence Agent (selects final BOARD_SIZE as a set)
             │              └─ NO  → GenerationAgent fills gaps → Coherence Agent
             └─ NO  → GenerationAgent (DALL·E 3, full generation) → Coherence Agent
         → Justification → return

Environment variables (.env):
    HYBRID_THRESHOLD   — score cutoff for retrieval vs generation (default 0.5)
    SIGLIP_WEIGHT      — weight for SigLIP image score (default 0.7)
    TEXT_WEIGHT        — weight for field text score (default 0.3)
    CANDIDATE_K        — how many candidates to retrieve before verification (default 20)
    BOARD_SIZE         — target number of images in the final mood board (default 9)
    MIN_VERIFIED       — minimum verified images before triggering generation fallback (default 6)
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
from agents.generation.agent import GenerationAgent
from agents.multimodal_verification.agent import MultimodalVerificationAgent
from src.agents.memory.memory_manager import MemoryManager
from agents.coherence.agent import CoherenceAgent

# ── Config ────────────────────────────────────────────────────────────────────
HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
SIGLIP_WEIGHT    = float(os.getenv("SIGLIP_WEIGHT",    "0.7"))
TEXT_WEIGHT      = float(os.getenv("TEXT_WEIGHT",      "0.3"))
CANDIDATE_K      = int(os.getenv("CANDIDATE_K", "20"))
BOARD_SIZE       = int(os.getenv("BOARD_SIZE",  "9"))
MIN_VERIFIED     = int(os.getenv("MIN_VERIFIED", "6"))

# Keep TOP_K for backward compatibility — used only in the uploaded-images branch
TOP_K = int(os.getenv("TOP_K", "3"))


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
    grounding_agent      = QwenVisualGroundingAgent()
    justification_agent  = QwenJustificationAgent()
    siglip_agent         = SiglipImageRetrievalAgent(top_k=CANDIDATE_K)
    text_agent           = FieldTextRetrievalAgent(top_k=CANDIDATE_K)
    generation_agent     = GenerationAgent()
    verification_agent   = MultimodalVerificationAgent(min_verified=MIN_VERIFIED)
    coherence_agent      = CoherenceAgent(target_count=BOARD_SIZE)

    # ── Step 1: Visual Grounding ──────────────────────────────────────────────
    # Converts abstract user text into structured visual concepts.
    # Full grounding output (including intent, lighting, color_palette) is
    # passed downstream to all agents that need it.
    print("\n[Step 1] Visual Grounding...")
    # grounding = grounding_agent.run(user_text)
    
    previous_grounding = memory.get_last()

    grounding = grounding_agent.run(
        user_text,
        previous_grounding=previous_grounding
    )
    memory.add(user_text, grounding)
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

    # ── Branch B: no uploads → hybrid retrieval → verification → coherence ──────
    print("\n[Pipeline] No uploads — running hybrid retrieval.")

    # Step 2: Hybrid retrieval — fetch CANDIDATE_K images to give verification
    # and coherence enough to work with
    print(f"\n[Step 2] Hybrid Retrieval (siglip={SIGLIP_WEIGHT}, text={TEXT_WEIGHT}, k={CANDIDATE_K})...")
    candidates = text_agent.merge_with_siglip(
        grounding_output=grounding,
        siglip_agent=siglip_agent,
    )

    top_score = candidates[0]["score"] if candidates else 0.0
    print(f"[Step 2] Top hybrid score: {top_score:.4f}  (threshold={HYBRID_THRESHOLD})")
    for i, c in enumerate(candidates[:5]):
        print(
            f"         [{i+1}] {c['photo_id']}  "
            f"hybrid={c['score']:.4f}  "
            f"siglip={c.get('siglip_score', 0):.4f}  "
            f"text={c.get('text_score', 0):.4f}"
        )

    # Step 3: Route — if score too low skip straight to generation
    if top_score < HYBRID_THRESHOLD:
        print(f"\n[Step 3] Score below threshold — generating full board with DALL·E 3.")
        images = generation_agent.run(
            grounding_output=grounding,
            n=BOARD_SIZE,
            user_text=user_text,
            uploaded_image_paths=None,
            style_ref_path=None,
            siglip_agent=siglip_agent,
        )
        print(f"[Step 3] {len(images)} image(s) generated.")
    else:
        # Step 3a: Verify retrieved candidates with Claude Vision
        print(f"\n[Step 3] Score sufficient — running Multimodal Verification...")
        verified_candidates = verification_agent.run(candidates, grounding)

        # Step 3b: If not enough verified, generation fills the gap
        if verification_agent.needs_generation(verified_candidates):
            n_needed = BOARD_SIZE - sum(1 for c in verified_candidates if c["verified"])
            print(f"\n[Step 3b] Only {sum(1 for c in verified_candidates if c['verified'])} verified — "
                  f"generating {n_needed} more with DALL·E 3.")
            generated = generation_agent.run(
                grounding_output=grounding,
                n=n_needed,
                user_text=user_text,
                uploaded_image_paths=None,
                style_ref_path=None,
                siglip_agent=siglip_agent,
            )
            pool = verified_candidates + generated
        else:
            pool = verified_candidates

        # Step 4: Coherence — pick the final board as a set
        print(f"\n[Step 4] Coherence Agent (selecting {BOARD_SIZE} from {len(pool)} candidates)...")
        images = coherence_agent.run(pool, grounding)
        print(f"[Step 4] {len(images)} image(s) selected for the board.")

    # Step 5: Justification
    print(f"\n[Step 5] Justification ({len(images)} image(s))...")
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
    memory = MemoryManager()

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
