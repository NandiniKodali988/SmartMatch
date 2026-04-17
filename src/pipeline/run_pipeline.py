"""
src/pipeline/run_pipeline.py

Full pipeline:

  User Text + [optional uploaded images]
    │
    ├─ Has uploaded images
    │    → Visual Grounding
    │    → GenerationAgent (Claude picks editing mode from full grounding output)
    │    → Justification (per-image + board summary)
    │    → MoodBoard Layout (gpt-image-1.5 composite)
    │    → return
    │
    └─ No uploaded images
         → Visual Grounding
         → Hybrid Retrieval: SigLIP (0.7) + FieldText (0.3), CANDIDATE_K candidates
         → top-1 hybrid score >= HYBRID_THRESHOLD?
             ├─ YES → Multimodal Verification (Claude Vision)
             │          → verified >= BOARD_SIZE?
             │              ├─ YES → Coherence Agent
             │              └─ NO  → GenerationAgent fills gap → Coherence Agent
             └─ NO  → GenerationAgent (DALL·E 3 full board) → Coherence Agent
         → Justification (per-image + board summary)
         → MoodBoard Layout (gpt-image-1.5 composite)
         → return

Environment variables (.env):
    HYBRID_THRESHOLD — min hybrid score to use retrieval path (default 0.5)
    SIGLIP_WEIGHT    — weight for SigLIP image score (default 0.7)
    TEXT_WEIGHT      — weight for field text score (default 0.3)
    CANDIDATE_K      — candidates fetched from hybrid retrieval (default 20)
    BOARD_SIZE       — target images in the final mood board (default 9)
    MIN_VERIFIED     — min verified images before generation fallback (default 6)
    SKIP_LAYOUT      — set to "1" to skip layout stitching (faster for testing)
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
from agents.coherence.agent import CoherenceAgent
from agents.moodboard_layout.agent import MoodBoardLayoutAgent

# ── Config ────────────────────────────────────────────────────────────────────
HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
SIGLIP_WEIGHT    = float(os.getenv("SIGLIP_WEIGHT",    "0.7"))
TEXT_WEIGHT      = float(os.getenv("TEXT_WEIGHT",      "0.3"))
CANDIDATE_K      = int(os.getenv("CANDIDATE_K", "20"))
BOARD_SIZE       = int(os.getenv("BOARD_SIZE",  "9"))
MIN_VERIFIED     = int(os.getenv("MIN_VERIFIED", "6"))
SKIP_LAYOUT      = os.getenv("SKIP_LAYOUT", "0") == "1"
TOP_K            = int(os.getenv("TOP_K", "3"))   # backward compat


def run_pipeline(
    user_text: str,
    uploaded_image_paths: list[str] | None = None,
    style_ref_path: str | None = None,
) -> dict:
    """
    Args:
        user_text             : raw text input from the user
        uploaded_image_paths  : local image file paths uploaded by the user (optional)
        style_ref_path        : style reference image for composite editing mode (optional)

    Returns:
        dict with:
            images        (list) : result dicts, each with photo_id, image_url,
                                   caption, score, source, verified,
                                   verification_reason, coherence_rank, justification
            board_summary (str)  : paragraph explaining the board as a whole
            board_path    (str)  : local path to the stitched mood board PNG
                                   (empty string if SKIP_LAYOUT=1)
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
    siglip_agent        = SiglipImageRetrievalAgent(top_k=CANDIDATE_K)
    text_agent          = FieldTextRetrievalAgent(top_k=CANDIDATE_K)
    generation_agent    = GenerationAgent()
    verification_agent  = MultimodalVerificationAgent(min_verified=MIN_VERIFIED)
    coherence_agent     = CoherenceAgent(target_count=BOARD_SIZE)
    layout_agent        = MoodBoardLayoutAgent()

    # ── Step 1: Visual Grounding ──────────────────────────────────────────────
    print("\n[Step 1] Visual Grounding...")
    grounding = grounding_agent.run(user_text)
    print(f"         visual_description : {grounding.get('visual_description', '')[:80]}...")
    print(f"         intent             : {grounding.get('intent', '')}")
    print(f"         mood               : {grounding.get('mood', '')}")
    print(f"         color_palette      : {grounding.get('color_palette', '')}")

    # ── Branch A: uploaded images → editing ───────────────────────────────────
    if uploaded_image_paths:
        print("\n[Pipeline] Uploaded images — routing to GenerationAgent (editing).")

        print("\n[Step 2] GenerationAgent (image editing)...")
        images = generation_agent.run(
            grounding_output=grounding,
            user_text=user_text,
            uploaded_image_paths=uploaded_image_paths,
            style_ref_path=style_ref_path,
            siglip_agent=siglip_agent,
        )
        print(f"[Step 2] {len(images)} image(s) returned.")

        print("\n[Step 3] Justification...")
        results, board_summary = justification_agent.run_with_board_summary(
            user_text, images,
        )

        board_path = _build_layout(layout_agent, results, grounding)
        return _package(results, board_summary, board_path)

    # ── Branch B: no uploads → retrieval → verification → coherence ──────────
    print("\n[Pipeline] No uploads — running hybrid retrieval.")

    # Step 2: Hybrid retrieval
    print(f"\n[Step 2] Hybrid Retrieval "
          f"(siglip={SIGLIP_WEIGHT}, text={TEXT_WEIGHT}, k={CANDIDATE_K})...")
    candidates = text_agent.merge_with_siglip(
        grounding_output=grounding,
        siglip_agent=siglip_agent,
    )

    top_score = candidates[0]["score"] if candidates else 0.0
    print(f"[Step 2] {len(candidates)} candidates. "
          f"Top score: {top_score:.4f} (threshold={HYBRID_THRESHOLD})")
    for i, c in enumerate(candidates[:5]):
        print(f"         [{i+1}] {c['photo_id']}  "
              f"hybrid={c['score']:.4f}  "
              f"siglip={c.get('siglip_score', 0):.4f}  "
              f"text={c.get('text_score', 0):.4f}")

    # Step 3: Route on score
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
        # Step 3a: Multimodal Verification
        print(f"\n[Step 3] Multimodal Verification ({len(candidates)} candidates)...")
        verified_candidates = verification_agent.run(candidates, grounding)
        n_verified = sum(1 for c in verified_candidates if c["verified"])
        print(f"[Step 3] {n_verified}/{len(candidates)} passed verification.")

        # Step 3b: Fill gap with generation if not enough verified images
        if n_verified < BOARD_SIZE:
            n_needed = BOARD_SIZE - n_verified
            print(f"\n[Step 3b] {n_verified} verified < {BOARD_SIZE} needed — "
                  f"generating {n_needed} more with DALL·E 3.")
            generated = generation_agent.run(
                grounding_output=grounding,
                n=n_needed,
                user_text=user_text,
                uploaded_image_paths=None,
                style_ref_path=None,
                siglip_agent=siglip_agent,
            )
            print(f"[Step 3b] Generated {len(generated)} image(s).")
            pool = verified_candidates + generated
        else:
            pool = verified_candidates

        # Step 4: Coherence
        print(f"\n[Step 4] Coherence Agent "
              f"(selecting {BOARD_SIZE} from {len(pool)} candidates)...")
        images = coherence_agent.run(pool, grounding)
        print(f"[Step 4] {len(images)} image(s) selected for the board.")

    # Step 5: Justification (per-image + board summary)
    print(f"\n[Step 5] Justification ({len(images)} image(s))...")
    results, board_summary = justification_agent.run_with_board_summary(
        user_text, images,
    )

    # Step 6: Mood Board Layout (gpt-image-1.5 composite)
    board_path = _build_layout(layout_agent, results, grounding)

    return _package(results, board_summary, board_path)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_layout(
    layout_agent: MoodBoardLayoutAgent,
    results: list,
    grounding: dict,
) -> str:
    """Run layout agent unless SKIP_LAYOUT=1. Passes grounding for aesthetic prompt."""
    if SKIP_LAYOUT:
        print("\n[Step 6] Layout skipped (SKIP_LAYOUT=1).")
        return ""
    print("\n[Step 6] Building mood board layout (gpt-image-1.5)...")
    try:
        return layout_agent.run(results, grounding_output=grounding)
    except Exception as e:
        print(f"[Step 6] Layout failed: {e}")
        return ""


def _package(images: list, board_summary: str, board_path: str) -> dict:
    print(f"\n[Pipeline] DONE — {len(images)} image(s)")
    print(f"           Board summary : {board_summary[:100]}...")
    if board_path:
        print(f"           Board path    : {board_path}")
    print("=" * 60 + "\n")
    return {
        "images":        images,
        "board_summary": board_summary,
        "board_path":    board_path,
    }


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartMatch mood board pipeline")
    parser.add_argument("--query", type=str,
                        default="Walking alone through a rainy city street at night")
    parser.add_argument("--images", type=str, nargs="*",
                        help="Uploaded image paths (optional)")
    parser.add_argument("--style",  type=str, default=None,
                        help="Style reference image (optional)")
    parser.add_argument("--skip-layout", action="store_true",
                        help="Skip layout stitching (faster for testing)")
    args = parser.parse_args()

    if args.skip_layout:
        os.environ["SKIP_LAYOUT"] = "1"

    output = run_pipeline(
        user_text=args.query,
        uploaded_image_paths=args.images,
        style_ref_path=args.style,
    )

    print("\n=== Final Results ===")
    for i, r in enumerate(output["images"], 1):
        print(f"\n{i}. [{r.get('source', '?')}] {r.get('photo_id')}")
        print(f"   score         : {r.get('score', 0):.4f}")
        print(f"   coherence_rank: {r.get('coherence_rank', '-')}")
        print(f"   verified      : {r.get('verified', '-')}")
        print(f"   justification : {r.get('justification', '')[:100]}...")
        print(f"   url           : {r.get('image_url', '')[:70]}...")

    print(f"\nBoard summary : {output['board_summary'][:200]}...")
    if output["board_path"]:
        print(f"Board PNG     : {output['board_path']}")