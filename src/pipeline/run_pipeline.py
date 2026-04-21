"""
src/pipeline/run_pipeline.py

Full pipeline:

  User Text + [optional uploaded images]
    │
    ├─ Has uploaded images
    │    → Visual Grounding → GenerationAgent → Justification → return
    │
    └─ No uploaded images
         → Visual Grounding
         → Hybrid Retrieval (CANDIDATE_K candidates)
         → Graph RAG:
             1. deduplicate  — remove near-identical images (concrete query problem)
             2. expand       — add graph neighbors of top seeds (abstract query problem)
             3. rerank       — boost coherence-connected candidates
         → score >= HYBRID_THRESHOLD?
             ├─ YES → Multimodal Verification
             │          → verified >= BOARD_SIZE? → Coherence → Justification
             │          → not enough → GenerationAgent fills gap → Coherence → Justification
             └─ NO  → GenerationAgent (full DALL·E 3 board) → Justification
         → return

Environment variables (.env):
    HYBRID_THRESHOLD  — min score for retrieval path (default 0.5)
    SIGLIP_WEIGHT     — SigLIP weight in hybrid score (default 0.7)
    TEXT_WEIGHT       — text embedding weight (default 0.3)
    CANDIDATE_K       — candidates from hybrid retrieval (default 20)
    BOARD_SIZE        — final board image count (default 9)
    MIN_VERIFIED      — min verified before generation fallback (default 6)
    GRAPH_WEIGHT      — graph score weight in reranking (default 0.25)
    DEDUP_THRESHOLD   — edge weight for near-duplicate detection (default 0.85)
    EXPAND_SEEDS      — top-K seeds for graph expansion (default 5)
    EXPAND_MAX_ADD    — max new candidates from expansion (default 10)
    SKIP_LAYOUT       — set to "1" to skip layout stitching
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
from agents.graph_rag.agent import GraphRAGAgent
from agents.memory.memory_manager import MemoryManager

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
    memory: MemoryManager | None = None,
) -> list[dict]:
    """
    Args:
        user_text             : raw text input from the user
        uploaded_image_paths  : local image file paths (optional)
        style_ref_path        : style reference for composite mode (optional)
        memory                : MemoryManager for multi-turn context (optional)

    Returns:
        list of result dicts, each with:
            photo_id, image_url, caption, score, source,
            graph_score, verified, verification_reason,
            coherence_rank, justification
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

    # Graph RAG — optional, skip gracefully if graph not built yet
    try:
        graph_agent = GraphRAGAgent()
        use_graph   = True
    except FileNotFoundError as e:
        print(f"[Pipeline] Graph RAG disabled: {e}")
        use_graph = False

    # ── Step 1: Visual Grounding ──────────────────────────────────────────────
    print("\n[Step 1] Visual Grounding...")
    previous_grounding = memory.get_last() if memory else None
    grounding = grounding_agent.run(user_text, previous_grounding=previous_grounding)
    if memory:
        memory.add(user_text, grounding)

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

        print(f"\n[Step 3] Justification...")
        results = justification_agent.run(user_text, images)
        _log_done(results)
        return results

    # ── Branch B: no uploads → retrieval + graph RAG + verification ───────────
    print("\n[Pipeline] No uploads — running hybrid retrieval.")

    # Step 2: Hybrid retrieval
    candidates = text_agent.merge_with_siglip(
        grounding_output=grounding,
        siglip_agent=siglip_agent,
    )

    top_hybrid_score = candidates[0]["score"] if candidates else 0.0  

    # Step 2b: Graph RAG
    if use_graph:
        candidates = graph_agent.deduplicate(candidates)
        candidates = graph_agent.expand(candidates, text_agent, grounding_output=grounding) 
        candidates = graph_agent.rerank(candidates, grounding)

    top_score = candidates[0]["score"] if candidates else 0.0  
    print(f"[Step 2] Top hybrid score: {top_hybrid_score:.4f} (threshold={HYBRID_THRESHOLD})")
    print(f"[Step 2] Top score after Graph RAG: {top_score:.4f}")


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

        # Step 3b: Fill gap with generation if not enough verified
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
            pool = verified_candidates + generated
        else:
            pool = verified_candidates

        # Step 4: Coherence
        print(f"\n[Step 4] Coherence Agent "
              f"(selecting {BOARD_SIZE} from {min(len(pool), 20)} candidates)...")
        images = coherence_agent.run(pool[:20], grounding)
        print(f"[Step 4] {len(images)} image(s) selected.")

    # Step 5: Justification
    print(f"\n[Step 5] Justification ({len(images)} image(s))...")
    results = justification_agent.run(user_text, images)

    _log_done(results)
    return results


# ── Helpers ───────────────────────────────────────────────────────────────────

def _log_done(results: list):
    print(f"\n[Pipeline] DONE — {len(results)} result(s)")
    if results:
        print(f"           Sample justification: "
              f"{results[0].get('justification', '')[:100]}...")
    print("=" * 60 + "\n")


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SmartMatch mood board pipeline")
    parser.add_argument("--query",  type=str,
                        default="Walking alone through a rainy city street at night")
    parser.add_argument("--images", type=str, nargs="*",
                        help="Uploaded image paths (optional)")
    parser.add_argument("--style",  type=str, default=None,
                        help="Style reference image (optional)")
    parser.add_argument("--skip-layout", action="store_true",
                        help="Skip layout stitching")
    args = parser.parse_args()

    if args.skip_layout:
        os.environ["SKIP_LAYOUT"] = "1"

    results = run_pipeline(
        user_text=args.query,
        uploaded_image_paths=args.images,
        style_ref_path=args.style,
    )

    print("\n=== Final Results ===")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.get('source', '?')}] {r.get('photo_id')}")
        print(f"   score         : {r.get('score', 0):.4f}")
        print(f"   graph_score   : {r.get('graph_score', 'n/a')}")
        print(f"   verified      : {r.get('verified', 'n/a')}")
        print(f"   justification : {r.get('justification', '')[:100]}...")
        print(f"   url           : {r.get('image_url', '')[:70]}...")