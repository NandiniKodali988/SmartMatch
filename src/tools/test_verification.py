"""
src/tools/test_verification.py

Quick test for MultimodalVerificationAgent.
Uses real Unsplash URLs from the dataset with a known query,
so you can visually check whether Claude's pass/fail decisions make sense.

Usage:
    python src/tools/test_verification.py --query "feeling burnt out after a long week"
    python src/tools/test_verification.py --query "fresh blueberries" --top_k 8
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
from agents.multimodal_verification.agent import MultimodalVerificationAgent


def main(query: str, top_k: int):
    print(f'\n=== Testing Verification for: "{query}" ===\n')

    # Step 1: Grounding
    print("[1/3] Running visual grounding...")
    grounding = QwenVisualGroundingAgent().run(query)
    print(f"      visual_description : {grounding.get('visual_description', '')[:80]}...")
    print(f"      mood               : {grounding.get('mood', '')}")
    print(f"      color_palette      : {grounding.get('color_palette', '')}\n")

    # Step 2: Hybrid retrieval — get candidates
    print(f"[2/3] Hybrid retrieval (top_k={top_k})...")
    siglip_agent = SiglipImageRetrievalAgent(top_k=top_k)
    text_agent   = FieldTextRetrievalAgent(top_k=top_k)
    candidates   = text_agent.merge_with_siglip(grounding, siglip_agent)
    print(f"      {len(candidates)} candidates retrieved.\n")

    # Step 3: Verification
    print(f"[3/3] Running multimodal verification...")
    verification_agent  = MultimodalVerificationAgent(min_verified=6)
    verified_candidates = verification_agent.run(candidates, grounding)

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    passed = [c for c in verified_candidates if c["verified"]]
    failed = [c for c in verified_candidates if not c["verified"]]

    print(f"\n✓ PASSED ({len(passed)}/{len(verified_candidates)})")
    for r in passed:
        print(f"\n  photo_id  : {r['photo_id']}")
        print(f"  score     : {r['score']:.4f}  "
              f"(siglip={r.get('siglip_score',0):.4f}, text={r.get('text_score',0):.4f})")
        print(f"  reason    : {r['verification_reason']}")
        print(f"  url       : {r['image_url'][:70]}...")

    print(f"\n✗ FAILED ({len(failed)}/{len(verified_candidates)})")
    for r in failed:
        print(f"\n  photo_id  : {r['photo_id']}")
        print(f"  score     : {r['score']:.4f}  "
              f"(siglip={r.get('siglip_score',0):.4f}, text={r.get('text_score',0):.4f})")
        print(f"  reason    : {r['verification_reason']}")
        print(f"  url       : {r['image_url'][:70]}...")

    print(f"\n{'=' * 60}")
    needs_gen = verification_agent.needs_generation(verified_candidates)
    print(f"needs_generation : {needs_gen}  "
          f"(min_verified={verification_agent.min_verified}, passed={len(passed)})")
    print("=" * 60)

    # Print all URLs in one block so you can open them and visually check
    print("\n--- All image URLs (open to verify manually) ---")
    for r in verified_candidates:
        status = "✓" if r["verified"] else "✗"
        print(f"{status} {r['photo_id']}: {r['image_url']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",  type=str, required=True)
    parser.add_argument("--top_k",  type=int, default=8)
    args = parser.parse_args()
    main(args.query, args.top_k)
