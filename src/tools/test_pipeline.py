"""
src/tools/test_pipeline.py

Complete pipeline integration test.
Tests three scenarios and reports pass/fail for each step.

Usage:
    # Quick test (skip layout stitching, use fewer candidates)
    python src/tools/test_pipeline.py --quick

    # Full test with layout
    python src/tools/test_pipeline.py

    # Specific query
    python src/tools/test_pipeline.py --query "golden hour on a quiet beach"

    # Test uploaded images branch
    python src/tools/test_pipeline.py --images uploads/photo1.png
"""

import os
import sys
import time
import argparse
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def section(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def check(label: str, value, expect_type=None, expect_min_len=None):
    """Assert a value looks right and print pass/fail."""
    try:
        if expect_type and not isinstance(value, expect_type):
            print(f"  ✗ {label}: expected {expect_type}, got {type(value)}")
            return False
        if expect_min_len is not None and len(value) < expect_min_len:
            print(f"  ✗ {label}: len={len(value)}, expected >= {expect_min_len}")
            return False
        print(f"  ✓ {label}: {str(value)[:80]}")
        return True
    except Exception as e:
        print(f"  ✗ {label}: error — {e}")
        return False


def test_text_only(query: str, quick: bool) -> bool:
    section(f"TEST 1: Text-only query — '{query}'")

    if quick:
        os.environ["CANDIDATE_K"] = "8"
        os.environ["BOARD_SIZE"]  = "4"
        os.environ["SKIP_LAYOUT"] = "1"
    else:
        os.environ.pop("CANDIDATE_K", None)
        os.environ.pop("BOARD_SIZE",  None)
        os.environ.pop("SKIP_LAYOUT", None)

    from pipeline.run_pipeline import run_pipeline

    t0 = time.time()
    try:
        output = run_pipeline(user_text=query)
    except Exception as e:
        print(f"\n  ✗ Pipeline raised exception: {e}")
        traceback.print_exc()
        return False

    elapsed = time.time() - t0
    print(f"\n  Pipeline completed in {elapsed:.1f}s")

    passed = True
    passed &= check("output is dict",       output,                  dict)
    passed &= check("images key exists",    output.get("images"),    list, expect_min_len=1)
    passed &= check("board_summary exists", output.get("board_summary", ""), str)

    images = output.get("images", [])
    if images:
        img = images[0]
        passed &= check("image has photo_id",      img.get("photo_id"),      str)
        passed &= check("image has image_url",     img.get("image_url"),      str)
        passed &= check("image has score",         img.get("score"),          (int, float))
        passed &= check("image has justification", img.get("justification"),  str)
        passed &= check("image has source",        img.get("source"),         str)

    if not quick and output.get("board_path"):
        board_path = output["board_path"]
        passed &= check("board_path is str",    board_path, str)
        passed &= check("board PNG exists",     os.path.exists(board_path), bool)

    print(f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def test_grounding_only(query: str) -> bool:
    section(f"TEST 2: Visual Grounding only — '{query}'")

    from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent

    try:
        agent    = QwenVisualGroundingAgent()
        grounding = agent.run(query)
    except Exception as e:
        print(f"  ✗ Grounding raised exception: {e}")
        return False

    passed = True
    for field in ["visual_description", "mood", "color_palette", "intent", "scene", "style"]:
        passed &= check(f"field '{field}'", grounding.get(field), str)

    print(f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def test_verification_only(query: str) -> bool:
    section(f"TEST 3: Verification Agent — '{query}'")

    from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
    from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
    from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
    from agents.multimodal_verification.agent import MultimodalVerificationAgent

    try:
        grounding = QwenVisualGroundingAgent().run(query)
        siglip    = SiglipImageRetrievalAgent(top_k=5)
        text      = FieldTextRetrievalAgent(top_k=5)
        candidates = text.merge_with_siglip(grounding, siglip)
        verified   = MultimodalVerificationAgent(min_verified=3).run(candidates, grounding)
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False

    passed = True
    passed &= check("candidates returned",   candidates, list, expect_min_len=1)
    passed &= check("verified list returned", verified,  list, expect_min_len=1)

    if verified:
        v = verified[0]
        passed &= check("has 'verified' field",            v.get("verified") is not None, bool)
        passed &= check("has 'verification_reason' field", v.get("verification_reason"), str)

    n_pass = sum(1 for v in verified if v.get("verified"))
    print(f"  info: {n_pass}/{len(verified)} images verified")
    print(f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def test_layout_only(image_urls: list) -> bool:
    section("TEST 4: MoodBoard Layout only (uses real URLs)")

    from agents.moodboard_layout.agent import MoodBoardLayoutAgent

    fake_images = [{"image_url": url} for url in image_urls]

    try:
        agent = MoodBoardLayoutAgent()
        path  = agent.run(fake_images)
    except Exception as e:
        print(f"  ✗ Exception: {e}")
        traceback.print_exc()
        return False

    passed = True
    passed &= check("returns a path",      path, str)
    passed &= check("PNG file exists",     os.path.exists(path), bool)
    passed &= check("file size > 0",       os.path.getsize(path) > 0, bool)
    print(f"  Saved to: {path}")

    print(f"\n  {'PASSED ✓' if passed else 'FAILED ✗'}")
    return passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",  type=str,
                        default="feeling burnt out after a long week")
    parser.add_argument("--images", type=str, nargs="*",
                        help="Local image paths to test uploaded-images branch")
    parser.add_argument("--quick",  action="store_true",
                        help="Faster: fewer candidates, smaller board, skip layout")
    parser.add_argument("--test",   type=str,
                        choices=["all", "grounding", "verification", "layout", "pipeline"],
                        default="all")
    args = parser.parse_args()

    results = {}

    if args.test in ("all", "grounding"):
        results["grounding"]     = test_grounding_only(args.query)

    if args.test in ("all", "verification"):
        results["verification"]  = test_verification_only(args.query)

    if args.test in ("all", "layout"):
        # Use a few known Unsplash URLs as test input
        test_urls = [
            "https://images.unsplash.com/photo-1565682459922-cb0d42a0e58d",
            "https://images.unsplash.com/photo-1489871270304-2aa631776b9f",
            "https://images.unsplash.com/photo-1465391422195-6be887eb93a9",
            "https://images.unsplash.com/photo-1551830180-7c66da1cb0b9",
            "https://images.unsplash.com/photo-1546689999-e1db7c13ef06",
        ]
        results["layout"] = test_layout_only(test_urls)

    if args.test in ("all", "pipeline"):
        results["pipeline"] = test_text_only(args.query, args.quick)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("TEST SUMMARY")
    all_passed = True
    for name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {status}  {name}")
        if not passed:
            all_passed = False

    print(f"\n  Overall: {'ALL PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
