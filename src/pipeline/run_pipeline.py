"""
src/pipeline/run_pipeline.py

Thin wrapper around OrchestratorAgent.

All pipeline logic now lives in src/agents/orchestrator/agent.py.
This file exists only for backward compatibility so that callers
(ablation.py, smoke_test.py, CLI scripts) don't need to change.

Usage (unchanged from before):
    from pipeline.run_pipeline import run_pipeline
    results = run_pipeline("feeling burnt out after a long week")
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agents.orchestrator.agent import OrchestratorAgent
from agents.memory.memory_manager import MemoryManager

_orchestrator = OrchestratorAgent()


def run_pipeline(
    user_text: str,
    uploaded_image_paths: list[str] | None = None,
    style_ref_path: str | None = None,
    memory: MemoryManager | None = None,
) -> list[dict]:
    """
    Run the SmartMatch pipeline and return a list of image dicts.

    Args:
        user_text             : raw text input from the user
        uploaded_image_paths  : local image file paths (optional)
        style_ref_path        : style reference for composite mode (optional)
        memory                : MemoryManager for multi-turn context (optional)

    Returns:
        list of result dicts, each with:
            photo_id, image_url, caption, score, source,
            siglip_score, text_score, graph_score,
            verified, verification_reason,
            coherence_rank, justification, board_reason
    """
    bundle = _orchestrator.run(
        query=user_text,
        uploaded_image_paths=uploaded_image_paths,
        style_ref_path=style_ref_path,
        memory=memory,
    )
    return bundle.to_legacy_list()


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description="SmartMatch pipeline")
    parser.add_argument("--query",  type=str,
                        default="Walking alone through a rainy city street at night")
    parser.add_argument("--images", type=str, nargs="*")
    parser.add_argument("--style",  type=str, default=None)
    args = parser.parse_args()

    results = run_pipeline(
        user_text=args.query,
        uploaded_image_paths=args.images,
        style_ref_path=args.style,
    )

    print("\n=== Results ===")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [{r.get('source', '?')}] {r.get('photo_id')}")
        print(f"   score         : {r.get('score', 0):.4f}")
        print(f"   verified      : {r.get('verified', 'n/a')}")
        print(f"   justification : {r.get('justification', '')[:100]}")
        print(f"   url           : {r.get('image_url', '')[:70]}")
