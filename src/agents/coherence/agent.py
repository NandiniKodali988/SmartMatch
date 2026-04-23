"""
src/agents/coherence/agent.py

Coherence Agent
---------------
Given a pool of verified candidate images, selects the final subset
(up to target_count) that work best together as a mood board.

Thinks about the board as a whole — consistent color palette, no visual
repetition, coherent mood across all images — rather than ranking images
individually by score.

Input:
    candidates      : list of dicts from MultimodalVerificationAgent.run()
                      each dict has at least: photo_id, caption, score,
                      verified, verification_reason
    grounding_output: dict from QwenVisualGroundingAgent
    target_count    : how many images to select (default 9)

Output:
    list of selected image dicts (up to target_count), each with a
    'coherence_rank' field added (1 = best fit for the board).
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("COHERENCE_MODEL", "claude-haiku-4-5-20251001")

SYSTEM_PROMPT = """You are a visual creative director selecting images for a mood board.

You will receive a list of candidate images (described by captions and metadata)
and a target visual concept. Your job is to select the subset that works best
together as a cohesive mood board — not just individually good images, but images
that tell a consistent visual story as a set.

When selecting, consider:
- Mood consistency: all images should evoke the same emotional tone
- Color palette harmony: avoid jarring clashes between very warm and very cold images
- Visual variety: avoid picking images that are essentially the same scene twice
- Narrative flow: the set should feel like it belongs together, like pages from the same book

CRITICAL: Visual diversity is required.
- The 9 selected images MUST vary in composition, scale, and perspective
- Do NOT select more than 2 images with nearly identical compositions
- Prefer a mix of: wide shots, close-ups, human presence, no people, different color registers
- A board with 9 similar images scores 1/5 on coherence regardless of relevance

Respond ONLY with valid JSON in this exact format:
{
  "selected_ids": ["photo_id_1", "photo_id_2", ...],
  "reason": "one sentence explaining why these images work together as a set"
}

Return only the photo_ids of the selected images, ordered from best fit to least fit.
Do not include any images that feel out of place even if they scored well individually."""


class CoherenceAgent:
    """
    Selects the most coherent subset of verified images for a mood board.

    Usage
    -----
        agent = CoherenceAgent(target_count=9)
        selected = agent.run(verified_candidates, grounding_output)
    """

    def __init__(self, target_count: int = 9):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.target_count = target_count

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        candidates: list[dict],
        grounding_output: dict,
    ) -> list[dict]:
        """
        Select the most coherent subset from the verified candidate pool.

        Args:
            candidates      : verified image dicts (from MultimodalVerificationAgent)
            grounding_output: structured visual concept from grounding agent

        Returns:
            selected image dicts with 'coherence_rank' added, ordered by fit.
            Falls back to top-scored verified images if Claude call fails.
        """
        verified = [c for c in candidates if c.get("verified", True)]

        if not verified:
            print("[Coherence] No verified candidates — returning empty list.")
            return []

        if len(verified) <= self.target_count:
            print(f"[Coherence] Only {len(verified)} verified candidates — skipping selection, returning all.")
            return self._add_ranks(verified)

        print(f"[Coherence] Selecting {self.target_count} from {len(verified)} verified candidates...")

        selected_ids, board_reason = self._select(verified, grounding_output)

        if not selected_ids:
            print("[Coherence] Claude returned no selections — falling back to top scores.")
            return self._add_ranks(verified[: self.target_count])

        id_to_candidate = {str(c["photo_id"]): c for c in verified}
        selected = []
        for rank, pid in enumerate(selected_ids[: self.target_count], start=1):
            candidate = id_to_candidate.get(str(pid))
            if candidate:
                selected.append({**candidate, "coherence_rank": rank, "board_reason": board_reason})

        # If Claude returned fewer than target, pad with remaining verified by score
        if len(selected) < self.target_count:
            selected_pid_set = {str(c["photo_id"]) for c in selected}
            remaining = [
                c for c in verified if str(c["photo_id"]) not in selected_pid_set
            ]
            remaining.sort(key=lambda x: x.get("score", 0), reverse=True)
            next_rank = len(selected) + 1
            for c in remaining[: self.target_count - len(selected)]:
                selected.append({**c, "coherence_rank": next_rank, "board_reason": board_reason})
                next_rank += 1

        print(f"[Coherence] Selected {len(selected)} image(s).")
        print(f"[Coherence] Board rationale: {board_reason}")
        return selected

    # ── Internal ──────────────────────────────────────────────────────────────

    def _select(
        self,
        candidates: list[dict],
        grounding_output: dict,
    ) -> tuple[list[str], str]:
        """Ask Claude to select the most coherent subset. Returns (id_list, reason)."""
        prompt = self._build_prompt(candidates, grounding_output)

        try:
            resp = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return self._parse_response(resp.content[0].text)

        except Exception as e:
            print(f"[Coherence] Claude call failed: {e}")
            return [], ""

    def _build_prompt(self, candidates: list[dict], grounding_output: dict) -> str:
        target_block = (
            f"Target concept: {grounding_output.get('visual_description', '')}\n"
            f"Target mood: {grounding_output.get('mood', '')}\n"
            f"Target color palette: {grounding_output.get('color_palette', '')}\n"
            f"Target style: {grounding_output.get('style', '')}\n"
        )

        candidate_lines = []
        for i, c in enumerate(candidates, start=1):
            caption = c.get("caption", "").replace("\n", " ").strip()
            reason = c.get("verification_reason", "")
            line = (
                f"{i}. photo_id={c['photo_id']}  "
                f"score={c.get('score', 0):.3f}  "
                f"caption: {caption[:120]}"
            )
            if reason:
                line += f"  [verified: {reason}]"
            candidate_lines.append(line)

        candidates_block = "\n".join(candidate_lines)

        return (
            f"{target_block}\n"
            f"Select up to {self.target_count} images from the following {len(candidates)} candidates "
            f"that work best together as a cohesive mood board:\n\n"
            f"{candidates_block}"
        )

    def _parse_response(self, text: str) -> tuple[list[str], str]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data = json.loads(text)
            selected_ids = [str(pid) for pid in data.get("selected_ids", [])]
            reason = str(data.get("reason", "")).strip()
            return selected_ids, reason
        except json.JSONDecodeError:
            print("[Coherence] Could not parse response.")
            return [], ""

    def _add_ranks(self, candidates: list[dict]) -> list[dict]:
        """Add coherence_rank to a list when no selection was needed."""
        return [
            {**c, "coherence_rank": i + 1, "board_reason": ""}
            for i, c in enumerate(candidates)
        ]


if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv, find_dotenv

    load_dotenv(find_dotenv())

    fake_grounding = {
        "visual_description": "soft morning light filtering through curtains onto a wooden table with a coffee cup",
        "mood": "calm, peaceful, quiet",
        "color_palette": "warm whites, soft beiges, golden tones",
        "style": "minimalist, editorial",
    }

    fake_candidates = [
        {"photo_id": "img_001", "caption": "A steaming coffee mug on a sunlit windowsill", "score": 0.82, "verified": True, "verification_reason": "warm tones match"},
        {"photo_id": "img_002", "caption": "Busy city street with crowds and neon signs", "score": 0.71, "verified": True, "verification_reason": "verified but off-mood"},
        {"photo_id": "img_003", "caption": "A cozy armchair beside a soft lamp and open book", "score": 0.79, "verified": True, "verification_reason": "quiet domestic mood matches"},
        {"photo_id": "img_004", "caption": "Fresh croissants on a linen cloth", "score": 0.76, "verified": True, "verification_reason": "warm tones and calm feel"},
        {"photo_id": "img_005", "caption": "A wooden table with scattered wildflowers", "score": 0.74, "verified": True, "verification_reason": "natural palette fits"},
        {"photo_id": "img_006", "caption": "Neon lights and rain-slicked pavement at night", "score": 0.65, "verified": True, "verification_reason": "mood mismatch but verified"},
        {"photo_id": "img_007", "caption": "Soft morning light through linen curtains", "score": 0.81, "verified": True, "verification_reason": "directly matches description"},
        {"photo_id": "img_008", "caption": "White ceramic bowls on a pale stone surface", "score": 0.73, "verified": True, "verification_reason": "minimalist and calm"},
        {"photo_id": "img_009", "caption": "Dew on grass in early morning sunlight", "score": 0.70, "verified": True, "verification_reason": "gentle and natural"},
        {"photo_id": "img_010", "caption": "A crowded festival with colorful banners", "score": 0.60, "verified": True, "verification_reason": "energy is wrong for this mood"},
    ]

    agent = CoherenceAgent(target_count=6)
    result = agent.run(fake_candidates, fake_grounding)

    print("\n=== Coherence Selection Result ===")
    for img in result:
        print(f"  [{img['coherence_rank']}] {img['photo_id']}  score={img.get('score', 0):.3f}  {img['caption'][:60]}")
    if result:
        print(f"\nBoard rationale: {result[0].get('board_reason', '')}")
