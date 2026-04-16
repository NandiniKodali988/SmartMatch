"""
src/agents/multimodal_verification/agent.py

Multimodal Verification Agent
-------------------------------
After hybrid retrieval, passes each candidate image URL to Claude Vision
along with the grounding output to verify the image actually matches
the user's intent — something embedding scores alone cannot guarantee.

Input:
    candidates      : list of dicts from FieldTextRetrievalAgent.merge_with_siglip()
    grounding_output: dict from QwenVisualGroundingAgent

Output:
    same list of dicts with two fields added per image:
        verified            (bool) : True if image passes visual check
        verification_reason (str)  : one brief sentence explaining the decision
"""

import os
import base64
import anthropic
import requests
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL    = os.getenv("VERIFICATION_MODEL", "claude-haiku-4-5-20251001")
REQUEST_TIMEOUT = int(os.getenv("VERIFICATION_TIMEOUT", "10"))

# Unsplash requires specific params to serve the actual image bytes
UNSPLASH_PARAMS = {"w": "400", "q": "60", "fm": "jpg", "fit": "crop"}

SYSTEM_PROMPT = """You are a visual verification assistant for a mood board generator.

Given an image and a target visual description, decide whether the image could
reasonably belong in a mood board for the described feeling or concept.

Rules:
- Focus primarily on MOOD and COLOR PALETTE — these matter most for a mood board.
- Be lenient about exact scene or subject: a contemplative portrait can represent
  "burnt out" even without a desk; a dark forest can represent "lonely" even without a person.
- Only reject if the image clearly contradicts the mood
  (e.g. a bright cheerful party scene for an exhausted/melancholic query).
- Accept images that evoke the right feeling even if the subject differs.

Respond ONLY with valid JSON:
{
  "verified": true or false,
  "reason": "one concise sentence"
}"""


class MultimodalVerificationAgent:
    """
    Verifies retrieved images using Claude Vision.

    Usage
    -----
        agent = MultimodalVerificationAgent(min_verified=6)
        verified = agent.run(candidates, grounding_output)
    """

    def __init__(self, min_verified: int = 6):
        """
        Args:
            min_verified : minimum number of verified images required before
                           triggering DALL·E 3 fallback in the pipeline.
        """
        self.client       = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.min_verified = min_verified

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        candidates: list[dict],
        grounding_output: dict,
    ) -> list[dict]:
        """
        Verify each candidate image against the grounding output.

        Returns:
            same list with 'verified' and 'verification_reason' added.
            Verified images sorted first.
        """
        print(f"[Verification] Verifying {len(candidates)} candidate(s)...")

        results = []
        for i, candidate in enumerate(candidates):
            photo_id  = candidate.get("photo_id", f"idx_{i}")
            image_url = candidate.get("image_url", "")

            print(f"  [{i+1}/{len(candidates)}] {photo_id}...", end=" ", flush=True)

            verified, reason = self._verify(image_url, grounding_output)
            status = "✓" if verified else "✗"
            print(f"{status} {reason}")

            results.append({
                **candidate,
                "verified":            verified,
                "verification_reason": reason,
            })

        passed = sum(1 for r in results if r["verified"])
        print(f"[Verification] {passed}/{len(results)} passed.")
        print(f"[Verification] Min required: {self.min_verified} — "
              f"{'sufficient' if passed >= self.min_verified else 'will trigger generation fallback'}.")

        # Verified images first, then failed
        results.sort(key=lambda x: x["verified"], reverse=True)
        return results

    def needs_generation(self, verified_results: list[dict]) -> bool:
        """Returns True if not enough images passed verification."""
        passed = sum(1 for r in verified_results if r["verified"])
        return passed < self.min_verified

    # ── Internal ──────────────────────────────────────────────────────────────

    def _verify(
        self,
        image_url: str,
        grounding_output: dict,
    ) -> tuple[bool, str]:
        """Call Claude Vision on a single image. Returns (verified, reason)."""
        if not image_url:
            return False, "No image URL available."

        image_data, media_type = self._fetch_image(image_url)
        if image_data is None:
            return False, "Could not fetch image."

        user_content = [
            {
                "type": "image",
                "source": {
                    "type":       "base64",
                    "media_type": media_type,
                    "data":       image_data,
                },
            },
            {
                "type": "text",
                "text": self._build_prompt(grounding_output),
            },
        ]

        try:
            resp = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            )
            return self._parse_response(resp.content[0].text)

        except Exception as e:
            print(f"\n    [Verification] Claude call failed: {e}")
            return False, "Verification error — skipped."

    def _fetch_image(self, url: str) -> tuple[str | None, str]:
        """
        Fetch image bytes and return (base64_string, media_type).
        Appends Unsplash params if URL is from images.unsplash.com
        to ensure the image is actually served (avoids 403).
        """
        try:
            # Unsplash CDN requires query params to serve image bytes
            if "images.unsplash.com" in url:
                resp = requests.get(url, params=UNSPLASH_PARAMS, timeout=REQUEST_TIMEOUT)
            else:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT)

            resp.raise_for_status()

            content_type = resp.headers.get("Content-Type", "image/jpeg")
            media_type   = content_type.split(";")[0].strip()

            if media_type not in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
                media_type = "image/jpeg"

            b64 = base64.standard_b64encode(resp.content).decode("utf-8")
            return b64, media_type

        except Exception as e:
            print(f"\n    [Verification] Fetch failed ({url[:60]}...): {e}")
            return None, ""

    def _build_prompt(self, grounding_output: dict) -> str:
        return (
            f"Target mood: {grounding_output.get('mood', '')}\n"
            f"Target color palette: {grounding_output.get('color_palette', '')}\n"
            f"Target visual description: {grounding_output.get('visual_description', '')}\n\n"
            f"Does this image evoke the right mood and color palette for the description above?"
        )

    def _parse_response(self, text: str) -> tuple[bool, str]:
        import json

        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
            text = text.strip()

        try:
            data     = json.loads(text)
            verified = bool(data.get("verified", False))
            reason   = str(data.get("reason", "")).strip()
            if len(reason) > 120:
                reason = reason[:117] + "..."
            return verified, reason

        except json.JSONDecodeError:
            return False, "Could not parse verification response."