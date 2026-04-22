"""
src/agents/guardrails/agent.py

GuardrailAgent
--------------
Two safety checks that wrap the SmartMatch pipeline:

  check_input(query)   — called BEFORE grounding
      Rejects queries that request harmful, violent, sexual, or hateful content.
      Returns (blocked: bool, reason: str).

  check_output(images) — called AFTER justification, BEFORE returning to user
      Filters out any images whose caption or justification contains flagged content.
      Returns the filtered image list (safe images only).

Both checks use Claude as the judge. The input check is a single fast API call.
The output check only runs on generated images (retrieved Unsplash images are
already moderated by Unsplash and are low-risk).

Design notes
------------
- On ANY API error, both checks fail open (allow the request through) so a
  transient API failure never silently breaks the pipeline for the user.
- Responses are parsed as JSON. If Claude returns malformed JSON, we also
  fail open — conservative about blocking, strict about not crashing.
- The system prompt is intentionally narrow: we only block content that is
  clearly harmful. Mood boards are creative tools, so we avoid over-blocking
  legitimate creative queries (dark aesthetics, melancholy themes, etc.).
"""

import os
import json
import anthropic
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("GUARDRAIL_MODEL", "claude-haiku-4-5-20251001")

# ── System prompts ────────────────────────────────────────────────────────────

INPUT_SYSTEM_PROMPT = """You are a content safety filter for an AI mood board generator.

Your job is to decide whether a user's query is safe to process.

BLOCK the query if it explicitly requests:
- Sexual or pornographic content
- Graphic violence or gore
- Content that sexualises minors
- Hate speech targeting a group by race, religion, gender, or sexuality
- Instructions for real-world harm (weapons, drugs, self-harm)

ALLOW everything else, including:
- Dark, melancholy, or sad moods ("feeling hopeless", "grief", "lonely")
- Edgy or unconventional aesthetics (gothic, industrial, dystopian)
- Mature themes that are not explicitly harmful (heartbreak, anxiety, burnout)
- Creative or abstract concepts, even if unusual

Respond ONLY with valid JSON:
{
  "blocked": true or false,
  "reason": "one sentence — why it is blocked, or 'safe' if allowed"
}"""

OUTPUT_SYSTEM_PROMPT = """You are a content safety filter reviewing AI-generated image descriptions.

You will receive the caption and justification for a single generated image.
Decide whether the content is safe to show to a user.

FLAG the image if its caption or justification describes:
- Sexual or pornographic content
- Graphic violence or gore
- Content that sexualises minors
- Hate speech

PASS everything else.

Respond ONLY with valid JSON:
{
  "safe": true or false,
  "reason": "one sentence"
}"""


class GuardrailAgent:
    """
    Input and output safety checks for the SmartMatch pipeline.

    Used by OrchestratorAgent — called automatically when this file exists.

    Usage
    -----
        agent = GuardrailAgent()

        blocked, reason = agent.check_input("campaign for a sustainable coffee brand")
        # → (False, "safe")

        blocked, reason = agent.check_input("generate explicit adult content")
        # → (True, "Request contains explicit sexual content.")

        safe_images = agent.check_output(images)
        # → filtered list with flagged images removed
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Public interface ──────────────────────────────────────────────────────

    def check_input(self, query: str) -> tuple[bool, str]:
        """
        Check whether a user query is safe to process.

        Args:
            query : raw user text input

        Returns:
            (blocked, reason)
            blocked=True  → pipeline should stop; reason explains why
            blocked=False → query is safe; reason is "safe"
        """
        if not query or not query.strip():
            return False, "safe"

        try:
            resp = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                system=INPUT_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": f"User query: {query.strip()}"
                }],
            )
            blocked, reason = self._parse_input_response(resp.content[0].text)
            status = "BLOCKED" if blocked else "ALLOWED"
            print(f"[Guardrail][input] {status} — {reason}")
            return blocked, reason

        except Exception as e:
            # Fail open: a transient API error should not block the user
            print(f"[Guardrail][input] API error ({e}) — failing open")
            return False, "safe"

    def check_output(self, images: list[dict]) -> list[dict]:
        """
        Filter out any generated images whose content is flagged as unsafe.

        Only checks images from generated sources (dalle3, image_editing/*).
        Retrieved Unsplash images are skipped — they are already moderated.

        Args:
            images : list of image dicts (from justification agent)

        Returns:
            Filtered list with unsafe images removed.
        """
        if not images:
            return images

        safe_images = []
        for img in images:
            source = img.get("source", "")

            # Only check generated images
            if not self._is_generated(source):
                safe_images.append(img)
                continue

            caption       = img.get("caption", "")
            justification = img.get("justification", "")

            if not caption and not justification:
                safe_images.append(img)
                continue

            is_safe, reason = self._check_image(caption, justification)
            if is_safe:
                safe_images.append(img)
            else:
                print(f"[Guardrail][output] Removed image {img.get('photo_id', '?')} — {reason}")

        n_removed = len(images) - len(safe_images)
        if n_removed > 0:
            print(f"[Guardrail][output] Removed {n_removed} image(s). {len(safe_images)} remain.")
        else:
            print(f"[Guardrail][output] All {len(safe_images)} image(s) passed.")

        return safe_images

    # ── Internal ──────────────────────────────────────────────────────────────

    def _check_image(self, caption: str, justification: str) -> tuple[bool, str]:
        """
        Check a single generated image. Returns (is_safe, reason).
        Fails open on error.
        """
        content = f"Caption: {caption[:300]}\nJustification: {justification[:300]}"
        try:
            resp = self.client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=100,
                system=OUTPUT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": content}],
            )
            return self._parse_output_response(resp.content[0].text)

        except Exception as e:
            print(f"[Guardrail][output] API error ({e}) — failing open")
            return True, "safe"

    def _is_generated(self, source: str) -> bool:
        return source == "dalle3" or source.startswith("image_editing/")

    def _parse_input_response(self, text: str) -> tuple[bool, str]:
        text = self._strip_fences(text)
        try:
            data    = json.loads(text)
            blocked = bool(data.get("blocked", False))
            reason  = str(data.get("reason", "")).strip() or ("blocked" if blocked else "safe")
            return blocked, reason
        except json.JSONDecodeError:
            return False, "safe"

    def _parse_output_response(self, text: str) -> tuple[bool, str]:
        text = self._strip_fences(text)
        try:
            data   = json.loads(text)
            safe   = bool(data.get("safe", True))
            reason = str(data.get("reason", "")).strip() or ("safe" if safe else "flagged")
            return safe, reason
        except json.JSONDecodeError:
            return True, "safe"

    def _strip_fences(self, text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("```", 2)[1]
            if text.startswith("json"):
                text = text[4:]
        return text.strip()


# ── CLI test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    agent = GuardrailAgent()

    print("=== Input Guardrail Tests ===\n")

    test_inputs = [
        # Should be ALLOWED
        "campaign visuals for a sustainable coffee brand",
        "feeling burnt out after a long week",
        "dark gothic aesthetic, melancholy and rain",
        "grief and loss, quiet and mournful",
        # Should be BLOCKED
        "generate explicit sexual content",
        "graphic violence and gore",
        "content sexualising children",
    ]

    for query in test_inputs:
        blocked, reason = agent.check_input(query)
        label = "BLOCKED" if blocked else "ALLOWED"
        print(f"[{label}] {query!r}")
        print(f"         → {reason}\n")

    print("=== Output Guardrail Tests ===\n")

    test_images = [
        {
            "photo_id": "safe_1",
            "source": "dalle3",
            "caption": "A cozy coffee shop with warm amber lighting and steam rising from a mug",
            "justification": "Evokes warmth and comfort, matching the campaign brief.",
        },
        {
            "photo_id": "safe_retrieved",
            "source": "hybrid",
            "caption": "Sunlit forest path in autumn",
            "justification": "Natural and peaceful, fits the mood.",
        },
    ]

    safe = agent.check_output(test_images)
    print(f"\n{len(safe)}/{len(test_images)} images passed output guardrail.")
