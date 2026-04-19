import os
import anthropic
from dotenv import load_dotenv

try:
    from prompts import (
        JUSTIFICATION_SYSTEM_PROMPT,
        JUSTIFICATION_USER_TEMPLATE,
        BOARD_SUMMARY_SYSTEM_PROMPT,
        BOARD_SUMMARY_USER_TEMPLATE,
    )
except ImportError:
    from src.agents.qwen_visual_grounding.prompts import (
        JUSTIFICATION_SYSTEM_PROMPT,
        JUSTIFICATION_USER_TEMPLATE,
        BOARD_SUMMARY_SYSTEM_PROMPT,
        BOARD_SUMMARY_USER_TEMPLATE,
    )

load_dotenv()

MODEL = os.getenv("GROUNDING_MODEL", "claude-haiku-4-5-20251001")


class QwenJustificationAgent:
    """
    Generates:
    1. Per-image justifications
    2. Board-level summary (why the set works together)
    """

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

    # ── Public Interface ──────────────────────────────────────────────────────

    def run(self, user_text: str, images: list) -> list:
        """
        Backward-compatible method:
        Adds 'justification' to each image dict.
        """
        print(f"Generating justifications for {len(images)} image(s)...")

        results = []
        for i, image in enumerate(images):
            caption = image.get("caption", "")

            print(
                f"  [{i+1}/{len(images)}] Justifying image {image.get('photo_id', '?')}..."
            )

            justification = self._justify(user_text, caption)

            results.append({
                **image,
                "justification": justification
            })

        print("Done generating justifications.")
        return results

    def run_with_board_summary(self, user_text: str, images: list) -> tuple[list, str]:
        """
        Full method:
        Returns per-image justifications + board-level summary.
        """
        # Step 1: Per-image justifications
        results = self.run(user_text, images)

        # Step 2: Board-level summary
        print("Generating board summary...")
        board_summary = self._summarize_board(user_text, results)
        print("Done.")

        return results, board_summary

    # ── Internal Methods ──────────────────────────────────────────────────────

    def _justify(self, user_text: str, caption: str) -> str:
        """Generate explanation for a single image."""
        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=200,
                system=JUSTIFICATION_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": JUSTIFICATION_USER_TEMPLATE.format(
                            user_text=user_text,
                            caption=caption,
                        ),
                    }
                ],
            )

            return response.content[0].text.strip()

        except Exception as e:
            print(f"    Warning: justification failed ({e})")
            return ""

    def _summarize_board(self, user_text: str, images: list) -> str:
        """
        Generate a cohesive explanation for the full image set.
        """

        # Build compact description of all images
        image_descriptions = "\n".join([
            f"- {img.get('caption', '')[:100]}"
            for img in images
            if img.get("caption", "").strip()
        ])

        if not image_descriptions:
            return ""

        try:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=250,
                system=BOARD_SUMMARY_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": BOARD_SUMMARY_USER_TEMPLATE.format(
                            user_text=user_text,
                            image_descriptions=image_descriptions,
                        ),
                    }
                ],
            )

            return response.content[0].text.strip()

        except Exception as e:
            print(f"    Warning: board summary failed ({e})")
            return ""