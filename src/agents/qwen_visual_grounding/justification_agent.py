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
    from agents.qwen_visual_grounding.prompts import (
        JUSTIFICATION_SYSTEM_PROMPT,
        JUSTIFICATION_USER_TEMPLATE,
        BOARD_SUMMARY_SYSTEM_PROMPT,
        BOARD_SUMMARY_USER_TEMPLATE,
    )

load_dotenv()

MODEL = os.getenv("GROUNDING_MODEL", "claude-haiku-4-5-20251001")


class QwenJustificationAgent:
    """
    Generates per-image justifications and an optional board-level summary.

    Methods
    -------
    run(user_text, images)
        Original method — adds a 'justification' field to each image dict.

    run_with_board_summary(user_text, images)
        Extended method — same as run(), but also returns a one-paragraph
        board_summary explaining why the images work together as a set.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, user_text: str, images: list) -> list:
        """
        Adds a 'justification' field to each image dict.

        Args:
            user_text : the original text the user submitted
            images    : list of dicts, each with at least photo_id, image_url,
                        caption, score

        Returns:
            same list with 'justification' added to each item
        """
        print(f"Generating justifications for {len(images)} image(s)...")

        results = []
        for i, image in enumerate(images):
            caption = image.get("caption", "")
            print(f"  [{i+1}/{len(images)}] Justifying image {image.get('photo_id', '?')}...")
            justification = self._justify(user_text, caption)
            results.append({**image, "justification": justification})

        print("Done generating justifications.")
        return results

    def run_with_board_summary(
        self,
        user_text: str,
        images: list,
    ) -> tuple[list, str]:
        """
        Same as run(), but also generates a board-level summary.

        Returns:
            (results, board_summary)
                results       : list of image dicts with 'justification' added
                board_summary : one paragraph explaining the board as a whole
        """
        # Per-image justifications
        results = self.run(user_text, images)

        # Board-level summary
        print("Generating board summary...")
        board_summary = self._summarize_board(user_text, results)
        print("Done.")

        return results, board_summary

    # ── Internal ──────────────────────────────────────────────────────────────

    def _justify(self, user_text: str, caption: str) -> str:
        """Generate a per-image justification."""
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
            print(f"    Warning: justification failed ({e}). Skipping.")
            return ""

    def _summarize_board(self, user_text: str, images: list) -> str:
        """
        Generate a one-paragraph summary explaining why the board
        works as a cohesive set.
        """
        # Build a compact description of all images for Claude
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
            print(f"    Warning: board summary failed ({e}).")
            return ""