import os
import anthropic
from dotenv import load_dotenv
try:
    from prompts import JUSTIFICATION_SYSTEM_PROMPT, JUSTIFICATION_USER_TEMPLATE  # when run directly
except ImportError:
    from agents.qwen_visual_grounding.prompts import JUSTIFICATION_SYSTEM_PROMPT, JUSTIFICATION_USER_TEMPLATE  # when imported by pipeline

load_dotenv()

MODEL = os.getenv("GROUNDING_MODEL", "claude-haiku-4-5-20251001")


class QwenJustificationAgent:
    """
    Given the user's original text and a list of retrieved images,
    generates a short natural-language explanation for why each image
    is a good match.
    """

    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def run(self, user_text: str, images: list) -> list:
        """
        Adds a 'justification' field to each image dict.

        Args:
            user_text: the original text the user submitted
            images: list of dicts from the retrieval agent, each with
                    at least 'photo_id', 'image_url', 'caption', 'score'

        Returns:
            the same list with a 'justification' key added to each item
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

    def _justify(self, user_text: str, caption: str) -> str:
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
                    },
                ],
            )
            return response.content[0].text.strip()

        except Exception as e:
            print(f"    Warning: justification failed ({e}). Skipping.")
            return ""


if __name__ == "__main__":
    # quick smoke test with fake retrieval results
    agent = QwenJustificationAgent()

    user_text = "feeling burnt out after a long week"

    fake_images = [
        {
            "photo_id": "abc123",
            "image_url": "https://example.com/img1.jpg",
            "caption": "a person sitting alone at a desk late at night, surrounded by empty coffee cups",
            "score": 0.87,
        },
        {
            "photo_id": "def456",
            "image_url": "https://example.com/img2.jpg",
            "caption": "a quiet park bench with fallen autumn leaves",
            "score": 0.81,
        },
    ]

    results = agent.run(user_text, fake_images)

    for r in results:
        print(f"\nImage: {r['photo_id']}")
        print(f"Justification: {r['justification']}")
