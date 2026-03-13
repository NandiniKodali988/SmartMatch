import os
import json
import time
import anthropic
from dotenv import load_dotenv
try:
    from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # when run directly
except ImportError:
    from agents.qwen_visual_grounding.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE  # when imported by pipeline

load_dotenv()

MODEL = os.getenv("GROUNDING_MODEL", "claude-haiku-4-5-20251001")
REQUIRED_KEYS = {"visual_description", "scene", "mood", "style"}


class QwenVisualGroundingAgent:
    """
    Converts abstract user text into a structured visual description
    that SigLIP-2 can use for image retrieval.
    """

    def __init__(self, max_retries=3):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.max_retries = max_retries

    def run(self, text: str) -> dict:
        """
        Main entry point. Takes a user's text and returns a dict with
        visual_description, scene, mood, and style fields.
        """
        print(f"Grounding visual concepts for: '{text}'")

        for attempt in range(1, self.max_retries + 1):
            try:
                result = self._call_model(text)
                self._validate(result)
                print("Visual grounding successful.")
                return result

            except (json.JSONDecodeError, ValueError) as e:
                print(f"Attempt {attempt} failed: {e}")
                if attempt < self.max_retries:
                    print("Retrying...")
                    time.sleep(1)
                else:
                    print("Max retries reached. Returning empty grounding.")
                    return self._fallback(text)

    def _call_model(self, text: str) -> dict:
        response = self.client.messages.create(
            model=MODEL,
            max_tokens=512,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
            ],
        )
        raw = response.content[0].text.strip()
        # strip markdown code fences if the model wraps its output
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    def _validate(self, result: dict):
        missing = REQUIRED_KEYS - result.keys()
        if missing:
            raise ValueError(f"Response missing fields: {missing}")

    def _fallback(self, text: str) -> dict:
        # if the model keeps failing, return a minimal usable dict so the
        # rest of the pipeline doesn't crash
        return {
            "visual_description": text,
            "scene": "",
            "mood": "",
            "style": "",
        }


if __name__ == "__main__":
    agent = QwenVisualGroundingAgent()

    test_inputs = [
        "feeling burnt out after a long week",
        "celebrating a new job offer",
        "that quiet Sunday morning feeling",
        "overwhelmed by city life",
        "finally finished my first marathon",
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        result = agent.run(text)
        print(json.dumps(result, indent=2))
