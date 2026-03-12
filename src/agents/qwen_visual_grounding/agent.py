import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

client = InferenceClient(
    api_key=os.environ["HF_TOKEN"],
)

MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"


def ground_visual_concepts(text: str) -> dict:
    """
    Takes a user text input and returns a structured visual description
    suitable for SigLip-2 embedding and image retrieval.

    Args:
        text: A social media post, caption, or article excerpt

    Returns:
        dict with keys: visual_description, scene, mood, style
    """
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)},
        ],
        response_format={"type": "json_object"},
        max_tokens=512,
    )

    raw = response.choices[0].message.content
    return json.loads(raw)


if __name__ == "__main__":
    test_inputs = [
        "feeling burnt out after a long week",
        "celebrating a new job offer",
        "that quiet Sunday morning feeling",
        "overwhelmed by city life",
        "finally finished my first marathon",
    ]

    for text in test_inputs:
        print(f"\nInput: {text}")
        result = ground_visual_concepts(text)
        print(json.dumps(result, indent=2))
