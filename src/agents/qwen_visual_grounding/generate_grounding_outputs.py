"""
generate_grounding_outputs.py

Takes real image captions from the dataset and runs them through the
QwenVisualGroundingAgent to produce structured grounding outputs.

This gives the SigLIP retrieval agent realistic queries derived from
actual image captions — so similarity scores should be higher than
with abstract user text.

Usage:
    python src/agents/qwen_visual_grounding/generate_grounding_outputs.py
"""

import csv
import json
import os
import sys

# allow imports from src/ when running this script directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent

INPUT_CSV = "src/data/processed/dataset_clean.csv"
OUTPUT_FILE = "src/data/processed/grounding_outputs.json"
N = 100  # number of captions to process


def load_captions(csv_path, n):
    captions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            caption = row.get("photo_description_clean", "").strip()
            if caption:  # skip rows with empty captions
                captions.append(caption)
            if len(captions) >= n:
                break
    return captions


def main():
    print(f"Loading {N} captions from {INPUT_CSV}...")
    captions = load_captions(INPUT_CSV, N)
    print(f"Loaded {len(captions)} captions.\n")

    agent = QwenVisualGroundingAgent()
    results = []

    for caption in captions:
        grounding = agent.run(caption)
        results.append({
            "input_text": caption,
            "grounding_output": grounding,
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} grounding outputs to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
