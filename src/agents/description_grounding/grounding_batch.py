"""
SmartMatch Grounding: Batch API version (50% discount, no rate limits).

Three-step workflow:
    1. Submit   — build requests and submit to Anthropic Batch API (max 10K per batch)
    2. Poll     — check batch status until complete
    3. Collect  — download results and save as grounding_outputs.json

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...

    # Step 1: Submit (splits 25K rows into 3 batches of 10K/10K/5K)
    python grounding_batch.py submit -i ../../data/processed/dataset_clean.csv

    # Step 2: Poll status
    python grounding_batch.py poll

    # Step 3: Collect results when done
    python grounding_batch.py collect -i ../../data/processed/dataset_clean.csv

    # Or do all three in one command (submits, waits, collects):
    python grounding_batch.py run -i ../../data/processed/dataset_clean.csv

    # Test mode: only first 10 rows
    python grounding_batch.py run -i ../../data/processed/dataset_clean.csv --test 10

Cost: ~$10 for 25K rows (50% off standard Haiku pricing)
Time: Usually 1-4 hours, max 24 hours
"""

import os
import json
import time
import argparse
import pandas as pd
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("Run: pip install anthropic")
    exit(1)


# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
MAX_TOKENS = 250
BATCH_LIMIT = 10_000  # Anthropic max per batch
STATE_FILE = "batch_state.json"  # tracks submitted batch IDs

SYSTEM_PROMPT = """You convert image metadata into a structured visual description for image retrieval.

You will receive:
- keywords: tags describing the image content
- colors: dominant color names in the image
- photo_description: human-written caption (may be empty)
- ai_description: AI-generated description (may be empty)

Return a JSON object with exactly 3 fields:
- visual_description: one rich sentence describing the visible scene (15-30 words), include subjects, environment, lighting, and dominant colors
- mood: comma-separated atmosphere/emotion keywords (2-5 keywords)
- color_palette: comma-separated dominant visual colors as seen in the image (2-5 colors, use common color names like "teal", "golden", "soft white", not CSS names like "darkslategray")

Rules:
- Describe only what is visually observable
- Be specific and concrete, avoid abstract language
- Use simple, common English words
- Translate CSS color names to natural visual terms (e.g. "darkslategray" → "deep teal", "saddlebrown" → "warm brown")
- Return valid JSON only, no extra text, no markdown fences"""


def get_state_path(input_path: str) -> Path:
    return Path(input_path).parent / STATE_FILE


def load_state(input_path: str) -> dict:
    sp = get_state_path(input_path)
    if sp.exists():
        with open(sp) as f:
            return json.load(f)
    return {"batch_ids": [], "model": DEFAULT_MODEL}


def save_state(input_path: str, state: dict):
    sp = get_state_path(input_path)
    with open(sp, "w") as f:
        json.dump(state, f, indent=2)


def build_user_prompt(row: pd.Series) -> str:
    parts = []

    kw = row.get("keyword_list", "")
    if pd.notna(kw) and str(kw).strip() and str(kw).strip() != "[]":
        parts.append(f"keywords: {kw}")

    cl = row.get("color_list", "")
    if pd.notna(cl) and str(cl).strip() and str(cl).strip() != "[]":
        parts.append(f"colors: {cl}")

    pd_desc = row.get("photo_description_clean", "")
    if pd.notna(pd_desc) and str(pd_desc).strip():
        parts.append(f"photo_description: {pd_desc}")

    ai_desc = row.get("ai_description_clean", "")
    if pd.notna(ai_desc) and str(ai_desc).strip():
        parts.append(f"ai_description: {ai_desc}")

    return "\n".join(parts) if parts else "No metadata available"


# ─────────────────────────────────────────
# Step 1: Submit
# ─────────────────────────────────────────


def cmd_submit(input_path: str, model: str, test_n: int | None):
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows from {input_path}")

    if test_n:
        df = df.head(test_n)
        print(f"TEST MODE: first {test_n} rows only")

    # Build requests
    requests = []
    for i, row in df.iterrows():
        photo_id = str(row["photo_id"])
        requests.append(
            {
                "custom_id": photo_id,
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": build_user_prompt(row)}],
                },
            }
        )

    # Cost estimate (Batch = 50% off: $0.50/M input, $2.50/M output for Haiku)
    est_input = len(requests) * 350
    est_output = len(requests) * 100
    est_cost = est_input / 1e6 * 0.50 + est_output / 1e6 * 2.50
    print(f"\nEstimated cost: ~${est_cost:.2f} (50% batch discount)")
    print(f"  {est_input / 1e6:.2f}M input + {est_output / 1e6:.2f}M output tokens")
    print(f"  Model: {model}")
    print(
        f"  Batches: {(len(requests) - 1) // BATCH_LIMIT + 1} "
        f"(max {BATCH_LIMIT} per batch)"
    )

    if not test_n:
        resp = input("\nSubmit? [y/N] ").strip().lower()
        if resp != "y":
            print("Aborted.")
            return

    # Submit in chunks of 10K
    client = anthropic.Anthropic()
    state = {"batch_ids": [], "model": model, "total_rows": len(df)}

    for chunk_start in range(0, len(requests), BATCH_LIMIT):
        chunk = requests[chunk_start : chunk_start + BATCH_LIMIT]
        chunk_num = chunk_start // BATCH_LIMIT + 1

        print(f"\nSubmitting batch {chunk_num} ({len(chunk)} requests)...")
        batch = client.messages.batches.create(requests=chunk)

        state["batch_ids"].append(batch.id)
        print(f"  Batch ID: {batch.id}")
        print(f"  Status: {batch.processing_status}")

    save_state(input_path, state)
    print(f"\nAll batches submitted! State saved to {get_state_path(input_path)}")
    print("Run 'poll' to check status, or 'collect' when done.")


# ─────────────────────────────────────────
# Step 2: Poll
# ─────────────────────────────────────────


def cmd_poll(input_path: str, wait: bool = False):
    state = load_state(input_path)
    if not state["batch_ids"]:
        print("No batches found. Run 'submit' first.")
        return False

    client = anthropic.Anthropic()
    all_done = True

    for batch_id in state["batch_ids"]:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        total = (
            counts.succeeded
            + counts.errored
            + counts.canceled
            + counts.expired
            + counts.processing
        )
        print(f"Batch {batch_id}:")
        print(f"  Status: {batch.processing_status}")
        print(f"  Succeeded: {counts.succeeded}/{total}")
        if counts.errored:
            print(f"  Errored: {counts.errored}")
        if batch.processing_status != "ended":
            all_done = False

    if all_done:
        print("\nAll batches complete! Run 'collect' to download results.")
    else:
        print("\nStill processing...")

    return all_done


# ─────────────────────────────────────────
# Step 3: Collect
# ─────────────────────────────────────────


def cmd_collect(input_path: str, output_path: str):
    state = load_state(input_path)
    if not state["batch_ids"]:
        print("No batches found. Run 'submit' first.")
        return

    client = anthropic.Anthropic()
    results = {}  # photo_id → grounding_output

    for batch_id in state["batch_ids"]:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status != "ended":
            print(
                f"Batch {batch_id} not done yet ({batch.processing_status}). Run 'poll' first."
            )
            return

        print(f"Downloading results from {batch_id}...")
        for result in client.messages.batches.results(batch_id):
            photo_id = result.custom_id

            if result.result.type == "succeeded":
                text = result.result.message.content[0].text.strip()

                # Strip markdown fences
                if text.startswith("```"):
                    text = text.split("\n", 1)[-1]
                if text.endswith("```"):
                    text = text.rsplit("```", 1)[0]
                text = text.strip()

                try:
                    grounding = json.loads(text)
                    for field in ["visual_description", "mood", "color_palette"]:
                        if field not in grounding:
                            grounding[field] = ""
                    results[photo_id] = {
                        "visual_description": grounding["visual_description"],
                        "mood": grounding["mood"],
                        "color_palette": grounding["color_palette"],
                    }
                except json.JSONDecodeError:
                    results[photo_id] = {
                        "visual_description": "",
                        "mood": "",
                        "color_palette": "",
                        "error": f"JSON parse error: {text[:100]}",
                    }
            else:
                results[photo_id] = {
                    "visual_description": "",
                    "mood": "",
                    "color_palette": "",
                    "error": str(result.result.type),
                }

    # Build output in original CSV order
    df = pd.read_csv(input_path)
    if state.get("total_rows") and state["total_rows"] < len(df):
        df = df.head(state["total_rows"])

    output = []
    for _, row in df.iterrows():
        photo_id = str(row["photo_id"])
        grounding = results.get(
            photo_id,
            {
                "visual_description": "",
                "mood": "",
                "color_palette": "",
                "error": "missing from results",
            },
        )

        entry = {"photo_id": photo_id, "grounding_output": {}}
        if "error" in grounding:
            entry["grounding_output"] = {
                "visual_description": grounding["visual_description"],
                "mood": grounding["mood"],
                "color_palette": grounding["color_palette"],
            }
            entry["error"] = grounding["error"]
        else:
            entry["grounding_output"] = grounding

        output.append(entry)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    success = sum(1 for e in output if e["grounding_output"]["visual_description"])
    errors = sum(1 for e in output if "error" in e)
    print(f"\nSaved {len(output)} entries to {output_path}")
    print(f"Success: {success}/{len(output)} ({success / len(output) * 100:.1f}%)")
    print(f"Errors: {errors}")

    # Samples
    print("\n--- Sample outputs ---")
    samples = [e for e in output if e["grounding_output"]["visual_description"]][:3]
    for e in samples:
        print(json.dumps(e, indent=2, ensure_ascii=False))
        print()


# ─────────────────────────────────────────
# Run: submit → poll → collect in one go
# ─────────────────────────────────────────


def cmd_run(input_path: str, output_path: str, model: str, test_n: int | None):
    # Submit
    cmd_submit(input_path, model, test_n)

    # Poll until done
    print("\nWaiting for batches to complete...")
    poll_interval = 30  # seconds
    while True:
        time.sleep(poll_interval)
        done = cmd_poll(input_path)
        if done:
            break
        print(f"  Checking again in {poll_interval}s...\n")

    # Collect
    cmd_collect(input_path, output_path)


# ─────────────────────────────────────────
# CLI
# ─────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="SmartMatch Grounding (Batch API)")
    sub = parser.add_subparsers(dest="command", required=True)

    # submit
    p_submit = sub.add_parser("submit", help="Submit batch requests")
    p_submit.add_argument("--input", "-i", required=True)
    p_submit.add_argument("--model", "-m", default=DEFAULT_MODEL)
    p_submit.add_argument("--test", "-t", type=int, default=None)

    # poll
    p_poll = sub.add_parser("poll", help="Check batch status")
    p_poll.add_argument("--input", "-i", required=True)

    # collect
    p_collect = sub.add_parser("collect", help="Download and save results")
    p_collect.add_argument("--input", "-i", required=True)
    p_collect.add_argument("--output", "-o", default=None)

    # run (all-in-one)
    p_run = sub.add_parser("run", help="Submit, wait, and collect")
    p_run.add_argument("--input", "-i", required=True)
    p_run.add_argument("--output", "-o", default=None)
    p_run.add_argument("--model", "-m", default=DEFAULT_MODEL)
    p_run.add_argument("--test", "-t", type=int, default=None)

    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        exit(1)

    if args.command == "submit":
        cmd_submit(args.input, args.model, args.test)

    elif args.command == "poll":
        cmd_poll(args.input)

    elif args.command == "collect":
        output = args.output or str(
            Path(args.input).parent / "description_grounding_outputs.json"
        )
        cmd_collect(args.input, output)

    elif args.command == "run":
        output = args.output or str(
            Path(args.input).parent / "description_grounding_outputs.json"
        )
        cmd_run(args.input, output, args.model, args.test)


if __name__ == "__main__":
    main()
