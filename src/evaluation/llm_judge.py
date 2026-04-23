"""
src/evaluation/llm_judge.py

LLM-as-Judge Evaluation
-------------------------
Runs 50 diverse queries through the SmartMatch pipeline,
then has Claude rate each mood board on 4 criteria (1-5 scale):

    Relevance       — do the images match the user's intent?
    Visual_Quality  — are the images high quality and aesthetically pleasing?
    Coherence       — do the images work together as a set?
    Aesthetics      — is the overall mood board visually compelling?

Saves results to:
    src/evaluation/results/llm_judge_results.csv
    src/evaluation/results/llm_judge_summary.json

Usage:
    python src/evaluation/llm_judge.py
    python src/evaluation/llm_judge.py --queries 10   # quick test with first 10
    python src/evaluation/llm_judge.py --output my_results.csv
"""

import os
import sys
import json
import time
import argparse
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import anthropic
from dotenv import load_dotenv

load_dotenv()

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── 50 diverse test queries ───────────────────────────────────────────────────
TEST_QUERIES = [
    # Emotional / personal
    "feeling burnt out after a long week",
    "celebrating a new job offer",
    "quiet sunday morning with coffee",
    "missing someone far away",
    "first day of a new chapter",
    "overwhelmed by city life",
    "finally finished my first marathon",
    "cozy rainy day indoors",
    "feeling hopeful about the future",
    "nostalgia for childhood summers",

    # Seasonal / nature
    "spring blossoms and fresh beginnings",
    "golden hour on a quiet beach",
    "first snow of winter",
    "autumn leaves and warm colors",
    "summer adventure road trip",
    "misty mountain morning",
    "tropical paradise escape",
    "desert sunset silence",
    "midnight forest walk",
    "wildflowers in an open field",

    # Lifestyle / aesthetic
    "minimalist workspace inspiration",
    "luxury travel and fine dining",
    "sustainable living and slow fashion",
    "urban street photography at night",
    "soft feminine bedroom aesthetic",
    "industrial loft modern design",
    "japanese zen garden calm",
    "bohemian festival free spirit",
    "dark academia library mood",
    "clean girl morning routine",

    # Commercial / campaign
    "sustainable coffee brand earthy tones",
    "luxury skincare brand soft and pure",
    "athletic performance energy and motion",
    "wedding day romance and elegance",
    "tech startup bold and innovative",
    "children's brand playful and colorful",
    "travel agency adventure and discovery",
    "wellness retreat peace and healing",
    "food brand fresh and appetizing",
    "fashion editorial moody and cinematic",

    # Abstract / conceptual
    "the feeling of being underwater",
    "time passing slowly",
    "connection between strangers",
    "the color of loneliness",
    "chaos turning into order",
    "silence before a storm",
    "the texture of memory",
    "warmth after the cold",
    "beginning and ending at once",
    "freedom with no destination",
]

JUDGE_SYSTEM = """You are an expert visual design critic evaluating AI-generated mood boards.

You will receive:
1. The user's original query
2. A list of images selected for their mood board (described by captions)
3. A board-level summary written by the AI

Rate the mood board on these 4 criteria, each on a scale of 1-5:

Relevance (1-5):
  5 = images perfectly match the user's query intent
  3 = mostly relevant with some mismatches
  1 = images do not match the query at all

Visual_Quality (1-5):
  5 = all images are high quality, well-composed, aesthetically strong
  3 = mixed quality, some strong some weak
  1 = poor quality images, badly composed

Coherence (1-5):
  5 = images work beautifully together as a set — consistent mood, palette, style
  3 = some images fit together but the set feels inconsistent
  1 = images feel random and disconnected

Aesthetics (1-5):
  5 = the overall board is visually compelling and would impress a creative professional
  3 = acceptable but unremarkable
  1 = visually unappealing or generic

Rules:
- Be honest and critical — do not default to high scores
- Base scores on the image descriptions and captions, not hypothetical images
- Consider whether the set tells a coherent visual story
- A board with 9 nearly identical images scores low on Coherence even if individually good

Respond ONLY with valid JSON:
{
  "Relevance": <1-5>,
  "Visual_Quality": <1-5>,
  "Coherence": <1-5>,
  "Aesthetics": <1-5>,
  "reasoning": "2-3 sentence explanation of your ratings"
}"""

JUDGE_USER = """User query: "{query}"

Board summary: "{board_summary}"

Images selected ({n_images} total):
{image_list}

Rate this mood board on the 4 criteria."""


class LLMJudge:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def rate(
        self,
        query: str,
        results: list[dict],
        board_summary: str = "",
    ) -> dict:
        image_lines = []
        for i, r in enumerate(results, 1):
            caption = r.get("caption", r.get("justification", ""))[:120]
            source  = r.get("source", "unknown")
            score   = r.get("score", 0.0)
            image_lines.append(
                f"  {i}. [{source}] score={score:.3f} — {caption}"
            )

        user_msg = JUDGE_USER.format(
            query=query,
            board_summary=board_summary[:300] if board_summary else "Not provided.",
            n_images=len(results),
            image_list="\n".join(image_lines),
        )

        try:
            resp = self.client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=300,
                system=JUDGE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()
            return json.loads(raw)

        except Exception as e:
            print(f"    [Judge] Rating failed: {e}")
            return {
                "Relevance":      0,
                "Visual_Quality": 0,
                "Coherence":      0,
                "Aesthetics":     0,
                "reasoning":      f"Rating failed: {e}",
            }


def run_evaluation(
    n_queries: int = 50,
    output_file: str = "llm_judge_results.csv",
    skip_layout: bool = True,
):
    if skip_layout:
        os.environ["SKIP_LAYOUT"] = "1"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / output_file

    # Use OrchestratorAgent directly (consistent with app.py)
    from agents.orchestrator.agent import OrchestratorAgent
    orchestrator = OrchestratorAgent()

    judge   = LLMJudge()
    queries = TEST_QUERIES[:n_queries]

    print(f"\n{'='*60}")
    print(f"LLM-as-Judge Evaluation")
    print(f"  Queries : {len(queries)}")
    print(f"  Output  : {output_path}")
    print(f"{'='*60}\n")

    fieldnames = [
        "query_id", "query",
        "Relevance", "Visual_Quality", "Coherence", "Aesthetics",
        "mean_score",
        "n_images", "n_retrieved", "n_generated",
        "routing_decision", "top_hybrid_score", "board_summary_len",
        "reasoning",
        "error",
    ]

    all_rows = []

    with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] Query: \"{query}\"")

            row = {
                "query_id": i + 1,
                "query":    query,
                "error":    "",
            }

            try:
                t0     = time.time()
                bundle = orchestrator.run(query=query)
                elapsed = time.time() - t0

                results       = bundle.to_legacy_list()
                board_summary = bundle.board_summary or ""
                routing       = bundle.routing_decision
                top_score     = bundle.top_hybrid_score

                print(f"  Pipeline: {len(results)} images in {elapsed:.1f}s  routing={routing}")

                # Count sources
                n_retrieved = sum(1 for r in results
                                  if r.get("source") in ("hybrid", "graph_expand", "field_text"))
                n_generated = sum(1 for r in results
                                  if r.get("source") in ("dalle3",) or
                                  str(r.get("source", "")).startswith("image_editing"))

                # Judge rating
                print(f"  Rating with Claude...")
                ratings = judge.rate(query, results, board_summary)

                mean_score = round(
                    sum([
                        ratings.get("Relevance",      0),
                        ratings.get("Visual_Quality",  0),
                        ratings.get("Coherence",       0),
                        ratings.get("Aesthetics",      0),
                    ]) / 4, 2
                )

                print(f"  Scores: R={ratings.get('Relevance')} "
                      f"VQ={ratings.get('Visual_Quality')} "
                      f"C={ratings.get('Coherence')} "
                      f"A={ratings.get('Aesthetics')} "
                      f"mean={mean_score}")

                row.update({
                    "Relevance":          ratings.get("Relevance",      0),
                    "Visual_Quality":     ratings.get("Visual_Quality", 0),
                    "Coherence":          ratings.get("Coherence",      0),
                    "Aesthetics":         ratings.get("Aesthetics",     0),
                    "mean_score":         mean_score,
                    "n_images":           len(results),
                    "n_retrieved":        n_retrieved,
                    "n_generated":        n_generated,
                    "routing_decision":   routing,
                    "top_hybrid_score":   round(top_score, 4),
                    "board_summary_len":  len(board_summary),
                    "reasoning":          ratings.get("reasoning", ""),
                })

            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback; traceback.print_exc()
                row.update({
                    "Relevance": 0, "Visual_Quality": 0,
                    "Coherence": 0, "Aesthetics":     0,
                    "mean_score": 0, "n_images": 0,
                    "n_retrieved": 0, "n_generated": 0,
                    "routing_decision": "error",
                    "top_hybrid_score": 0, "board_summary_len": 0,
                    "reasoning": "", "error": str(e),
                })

            writer.writerow(row)
            csvfile.flush()
            all_rows.append(row)

            # Delay between runs to avoid rate limits
            if i < len(queries) - 1:
                time.sleep(2)

    # Summary stats
    valid = [r for r in all_rows if not r["error"]]
    if valid:
        summary = {
            "n_total":             len(all_rows),
            "n_valid":             len(valid),
            "mean_Relevance":      round(sum(r["Relevance"]      for r in valid) / len(valid), 2),
            "mean_Visual_Quality": round(sum(r["Visual_Quality"] for r in valid) / len(valid), 2),
            "mean_Coherence":      round(sum(r["Coherence"]      for r in valid) / len(valid), 2),
            "mean_Aesthetics":     round(sum(r["Aesthetics"]     for r in valid) / len(valid), 2),
            "mean_overall":        round(sum(r["mean_score"]     for r in valid) / len(valid), 2),
            "pct_retrieval":       round(
                sum(1 for r in valid if r["routing_decision"] == "retrieval") / len(valid) * 100, 1),
            "pct_generation":      round(
                sum(1 for r in valid if r["routing_decision"] == "generation") / len(valid) * 100, 1),
        }

        summary_path = RESULTS_DIR / "llm_judge_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"  Valid runs         : {summary['n_valid']}/{summary['n_total']}")
        print(f"  Mean Relevance     : {summary['mean_Relevance']}/5")
        print(f"  Mean Vis Quality   : {summary['mean_Visual_Quality']}/5")
        print(f"  Mean Coherence     : {summary['mean_Coherence']}/5")
        print(f"  Mean Aesthetics    : {summary['mean_Aesthetics']}/5")
        print(f"  Mean Overall       : {summary['mean_overall']}/5")
        print(f"  % retrieval boards : {summary['pct_retrieval']}%")
        print(f"  % generation boards: {summary['pct_generation']}%")
        print(f"\n  Results → {output_path}")
        print(f"  Summary → {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-Judge evaluation")
    parser.add_argument("--queries",        type=int, default=50)
    parser.add_argument("--output",         type=str, default="llm_judge_results.csv")
    parser.add_argument("--no-skip-layout", action="store_true")
    args = parser.parse_args()

    run_evaluation(
        n_queries=args.queries,
        output_file=args.output,
        skip_layout=not args.no_skip_layout,
    )