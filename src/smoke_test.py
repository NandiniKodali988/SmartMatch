"""
src/smoke_test.py

Smoke tests for the SmartMatch pipeline.

Run:
    cd /path/to/project
    pytest src/smoke_test.py -v

    # Skip slow tests (model loading / API calls):
    pytest src/smoke_test.py -v -m "not slow"

    # Only environment checks:
    pytest src/smoke_test.py -v -m "env"

Markers:
    env   — no API calls, no model loading
    slow  — loads SigLIP model (10-30s) or makes real API calls
"""

import os
import sys
import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent
BASE_DIR = SRC_DIR.parent
sys.path.insert(0, str(SRC_DIR))

# Load .env so API keys are available during tests
load_dotenv(BASE_DIR / ".env")

# ── Shared paths ──────────────────────────────────────────────────────────────
IMAGE_EMB_PATH  = SRC_DIR / "data/embeddings/image_embeddings.npy"
FIELD_EMB_PATH  = SRC_DIR / "data/embeddings/field_embeddings.npz"
DATASET_CSV     = SRC_DIR / "data/processed/dataset_clean.csv"
GROUNDING_JSON  = SRC_DIR / "data/processed/description_grounding_outputs.json"

# ── Test queries ──────────────────────────────────────────────────────────────
# Diverse set: emotional, concrete-visual, abstract, multi-word
TEST_QUERIES = [
    "feeling burnt out after a long week",
    "walking alone through a rainy city street at night",
    "a quiet Sunday morning with coffee and sunlight",
    "celebrating a new job offer with friends",
    "the calm before a thunderstorm",
    "nostalgia for childhood summers",
    "overwhelmed by city life",
    "a peaceful mountain hike at golden hour",
]

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def siglip_agent():
    """Load SigLIP agent once per module (slow: model download + load)."""
    from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
    return SiglipImageRetrievalAgent(top_k=3)


@pytest.fixture(scope="module")
def field_agent():
    """Load FieldTextRetrieval agent once per module."""
    from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
    return FieldTextRetrievalAgent(top_k=3)


@pytest.fixture(scope="module")
def grounding_agent():
    from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
    return QwenVisualGroundingAgent()


@pytest.fixture(scope="module")
def justification_agent():
    from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
    return QwenJustificationAgent()


MOCK_GROUNDING = {
    "visual_description": "A person sitting alone at a desk late at night, surrounded by dim lamp light",
    "scene":              "indoor, office, nighttime",
    "mood":               "tired, melancholic, introspective",
    "style":              "cinematic, documentary",
    "lighting":           "warm desk lamp, low ambient light",
    "color_palette":      "muted amber, deep navy, soft grey",
    "intent":             "editorial",
}

MOCK_IMAGE = {
    "photo_id":   "test_001",
    "image_url":  "https://images.unsplash.com/photo-test",
    "caption":    "a person sitting alone late at night with a lamp",
    "score":      0.82,
    "source":     "unsplash",
    "justification": "",
}


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 0 — Environment & data files (no API, no model)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.env
class TestEnvironment:

    def test_image_embeddings_exist(self):
        assert IMAGE_EMB_PATH.exists(), f"Missing: {IMAGE_EMB_PATH}"

    def test_field_embeddings_exist(self):
        assert FIELD_EMB_PATH.exists(), f"Missing: {FIELD_EMB_PATH}"

    def test_dataset_csv_exists(self):
        assert DATASET_CSV.exists(), f"Missing: {DATASET_CSV}"

    def test_grounding_json_exists(self):
        assert GROUNDING_JSON.exists(), f"Missing: {GROUNDING_JSON}"

    def test_anthropic_api_key_set(self):
        assert os.getenv("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set"

    def test_openai_api_key_set(self):
        assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY not set"

    def test_image_embeddings_shape(self):
        emb = np.load(IMAGE_EMB_PATH)
        assert emb.ndim == 2, f"Expected 2D array, got {emb.ndim}D"
        assert emb.shape[1] == 768, f"Expected 768 dims, got {emb.shape[1]}"
        print(f"\n  image_embeddings shape: {emb.shape}")

    def test_field_embeddings_keys_and_shape(self):
        npz = np.load(FIELD_EMB_PATH)
        required_keys = {"visual_description", "mood", "color_palette"}
        assert required_keys.issubset(set(npz.keys())), \
            f"Missing keys: {required_keys - set(npz.keys())}"
        for key in required_keys:
            assert npz[key].ndim == 2, f"{key}: expected 2D"
        print(f"\n  field_embeddings shape (per field): {npz['visual_description'].shape}")

    def test_embedding_count_matches_grounding_json(self):
        """
        CRITICAL: SigLIP and FieldText must cover the same number of records
        for merge_with_siglip() to work. If this fails, the hybrid pipeline
        will crash with a shape mismatch error.
        """
        siglip_n = np.load(IMAGE_EMB_PATH).shape[0]
        field_n  = np.load(FIELD_EMB_PATH)["visual_description"].shape[0]
        with open(GROUNDING_JSON) as f:
            json_n = len(json.load(f))

        print(f"\n  SigLIP embeddings : {siglip_n}")
        print(f"  Field embeddings  : {field_n}")
        print(f"  Grounding JSON    : {json_n}")

        assert siglip_n == field_n, (
            f"Shape mismatch! SigLIP has {siglip_n} records but "
            f"field_embeddings has {field_n}. merge_with_siglip() will crash."
        )

    def test_dataset_csv_has_required_columns(self):
        import pandas as pd
        df = pd.read_csv(DATASET_CSV, nrows=5)
        required = {"photo_id", "photo_image_url"}
        missing = required - set(df.columns)
        assert not missing, f"dataset_clean.csv missing columns: {missing}"


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 1 — SigLIP Agent (model load, no external API)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestSiglipAgent:

    def test_init_loads_embeddings(self, siglip_agent):
        assert siglip_agent.image_embeddings is not None
        assert siglip_agent.image_embeddings.shape[1] == 768

    def test_normalized_embeddings_unit_norm(self, siglip_agent):
        """Pre-normalization at init — each non-zero row should have norm ≈ 1.
        Zero vectors are expected for images that failed to download."""
        norms = np.linalg.norm(siglip_agent.image_embeddings_normalized, axis=1)
        valid = norms > 1e-6  # skip zero vectors (failed downloads)
        print(f"\n  zero vectors (failed downloads): {(~valid).sum()}/{len(norms)}")
        assert valid.sum() > 0, "All embeddings are zero vectors!"
        assert np.allclose(norms[valid], 1.0, atol=1e-5), \
            f"Max norm deviation on valid rows: {np.abs(norms[valid] - 1.0).max():.6f}"

    def test_embed_text_shape_and_norm(self, siglip_agent):
        vec = siglip_agent.embed_text("a quiet rainy evening")
        assert vec.shape == (768,), f"Expected (768,), got {vec.shape}"
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-5, "Output vector not normalized"

    def test_embed_text_long_input_no_crash(self, siglip_agent):
        """Truncation (max_length=64) should silently handle very long text."""
        long_text = "beautiful sunset over the mountains " * 50
        vec = siglip_agent.embed_text(long_text)
        assert vec.shape == (768,)

    def test_retrieve_all_scores_shape(self, siglip_agent):
        scores = siglip_agent.retrieve_all_scores(MOCK_GROUNDING)
        n = siglip_agent.image_embeddings.shape[0]
        assert scores.shape == (n,), f"Expected ({n},), got {scores.shape}"
        assert scores.min() >= -1.01 and scores.max() <= 1.01, \
            "Cosine scores out of [-1, 1] range"

    def test_retrieve_returns_top_k(self, siglip_agent):
        results = siglip_agent.retrieve(MOCK_GROUNDING)
        assert len(results) == siglip_agent.top_k
        assert all("photo_id" in r and "image_url" in r and
                   "caption" in r and "score" in r for r in results)

    def test_retrieve_scores_descending(self, siglip_agent):
        results = siglip_agent.retrieve(MOCK_GROUNDING)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"

    @pytest.mark.parametrize("query", TEST_QUERIES[:4])
    def test_retrieve_multiple_queries(self, siglip_agent, query):
        grounding = {**MOCK_GROUNDING, "visual_description": query}
        results = siglip_agent.retrieve(grounding)
        assert len(results) > 0
        assert results[0]["score"] >= 0.0
        print(f"\n  Query: '{query[:50]}...' → top score: {results[0]['score']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 2 — Visual Grounding Agent (Claude API)
# ══════════════════════════════════════════════════════════════════════════════

GROUNDING_REQUIRED_KEYS = {
    "visual_description", "scene", "mood", "style",
    "lighting", "color_palette", "intent",
}
VALID_INTENTS = {"social", "artistic", "editorial", "commercial", "personal"}

@pytest.mark.slow
class TestGroundingAgent:

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_grounding_returns_all_fields(self, grounding_agent, query):
        result = grounding_agent.run(query)
        missing = GROUNDING_REQUIRED_KEYS - result.keys()
        assert not missing, f"Query '{query}' missing fields: {missing}"
        assert result["visual_description"], "visual_description is empty"
        print(f"\n  [{query[:40]}]")
        print(f"    visual_description: {result['visual_description'][:70]}...")
        print(f"    intent: {result['intent']}  mood: {result['mood'][:40]}")

    @pytest.mark.parametrize("query", TEST_QUERIES)
    def test_grounding_intent_is_valid(self, grounding_agent, query):
        result = grounding_agent.run(query)
        assert result.get("intent") in VALID_INTENTS, \
            f"Unexpected intent '{result.get('intent')}' for query: '{query}'"

    def test_grounding_empty_string_no_crash(self, grounding_agent):
        """Empty input should fall back gracefully, not raise."""
        result = grounding_agent.run("")
        assert isinstance(result, dict)
        assert "visual_description" in result


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 3 — Generation Agent (unit-level, no API)
# ══════════════════════════════════════════════════════════════════════════════

class TestGenerationAgentUnit:

    @pytest.fixture
    def gen_agent(self):
        from agents.generation.agent import GenerationAgent
        return GenerationAgent()

    def test_build_generation_prompt_contains_description(self, gen_agent):
        prompt = gen_agent._build_generation_prompt(MOCK_GROUNDING)
        assert "sitting alone" in prompt or "desk" in prompt, \
            f"Prompt doesn't include visual_description content: {prompt[:100]}"

    def test_build_generation_prompt_includes_mood_and_scene(self, gen_agent):
        prompt = gen_agent._build_generation_prompt(MOCK_GROUNDING)
        assert "tired" in prompt or "melancholic" in prompt, \
            "Prompt missing mood content"
        assert "office" in prompt or "indoor" in prompt, \
            "Prompt missing scene content"

    def test_build_generation_prompt_empty_grounding(self, gen_agent):
        """Empty grounding should return 'A photograph', not crash."""
        prompt = gen_agent._build_generation_prompt({})
        assert isinstance(prompt, str)
        assert len(prompt) > 0

    def test_select_mode_no_uploads_forced_dalle3(self, gen_agent):
        """Safety guard: has_uploads=False must always return 'dalle3'."""
        with patch.object(gen_agent.claude.messages, "create") as mock_create:
            mock_create.return_value = MagicMock(
                content=[MagicMock(text='{"mode": "restyle", "prompt": "some prompt"}')]
            )
            mode, prompt = gen_agent._select_mode(
                user_text="test",
                grounding_output=MOCK_GROUNDING,
                image_paths=None,
                style_ref_path=None,
                has_uploads=False,
            )
        assert mode == "dalle3", \
            f"Safety guard failed: mode should be 'dalle3' but got '{mode}'"

    def test_blend_requires_2_images(self, gen_agent, tmp_path):
        img_path = tmp_path / "test.png"
        from PIL import Image
        Image.new("RGB", (64, 64), color="red").save(img_path)
        with pytest.raises(ValueError, match="blend requires at least 2"):
            gen_agent._blend([str(img_path)], "test prompt")

    def test_composite_requires_style_ref(self, gen_agent, tmp_path):
        img1 = tmp_path / "a.png"
        img2 = tmp_path / "b.png"
        from PIL import Image
        Image.new("RGB", (64, 64), color="blue").save(img1)
        Image.new("RGB", (64, 64), color="green").save(img2)
        with pytest.raises(ValueError, match="composite requires a style_ref_path"):
            gen_agent._composite([str(img1), str(img2)], "test", style_ref_path=None)


# # ══════════════════════════════════════════════════════════════════════════════
# # LAYER 3b — Justification Agent (Claude API)
# # ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestJustificationAgent:

    def test_justification_appends_field(self, justification_agent):
        images = [MOCK_IMAGE.copy()]
        results = justification_agent.run("feeling burnt out", images)
        assert len(results) == 1
        assert "justification" in results[0]
        assert isinstance(results[0]["justification"], str)
        print(f"\n  justification: {results[0]['justification'][:100]}...")

    def test_justification_preserves_original_fields(self, justification_agent):
        images = [MOCK_IMAGE.copy()]
        results = justification_agent.run("rainy city street", images)
        r = results[0]
        assert r["photo_id"] == MOCK_IMAGE["photo_id"]
        assert r["image_url"] == MOCK_IMAGE["image_url"]
        assert r["score"] == MOCK_IMAGE["score"]

    def test_justification_empty_image_list(self, justification_agent):
        results = justification_agent.run("anything", [])
        assert results == []


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 4 — Pipeline end-to-end
# ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestPipelineEndToEnd:

    def test_branch_b_retrieval_path(self):
        """Force retrieval by setting threshold=0.0 — top score always passes."""
        with patch.dict(os.environ, {"HYBRID_THRESHOLD": "0.0"}):
            # Re-import pipeline so it picks up the new env var
            import importlib
            import pipeline.run_pipeline as rp
            importlib.reload(rp)

            results = rp.run_pipeline("a quiet morning in the city")

        assert len(results) > 0, "Pipeline returned no results"
        assert all("image_url" in r for r in results)
        assert all("justification" in r for r in results)
        assert all(r.get("source") in {"hybrid", "unsplash", "field_text"}
                   for r in results), f"Unexpected source: {[r.get('source') for r in results]}"
        print(f"\n  top result: {results[0]['photo_id']}  score={results[0]['score']:.4f}")
        print(f"  justification: {results[0]['justification'][:80]}...")

    def test_branch_b_fallback_to_generation(self):
        """Force DALL·E 3 fallback by setting threshold=1.0 — no retrieval can pass."""
        with patch.dict(os.environ, {"HYBRID_THRESHOLD": "1.0", "TOP_K": "1"}):
            import importlib
            import pipeline.run_pipeline as rp
            importlib.reload(rp)

            results = rp.run_pipeline("the feeling before a thunderstorm")

        assert len(results) > 0, "Pipeline returned no results"
        assert all("image_url" in r for r in results)
        assert all("justification" in r for r in results)
        assert all(r.get("source") == "dalle3" for r in results), \
            f"Expected source='dalle3', got: {[r.get('source') for r in results]}"
        print(f"\n  generated image_url: {results[0]['image_url'][:70]}...")

    @pytest.mark.parametrize("query", [
        "feeling lost and uncertain about the future",
        "celebrating a new job offer with friends",
        "overwhelming noise of city life",
    ])
    def test_branch_b_multiple_queries(self, query):
        """Multiple diverse queries should all complete without crashing."""
        with patch.dict(os.environ, {"HYBRID_THRESHOLD": "0.0", "TOP_K": "2"}):
            import importlib
            import pipeline.run_pipeline as rp
            importlib.reload(rp)

            results = rp.run_pipeline(query)

        assert len(results) > 0, f"No results for query: '{query}'"
        assert all("justification" in r for r in results)
        print(f"\n  '{query[:45]}' → {len(results)} result(s), "
              f"top score={results[0]['score']:.4f}")

    def test_result_schema_completeness(self):
        """Every result must have the full required schema."""
        with patch.dict(os.environ, {"HYBRID_THRESHOLD": "0.0", "TOP_K": "3"}):
            import importlib
            import pipeline.run_pipeline as rp
            importlib.reload(rp)

            results = rp.run_pipeline("a peaceful mountain hike at golden hour")

        required_keys = {"photo_id", "image_url", "caption", "score", "source", "justification"}
        for i, r in enumerate(results):
            missing = required_keys - r.keys()
            assert not missing, f"Result {i} missing keys: {missing}"


# # ══════════════════════════════════════════════════════════════════════════════
# # LAYER 5 — Hybrid retrieval shape compatibility
# # ══════════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestHybridRetrieval:

    def test_merge_with_siglip_shape_compatibility(self, siglip_agent, field_agent):
        """
        SigLIP returns (N,) and FieldText returns (M,).
        If N != M, merge_with_siglip crashes. This test catches that bug.
        """
        siglip_scores = siglip_agent.retrieve_all_scores(MOCK_GROUNDING)
        text_scores, _ = field_agent._score_all(MOCK_GROUNDING)

        assert siglip_scores.shape == text_scores.shape, (
            f"SHAPE MISMATCH: SigLIP={siglip_scores.shape}, "
            f"FieldText={text_scores.shape}. "
            f"merge_with_siglip() will fail. "
            f"You need to either expand image_embeddings to 25k or "
            f"restrict field_text to 200 records."
        )

    def test_merge_with_siglip_returns_top_k(self, siglip_agent, field_agent):
        results = field_agent.merge_with_siglip(MOCK_GROUNDING, siglip_agent)
        assert len(results) == field_agent.top_k
        for r in results:
            assert "score" in r and "siglip_score" in r and "text_score" in r
            assert r["source"] == "hybrid"

    def test_hybrid_scores_are_weighted_correctly(self, siglip_agent, field_agent):
        """Spot-check: final_score should equal SIGLIP_WEIGHT * siglip + TEXT_WEIGHT * text.
        Reads weights from the agent directly to stay in sync with agent defaults."""
        from agents.field_text_retrieval.agent import SIGLIP_WEIGHT, TEXT_WEIGHT
        results = field_agent.merge_with_siglip(MOCK_GROUNDING, siglip_agent)
        for r in results:
            expected = SIGLIP_WEIGHT * r["siglip_score"] + TEXT_WEIGHT * r["text_score"]
            assert abs(r["score"] - expected) < 1e-4, \
                f"Hybrid score {r['score']:.6f} != expected {expected:.6f}"

    @pytest.mark.parametrize("query", TEST_QUERIES[:3])
    def test_hybrid_retrieval_multiple_queries(self, siglip_agent, field_agent, query):
        grounding = {**MOCK_GROUNDING, "visual_description": query, "mood": query}
        results = field_agent.merge_with_siglip(grounding, siglip_agent)
        assert len(results) > 0
        assert results[0]["score"] >= results[-1]["score"], "Results not sorted by score"
        print(f"\n  '{query[:45]}' → top hybrid={results[0]['score']:.4f}  "
              f"siglip={results[0]['siglip_score']:.4f}  "
              f"text={results[0]['text_score']:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# LAYER 6 — Evaluation Framework
# Runs all TEST_QUERIES through the full pipeline and records scores.
# Produces a human-readable report of preliminary results.
# ══════════════════════════════════════════════════════════════════════════════

EVAL_QUERIES = [
    "feeling burnt out after a long week",
    "walking alone through a rainy city street at night",
    "a quiet Sunday morning with coffee and sunlight",
    "celebrating a new job offer with friends",
    "the calm before a thunderstorm",
    "nostalgia for childhood summers",
    "overwhelmed by city life",
    "a peaceful mountain hike at golden hour",
]


@pytest.mark.slow
class TestEvaluation:
    """
    Evaluation framework for Milestone 3.
    Runs all queries through the pipeline, records scores and decisions,
    and prints a structured report of preliminary results.
    """

    @pytest.fixture(scope="class")
    def eval_results(self, siglip_agent, field_agent):
        """
        Run all EVAL_QUERIES through the FULL pipeline:
          1. Hybrid retrieval to get scores
          2. If score >= threshold → use retrieval results
          3. If score <  threshold → actually call GenerationAgent (DALL·E 3)
          4. Always run Justification on final results

        Generation queries: ~12s rate limit each, plan accordingly.
        """
        from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
        from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
        from agents.generation.agent import GenerationAgent

        grounding_agent    = QwenVisualGroundingAgent()
        justification_agent = QwenJustificationAgent()
        generation_agent   = GenerationAgent()
        HYBRID_THRESHOLD   = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
        TOP_K              = int(os.getenv("TOP_K", "3"))

        records = []
        for query in EVAL_QUERIES:
            print(f"\n[Eval] Running full pipeline for: '{query}'")
            grounding  = grounding_agent.run(query)
            candidates = field_agent.merge_with_siglip(grounding, siglip_agent)
            top        = candidates[0] if candidates else {}
            top_score  = top.get("score", 0.0)
            decision   = "retrieval" if top_score >= HYBRID_THRESHOLD else "generation"

            if decision == "retrieval":
                print(f"[Eval]   → retrieval  (score={top_score:.4f} >= {HYBRID_THRESHOLD})")
                final_images = candidates[:TOP_K]
            else:
                print(f"[Eval]   → generation (score={top_score:.4f} < {HYBRID_THRESHOLD})")
                final_images = generation_agent.run(
                    grounding_output=grounding,
                    n=1,
                    user_text=query,
                    uploaded_image_paths=None,
                    style_ref_path=None,
                    siglip_agent=siglip_agent,
                )

            final_images = justification_agent.run(query, final_images)

            records.append({
                "query":        query,
                "grounding":    grounding,
                "top_score":    top_score,
                "siglip_score": top.get("siglip_score", 0.0),
                "text_score":   top.get("text_score",   0.0),
                "decision":     decision,
                "candidates":   candidates,       # raw hybrid top-k (always available)
                "final_images": final_images,     # what pipeline actually returned
            })

        return records

    # ── Individual query checks ───────────────────────────────────────────────

    @pytest.mark.parametrize("query", EVAL_QUERIES)
    def test_eval_query_returns_results(self, eval_results, query):
        """Every query must return at least one final image."""
        record = next(r for r in eval_results if r["query"] == query)
        assert len(record["final_images"]) > 0, \
            f"No final images for: '{query}' (decision={record['decision']})"

    @pytest.mark.parametrize("query", EVAL_QUERIES)
    def test_eval_scores_in_valid_range(self, eval_results, query):
        """Cosine similarity ranges from -1 to 1; hybrid/text scores in [0, 1]."""
        record = next(r for r in eval_results if r["query"] == query)
        assert -1.0 <= record["siglip_score"] <= 1.0   # cosine: [-1, 1]
        assert  0.0 <= record["text_score"]   <= 1.0   # normalized text sim: [0, 1]
        assert  0.0 <= record["top_score"]    <= 1.0   # weighted hybrid: [0, 1]

    @pytest.mark.parametrize("query", EVAL_QUERIES)
    def test_eval_decision_is_valid(self, eval_results, query):
        """Decision must be either 'retrieval' or 'generation'."""
        record = next(r for r in eval_results if r["query"] == query)
        assert record["decision"] in {"retrieval", "generation"}

    # ── Aggregate report ──────────────────────────────────────────────────────

    def test_eval_print_full_report(self, eval_results):
        """
        Prints a full evaluation report.
        Always passes — the report is for human review.
        """
        HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
        retrieval_count  = sum(1 for r in eval_results if r["decision"] == "retrieval")
        generation_count = len(eval_results) - retrieval_count
        avg_hybrid  = sum(r["top_score"]    for r in eval_results) / len(eval_results)
        avg_siglip  = sum(r["siglip_score"] for r in eval_results) / len(eval_results)
        avg_text    = sum(r["text_score"]   for r in eval_results) / len(eval_results)

        sep = "═" * 72

        print(f"\n\n{sep}")
        print("  SMARTMATCH — PRELIMINARY EVALUATION RESULTS")
        print(f"  Queries: {len(eval_results)}  |  Threshold: {HYBRID_THRESHOLD}")
        print(sep)

        print(f"\n{'Query':<45} {'Hybrid':>7} {'SigLIP':>7} {'Text':>7}  Decision")
        print("─" * 72)
        for r in eval_results:
            flag = "✓" if r["decision"] == "retrieval" else "↳ gen"
            print(
                f"  {r['query'][:43]:<43} "
                f"{r['top_score']:>7.4f} "
                f"{r['siglip_score']:>7.4f} "
                f"{r['text_score']:>7.4f}  {flag}"
            )

        print("─" * 72)
        print(
            f"  {'Average':<43} "
            f"{avg_hybrid:>7.4f} "
            f"{avg_siglip:>7.4f} "
            f"{avg_text:>7.4f}"
        )

        print(f"\n  Routing:  retrieval={retrieval_count}  generation={generation_count}  "
              f"(threshold={HYBRID_THRESHOLD})")

        print(f"\n  Full pipeline results per query:")
        print("─" * 72)
        for r in eval_results:
            decision = r["decision"]
            print(f"\n  [{r['query'][:60]}]  → {decision}")

            if decision == "retrieval":
                for i, c in enumerate(r["candidates"][:3], 1):
                    print(
                        f"    {i}. {c.get('photo_id','?'):<20}  "
                        f"hybrid={c['score']:.4f}  "
                        f"siglip={c.get('siglip_score',0):.4f}  "
                        f"text={c.get('text_score',0):.4f}"
                    )
                    print(f"       caption: {c.get('caption','')[:70]}")
                    justification = r["final_images"][i-1].get("justification", "") if i-1 < len(r["final_images"]) else ""
                    if justification:
                        print(f"       justification: {justification[:80]}")
            else:
                for i, img in enumerate(r["final_images"], 1):
                    if not img.get("image_url"):
                        print(f"    {i}. ⚠ BLOCKED by DALL·E 3 content policy")
                        continue
                    print(f"    {i}. {img.get('photo_id','?'):<20}  score={img.get('score',0):.4f}  source={img.get('source')}")
                    print(f"       prompt : {img.get('caption','')[:70]}")
                    print(f"       saved  : {img.get('local_path', 'not saved')}")
                    justification = img.get("justification", "")
                    if justification:
                        print(f"       justification: {justification[:80]}")

        print(f"\n{sep}\n")
        assert True  # report is for human review
