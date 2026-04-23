"""
src/agents/field_text_retrieval/agent.py
 
Field-level Text Retrieval Agent
----------------------------------
Computes per-field cosine similarity between a query grounding output
and pre-embedded dataset grounding outputs (visual_description / mood / color_palette).
 
Uses OpenAI text-embedding-3-large embeddings cached in field_embeddings.npz.
photo_image_url is looked up from dataset_clean.csv by photo_id.
 
Hybrid retrieval merges full-dataset SigLIP scores with full-dataset text scores
before ranking, so both methods compete on equal footing.
"""
 
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
 
load_dotenv()
 
# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).resolve().parents[2]
DATASET_JSON    = BASE_DIR / "data/processed/description_grounding_outputs.json"
EMBEDDING_NPZ   = BASE_DIR / "data/embeddings/field_embeddings.npz"
METADATA_CSV    = BASE_DIR / "data/processed/dataset_clean.csv"
 
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
 
# Per-field weights — must sum to 1.0
FIELD_WEIGHTS = {
    "visual_description": 1/2,
    "mood":               1/3,
    "color_palette":      1/6,
}
 
SIGLIP_WEIGHT = float(os.getenv("SIGLIP_WEIGHT", "0.3"))
TEXT_WEIGHT   = float(os.getenv("TEXT_WEIGHT",   "0.7"))
 
 
class FieldTextRetrievalAgent:
    """
    Retrieves images by comparing query grounding output fields
    against pre-embedded dataset grounding outputs.
 
    Usage
    -----
        agent = FieldTextRetrievalAgent(top_k=5)
 
        # Standalone text-only retrieval
        results = agent.retrieve(grounding_output)
 
        # Full-dataset hybrid merge with SigLIP
        results = agent.merge_with_siglip(grounding_output, siglip_agent)
    """
 
    def __init__(self, top_k: int = 5):
        self.top_k  = top_k
        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
 
        # Load dataset grounding records
        print("[FieldTextRetrieval] Loading dataset...")
        with open(DATASET_JSON, encoding="utf-8") as f:
            self.dataset = json.load(f)
        print(f"[FieldTextRetrieval] {len(self.dataset)} records loaded.")
 
        # Build photo_id → photo_image_url lookup from dataset_clean.csv
        print("[FieldTextRetrieval] Loading image URL lookup from CSV...")
        meta_df = pd.read_csv(METADATA_CSV, usecols=["photo_id", "photo_image_url"], on_bad_lines="skip")
        self.url_lookup = dict(zip(
            meta_df["photo_id"].astype(str),
            meta_df["photo_image_url"].astype(str),
        ))
        print(f"[FieldTextRetrieval] URL lookup ready — {len(self.url_lookup)} entries.")
 
        # Load pre-computed field embeddings
        print("[FieldTextRetrieval] Loading field embeddings...")
        # npz = np.load(EMBEDDING_NPZ)
        npz = np.load(EMBEDDING_NPZ, allow_pickle=True)
        self.field_embeddings = {field: npz[field] for field in FIELD_WEIGHTS}
        shape = next(iter(self.field_embeddings.values())).shape
        print(f"[FieldTextRetrieval] Embeddings loaded — shape per field: {shape}")
 
    # ── Public interface ──────────────────────────────────────────────────────
 
    def retrieve(self, grounding_output: dict) -> list[dict]:
        """
        Text-only retrieval over the full dataset.
        Returns top-k results ranked by weighted field similarity.
        """
        scores, field_scores_matrix = self._score_all(grounding_output)
        return self._build_results(scores, field_scores_matrix, source="field_text")
 
    def merge_with_siglip(
        self,
        grounding_output: dict,
        siglip_agent,                  # SiglipImageRetrievalAgent instance
        exclude_ids: set | None = None,  # photo_ids to skip (e.g. already liked)
        n: int | None = None,            # override top_k for this call
    ) -> list[dict]:
        """
        Full-dataset hybrid retrieval.

        Computes SigLIP scores and text scores over ALL records independently,
        then merges them before ranking — so both methods compete on equal footing.

            final_score = SIGLIP_WEIGHT * siglip_score + TEXT_WEIGHT * text_score

        Args:
            grounding_output : dict from QwenVisualGroundingAgent
            siglip_agent     : initialized SiglipImageRetrievalAgent
            exclude_ids      : set of photo_ids to exclude from results (e.g. liked images)
            n                : number of results to return (default: self.top_k)

        Returns:
            top-k result dicts with siglip_score, text_score, and final score
        """
        exclude = set(exclude_ids) if exclude_ids else set()
        k       = n if n is not None else self.top_k

        # Score all N records with both methods independently
        siglip_scores             = siglip_agent.retrieve_all_scores(grounding_output)  # (N,)
        text_scores, field_matrix = self._score_all(grounding_output)                   # (N,)

        # Weighted merge over full dataset
        final_scores = SIGLIP_WEIGHT * siglip_scores + TEXT_WEIGHT * text_scores        # (N,)

        # Rank all, then filter excluded ids until we have k results
        ranked_all = np.argsort(final_scores)[::-1]
        results = []
        for idx in ranked_all:
            if len(results) >= k:
                break
            row      = self.dataset[idx]
            photo_id = str(row.get("photo_id", f"idx_{idx}"))
            if photo_id in exclude:
                continue
            grounding_row = row.get("grounding_output", {})
            results.append({
                "photo_id":     photo_id,
                "image_url":    self.url_lookup.get(photo_id, ""),
                "caption":      grounding_row.get("visual_description", ""),
                "score":        float(final_scores[idx]),
                "siglip_score": float(siglip_scores[idx]),
                "text_score":   float(text_scores[idx]),
                "source":       "hybrid",
                "field_scores": {
                    field: float(field_matrix[field][idx])
                    for field in FIELD_WEIGHTS
                },
            })

        return results
 
    # ── Internal ──────────────────────────────────────────────────────────────
 
    def _embed_text(self, text: str) -> np.ndarray:
        """Embed a single string and return an L2-normalized vector."""
        text = text.strip() if text.strip() else "no description"
        resp = self.client.embeddings.create(model=EMBEDDING_MODEL, input=[text])
        vec  = np.array(resp.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec
 
    def _score_all(self, grounding_output: dict) -> tuple[np.ndarray, dict]:
        """
        Compute weighted text similarity scores for ALL dataset records.
 
        Returns:
            final_scores      (N,)          -- weighted sum across fields
            field_scores_dict {field: (N,)} -- per-field scores for debugging
        """
        N = next(iter(self.field_embeddings.values())).shape[0]
        final_scores      = np.zeros(N, dtype=np.float32)
        field_scores_dict = {}
 
        for field, weight in FIELD_WEIGHTS.items():
            query_text = str(grounding_output.get(field, "") or "").strip()
 
            if not query_text:
                field_scores_dict[field] = np.zeros(N, dtype=np.float32)
                continue
 
            q_emb        = self._embed_text(query_text)
            field_scores = self.field_embeddings[field] @ q_emb  # (N,)
            final_scores += weight * field_scores
            field_scores_dict[field] = field_scores
 
        return final_scores, field_scores_dict
 
    def _build_results(
        self,
        scores: np.ndarray,
        field_scores_matrix: dict,
        source: str,
    ) -> list[dict]:
        """Build result dicts for the top-k records."""
        ranked  = np.argsort(scores)[::-1][:self.top_k]
        results = []
 
        for idx in ranked:
            row      = self.dataset[idx]
            photo_id = str(row.get("photo_id", f"idx_{idx}"))
            grounding = row.get("grounding_output", {})
 
            results.append({
                "photo_id":    photo_id,
                "image_url":   self.url_lookup.get(photo_id, ""),
                "caption":     grounding.get("visual_description", ""),
                "score":       float(scores[idx]),
                "source":      source,
                "field_scores": {
                    field: float(field_scores_matrix[field][idx])
                    for field in FIELD_WEIGHTS
                },
            })
 
        return results