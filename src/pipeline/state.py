"""
src/pipeline/state.py

Pydantic models for the SmartMatch pipeline.

All agents accept and return these models instead of raw dicts.
This makes field mismatches fail loudly at parse time rather than
silently breaking downstream agents.

Models
------
    GroundingOutput   — output from QwenVisualGroundingAgent
    ImageResult       — a single image candidate (retrieval or generated)
    PipelineState     — mutable state flowing through OrchestratorAgent
    MoodBoardBundle   — final output artifact returned to the caller
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field


# ── 1. GroundingOutput ────────────────────────────────────────────────────────

class GroundingOutput(BaseModel):
    """
    Structured visual concept produced by QwenVisualGroundingAgent.

    All fields are optional strings so the pipeline can degrade gracefully
    when the grounding model omits a field (e.g. on a short / vague query).
    """
    visual_description: str = ""
    scene:              str = ""
    mood:               str = ""
    style:              str = ""
    lighting:           str = ""
    color_palette:      str = ""
    intent:             str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "GroundingOutput":
        """Convert a raw dict (from the grounding agent) to a GroundingOutput."""
        return cls(**{k: v for k, v in d.items() if k in cls.model_fields})

    def to_dict(self) -> dict:
        """Convert back to a plain dict for agents that still expect dicts."""
        return self.model_dump()


# ── 2. ImageResult ────────────────────────────────────────────────────────────

class ImageResult(BaseModel):
    """
    A single image candidate at any stage of the pipeline.

    Fields are populated incrementally:
      - photo_id / image_url / caption / score come from retrieval
      - siglip_score / text_score / field_scores come from hybrid retrieval
      - graph_score comes from GraphRAGAgent.rerank()
      - verified / verification_reason come from MultimodalVerificationAgent
      - coherence_rank / board_reason come from CoherenceAgent
      - justification comes from QwenJustificationAgent
      - local_path comes from GenerationAgent (downloaded DALL·E images)
    """
    photo_id:            str
    image_url:           str   = ""
    local_path:          Optional[str] = None
    caption:             str   = ""
    score:               float = 0.0
    source: Literal[
        "hybrid",
        "field_text",
        "graph_expand",
        "dalle3",
        "image_editing/inpaint",
        "image_editing/restyle",
        "image_editing/blend",
        "image_editing/collage",
        "image_editing/composite",
        "unknown",
    ] = "unknown"

    # Retrieval detail scores
    siglip_score:        float = 0.0
    text_score:          float = 0.0
    graph_score:         float = 0.0
    field_scores:        dict  = Field(default_factory=dict)

    # Verification
    verified:            bool  = False
    verification_reason: str   = ""

    # Coherence selection
    coherence_rank:      Optional[int] = None
    board_reason:        str   = ""

    # Justification
    justification:       str   = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ImageResult":
        """Convert a raw pipeline dict to an ImageResult."""
        source = d.get("source", "unknown")
        if source not in cls.model_fields["source"].annotation.__args__:
            source = "unknown"
        return cls(
            photo_id            = str(d.get("photo_id", "")),
            image_url           = d.get("image_url", ""),
            local_path          = d.get("local_path"),
            caption             = d.get("caption", ""),
            score               = float(d.get("score", 0.0)),
            source              = source,
            siglip_score        = float(d.get("siglip_score", 0.0)),
            text_score          = float(d.get("text_score", 0.0)),
            graph_score         = float(d.get("graph_score", 0.0)),
            field_scores        = d.get("field_scores", {}),
            verified            = bool(d.get("verified", False)),
            verification_reason = d.get("verification_reason", ""),
            coherence_rank      = d.get("coherence_rank"),
            board_reason        = d.get("board_reason", ""),
            justification       = d.get("justification", ""),
        )

    def to_dict(self) -> dict:
        return self.model_dump()


# ── 3. PipelineState ──────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    Mutable state object that flows through OrchestratorAgent.

    The orchestrator updates this object at each step so every routing
    decision and intermediate result is captured in one place.
    """
    # Run tracking
    run_id:               Optional[str]    = None

    # Input
    query:                str
    uploaded_image_paths: list[str]        = Field(default_factory=list)
    style_ref_path:       Optional[str]    = None

    # Step outputs (populated as the pipeline runs)
    grounding:            Optional[GroundingOutput] = None
    candidates:           list[ImageResult]         = Field(default_factory=list)
    verified_candidates:  list[ImageResult]         = Field(default_factory=list)
    final_images:         list[ImageResult]         = Field(default_factory=list)

    # Routing
    routing_decision:     Literal["retrieval", "generation", "editing", "pending"] = "pending"
    top_hybrid_score:     float = 0.0
    use_graph:            bool  = True

    # Step timing (seconds)
    step_timings:         dict[str, float] = Field(default_factory=dict)


# ── 4. MoodBoardBundle ────────────────────────────────────────────────────────

class MoodBoardBundle(BaseModel):
    """
    Final output artifact from OrchestratorAgent.

    Contains everything needed to:
      - display the mood board (images)
      - explain the result (board_summary, justifications)
      - reproduce or debug the run (query, grounding, routing metadata)
      - feed into LLM-as-judge evaluation (all fields are serialisable)
    """
    query:             str
    grounding:         GroundingOutput
    images:            list[ImageResult]
    routing_decision:  Literal["retrieval", "generation", "editing"]
    top_hybrid_score:  float
    board_summary:     str   = ""
    timestamp:         str   = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON logging and MCP responses)."""
        return self.model_dump()

    def to_legacy_list(self) -> list[dict]:
        """
        Convert to the list-of-dicts format that app.py and the old pipeline
        currently expect, so existing callers don't need to change at once.
        """
        return [img.to_dict() for img in self.images]
