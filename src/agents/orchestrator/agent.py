"""
src/agents/orchestrator/agent.py

OrchestratorAgent
-----------------
Coordinates all pipeline steps and returns a MoodBoardBundle.

Pipeline (no uploaded images):
    query
      → [Guardrail] input safety check
      → GroundingAgent
      → Hybrid Retrieval (SigLIP + FieldText)
      → GraphRAG (deduplicate → expand → rerank)
      → score >= HYBRID_THRESHOLD?
          ├─ YES → MultimodalVerification
          │          → enough verified? → CoherenceAgent
          │          → not enough      → GenerationAgent fills gap → CoherenceAgent
          └─ NO  → GenerationAgent (full DALL·E 3 board)
      → JustificationAgent (per-image + board summary)
      → [Guardrail] output safety check
      → MoodBoardBundle

Pipeline (with uploaded images):
    query + images
      → [Guardrail] input safety check
      → GroundingAgent
      → GenerationAgent (editing mode)
      → JustificationAgent
      → MoodBoardBundle
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
from agents.generation.agent import GenerationAgent
from agents.multimodal_verification.agent import MultimodalVerificationAgent
from agents.coherence.agent import CoherenceAgent
from agents.graph_rag.agent import GraphRAGAgent
from agents.memory.memory_manager import MemoryManager
from pipeline.state import GroundingOutput, ImageResult, PipelineState, MoodBoardBundle
from pipeline.logger import PipelineLogger

# ── Config ────────────────────────────────────────────────────────────────────
HYBRID_THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
CANDIDATE_K      = int(os.getenv("CANDIDATE_K",  "20"))
BOARD_SIZE       = int(os.getenv("BOARD_SIZE",   "9"))
MIN_VERIFIED     = int(os.getenv("MIN_VERIFIED", "6"))


class OrchestratorAgent:
    """
    Coordinates all SmartMatch agents and returns a MoodBoardBundle.

    Usage
    -----
        orchestrator = OrchestratorAgent()
        bundle = orchestrator.run("campaign visuals for a sustainable coffee brand")

        # Multi-turn (pass memory from previous turn)
        bundle = orchestrator.run("make it warmer", memory=memory)

        # With uploaded images
        bundle = orchestrator.run("restyle this", uploaded_image_paths=["/tmp/img.jpg"])
    """

    def __init__(self, logger: PipelineLogger | None = None):
        # Lazy-initialise heavy agents (SigLIP loads a model — only once per instance)
        self._grounding_agent     = None
        self._justification_agent = None
        self._siglip_agent        = None
        self._text_agent          = None
        self._generation_agent    = None
        self._verification_agent  = None
        self._coherence_agent     = None
        self._graph_agent         = None
        self._graph_available     = None   # None = not yet checked
        self.logger               = logger or PipelineLogger()

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        query: str,
        uploaded_image_paths: list[str] | None = None,
        style_ref_path: str | None = None,
        memory: MemoryManager | None = None,
    ) -> MoodBoardBundle:
        """
        Run the full pipeline and return a MoodBoardBundle.

        Args:
            query                : user text input
            uploaded_image_paths : local image paths for editing mode (optional)
            style_ref_path       : style reference image for composite mode (optional)
            memory               : MemoryManager for multi-turn context (optional)

        Returns:
            MoodBoardBundle with images, grounding, routing metadata, and board summary
        """
        state = PipelineState(
            query=query,
            uploaded_image_paths=uploaded_image_paths or [],
            style_ref_path=style_ref_path,
        )

        run_id = self.logger.start(query)
        state.run_id = run_id

        self._log("START", f"query={query!r}")

        # ── Step 1: Input guardrail ───────────────────────────────────────────
        t = time.time()
        blocked, block_reason = self._check_input(query)
        state.step_timings["guardrail_input"] = round(time.time() - t, 3)

        if blocked:
            self._log("GUARDRAIL", f"Input blocked: {block_reason}")
            bundle = self._blocked_bundle(state, block_reason)
            self.logger.log_routing(run_id, "blocked", 0.0)
            self.logger.log_final(run_id, bundle)
            return bundle

        # ── Step 2: Visual grounding ──────────────────────────────────────────
        t = time.time()
        previous_grounding = memory.get_last() if memory else None
        raw_grounding = self._grounding().run(query, previous_grounding=previous_grounding)
        state.grounding = GroundingOutput.from_dict(raw_grounding)
        state.step_timings["grounding"] = round(time.time() - t, 3)
        self.logger.log_grounding(run_id, state.grounding)
        self.logger.log_step_timing(run_id, "grounding", state.step_timings["grounding"])

        if memory:
            memory.add(query, raw_grounding)

        self._log("GROUNDING", (
            f"mood={state.grounding.mood!r}  "
            f"intent={state.grounding.intent!r}  "
            f"color_palette={state.grounding.color_palette!r}"
        ))

        # ── Branch A: uploaded images → editing ───────────────────────────────
        if uploaded_image_paths:
            return self._run_editing(state)

        # ── Branch B: no uploads → retrieval + graph RAG + verification ───────
        return self._run_retrieval(state)

    # ── Branch A ──────────────────────────────────────────────────────────────

    def _run_editing(self, state: PipelineState) -> MoodBoardBundle:
        state.routing_decision = "editing"
        self._log("ROUTE", "editing (uploaded images)")

        t = time.time()
        raw_images = self._generation().run(
            grounding_output=state.grounding.to_dict(),
            user_text=state.query,
            uploaded_image_paths=state.uploaded_image_paths,
            style_ref_path=state.style_ref_path,
            siglip_agent=self._siglip(),
        )
        state.step_timings["generation_editing"] = round(time.time() - t, 3)
        self._log("EDITING", f"{len(raw_images)} image(s) produced")

        return self._finalise(state, raw_images)

    # ── Branch B ──────────────────────────────────────────────────────────────

    def _run_retrieval(self, state: PipelineState) -> MoodBoardBundle:
        run_id = state.run_id
        grounding_dict = state.grounding.to_dict()

        # Step 3: Hybrid retrieval
        t = time.time()
        raw_candidates = self._text().merge_with_siglip(
            grounding_output=grounding_dict,
            siglip_agent=self._siglip(),
        )
        state.step_timings["hybrid_retrieval"] = round(time.time() - t, 3)
        state.top_hybrid_score = raw_candidates[0]["score"] if raw_candidates else 0.0
        self.logger.log_retrieval(run_id, raw_candidates)
        self.logger.log_step_timing(run_id, "hybrid_retrieval", state.step_timings["hybrid_retrieval"])
        self._log("RETRIEVAL", f"{len(raw_candidates)} candidates  top_score={state.top_hybrid_score:.4f}")

        # Step 4: Graph RAG
        if self._graph_available is None:
            try:
                self._graph()
                self._graph_available = True
            except FileNotFoundError as e:
                self._log("GRAPH_RAG", f"disabled — {e}")
                self._graph_available = False

        if self._graph_available:
            t = time.time()
            before_count   = len(raw_candidates)
            raw_candidates = self._graph().deduplicate(raw_candidates)
            after_dedup    = len(raw_candidates)
            raw_candidates = self._graph().expand(raw_candidates, self._text(), grounding_output=grounding_dict)
            added_by_expand = len(raw_candidates) - after_dedup
            raw_candidates = self._graph().rerank(raw_candidates, grounding_dict)
            state.step_timings["graph_rag"] = round(time.time() - t, 3)
            top_after_rerank = raw_candidates[0]["score"] if raw_candidates else 0.0
            self.logger.log_graph_rag(run_id, before_count, after_dedup, added_by_expand, top_after_rerank)
            self.logger.log_step_timing(run_id, "graph_rag", state.step_timings["graph_rag"])

        top_score = raw_candidates[0]["score"] if raw_candidates else 0.0
        state.candidates = [ImageResult.from_dict(c) for c in raw_candidates]

        # Step 5: Route on score
        if top_score < HYBRID_THRESHOLD:
            state.routing_decision = "generation"
            self.logger.log_routing(run_id, "generation", top_score, HYBRID_THRESHOLD)
            self._log("ROUTE", f"generation (score {top_score:.4f} < threshold {HYBRID_THRESHOLD})")

            t = time.time()
            raw_images = self._generation().run(
                grounding_output=grounding_dict,
                n=BOARD_SIZE,
                user_text=state.query,
                uploaded_image_paths=None,
                style_ref_path=None,
                siglip_agent=self._siglip(),
            )
            state.step_timings["generation_fallback"] = round(time.time() - t, 3)
            self._log("GENERATION", f"{len(raw_images)} image(s) generated")

            return self._finalise(state, raw_images)

        # Step 5b: Retrieval path → multimodal verification
        state.routing_decision = "retrieval"
        self.logger.log_routing(run_id, "retrieval", top_score, HYBRID_THRESHOLD)
        self._log("ROUTE", f"retrieval (score {top_score:.4f} >= threshold {HYBRID_THRESHOLD})")

        t = time.time()
        verified_raw = self._verification().run(raw_candidates[:CANDIDATE_K], grounding_dict)
        state.step_timings["verification"] = round(time.time() - t, 3)
        self.logger.log_verification(run_id, verified_raw)
        self.logger.log_step_timing(run_id, "verification", state.step_timings["verification"])

        n_verified = sum(1 for c in verified_raw if c.get("verified"))
        self._log("VERIFICATION", f"{n_verified}/{len(verified_raw)} passed")

        # Step 5c: Fill gap with generation if not enough verified
        if n_verified < BOARD_SIZE:
            n_needed = BOARD_SIZE - n_verified
            self._log("GENERATION_GAP", f"need {n_needed} more image(s)")

            t = time.time()
            generated_raw = self._generation().run(
                grounding_output=grounding_dict,
                n=n_needed,
                user_text=state.query,
                uploaded_image_paths=None,
                style_ref_path=None,
                siglip_agent=self._siglip(),
            )
            state.step_timings["generation_gap"] = round(time.time() - t, 3)
            pool = verified_raw + generated_raw
        else:
            pool = verified_raw

        state.verified_candidates = [ImageResult.from_dict(c) for c in pool]

        # Step 6: Coherence selection
        t = time.time()
        coherent_raw = self._coherence().run(pool[:20], grounding_dict)
        state.step_timings["coherence"] = round(time.time() - t, 3)
        self.logger.log_coherence(run_id, coherent_raw)
        self.logger.log_step_timing(run_id, "coherence", state.step_timings["coherence"])
        self._log("COHERENCE", f"{len(coherent_raw)} image(s) selected from {len(pool)}")

        return self._finalise(state, coherent_raw)

    # ── Shared finalisation ───────────────────────────────────────────────────

    def _finalise(self, state: PipelineState, raw_images: list[dict]) -> MoodBoardBundle:
        """Run justification, output guardrail, and pack into a MoodBoardBundle."""

        # Step 7: Justification (per-image + board summary)
        t = time.time()
        justified_images, board_summary = self._justification().run_with_board_summary(
            state.query, raw_images
        )
        state.step_timings["justification"] = round(time.time() - t, 3)
        self._log("JUSTIFICATION", f"{len(justified_images)} justification(s) generated")

        # Step 8: Output guardrail
        t = time.time()
        justified_images = self._check_output(justified_images)
        state.step_timings["guardrail_output"] = round(time.time() - t, 3)

        state.final_images = [ImageResult.from_dict(img) for img in justified_images]

        bundle = MoodBoardBundle(
            query=state.query,
            grounding=state.grounding,
            images=state.final_images,
            routing_decision=state.routing_decision,
            top_hybrid_score=state.top_hybrid_score,
            board_summary=board_summary,
        )

        run_id = state.run_id
        if run_id:
            self.logger.log_step_timing(run_id, "justification", state.step_timings.get("justification", 0.0))
            self.logger.log_final(run_id, bundle)

        total = sum(state.step_timings.values())
        self._log("DONE", (
            f"{len(bundle.images)} image(s)  "
            f"routing={bundle.routing_decision}  "
            f"total={total:.2f}s  "
            f"timings={state.step_timings}"
        ))

        return bundle

    # ── Guardrails ────────────────────────────────────────────────────────────

    def _check_input(self, query: str) -> tuple[bool, str]:
        """
        Returns (blocked, reason).
        Calls GuardrailAgent if available; always allows if not yet implemented.
        """
        try:
            from agents.guardrails.agent import GuardrailAgent
            agent = GuardrailAgent()
            return agent.check_input(query)
        except ImportError:
            return False, ""

    def _check_output(self, images: list[dict]) -> list[dict]:
        """
        Filters out any images flagged by the output guardrail.
        Returns the same list if guardrail is not yet implemented.
        """
        try:
            from agents.guardrails.agent import GuardrailAgent
            agent = GuardrailAgent()
            return agent.check_output(images)
        except ImportError:
            return images

    # ── Lazy agent initialisation ─────────────────────────────────────────────

    def _grounding(self) -> QwenVisualGroundingAgent:
        if self._grounding_agent is None:
            self._grounding_agent = QwenVisualGroundingAgent()
        return self._grounding_agent

    def _justification(self) -> QwenJustificationAgent:
        if self._justification_agent is None:
            self._justification_agent = QwenJustificationAgent()
        return self._justification_agent

    def _siglip(self) -> SiglipImageRetrievalAgent:
        if self._siglip_agent is None:
            self._siglip_agent = SiglipImageRetrievalAgent(top_k=CANDIDATE_K)
        return self._siglip_agent

    def _text(self) -> FieldTextRetrievalAgent:
        if self._text_agent is None:
            self._text_agent = FieldTextRetrievalAgent(top_k=CANDIDATE_K)
        return self._text_agent

    def _generation(self) -> GenerationAgent:
        if self._generation_agent is None:
            self._generation_agent = GenerationAgent()
        return self._generation_agent

    def _verification(self) -> MultimodalVerificationAgent:
        if self._verification_agent is None:
            self._verification_agent = MultimodalVerificationAgent(min_verified=MIN_VERIFIED)
        return self._verification_agent

    def _coherence(self) -> CoherenceAgent:
        if self._coherence_agent is None:
            self._coherence_agent = CoherenceAgent(target_count=BOARD_SIZE)
        return self._coherence_agent

    def _graph(self) -> GraphRAGAgent:
        if self._graph_agent is None:
            self._graph_agent = GraphRAGAgent()
        return self._graph_agent

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log(self, step: str, msg: str):
        print(f"[Orchestrator][{step}] {msg}")

    def _blocked_bundle(self, state: PipelineState, reason: str) -> MoodBoardBundle:
        """Return an empty bundle when the input guardrail blocks the query."""
        return MoodBoardBundle(
            query=state.query,
            grounding=state.grounding or GroundingOutput(),
            images=[],
            routing_decision="generation",
            top_hybrid_score=0.0,
            board_summary=f"This request could not be processed: {reason}",
        )


# ── CLI entry ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())

    parser = argparse.ArgumentParser(description="SmartMatch OrchestratorAgent")
    parser.add_argument("--query", type=str,
                        default="Walking alone through a rainy city street at night")
    parser.add_argument("--images", type=str, nargs="*")
    parser.add_argument("--style",  type=str, default=None)
    args = parser.parse_args()

    orch   = OrchestratorAgent()
    bundle = orch.run(
        query=args.query,
        uploaded_image_paths=args.images,
        style_ref_path=args.style,
    )

    print("\n=== MoodBoardBundle ===")
    print(f"query          : {bundle.query}")
    print(f"routing        : {bundle.routing_decision}")
    print(f"top_score      : {bundle.top_hybrid_score:.4f}")
    print(f"images         : {len(bundle.images)}")
    print(f"board_summary  : {bundle.board_summary[:120]}")
    print(f"timestamp      : {bundle.timestamp}")
    for i, img in enumerate(bundle.images, 1):
        print(f"\n  {i}. [{img.source}] {img.photo_id}")
        print(f"     score={img.score:.4f}  verified={img.verified}  rank={img.coherence_rank}")
        print(f"     justification: {img.justification[:80]}")
