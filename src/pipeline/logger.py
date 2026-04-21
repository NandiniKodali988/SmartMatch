"""
src/pipeline/logger.py

PipelineLogger
--------------
Structured JSON logging for every SmartMatch pipeline run.

Every run produces one log entry containing:
    - run_id       : unique identifier (timestamp + short hash)
    - timestamp    : ISO-8601 UTC
    - query        : raw user input
    - grounding    : full GroundingOutput fields
    - retrieval    : top-k candidates with scores
    - graph_rag    : dedup / expand / rerank summary
    - verification : per-image decision (verified / rejected + reason)
    - routing      : "retrieval" | "generation" | "editing", top_hybrid_score
    - coherence    : selected image ids + board rationale
    - final_images : id, source, score, verified, coherence_rank, justification
    - board_summary: board-level narrative from JustificationAgent
    - step_timings : seconds per step
    - total_seconds: total wall-clock time

Log entries are appended to a newline-delimited JSON file (one JSON object
per line) so they can be streamed and processed without loading the full file.

Default log path: logs/pipeline_runs.jsonl
Override with env var: PIPELINE_LOG_PATH

Usage
-----
    from pipeline.logger import PipelineLogger

    logger = PipelineLogger()
    run_id = logger.start(query)

    logger.log_grounding(run_id, grounding_output)
    logger.log_retrieval(run_id, candidates)
    logger.log_graph_rag(run_id, before_count, after_count, added_count)
    logger.log_verification(run_id, verified_candidates)
    logger.log_routing(run_id, decision, top_score)
    logger.log_coherence(run_id, selected_images)
    logger.log_final(run_id, bundle)       # writes the complete entry to disk
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pipeline.state import GroundingOutput, ImageResult, MoodBoardBundle

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_LOG_PATH = Path(os.getenv("PIPELINE_LOG_PATH", "logs/pipeline_runs.jsonl"))


class PipelineLogger:
    """
    Collects structured data throughout a pipeline run and writes one
    JSON log entry when the run completes.

    Thread-safety: each run_id maps to its own dict, so concurrent runs
    with different run_ids are safe. Do not share a run_id across threads.
    """

    def __init__(self, log_path: Path | str | None = None):
        self.log_path = Path(log_path) if log_path else DEFAULT_LOG_PATH
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._runs: dict[str, dict] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self, query: str) -> str:
        """
        Begin tracking a new pipeline run.

        Returns a run_id string that must be passed to all subsequent calls.
        """
        ts     = datetime.now(timezone.utc)
        digest = hashlib.sha1(f"{query}{ts.isoformat()}".encode()).hexdigest()[:8]
        run_id = f"{ts.strftime('%Y%m%dT%H%M%S')}_{digest}"

        self._runs[run_id] = {
            "run_id":       run_id,
            "timestamp":    ts.isoformat(),
            "query":        query,
            "grounding":    {},
            "retrieval":    {},
            "graph_rag":    {},
            "verification": {},
            "routing":      {},
            "coherence":    {},
            "final_images": [],
            "board_summary": "",
            "step_timings": {},
            "total_seconds": 0.0,
            "_start_wall":  time.time(),
        }

        print(f"[Logger] Run started — id={run_id}")
        return run_id

    # ── Step loggers ──────────────────────────────────────────────────────────

    def log_grounding(self, run_id: str, grounding: GroundingOutput | dict):
        """Log the full grounding output."""
        entry = self._get(run_id)
        if entry is None:
            return
        if isinstance(grounding, GroundingOutput):
            grounding = grounding.to_dict()
        entry["grounding"] = grounding

    def log_retrieval(self, run_id: str, candidates: list[dict | ImageResult]):
        """
        Log retrieval candidates with their scores.
        Stores the top 20 to keep log size manageable.
        """
        entry = self._get(run_id)
        if entry is None:
            return

        rows = []
        for c in candidates[:20]:
            d = c.to_dict() if isinstance(c, ImageResult) else c
            rows.append({
                "photo_id":     d.get("photo_id", ""),
                "score":        round(float(d.get("score",        0.0)), 4),
                "siglip_score": round(float(d.get("siglip_score", 0.0)), 4),
                "text_score":   round(float(d.get("text_score",   0.0)), 4),
                "source":       d.get("source", ""),
            })

        entry["retrieval"] = {
            "total_candidates": len(candidates),
            "top_score":        rows[0]["score"] if rows else 0.0,
            "candidates":       rows,
        }

    def log_graph_rag(
        self,
        run_id: str,
        before_count: int,
        after_dedup_count: int,
        added_by_expand: int,
        after_rerank_top_score: float,
    ):
        """Log Graph RAG step summary."""
        entry = self._get(run_id)
        if entry is None:
            return
        entry["graph_rag"] = {
            "before_dedup":          before_count,
            "after_dedup":           after_dedup_count,
            "removed_by_dedup":      before_count - after_dedup_count,
            "added_by_expand":       added_by_expand,
            "after_rerank_top_score": round(after_rerank_top_score, 4),
        }

    def log_verification(self, run_id: str, verified_candidates: list[dict | ImageResult]):
        """Log per-image verification decisions."""
        entry = self._get(run_id)
        if entry is None:
            return

        decisions = []
        for c in verified_candidates:
            d = c.to_dict() if isinstance(c, ImageResult) else c
            decisions.append({
                "photo_id": d.get("photo_id", ""),
                "verified": bool(d.get("verified", False)),
                "reason":   d.get("verification_reason", ""),
                "score":    round(float(d.get("score", 0.0)), 4),
            })

        n_verified = sum(1 for dec in decisions if dec["verified"])
        entry["verification"] = {
            "total":      len(decisions),
            "passed":     n_verified,
            "rejected":   len(decisions) - n_verified,
            "pass_rate":  round(n_verified / len(decisions), 3) if decisions else 0.0,
            "decisions":  decisions,
        }

    def log_routing(
        self,
        run_id: str,
        decision: str,
        top_score: float,
        threshold: float = 0.5,
    ):
        """Log the retrieval-vs-generation routing decision."""
        entry = self._get(run_id)
        if entry is None:
            return
        entry["routing"] = {
            "decision":       decision,
            "top_score":      round(top_score, 4),
            "threshold":      round(threshold, 4),
            "score_vs_threshold": round(top_score - threshold, 4),
        }

    def log_coherence(self, run_id: str, selected_images: list[dict | ImageResult]):
        """Log coherence selection result."""
        entry = self._get(run_id)
        if entry is None:
            return

        selected = []
        board_reason = ""
        for img in selected_images:
            d = img.to_dict() if isinstance(img, ImageResult) else img
            selected.append({
                "photo_id":      d.get("photo_id", ""),
                "coherence_rank": d.get("coherence_rank"),
                "score":         round(float(d.get("score", 0.0)), 4),
            })
            if not board_reason:
                board_reason = d.get("board_reason", "")

        entry["coherence"] = {
            "selected_count": len(selected),
            "board_reason":   board_reason,
            "selected":       selected,
        }

    def log_step_timing(self, run_id: str, step: str, seconds: float):
        """Record the wall-clock time for a named step."""
        entry = self._get(run_id)
        if entry is None:
            return
        entry["step_timings"][step] = round(seconds, 3)

    def log_final(self, run_id: str, bundle: MoodBoardBundle):
        """
        Record the final MoodBoardBundle and flush the complete log entry to disk.
        Must be called once per run after all other log_* calls.
        """
        entry = self._get(run_id)
        if entry is None:
            return

        # Final images summary
        entry["final_images"] = [
            {
                "photo_id":       img.photo_id,
                "source":         img.source,
                "score":          round(img.score, 4),
                "verified":       img.verified,
                "coherence_rank": img.coherence_rank,
                "justification":  img.justification[:200],
            }
            for img in bundle.images
        ]

        entry["board_summary"]  = bundle.board_summary
        entry["routing"]        = entry.get("routing") or {
            "decision":  bundle.routing_decision,
            "top_score": round(bundle.top_hybrid_score, 4),
        }
        entry["total_seconds"]  = round(time.time() - entry.pop("_start_wall", time.time()), 3)

        self._write(entry)
        self._runs.pop(run_id, None)
        print(f"[Logger] Run logged — id={run_id}  "
              f"images={len(bundle.images)}  "
              f"total={entry['total_seconds']}s  "
              f"→ {self.log_path}")

    # ── Read-back ─────────────────────────────────────────────────────────────

    def read_all(self) -> list[dict]:
        """
        Load all log entries from disk.
        Useful for evaluation scripts and the ablation study.
        """
        if not self.log_path.exists():
            return []
        entries = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return entries

    def read_latest(self, n: int = 10) -> list[dict]:
        """Return the n most recent log entries."""
        return self.read_all()[-n:]

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get(self, run_id: str) -> dict | None:
        entry = self._runs.get(run_id)
        if entry is None:
            print(f"[Logger] Warning: unknown run_id={run_id!r}")
        return entry

    def _write(self, entry: dict):
        """Append one JSON line to the log file."""
        # Remove internal fields before writing
        clean = {k: v for k, v in entry.items() if not k.startswith("_")}
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")
