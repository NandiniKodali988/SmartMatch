"""
src/agents/generation/agent.py

Agent 4: Generation Agent
--------------------------
Two paths:

  Path A — user uploaded images
      → Claude reads full grounding output (including intent) to select editing mode
      → Image editing (inpaint / restyle / blend / collage / composite)
      → Score check; if low, Claude refines prompt and retries
      → Retries exhausted → fall back to Path B

  Path B — no uploaded images / fallback
      → gpt-image-1.5 3 pure generation using full grounding output
      → Returns n images with style variants for visual diversity

Triggered when: hybrid retrieval score < HYBRID_THRESHOLD, or user uploaded images.
"""

import os
import json
import time
import base64
from io import BytesIO
from pathlib import Path

import anthropic
from openai import OpenAI
from PIL import Image as PILImage
from dotenv import load_dotenv
import concurrent.futures

try:
    from prompts import (
        MODE_SELECTION_SYSTEM,
        MODE_SELECTION_USER,
        REFINE_SYSTEM,
        REFINE_USER,
        STYLE_VARIANTS,
    )
except ImportError:
    from agents.generation.prompts import (
        MODE_SELECTION_SYSTEM,
        MODE_SELECTION_USER,
        REFINE_SYSTEM,
        REFINE_USER,
        STYLE_VARIANTS,
    )

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
DALLE_MODEL          = os.getenv("DALLE_MODEL",      "gpt-image-1.5")
EDIT_MODEL           = os.getenv("EDIT_MODEL",       "gpt-image-1.5")
DALLE_SIZE           = os.getenv("DALLE_SIZE",       "1024x1024")
DALLE_QUALITY        = os.getenv("DALLE_QUALITY",    "standard")
CLAUDE_MODEL         = os.getenv("GROUNDING_MODEL",  "claude-haiku-4-5-20251001")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.25"))
MAX_EDIT_RETRIES     = int(os.getenv("MAX_EDIT_RETRIES", "2"))
RATE_LIMIT_SLEEP     = 2   # seconds between DALL·E 3 calls (Tier 1: ~5 img/min)

EDITING_MODES = {"inpaint", "restyle", "blend", "collage", "composite"}


class GenerationAgent:
    """
    Agent 4: image editing (with uploads) + pure DALL·E 3 generation (without uploads).
    """

    def __init__(self):
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # ── Public interface ──────────────────────────────────────────────────────

    def run(
        self,
        grounding_output: dict,
        n: int = 3,
        user_text: str = "",
        uploaded_image_paths: list[str] | None = None,
        style_ref_path: str | None = None,
        siglip_agent=None,
    ) -> list[dict]:
        """
        Args:
            grounding_output     : full dict from VisualGroundingAgent
                                   (visual_description, scene, mood, style,
                                    lighting, color_palette, intent, ...)
            n                    : number of images to generate (no-upload path)
            user_text            : original user text
            uploaded_image_paths : user-uploaded image file paths
            style_ref_path       : style reference image for composite mode
            siglip_agent         : SiglipImageRetrievalAgent instance (for scoring)

        Returns:
            list of pipeline-compatible dicts:
            { photo_id, image_url, caption, score, source, justification }
        """
        if uploaded_image_paths:
            # Path A: user has images → Claude picks editing mode from full grounding
            return self._run_editing(
                user_text,
                grounding_output,
                uploaded_image_paths,
                style_ref_path,
                siglip_agent,
            )
        else:
            # Path B: no uploads → DALL·E 3 generation
            return self._run_generation(grounding_output, n, user_text, siglip_agent)

    # ── Path A: image editing ─────────────────────────────────────────────────

    def _run_editing(
        self,
        user_text: str,
        grounding_output: dict,
        image_paths: list[str],
        style_ref_path: str | None,
        siglip_agent,
    ) -> list[dict]:
        print(f"[Agent 4] Edit path — {len(image_paths)} uploaded image(s).")

        # Claude reads full grounding output (including intent) to select mode
        mode, prompt = self._select_mode(
            user_text, grounding_output, image_paths, style_ref_path,
            has_uploads=True,
        )
        print(f"[Agent 4] Claude selected mode='{mode}', prompt: {prompt[:80]}...")

        for attempt in range(1, MAX_EDIT_RETRIES + 1):
            print(f"[Agent 4] Editing attempt {attempt}/{MAX_EDIT_RETRIES}...")
            try:
                result    = self._edit(mode, image_paths, prompt, style_ref_path)
                image_url = result["image_url"]
                score     = self._score(siglip_agent, grounding_output, prompt)
                print(f"[Agent 4] Score={score:.4f} (threshold={SIMILARITY_THRESHOLD})")

                if score >= SIMILARITY_THRESHOLD or siglip_agent is None:
                    return [self._pack(f"edited_{attempt}", image_url, prompt,
                                       score, f"image_editing/{mode}")]

                if attempt < MAX_EDIT_RETRIES:
                    print("[Agent 4] Score low — refining prompt with Claude...")
                    prompt = self._refine_prompt(user_text, grounding_output, prompt, score)

            except Exception as e:
                print(f"[Agent 4] Editing attempt {attempt} failed: {e}")

        # Retries exhausted → fall back to pure generation
        print("[Agent 4] Edit retries exhausted — falling back to DALL·E 3.")
        return self._run_generation(grounding_output, n=1, user_text=user_text)

    def _edit(
        self,
        mode: str,
        image_paths: list[str],
        prompt: str,
        style_ref_path: str | None,
        mask_path: str | None = None,
    ) -> dict:
        mode = mode.lower().strip()
        if mode == "inpaint":
            return self._inpaint(image_paths[0], prompt, mask_path)
        elif mode == "restyle":
            return self._restyle(image_paths, prompt)
        elif mode == "blend":
            return self._blend(image_paths, prompt)
        elif mode == "collage":
            return self._collage(image_paths, prompt)
        elif mode == "composite":
            return self._composite(image_paths, prompt, style_ref_path)
        else:
            raise ValueError(f"Unknown mode: '{mode}'")

    # ── Editing sub-methods ───────────────────────────────────────────────────

    def _inpaint(self, image_path: str, prompt: str, mask_path: str | None) -> dict:
        print(f"[Agent 4][inpaint] {image_path}")
        image_bytes = self._to_png_bytes(image_path)
        kwargs = dict(
            model=EDIT_MODEL,
            image=("image.png", image_bytes, "image/png"),
            prompt=prompt,
            size=DALLE_SIZE,
            n=1,
        )
        if mask_path:
            kwargs["mask"] = ("mask.png", self._to_png_bytes(mask_path), "image/png")
        resp = self.openai.images.edit(**kwargs)
        return self._package_edit_resp(resp, "inpaint", prompt)

    def _restyle(self, image_paths: list[str], prompt: str) -> dict:
        print(f"[Agent 4][restyle] {len(image_paths)} reference image(s)")
        buf = BytesIO(self._to_png_bytes(image_paths[0]))
        buf.name = "image.png"
        resp = self.openai.images.edit(
            model=EDIT_MODEL, image=buf, prompt=prompt, size=DALLE_SIZE, n=1,
        )
        return self._package_edit_resp(resp, "restyle", prompt)

    def _blend(self, image_paths: list[str], prompt: str) -> dict:
        if len(image_paths) < 2:
            raise ValueError("blend requires at least 2 images.")
        print(f"[Agent 4][blend] {len(image_paths)} images")
        images = []
        for i, p in enumerate(image_paths):
            buf = BytesIO(self._to_png_bytes(p))
            buf.name = f"image_{i}.png"
            images.append(buf)
        resp = self.openai.images.edit(
            model=EDIT_MODEL, image=images, prompt=prompt, size=DALLE_SIZE, n=1,
        )
        return self._package_edit_resp(resp, "blend", prompt)

    def _collage(self, image_paths: list[str], prompt: str) -> dict:
        print(f"[Agent 4][collage] {len(image_paths)} images")
        grid_bytes = self._make_grid(image_paths)
        if not prompt.strip():
            b64 = base64.b64encode(grid_bytes).decode()
            return {"image_url": f"data:image/png;base64,{b64}",
                    "mode": "collage", "prompt": "", "revised_prompt": ""}
        resp = self.openai.images.edit(
            model=EDIT_MODEL,
            image=("collage.png", grid_bytes, "image/png"),
            prompt=prompt, size=DALLE_SIZE, n=1,
        )
        return self._package_edit_resp(resp, "collage", prompt)

    def _composite(
        self,
        image_paths: list[str],
        prompt: str,
        style_ref_path: str | None,
    ) -> dict:
        if len(image_paths) < 2:
            raise ValueError("composite requires at least 2 source images.")
        if not style_ref_path:
            raise ValueError("composite requires a style_ref_path.")
        print(f"[Agent 4][composite] {len(image_paths)} sources + style ref")

        grid_buf = BytesIO(self._make_grid(image_paths))
        grid_buf.name = "grid.png"
        style_buf = BytesIO(self._to_png_bytes(style_ref_path))
        style_buf.name = "style_ref.png"

        final_prompt = prompt.strip() or (
            "Merge and blend all the scenes shown in the grid into one cohesive image, "
            "following the visual style, color palette, and mood of the style reference image."
        )
        resp = self.openai.images.edit(
            model=EDIT_MODEL,
            image=[grid_buf, style_buf],
            prompt=final_prompt, size=DALLE_SIZE, n=1,
        )
        return self._package_edit_resp(resp, "composite", final_prompt)

    # ── Path B: DALL·E 3 pure generation ─────────────────────────────────────
    def _download_image(self, url: str, filename: str) -> str | None:
        """Download a DALL·E image URL to local disk and return the local path."""
        import requests
        from pathlib import Path

        save_dir = Path("outputs/generated")
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / filename

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            save_path.write_bytes(resp.content)
            print(f"[Agent 4] Saved → {save_path}")
            return str(save_path)
        except Exception as e:
            print(f"[Agent 4] Download failed: {e}")
            return None
    
    def _generate_diverse_prompts(
        self,
        grounding_output: dict,
        user_text: str,
        n: int,
    ) -> list[str]:
        """
        Ask Claude to generate n visually distinct image descriptions
        for the same query, each from a different angle.

        Diversity axes enforced in the prompt:
          - Subject type: person / environment / object / abstract
          - Scale: wide / medium / close-up / macro
          - Emotional register: literal / symbolic / metaphorical
          - Composition: centered / asymmetric / overhead / silhouette
        """
        mood    = grounding_output.get("mood",          "").strip()
        style   = grounding_output.get("style",         "").strip()
        palette = grounding_output.get("color_palette", "").strip()
        scene   = grounding_output.get("scene",         "").strip()

        system = (
            "You are a creative director writing image generation prompts. "
            "Your prompts must be visually specific, photorealistic, and maximally diverse from each other. "
            "Each prompt should feel like a completely different photograph."
        )
        user = (
            f"Generate {n} visually DISTINCT image prompts for this mood board:\n"
            f"Query: {user_text}\n"
            f"Mood: {mood}\n"
            f"Scene: {scene}\n"
            f"Style: {style}\n"
            f"Color palette: {palette}\n\n"
            f"Requirements:\n"
            f"- Each prompt MUST have a different subject type: "
            f"cycle through (person, interior space, outdoor environment, "
            f"still life/objects, abstract texture, urban architecture, "
            f"nature detail, silhouette, aerial/overhead)\n"
            f"- Each prompt MUST have a different scale: "
            f"(extreme close-up, close-up, medium, wide, aerial)\n"
            f"- No two prompts should describe the same scene or composition\n"
            f"- Every prompt must be one sentence, specific and visual\n\n"
            f"Return ONLY a valid JSON array of {n} strings. No explanation, no markdown.\n"
            f'Example format: ["prompt 1", "prompt 2", ...]'
        )

        try:
            resp = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=1000,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            import json as _json
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            prompts = _json.loads(raw.strip())
            if isinstance(prompts, list) and len(prompts) == n:
                # Append style + palette to each for consistency
                suffix = ""
                if style:   suffix += f" {style} photography style."
                if palette: suffix += f" Color palette: {palette}."
                return [p.rstrip(".") + suffix for p in prompts]
        except Exception as e:
            print(f"[Agent 4] Diverse prompt generation failed: {e} — using base prompt.")

        # Fallback: use base prompt for all
        base = self._build_generation_prompt(grounding_output)
        return [base] * n

    def _run_generation(
        self,
        grounding_output: dict,
        n: int = 3,
        user_text: str = "",
        siglip_agent=None,
    ) -> list[dict]:
        """
        Generate n images using DALL·E 3.
        Step 1: Claude generates n diverse prompts (different scale/composition/subject).
        Step 2: API calls run concurrently (max 3 threads) — only network I/O.
        Step 3: Save + score back in main thread (siglip is not thread-safe).
        Step 4: Retry with _refine_prompt if score < 0.15.
        """
        import uuid
        print(f"[Agent 4] Generation path — {n} image(s).")
        print(f"[Agent 4] Generating {n} diverse prompts with Claude...")
        diverse_prompts = self._generate_diverse_prompts(grounding_output, user_text, n)

        prompts = [
            f"{diverse_prompts[i]} {STYLE_VARIANTS[i % len(STYLE_VARIANTS)]}"
            for i in range(n)
        ]

        # ── Step 2: parallel DALL·E API calls ────────────────────────────────
        def call_one(args):
            i, prompt = args
            print(f"[Agent 4] Generating image {i+1}/{n}...")
            url = self._call_dalle(prompt)
            return (i, prompt, url)

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            raw = list(executor.map(call_one, enumerate(prompts)))

        # ── Step 3: save + score in main thread ──────────────────────────────
        save_dir = Path("outputs/generated")
        save_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, prompt, url in raw:
            if not url:
                continue

            filename   = f"generated_{uuid.uuid4().hex[:8]}_{i+1}.png"
            base_prompt = diverse_prompts[i]

            if url.startswith("data:image"):
                b64_data  = url.split(",", 1)[1]
                img_bytes = base64.b64decode(b64_data)
                local_path = str(save_dir / filename)
                with open(local_path, "wb") as f:
                    f.write(img_bytes)
                print(f"[Agent 4] Saved → {local_path}")
            else:
                local_path = self._download_image(url, filename)

            score = self._score(siglip_agent, grounding_output, base_prompt)
            print(f"[Agent 4] Image {i+1}/{n} score={score:.4f}")

            # ── Step 4: retry once with refined prompt if score too low ───────
            if score < 0.15 and siglip_agent is not None:
                print(f"[Agent 4] Score too low — refining prompt with Claude...")
                refined_prompt = self._refine_prompt(
                    user_text=user_text,
                    grounding_output=grounding_output,
                    previous_prompt=prompt,
                    previous_score=score,
                )
                alt_url = self._call_dalle(refined_prompt)
                if alt_url:
                    alt_score = self._score(siglip_agent, grounding_output, refined_prompt)
                    if alt_score > score:
                        url, prompt, score = alt_url, refined_prompt, alt_score
                        if url.startswith("data:image"):
                            b64_data  = url.split(",", 1)[1]
                            img_bytes = base64.b64decode(b64_data)
                            with open(local_path, "wb") as f:
                                f.write(img_bytes)
                        print(f"[Agent 4] Retry improved score to {score:.4f}")

            results.append(
                self._pack(
                    f"generated_{uuid.uuid4().hex[:8]}_{i+1}",
                    url, base_prompt, score, "dalle3", local_path,
                )
            )

        print(f"[Agent 4] Generated {len(results)} image(s).")
        return results

    def _call_dalle(self, prompt: str) -> str | None:
        try:
            resp = self.openai.images.generate(
                model=DALLE_MODEL,
                prompt=prompt,
                size=DALLE_SIZE,
                n=1,
            )
            item = resp.data[0]
            # gpt-image-1.5 returns base64, dall-e-3 returns URL
            if hasattr(item, "b64_json") and item.b64_json:
                return f"data:image/png;base64,{item.b64_json}"
            return item.url
        except Exception as e:
            print(f"[Agent 4] Generation failed: {e}")
            return None

    def _build_generation_prompt(self, grounding_output: dict) -> str:
        """
        Build a DALL·E 3 prompt from the full grounding output.
        Uses visual_description as the base, then appends scene, mood,
        style, lighting, and color_palette for richness.
        """
        desc     = grounding_output.get("visual_description", "").strip()
        scene    = grounding_output.get("scene",         "").strip()
        mood     = grounding_output.get("mood",          "").strip()
        style    = grounding_output.get("style",         "").strip()
        lighting = grounding_output.get("lighting",      "").strip()
        palette  = grounding_output.get("color_palette", "").strip()

        base   = desc if desc else "A photograph"
        extras = []
        if scene:    extras.append(f"set in {scene}")
        if mood:     extras.append(f"with a {mood} atmosphere")
        if style:    extras.append(f"in the style of {style}")
        if lighting: extras.append(f"lit by {lighting}")
        if palette:  extras.append(f"color palette: {palette}")

        if extras:
            base = base.rstrip(".") + ", " + ", ".join(extras) + "."

        return base

    # ── Claude: mode selection ────────────────────────────────────────────────

    def _select_mode(
        self,
        user_text: str,
        grounding_output: dict,
        image_paths: list[str] | None,
        style_ref_path: str | None,
        has_uploads: bool = False,
    ) -> tuple[str, str]:
        """
        Claude reads the full grounding output (including intent) and decides
        the best generation mode and prompt.

        When has_uploads=False, Claude must choose dalle3.
        When has_uploads=True, Claude chooses from the editing modes.
        """
        num_images = len(image_paths) if image_paths else 0
        has_style  = bool(style_ref_path)

        system = MODE_SELECTION_SYSTEM.format(
            num_images=num_images,
            has_uploads=has_uploads,
            has_style=has_style,
        )
        user_msg = MODE_SELECTION_USER.format(
            user_text=user_text,
            intent=grounding_output.get("intent",            ""),
            visual_description=grounding_output.get("visual_description", ""),
            scene=grounding_output.get("scene",              ""),
            mood=grounding_output.get("mood",                ""),
            style=grounding_output.get("style",              ""),
            lighting=grounding_output.get("lighting",        ""),
            color_palette=grounding_output.get("color_palette", ""),
        )

        try:
            resp = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                system=system,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = resp.content[0].text.strip()
            if raw.startswith("```"):
                raw = raw.split("```", 2)[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            data   = json.loads(raw.strip())
            mode   = data.get("mode", "dalle3")
            prompt = data.get("prompt", user_text)

            # Safety guards
            if not has_uploads:
                mode = "dalle3"
            elif mode not in EDITING_MODES:
                mode = "inpaint"
            elif mode == "composite" and not has_style:
                mode = "blend" if num_images >= 2 else "inpaint"
            elif mode in {"blend", "collage", "composite"} and num_images < 2:
                mode = "inpaint"

            return mode, prompt

        except Exception as e:
            print(f"[Agent 4] Mode selection failed: {e} — safe default.")
            if not has_uploads:
                return "dalle3", user_text
            return ("blend" if num_images >= 2 else "inpaint"), user_text

    # ── Claude: prompt refinement ─────────────────────────────────────────────

    def _refine_prompt(
        self,
        user_text: str,
        grounding_output: dict,
        previous_prompt: str,
        previous_score: float,
    ) -> str:
        """
        When a generated/edited image scores below threshold,
        Claude rewrites the prompt using the full grounding output.
        """
        user_msg = REFINE_USER.format(
            user_text=user_text,
            intent=grounding_output.get("intent",            ""),
            visual_description=grounding_output.get("visual_description", ""),
            scene=grounding_output.get("scene",              ""),
            mood=grounding_output.get("mood",                ""),
            color_palette=grounding_output.get("color_palette", ""),
            previous_score=previous_score,
            previous_prompt=previous_prompt,
        )
        try:
            resp = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=300,
                system=REFINE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            refined = resp.content[0].text.strip()
            print(f"[Agent 4] Refined prompt: {refined[:80]}...")
            return refined
        except Exception as e:
            print(f"[Agent 4] Prompt refinement failed: {e}")
            return previous_prompt

    # ── Image helpers ─────────────────────────────────────────────────────────

    def _to_png_bytes(self, path: str) -> bytes:
        with PILImage.open(path) as img:
            img = img.convert("RGBA")
            buf = BytesIO()
            img.save(buf, format="PNG")
            return buf.getvalue()

    def _make_grid(self, image_paths: list[str], thumb_size: int = 512) -> bytes:
        images = [
            PILImage.open(p).convert("RGB").resize((thumb_size, thumb_size))
            for p in image_paths
        ]
        cols = min(len(images), 2)
        rows = (len(images) + cols - 1) // cols
        grid = PILImage.new("RGB", (cols * thumb_size, rows * thumb_size), color=(255, 255, 255))
        for idx, img in enumerate(images):
            r, c = divmod(idx, cols)
            grid.paste(img, (c * thumb_size, r * thumb_size))
        buf = BytesIO()
        grid.save(buf, format="PNG")
        return buf.getvalue()

    def _package_edit_resp(self, response, mode: str, prompt: str) -> dict:
        item = response.data[0]
        if hasattr(item, "b64_json") and item.b64_json:
            image_url = f"data:image/png;base64,{item.b64_json}"
        else:
            image_url = item.url
        return {
            "image_url":      image_url,
            "mode":           mode,
            "prompt":         prompt,
            "revised_prompt": getattr(item, "revised_prompt", ""),
        }

    def _score(self, siglip_agent, grounding_output: dict, caption: str) -> float:
        """Text-text proxy score via SigLIP text encoder."""
        if siglip_agent is None:
            return 1.0
        try:
            import numpy as np
            q_emb   = siglip_agent.embed_text(siglip_agent.build_query(grounding_output))
            cap_emb = siglip_agent.embed_text(caption)
            return float(np.dot(q_emb, cap_emb))
        except Exception as e:
            print(f"[Agent 4] Scoring failed: {e}")
            return 0.0

    def _pack(self, photo_id, image_url, caption, score, source, local_path=None) -> dict:
        return {
            "photo_id":      photo_id,
            "image_url":     image_url,
            "local_path":    local_path,
            "caption":       caption,
            "score":         score,
            "source":        source,
            "justification": "",
        }