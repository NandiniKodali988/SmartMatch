"""
src/agents/moodboard_layout/agent.py

Mood Board Layout Agent
------------------------
Two-step process:

  Step 1: Claude picks the best HTML template based on image count and aspect ratios.
          Templates are mapped by image count:
            6 images → F
            7 images → G or B (Claude picks)
            8 images → D
            9 images → E or D (Claude picks)

  Step 2: Fill the HTML template with base64-embedded images + extracted colors,
          then render to PNG via Playwright. The rendered PNG is the final output.

  Note: gpt-image-1 composite was removed because its fixed output sizes
  (1024x1024, 1024x1792) do not match the template aspect ratio (~2400x3100),
  causing cropping. The Playwright render is full-resolution and looks better.

Image source handling:
  - local_path (generated images): read directly from disk
  - data:image/... (base64 URL): decode inline
  - https:// (Unsplash or other): download via requests

Setup:
    pip install playwright colorthief pillow requests anthropic
    playwright install chromium

HTML templates:
    src/agents/moodboard_layout/html/template_F.html  (6 images)
    src/agents/moodboard_layout/html/template_G.html  (7 images)
    src/agents/moodboard_layout/html/template_B.html  (7 images, alt)
    src/agents/moodboard_layout/html/template_D.html  (8 images)
    src/agents/moodboard_layout/html/template_E.html  (9 images)
"""

import os
import time
import base64
import shutil
import argparse
import requests
import anthropic
from PIL import Image
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv

load_dotenv()

CLAUDE_MODEL = os.getenv("GROUNDING_MODEL", "claude-haiku-4-5-20251001")

BASE_DIR     = Path(__file__).resolve().parents[3]
HTML_DIR     = Path(__file__).resolve().parent / "html"
TEMPLATE_DIR = Path(__file__).resolve().parent / "rendered_templates"
OUTPUT_DIR   = BASE_DIR / "outputs/moodboard"

UNSPLASH_PARAMS = {"w": "1200", "q": "85", "fm": "jpg"}

# Maps image count to valid template keys
TEMPLATES_BY_COUNT = {
    6: ["F"],
    7: ["G", "B"],
    8: ["D"],
    9: ["E", "D"],
}

TEMPLATE_META = {
    "F": {
        "description": (
            "Full-width hero at top. Two equal images. "
            "Color swatches + two images. Full-width bottom. "
            "Best for: 6 images, clean editorial look."
        ),
        "image_slots": ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6"],
        "n_images": 6,
    },
    "G": {
        "description": (
            "Large left + two stacked right. Three equal middle row. "
            "Color swatches + wide image bottom. "
            "Best for: 7 images, mix of tall and wide."
        ),
        "image_slots": ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"],
        "n_images": 7,
    },
    "B": {
        "description": (
            "Large image top-left, medium top-right with color swatches. "
            "Three equal images in middle row. Small + wide in bottom row. "
            "Best for: 7 images, mix of portrait and landscape."
        ),
        "image_slots": ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7"],
        "n_images": 7,
    },
    "D": {
        "description": (
            "One full-width hero image at top. Two equal images below. "
            "Three images + color swatches in third row. Wide + small at bottom. "
            "Best for: 8 images, at least one strong wide landscape image."
        ),
        "image_slots": ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8"],
        "n_images": 8,
    },
    "E": {
        "description": (
            "Left column: tall portrait + color swatches + portrait. "
            "Right column: wide top, two equal middle, three small bottom. "
            "Best for: 9 images, mix with at least 2 strong portrait images."
        ),
        "image_slots": ["image_1", "image_2", "image_3", "image_4", "image_5", "image_6", "image_7", "image_8", "image_9"],
        "n_images": 9,
    },
}


class MoodBoardLayoutAgent:
    """
    Renders a mood board from a list of image dicts and optional grounding output.

    Usage:
        agent = MoodBoardLayoutAgent()
        path  = agent.run(images, grounding_output)
    """

    def __init__(self):
        self.claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, images: list[dict], grounding_output: dict | None = None) -> str:
        """
        Build a mood board PNG from image dicts.

        Args:
            images          : list of image dicts. Each dict may have:
                                - local_path  : path to a locally saved file (generated images)
                                - image_url   : https:// URL or data:image/... base64 URI
                              Uses up to 9 images; requires at least 6.
            grounding_output: grounding dict for color palette and prompt context.

        Returns:
            Local path to the final mood board PNG.
        """
        grounding_output = grounding_output or {}
        n = len(images)
        print(f"[MoodBoard] Building board from {n} image(s)...")

        if n < 6:
            raise ValueError(f"[MoodBoard] Need at least 6 images, got {n}.")

        # Load PIL images — handles local_path, base64, and http URLs
        pil_images = self._load_images(images)

        # Deduplicate
        unique, seen = [], set()
        for img in pil_images:
            h = hash(img.tobytes())
            if h not in seen:
                seen.add(h)
                unique.append(img)

        # Clamp to 6–9
        pil_images = unique[:9]
        n_actual   = len(pil_images)
        images     = images[:n_actual]

        if n_actual < 6:
            raise ValueError(f"[MoodBoard] Could only load {n_actual} images (need 6+).")
        print(f"[MoodBoard] Loaded {n_actual} image(s).")

        # Pick template based on count + aspect ratios
        key = self._pick_template(pil_images, n_actual)
        print(f"[MoodBoard] Template: '{key}' ({n_actual} images)")

        # Extract color palette
        colors = self._extract_colors(pil_images, grounding_output)
        print(f"[MoodBoard] Colors: {colors}")

        # Render HTML → PNG
        template_png = TEMPLATE_DIR / f"template_{key}_filled.png"
        self._render_html(key, images, colors, template_png)
        print(f"[MoodBoard] Template rendered → {template_png}")

        # Copy rendered PNG to output directory as the final result
        out = OUTPUT_DIR / f"moodboard_{int(time.time())}.png"
        shutil.copy(template_png, out)
        print(f"[MoodBoard] Saved → {out}")
        return str(out)

    # ── Image loading ─────────────────────────────────────────────────────────

    def _load_images(self, images: list[dict]) -> list:
        """
        Load PIL images from a list of image dicts.
        Priority: local_path > base64 data URI > https URL.
        """
        result = []
        for i, d in enumerate(images):
            try:
                pil = self._load_one(d)
                if pil:
                    result.append(pil)
                    print(f"  [{i+1}] {pil.size} OK")
                else:
                    print(f"  [{i+1}] No loadable source found")
            except Exception as e:
                print(f"  [{i+1}] Failed to load: {e}")
        return result

    def _load_one(self, d: dict) -> Image.Image | None:
        """Load a single image from a result dict."""
        # 1. Local file (generated images saved to disk)
        local = d.get("local_path")
        if local and os.path.exists(local):
            return Image.open(local).convert("RGB")

        url = d.get("image_url", "")
        if not url:
            return None

        # 2. Inline base64 data URI
        if url.startswith("data:image"):
            b64_data = url.split(",", 1)[1]
            return Image.open(BytesIO(base64.b64decode(b64_data))).convert("RGB")

        # 3. Remote URL (Unsplash or other)
        if url.startswith("http"):
            params = UNSPLASH_PARAMS if "images.unsplash.com" in url else {}
            resp   = requests.get(url, params=params, timeout=20)
            resp.raise_for_status()
            return Image.open(BytesIO(resp.content)).convert("RGB")

        return None

    # ── Image embedding for HTML ──────────────────────────────────────────────

    def _img_to_base64(self, d: dict) -> str:
        """
        Convert an image dict to a base64 data URI for embedding in HTML.
        Handles local_path, base64 URL, and http URL.
        Resizes to max 1600px on the long side to keep HTML file size manageable.
        """
        try:
            img = self._load_one(d)
            if img is None:
                return ""
            # Resize if needed
            w, h  = img.size
            scale = min(1600 / w, 1600 / h, 1.0)
            if scale < 1.0:
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            b64 = base64.b64encode(buf.getvalue()).decode()
            return f"data:image/jpeg;base64,{b64}"
        except Exception as e:
            print(f"  [embed] Failed: {e}")
            return ""

    # ── Template selection ────────────────────────────────────────────────────

    def _pick_template(self, pil_images: list, n: int) -> str:
        """
        Select the best template key for the given image count and orientations.
        If only one template is valid for the count, return it immediately.
        Otherwise ask Claude to pick based on portrait/landscape/square counts.
        """
        valid_keys = TEMPLATES_BY_COUNT.get(n)
        if not valid_keys:
            if n <= 6:   valid_keys = ["F"]
            elif n == 7: valid_keys = ["G", "B"]
            elif n == 8: valid_keys = ["D"]
            else:        valid_keys = ["E"]

        if len(valid_keys) == 1:
            return valid_keys[0]

        # Classify orientations
        orientations = []
        for img in pil_images:
            w, h = img.size
            r    = w / h
            if r >= 1.3:   orientations.append("landscape")
            elif r <= 0.8: orientations.append("portrait")
            else:          orientations.append("square")

        counts  = {k: orientations.count(k) for k in ["landscape", "portrait", "square"]}
        options = "\n".join(
            f"  {k}: {TEMPLATE_META[k]['description']}" for k in valid_keys
        )

        prompt = (
            f"I have {n} mood board images:\n"
            f"  portrait={counts['portrait']}, landscape={counts['landscape']}, "
            f"square={counts['square']}\n\n"
            f"Templates:\n{options}\n\n"
            f"Reply with ONLY one letter: {'/'.join(valid_keys)}."
        )

        try:
            resp = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = resp.content[0].text.strip().upper()
            if choice in valid_keys:
                return choice
        except Exception as e:
            print(f"[MoodBoard] Template selection failed: {e}")

        return valid_keys[0]

    # ── Color extraction ──────────────────────────────────────────────────────

    def _extract_colors(self, pil_images: list, grounding_output: dict, n: int = 5) -> list[str]:
        """
        Extract a magazine-style color palette from the images.
        Uses colorthief to gather candidates, then scores and deduplicates them.
        Falls back to neutral editorial tones if colorthief is not installed.
        """
        try:
            from colorthief import ColorThief

            candidates = []
            for img in pil_images[:6]:
                buf = BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                buf.seek(0)
                candidates.extend(ColorThief(buf).get_palette(color_count=6, quality=4))

            if not candidates:
                return self._fallback_palette()

            scored = sorted(
                [(rgb, self._color_score(rgb, grounding_output)) for rgb in candidates],
                key=lambda x: x[1], reverse=True,
            )

            selected = []
            for rgb, _ in scored:
                if not self._too_close(rgb, selected):
                    selected.append(rgb)
                if len(selected) >= n:
                    break

            while len(selected) < n:
                selected.append((200, 200, 195))

            return [self._rgb_to_hex(c) for c in selected[:n]]

        except ImportError:
            print("[MoodBoard] colorthief not installed — using fallback palette.")
            return self._fallback_palette()

    def _color_score(self, rgb, grounding_output) -> float:
        """
        Score a candidate color for editorial quality and query alignment.
        Prefers mid-brightness, low-saturation tones; adjusts for warm/cool/neutral palette hints.
        """
        r, g, b    = rgb
        brightness = (r + g + b) / 3
        saturation = max(rgb) - min(rgb)
        warmth     = r - b

        text  = grounding_output.get("color_palette", "").lower()
        mood  = grounding_output.get("mood", "").lower()

        score = 100 - abs(brightness - 170)  # prefer mid-brightness

        if saturation > 180:   score -= 80
        elif saturation > 120: score -= 30
        else:                  score += 20

        if any(k in text for k in ["warm", "beige", "peach", "pink"]):
            score += max(0, warmth) * 0.5
        if any(k in text for k in ["cool", "blue", "green"]):
            score += max(0, b - r) * 0.5
        if any(k in text for k in ["neutral", "soft", "muted"]):
            score += 40 - abs(saturation - 60)
        if any(k in mood for k in ["luxury", "editorial"]):
            if brightness > 230 or brightness < 50: score -= 40
            if saturation > 150:                     score -= 50

        if r < 40 and g < 40 and b < 40:   score -= 100  # too dark
        if r > 240 and g > 240 and b > 240: score -= 80   # too white

        return score

    def _too_close(self, rgb, selected, threshold: int = 45) -> bool:
        """Return True if rgb is too similar to any already-selected color."""
        return any(self._cdist(rgb, s) < threshold for s in selected)

    def _rgb_to_hex(self, rgb) -> str:
        return "#{:02x}{:02x}{:02x}".format(*rgb)

    def _fallback_palette(self) -> list[str]:
        return ["#d8cfc4", "#a8b5ad", "#c9a27e", "#6e8577", "#eee6da"]

    def _cdist(self, c1, c2) -> float:
        return sum((a - b) ** 2 for a, b in zip(c1, c2)) ** 0.5

    # ── HTML rendering ────────────────────────────────────────────────────────

    def _render_html(
        self,
        key: str,
        images: list[dict],
        colors: list[str],
        out_path: Path,
    ) -> None:
        """
        Fill the HTML template with base64-embedded images and color swatches,
        then render to a PNG via Playwright (headless Chromium).

        Images are embedded as base64 data URIs so Playwright needs no network
        requests — the screenshot is always complete regardless of network speed.
        """
        html_path = HTML_DIR / f"template_{key}.html"
        if not html_path.exists():
            raise FileNotFoundError(
                f"HTML template not found: {html_path}\n"
                f"Expected templates in: {HTML_DIR}"
            )

        html  = html_path.read_text(encoding="utf-8")
        slots = TEMPLATE_META[key]["image_slots"]

        # Embed each image as a base64 data URI
        print(f"[MoodBoard] Embedding {len(slots)} images as base64...")
        for i, slot in enumerate(slots):
            data_uri = self._img_to_base64(images[i]) if i < len(images) else ""
            html = html.replace(f"{{{{{slot}}}}}", data_uri)

        # Fill color swatches
        for i in range(1, 6):
            color = colors[i - 1] if i - 1 < len(colors) else "#cccccc"
            html  = html.replace(f"{{{{color_{i}}}}}", color)

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page    = browser.new_page(viewport={"width": 2400, "height": 4000})
                # domcontentloaded is sufficient since all images are inline base64
                page.set_content(html, wait_until="domcontentloaded")
                page.wait_for_timeout(800)
                # Resize viewport to actual content height before screenshot
                height = page.evaluate("document.body.scrollHeight")
                page.set_viewport_size({"width": 2400, "height": max(height, 100)})
                page.wait_for_timeout(200)
                page.screenshot(path=str(out_path), full_page=True)
                browser.close()
        except ImportError:
            raise ImportError(
                "Playwright not installed.\n"
                "Run: pip install playwright && playwright install chromium"
            )

    # ── Color helpers (kept for potential future use) ─────────────────────────

    def _resize_keep_aspect(self, img: Image.Image, max_px: int = 1024) -> Image.Image:
        """Resize image so the longest side is max_px, preserving aspect ratio."""
        w, h  = img.size
        scale = min(max_px / w, max_px / h, 1.0)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img


# ── CLI: pre-render blank template PNGs for inspection ───────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render-templates", action="store_true",
        help="Render all HTML templates to PNG with placeholder images (run once to inspect layouts)",
    )
    args = parser.parse_args()

    if args.render_templates:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("Run: pip install playwright && playwright install chromium")
            exit(1)

        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

        # 1x1 transparent PNG as placeholder
        placeholder = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="
        )
        neutral = ["#d4c5b5", "#9fb0a8", "#c8a882", "#6b8275", "#e8ddd0"]

        for key in TEMPLATE_META:
            html_path = HTML_DIR / f"template_{key}.html"
            if not html_path.exists():
                print(f"  Missing: {html_path}")
                continue

            html = html_path.read_text(encoding="utf-8")
            for slot in TEMPLATE_META[key]["image_slots"]:
                html = html.replace(f"{{{{{slot}}}}}", placeholder)
            for i in range(1, 6):
                html = html.replace(f"{{{{color_{i}}}}}", neutral[i - 1])

            out = TEMPLATE_DIR / f"template_{key}_blank.png"
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page    = browser.new_page(viewport={"width": 2400, "height": 4000})
                page.set_content(html, wait_until="domcontentloaded")
                page.wait_for_timeout(500)
                height = page.evaluate("document.body.scrollHeight")
                page.set_viewport_size({"width": 2400, "height": max(height, 100)})
                page.screenshot(path=str(out), full_page=True)
                browser.close()
            print(f"  Rendered → {out}")