"""
src/agents/moodboard_layout/agent.py

Mood Board Layout Agent (v3)
-----------------------------
Three-step process:

  Step 1: Claude picks best HTML template (B/D/E) based on image aspect ratios
  Step 2: Fill HTML template with real image URLs + extracted colors,
          render via Playwright → template reference PNG
  Step 3: gpt-image-1.5 composite:
            input = [9 source images (aspect-ratio preserved) + template PNG]
            output = final mood board PNG

Key changes from v2:
  - Images passed individually (not squashed into a grid) so model sees real proportions
  - Prompt uses aesthetic language instead of precise layout instructions
  - Prompt is dynamically built from grounding output (mood, style, color palette)

Setup (run once):
    pip install playwright colorthief pillow requests openai anthropic
    playwright install chromium

HTML template files must live at:
    src/agents/moodboard_layout/html/template_B.html
    src/agents/moodboard_layout/html/template_D.html
    src/agents/moodboard_layout/html/template_E.html
"""

import os
import time
import base64
import argparse
import requests
import anthropic
from PIL import Image
from pathlib import Path
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

CLAUDE_MODEL = os.getenv("GROUNDING_MODEL", "claude-haiku-4-5-20251001")
IMAGE_MODEL  = os.getenv("EDIT_MODEL",      "gpt-image-1.5")
DALLE_SIZE   = os.getenv("DALLE_SIZE",      "1024x1024")

BASE_DIR     = Path(__file__).resolve().parents[3]
HTML_DIR     = Path(__file__).resolve().parent / "html"
TEMPLATE_DIR = Path(__file__).resolve().parent / "rendered_templates"
OUTPUT_DIR   = BASE_DIR / "outputs/moodboard"

UNSPLASH_PARAMS = {"w": "1200", "q": "85", "fm": "jpg"}

TEMPLATE_META = {
    "B": {
        "description": (
            "Large image top-left, medium top-right with color swatches. "
            "Three equal images in middle row. Small + wide in bottom row. "
            "Best for: mix of portrait and landscape images."
        ),
        "image_slots": ["image_1","image_2","image_3","image_4","image_5","image_6","image_7"],
    },
    "D": {
        "description": (
            "One full-width hero image at top. Two equal images below. "
            "Three images + color swatches in third row. Wide + small at bottom. "
            "Best for: at least one strong wide landscape image."
        ),
        "image_slots": ["image_1","image_2","image_3","image_4","image_5","image_6","image_7","image_8"],
    },
    "E": {
        "description": (
            "Left column: tall portrait + color swatches + portrait. "
            "Right column: wide top, two equal middle, three small bottom. "
            "Best for: mix with at least 2 strong portrait/vertical images."
        ),
        "image_slots": ["image_1","image_2","image_3","image_4","image_5","image_6","image_7","image_8"],
    },
}


class MoodBoardLayoutAgent:
    """
    Builds a mood board using HTML template + gpt-image-1.5 composite.

    Usage
    -----
        agent = MoodBoardLayoutAgent()
        path  = agent.run(images, grounding_output)
    """

    def __init__(self):
        self.claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, images: list[dict], grounding_output: dict | None = None) -> str:
        """
        Args:
            images          : list of image dicts with 'image_url'. Uses up to 9.
            grounding_output: full grounding dict (mood, style, color_palette, etc.)
                              Used to build the aesthetic prompt. Optional.
        Returns:
            Local path to final mood board PNG.
        """
        images = images[:9]
        grounding_output = grounding_output or {}
        print(f"[MoodBoard] Building board from {len(images)} image(s)...")

        # Download source images preserving original aspect ratios
        pil_images = self._download(images)
        unique = []
        seen = set()

        for img in pil_images:
            h = hash(img.tobytes())
            if h not in seen:
                seen.add(h)
                unique.append(img)

        pil_images = unique[:9]
        if not pil_images:
            raise ValueError("[MoodBoard] No images could be downloaded.")
        print(f"[MoodBoard] Downloaded {len(pil_images)} image(s).")

        # Pick template based on orientations
        key = self._pick_template(pil_images)
        print(f"[MoodBoard] Template: '{key}'")

        # Extract color palette from images
        colors = self._extract_colors(pil_images, grounding_output)
        print(f"[MoodBoard] Colors: {colors}")

        # Render filled HTML → PNG (layout reference)
        template_png = TEMPLATE_DIR / f"template_{key}_filled.png"
        self._render_html(key, images, colors, template_png)
        print(f"[MoodBoard] Template rendered → {template_png}")

        # Build aesthetic prompt from grounding output
        prompt = self._build_prompt(grounding_output, colors)
        print(f"[MoodBoard] Prompt: {prompt[:100]}...")

        # gpt-image-1.5 composite
        print("[MoodBoard] Running gpt-image-1.5 composite...")
        out_path = self._composite(pil_images, template_png, prompt)
        print(f"[MoodBoard] Saved → {out_path}")

        return str(out_path)

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, grounding_output: dict, colors: list[str]) -> str:
        """
        Build an aesthetic prompt from grounding output.
        Uses mood/style/color as creative direction — not precise layout commands.
        The template image handles layout; the prompt handles feel.
        """
        mood       = grounding_output.get("mood", "calm, cohesive, editorial")
        style      = grounding_output.get("style", "editorial, documentary photography")
        color_desc = grounding_output.get("color_palette", "soft neutral tones")
        color_hex  = ", ".join(colors[:3]) if colors else ""

        return (
            "Create a high-end editorial mood board collage using the provided photos. "
            "The composition and color harmony should feel like a high-end magazine or Pinterest mood board. "
            "Use the last image as a loose composition guide — follow its general panel arrangement. "

            "\n\nIMPORTANT LAYOUT RULES: "
            "Do NOT reuse or duplicate any image. Each source photo must appear only once. "
            "Leave generous white margins around the entire canvas (at least 5-10% padding). "
            "Maintain clear spacing between panels — do not let images touch or crowd each other. "

            f"\n\nAesthetic direction: {mood}. "
            f"Visual style: {style}. "
            f"Color palette: {color_desc}"
            + (f" ({color_hex})" if color_hex else "") + ". "

            "\n\nLayout feel: editorial, airy, minimal. "
            "Balanced negative space, not dense. "
            "Some panels should be intentionally smaller to create breathing room. "

            "\n\nUse the actual photo content from the source images — do not alter them. "
            "No duplicates. No cropping that removes key subjects. "
            "White background. Photorealistic result."
        )

    # ── Template selection ────────────────────────────────────────────────────

    def _pick_template(self, pil_images: list) -> str:
        """Claude picks template based on image aspect ratios."""
        orientations = []
        for img in pil_images:
            w, h = img.size
            r = w / h
            if r >= 1.3:
                orientations.append("landscape")
            elif r <= 0.8:
                orientations.append("portrait")
            else:
                orientations.append("square")

        counts = {k: orientations.count(k) for k in ["landscape", "portrait", "square"]}
        options = "\n".join(f"  {k}: {v['description']}" for k, v in TEMPLATE_META.items())

        prompt = (
            f"I have {len(pil_images)} mood board images:\n"
            f"  portrait={counts['portrait']}, landscape={counts['landscape']}, "
            f"square={counts['square']}\n\n"
            f"Templates:\n{options}\n\n"
            f"Reply with ONLY one letter: B, D, or E."
        )

        try:
            resp = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=5,
                messages=[{"role": "user", "content": prompt}],
            )
            choice = resp.content[0].text.strip().upper()
            if choice in TEMPLATE_META:
                return choice
        except Exception as e:
            print(f"[MoodBoard] Template selection failed: {e}")

        # Fallback heuristic
        if counts["portrait"] >= 2:
            return "E"
        if counts["landscape"] >= 3:
            return "D"
        return "B"

    # ── Color extraction ──────────────────────────────────────────────────────

    def _extract_colors(self, pil_images: list, grounding_output: dict, n: int = 5) -> list[str]:
        """
        Extracts a high-quality, magazine-style color palette.

        Strategy:
        1. Extract many candidate colors from images
        2. Score colors using aesthetic + prompt alignment
        3. Enforce diversity + harmony (no muddy or duplicate tones)
        4. Return a curated palette (like Pinterest / editorial boards)
        """
        try:
            from colorthief import ColorThief

            # ── Step 1: collect candidate colors ───────────────────────────────
            candidates = []

            for img in pil_images[:6]:
                buf = BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                buf.seek(0)

                palette = ColorThief(buf).get_palette(color_count=6, quality=4)
                candidates.extend(palette)

            if not candidates:
                return self._fallback_palette()

            # ── Step 2: score colors using prompt + aesthetics ─────────────────
            scored = [(rgb, self._color_score(rgb, grounding_output)) for rgb in candidates]

            # Sort best → worst
            scored.sort(key=lambda x: x[1], reverse=True)

            # ── Step 3: pick diverse + harmonious colors ───────────────────────
            selected = []

            for rgb, _ in scored:
                if not self._too_close(rgb, selected):
                    selected.append(rgb)
                if len(selected) >= n:
                    break

            # Fill if needed
            while len(selected) < n:
                selected.append((200, 200, 195))

            # ── Step 4: return hex palette ─────────────────────────────────────
            return [self._rgb_to_hex(c) for c in selected[:n]]

        except ImportError:
            print("[MoodBoard] colorthief not installed — using fallback palette.")
            return self._fallback_palette()
        
    def _color_score(self, rgb, grounding_output):
        """
        Scores a color based on:
        - brightness balance
        - saturation (avoid overly harsh colors)
        - prompt alignment (warm / cool / neutral / luxury)
        """
        r, g, b = rgb

        # Basic features
        brightness = (r + g + b) / 3
        saturation = max(rgb) - min(rgb)
        warmth = r - b

        text  = grounding_output.get("color_palette", "").lower()
        mood  = grounding_output.get("mood", "").lower()
        style = grounding_output.get("style", "").lower()

        score = 0

        # ── 1. Prefer balanced brightness (editorial look) ──
        score += 100 - abs(brightness - 170)

        # ── 2. Penalize overly saturated colors (too "cheap looking") ──
        if saturation > 180:
            score -= 80
        elif saturation > 120:
            score -= 30
        else:
            score += 20

        # ── 3. Prompt alignment ──
        if any(k in text for k in ["warm", "beige", "peach", "pink"]):
            score += max(0, warmth) * 0.5

        if any(k in text for k in ["cool", "blue", "green"]):
            score += max(0, b - r) * 0.5

        if any(k in text for k in ["neutral", "soft", "muted"]):
            score += 40 - abs(saturation - 60)

        # ── 4. Luxury/editorial filtering ──
        if any(k in mood for k in ["luxury", "editorial"]):
            if brightness > 230 or brightness < 50:
                score -= 40
            if saturation > 150:
                score -= 50

        # ── 5. Avoid ugly colors ──
        if r < 40 and g < 40 and b < 40:
            score -= 100  # too dark
        if r > 240 and g > 240 and b > 240:
            score -= 80   # too white

        return score


    def _too_close(self, rgb, selected, threshold=45):
        """Avoid similar colors (keeps palette clean & diverse)."""
        for s in selected:
            if self._cdist(rgb, s) < threshold:
                return True
        return False


    def _rgb_to_hex(self, rgb):
        return "#{:02x}{:02x}{:02x}".format(*rgb)


    def _fallback_palette(self):
        """Fallback palette (soft editorial neutral tones)."""
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
    ):
        html_path = HTML_DIR / f"template_{key}.html"
        if not html_path.exists():
            raise FileNotFoundError(
                f"HTML template not found: {html_path}\n"
                f"Place template HTML files in: {HTML_DIR}"
            )

        html = html_path.read_text(encoding="utf-8")

        # Fill image slots with URLs (add Unsplash params if needed)
        for i, slot in enumerate(TEMPLATE_META[key]["image_slots"]):
            if i < len(images):
                url = images[i].get("image_url", "")
                if "images.unsplash.com" in url and "?" not in url:
                    url += "?w=1200&q=85&fm=jpg"
            else:
                url = ""
            html = html.replace(f"{{{{{slot}}}}}", url)

        # Fill color slots
        for i in range(1, 6):
            color = colors[i - 1] if i - 1 < len(colors) else "#cccccc"
            html = html.replace(f"{{{{color_{i}}}}}", color)

        try:
            from playwright.sync_api import sync_playwright
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page    = browser.new_page(viewport={"width": 2400, "height": 3200})
                page.set_content(html, wait_until="networkidle")
                page.screenshot(path=str(out_path), full_page=True)
                browser.close()
        except ImportError:
            raise ImportError(
                "Playwright not installed.\n"
                "Run: pip install playwright && playwright install chromium"
            )

    # ── gpt-image-1.5 composite ───────────────────────────────────────────────

    def _composite(
        self,
        pil_images: list,
        template_png: Path,
        prompt: str,
    ) -> Path:
        """
        Pass each source image individually (aspect-ratio preserved, not squashed into a grid)
        + template PNG as the last image.

        gpt-image-1.5 sees the real proportions of each photo, which lets it make
        better layout decisions than when all images are resquished into equal squares.
        """
        image_files = []

        # Add each source image individually, preserving aspect ratio
        # Resize to max 1024px on the long side to stay within token limits
        for i, pil in enumerate(pil_images):
            buf = BytesIO()
            img = self._resize_keep_aspect(pil, max_px=1024)
            img.convert("RGB").save(buf, format="PNG")
            buf.seek(0)
            buf.name = f"photo_{i+1}.png"
            image_files.append(buf)

        # Template PNG goes last — model uses it as layout reference
        tpl_buf = BytesIO()
        Image.open(template_png).convert("RGB").save(tpl_buf, format="PNG")
        tpl_buf.seek(0)
        tpl_buf.name = "template.png"
        image_files.append(tpl_buf)

        try:
            resp = self.openai.images.edit(
                model=IMAGE_MODEL,
                image=image_files,
                prompt=prompt,
                size=DALLE_SIZE,
                n=1,
            )
            item = resp.data[0]
            if hasattr(item, "b64_json") and item.b64_json:
                img_bytes = base64.b64decode(item.b64_json)
            else:
                img_bytes = requests.get(item.url, timeout=30).content

        except Exception as e:
            print(f"[MoodBoard] Composite failed: {e} — using rendered template as fallback.")
            img_bytes = template_png.read_bytes()

        out = OUTPUT_DIR / f"moodboard_{int(time.time())}.png"
        out.write_bytes(img_bytes)
        return out

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resize_keep_aspect(self, img: Image.Image, max_px: int = 1024) -> Image.Image:
        """Resize image so the longest side is max_px, preserving aspect ratio."""
        w, h  = img.size
        scale = min(max_px / w, max_px / h, 1.0)  # never upscale
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def _download(self, images: list[dict]) -> list:
        result = []
        for i, d in enumerate(images):
            url = d.get("image_url", "")
            if not url:
                continue
            try:
                params = UNSPLASH_PARAMS if "images.unsplash.com" in url else {}
                resp   = requests.get(url, params=params, timeout=20)
                resp.raise_for_status()
                pil = Image.open(BytesIO(resp.content))
                result.append(pil)
                print(f"  [{i+1}] {pil.size} OK")
            except Exception as e:
                print(f"  [{i+1}] Failed: {e}")
        return result


# ── CLI: render blank template PNGs ──────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-templates", action="store_true",
                        help="Render HTML templates to PNG with placeholders (run once)")
    args = parser.parse_args()

    if args.render_templates:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("Run: pip install playwright && playwright install chromium")
            exit(1)

        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        placeholder = (
            "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
            "AAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="
        )
        neutral = ["#d4c5b5", "#9fb0a8", "#c8a882", "#6b8275", "#e8ddd0"]

        for key in ["B", "D", "E"]:
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
                page    = browser.new_page(viewport={"width": 2400, "height": 3200})
                page.set_content(html, wait_until="domcontentloaded")
                page.screenshot(path=str(out), full_page=False)
                browser.close()
            print(f"  Rendered → {out}")