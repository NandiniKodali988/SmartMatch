"""
src/agents/moodboard_layout/agent.py

Mood Board Layout Agent (v2)
-----------------------------
Three-step process:

  Step 1: Claude picks best HTML template (B/D/E) based on image aspect ratios
  Step 2: Fill HTML template with real image URLs + extracted colors,
          render via Playwright → template reference PNG
  Step 3: gpt-image-1.5 composite:
            input = [source images grid, template PNG]
            output = final mood board PNG

Setup (run once):
    pip install playwright colorthief pillow requests openai anthropic
    playwright install chromium
    python src/agents/moodboard_layout/agent.py --render-templates

HTML template files must live at:
    src/agents/moodboard_layout/html/template_B.html
    src/agents/moodboard_layout/html/template_D.html
    src/agents/moodboard_layout/html/template_E.html
"""

import os
import io
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

UNSPLASH_PARAMS = {"w": "800", "q": "80", "fm": "jpg"}

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

COMPOSITE_PROMPT = (
    "The last image is a layout template. Use it ONLY as a structural guide — "
    "replicate its exact grid layout, panel proportions, white spacing, and color dot positions. "
    "Fill every photo panel with the actual content from the source images in the grid. "
    "Preserve the white background and gaps between panels exactly as shown in the template. "
    "Keep the color dots in their template positions using the same colors shown. "
    "Do not add text, logos, filters, or any elements not in the template. "
    "Do not alter the photos. The result should look like a clean editorial photo mood board."
)


class MoodBoardLayoutAgent:
    """
    Builds a mood board using HTML template + gpt-image-1.5 composite.

    Usage
    -----
        agent = MoodBoardLayoutAgent()
        path  = agent.run(images)
    """

    def __init__(self):
        self.claude = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        self.openai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)

    # ── Public interface ──────────────────────────────────────────────────────

    def run(self, images: list[dict]) -> str:
        """
        Args:
            images : list of image dicts with 'image_url'. Uses up to 9.
        Returns:
            Local path to final mood board PNG.
        """
        images = images[:9]
        print(f"[MoodBoard] Building board from {len(images)} image(s)...")

        # Download source images
        pil_images = self._download(images)
        if not pil_images:
            raise ValueError("[MoodBoard] No images could be downloaded.")
        print(f"[MoodBoard] Downloaded {len(pil_images)} image(s).")

        # Pick template
        key = self._pick_template(pil_images)
        print(f"[MoodBoard] Template: '{key}'")

        # Extract palette
        colors = self._extract_colors(pil_images)
        print(f"[MoodBoard] Colors: {colors}")

        # Render filled HTML → PNG
        template_png = TEMPLATE_DIR / f"template_{key}_filled.png"
        self._render_html(key, images, colors, template_png)
        print(f"[MoodBoard] Template rendered → {template_png}")

        # gpt-image-1.5 composite
        print("[MoodBoard] Running gpt-image-1.5 composite...")
        out_path = self._composite(pil_images, template_png)
        print(f"[MoodBoard] Saved → {out_path}")

        return str(out_path)

    # ── Template selection ────────────────────────────────────────────────────

    def _pick_template(self, pil_images: list) -> str:
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
            f"  portrait={counts['portrait']}, landscape={counts['landscape']}, square={counts['square']}\n\n"
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

    def _extract_colors(self, pil_images: list, n: int = 5) -> list[str]:
        try:
            from colorthief import ColorThief
            all_colors = []
            for img in pil_images[:5]:
                buf = BytesIO()
                img.convert("RGB").save(buf, format="PNG")
                buf.seek(0)
                palette = ColorThief(buf).get_palette(color_count=3, quality=5)
                all_colors.extend(palette)

            seen = []
            for rgb in all_colors:
                if not any(self._cdist(rgb, s) < 40 for s in seen):
                    seen.append(rgb)
                if len(seen) >= n:
                    break
            while len(seen) < n:
                seen.append((200, 200, 195))
            return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in seen[:n]]

        except ImportError:
            print("[MoodBoard] colorthief not installed — using neutral colors.")
            return ["#d4c5b5", "#9fb0a8", "#c8a882", "#6b8275", "#e8ddd0"]

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

        # Fill image slots with URLs
        for i, slot in enumerate(TEMPLATE_META[key]["image_slots"]):
            if i < len(images):
                url = images[i].get("image_url", "")
                if "images.unsplash.com" in url and "?" not in url:
                    url += "?w=800&q=80&fm=jpg"
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
                page.screenshot(path=str(out_path), full_page=False)
                browser.close()
        except ImportError:
            raise ImportError(
                "Playwright not installed.\n"
                "Run: pip install playwright && playwright install chromium"
            )

    # ── gpt-image-1.5 composite ───────────────────────────────────────────────

    def _composite(self, pil_images: list, template_png: Path) -> Path:
        # Pack source images into a grid
        grid_buf  = BytesIO(self._make_grid(pil_images))
        grid_buf.name = "sources.png"

        # Template
        tpl_buf   = BytesIO()
        Image.open(template_png).save(tpl_buf, format="PNG")
        tpl_buf.seek(0)
        tpl_buf.name = "template.png"

        try:
            resp = self.openai.images.edit(
                model=IMAGE_MODEL,
                image=[grid_buf, tpl_buf],
                prompt=COMPOSITE_PROMPT,
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

    def _make_grid(self, pil_images: list, thumb: int = 512) -> bytes:
        imgs = [img.convert("RGB").resize((thumb, thumb), Image.LANCZOS)
                for img in pil_images]
        cols = 3
        rows = (len(imgs) + cols - 1) // cols
        grid = Image.new("RGB", (cols * thumb, rows * thumb), (255, 255, 255))
        for idx, img in enumerate(imgs):
            r, c = divmod(idx, cols)
            grid.paste(img, (c * thumb, r * thumb))
        buf = BytesIO()
        grid.save(buf, format="PNG")
        return buf.getvalue()

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
                        help="Render HTML templates to PNG with placeholders")
    args = parser.parse_args()

    if args.render_templates:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            print("Run: pip install playwright && playwright install chromium")
            exit(1)

        TEMPLATE_DIR.mkdir(parents=True, exist_ok=True)
        placeholder = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVQI12NgAAIABQAABjE+ibYAAAAASUVORK5CYII="
        neutral     = ["#d4c5b5","#9fb0a8","#c8a882","#6b8275","#e8ddd0"]

        for key in ["B", "D", "E"]:
            html_path = HTML_DIR / f"template_{key}.html"
            if not html_path.exists():
                print(f"  Missing: {html_path}")
                continue
            html = html_path.read_text(encoding="utf-8")
            for slot in TEMPLATE_META[key]["image_slots"]:
                html = html.replace(f"{{{{{slot}}}}}", placeholder)
            for i in range(1, 6):
                html = html.replace(f"{{{{color_{i}}}}}", neutral[i-1])

            out = TEMPLATE_DIR / f"template_{key}_blank.png"
            with sync_playwright() as p:
                browser = p.chromium.launch()
                page    = browser.new_page(viewport={"width": 2400, "height": 3200})
                page.set_content(html, wait_until="domcontentloaded")
                page.screenshot(path=str(out), full_page=False)
                browser.close()
            print(f"  Rendered → {out}")