"""
src/app.py

SmartMatch — multi-turn mood board generator.

Run with:
    streamlit run src/app.py
"""

import io
import math
import os
import sys
import tempfile

import streamlit as st
from dotenv import find_dotenv, load_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
load_dotenv(find_dotenv())

from pipeline.run_pipeline import run_pipeline
from agents.memory.memory_manager import MemoryManager

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartMatch", page_icon="🖼️", layout="wide")

COLS_PER_ROW = 3


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defaults = {
        "memory":        MemoryManager(),
        "results":       [],
        "board_summary": "",
        "liked":         set(),
        "disliked":      set(),
        "history":       [],   # [{"query": str, "n": int}]
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save_uploads(files):
    paths, tmps = [], []
    for f in files:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(f.name)[1]
        )
        tmp.write(f.read())
        tmp.flush()
        paths.append(tmp.name)
        tmps.append(tmp)
    return paths, tmps


def _feedback_prefix(results):
    liked_caps    = [r["caption"] for r in results if r.get("photo_id") in st.session_state.liked    and r.get("caption")]
    disliked_caps = [r["caption"] for r in results if r.get("photo_id") in st.session_state.disliked and r.get("caption")]
    parts = []
    if liked_caps:
        parts.append("I liked images showing: " + "; ".join(liked_caps[:3]))
    if disliked_caps:
        parts.append("I did not like images showing: " + "; ".join(disliked_caps[:3]))
    return (" ".join(parts) + ". ") if parts else ""


def _export_grid(results):
    """Build a 3-column PNG grid from result images. Returns bytes or None."""
    try:
        import requests
        from PIL import Image
    except ImportError:
        return None

    cell_w, cell_h = 400, 400
    tiles = []
    for r in results:
        try:
            local = r.get("local_path")
            url   = r.get("image_url", "")
            if local and os.path.exists(local):
                img = Image.open(local).convert("RGB")
            elif url:
                resp = requests.get(url, timeout=10)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            else:
                continue
            tiles.append(img.resize((cell_w, cell_h), Image.LANCZOS))
        except Exception:
            continue

    if not tiles:
        return None

    cols = COLS_PER_ROW
    rows = math.ceil(len(tiles) / cols)
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), (240, 240, 240))
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        grid.paste(tile, (c * cell_w, r * cell_h))

    buf = io.BytesIO()
    grid.save(buf, format="PNG")
    return buf.getvalue()


def _reset():
    for k in ["memory", "results", "board_summary", "liked", "disliked", "history"]:
        st.session_state.pop(k, None)
    st.rerun()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")

    uploaded_files = st.file_uploader(
        "Upload your own images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Upload images to edit/blend instead of retrieving.",
    )
    style_file = st.file_uploader(
        "Style reference (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Used for composite/style-transfer editing.",
    )

    st.divider()

    if st.button("Start Over", use_container_width=True):
        _reset()

    # Export button (only shown when there are results)
    if st.session_state.results:
        grid_bytes = _export_grid(st.session_state.results)
        if grid_bytes:
            st.download_button(
                label="Export Board (PNG)",
                data=grid_bytes,
                file_name="moodboard.png",
                mime="image/png",
                use_container_width=True,
            )

    st.divider()
    st.caption("SmartMatch · DSAN 6725 · Team 04")


# ── Header ────────────────────────────────────────────────────────────────────
st.title("SmartMatch")
st.markdown(
    "Describe a feeling, moment, or idea — SmartMatch builds a mood board for it. "
    "Refine it turn by turn with natural language."
)
st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.history:
    with st.expander(f"Previous turns ({len(st.session_state.history)})", expanded=False):
        for i, turn in enumerate(st.session_state.history, 1):
            st.markdown(f"**Turn {i}:** {turn['query']}  →  {turn['n']} images")


# ── Query input ───────────────────────────────────────────────────────────────
is_refinement = bool(st.session_state.results)
label       = "Refine your board" if is_refinement else "What are you looking for?"
placeholder = "Add more warmth, change the mood to hopeful..." if is_refinement else \
              "e.g. quiet Sunday morning, feeling burnt out after a long week..."
btn_label   = "Refine Board" if is_refinement else "Generate Board"

query = st.text_area(label, placeholder=placeholder, height=80, key="query_input")
run_btn = st.button(btn_label, type="primary", use_container_width=True)


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a description.")
        st.stop()

    # Build enriched query with like/dislike feedback for refinements
    enriched = _feedback_prefix(st.session_state.results) + query.strip() if is_refinement else query.strip()

    # Save uploaded files
    uploaded_paths, _tmps = _save_uploads(uploaded_files) if uploaded_files else (None, [])
    style_path = None
    if style_file:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(style_file.name)[1])
        tmp.write(style_file.read())
        tmp.flush()
        style_path = tmp.name
        _tmps.append(tmp)

    with st.spinner("Building your mood board..."):
        try:
            results, board_summary = run_pipeline(
                user_text=enriched,
                uploaded_image_paths=uploaded_paths,
                style_ref_path=style_path,
                memory=st.session_state.memory,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
        finally:
            for t in _tmps:
                try:
                    os.unlink(t.name)
                except Exception:
                    pass

    if not results:
        st.info("No results returned. Try a different query.")
        st.stop()

    # Save turn to history and update state
    st.session_state.history.append({"query": query.strip(), "n": len(results)})
    st.session_state.results       = results
    st.session_state.board_summary = board_summary
    st.session_state.liked.clear()
    st.session_state.disliked.clear()
    st.rerun()


# ── Board grid ────────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results
    n = len(results)

    st.subheader(f"Mood Board  ({n} image{'s' if n != 1 else ''})")

    for row_start in range(0, n, COLS_PER_ROW):
        row_items = results[row_start : row_start + COLS_PER_ROW]
        # Pad row to full width so layout doesn't shift
        cols = st.columns(COLS_PER_ROW)

        for col, result in zip(cols, row_items):
            photo_id   = result.get("photo_id", f"img_{row_start}")
            source     = result.get("source", "unknown")
            score      = result.get("score", 0.0)
            caption    = result.get("caption", "")
            just       = result.get("justification", "")
            image_url  = result.get("image_url", "")
            local_path = result.get("local_path")

            with col:
                if local_path and os.path.exists(local_path):
                    st.image(local_path, use_container_width=True)
                elif image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.empty()

                source_label = {
                    "unsplash": "📷 Retrieved",
                    "dalle3":   "🎨 Generated",
                }.get(source, f"🔧 {source}")

                st.markdown(f"**{source_label}** · `{score:.3f}`")

                if just:
                    st.caption(just)

                if caption:
                    with st.expander("Caption", expanded=False):
                        st.write(caption)

                # Like / dislike
                liked_state    = photo_id in st.session_state.liked
                disliked_state = photo_id in st.session_state.disliked

                lc, dc = st.columns(2)
                with lc:
                    like_icon = "✅" if liked_state else "👍"
                    if st.button(like_icon, key=f"like_{photo_id}", use_container_width=True):
                        if liked_state:
                            st.session_state.liked.discard(photo_id)
                        else:
                            st.session_state.liked.add(photo_id)
                            st.session_state.disliked.discard(photo_id)
                        st.rerun()
                with dc:
                    dislike_icon = "❌" if disliked_state else "👎"
                    if st.button(dislike_icon, key=f"dislike_{photo_id}", use_container_width=True):
                        if disliked_state:
                            st.session_state.disliked.discard(photo_id)
                        else:
                            st.session_state.disliked.add(photo_id)
                            st.session_state.liked.discard(photo_id)
                        st.rerun()

    # ── Board summary ─────────────────────────────────────────────────────────
    if st.session_state.board_summary:
        st.divider()
        st.markdown("**Why this board works together**")
        st.info(st.session_state.board_summary)
