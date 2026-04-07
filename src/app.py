"""
src/app.py

SmartMatch — Streamlit demo app.

Run with:
    streamlit run src/app.py
"""

import os
import sys
import tempfile

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# ── Path setup ────────────────────────────────────────────────────────────────
# Allows imports like: from agents.qwen_visual_grounding.agent import ...
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Load .env before importing any agent (agents use os.environ["ANTHROPIC_API_KEY"])
load_dotenv(find_dotenv())

from pipeline.run_pipeline import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartMatch",
    page_icon="🖼️",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("SmartMatch")
st.markdown(
    "Describe a feeling, moment, or idea — SmartMatch finds or generates the perfect image for it."
)
st.divider()

# ── Sidebar: options ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    uploaded_files = st.file_uploader(
        "Upload your own images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        help="Upload images to edit/blend. If omitted, SmartMatch retrieves or generates images.",
    )
    style_file = st.file_uploader(
        "Style reference image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        help="Used for composite/style-transfer editing when you also upload images.",
    )
    st.divider()
    st.caption("SmartMatch · DSAN 6725 · Team 04")

# ── Main: query input ─────────────────────────────────────────────────────────
query = st.text_area(
    "Describe what you're looking for",
    placeholder="e.g. feeling burnt out after a long week, celebrating a new job, quiet Sunday morning...",
    height=100,
)

run_btn = st.button("Find Images", type="primary", use_container_width=True)

# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a description.")
        st.stop()

    # Save uploaded files to temp paths so run_pipeline can read them
    uploaded_image_paths = None
    style_ref_path = None
    _tmp_files = []  # keep references so they aren't GC'd

    if uploaded_files:
        uploaded_image_paths = []
        for uf in uploaded_files:
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=os.path.splitext(uf.name)[1]
            )
            tmp.write(uf.read())
            tmp.flush()
            uploaded_image_paths.append(tmp.name)
            _tmp_files.append(tmp)

    if style_file:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(style_file.name)[1]
        )
        tmp.write(style_file.read())
        tmp.flush()
        style_ref_path = tmp.name
        _tmp_files.append(tmp)

    with st.spinner("Running SmartMatch pipeline..."):
        try:
            results = run_pipeline(
                user_text=query,
                uploaded_image_paths=uploaded_image_paths,
                style_ref_path=style_ref_path,
            )
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    # Clean up temp files
    for tmp in _tmp_files:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if not results:
        st.info("No results returned. Try a different query.")
        st.stop()

    # ── Display results ───────────────────────────────────────────────────────
    st.subheader(f"Results ({len(results)} image{'s' if len(results) != 1 else ''})")

    cols = st.columns(len(results))

    for col, result in zip(cols, results):
        source     = result.get("source", "unknown")
        score      = result.get("score", 0.0)
        caption    = result.get("caption", "")
        just       = result.get("justification", "")
        image_url  = result.get("image_url", "")
        local_path = result.get("local_path")

        with col:
            # Show image
            if local_path and os.path.exists(local_path):
                st.image(local_path, use_container_width=True)
            elif image_url:
                st.image(image_url, use_container_width=True)
            else:
                st.info("Image not available.")

            # Source badge
            source_label = {
                "unsplash":  "📷 Retrieved",
                "dalle3":    "🎨 Generated",
            }.get(source, f"🔧 {source}")

            st.markdown(f"**{source_label}** &nbsp; `score: {score:.3f}`")

            # Justification
            if just:
                st.markdown(f"> {just}")

            # Caption (collapsed by default)
            if caption:
                with st.expander("Caption"):
                    st.write(caption)
