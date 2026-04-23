"""
src/app.py

SmartMatch — Streamlit demo app.

Run with:
    streamlit run src/app.py

Features:
    - Multi-turn memory between queries
    - Like / Dislike with color border
    - Keep liked + replace rest with new prompt
    - Per-image edit (Image 1, Image 2…)
    - Download each image individually
    - Download full moodboard as stitched PNG
    - Download liked-only moodboard
    - Start Over
"""

import base64
import io
import os
import sys
import tempfile

import requests
import streamlit as st
from dotenv import load_dotenv, find_dotenv
from PIL import Image as PILImage

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
load_dotenv(find_dotenv())

from agents.orchestrator.agent import OrchestratorAgent
from agents.generation.agent import GenerationAgent
from agents.memory.memory_manager import MemoryManager
from agents.field_text_retrieval.agent import FieldTextRetrievalAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent


# ── Cached agents (load once, survive reruns) ─────────────────────────────────
@st.cache_resource
def get_orchestrator():
    return OrchestratorAgent()

@st.cache_resource
def get_generation_agent():
    return GenerationAgent()

@st.cache_resource
def get_text_agent():
    return FieldTextRetrievalAgent(top_k=20)

@st.cache_resource
def get_siglip_agent():
    return SiglipImageRetrievalAgent(top_k=20)

@st.cache_resource
def get_grounding_agent():
    return QwenVisualGroundingAgent()


# ── Session state ─────────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "memory":          MemoryManager(),
        "results":         [],
        "liked":           set(),
        "disliked":        set(),
        "last_grounding":  None,
        "last_query":      "",
        "board_summary":   "",
        "turn":            0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="SmartMatch", page_icon="🖼️", layout="wide")
st.title("SmartMatch 🖼️")
st.markdown("Describe a feeling, moment, or idea — SmartMatch finds or generates the perfect mood board.")
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────
def _save_uploads(files, style):
    paths, style_path, tmp_refs = [], None, []
    for uf in (files or []):
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(uf.name)[1])
        tmp.write(uf.read()); tmp.flush()
        paths.append(tmp.name); tmp_refs.append(tmp)
    if style:
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(style.name)[1])
        tmp.write(style.read()); tmp.flush()
        style_path = tmp.name; tmp_refs.append(tmp)
    return paths or None, style_path, tmp_refs

def _cleanup(tmp_refs):
    for t in tmp_refs:
        try: os.unlink(t.name)
        except Exception: pass

def _fetch_image_bytes(result: dict) -> bytes | None:
    """Return raw PNG bytes from a result dict."""
    local = result.get("local_path")
    if local and os.path.exists(local):
        with open(local, "rb") as f:
            return f.read()
    url = result.get("image_url", "")
    if url.startswith("data:image"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith("http"):
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            return r.content
        except Exception:
            return None
    return None

def _build_moodboard_png(results: list[dict], cols: int = 3) -> bytes:
    """Stitch images into a single moodboard PNG."""
    THUMB = 512
    images = []
    for r in results:
        raw = _fetch_image_bytes(r)
        if raw:
            try:
                img = PILImage.open(io.BytesIO(raw)).convert("RGB").resize((THUMB, THUMB))
                images.append(img)
            except Exception:
                pass

    if not images:
        blank = PILImage.new("RGB", (THUMB, THUMB), color=(30, 30, 30))
        images = [blank]

    rows = (len(images) + cols - 1) // cols
    board = PILImage.new("RGB", (cols * THUMB, rows * THUMB), color=(20, 20, 20))
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        board.paste(img, (c * THUMB, r * THUMB))

    buf = io.BytesIO()
    board.save(buf, format="PNG")
    return buf.getvalue()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Options")
    uploaded_files = st.file_uploader(
        "Upload your own images (optional)",
        type=["png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )
    style_file = st.file_uploader(
        "Style reference image (optional)",
        type=["png", "jpg", "jpeg", "webp"],
    )
    st.divider()

    # ── Export ────────────────────────────────────────────────────────────────
    if st.session_state.results:
        st.subheader("📥 Export")

        # Full moodboard
        if st.button("Build full moodboard PNG", use_container_width=True):
            with st.spinner("Stitching…"):
                st.session_state["_mb_all"] = _build_moodboard_png(
                    st.session_state.results)
        if "_mb_all" in st.session_state:
            st.download_button(
                "⬇️ Download full moodboard",
                data=st.session_state["_mb_all"],
                file_name="smartmatch_moodboard.png",
                mime="image/png",
                use_container_width=True,
            )

        # Liked-only moodboard
        liked_results = [r for r in st.session_state.results
                         if r.get("photo_id") in st.session_state.liked]
        if liked_results:
            if st.button(
                f"Build liked moodboard ({len(liked_results)} images)",
                use_container_width=True,
            ):
                with st.spinner("Stitching…"):
                    st.session_state["_mb_liked"] = _build_moodboard_png(liked_results)
            if "_mb_liked" in st.session_state:
                st.download_button(
                    "⬇️ Download liked moodboard",
                    data=st.session_state["_mb_liked"],
                    file_name="smartmatch_liked.png",
                    mime="image/png",
                    use_container_width=True,
                )

    st.divider()

    if st.session_state.turn > 0:
        st.caption(f"Turn {st.session_state.turn}")
        if st.button("🔄 Start Over", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            _init_state()
            st.rerun()

    st.caption("SmartMatch · DSAN 6725 · Team 04")


# ── Query input ───────────────────────────────────────────────────────────────
query = st.text_area(
    "Describe what you're looking for" if st.session_state.turn == 0
    else "Refine or change direction",
    placeholder=(
        "e.g. feeling burnt out after a long week, quiet Sunday morning..."
        if st.session_state.turn == 0
        else "e.g. make it warmer, more minimalist..."
    ),
    height=90,
)

col_run, col_regen = st.columns([3, 1])

with col_run:
    run_btn = st.button(
        "✨ Generate Board" if st.session_state.turn == 0 else "✨ New Query",
        type="primary",
        use_container_width=True,
    )

with col_regen:
    n_liked    = len(st.session_state.liked)
    n_disliked = len(st.session_state.disliked)
    if n_liked > 0:
        regen_label = f"🔁 Keep {n_liked} liked, replace rest"
    elif n_disliked > 0:
        regen_label = f"🔁 Replace {n_disliked} disliked"
    else:
        regen_label = "🔁 Regenerate all"

    regen_btn = st.button(
        regen_label,
        use_container_width=True,
        disabled=(len(st.session_state.results) == 0),
    )

# Optional new prompt for replacement images
regen_prompt = ""
if st.session_state.results:
    regen_prompt = st.text_input(
        "Optional: new prompt for replacement images (leave blank to reuse original query)",
        placeholder="e.g. warmer tones, add greenery, more abstract...",
        key="regen_prompt_input",
    )


# ── Action: full pipeline run ─────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please enter a description.")
        st.stop()

    uploaded_image_paths, style_ref_path, tmp_refs = _save_uploads(
        uploaded_files, style_file)

    with st.spinner("Running SmartMatch pipeline…"):
        try:
            bundle = get_orchestrator().run(
                query=query,
                uploaded_image_paths=uploaded_image_paths,
                style_ref_path=style_ref_path,
                memory=st.session_state.memory,
            )
            st.session_state.results        = bundle.to_legacy_list()
            st.session_state.board_summary  = bundle.board_summary or ""
            st.session_state.last_grounding = bundle.grounding.to_dict()
            st.session_state.last_query     = query
            st.session_state.turn          += 1
            st.session_state.liked          = set()
            st.session_state.disliked       = set()
            for k in ["_mb_all", "_mb_liked"]:
                st.session_state.pop(k, None)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()
        finally:
            _cleanup(tmp_refs)

    if not st.session_state.results:
        st.info("No results returned. Try a different query.")
        st.stop()


# ── Action: partial regeneration (retrieval-first, no graph/coherence) ──────
if regen_btn and st.session_state.results:
    if st.session_state.last_grounding is None:
        st.warning("Run a query first.")
    else:
        liked_ids = st.session_state.liked
        kept      = [r for r in st.session_state.results
                     if r.get("photo_id") in liked_ids]
        n_needed  = len(st.session_state.results) - len(kept)

        # All photo_ids already in the board — don't retrieve duplicates
        all_board_ids = {r.get("photo_id") for r in st.session_state.results}
        exclude_ids   = all_board_ids  # skip everything already shown

        # Build grounding for the replacement query
        eff_query = regen_prompt.strip() or st.session_state.last_query
        with st.spinner(f"Finding {n_needed} replacement image(s)…"):
            try:
                # Re-ground if user gave a new prompt
                if regen_prompt.strip():
                    new_grounding = get_grounding_agent().run(eff_query)
                else:
                    new_grounding = dict(st.session_state.last_grounding)

                # Hybrid retrieval, skipping already-shown images
                # Request n_needed*3 so we have buffer after filtering
                candidates = get_text_agent().merge_with_siglip(
                    grounding_output=new_grounding,
                    siglip_agent=get_siglip_agent(),
                    exclude_ids=exclude_ids,
                    n=n_needed * 3,
                )

                # Check threshold — use retrieval if score is good enough
                THRESHOLD = float(os.getenv("HYBRID_THRESHOLD", "0.5"))
                top_score = candidates[0]["score"] if candidates else 0.0

                if top_score >= THRESHOLD:
                    # Retrieval path: take top n_needed
                    new_imgs = candidates[:n_needed]
                else:
                    # Generation fallback
                    st.caption(
                        f"Top retrieval score {top_score:.3f} < {THRESHOLD} "
                        f"— generating images instead."
                    )
                    new_imgs = get_generation_agent().run(
                        grounding_output=new_grounding,
                        n=n_needed,
                        user_text=eff_query,
                    )

                st.session_state.results  = kept + new_imgs
                st.session_state.disliked = set()
                for k in ["_mb_all", "_mb_liked"]:
                    st.session_state.pop(k, None)
            except Exception as e:
                st.error(f"Regeneration error: {e}")


# ── Display board ─────────────────────────────────────────────────────────────
if st.session_state.results:
    results = st.session_state.results

    if st.session_state.board_summary:
        st.info(f"🎨 **Board summary:** {st.session_state.board_summary}")

    if n_liked or n_disliked:
        st.caption(
            f"👍 {n_liked} liked  ·  👎 {n_disliked} disliked  ·  "
            f"{len(results) - n_liked - n_disliked} neutral"
        )

    st.divider()

    COLS = 3
    for row_start in range(0, len(results), COLS):
        row_results = results[row_start : row_start + COLS]
        cols = st.columns(COLS)

        for col, result in zip(cols, row_results):
            photo_id   = result.get("photo_id", f"img_{row_start}")
            source     = result.get("source", "unknown")
            score      = result.get("score", 0.0)
            just       = result.get("justification", "")
            caption    = result.get("caption", "")
            image_url  = result.get("image_url", "")
            local_path = result.get("local_path")
            img_idx    = results.index(result) + 1

            is_liked    = photo_id in st.session_state.liked
            is_disliked = photo_id in st.session_state.disliked
            border      = "#2ecc71" if is_liked else "#e74c3c" if is_disliked else "#2a2a2a"

            with col:
                # Image number label
                st.markdown(
                    f"<p style='font-size:11px;color:#888;margin:0 0 3px 0;'>"
                    f"Image {img_idx}</p>",
                    unsafe_allow_html=True,
                )

                # Image with colored border
                st.markdown(
                    f'<div style="border:2px solid {border};border-radius:8px;'
                    f'padding:3px;margin-bottom:6px;">',
                    unsafe_allow_html=True,
                )
                if local_path and os.path.exists(local_path):
                    st.image(local_path, use_container_width=True)
                elif image_url:
                    st.image(image_url, use_container_width=True)
                else:
                    st.info("Image not available.")
                st.markdown("</div>", unsafe_allow_html=True)

                # Source + score
                source_label = {
                    "hybrid":       "📷 Retrieved",
                    "graph_expand": "🕸️ Graph",
                    "dalle3":       "🎨 Generated",
                }.get(source, f"🔧 {source}")
                st.caption(f"{source_label} · {score:.3f}")

                # ── Like / Dislike ────────────────────────────────────────────
                b1, b2 = st.columns(2)
                with b1:
                    if st.button(
                        "✅ Liked" if is_liked else "👍 Like",
                        key=f"like_{photo_id}",
                        use_container_width=True,
                    ):
                        st.session_state.liked.add(photo_id)
                        st.session_state.disliked.discard(photo_id)
                        st.rerun()
                with b2:
                    if st.button(
                        "❌ Disliked" if is_disliked else "👎 Dislike",
                        key=f"dislike_{photo_id}",
                        use_container_width=True,
                    ):
                        st.session_state.disliked.add(photo_id)
                        st.session_state.liked.discard(photo_id)
                        st.rerun()

                # ── Download individual image ─────────────────────────────────
                raw = _fetch_image_bytes(result)
                if raw:
                    st.download_button(
                        f"⬇️ Download Image {img_idx}",
                        data=raw,
                        file_name=f"smartmatch_{img_idx}_{photo_id[:8]}.png",
                        mime="image/png",
                        key=f"dl_{photo_id}",
                        use_container_width=True,
                    )

                # ── Per-image edit ────────────────────────────────────────────
                with st.expander(f"✏️ Edit Image {img_idx}"):
                    st.caption(
                        "Describe what to change about this specific image. "
                        "The original image will be used as a reference."
                    )
                    edit_prompt = st.text_input(
                        "Edit prompt",
                        placeholder="e.g. warmer lighting, remove people, add fog, make it night...",
                        key=f"ep_{photo_id}",
                        label_visibility="collapsed",
                    )
                    if st.button("Apply", key=f"eb_{photo_id}", type="primary"):
                        if not edit_prompt.strip():
                            st.warning("Enter an edit prompt.")
                        else:
                            edit_grounding = dict(st.session_state.last_grounding or {})
                            edit_grounding["visual_description"] = edit_prompt.strip()

                            # Save current image to temp file for editing reference
                            img_bytes = _fetch_image_bytes(result)
                            tmp_path  = None
                            if img_bytes:
                                with tempfile.NamedTemporaryFile(
                                    delete=False, suffix=".png"
                                ) as tf:
                                    img = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
                                    img.thumbnail((1024, 1024), PILImage.LANCZOS)
                                    img.save(tf.name, format="PNG")
                                    tmp_path = tf.name

                            with st.spinner(f"Editing Image {img_idx}…"):
                                try:
                                    edited = get_generation_agent().run(
                                        grounding_output=edit_grounding,
                                        n=1,
                                        user_text=edit_prompt,
                                        uploaded_image_paths=(
                                            [tmp_path] if tmp_path else None
                                        ),
                                    )
                                    if edited:
                                        idx = next(
                                            i for i, r in enumerate(
                                                st.session_state.results)
                                            if r.get("photo_id") == photo_id
                                        )
                                        st.session_state.results[idx] = edited[0]
                                        st.session_state.liked.discard(photo_id)
                                        st.session_state.disliked.discard(photo_id)
                                        for k in ["_mb_all", "_mb_liked"]:
                                            st.session_state.pop(k, None)
                                        st.success(f"Image {img_idx} updated!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(f"Edit failed: {e}")
                                finally:
                                    if tmp_path:
                                        try: os.unlink(tmp_path)
                                        except: pass

                # Justification + caption
                if just:
                    with st.expander("Why this image?"):
                        st.write(just)
                if caption:
                    with st.expander("Caption"):
                        st.write(caption)

        st.divider()