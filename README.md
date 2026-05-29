# SmartMatch: Multi-Agent Mood Board Generation via Graph-Augmented Retrieval and Multimodal Synthesis

---

## Team 04

[Qingyang Wang](www.linkedin.com/in/qingyang-wang-47675730b/) · [Nandini Kodali](https://www.linkedin.com/in/nandini-kodali/) · [Caroline Delva](https://www.linkedin.com/in/caroline-delva-5184a172/)  · [Xinzhou Li](https://www.linkedin.com/in/xinzhou-li-4189452a4/)

Georgetown University — DSAN 6725: Applied Generative AI for Developers — Spring 2026

---

## Abstract

Visual content selection is a recurring challenge for creatives and marketers who must identify images that match not just a topic but a specific emotional tone, aesthetic intent, and compositional style. Conventional image search fails because abstract or emotionally rich language does not map naturally to the visual feature spaces that retrieval models operate in.

SmartMatch is a multi-agent system that generates cohesive nine-image mood boards from free-form natural language. The pipeline comprises five stages: a **Visual Concept Grounding Agent** that uses Claude to decompose user intent into structured visual descriptors; a **Hybrid Retrieval** system combining SigLIP-2 visual embeddings with per-field OpenAI text embeddings over 25,000 Unsplash images; a **Graph RAG Agent** that builds a knowledge graph over the corpus and performs candidate deduplication, expansion, and reranking; a **Multimodal Verification and Coherence Agent** that selects a visually consistent final set; and a **Justification Agent** that produces natural-language explanations for each image alongside a board-level narrative. When retrieval scores fall below a threshold, the system falls back to gpt-image-1 with Claude-driven diverse prompt synthesis.

LLM-as-judge evaluation across 50 diverse queries yields an overall mean score of **3.45 / 5.0**, with relevance at 4.18, coherence at 3.06, and aesthetics at 3.16.

---

## Pipeline

```
User Input (text + optional images)
  → [1] Input Guardrail
  → [2] Visual Concept Grounding Agent (Claude)
       → visual_description, scene, mood, style, lighting, color_palette, intent

  Branch A (uploaded images)
       → Generation Agent (gpt-image-1, editing mode)

  Branch B (text-only)
       → Hybrid Retrieval (SigLIP-2 × 0.3 + Field Text × 0.7, top-20)
       → Graph RAG: deduplicate → expand → rerank
       → score ≥ 0.5?
           YES → Multimodal Verification → Coherence Agent
           NO  → Generation Agent (diverse prompt synthesis)

  → Justification Agent (per-image + board summary)
  → Output Guardrail
  → Mood Board UI (like/dislike · chat refinement · download)
```

---

## System Components

| Component | Description | Technology |
|---|---|---|
| Visual Concept Grounding | Converts abstract text into structured visual descriptors | Claude (haiku) |
| Hybrid Retrieval | Cosine similarity over SigLIP-2 embeddings + per-field text embeddings | SigLIP-2 + OpenAI + FAISS |
| Graph RAG | Knowledge graph over 25,000 images; dedup, expand, rerank | FAISS + NetworkX |
| Multimodal Verification | Filters candidates against query mood/palette/intent | Claude Vision |
| Coherence Agent | Selects final 9 images balancing consistency and diversity | Claude |
| Generation Agent | Synthesizes images with diverse prompt synthesis + quality retry | gpt-image-1 |
| Justification Agent | Per-image explanations + board narrative | Claude |
| Multi-turn Refinement | Chat interface: like/dislike signals + natural-language feedback steer retrieval across turns | MemoryManager + Streamlit |

---

## Running Locally

### Prerequisites

- Python 3.10+
- API keys: Anthropic, OpenAI, HuggingFace

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Set up environment variables

```bash
cp .env.example .env
```

Fill in `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
HF_TOKEN=hf_...
```

### 3. Download data files

Large data files are hosted on HuggingFace and auto-downloaded on first startup if `HF_TOKEN` is set. To download manually:

```bash
python download_data.py
```

Downloads into `src/data/` (~1 GB total):

| File | Size |
|---|---|
| `embeddings/image_embeddings.npy` | 154 MB |
| `embeddings/field_embeddings.npz` | 922 MB |
| `graph/image_graph.pkl` | ~200 MB |
| `processed/dataset_clean.csv` | — |
| `processed/description_grounding_outputs.json` | — |

### 4. Run the app

```bash
streamlit run src/app.py
```

Open [http://localhost:8501](http://localhost:8501) and describe a mood, feeling, or idea to generate a mood board.

---

## Deploying to HuggingFace Spaces

Live demo: [huggingface.co/spaces/NandiniKodali/smartmatch](https://huggingface.co/spaces/NandiniKodali/smartmatch)

### First-time setup

1. Create a Space — SDK: **Streamlit**, Hardware: **CPU Basic**
2. Add secrets in Space Settings → Variables and Secrets:
   - `ANTHROPIC_API_KEY`
   - `OPENAI_API_KEY`
   - `HF_TOKEN`
3. Add remote: `git remote add space https://huggingface.co/spaces/NandiniKodali/smartmatch`

### Redeploy after changes

HF Spaces rejects binary/large files via standard git push. Use a clean orphan branch:

```bash
git checkout --orphan space-deploy
git add -A
git rm --cached -r outputs/ ProjectPaper/ spring-2025/ spring-2026/ "src/agents/moodboard_layout/rendered_templates/"
git commit -m "deploy: your message here"
git push space space-deploy:main --force
git checkout -f main
git branch -D space-deploy
```

On first startup the Space auto-downloads embedding files (~1 min). Subsequent loads are fast.

---

## Project Structure

```
src/
├── app.py                          # Streamlit UI + multi-turn chat
├── api/server.py                   # FastAPI server
├── pipeline/
│   ├── state.py                    # Pydantic models (GroundingOutput, ImageResult, MoodBoardBundle)
│   ├── logger.py                   # Structured JSONL pipeline logger
│   └── run_pipeline.py             # Pipeline runner
├── agents/
│   ├── orchestrator/               # Coordinates all pipeline stages
│   ├── guardrails/                 # Input / output safety checks
│   ├── qwen_visual_grounding/      # Visual Concept Grounding Agent + Justification Agent
│   ├── siglip_image_retrieval/     # SigLIP-2 visual embedding + FAISS retrieval
│   ├── field_text_retrieval/       # Per-field text embedding + hybrid scoring
│   ├── graph_rag/                  # Graph RAG: dedup → expand → rerank
│   ├── multimodal_verification/    # Claude Vision candidate filtering
│   ├── coherence/                  # Final 9-image coherence selection
│   ├── generation/                 # gpt generation + diverse prompt synthesis
│   ├── moodboard_layout/           # HTML template
│   ├── memory/memory_manager.py    # Cross-turn grounding state
│   └── content_router/             # Branch A / B routing logic
├── data/
│   ├── data_prep/                  # Graph construction, embedding generation, data cleaning
│   ├── embeddings/                 # image_embeddings.npy, field_embeddings.npz (HF-hosted)
│   ├── graph/                      # image_graph.pkl (HF-hosted)
│   └── processed/                  # dataset_clean.csv, description_grounding_outputs.json
├── evaluation/
│   ├── llm_judge.py                # LLM-as-judge scorer (50-query evaluation)
│   └── ablation.py                 # Ablation study runner
└── tools/                          # Diagnostics, patch scripts, comparison utilities
```

---

## Key Contributions

- **Visual Concept Grounding** — Claude converts abstract user text into structured visual descriptors (mood, palette, lighting, intent), directly addressing SigLIP-2's weakness on non-literal language.
- **Graph RAG over image corpus** — A weighted knowledge graph (750K edges, avg degree 30) enables connectivity-based reranking that improves coherence beyond flat similarity search.
- **Routing with pre-Graph RAG score** — Fallback to generation is triggered by the raw hybrid score, preventing artificially inflated post-reranking scores from masking low retrieval quality.
- **Diverse prompt synthesis + quality retry** — Claude generates visually distinct prompts before generation; images scoring below threshold are individually re-prompted without discarding the full batch.
- **Multi-turn refinement loop** — Like/dislike signals and natural-language chat feedback are folded into subsequent grounding calls, steering retrieval toward the user's aesthetic intent across turns.

---

## Deliverables

| Item | File |
|---|---|
| Paper | `Deliverables/FinalPaper.pdf` |
| Slides | `Deliverables/SmartMatch-slides.pdf` |
| Poster | `Deliverables/Poster.pdf` |
| Live demo | [huggingface.co/spaces/NandiniKodali/smartmatch](https://huggingface.co/spaces/NandiniKodali/smartmatch) |

