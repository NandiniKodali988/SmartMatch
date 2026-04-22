# SmartMatch: A Multi-Agent Pipeline for Mood Board Generation via Graph-Augmented Retrieval and Multimodal Synthesis

**Team 04 · DSAN 6725 · Georgetown University · Spring 2026**

Xinzhou Li · Nandini Kodali · Caroline Delva · Qingyang Wang 

---

## Abstract

Visual content selection is a recurring challenge for creatives, marketers, and designers who must identify images that match not just a topic but a specific emotional tone, aesthetic, and compositional intent. Text-based search engines fail at this task because abstract or emotionally rich language does not map naturally to the visual feature spaces that retrieval models operate in. We present SmartMatch, a multi-agent AI system that addresses this gap by generating cohesive nine-image mood boards from free-form natural language input. The system introduces a five-stage pipeline: (1) a Visual Concept Grounding Agent that uses Claude to decompose user intent into structured visual descriptors; (2) a Hybrid Retrieval system combining SigLIP-2 visual embeddings with per-field OpenAI text embeddings over 25,000 Unsplash images; (3) a Graph RAG Agent that builds a knowledge graph over the image corpus and performs candidate deduplication, neighbor expansion, and connectivity-based reranking; (4) a Multimodal Verification and Coherence Agent that selects a visually consistent final set; and (5) a Justification Agent that produces natural-language explanations for each selected image alongside a board-level narrative. When retrieval scores fall below a threshold, the system falls back to gpt-image-1.5 for on-demand image generation. LLM-as-judge evaluation across 50 diverse queries rates the system at 4.1/5.0 on relevance, coherence, visual quality, and aesthetic consistency. An ablation study confirms that each pipeline stage contributes measurably to final board quality, with Graph RAG providing the largest improvement on abstract queries.

---

## 1. Introduction

Selecting images for a mood board is a creative task that requires matching visual output to emotional intent. A user searching for images that evoke "the feeling of being burnt out after a long week" is not describing a literal scene—they are expressing an affective state that should translate into a specific palette, lighting, compositional weight, and subject matter. Conventional image search engines, which rely on keyword matching or single-vector similarity, routinely fail at this translation. The user either receives irrelevant literal results ("tired person at desk") or generic stock imagery that misses the intended tone entirely.

This failure has a structural cause. Vision-language models such as SigLIP-2 learn alignment between text and image in a shared embedding space, but that space is trained primarily on literal descriptions. Emotionally rich language, abstract concepts, and aesthetic intentions are systematically underrepresented. A system designed to bridge this gap must do more than retrieve images—it must interpret intent, structure the retrieval problem, evaluate coherence across a set of images, and explain its selections.

SmartMatch addresses all four requirements through a coordinated multi-agent architecture. The system:

1. Translates free-form text into structured visual descriptors using an LLM grounding step
2. Retrieves candidates via a hybrid embedding strategy that scores images on multiple semantic dimensions simultaneously
3. Applies graph-based reasoning over the image corpus to diversify and rerank candidates
4. Selects a coherent final set using a multimodal verification and coherence agent
5. Provides natural-language justifications for every selected image and a narrative summary for the board as a whole

The key research questions this work addresses are: Does LLM-based visual grounding improve retrieval quality for abstract queries? Does graph-augmented retrieval improve mood board coherence compared to flat similarity search? How well does LLM-as-judge evaluation correlate with the structural properties of the pipeline?

---

## 2. Related Work

### 2.1 Vision-Language Models and Image Retrieval

Contrastive vision-language models such as CLIP [Radford et al., 2021] and SigLIP [Zhai et al., 2023] enable zero-shot image retrieval by projecting text and images into a shared embedding space. SigLIP-2 improves on CLIP by using a sigmoid loss instead of softmax normalization, which improves performance on multi-label and fine-grained retrieval tasks. These models form the backbone of modern semantic image search, but they share a limitation: their text encoders are optimized for literal, descriptive language rather than affective or abstract intent.

Recent work on dense retrieval [Karpukhin et al., 2020] and multi-vector representations [Colbert; Khattab and Zaharia, 2020] suggests that per-field scoring—where different semantic aspects of a query receive independent representations—can improve retrieval quality over single-vector approaches. SmartMatch's hybrid retrieval agent applies this insight directly, scoring images on visual description, mood, and color palette independently before combining scores.

### 2.2 LLM-Augmented Retrieval

Retrieval-Augmented Generation (RAG) [Lewis et al., 2020] and its variants have demonstrated that LLMs can substantially improve retrieval quality by reformulating queries, expanding them with relevant context, or reranking retrieved results. HyDE [Gao et al., 2022] showed that generating a hypothetical document from a query and using that document as the retrieval anchor can outperform direct query embedding, particularly for abstract queries. SmartMatch's grounding agent applies a related principle: rather than generating a hypothetical image, it generates a structured set of visual descriptors that are better aligned with the embedding model's training distribution.

### 2.3 Knowledge Graphs for Recommendation

Graph-based approaches to recommendation systems exploit structural relationships between items to improve diversity and coverage beyond what similarity scores alone can achieve. Classical collaborative filtering and more recent graph neural network approaches [Hamilton et al., 2017; Wang et al., 2019] demonstrate that connectivity in an item graph is a reliable signal for recommendation quality. SmartMatch's Graph RAG agent applies this principle to the image retrieval setting, constructing a graph where edges encode mood-color similarity with a visual diversity penalty and using graph traversal to expand and rerank candidate sets.

### 2.4 Mood Board Generation

Prior work on computational mood board generation [O'Donovan et al., 2014; Kita and Cao, 2015] focused primarily on color harmony and style matching using hand-crafted features. More recent approaches using neural style embeddings [Gatys et al., 2016] improved aesthetic consistency but lacked the ability to incorporate semantic or emotional intent. To our knowledge, SmartMatch is the first system to combine LLM-based intent grounding, hybrid semantic retrieval, graph-augmented candidate expansion, and LLM-based coherence selection in a unified mood board generation pipeline.

---

## 3. System Architecture

SmartMatch is a multi-agent pipeline where each agent handles a distinct transformation of the input before passing structured output to the next stage. The system accepts free-form natural language text (and optionally uploaded images) and returns a nine-image mood board with per-image justifications and a board-level narrative summary. All agents are coordinated by an Orchestrator that manages routing decisions, step timing, structured logging, and safety checks.

![Figure 1: SmartMatch System Architecture](images/Pipeline.png)

### 3.1 Pipeline Overview

The pipeline follows one of two branches depending on whether the user provides uploaded images.

**Branch A (uploaded images):** The Grounding Agent processes user text and uploaded images, then routes directly to the Generation Agent for image editing (inpaint, restyle, blend, collage, or composite). The result passes to the Justification Agent and returns as a MoodBoardBundle.

**Branch B (no uploads):** The Grounding Agent processes user text, then the Hybrid Retrieval system (SigLIP-2 + Field Text) retrieves candidates from the 25,000-image corpus. The Graph RAG Agent refines the candidate set. If the top hybrid score meets or exceeds the threshold of 0.4, the system proceeds through Multimodal Verification and Coherence selection. If not, the system falls back to gpt-image-1.5 for full board generation. In both sub-branches, the Justification Agent produces per-image explanations and a board summary before output.

The threshold routing uses the pre-Graph RAG hybrid score to avoid the confound of graph reranking inflating scores for abstract queries that genuinely lack retrievable matches.

### 3.2 Visual Concept Grounding Agent

The grounding agent addresses the core challenge of the system: emotionally rich or abstract text does not map naturally to visual feature spaces. The agent takes raw user text and transforms it into a structured JSON output using Claude (claude-haiku-4-5-20251001) via the Anthropic API.

| Field | Description |
|---|---|
| `visual_description` | A rich sentence describing what the ideal image would look like |
| `scene` | Scene and setting keywords |
| `mood` | Emotional and atmospheric keywords |
| `style` | Visual style keywords |
| `lighting` | Lighting conditions |
| `color_palette` | Dominant colors and tones |
| `intent` | Likely use case: professional, editorial, or social |

The agent supports multi-turn context: in a follow-up query (e.g., "make it warmer"), it receives the previous grounding output and modifies it rather than starting from scratch. This is managed by the MemoryManager, which stores grounding outputs per session.

For offline preprocessing, the system uses the Anthropic Batch API to compute grounding outputs for all 25,000 images in the corpus at a 50% cost reduction compared to live API calls. These precomputed outputs form the basis for the field-level text embeddings used in retrieval.

### 3.3 Hybrid Retrieval

Retrieval combines two complementary scoring signals.

**SigLIP-2 Retrieval Agent** performs cosine similarity search over a FAISS index of 25,000 pre-computed image embeddings using the `google/siglip-base-patch16-224` model. The concatenated grounding output fields form the query, which is encoded and compared against image embeddings using L2-normalized dot product similarity.

**Field Text Retrieval Agent** scores images using per-field semantic embeddings via OpenAI's `text-embedding-3-large` model. Three fields are embedded independently—`visual_description`, `mood`, and `color_palette`—and scored against the corresponding grounding output fields. This per-field decomposition allows the system to weight semantic alignment across aesthetic dimensions independently.

The final hybrid score combines both signals: `score = 0.7 × siglip_score + 0.3 × text_score`. In practice, SigLIP-2 scores are often negative in high-dimensional space, making the text embedding component the dominant signal for abstract queries. The system retrieves a candidate pool of up to 20 images before Graph RAG processing.

### 3.4 Graph RAG Agent

The Graph RAG Agent applies graph-based reasoning over the image corpus to improve candidate diversity and coherence. The agent operates in three steps: **deduplicate**, **expand**, and **rerank**.

**Graph Construction.** An offline script builds a weighted adjacency list over all 25,000 images using FAISS-based nearest neighbor search per embedding field. The edge weight formula is:

```
edge_weight = 0.5 × mood_sim + 0.5 × color_sim − 0.3 × visual_sim
edge_weight = max(0.0, edge_weight)
```

The positive mood and color terms connect images that share atmosphere and palette. The negative visual description term penalizes near-identical content—images that look the same even if the mood is right. This produces a graph with 743,839 edges across 25,000 nodes (average degree 29.75), where high-weight edges connect images that are tonally coherent but visually distinct.

**Deduplicate.** For each pair of candidates whose edge weight exceeds a threshold (default 0.60), the lower-scoring image is removed. This eliminates near-duplicate images that frequently appear at the top of retrieval results for concrete queries (e.g., multiple near-identical beach sunset photos).

**Expand.** The top-5 seed candidates traverse the graph to contribute up to 2 neighbors each (MAX_PER_SEED=2), adding at most 10 new candidates to the pool. Expanded candidates are scored using real text similarity via `_score_all()` scaled by `TEXT_WEIGHT` (0.3), so they compete on the same scale as hybrid retrieval candidates rather than relying on a proxy score.

**Rerank.** Each candidate's connectivity score is computed as the average edge weight to other candidates in the pool. The final score is a weighted combination of the hybrid score and the connectivity score: `final = (1 − GRAPH_WEIGHT) × hybrid + GRAPH_WEIGHT × graph`, with `GRAPH_WEIGHT = 0.25`. Images that are well-connected to their peers are boosted, rewarding candidates that form coherent clusters within the selected set.

The routing decision always uses the pre-Graph RAG score to avoid inflating scores for abstract queries that should fall back to generation.

### 3.5 Multimodal Verification and Coherence Agents

**Multimodal Verification Agent** passes each candidate image and the grounding output to Claude Vision (`claude-haiku-4-5-20251001`) for direct visual evaluation. The agent checks whether each image genuinely matches the mood, color palette, and intent of the query—not just whether its text caption is semantically similar. Images that fail verification are excluded; if fewer than `MIN_VERIFIED` images pass (default 6), the Generation Agent fills the gap.

**Coherence Agent** receives the top 20 verified candidates and selects the final 9 images that best form a visually consistent mood board. Using Claude, the agent evaluates the candidate set holistically and selects the subset that balances thematic consistency, visual diversity, and aesthetic harmony.

### 3.6 Generation Agent

When retrieval fails (score < `HYBRID_THRESHOLD`), or when the verified pool is insufficient, the Generation Agent synthesizes images using gpt-image-1.5. For full board generation, 9 images are created using style variants that ensure visual diversity: cinematic wide shot, documentary medium shot, and high-contrast expressive. Each variant receives the grounding-derived prompt with different photographic style specifications.

For uploaded-image editing, the agent selects among five modes—inpaint, restyle, blend, collage, composite—using Claude's judgment based on the grounding output and the nature of the uploaded images. A scoring step evaluates generated images against the original text using SigLIP-2 text-text similarity, and a Claude-based prompt refinement step retries if the score falls below a quality threshold.

### 3.7 Justification Agent

The Justification Agent takes the final selected images and the user's original query text and produces: (1) a 2–3 sentence per-image justification explaining concretely why that image matches the query's mood, palette, and intent, and (2) a board-level narrative summary of 3–5 sentences describing the mood board as a whole. Both outputs are generated by Claude (`claude-haiku-4-5-20251001`) and are written in plain language without reference to technical scores or model internals.

### 3.8 Orchestrator and Supporting Components

**OrchestratorAgent** coordinates all pipeline stages, manages lazy agent initialization, handles guardrail checks, and produces a `MoodBoardBundle`—a Pydantic model containing the query, structured grounding output, final image list, routing decision, top hybrid score, board summary, and timestamp.

**GuardrailAgent** performs input and output safety checks using Claude. Input checks block queries that explicitly request harmful content while allowing dark, melancholy, or unconventional creative queries. Output checks filter generated images whose descriptions suggest unsafe content. Both checks fail open on API error to avoid blocking legitimate requests.

**PipelineLogger** writes structured JSONL logs for every run, capturing step timings, routing decisions, retrieval scores, graph RAG statistics, verification outcomes, and the final bundle. These logs are the basis for the ablation study in Section 5.

**MemoryManager** stores grounding outputs per session to support multi-turn refinement (e.g., "make it darker", "add more contrast").

### 3.9 Pydantic State Models

All inter-agent communication uses Pydantic models (`state.py`) rather than raw dictionaries. Key models include `GroundingOutput`, `ImageResult`, `PipelineState`, and `MoodBoardBundle`. Pydantic validation ensures that field mismatches fail loudly at parse time rather than silently propagating through the pipeline.

### 3.10 External Services and Tools

| Component | Service / Tool |
|---|---|
| Visual grounding, verification, coherence, justification | Anthropic API (`claude-haiku-4-5-20251001`) |
| Image retrieval embeddings | SigLIP-2 (`google/siglip-base-patch16-224`) |
| Field-level text embeddings | OpenAI (`text-embedding-3-large`) |
| Fallback image generation and editing | OpenAI (`gpt-image-1`) |
| Similarity search index | FAISS (CPU) |
| Image dataset | Unsplash (25,000 images with pre-computed embeddings) |
| Batch grounding preprocessing | Anthropic Batch API |
| Safety checks | Anthropic API (`claude-haiku-4-5-20251001`) |
| Structured logging | JSONL (custom PipelineLogger) |

---

## 4. Data and Preprocessing

### 4.1 Image Dataset

The image corpus consists of 25,000 photographs from the Unsplash dataset, selected for diversity across subjects, moods, and visual styles. Each image has an associated caption and metadata. The corpus was chosen for its Creative Commons licensing and broad coverage of the aesthetic range relevant to mood board use cases (lifestyle, nature, architecture, portraiture, abstract).

### 4.2 Preprocessing Pipeline

**Grounding outputs** for all 25,000 images were generated offline using the Anthropic Batch API, which processes requests at half the cost of the live API. Each image's caption was passed to the Grounding Agent, producing structured JSON with the seven grounding fields described in Section 3.2.

**SigLIP-2 image embeddings** were computed for all 25,000 images using the `google/siglip-base-patch16-224` model and stored in a FAISS flat inner product index with L2 normalization.

**Field text embeddings** were computed for the `visual_description`, `mood`, and `color_palette` fields of each image's grounding output using OpenAI's `text-embedding-3-large` model and stored as NumPy arrays. Each field produces a (25,000 × 3,072) matrix.

**The image knowledge graph** was built offline using FAISS-based nearest-neighbor search across the three embedding fields, with the edge weight formula described in Section 3.4. Building the graph for 25,000 images takes approximately 8 minutes on CPU.

### 4.3 Data Quality

SigLIP-2 scores in the raw retrieval results are frequently negative (high-dimensional cosine similarities for non-matching pairs are typically near zero or slightly negative). This is expected behavior for the model and is why the hybrid retrieval formula weights text similarity at 70% for most queries. The Graph RAG edge formula uses additive combination rather than multiplicative to preserve sufficient edge weight magnitude for the deduplication and reranking steps.

---

## 5. Evaluation

### 5.1 LLM-as-Judge Evaluation

We evaluate SmartMatch using an LLM-as-judge framework with 50 diverse queries spanning five categories: emotional/lifestyle (e.g., "feeling burnt out after a long week"), seasonal/nature (e.g., "spring blossoms and fresh beginnings"), abstract/conceptual (e.g., "love you to the moon and back"), commercial/editorial (e.g., "minimalist workspace inspiration"), and sensory/atmospheric (e.g., "golden hour on a quiet beach").

For each query, Claude (`claude-haiku-4-5-20251001`) evaluates the resulting mood board on four criteria, each scored on a 1–5 scale:

| Criterion | Description |
|---|---|
| **Relevance** | Do the images match the emotional intent and mood of the query? |
| **Visual Quality** | Are the images high-quality, well-composed, and aesthetically appropriate? |
| **Coherence** | Do the nine images form a visually and tonally consistent set? |
| **Aesthetics** | Does the overall board have a pleasing, unified aesthetic? |

Results across 50 queries:

| Criterion | Mean Score | Std Dev |
|---|---|---|
| Relevance | 4.1 | 0.7 |
| Visual Quality | 4.3 | 0.5 |
| Coherence | 3.9 | 0.8 |
| Aesthetics | 4.0 | 0.6 |
| **Overall** | **4.1** | **0.6** |

Scores are highest for concrete visual queries (e.g., "golden hour on a quiet beach", avg 4.6) and lower for highly abstract emotional queries (e.g., "love you to the moon and back", avg 3.7), consistent with the fundamental challenge of mapping abstract language to visual search.

### 5.2 Ablation Study

To quantify the contribution of each pipeline component, we run an ablation study comparing four conditions across 20 representative queries:

| Condition | Description |
|---|---|
| **A: Baseline** | SigLIP-2 retrieval only, no grounding, no graph RAG |
| **B: + Grounding** | Hybrid retrieval with LLM grounding |
| **C: + Graph RAG** | Hybrid retrieval + grounding + graph RAG (dedup + expand + rerank) |
| **D: Full system** | All components including verification, coherence, and generation fallback |

| Condition | Avg Relevance | Avg Coherence | Dedup Removed | Expand Added |
|---|---|---|---|---|
| A: Baseline | 2.8 | 2.6 | — | — |
| B: + Grounding | 3.5 | 3.1 | — | — |
| C: + Graph RAG | 3.8 | 3.7 | 0.4 avg | 10 avg |
| D: Full system | 4.1 | 3.9 | 0.4 avg | 10 avg |

Key findings:

- **LLM grounding** contributes the largest single improvement (+0.7 relevance), confirming that decomposing abstract queries into structured visual descriptors substantially improves retrieval alignment.
- **Graph RAG** improves coherence more than relevance (+0.6 coherence vs +0.3 relevance), consistent with the design intent: graph-based reranking promotes candidates that are well-connected to their peers, producing more tonally unified sets.
- **Verification and coherence** contribute an additional +0.3 relevance and +0.2 coherence over graph RAG alone.
- For concrete queries, deduplication removes an average of 0.4 near-duplicate candidates per query; for abstract queries, deduplication rarely triggers, confirming that the graph correctly identifies structural repetition rather than over-removing diverse results.

### 5.3 Baseline Comparison

We compare SmartMatch against three baselines using the preliminary evaluation results (8 queries, full pipeline vs. baseline conditions):

| System | Avg Hybrid Score | Routing (retrieval/gen) |
|---|---|---|
| Raw SigLIP-2 | 0.38 | — |
| Hybrid retrieval (no grounding) | 0.44 | 3 / 5 |
| Hybrid retrieval + grounding | 0.48 | 4 / 4 |
| **SmartMatch (full)** | **0.48** | **4 / 4** |

The full SmartMatch system routes 4 of 8 queries to retrieval and 4 to generation, reflecting the distribution of abstract vs. concrete queries in the test set. Retrieval scores above 0.5 are achieved for concrete visual queries; abstract emotional queries fall below threshold and benefit from on-demand generation.

---

## 6. Models and Technologies

### 6.1 Language Models

**Claude (`claude-haiku-4-5-20251001`)** is used for all LLM-dependent steps: visual concept grounding, multimodal verification, coherence selection, justification generation, and safety checks. Claude Haiku was chosen for its balance of speed and quality for structured JSON generation tasks. All Claude API calls use the Anthropic Python SDK.

### 6.2 Vision-Language Models

**SigLIP-2 (`google/siglip-base-patch16-224`)** provides the visual embedding backbone. The model is loaded locally using HuggingFace Transformers and runs on CPU. Text and image embeddings are pre-computed offline; at inference, only the text query embedding is computed live.

### 6.3 Text Embedding Models

**OpenAI `text-embedding-3-large`** provides field-level text embeddings for the retrieval and graph construction steps. The 3,072-dimensional embeddings are stored as NumPy float32 arrays and indexed using FAISS.

### 6.4 Image Generation

**OpenAI `gpt-image-1`** is used for both generation fallback (text-to-image) and uploaded-image editing. The model returns base64-encoded PNG images, which the system saves locally and provides to the user.

### 6.5 Infrastructure

The system runs entirely on consumer hardware (no GPU required). FAISS CPU provides sub-second retrieval over 25,000 images. The Orchestrator initializes agents lazily to avoid loading all models at startup. A complete mood board pipeline run (retrieval path) takes approximately 45–90 seconds, dominated by multiple Claude API calls.

---

## 7. Responsible AI Considerations

### 7.1 Content Safety

The GuardrailAgent applies two layers of safety filtering. Input guardrails block queries that explicitly request harmful content while deliberately allowing dark, melancholy, or unconventional creative queries that are legitimate mood board requests. Output guardrails filter generated images whose captions or justifications describe unsafe content. Both checks use Claude as the judge and fail open on API error to avoid blocking legitimate users.

The system distinguishes between emotional darkness (allowed: grief, burnout, melancholy) and harmful content (blocked: explicit sexual content, graphic violence, content involving minors). This boundary is implemented through carefully designed system prompts in the GuardrailAgent.

### 7.2 Bias and Representation

The Unsplash dataset has known demographic skews toward certain geographies, subjects, and aesthetic styles. This is a limitation of the retrieval path—the system can only recommend what is present in the corpus. For abstract or underrepresented queries, the generation fallback provides an alternative, though generated images may themselves reflect biases from the DALL-E 3/gpt-image-1 training data.

Future work should include demographic analysis of the corpus and evaluation of whether certain query types (e.g., cultural references, non-Western aesthetics) consistently route to generation rather than retrieval due to corpus gaps.

### 7.3 Hallucination and Accuracy

The Justification Agent produces natural-language explanations that describe concrete visual elements of each image. Justifications are grounded in the image caption and the user's query; they do not make factual claims about the world and are therefore low-risk for hallucination. Multimodal Verification provides an additional check against cases where caption-based similarity does not reflect actual visual match.

### 7.4 Privacy

The system does not store user queries or uploaded images beyond the duration of a single session. The MemoryManager maintains in-memory session state that is discarded when the session ends. No user data is logged to persistent storage.

---

## 8. Findings and Discussion

### 8.1 Visual Grounding Is the Largest Single Improvement

The most surprising finding from the ablation study is the magnitude of the grounding agent's contribution. Replacing raw user text with structured visual descriptors raises the average relevance score by 0.7 points. This confirms the hypothesis that abstract language is structurally ill-suited for embedding-based retrieval, and that LLM-based intent decomposition can bridge this gap effectively.

### 8.2 Graph RAG Improves Coherence More Than Relevance

Graph-based reranking increases coherence scores more than relevance scores, which aligns with its design: the connectivity score rewards images that are well-connected to their peers, not images that are necessarily closest to the query. This produces mood boards where images share a consistent visual register, even if individual image-query scores are not maximized. This trade-off is appropriate for the mood board use case, where the board as a whole must feel unified.

### 8.3 SigLIP-2 Scores Are Often Negative

A consistent observation across all evaluations is that SigLIP-2 scores are frequently negative for text queries, even when retrieved images appear visually relevant. This is a known behavior of high-dimensional embedding spaces and is not a model failure—negative cosine similarity simply means the vectors are oriented away from each other in a space where most pairs are near-orthogonal. The hybrid retrieval formula compensates by weighting text similarity at 70%, but this means the visual embedding is contributing less signal than intended. Future work should investigate normalizing or recalibrating SigLIP-2 scores before combining them with text scores.

### 8.4 Abstract Queries Reliably Route to Generation

Queries expressing emotional states or abstract concepts ("feeling burnt out", "love you to the moon and back") consistently score below the 0.5 threshold and route to generation. This is the intended behavior—the 25,000-image corpus does not contain images whose captions literally describe these states. However, it raises a question about whether a larger or differently curated corpus could improve retrieval coverage for emotional queries.

### 8.5 Generation Quality Is Variable

`gpt-image-1` produces high-quality images for concrete visual concepts but is less consistent for abstract emotional queries. For "nostalgia for childhood summers", the system produced a highly scored image (0.93 SigLIP-2 text proxy), suggesting strong prompt-image alignment. For "feeling burnt out", the generation score was lower (0.34), indicating that the generation prompt did not fully capture the intended emotional register. Improving the prompt construction for abstract emotional queries is a key direction for future work.

---

## 9. Conclusion and Future Work

SmartMatch demonstrates that multi-agent coordination, LLM-based grounding, and graph-augmented retrieval can together produce mood boards that align with both the explicit and implicit intent of natural language queries. The system moves beyond single-vector similarity search and keyword matching to produce coherent nine-image boards with natural-language explanations, addressing a genuine gap in existing tools for creative visual content selection.

The key contributions of this work are: (1) a Visual Concept Grounding Agent that maps abstract language to structured visual descriptors; (2) a Graph RAG architecture for image retrieval that uses mood-color similarity with visual diversity penalties to construct a knowledge graph over the image corpus; (3) a multi-stage coherence pipeline combining graph-based reranking, multimodal verification, and LLM-based coherence selection; and (4) an LLM-as-judge evaluation framework for mood board quality.

**Limitations.** The 25,000-image corpus limits coverage for niche aesthetic styles and cultural references. SigLIP-2 negative scores reduce the effectiveness of the visual embedding component. Multi-turn refinement is supported but not yet evaluated systematically.

**Future Work.** Promising directions include: (1) corpus expansion and demographic analysis; (2) recalibration of SigLIP-2 scores for hybrid retrieval; (3) scene-based prompt diversification for the generation agent; (4) systematic evaluation of multi-turn refinement; (5) user study comparing SmartMatch output to human-curated mood boards.

---

## References

[1] Radford, A., Kim, J. W., Hallacy, C., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML 2021*.

[2] Zhai, X., Mustafa, B., Kolesnikov, A., et al. (2023). Sigmoid Loss for Language Image Pre-Training. *ICCV 2023*.

[3] Lewis, P., Perez, E., Piktus, A., et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. *NeurIPS 2020*.

[4] Gao, L., Ma, X., Lin, J., & Callan, J. (2022). Precise Zero-Shot Dense Retrieval without Relevance Labels. *arXiv:2212.10496*.

[5] Karpukhin, V., Oğuz, B., Min, S., et al. (2020). Dense Passage Retrieval for Open-Domain Question Answering. *EMNLP 2020*.

[6] Khattab, O., & Zaharia, M. (2020). ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT. *SIGIR 2020*.

[7] Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *NeurIPS 2017*.

[8] Wang, X., He, X., Wang, M., et al. (2019). Neural Graph Collaborative Filtering. *SIGIR 2019*.

[9] O'Donovan, P., Agarwala, A., & Hertzmann, A. (2014). Color Compatibility From Large Datasets. *SIGGRAPH 2011*.

[10] Gatys, L. A., Ecker, A. S., & Bethge, M. (2016). Image Style Transfer Using Convolutional Neural Networks. *CVPR 2016*.

[11] Anthropic. (2024). Claude Model Card. *anthropic.com*.

[12] Johnson, J., Douze, M., & Jégou, H. (2019). Billion-Scale Similarity Search with GPUs. *IEEE TBDM*.
