# SmartMatch: Multimodal Image Recommendation via CLIP and GPT-4o

## Abstract

Selecting an appropriate image to accompany a social media post or article is a common yet time-consuming challenge. Users frequently struggle to identify visuals that align with the tone, content, and emotional intent of their text. This project proposes **SmartMatch**, an AI-powered image recommendation system that leverages CLIP (Contrastive Language-Image Pretraining) and GPT-4o to bridge the semantic gap between textual content and visual media.

The core challenge addressed is the mismatch between abstract or emotionally rich text and the concrete visual representations that CLIP can match. To overcome this, we introduce a **Visual Concept Grounding** step in which GPT-4o transforms user input into visually descriptive language before retrieval. The system then performs semantic search over a curated image library using CLIP embeddings, and falls back to DALL·E 3 image generation when no sufficiently similar image is found. Final candidate images are re-ranked by CLIP similarity and presented to the user with natural-language justifications.

The system is evaluated against three baselines — random selection, text-only keyword search, and image-only retrieval — using CLIP similarity score, user preference rating, and retrieval relevance (NDCG). This project contributes a practical, end-to-end multimodal pipeline that demonstrates the synergy between vision-language models and large language models for real-world creative assistance.

---

## Agent Flow

```
User Text
  → [1] Input Handler
  → [2] Visual Concept Grounding (GPT-4o)
  → [3] CLIP Encoder
  → [4] Semantic Retrieval (FAISS)
  → Score ≥ threshold?
      ├── YES → [6] Re-ranker
      └── NO  → [5] DALL·E 3 Generation → [6] Re-ranker
  → [7] Justification Generator (GPT-4o)
  → [8] Output UI
```

---

## System Pipeline

| Step                        | Description                                                                                                                      | Technology   |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ------------ |
| 1. Input                    | User submits a text snippet (post, caption, article excerpt)                                                                     | —            |
| 2. Visual Concept Grounding | GPT-4o transforms abstract text into visually descriptive language; extracts scene, emotion, style keywords                      | GPT-4o       |
| 3. Semantic Retrieval       | Visual description encoded as CLIP embedding; cosine similarity search against image library (Unsplash / LAION subset via FAISS) | CLIP + FAISS |
| 4. Fallback Generation      | If top similarity < threshold, DALL·E 3 generates 3 candidate images from the visual description                                 | DALL·E 3     |
| 5. Re-ranking               | All candidates (retrieved + generated) re-ranked by CLIP similarity to original text                                             | CLIP         |
| 6. Output                   | Top-3 images returned with natural-language justification and style labels; user feedback collected                              | GPT-4o + UI  |

---

## Key Contributions

- **Visual Concept Grounding**: A novel preprocessing step using GPT-4o to convert abstract user text into CLIP-compatible visual descriptions, directly addressing CLIP's weakness on non-literal language.

- **Hybrid Retrieval-Generation Pipeline**: Combines semantic image retrieval with on-demand DALL·E 3 generation, ensuring relevant output even when the image library lacks suitable matches.

- **Explainable Recommendations**: Each recommended image is accompanied by a GPT-4o-generated natural-language justification, improving user trust and interpretability.

- **Rigorous Baseline Comparison**: System is benchmarked against three baselines with quantitative metrics (CLIP score, NDCG, user preference), ensuring the contribution is measurable and reproducible.

---

## Potential Data Sources

- **Public image datasets**: e.g., Unsplash, Pexels (free stock photo platforms), or academic datasets such as MS-COCO and ImageNet
- **Web-crawled data**: e.g., LAION-5B, which is also the type of large-scale image-text pair data that CLIP itself was trained on
- **Custom annotated libraries**: manually curated and labeled thematic image collections