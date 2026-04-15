# SmartMatch — Mood Board Generator Plan

## What We're Building

We're evolving SmartMatch from a basic image recommender into an **AI-powered mood board generator**. The idea is simple — a user describes a feeling, a campaign, or a concept, and the system builds a cohesive visual board of 6-9 images that work together. They can then refine it through conversation ("make it warmer", "remove the corporate-looking ones") and the system adapts.

The target user is a **marketing professional or content creator** who needs visuals for a campaign, social post, or article — someone who would normally spend hours manually pulling images from Pinterest or Unsplash and arranging them into a mood board.

---

## Why This Is Different

Most image retrieval systems are one-shot — you type something, you get images, done. What makes this interesting is:

- **The system actually looks at the images it retrieves** (using Claude Vision) to check they match before showing them to the user. Most retrieval systems skip this step entirely.
- **It thinks about the board as a whole**, not just individual images. A good mood board tells a story — the images should feel like they belong together.
- **It remembers the conversation.** If you say "I don't like the city shots", it won't show you city shots again.
- **It can generate when it can't find.** If retrieval doesn't surface enough good images, DALL-E 3 fills the gaps.

---

## The Full Pipeline

```
User Text
  → Visual Grounding          (Claude turns abstract text into structured visual fields)
  → Hybrid Retrieval          (SigLIP image search + field text search, merged)
  → Multimodal Verification   (Claude Vision checks images actually match the query)
  → Score Check
      ├── Enough good images → Coherence Ranking
      └── Not enough         → DALL-E 3 fills gaps → Coherence Ranking
  → Coherence Ranking         (picks images that work well together as a set)
  → Justification             (explains each image + writes a summary for the whole board)
  → Memory Update             (saves what the user liked, disliked, refined)
  → Output: Mood board
```

---

## Agent by Agent

### Visual Grounding Agent — minor update needed
Already exists and works well. The only change is it needs to accept the **previous grounding output** alongside the new refinement request, so it can update specific fields (like color_palette or mood) without starting from scratch every turn.

### Hybrid Retrieval — no changes
Already working. We just need to increase how many candidates it returns (from 3 to around 15-20) so the verification and coherence steps have enough to work with.

### Multimodal Verification Agent — new
This is one of the more interesting additions. After retrieval, we pass each image URL to Claude along with the grounding output and ask it to confirm whether the image actually matches. This is something most retrieval systems don't do — they just trust the embedding score. By adding this step we catch cases where the score is high but the image is clearly wrong for the context.

### Generation Fallback — no changes
DALL-E 3 already works. It gets triggered when the verified pool is too small to fill a mood board.

### Coherence Agent — new
Once we have a pool of verified images, we need to pick the final 6-9 that work together. This agent looks at the set as a whole — are the color palettes consistent? Is there too much visual repetition? Does the mood feel coherent across all images? It scores and selects accordingly.

### Justification Agent — minor update needed
Already writes per-image justifications. We want to add a **board-level summary** — one short paragraph that explains why these images work together as a set.

### Memory Manager — new
Keeps track of the session: what the user has seen, what they liked, what they asked to remove, and how the grounding output has evolved across turns. On each new turn, this context gets passed back to the grounding agent so refinements are actually meaningful.

---

## App Changes

The Streamlit app needs a few key updates:

- **Grid layout** instead of 3 columns — 6-9 images arranged like an actual mood board
- **Chat input** that stays visible so the user can refine at any point
- **Like/dislike on each image** — this feeds directly into the memory manager
- **Board-level summary** shown below the grid
- **Start Over button** to reset the session completely
- **Export option** to save the board as an image

The multi-turn flow would feel something like:

> Turn 1: *"campaign visuals for a sustainable coffee brand"* → full board generated
>
> Turn 2: *"make it more earthy, less corporate"* → grounding updates mood and color_palette, retrieval reruns, board refreshes
>
> Turn 3: *"I like the outdoor shots but remove anything with people"* → memory filters rejections, coherence fills the gaps

---

## Evaluation

We need two types of evaluation:

**LLM-as-Judge (automated)**
Run 50 diverse queries through the pipeline. Use Claude to rate each result on Relevance, Visual Quality, Coherence, and Aesthetics (1-5). This gives us a scalable, structured dataset to analyze. We can compare retrieval vs. generated results, and with vs. without the multimodal verification step.

**Human Evaluation**
Same 50 queries, rated by 5 evaluators using the same 4 criteria. The interesting angle here is comparing human ratings to LLM ratings — do they agree? Where do they diverge? That's a finding worth writing about.

**Baselines**
- Random selection
- Text-only search (no grounding agent)
- Image-only retrieval (no field text)
- Single-turn vs multi-turn (does refining actually improve scores?)

---

## What's Already Done and Staying

Everything in the core pipeline is solid and stays as is — grounding agent, hybrid retrieval, DALL-E 3 fallback, justification agent, and the base Streamlit app. The new work is the verification agent, coherence agent, memory manager, and the multi-turn app layer. The underlying architecture doesn't change, we're building on top of it.
