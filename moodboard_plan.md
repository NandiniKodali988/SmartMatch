# SmartMatch - Mood Board Generator Plan

## What We're Building

We're evolving SmartMatch from a basic image recommender into an **AI-powered mood board generator**. The idea is simple - a user describes a feeling, a campaign, or a concept, and the system builds a cohesive visual board of 6-9 images that work together. They can then refine it through conversation ("make it warmer", "remove the corporate-looking ones") and the system adapts.

The target user is a **marketing professional or content creator** who needs visuals for a campaign, social post, or article - someone who would normally spend hours manually pulling images from Pinterest or Unsplash and arranging them into a mood board.

---

## Why This Is Different

Most image retrieval systems are one-shot - you type something, you get images, done. What makes this interesting is:

- **The system actually looks at the images it retrieves** (using Claude Vision) to check they match before showing them to the user. Most retrieval systems skip this step entirely.
- **It thinks about the board as a whole**, not just individual images. A good mood board tells a story - the images should feel like they belong together.
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

### Visual Grounding Agent - minor update needed
Already exists and works well. The only change is it needs to accept the **previous grounding output** alongside the new refinement request, so it can update specific fields (like color_palette or mood) without starting from scratch every turn.

### Hybrid Retrieval - no changes
Already working. We just need to increase how many candidates it returns (from 3 to around 15-20) so the verification and coherence steps have enough to work with.

### Multimodal Verification Agent - new
This is one of the more interesting additions. After retrieval, we pass each image URL to Claude along with the grounding output and ask it to confirm whether the image actually matches. This is something most retrieval systems don't do - they just trust the embedding score. By adding this step we catch cases where the score is high but the image is clearly wrong for the context.

### Generation Fallback - no changes
DALL-E 3 already works. It gets triggered when the verified pool is too small to fill a mood board.

### Coherence Agent - new
Once we have a pool of verified images, we need to pick the final 6-9 that work together. This agent looks at the set as a whole - are the color palettes consistent? Is there too much visual repetition? Does the mood feel coherent across all images? It scores and selects accordingly.

### Justification Agent - minor update needed
Already writes per-image justifications. We want to add a **board-level summary** - one short paragraph that explains why these images work together as a set.

### Memory Manager - new
Keeps track of the session: what the user has seen, what they liked, what they asked to remove, and how the grounding output has evolved across turns. On each new turn, this context gets passed back to the grounding agent so refinements are actually meaningful.

---

## App Changes

The Streamlit app needs a few key updates:

- **Grid layout** instead of 3 columns - 6-9 images arranged like an actual mood board
- **Chat input** that stays visible so the user can refine at any point
- **Like/dislike on each image** - this feeds directly into the memory manager
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
Same 50 queries, rated by 5 evaluators using the same 4 criteria. The interesting angle here is comparing human ratings to LLM ratings - do they agree? Where do they diverge? That's a finding worth writing about.

**Baselines**
- Random selection
- Text-only search (no grounding agent)
- Image-only retrieval (no field text)
- Single-turn vs multi-turn (does refining actually improve scores?)

---

## Additions from the Course

These are additions inspired by what we covered in class. Each one strengthens a different part of the project.

### Graph RAG

Right now retrieval ranks images independently by score. The problem is that a mood board needs images that work *together*, not just images that are individually relevant. Graph RAG solves this.

We build a knowledge graph over the 25,000 images using the grounding outputs we already have. Each image is a node. Edges connect images that share attributes like mood cluster, overlapping color palette, similar scene type, and style. When a user submits a query, retrieval traverses the graph to find clusters of images that are mutually related, not just individually similar.

The data to build this graph already exists in `description_grounding_outputs.json`. This feeds directly into the coherence agent and makes mood board quality noticeably better.

### MCP Server

We can wrap the SmartMatch pipeline as an MCP server, which means any MCP-compatible client (including Claude Code) can call the mood board generator as a tool, just like calling a function. The server would expose two tools: `generate_moodboard(query)` and `refine_moodboard(query, previous_state)`.

MCP is relatively new and very few people have built production MCP servers. It shows the system is designed to plug into real-world AI tooling rather than just run as a standalone demo, which is a strong talking point.

### Observability and Logging

Add structured logging throughout the pipeline so every query, routing decision, retrieval score, verification result, and generation event gets captured. This gives us real data for the evaluation section of the paper and also shows the system was built with production practices in mind.

Every pipeline run would log the query, grounding output, retrieval scores, verification decisions, final images selected, and the board-level coherence score. This feeds directly into the LLM-as-judge evaluation we have planned.

### Guardrails

Add a content safety check at the input stage. Before the query reaches the grounding agent, Claude checks whether the request is appropriate. It is a small addition but it cleanly covers the Responsible AI section of the paper and shows the system handles real-world edge cases.

We can also add output guardrails after verification, flagging any generated images that may contain inappropriate content before they reach the mood board.

### LLM-as-Judge Evaluation

Use Claude to automatically evaluate pipeline outputs at scale. Run 50 queries, have Claude rate each mood board on Relevance, Visual Quality, Coherence, and Aesthetics on a 1 to 5 scale. Compare these ratings against human evaluator ratings. The agreement and disagreement between LLM and human judgement is itself an interesting finding worth writing about.

This also lets us run ablation studies, comparing results with and without the multimodal verification step, and with and without graph RAG, to show that each addition actually moves the needle.

## What Is Already Done and Staying

Everything in the core pipeline is solid and stays as is. The grounding agent, hybrid retrieval, DALL-E 3 fallback, justification agent, and the base Streamlit app are all staying. The new work is the verification agent, coherence agent, memory manager, multi-turn app layer, graph RAG, MCP server, observability, and guardrails. The underlying architecture does not change. We are building on top of what is already there.
