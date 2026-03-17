# Risks and Mitigation Plans - Milestone 2

## Visual Grounding + Justification Agents (Claude)

### Risk 1: Dependency on the Anthropic API
Both agents make live calls to the Anthropic API (currently `claude-haiku-4-5-20251001`), which means the pipeline is only as reliable as that external service. If the API goes down, hits a rate limit, or the key expires, requests will fail.

**Mitigation:** We added retry logic — if a call fails, the agent tries up to 3 times before giving up and returning a safe fallback response. This way the rest of the pipeline doesn't crash even if the model is temporarily unavailable. The model itself is also set through an environment variable, so switching to a different one doesn't require any code changes.

---

### Risk 2: Grounding Quality Directly Impacts Retrieval
The SigLIP-2 retrieval is only as good as what the grounding agent gives it. If the grounding output is too vague or generic, the similarity scores drop and users get poor recommendations — so the quality of the prompts really matters here.

**Mitigation:** We kept all prompts in a separate `prompts.py` file so they can be adjusted and tested without touching the agent logic. This makes it much easier to iterate and improve the grounding quality over time without breaking anything else in the pipeline.
