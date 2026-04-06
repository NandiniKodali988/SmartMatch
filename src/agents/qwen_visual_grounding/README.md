# Visual Concept Grounding Agent (powered by Claude)

Transforms abstract user text into structured visual descriptions for downstream SigLIP-2 embedding and image retrieval. Also generates natural-language justifications for recommended images.

## Setup

Install dependencies and configure your environment from the project root:

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your Anthropic API key to .env
```

## Usage

### Visual Grounding

```python
from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent

agent = QwenVisualGroundingAgent()
result = agent.run("feeling burnt out after a long week")
print(result)
```

Output:
```json
{
  "visual_description": "A person slumped at a cluttered desk, head resting on their arms, surrounded by empty coffee cups and scattered papers under dim lamp light",
  "scene": "home office, nighttime, desk",
  "mood": "exhausted, burnt out, drained",
  "style": "candid, warm low-light, realistic"
}
```

### Justification

```python
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent

agent = QwenJustificationAgent()
results = agent.run(user_text, images)  # images is the list returned by the retrieval agent
```

Each image dict in the returned list will have a `"justification"` field added.

## Pipeline Position

```
User Text → [Visual Grounding] → visual_description → SigLIP-2 Encoder → FAISS Retrieval
                                                                                  ↓
                                               User Text + Images → [Justification] → Output
```
