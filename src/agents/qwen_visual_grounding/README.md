# Qwen2.5-VL Visual Concept Grounding Agent

Transforms abstract user text into structured visual descriptions for downstream SigLip-2 embedding and image retrieval.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your HuggingFace token to .env
```

## Usage

```python
from agent import ground_visual_concepts

result = ground_visual_concepts("feeling burnt out after a long week")
print(result)
```

## Output Format

```json
{
  "visual_description": "A person slumped at a cluttered desk, head resting on their arms, surrounded by empty coffee cups and scattered papers under dim lamp light",
  "scene": "home office, nighttime, desk",
  "mood": "exhausted, burnt out, drained",
  "style": "candid, warm low-light, realistic"
}
```

## Pipeline Position

```
User Text → [This Agent] → visual_description → SigLip-2 Encoder → FAISS Retrieval
```
