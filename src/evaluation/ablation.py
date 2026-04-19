import json
from src.pipeline.run_pipeline import run_pipeline
from src.agents.memory.memory_manager import MemoryManager
from src.agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from src.agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent


test_queries = [
    "feeling burnt out after a long week",
    "celebrating a new job offer",
    "quiet sunday morning with coffee",
]


def run_baseline(query):
    """
    No grounding — just raw text into SigLIP
    """
    siglip = SiglipImageRetrievalAgent()
    fake_grounding = {
        "visual_description": query,
        "scene": "",
        "mood": "",
        "style": "",
    }
    results = siglip.retrieve(fake_grounding)
    return results


def run_with_grounding(query):
    """
    Grounding but NO memory
    """
    grounding_agent = QwenVisualGroundingAgent()
    siglip = SiglipImageRetrievalAgent()

    grounding = grounding_agent.run(query)
    results = siglip.retrieve(grounding)
    return results


def run_with_memory(query1, query2):
    """
    Multi-turn grounding (your contribution)
    """
    memory = MemoryManager()
    grounding_agent = QwenVisualGroundingAgent()
    siglip = SiglipImageRetrievalAgent()

    g1 = grounding_agent.run(query1)
    memory.add(query1, g1)

    g2 = grounding_agent.run(query2, previous_grounding=memory.get_last())
    memory.add(query2, g2)

    results = siglip.retrieve(g2)
    return results


def summarize_results(name, results):
    if not results:
        return {"name": name, "top_score": 0}

    return {
        "name": name,
        "top_score": results[0]["score"],
        "top_caption": results[0].get("caption", "")[:80],
    }


def run_ablation():

    print("\n=== ABLATION STUDY ===\n")

    all_results = []

    for query in test_queries:

        print(f"\nQuery: {query}")

        baseline = run_baseline(query)
        grounding = run_with_grounding(query)

        print("  Baseline done")
        print("  Grounding done")

        all_results.append({
            "query": query,
            "baseline": summarize_results("baseline", baseline),
            "grounding": summarize_results("grounding", grounding),
        })

    # Multi-turn test
    print("\nMulti-turn test...")
    multi = run_with_memory(
        "feeling burnt out after a long week",
        "now I'm outside getting fresh air"
    )

    all_results.append({
        "query": "multi-turn",
        "memory": summarize_results("memory", multi)
    })

    return all_results


if __name__ == "__main__":

    results = run_ablation()

    print("\n=== RESULTS ===\n")

    for r in results:
        print(json.dumps(r, indent=2))