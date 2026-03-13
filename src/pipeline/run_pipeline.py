from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.content_router.agent import ContentRouterAgent


def run_pipeline(user_text):

    qwen_agent = QwenVisualGroundingAgent()
    justification_agent = QwenJustificationAgent()
    router = ContentRouterAgent()
    siglip_agent = SiglipImageRetrievalAgent()

    # Step 1: visual grounding — convert abstract text into structured visual concepts
    grounding = qwen_agent.run(user_text)

    # Step 2: choose retrieval strategy based on grounding output
    route = router.route(grounding)

    if route == "unsplash":

        # Step 3: retrieve top-K matching images via SigLIP-2 + FAISS
        images = siglip_agent.retrieve(grounding)

        # Step 4: explain why each image matches the user's original text
        images = justification_agent.run(user_text, images)

        return images

    return []


if __name__ == "__main__":

    query = "Walking alone through a rainy city street at night"

    results = run_pipeline(query)

    print(results)