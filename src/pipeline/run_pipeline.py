from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.content_router.agent import ContentRouterAgent


def run_pipeline(user_text):

    qwen_agent = QwenVisualGroundingAgent()
    router = ContentRouterAgent()
    siglip_agent = SiglipImageRetrievalAgent()

    # Step 1: visual grounding
    grounding = qwen_agent.run(user_text)

    # Step 2: choose strategy
    route = router.route(grounding)

    if route == "unsplash":

        images = siglip_agent.retrieve(grounding)

        return images

    return []


if __name__ == "__main__":

    query = "Walking alone through a rainy city street at night"

    results = run_pipeline(query)

    print(results)