# # from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
# # from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
# # from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
# # from agents.content_router.agent import ContentRouterAgent


# # def run_pipeline(user_text):

# #     qwen_agent = QwenVisualGroundingAgent()
# #     justification_agent = QwenJustificationAgent()
# #     router = ContentRouterAgent()
# #     siglip_agent = SiglipImageRetrievalAgent()

# #     # Step 1: visual grounding — convert abstract text into structured visual concepts
# #     grounding = qwen_agent.run(user_text)

# #     # Step 2: choose retrieval strategy based on grounding output
# #     route = router.route(grounding)

# #     if route == "unsplash":

# #         # Step 3: retrieve top-K matching images via SigLIP-2 + FAISS
# #         images = siglip_agent.retrieve(grounding)

# #         # Step 4: explain why each image matches the user's original text
# #         images = justification_agent.run(user_text, images)

# #         return images

# #     return []


# # if __name__ == "__main__":

# #     query = "Walking alone through a rainy city street at night"

# #     results = run_pipeline(query)

# #     print(results)

# from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
# from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
# from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
# from agents.content_router.agent import ContentRouterAgent
# from agents.generation.agent import GenerationAgent  # ADDED


# def run_pipeline(
#     user_text,
#     uploaded_image_paths=None,  # ADDED: 用户上传的图片路径列表（可选）
#     style_ref_path=None,  # ADDED: composite 模式的风格参考图（可选）
# ):

#     qwen_agent = QwenVisualGroundingAgent()
#     justification_agent = QwenJustificationAgent()
#     router = ContentRouterAgent()
#     siglip_agent = SiglipImageRetrievalAgent()
#     generation_agent = GenerationAgent()  # ADDED

#     # Step 1: visual grounding — convert abstract text into structured visual concepts
#     grounding = qwen_agent.run(user_text)

#     # Step 2: choose retrieval strategy based on grounding output
#     route = router.route(grounding)

#     if route == "unsplash":

#         # Step 3: retrieve top-K matching images via SigLIP-2 + FAISS
#         images = siglip_agent.retrieve(grounding)

#         # ADDED: Step 3.5 — check if retrieval score is sufficient
#         # If not, fall back to GenerationAgent instead of returning low-quality results
#         top_score = images[0]["score"] if images else 0.0
#         SIMILARITY_THRESHOLD = 0.25  # ADDED: same value as in generation/agent.py

#         if top_score < SIMILARITY_THRESHOLD:  # ADDED
#             images = generation_agent.run(  # ADDED
#                 grounding_output=grounding,  # ADDED
#                 n=3,  # ADDED
#                 user_text=user_text,  # ADDED
#                 uploaded_image_paths=uploaded_image_paths,  # ADDED
#                 style_ref_path=style_ref_path,  # ADDED
#                 siglip_agent=siglip_agent,  # ADDED
#             )  # ADDED

#         # Step 4: explain why each image matches the user's original text
#         images = justification_agent.run(user_text, images)

#         return images

#     # ADDED: handle non-unsplash routes (ai_generation, ai_compositing, etc.)
#     # Original code returned [] for these routes — now we generate instead
#     # return []  # COMMENTED OUT: replaced by generation below
#     generated = generation_agent.run(  # ADDED
#         grounding_output=grounding,  # ADDED
#         n=3,  # ADDED
#         user_text=user_text,  # ADDED
#         uploaded_image_paths=uploaded_image_paths,  # ADDED
#         style_ref_path=style_ref_path,  # ADDED
#         siglip_agent=siglip_agent,  # ADDED
#     )  # ADDED
#     return justification_agent.run(user_text, generated)  # ADDED


# if __name__ == "__main__":

#     query = ""

#     results = run_pipeline(query)

#     print(results)

from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.content_router.agent import ContentRouterAgent
from agents.generation.agent import GenerationAgent  # ADDED


def run_pipeline(
    user_text,
    uploaded_image_paths=None,  # ADDED: 用户上传的图片路径列表（可选）
    style_ref_path=None,  # ADDED: composite 模式的风格参考图（可选）
):
    print("\n" + "=" * 60)
    print(f"[Pipeline] START")
    print(f"[Pipeline] user_text        : {user_text}")
    print(f"[Pipeline] uploaded_images  : {uploaded_image_paths}")
    print(f"[Pipeline] style_ref_path   : {style_ref_path}")
    print("=" * 60)

    qwen_agent = QwenVisualGroundingAgent()
    justification_agent = QwenJustificationAgent()
    router = ContentRouterAgent()
    siglip_agent = SiglipImageRetrievalAgent()
    generation_agent = GenerationAgent()  # ADDED

    # ── Step 1: Visual Grounding ───────────────────────────────────────────
    print("\n[Step 1] Visual Grounding...")
    grounding = qwen_agent.run(user_text)
    print(f"[Step 1] Result:")
    print(f"         visual_description : {grounding.get('visual_description', '')}")
    print(f"         scene              : {grounding.get('scene', '')}")
    print(f"         mood               : {grounding.get('mood', '')}")
    print(f"         style              : {grounding.get('style', '')}")

    # ── Step 2: Content Routing ────────────────────────────────────────────
    print("\n[Step 2] Content Routing...")
    route = router.route(grounding)
    print(f"[Step 2] Route: '{route}'")

    if route == "unsplash":

        # ── Step 3: SigLIP Retrieval ───────────────────────────────────────
        print("\n[Step 3] SigLIP Retrieval...")
        images = siglip_agent.retrieve(grounding)
        top_score = images[0]["score"] if images else 0.0
        print(f"[Step 3] Retrieved {len(images)} image(s), top_score={top_score:.4f}")
        for i, img in enumerate(images):
            print(
                f"         [{i+1}] photo_id={img.get('photo_id')}  score={img.get('score', 0):.4f}"
            )

        # ── Step 3.5: Fallback to Generation if score insufficient ─────────
        # SIMILARITY_THRESHOLD = 0.25  # ADDED
        # if top_score < SIMILARITY_THRESHOLD:  # ADDED
        #     print(
        #         f"\n[Step 3.5] Score {top_score:.4f} < threshold {SIMILARITY_THRESHOLD}"
        #     )
        #     print(f"[Step 3.5] Falling back to GenerationAgent...")
        #     images = generation_agent.run(  # ADDED
        #         grounding_output=grounding,
        #         n=3,
        #         user_text=user_text,
        #         uploaded_image_paths=uploaded_image_paths,
        #         style_ref_path=style_ref_path,
        #         siglip_agent=siglip_agent,
        #     )
        #     print(f"[Step 3.5] Generated {len(images)} image(s):")
        #     for i, img in enumerate(images):
        #         print(
        #             f"           [{i+1}] photo_id={img.get('photo_id')}  source={img.get('source')}  score={img.get('score', 0):.4f}"
        #         )
        # else:
        #     print(
        #         f"\n[Step 3.5] Score {top_score:.4f} >= threshold {SIMILARITY_THRESHOLD} — using retrieval results"
        #     )

        # # ── Step 4: Justification ──────────────────────────────────────────
        # print(f"\n[Step 4] Justification ({len(images)} image(s))...")
        # images = justification_agent.run(user_text, images)
        # print(f"[Step 4] Done. Sample justification:")
        # if images:
        #     print(f"         {images[0].get('justification', '')[:120]}...")

        # print(f"\n[Pipeline] DONE — returning {len(images)} result(s)")
        # print("=" * 60 + "\n")
        return images

    # ── Non-unsplash route: go directly to GenerationAgent ────────────────
    print(
        f"\n[Step 3] Route='{route}' — skipping retrieval, going directly to GenerationAgent..."
    )  # ADDED
    generated = generation_agent.run(  # ADDED
        grounding_output=grounding,
        n=3,
        user_text=user_text,
        uploaded_image_paths=uploaded_image_paths,
        style_ref_path=style_ref_path,
        siglip_agent=siglip_agent,
    )
    print(f"[Step 3] Generated {len(generated)} image(s):")
    for i, img in enumerate(generated):
        print(
            f"         [{i+1}] photo_id={img.get('photo_id')}  source={img.get('source')}  score={img.get('score', 0):.4f}"
        )

    print(f"\n[Step 4] Justification ({len(generated)} image(s))...")
    result = justification_agent.run(user_text, generated)
    print(f"[Step 4] Done.")

    print(f"\n[Pipeline] DONE — returning {len(result)} result(s)")
    print("=" * 60 + "\n")
    return result  # ADDED


if __name__ == "__main__":

    query = "Watching a dramatic sunset"

    results = run_pipeline(query)

    print(results)
