from agents.qwen_visual_grounding.agent import QwenVisualGroundingAgent
from agents.qwen_visual_grounding.justification_agent import QwenJustificationAgent
from agents.siglip_image_retrieval.agent import SiglipImageRetrievalAgent
from agents.content_router.agent import ContentRouterAgent
from agents.generation.agent import GenerationAgent  


def run_pipeline(
    user_text,
    uploaded_image_paths=None, 
    style_ref_path=None,
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
    generation_agent = GenerationAgent()  

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
        SIMILARITY_THRESHOLD = 0.25  
        if top_score < SIMILARITY_THRESHOLD:  
            print(
                f"\n[Step 3.5] Score {top_score:.4f} < threshold {SIMILARITY_THRESHOLD}"
            )
            print(f"[Step 3.5] Falling back to GenerationAgent...")
            images = generation_agent.run(  
                grounding_output=grounding,
                n=3,
                user_text=user_text,
                uploaded_image_paths=uploaded_image_paths,
                style_ref_path=style_ref_path,
                siglip_agent=siglip_agent,
            )
            print(f"[Step 3.5] Generated {len(images)} image(s):")
            for i, img in enumerate(images):
                print(
                    f"           [{i+1}] photo_id={img.get('photo_id')}  source={img.get('source')}  score={img.get('score', 0):.4f}"
                )
        else:
            print(
                f"\n[Step 3.5] Score {top_score:.4f} >= threshold {SIMILARITY_THRESHOLD} — using retrieval results"
            )

        # ── Step 4: Justification ──────────────────────────────────────────
        print(f"\n[Step 4] Justification ({len(images)} image(s))...")
        images = justification_agent.run(user_text, images)
        print(f"[Step 4] Done. Sample justification:")
        if images:
            print(f"         {images[0].get('justification', '')[:120]}...")

        print(f"\n[Pipeline] DONE — returning {len(images)} result(s)")
        print("=" * 60 + "\n")
        return images

    # ── Non-unsplash route: go directly to GenerationAgent ────────────────
    print(
        f"\n[Step 3] Route='{route}' — skipping retrieval, going directly to GenerationAgent..."
    )  
    generated = generation_agent.run(  
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
    return result  


if __name__ == "__main__":

    query = "Watching a dramatic sunset"

    results = run_pipeline(query)

    print(results)
