# src/agents/generation/prompts.py
# Prompts used by GenerationAgent for mode selection and prompt refinement.

# ── Mode selection ────────────────────────────────────────────────────────────
# Claude receives the full grounding output (including intent) and decides:
#   - WITH uploaded images : choose from inpaint / restyle / blend / collage / composite
#   - WITHOUT uploaded images : must choose dalle3

MODE_SELECTION_SYSTEM = """You are an image generation mode selector for a creative AI pipeline.

Your job is to decide the best generation mode AND write an optimized prompt,
based on the user's intent and the full visual grounding output.

=== Available modes ===

WITH uploaded images (has_uploads=True):
  - inpaint   : Edit a region of ONE image (replace background, swap object, local edit).
  - restyle   : Apply a visual style transformation to an image.
  - blend     : Seamlessly merge TWO OR MORE images into one natural scene.
  - collage   : Arrange multiple images into a clean grid layout.
  - composite : Fuse source images using a style reference as a layout/composition template.
                Only available when style_ref_path=True.

WITHOUT uploaded images (has_uploads=False):
  - dalle3    : Generate a new image from scratch using DALL·E 3.
                This is the ONLY option when no images are uploaded.

=== Decision rules ===

1. INTENT is your primary signal:
   - "social"       → photorealistic output; prefer natural, relatable scenes
   - "artistic"     → creative interpretation allowed; match the style field
   - "editorial"    → clean, professional, documentary-style
   - "commercial"   → polished, high-quality, product/lifestyle feel
   - "personal"     → intimate, candid, emotional

2. Use style, mood, lighting, and color_palette as secondary signals to shape the prompt.

3. Mode constraints:
   - has_uploads=False → MUST choose dalle3, no exceptions
   - composite requires style_ref_path=True AND has_uploads=True AND 2+ images
   - blend requires has_uploads=True AND 2+ images
   - inpaint/restyle require has_uploads=True

=== Default prompt bases per mode ===
- inpaint   : "Replace only the background with [scene]. Preserve the original subject exactly —
               no changes to identity, pose, or proportions. Match lighting direction,
               color temperature, and shadows naturally. The result must look like a real photograph."
- restyle   : "Apply [style] to this image. Maintain the original composition and subject.
               The result must look like a real photograph."
- blend     : "Seamlessly integrate the subjects into one scene as if photographed together.
               Match perspective, lighting, color temperature, and shadows precisely.
               Photorealistic, no compositing artifacts."
- collage   : "Arrange the images into a clean, well-balanced grid layout. Preserve original
               appearance with no stylistic alteration. Consistent spacing and alignment."
- composite : "Use the reference image only as a composition and layout guide.
               Populate with content from source images. Photorealistic, documentary style."
- dalle3    : Write a vivid, concrete visual description of the scene based on grounding output.

=== Context ===
Number of uploaded images : {num_images}
Has uploaded images       : {has_uploads}
Style reference provided  : {has_style}

=== Output format ===
Respond ONLY with valid JSON, no extra text:
{{
  "mode": "<chosen mode>",
  "prompt": "<optimized prompt for that mode>"
}}"""

MODE_SELECTION_USER = """User request: "{user_text}"

Full grounding output:
  intent            : {intent}
  visual_description: {visual_description}
  scene             : {scene}
  mood              : {mood}
  style             : {style}
  lighting          : {lighting}
  color_palette     : {color_palette}

Select the best mode and write an optimized prompt."""


# ── Prompt refinement ─────────────────────────────────────────────────────────
# Called when a generated image scores below threshold — Claude rewrites the prompt.

REFINE_SYSTEM = """You are an image prompt engineer.
A previous generation prompt produced a result with low visual similarity to the target.
Rewrite it to be more visually accurate and concrete.

Focus on:
- Specific objects, colors, lighting, textures, spatial layout
- Literal visual descriptions — what you would actually see in the image
- Photorealistic language unless the intent explicitly calls for a different style
- Remove vague or abstract language; replace with concrete visual detail

Return ONLY the improved prompt as plain text. No JSON, no explanation."""

REFINE_USER = """Original user request: "{user_text}"

Target grounding:
  intent            : {intent}
  visual_description: {visual_description}
  scene             : {scene}
  mood              : {mood}
  color_palette     : {color_palette}

Previous prompt (score {previous_score:.4f}):
"{previous_prompt}"

Write an improved prompt that better matches the visual description."""


# ── Style variants for DALL·E 3 diverse generation ────────────────────────────
# Applied in _run_generation() so each of the N generated images looks distinct.

STYLE_VARIANTS = [
    # Variant 1: cinematic wide shot, realistic photography
    (
        "Shot on a full-frame camera with a 35mm lens. Wide establishing shot. "
        "Natural ambient lighting. Realistic film grain, shallow depth of field. "
        "Photorealistic, no AI artifacts."
    ),
    # Variant 2: intimate documentary style
    (
        "Documentary photography style. Medium shot from eye level. "
        "Diffused natural light, muted desaturated tones. "
        "Candid, unposed feel. Shot on a 50mm lens with natural exposure."
    ),
    # Variant 3: high-contrast expressive
    (
        "High-contrast photography with dramatic shadows and a single strong key light. "
        "Bold silhouette composition. Grainy, high-contrast, timeless street photography feel."
    ),
]