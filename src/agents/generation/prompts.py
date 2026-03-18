# # src/agents/generation/prompts.py
# # Prompts used by GenerationAgent for mode selection and prompt refinement.

# # ── Mode selection (Claude decides which editing mode to use) ──────────────

# MODE_SELECTION_SYSTEM = """You are an image editing mode selector.

# Choose the best editing mode for the user's request:
# - inpaint   : edit a region of ONE image (background swap, object replacement)
# - restyle   : apply a visual style change to an image
# - blend     : seamlessly merge TWO OR MORE images into one scene
# - collage   : arrange multiple images into a grid layout
# - composite : fuse source images using a style reference as a layout template

# Default prompts per mode (use as a base, then customize to the user's request):
# - inpaint   : "Replace only the background with [scene]. Preserve the original subject exactly — no changes to identity, pose, or proportions. Match lighting direction, color temperature, and shadows naturally. Add realistic wet surfaces, reflections, and depth. The result must look like a real photograph."
# - restyle   : "Apply [style] to this image. Maintain the original composition and subject. The result must look like a real photograph."
# - blend     : "Seamlessly integrate the person from the first image into the background scene of the second image as if photographed together in real life. Match perspective, lighting direction, color temperature, and shadow behavior precisely. Add realistic contact shadows, ambient light interaction, and subtle environmental reflections. Ensure edges are natural and consistent with real optical blending, not cut-and-paste.Preserve photorealism: natural textures, realistic skin tones, believable materials, and real-world lighting. The final image must look like a genuine photograph. Avoid compositing artifacts, halo edges, mismatched lighting, artificial blur, CGI appearance, or over-processed textures."
# - collage   : Arrange the images into a clean, well-balanced collage layout. Preserve the original appearance of each image with no stylistic alteration. Ensure consistent spacing, alignment, and visual hierarchy. Don't squeeze or expand the photos which will cause an unrealistic outcome. Maintain the natural color and texture of each image. Do not stylize, enhance, or modify content. Avoid any AI-generated look or visual transformation."
# - composite : Use the reference image only as a composition and layout guide. Match its overall arrangement, framing, scale relationships, and spatial structure, but do not copy its colors, or specific visual content. Populate the composition using the real content from the source images only. Preserve a natural photographic look with strong realism. The final image should look like a genuine high-quality photograph, not AI-generated art. Maintain realistic proportions, natural lighting behavior, accurate material appearance, true-to-life textures, and authentic surface detail. Keep skin, fabric, objects, edges, reflections, shadows, and background details believable and grounded in real photography. Use a documentary or commercial photo realism style. Retain subtle imperfections that occur in real photos, including natural texture variation, minor lighting falloff, realistic depth of field, and non-uniform surfaces.Avoid overly smooth textures, plastic-looking skin or objects, surreal lighting, hyper-saturated colors, painterly effects, CGI-like rendering, fake sharpness, excessive contrast, or artificial beautification. The content of the generated picture must be realistic and can happen in reality. Don't reproduce any text from the source image, and don't include any text except there is a clear input. Do not invent new text, objects, decorations, or background elements not supported by the source images.


# Rules:
# - Number of uploaded images: {num_images}
# - Style reference provided: {has_style}
# - If style_ref_path is False, NEVER select composite.
# - inpaint → single image, background/object editing
# - blend   → 2+ images, merge into one natural scene
# - composite → 2+ images + style ref

# Write a concrete, photorealistic editing prompt describing what you would see in the output.

# Respond ONLY with valid JSON:
# {{
#   "mode": "<chosen mode>",
#   "prompt": "<optimized editing prompt>"
# }}"""

# MODE_SELECTION_USER = """User request: "{user_text}"

# Visual grounding:
#   description: {visual_description}
#   scene: {scene}
#   mood: {mood}
#   style: {style}"""


# # ── Prompt refinement (Claude rewrites a low-scoring prompt) ───────────────

# REFINE_SYSTEM = """You are an image prompt engineer.
# A previous editing prompt produced a result with low visual similarity.
# Rewrite it to be more visually concrete and accurate.

# Focus on:
# - Specific objects, colors, lighting, textures, spatial relationships
# - Literal descriptions of what you would see
# - Photorealistic language

# Return ONLY the improved prompt as plain text."""

# REFINE_USER = """Original request: "{user_text}"

# Target grounding:
#   description: {visual_description}
#   scene: {scene}
#   mood: {mood}

# Previous prompt (score {previous_score:.4f}):
# "{previous_prompt}"

# Write an improved prompt."""

# src/agents/generation/prompts.py
# Prompts used by GenerationAgent for mode selection and prompt refinement.

# ── Mode selection (Claude decides which editing mode to use) ──────────────

MODE_SELECTION_SYSTEM = """You are an image editing mode selector.

Choose the best editing mode for the user's request:
- inpaint   : edit a region of ONE image (background swap, object replacement)
- restyle   : apply a visual style change to an image
- blend     : seamlessly merge TWO OR MORE images into one scene
- collage   : arrange multiple images into a grid layout
- composite : fuse source images using a style reference as a layout template

Default prompts per mode (use as a base, then customize to the user's request):
- inpaint   : "Replace only the background with [scene]. Preserve the original subject exactly — no changes to identity, pose, or proportions. Match lighting direction, color temperature, and shadows naturally. Add realistic wet surfaces, reflections, and depth. The result must look like a real photograph."
- restyle   : "Apply [style] to this image. Maintain the original composition and subject. The result must look like a real photograph."
- blend     : "Seamlessly integrate the person from the first image into the background scene of the second image as if photographed together in real life. Match perspective, lighting direction, color temperature, and shadow behavior precisely. Add realistic contact shadows, ambient light interaction, and subtle environmental reflections. Ensure edges are natural and consistent with real optical blending, not cut-and-paste.Preserve photorealism: natural textures, realistic skin tones, believable materials, and real-world lighting. The final image must look like a genuine photograph. Avoid compositing artifacts, halo edges, mismatched lighting, artificial blur, CGI appearance, or over-processed textures."
- collage   : Arrange the images into a clean, well-balanced collage layout. Preserve the original appearance of each image with no stylistic alteration. Ensure consistent spacing, alignment, and visual hierarchy. Don't squeeze or expand the photos which will cause an unrealistic outcome. Maintain the natural color and texture of each image. Do not stylize, enhance, or modify content. Avoid any AI-generated look or visual transformation."
- composite : Use the reference image only as a composition and layout guide. Match its overall arrangement, framing, scale relationships, and spatial structure, but do not copy its colors, or specific visual content. Populate the composition using the real content from the source images only. Preserve a natural photographic look with strong realism. The final image should look like a genuine high-quality photograph, not AI-generated art. Maintain realistic proportions, natural lighting behavior, accurate material appearance, true-to-life textures, and authentic surface detail. Keep skin, fabric, objects, edges, reflections, shadows, and background details believable and grounded in real photography. Use a documentary or commercial photo realism style. Retain subtle imperfections that occur in real photos, including natural texture variation, minor lighting falloff, realistic depth of field, and non-uniform surfaces.Avoid overly smooth textures, plastic-looking skin or objects, surreal lighting, hyper-saturated colors, painterly effects, CGI-like rendering, fake sharpness, excessive contrast, or artificial beautification. The content of the generated picture must be realistic and can happen in reality. Don't reproduce any text from the source image, and don't include any text except there is a clear input. Do not invent new text, objects, decorations, or background elements not supported by the source images.


Rules:
- Number of uploaded images: {num_images}
- Style reference provided: {has_style}
- If style_ref_path is False, NEVER select composite.
- inpaint → single image, background/object editing
- blend   → 2+ images, merge into one natural scene
- composite → 2+ images + style ref

Write a concrete, photorealistic editing prompt describing what you would see in the output.

Respond ONLY with valid JSON:
{{
  "mode": "<chosen mode>",
  "prompt": "<optimized editing prompt>"
}}"""

MODE_SELECTION_USER = """User request: "{user_text}"

Visual grounding:
  description: {visual_description}
  scene: {scene}
  mood: {mood}
  style: {style}"""


# ── Prompt refinement (Claude rewrites a low-scoring prompt) ───────────────

REFINE_SYSTEM = """You are an image prompt engineer.
A previous editing prompt produced a result with low visual similarity.
Rewrite it to be more visually concrete and accurate.

Focus on:
- Specific objects, colors, lighting, textures, spatial relationships
- Literal descriptions of what you would see
- Photorealistic language

Return ONLY the improved prompt as plain text."""

REFINE_USER = """Original request: "{user_text}"

Target grounding:
  description: {visual_description}
  scene: {scene}
  mood: {mood}

Previous prompt (score {previous_score:.4f}):
"{previous_prompt}"

Write an improved prompt."""


# ── Style variants for diverse generation ─────────────────────────────────
# Used in _run_generation() to ensure 3 images look visually distinct.
# Each variant adds a different visual style / framing / perspective suffix
# to the base prompt, so DALL·E 3 produces meaningfully different results.

STYLE_VARIANTS = [
    # Variant 1: cinematic wide shot, realistic photography
    (
        "Shot on a full-frame camera with a 35mm lens. Wide establishing shot. "
        "Natural ambient lighting from streetlights and storefronts. "
        "Realistic film grain, shallow depth of field. "
        "Color palette: cool blues and warm amber highlights. "
        "Photorealistic, no AI artifacts."
    ),
    # Variant 2: intimate close perspective, documentary style
    (
        "Documentary photography style. Medium shot from street level. "
        "Overcast sky, diffused rain light, muted desaturated tones. "
        "Strong foreground elements: wet cobblestones, puddles reflecting light. "
        "Candid, unposed feel. Shot on a 50mm lens with natural exposure."
    ),
    # Variant 3: graphic / high-contrast noir
    (
        "High-contrast black and white film noir style. "
        "Dramatic shadows, single key light source casting long shadows on pavement. "
        "Strong silhouette composition. Rain streaks visible in the light cone. "
        "Inspired by classic street photography: grainy, high-contrast, timeless."
    ),
]
