# Project Paper

## System Architecture 

- built SmartMatch as a multi-agent pipeline in which each agent handles a distinct stage of image recommendation process
- the system takes a piece of user-written text (and optionally uploaded images) as input and returns a ranked set of images, accompanied by an explanation
- if the user uploads images > grounding > image editing via GenerationAgent > justification
- if no uploads > grounding > hybrid retrieval > score check (threshold 0.5) > justification or DALL·E 3 fallback > justification
- all agents are orchestrated through a single entry point which initializes each agent, passes outputs between stages, and handles the retrieval-vs-gen decision
- pipeline overview: add a flow chart here

### agent descriptions:

- visual concept grounding agent (GA)
  - the first stage of the pipeline addresses a core limitation of embedding-based retrieval
  - abstract or emotionally rich text does not map well to visual feature spaces
  - GA takes raw user text and transforms into a structural description using the claude model
  - list of the json fields
  - the agent includes retry logic and safe fallback response if the model fails
  - prompts are isolated in a separate file to allow iterative improvement without having to modify the agent logic (same discussed in the class)

semantic retrieval agent:
- performs cosine similarity search over pre-computed embeddings
- the GA output fields are concatenated into a single query string which is then encoded using `google/siglip-base-patch16-224`
- the text embedding is compared against pre-computed image embeddings using l2-normalized dot product similarity
- the agent returns the top-k images ranked by similarity score along with the images' id, url and the caption

hybrid field text retrieval agent: 
- to improve the retrieval quality beyond a single query string we used a second retrieval agent to score images using per-field semantic embeddings 
- the final score for each image is a weighted combination of SigLIP similarity (70%) and field-level text similarity (30%), with each field contributing equally (visual_description, mood, color_palette)
- it is to capture both visual and semantic alignment 

dall.e 3 fallback generation 
- if the top hybrid retrieval score falls below a similarity threshold of 0.5, the system falls back to generative image synthesis
- the GA output is used to construct a photorealistic prompt with strict realism constraints
- 3 style variants are generated 
- generated images are scored against the original text embedding and the best-scoring result is returned

justification generation agent
- the final stage adds an explanation to each recommended image
- it takes the user's original text and each image's caption and produces 2-3 sentences explanation of why the image is a good match
- helps improve transparency and user trust in the recommendations

### agent coordination 

- agents communicate through structured dictionaries passed directly between pipeline stages
- the grounding output dictionary is the central data structure — consumed by both retrieval agents and the generation agent
- retrieval agents return a list of image dictionaries, each with photo_id, image_url, caption, and score
- the justification agent appends a justification field to each image before results are returned
- no message-passing framework or shared state — coordination is handled through the pipeline function

### external services and tools 

[create a table]

Component ---- Service / Tool
Visual grounding & justification ---- Anthropic API (claude-haiku-4-5-20251001)
Image retrieval embeddings ---- SigLIP-2 (google/siglip-base-patch16-224)
Field-level text embeddings ---- OpenAI (text-embedding-3-large)
Fallback image generation ---- OpenAI DALL·E 3
Similarity search index ---- FAISS (CPU)
Image dataset ---- Unsplash (50K+ images with pre-computed embeddings)
Batch grounding processing ---- Anthropic Batch API (50% cost reduction vs live API)
