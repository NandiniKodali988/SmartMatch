# Project Paper

## System Architecture 

- built SmartMatch as a multi-agent pipeline in which each agent handles a distinct stage of image recommendation process
- the system takes a piece of user-written text as input and returns a ranked set of images, accompanied by an explanation 
- pipeline overview: add a flow chart here
- all the agents are orchestrated through a single entry point which initializes each agent, passes outputs between stages, and handles the retrieval-vs-gen. decision

### agent descriptions:

- visual concept grounding agent (GA)
  - the first stage of the pipeline addresses a core limitation of embedding-based retrieval
  - abstract of emotionally rich text does not map well to visual feature spaces
  - GA takes raw user text and transforms into a structural description using the claude model
  - list of the json fields
  - the agent the agent includes retry logic and safe fallback response if the model fails
  - prompts are isolated in a separate file to allow iterative improvement without having to modify the agent logic (same discussed in the class)

content router:
- examines the GA output and routes the request to the appropriate retrieval or generation path 
- uses keyword matching to identify special cases and routes those to specialized handlers
- the default route sends requests to the siglip2 semantic retrieval path