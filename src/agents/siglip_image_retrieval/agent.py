# # import numpy as np
# import pandas as pd
# import torch
# import time
# from transformers import AutoModel, AutoProcessor
# from pathlib import Path


# class SiglipImageRetrievalAgent:

#     def __init__(
#         self,
#         embedding_file="src/data/embeddings/image_embeddings.npy",
#         metadata_file="src/data/processed/dataset_clean.csv",
#         top_k=5
#     ):

#         self.top_k = top_k

#         # Resolve project root dynamically
#         BASE_DIR = Path(__file__).resolve().parents[2]

#         embedding_path = BASE_DIR / "data/embeddings/image_embeddings.npy"
#         metadata_path = BASE_DIR / "data/processed/dataset_clean.csv"

#         # Load embeddings
#         self.image_embeddings = np.load(embedding_path)

#         # Load metadata
#         # self.metadata = pd.read_csv(metadata_path).to_dict("records")
#         self.metadata = pd.read_csv(metadata_path).head(len(self.image_embeddings)).to_dict("records")

#         # Safety check
#         if len(self.image_embeddings) != len(self.metadata):
#             raise ValueError(
#                 f"Embeddings ({len(self.image_embeddings)}) and metadata ({len(self.metadata)}) must match."
#             )

#         model_name = "google/siglip-base-patch16-224"

#         self.model = AutoModel.from_pretrained(model_name)
#         self.processor = AutoProcessor.from_pretrained(model_name)


#     def build_query(self, grounding_output):

#         return " ".join([
#             grounding_output.get("visual_description", ""),
#             grounding_output.get("scene", ""),
#             grounding_output.get("mood", ""),
#             grounding_output.get("style", "")
#         ])


#     def embed_text(self, text):

#         inputs = self.processor(text=[text], return_tensors="pt")

#         with torch.no_grad():
#             outputs = self.model.get_text_features(**inputs)

#         # Extract tensor if needed
#         if hasattr(outputs, "pooler_output"):
#             features = outputs.pooler_output
#         else:
#             features = outputs

#         features = torch.nn.functional.normalize(features, dim=-1)

#         return features.cpu().numpy()[0]


#     # def retrieve(self, grounding_output):

#     #     query = self.build_query(grounding_output)

#     #     text_embedding = self.embed_text(query)

#     #     image_embeddings = self.image_embeddings
#     #     norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)

#     #     # prevent divide-by-zero
#     #     norms[norms == 0] = 1

#     #     image_embeddings = image_embeddings / norms
#     #     similarities = image_embeddings @ text_embedding

#     #     ranked = np.argsort(similarities)[::-1]

#     #     results = []

#     #     for idx in ranked[:self.top_k]:

#     #         item = self.metadata[idx]

#     #         results.append({
#     #             "photo_id": item.get("photo_id"),
#     #             "image_url": item.get("photo_image_url"),
#     #             "caption": item.get("photo_description_clean", ""),
#     #             "score": float(similarities[idx])
#     #         })

#     #     return results
#     def retrieve_all_scores(self, grounding_output: dict) -> np.ndarray:
#         """Return cosine similarity scores for ALL records, shape (N,)."""
#         query = self.build_query(grounding_output)
#         text_embedding = self.embed_text(query)

#         image_embeddings = self.image_embeddings
#         norms = np.linalg.norm(image_embeddings, axis=1, keepdims=True)
#         norms[norms == 0] = 1
#         image_embeddings = image_embeddings / norms

#         return image_embeddings @ text_embedding


# if __name__ == "__main__":

#     import json
#     from pathlib import Path

#     print("\n--- Testing SigLIP Retrieval Agent ---\n")

#     BASE_DIR = Path(__file__).resolve().parents[2]

#     # choose which file to use
#     sample_path = BASE_DIR / "data/processed/sample_grounding_outputs.json"
#     # or:
#     # sample_path = BASE_DIR / "data/processed/grounding_outputs.json"

#     with open(sample_path) as f:
#         data = json.load(f)

#     # handle both formats safely
#     if isinstance(data, list):
#         grounding_output = data[0]["grounding_output"]
#     else:
#         grounding_output = data["grounding_output"]

#     agent = SiglipImageRetrievalAgent()

#     start = time.time()

#     query = agent.build_query(grounding_output)
#     print("Generated Query:\n", query, "\n")

#     results = agent.retrieve_all_scores(grounding_output)

#     end = time.time()

#     print("Top Results:\n")

#     for i, r in enumerate(results):

#         print(f"{i+1}. Photo ID: {r['photo_id']}")
#         print(f"   Score: {r['score']:.4f}")
#         print(f"   Caption: {r['caption']}")
#         print()

#     print("Retrieval time:", round(end - start, 3), "seconds")

import numpy as np
import pandas as pd
import torch
import time
from transformers import AutoModel, AutoProcessor
from pathlib import Path


class SiglipImageRetrievalAgent:

    def __init__(
        self,
        embedding_file="src/data/embeddings/image_embeddings.npy",
        metadata_file="src/data/processed/dataset_clean.csv",
        top_k=5,
    ):

        self.top_k = top_k

        # Resolve project root dynamically
        BASE_DIR = Path(__file__).resolve().parents[2]

        embedding_path = BASE_DIR / "data/embeddings/image_embeddings.npy"
        metadata_path = BASE_DIR / "data/processed/dataset_clean.csv"

        # Load embeddings
        self.image_embeddings = np.load(embedding_path)

        # Pre-normalize image embeddings once at load time (saves work on every query)
        norms = np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.image_embeddings_normalized = self.image_embeddings / norms

        # Load metadata
        self.metadata = (
            pd.read_csv(metadata_path, on_bad_lines="skip", engine="python")
            .head(len(self.image_embeddings))
            .to_dict("records")
        )

        # Safety check
        if len(self.image_embeddings) != len(self.metadata):
            raise ValueError(
                f"Embeddings ({len(self.image_embeddings)}) and metadata ({len(self.metadata)}) must match."
            )

        model_name = "google/siglip-base-patch16-224"
        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)

    def build_query(self, grounding_output):
        return " ".join(
            [
                grounding_output.get("visual_description", ""),
                grounding_output.get("scene", ""),
                grounding_output.get("mood", ""),
                grounding_output.get("style", ""),
            ]
        )

    # def embed_text(self, text):
    #     inputs = self.processor(text=[text], return_tensors="pt")
    def embed_text(self, text):
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            truncation=True,
            max_length=64,
        )

        with torch.no_grad():
            outputs = self.model.get_text_features(**inputs)

        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        features = torch.nn.functional.normalize(features, dim=-1)
        return features.cpu().numpy()[0]

    def retrieve_all_scores(self, grounding_output) -> np.ndarray:
        """
        Return cosine similarity scores for ALL records in the image library.
        Shape: (N,) — one score per record, in the same order as self.metadata.
        Used by hybrid retrieval to merge with text scores over the full dataset.
        """
        query = self.build_query(grounding_output)
        text_embedding = self.embed_text(query)
        return self.image_embeddings_normalized @ text_embedding  # (N,)

    def retrieve(self, grounding_output):
        """Return top-k results as a list of dicts."""
        similarities = self.retrieve_all_scores(grounding_output)
        ranked = np.argsort(similarities)[::-1]

        results = []
        for idx in ranked[: self.top_k]:
            item = self.metadata[idx]
            results.append(
                {
                    "photo_id": item.get("photo_id"),
                    "image_url": item.get("photo_image_url"),
                    "caption": item.get("photo_description_clean", ""),
                    "score": float(similarities[idx]),
                }
            )

        return results


if __name__ == "__main__":

    import json
    from pathlib import Path
    import time

    print("\n--- Testing SigLIP Retrieval Agent (Saving Output) ---\n")

    BASE_DIR = Path(__file__).resolve().parents[2]

    grounding_path = BASE_DIR / "data/processed/description_grounding_outputs.json"
    output_path = BASE_DIR / "outputs/siglip_results.json"

    # make sure outputs folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(grounding_path) as f:
        data = json.load(f)

    agent = SiglipImageRetrievalAgent()

    all_results = []

    start = time.time()

    for i, item in enumerate(data[:5]):

        print(f"\n--- Example {i+1} ---\n")

        # handle both formats
        if isinstance(item, dict) and "grounding_output" in item:
            grounding_output = item["grounding_output"]
        else:
            grounding_output = item

        query = agent.build_query(grounding_output)

        results = agent.retrieve(grounding_output)

        # store everything
        all_results.append({
            "query": query,
            "grounding_output": grounding_output,
            "results": results
        })

        # still print for debugging
        print("Query:", query)
        for r in results:
            print(r)

    end = time.time()

    # save to file
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved results to: {output_path}")
    print("Total runtime:", round(end - start, 3), "seconds")