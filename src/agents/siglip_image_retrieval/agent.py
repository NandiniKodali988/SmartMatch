import numpy as np
import json
import torch
from transformers import AutoModel, AutoProcessor


class SiglipImageRetrievalAgent:

    def __init__(
        self,
        embedding_file="data/image_embeddings.npy",
        metadata_file="data/image_metadata.json",
        top_k=5
    ):

        self.top_k = top_k

        # Load embeddings
        self.image_embeddings = np.load(embedding_file)

        with open(metadata_file, "r") as f:
            self.metadata = json.load(f)

        model_name = "google/siglip-base-patch16-224"

        self.model = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)


    def build_query(self, grounding_output):

        return " ".join([
            grounding_output.get("visual_description", ""),
            grounding_output.get("scene", ""),
            grounding_output.get("mood", ""),
            grounding_output.get("style", "")
        ])


    def embed_text(self, text):

        inputs = self.processor(text=[text], return_tensors="pt")

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)

        features = features / features.norm(dim=-1, keepdim=True)

        return features.numpy()[0]


    def retrieve(self, grounding_output):

        query = self.build_query(grounding_output)

        text_embedding = self.embed_text(query)

        image_embeddings = self.image_embeddings
        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings,
            axis=1,
            keepdims=True
        )

        similarities = image_embeddings @ text_embedding

        ranked = np.argsort(similarities)[::-1]

        results = []

        for idx in ranked[:self.top_k]:

            item = self.metadata[idx]

            results.append({
                "photo_id": item["photo_id"],
                "image_url": item["photo_image_url"],
                "caption": item.get("photo_description_clean", ""),
                "score": float(similarities[idx])
            })

        return results