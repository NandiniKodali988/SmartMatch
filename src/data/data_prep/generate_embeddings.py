import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from pathlib import Path


# Get project root
BASE_DIR = Path(__file__).resolve().parents[2]

dataset_path = BASE_DIR / "data/processed/dataset_clean.csv"
embedding_path = BASE_DIR / "data/embeddings/image_embeddings.npy"


# Load dataset (limit to 200 for testing)
df = pd.read_csv(dataset_path).sample(200, random_state=42)


# Load SigLIP model
model_name = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

model.eval()


embeddings = []

print("\nGenerating image embeddings...\n")


for _, row in tqdm(df.iterrows(), total=len(df)):

    url = row["photo_image_url"]

    try:

        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # Extract tensor if wrapped in output object
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        # Normalize embedding
        features = torch.nn.functional.normalize(features, dim=-1)

        embeddings.append(features.cpu().numpy()[0])

    except Exception as e:

        print("\nFAILED:", url)
        print(e)

        embeddings.append(np.zeros(768))


embeddings = np.array(embeddings)

print("\nEmbedding matrix shape:", embeddings.shape)


# Save embeddings
np.save(embedding_path, embeddings)

print("\nEmbeddings saved to:", embedding_path)