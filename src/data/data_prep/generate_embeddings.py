import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm
from pathlib import Path


# ===== CONFIG =====
BATCH_SIZE = 500
EMBED_DIM = 768

# ===== PATHS =====
BASE_DIR = Path(__file__).resolve().parents[2]

dataset_path = BASE_DIR / "data/processed/dataset_clean.csv"
embedding_path = BASE_DIR / "data/embeddings/image_embeddings.npy"


# ===== LOAD DATA =====
df = pd.read_csv(dataset_path)

total_rows = len(df)


# ===== LOAD MODEL =====
model_name = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_name)
processor = AutoProcessor.from_pretrained(model_name)

model.eval()


# ===== RESUME LOGIC =====
if embedding_path.exists():
    print("Found existing embeddings — resuming...")

    existing_embeddings = np.load(embedding_path)
    start_idx = len(existing_embeddings)

    embeddings = existing_embeddings.tolist()

else:
    print("Starting fresh embeddings...")
    embeddings = []
    start_idx = 0


print(f"Starting from row: {start_idx}")


# ===== MAIN LOOP =====
for i in tqdm(range(start_idx, total_rows)):

    row = df.iloc[i]
    url = row["photo_image_url"]

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        image = Image.open(BytesIO(response.content)).convert("RGB")

        inputs = processor(images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = model.get_image_features(**inputs)

        # Handle SigLIP output
        if hasattr(outputs, "pooler_output"):
            features = outputs.pooler_output
        else:
            features = outputs

        features = torch.nn.functional.normalize(features, dim=-1)

        embeddings.append(features.cpu().numpy()[0])

    except Exception as e:
        print(f"\nFAILED at index {i}: {url}")
        print(e)

        embeddings.append(np.zeros(EMBED_DIM))


    # ===== SAVE EVERY BATCH =====
    if (i + 1) % BATCH_SIZE == 0:

        print(f"\nSaving checkpoint at {i+1} rows...")

        np.save(embedding_path, np.array(embeddings))

        print("Checkpoint saved.")


# ===== FINAL SAVE =====
embeddings = np.array(embeddings)

np.save(embedding_path, embeddings)

print("\nFinal embeddings saved.")
print("Shape:", embeddings.shape)