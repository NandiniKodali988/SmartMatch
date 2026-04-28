"""
Run once to download large data files from HuggingFace.
Usage: python download_data.py
Requires HF_TOKEN in .env or environment.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

REPO_ID   = "NandiniKodali/smartmatch_data"
DEST_DIR  = Path(__file__).parent / "src/data/embeddings"
DEST_DIR.mkdir(parents=True, exist_ok=True)

FILES = [
    "image_embeddings.npy",
    "field_embeddings.npz",
]

token = os.environ.get("HF_TOKEN")
if not token:
    raise EnvironmentError("HF_TOKEN not set — add it to your .env file")

for filename in FILES:
    dest = DEST_DIR / filename
    if dest.exists():
        print(f"[skip] {filename} already exists")
        continue
    print(f"[download] {filename} → {dest}")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        repo_type="dataset",
        local_dir=str(DEST_DIR),
        token=token,
    )
    print(f"[done] {filename}")

print("\nAll files ready.")
