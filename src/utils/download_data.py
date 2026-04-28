"""
src/utils/download_data.py

Downloads large data files from HuggingFace Hub into their expected local paths
if they are not already present. Called once at Streamlit app startup.
"""

import os
from pathlib import Path

HF_DATASET_REPO = "NandiniKodali/smartmatch_data"

# HF filename → path relative to src/
_FILES = {
    "field_embeddings.npz":               "data/embeddings/field_embeddings.npz",
    "image_embeddings.npy":               "data/embeddings/image_embeddings.npy",
    "image_graph.pkl":                    "data/graph/image_graph.pkl",
    "dataset_clean.csv":                  "data/processed/dataset_clean.csv",
    "description_grounding_outputs.json": "data/processed/description_grounding_outputs.json",
}

SRC_DIR = Path(__file__).parent.parent  # src/


def ensure_data(verbose: bool = True) -> None:
    """Download any missing data files from HF Hub. Already-present files are skipped."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError("huggingface_hub is required: pip install huggingface_hub")

    token = os.getenv("HF_TOKEN")

    for hf_filename, rel_path in _FILES.items():
        local_path = SRC_DIR / rel_path
        if local_path.exists():
            if verbose:
                print(f"[data] already present: {rel_path}")
            continue

        local_path.parent.mkdir(parents=True, exist_ok=True)

        if verbose:
            print(f"[data] downloading {hf_filename} from {HF_DATASET_REPO} ...")

        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=hf_filename,
            repo_type="dataset",
            local_dir=str(local_path.parent),
            token=token,
        )

        if verbose:
            size_mb = os.path.getsize(downloaded) / 1e6
            print(f"[data] saved {rel_path} ({size_mb:.1f} MB)")
