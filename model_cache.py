"""
model_cache.py  —  place this in your project root.

Run once to download all models:
    python model_cache.py

After that, all NLP modules will load from local disk — zero internet needed.
"""

import os
from pathlib import Path
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# ── Where to store models ─────────────────────────────────────────────────────
# Change this path if you want models stored elsewhere
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HOME"]            = CACHE_DIR

MODELS = {
    "sentiment":    "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "emotion":      "SamLowe/roberta-base-go_emotions",          # ← upgraded from j-hartmann
    "toxicity":     "unitary/toxic-bert",
    "sarcasm":      "cardiffnlp/twitter-roberta-base-irony",
    "multilingual": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
}


def download_all_models():
    """Download and cache all models to ./models/ folder."""
    Path(CACHE_DIR).mkdir(exist_ok=True)
    print(f"\n📦 Downloading models to: {CACHE_DIR}\n")

    for name, model_id in MODELS.items():
        print(f"⬇️  Downloading {name}: {model_id} ...")
        try:
            AutoTokenizer.from_pretrained(model_id, cache_dir=CACHE_DIR)
            AutoModelForSequenceClassification.from_pretrained(model_id, cache_dir=CACHE_DIR)
            print(f"   ✅ {name} saved to {CACHE_DIR}")
        except Exception as e:
            print(f"   ❌ Failed to download {name}: {e}")

    print("\n✅ All models downloaded! They will now load from disk.")
    print(f"   Location: {CACHE_DIR}")


def get_cache_dir() -> str:
    """Return the model cache directory path."""
    return CACHE_DIR


def is_downloaded(model_key: str) -> bool:
    """Check if a model has been downloaded."""
    model_id  = MODELS.get(model_key, "")
    model_dir = Path(CACHE_DIR)
    if not model_dir.exists():
        return False
    safe_name = model_id.replace("/", "--")
    return any(safe_name in str(p) for p in model_dir.rglob("*") if p.is_dir())


def model_status() -> dict:
    """Return download status for all models."""
    return {name: is_downloaded(name) for name in MODELS}


if __name__ == "__main__":
    print("=" * 55)
    print("  YT Comment Intelligence — Model Downloader")
    print("=" * 55)

    status = model_status()
    already = [k for k, v in status.items() if v]
    missing = [k for k, v in status.items() if not v]

    if already:
        print(f"\n✅ Already downloaded: {', '.join(already)}")
    if missing:
        print(f"⬇️  Will download: {', '.join(missing)}\n")
        download_all_models()
    else:
        print("\n🎉 All models already cached! Nothing to download.")