import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_cache import CACHE_DIR

from langdetect import detect
from transformers import pipeline


_multilingual_model = None

HINGLISH_MARKERS = [
    "nahi", "hai", "bhai", "yaar", "kya", "aur", "tha", "hoga",
    "bahut", "accha", "theek", "bilkul", "matlab", "lekin", "phir",
    "agar", "toh", "mujhe", "tumhe", "unhe", "apna", "mera", "tera",
    "dekho", "suno", "bata", "kar", "karo", "kiya", "kyun", "kyunki",
]


def get_multilingual_model():
    global _multilingual_model
    if _multilingual_model is None:
        print("[NLP] Loading multilingual model from cache...")
        _multilingual_model = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
            top_k=None,
            model_kwargs={"cache_dir": CACHE_DIR},
        )
    return _multilingual_model


def detect_language(text: str) -> str:
    try:
        if any(m in text.lower() for m in HINGLISH_MARKERS):
            return "hinglish"
        return detect(text)
    except Exception:
        return "unknown"


def detect_languages(texts: list[str]) -> list[str]:
    return [detect_language(t) for t in texts]


def analyze_hinglish_sentiment(texts: list[str]) -> list[dict]:
    model   = get_multilingual_model()
    outputs = model([t[:512] for t in texts], batch_size=16, truncation=True)
    results = []
    for out in outputs:
        scores   = {item["label"].lower(): round(item["score"], 4) for item in out}
        dominant = max(scores, key=scores.get)
        results.append({
            "sentiment":      dominant,
            "positive_score": scores.get("positive", 0),
            "negative_score": scores.get("negative", 0),
            "neutral_score":  scores.get("neutral",  0),
        })
    return results


def route_by_language(comments: list[dict]) -> tuple[list, list, list]:
    english  = []
    hinglish = []
    other    = []
    for i, c in enumerate(comments):
        lang       = detect_language(c["text"])
        c["language"] = lang
        if lang == "hinglish":
            hinglish.append(i)
        elif lang == "en":
            english.append(i)
        else:
            other.append(i)
    return english, hinglish, other