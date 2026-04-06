import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_cache import CACHE_DIR

import streamlit as st
from transformers import pipeline

# ─────────────────────────────────────────────────────────────────────────────
# WHY go_emotions?
# j-hartmann was trained on tweets only — fails on thank-you comments,
# study/learning content, Hinglish, and anything not tweet-like.
#
# SamLowe/roberta-base-go_emotions is trained on 58,000 Reddit comments
# across 27 diverse topics — news, sports, gaming, relationships, science etc.
# It has 28 fine-grained emotion labels which we map down to 7 display emotions.
# ─────────────────────────────────────────────────────────────────────────────

GO_EMOTION_MAP = {
    "admiration":    "joy",
    "amusement":     "joy",
    "approval":      "joy",
    "excitement":    "joy",
    "gratitude":     "joy",
    "joy":           "joy",
    "love":          "joy",
    "optimism":      "joy",
    "pride":         "joy",
    "relief":        "joy",
    "surprise":      "surprise",
    "realization":   "surprise",
    "anger":         "anger",
    "annoyance":     "anger",
    "disapproval":   "anger",
    "sadness":       "sadness",
    "disappointment":"sadness",
    "grief":         "sadness",
    "remorse":       "sadness",
    "fear":          "fear",
    "nervousness":   "fear",
    "disgust":       "disgust",
    "embarrassment": "disgust",
    "neutral":       "neutral",
    "curiosity":     "neutral",
    "confusion":     "neutral",
    "caring":        "neutral",
    "desire":        "neutral",
}

CONFIDENCE_THRESHOLD = 0.30

# ─────────────────────────────────────────────────────────────────────────────
# Praise-with-negation patterns — the emotion model reads "cannot", "no one",
# "never" and fires ANGER/SADNESS even though the sentence is clearly praise.
# e.g. "Even PhD holder faculties CANNOT explain with such ease — Hats off"
#      "NO ONE has given this type of explanation"
#      "ye idea toh notebook llm JAISA lag raha hai"  (comparison, not sadness)
# We detect these and force JOY.
# ─────────────────────────────────────────────────────────────────────────────
PRAISE_NEGATION_PATTERNS = [
    # English patterns
    "cannot explain", "can't explain", "could not explain",
    "no one explains", "no one has explained", "nobody explains",
    "no one can", "nobody can", "none of them",
    "hats off", "hat's off", "hats off to",
    "even phd", "even professor", "even teachers",
    "better than any", "better than all",
    "never seen", "never found", "never heard",
    "not found anywhere", "nowhere else",
    # Hinglish comparison / praise patterns mistaken for sadness
    "jaisa lag", "jaisa nahi", "se better", "se acha",
    "jitna acha", "utna nahi", "itna acha",
    # General emphatic praise phrases
    "god level", "god bless", "bless you",
    "truly amazing", "truly great", "truly the best",
    "lives his teachings", "truly lives",
    "completely different", "unique explanation",
    "mind blowing", "mind blown",
]

# Explicit negative-sentiment phrases that should NOT be overridden
GENUINE_ANGER_PATTERNS = [
    "hate this", "this is terrible", "this is worst",
    "waste of time", "so boring", "just for views",
    "clickbait", "watch hours", "not worth",
]


@st.cache_resource(show_spinner="Loading sentiment model...")
def get_sentiment_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        top_k=None,
        model_kwargs={"cache_dir": CACHE_DIR},
    )


@st.cache_resource(show_spinner="Loading emotion model (go_emotions — 28 categories)...")
def get_emotion_model():
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        model_kwargs={"cache_dir": CACHE_DIR},
    )


def _rule_based_emotion(text: str, sentiment: str) -> str:
    """Last-resort fallback using keywords when model is not confident."""
    t = text.lower()
    if any(w in t for w in ["thank", "thanks", "grateful", "appreciate",
                             "helped", "got placed", "got job", "got selected",
                             "got offer", "because of you", "❤", "🙏", "😍"]):
        return "joy"
    if any(w in t for w in ["amazing", "awesome", "incredible", "best ever",
                             "love this", "brilliant", "outstanding"]):
        return "joy"
    if any(w in t for w in ["wow", "didn't know", "never knew", "mind blown",
                             "surprised", "omg", "wait what"]):
        return "surprise"
    if any(w in t for w in ["scared", "worried", "nervous", "afraid", "anxious"]):
        return "fear"
    if any(w in t for w in ["sad", "miss", "lost", "gone", "rip", "crying", "😢", "😭"]):
        return "sadness"
    if any(w in t for w in ["angry", "hate", "worst", "useless", "waste",
                             "pathetic", "disgusting", "terrible"]):
        return "anger"
    if sentiment == "positive":
        return "joy"
    if sentiment == "negative":
        return "anger"
    return "neutral"


def _fix_praise_negation(text: str, emotion: str, sentiment: str) -> str:
    """
    Correct ANGER/SADNESS mis-fires on praise sentences that contain
    negation words like "cannot", "no one", "never", "jaisa", etc.

    Only overrides ANGER or SADNESS → JOY.
    Never touches genuinely negative comments.
    """
    if emotion not in ("anger", "sadness"):
        return emotion

    t = text.lower()

    # If the text has genuine anger phrases, don't override
    if any(p in t for p in GENUINE_ANGER_PATTERNS):
        return emotion

    # If the text matches a praise-with-negation pattern, correct to joy
    if any(p in t for p in PRAISE_NEGATION_PATTERNS):
        return "joy"

    # Additional heuristic: sentiment is positive but model said anger/sadness
    # and the comment has praise keywords → force joy
    PRAISE_KEYWORDS = [
        "hats off", "amazing", "brilliant", "fantastic", "superb",
        "excellent", "well done", "best teacher", "best video",
        "great explanation", "love this", "appreciate", "thank",
        "god bless", "salute", "keep it up", "bless", "🙏", "❤", "👏",
        "bawaal", "badhiya", "zabardast", "kamaal", "shandaar",
    ]
    if sentiment == "positive" and any(kw in t for kw in PRAISE_KEYWORDS):
        return "joy"

    return emotion


def analyze_sentiment(texts: list[str]) -> list[dict]:
    model   = get_sentiment_model()
    outputs = model([t[:512] for t in texts], batch_size=32, truncation=True)
    results = []
    for out in outputs:
        scores   = {item["label"].lower(): round(item["score"], 4) for item in out}
        dominant = max(scores, key=scores.get)
        results.append({
            "sentiment":      dominant,
            "positive_score": scores.get("positive", scores.get("pos", 0)),
            "negative_score": scores.get("negative", scores.get("neg", 0)),
            "neutral_score":  scores.get("neutral",  scores.get("neu", 0)),
        })
    return results


def analyze_emotions(texts: list[str], sentiments: list[dict] = None) -> list[dict]:
    """
    Analyze emotions using go_emotions (28 labels → mapped to 7).
    Falls back to rule-based when confidence is low.
    Applies praise-negation correction to fix ANGER/SADNESS on praise sentences.
    """
    model   = get_emotion_model()
    outputs = model([t[:512] for t in texts], batch_size=32, truncation=True)
    results = []

    for i, out in enumerate(outputs):
        raw_scores = {item["label"]: round(item["score"], 4) for item in out}

        grouped = {}
        for raw_label, score in raw_scores.items():
            mapped          = GO_EMOTION_MAP.get(raw_label, "neutral")
            grouped[mapped] = grouped.get(mapped, 0) + round(score, 4)

        top_emotion = max(grouped, key=grouped.get)
        top_score   = grouped[top_emotion]

        # Confidence gate — use rule-based fallback if not confident
        sentiment_str = "neutral"
        if sentiments and i < len(sentiments) and sentiments[i]:
            sentiment_str = sentiments[i].get("sentiment", "neutral")

        if top_score < CONFIDENCE_THRESHOLD:
            top_emotion = _rule_based_emotion(texts[i], sentiment_str)

        # Fix praise-with-negation misclassifications (ANGER/SADNESS → JOY)
        top_emotion = _fix_praise_negation(texts[i], top_emotion, sentiment_str)

        results.append({
            "dominant_emotion": top_emotion,
            **grouped,
            "_raw_top": max(raw_scores, key=raw_scores.get),
        })

    return results