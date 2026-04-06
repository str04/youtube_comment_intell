from transformers import pipeline

_sarcasm_model = None

# Raised from 0.80 → 0.92 to eliminate false positives on sincere praise.
# The twitter-roberta-irony model was trained on tweets and confuses
# emphatic/short positive statements ("The online teacher I appreciate the most",
# "A teacher who truly lives his teachings 🎉") with irony.
# At 0.92 only genuinely sarcastic comments get flagged.
SARCASM_THRESHOLD = 0.92

# If a comment contains ANY of these markers it is almost certainly sincere —
# override the model even if it fires above threshold.
SINCERE_MARKERS = [
    # Gratitude / appreciation
    "thank", "thanks", "grateful", "appreciate", "appreciated",
    "bless", "god bless", "hats off", "hat's off", "salute",
    "well done", "keep it up", "keep going", "great work", "good work",
    "nice work", "nice explanation", "great explanation",
    # Praise phrases that the irony model often misreads
    "i appreciate", "i love", "i learned", "i have learned",
    "best teacher", "best explanation", "best video", "best channel",
    "no one explains", "no one has explained", "better than",
    "lives his teachings", "truly lives",
    # Hinglish appreciation
    "bhaiya", "sir aap", "nitish sir", "iss se best",
    "sabse best", "bohot acha", "bahut acha", "bhot acha",
    "maza aa gaya", "maza aaya", "maja aa gaya",
    # Positive emojis (sincere signal)
    "🙏", "❤", "😍", "🥰", "🎉", "🎊", "💯", "👏", "🌟", "⭐",
]

# Hard sarcasm signals — if present, trust the model even below threshold
SARCASM_SIGNALS = [
    "yeah right", "oh sure", "totally not", "clearly the best",
    "wow so helpful", "very helpful indeed", "great job as always",
    "as if", "oh wow amazing", "what a surprise", "shocking",
    "/s",   # Reddit-style sarcasm tag
]


def get_sarcasm_model():
    global _sarcasm_model
    if _sarcasm_model is None:
        _sarcasm_model = pipeline(
            "text-classification",
            model="cardiffnlp/twitter-roberta-base-irony",
        )
    return _sarcasm_model


def _override_check(text: str, model_says_sarcastic: bool, score: float) -> bool:
    """
    Apply rule-based overrides on top of the model result.

    Returns the final is_sarcastic boolean.
    """
    t = text.lower()

    # 1. Hard sarcasm signals → trust model regardless
    if any(sig in t for sig in SARCASM_SIGNALS):
        return model_says_sarcastic

    # 2. Sincere markers → override to NOT sarcastic, no matter what model says
    if any(marker in t for marker in SINCERE_MARKERS):
        return False

    # 3. Very short positive comments (≤ 8 words) — the irony model is
    #    notoriously unreliable on these; require a higher bar (0.95)
    word_count = len(text.split())
    if word_count <= 8 and model_says_sarcastic and score < 0.95:
        return False

    return model_says_sarcastic


def detect_sarcasm(texts: list[str]) -> list[dict]:
    model   = get_sarcasm_model()
    batch   = [t[:512] for t in texts]
    outputs = model(batch, batch_size=32, truncation=True)
    results = []

    for i, out in enumerate(outputs):
        raw_score     = round(out["score"], 4)
        model_irony   = out["label"] == "irony"
        sarcasm_score = raw_score if model_irony else round(1 - raw_score, 4)

        # Step 1: threshold gate
        model_says_sarcastic = model_irony and raw_score >= SARCASM_THRESHOLD

        # Step 2: rule-based override
        is_sarcastic = _override_check(texts[i], model_says_sarcastic, raw_score)

        results.append({
            "is_sarcastic":  is_sarcastic,
            "sarcasm_score": sarcasm_score,
        })

    return results