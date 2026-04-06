from transformers import pipeline

_toxicity_model = None

# Raise threshold to 0.85 to avoid false positives on normal comments
TOXIC_THRESHOLD = 0.85


def get_toxicity_model():
    global _toxicity_model
    if _toxicity_model is None:
        _toxicity_model = pipeline(
            "text-classification",
            model="unitary/toxic-bert",
        )
    return _toxicity_model


def analyze_toxicity(texts: list[str]) -> list[dict]:
    model   = get_toxicity_model()
    batch   = [t[:512] for t in texts]
    outputs = model(batch, batch_size=32, truncation=True)
    results = []
    for out in outputs:
        if out["label"] == "toxic":
            toxic_score = round(out["score"], 4)
            # Only mark as toxic if confidence is above threshold
            is_toxic = toxic_score >= TOXIC_THRESHOLD
        else:
            toxic_score = round(1 - out["score"], 4)
            is_toxic = False
        results.append({
            "is_toxic":    is_toxic,
            "toxic_score": toxic_score,
        })
    return results