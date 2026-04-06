import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__))
MODEL_PATH  = os.path.join(MODEL_DIR, "viral_model.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "viral_scaler.joblib")

_model  = None
_scaler = None


def build_features(comment: dict) -> list:
    return [
        min(len(comment.get("text", "")), 500),
        comment.get("positive_score", 0),
        comment.get("negative_score", 0),
        comment.get("neutral_score",  0),
        1 if comment.get("intent") == "question"  else 0,
        1 if comment.get("intent") == "praise"    else 0,
        1 if comment.get("intent") == "complaint" else 0,
        1 if comment.get("is_sarcastic", False)   else 0,
        1 if comment.get("is_toxic", False)        else 0,
        1 if comment.get("has_emoji", False)       else 0,
        min(comment.get("reply_count", 0), 50),
    ]


def _train(enriched_comments: list[dict]) -> bool:
    """Train on current comments and save model to disk. Returns True if successful."""
    global _model, _scaler

    labeled = [c for c in enriched_comments if c.get("like_count", 0) > 0]
    if len(labeled) < 15:
        return False  # not enough data to train

    X = [build_features(c) for c in labeled]
    y = [np.log1p(c["like_count"]) for c in labeled]

    _scaler  = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _model   = GradientBoostingRegressor(n_estimators=100, random_state=42)
    _model.fit(X_scaled, y)

    # Save so next run loads from disk
    joblib.dump(_model,  MODEL_PATH)
    joblib.dump(_scaler, SCALER_PATH)
    return True


def _load() -> bool:
    """Load model from disk if it exists. Returns True if loaded."""
    global _model, _scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        _model  = joblib.load(MODEL_PATH)
        _scaler = joblib.load(SCALER_PATH)
        return True
    return False


def predict_virality(enriched_comments: list[dict]) -> list[dict]:
    global _model, _scaler

    # Step 1 — try loading from disk
    if _model is None:
        _load()

    # Step 2 — if still None, train from current comments
    if _model is None:
        trained = _train(enriched_comments)
        if not trained:
            # Not enough data — just set defaults and return
            for c in enriched_comments:
                c["predicted_likes"] = 0
                c["virality_score"]  = 0.0
            return enriched_comments

    # Step 3 — predict
    try:
        X        = [build_features(c) for c in enriched_comments]
        X_scaled = _scaler.transform(X)
        preds    = _model.predict(X_scaled)
        max_pred = max(preds) if max(preds) > 0 else 1

        for i, c in enumerate(enriched_comments):
            c["predicted_likes"] = int(np.expm1(preds[i]))
            c["virality_score"]  = round(float(preds[i]) / max_pred, 3)

    except Exception:
        # Fallback — don't crash the whole pipeline
        for c in enriched_comments:
            c["predicted_likes"] = 0
            c["virality_score"]  = 0.0

    return enriched_comments