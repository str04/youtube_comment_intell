"""
audiance_segmentation.py

Score-based audience segmentation on 3 axes: positivity, criticism, curiosity.

--- Fixes in this version ---

FIX 1: Curious Mind false positives
  "I appreciate HOW you broke down..." → was Curious Mind because "how" fired.
  Root cause: CURIOUS_MARKERS matched any word containing "how", "what", etc.
  even inside praise sentences.
  Fix: require that a "how/what/why/..." word is genuinely interrogative —
  i.e. followed by a verb phrase, OR that there is actually a "?" in the text,
  OR the sentence starts with the question word.
  Also: strong positivity now beats curiosity in _assign_segment.

FIX 2: Critic false positives
  "my langchain is not working giving me error" → was Critic.
  Root cause: neg score from sentiment model was high enough to pass threshold.
  Fix: added HELP_REQUEST_MARKERS — if a comment is asking for help/support
  it's NOT a Critic even if the model scores it negative. Route to Curious Minds.

FIX 3: Emotion misclassification (ANGER on "Even PhD holder faculties cannot...")
  This is handled in sentiment.py / ensemble_classifier.py, not here.
  But we add a segment-level guard: if intent == "praise" or positive_score > 0.7,
  never assign Critic regardless of raw neg score.

FIX 4: Casual Viewer under-scoring on Hinglish praise
  "bhaiya litreally isse best kisine explain nhi kiya" → Casual Viewer.
  Root cause: 7 likes < 10 threshold AND positivity score just below 0.45.
  Fix: expanded HINGLISH_POSITIVE list + lower Super Fan likes threshold to 5.
"""

import numpy as np
import pandas as pd

SEGMENT_LABELS = {
    0: {"name": "Super Fans",     "color": "#1fd99a", "icon": "⭐",
        "desc": "Highly positive, loyal, engaged supporters"},
    1: {"name": "Critics",        "color": "#ff5b5b", "icon": "🗣️",
        "desc": "Genuinely negative or critical feedback"},
    2: {"name": "Curious Minds",  "color": "#6c63ff", "icon": "🧐",
        "desc": "Genuinely curious, asking real questions"},
    3: {"name": "Casual Viewers", "color": "#7a7f95", "icon": "😐",
        "desc": "Neutral engagement, watching but not deeply invested"},
}

HINGLISH_POSITIVE = [
    "bawaal", "bawal", "mast", "ekdum", "badhiya", "badiya",
    "zabardast", "jhakaas", "ekdam", "dhamaal", "kamaal",
    "shandaar", "bindaas", "sahi hai", "bilkul sahi", "jai hind",
    "jai ho", "bharat mata", "vande mataram",
    # FIX 4: added common Hinglish praise that was being missed
    "best kisine", "isse best", "sabse best", "itna acha",
    "bohot acha", "bahut acha", "bhot acha", "bhot badiya",
    "maja aa", "maza aa", "maza aaya", "maja aaya",
    "hats off", "hat off", "salute", "jai ho",
    "kya baat", "kya bat", "wah wah", "waah",
    "superb sir", "amazing sir", "great sir", "best sir",
    "bhaiya best", "sir best", "nitish sir", "campusx",
    "god level", "mind blowing", "mind blown",
]

# FIX 2: help-request patterns → route to Curious Minds, NOT Critics
HELP_REQUEST_MARKERS = [
    "not working", "giving error", "getting error", "error in",
    "how to fix", "how do i fix", "can you help", "please help",
    "help me", "i am stuck", "i'm stuck", "stuck on",
    "not able to", "unable to", "facing issue", "facing error",
    "installation error", "import error", "module not found",
    "any solution", "any fix", "solved?", "how to solve",
]

# FIX 1: these phrases contain question words but are NOT questions
PRAISE_WITH_QUESTION_WORD = [
    "appreciate how", "appreciate what", "love how", "love what",
    "like how", "like what", "amazing how", "incredible how",
    "great how", "best how", "understand how", "learned how",
    "shows how", "explains how", "taught how", "teaching how",
    "broke down how", "broken down how",
    "no one explains", "no one has explained",
    "better than", "the best way",
]

CURIOUS_MARKERS = [
    "can you explain", "could you explain", "could you please",
    "is it possible", "i want to know", "can someone tell",
    "does anyone know", "please explain", "please tell",
    "what about", "how does this", "why does this",
    "what is the difference", "how can i", "when will",
    "where can i", "which one is", "who can",
]

# Short question-word openers that ARE genuine questions (sentence starts with these)
QUESTION_OPENERS = [
    "why ", "how ", "what ", "when ", "where ", "which ", "who ",
    "is it ", "are there ", "can i ", "will this ", "does this ",
    "should i ", "do i need ", "what if ",
]

CRITICISM_MARKERS = [
    "shameful", "disgusting", "pathetic", "worst", "useless", "waste",
    "corrupt", "corruption", "failure", "failed", "incompetent",
    "disappointed", "misleading", "fake", "propaganda", "liar", "lies",
    "should resign", "step down", "embarrassing", "unacceptable",
    "shame", "terrible", "horrible", "awful", "disgrace",
    # Video-specific criticism (not help requests)
    "clickbait", "waste of time", "dislike", "thumbs down",
    "not worth", "mislead", "copy paste", "plagiar",
    "talking too much", "unnecessary talks", "watch hours only",
    "less on content", "not gripping",
]


def _is_genuine_question(text: str) -> bool:
    """
    Returns True only if the comment is ACTUALLY asking a question,
    not just using a question word inside a praise/statement sentence.
    """
    t       = text.lower().strip()
    has_q   = "?" in text

    # If it contains a phrase that's praise-with-question-word, it's NOT curious
    if any(phrase in t for phrase in PRAISE_WITH_QUESTION_WORD):
        return False

    # If it contains a help request, it IS curious (but routed differently)
    if any(h in t for h in HELP_REQUEST_MARKERS):
        return True

    # Explicit question markers (multi-word phrases — high precision)
    if any(marker in t for marker in CURIOUS_MARKERS):
        return True

    # Starts with a question word AND has "?" → genuine question
    if has_q and any(t.startswith(opener) for opener in QUESTION_OPENERS):
        return True

    # Has "?" and is short (≤ 15 words) — likely a real question
    if has_q and len(text.split()) <= 15:
        return True

    return False


def _correct_sentiment(comment: dict) -> dict:
    text = comment.get("text", "").lower()
    pos  = comment.get("positive_score", 0)
    neg  = comment.get("negative_score", 0)

    if any(sig in text for sig in HINGLISH_POSITIVE):
        comment["positive_score"] = max(pos, 0.72)
        comment["negative_score"] = min(neg, 0.12)
        comment["sentiment"]      = "positive"
        return comment

    has_request    = any(w in text for w in ["please", "kindly", "request"])
    has_compliment = any(w in text for w in [
        "best", "great", "amazing", "no one", "better than",
        "love", "thank", "outstanding", "excellent",
    ])
    if has_request and has_compliment and neg > pos:
        comment["positive_score"] = max(pos, 0.60)
        comment["negative_score"] = min(neg, 0.20)
        comment["sentiment"]      = "positive"

    pos_emojis = ["🙏", "❤", "😍", "🔥", "💪", "👏", "🥰", "😊", "✅", "🎉", "🙌"]
    if sum(1 for e in pos_emojis if e in comment.get("text", "")) >= 2 and neg > pos:
        comment["positive_score"] = max(pos, 0.58)
        comment["negative_score"] = min(neg, 0.22)
        comment["sentiment"]      = "positive"

    return comment


def _compute_scores(comment: dict) -> tuple[float, float, float]:
    text        = comment.get("text", "").lower()
    raw_text    = comment.get("text", "")
    pos         = comment.get("positive_score", 0)
    neg         = comment.get("negative_score", 0)
    intent      = comment.get("intent", "general")
    likes       = min(comment.get("like_count", 0), 500)
    length      = len(raw_text)
    is_question = _is_genuine_question(raw_text)

    # ── Positivity score ──────────────────────────────────────────────────────
    positivity = (
        pos * 0.50
        + (0.20 if intent == "praise" else 0)
        + (0.10 if intent == "suggestion" and pos > 0.4 else 0)
        + (0.15 if likes > 50 else 0.07 if likes > 10 else 0.03 if likes > 4 else 0)
        + (0.05 if length > 100 else 0)
    )

    # ── Criticism score ───────────────────────────────────────────────────────
    has_crit_word  = any(w in text for w in CRITICISM_MARKERS)
    is_sarcastic   = comment.get("is_sarcastic", False)
    is_help        = any(h in text for h in HELP_REQUEST_MARKERS)
    is_praise      = (intent == "praise") or (pos > 0.70)

    # FIX 3: never push toward Critic if the comment is clearly praise
    # FIX 2: help requests do not count as criticism
    crit_base = 0 if (is_praise or is_help) else neg * 0.45
    criticism = (
        crit_base
        + (0.30 if has_crit_word and not is_help else 0)
        + (0.15 if intent == "complaint" and not is_help else 0)
        + (0.10 if is_sarcastic and neg > 0.3 and not is_praise else 0)
        + (0.10 if likes > 20 and neg > 0.5 and not is_praise else 0)
    )

    # ── Curiosity score ───────────────────────────────────────────────────────
    # FIX 1: use _is_genuine_question() instead of raw word matching
    has_crit_word_strict = any(w in text for w in CRITICISM_MARKERS)
    is_rhetorical = has_crit_word_strict and neg > 0.55

    curiosity = (
        (0.50 if is_question and not is_rhetorical else 0)
        + (0.20 if intent == "question" and not is_rhetorical else 0)
        + (0.10 if is_help else 0)   # help requests are curious-flavoured
    )

    return round(positivity, 3), round(criticism, 3), round(curiosity, 3)


def _assign_segment(pos_s: float, crit_s: float, cur_s: float,
                    comment: dict) -> int:
    neg    = comment.get("negative_score", 0)
    pos    = comment.get("positive_score", 0)
    likes  = comment.get("like_count", 0)
    intent = comment.get("intent", "general")
    text   = comment.get("text", "").lower()
    is_help = any(h in text for h in HELP_REQUEST_MARKERS)

    # FIX 3: if model said praise OR strongly positive → never Critic
    is_clearly_positive = (intent == "praise") or (pos > 0.70) or (pos_s > 0.60)

    # Critics — must be genuinely negative, not a help request, not praise
    if (crit_s > 0.35
            and crit_s > pos_s
            and crit_s > cur_s
            and neg > 0.40
            and not is_clearly_positive
            and not is_help):
        return 1

    # FIX 1: Super Fans beat Curious Minds when positivity is strong.
    # This fixes "I appreciate HOW you broke down..." being sent to Curious Minds.
    if (pos_s > 0.45
            and pos_s > crit_s
            and pos_s >= cur_s          # positivity wins ties with curiosity
            and (likes >= 5 or pos > 0.65)):   # FIX 4: lowered from 10 → 5 likes
        return 0

    # Curious Minds — real questions or help requests, non-hostile tone
    if (cur_s > 0.35
            and cur_s > crit_s
            and neg < 0.55):
        return 2

    # Super Fans (second chance — lower bar for strong positive score alone)
    if (pos_s > 0.38
            and pos_s > crit_s
            and pos > 0.55):
        return 0

    return 3  # Casual Viewers


def segment_audience(enriched_comments: list[dict]) -> list[dict]:
    if not enriched_comments:
        return enriched_comments

    for c in enriched_comments:
        _correct_sentiment(c)

    for c in enriched_comments:
        pos_s, crit_s, cur_s = _compute_scores(c)
        seg_id               = _assign_segment(pos_s, crit_s, cur_s, c)
        info                 = SEGMENT_LABELS[seg_id]
        c["segment"]         = seg_id
        c["segment_label"]   = info["name"]
        c["segment_color"]   = info["color"]
        c["segment_icon"]    = info["icon"]

    return enriched_comments


def get_segment_summary(enriched_comments: list[dict]) -> dict:
    if not enriched_comments:
        return {
            info["name"]: {
                "count": 0, "color": info["color"], "icon": info["icon"],
                "description": info["desc"], "avg_likes": 0, "top_comment": "",
            }
            for info in SEGMENT_LABELS.values()
        }

    df      = pd.DataFrame(enriched_comments)
    summary = {}

    for seg_id, info in SEGMENT_LABELS.items():
        if "segment" in df.columns:
            seg_df = df[df["segment"] == seg_id]
        else:
            seg_df = pd.DataFrame()

        avg_likes   = round(seg_df["like_count"].mean(), 1) if len(seg_df) > 0 else 0
        top_comment = ""
        if len(seg_df) > 0 and "text" in seg_df.columns and "like_count" in seg_df.columns:
            top_row     = seg_df.sort_values("like_count", ascending=False).iloc[0]
            top_comment = str(top_row.get("text", ""))[:120]

        summary[info["name"]] = {
            "count":       len(seg_df),
            "color":       info["color"],
            "icon":        info["icon"],
            "description": info["desc"],
            "avg_likes":   avg_likes,
            "top_comment": top_comment,
        }

    return summary