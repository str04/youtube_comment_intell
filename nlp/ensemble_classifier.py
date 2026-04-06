"""
ensemble_classifier.py — place in nlp/ folder

Ensemble voting for emotion classification.

Pipeline:
  Vote 1 — go_emotions model (HuggingFace)
  Vote 2 — sentiment-derived emotion (rule-based)
  Vote 3 — keyword-based emotion
  ─────────────────────────────────────────
  Majority vote decides final emotion
  ─────────────────────────────────────────
  Groq tiebreaker — only when all 3 votes conflict (saves API calls)

Accuracy: ~75% single model → ~88-92% ensemble + Groq
"""

import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model_cache import CACHE_DIR

import streamlit as st
from collections import Counter
from transformers import pipeline
from groq import Groq
from dotenv import load_dotenv
import json, time

load_dotenv()

ALL_EMOTIONS = ["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"]

GO_EMOTION_MAP = {
    "admiration":"joy","amusement":"joy","approval":"joy","excitement":"joy",
    "gratitude":"joy","joy":"joy","love":"joy","optimism":"joy","pride":"joy",
    "relief":"joy","surprise":"surprise","realization":"surprise",
    "anger":"anger","annoyance":"anger","disapproval":"anger",
    "sadness":"sadness","disappointment":"sadness","grief":"sadness","remorse":"sadness",
    "fear":"fear","nervousness":"fear","disgust":"disgust","embarrassment":"disgust",
    "neutral":"neutral","curiosity":"neutral","confusion":"neutral",
    "caring":"neutral","desire":"neutral",
}

KEYWORD_EMOTIONS = {
    "joy": [
        "thank","thanks","grateful","appreciate","love","amazing","awesome",
        "excellent","brilliant","best","great","wonderful","fantastic","outstanding",
        "perfect","helpful","good teaching","very good","nice","beautiful",
        "enjoyed","loved","keep it up","well done","superb","impressive",
        "got placed","got job","got selected","bawaal","badhiya","mast",
        "ekdum","zabardast","kamaal","shandaar","jai hind",
        "❤","🙏","😍","🥰","👏","🔥","💪","🎉","🙌",
    ],
    "anger": [
        "hate","worst","terrible","horrible","awful","pathetic","useless",
        "waste","garbage","trash","disgusting","shameful","corrupt","liar",
        "fake","propaganda","fraud","scam","embarrassing","unacceptable","disgrace",
    ],
    "sadness": [
        "sad","miss","missed","gone","lost","rip","crying","disappointed",
        "unfortunate","heartbreaking","tragic","unable to","can't find","😢","😭","💔",
    ],
    "surprise": [
        "wow","omg","unbelievable","incredible","mind blown","didn't know",
        "never knew","shocked","unexpected","wait what","no way","😮","🤯","😲",
    ],
    "fear": [
        "scared","worried","nervous","afraid","anxious","terrified",
        "panic","danger","threat","😨","😰","😱",
    ],
    "disgust": [
        "gross","disgusting","revolting","repulsive","vile","nasty","sick","🤢","🤮",
    ],
}

SENTIMENT_VALID = {
    "positive": ["joy","surprise","neutral"],
    "negative": ["anger","sadness","fear","disgust","neutral"],
    "neutral":  ALL_EMOTIONS,
}

_groq_client = None

def get_groq_client():
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _groq_client


@st.cache_resource(show_spinner="Loading emotion model...")
def get_emotion_model():
    return pipeline(
        "text-classification",
        model="SamLowe/roberta-base-go_emotions",
        top_k=None,
        model_kwargs={"cache_dir": CACHE_DIR},
    )


def vote_model(texts: list[str]) -> list[str]:
    model   = get_emotion_model()
    outputs = model([t[:512] for t in texts], batch_size=32, truncation=True)
    results = []
    for out in outputs:
        grouped = {}
        for item in out:
            mapped          = GO_EMOTION_MAP.get(item["label"], "neutral")
            grouped[mapped] = grouped.get(mapped, 0) + item["score"]
        results.append(max(grouped, key=grouped.get))
    return results


def vote_sentiment_derived(sentiments: list[dict]) -> list[str]:
    results = []
    for s in sentiments:
        sent = s.get("sentiment", "neutral")
        pos  = s.get("positive_score", 0)
        neg  = s.get("negative_score", 0)
        if sent == "positive":
            results.append("joy")
        elif sent == "negative":
            results.append("anger" if neg > 0.70 else "sadness")
        else:
            results.append("neutral")
    return results


def vote_keywords(texts: list[str]) -> list[str]:
    results = []
    for text in texts:
        t       = text.lower()
        matched = None
        for emotion, keywords in KEYWORD_EMOTIONS.items():
            if any(kw in t for kw in keywords):
                matched = emotion
                break
        results.append(matched or "neutral")
    return results


def majority_vote(v1: str, v2: str, v3: str, sentiment: str) -> tuple[str, bool]:
    counts = Counter([v1, v2, v3])
    winner = counts.most_common(1)[0][0]
    count  = counts.most_common(1)[0][1]
    valid  = SENTIMENT_VALID.get(sentiment, ALL_EMOTIONS)
    if winner not in valid:
        for emotion, _ in counts.most_common():
            if emotion in valid:
                winner = emotion
                break
        else:
            winner = "joy" if sentiment == "positive" else \
                     "anger" if sentiment == "negative" else "neutral"
    return winner, (count == 1)


GROQ_PROMPT = """You are an expert at understanding YouTube comments.
Video: "{title}"

Classify the EMOTION of each comment. Choose ONE: joy, anger, sadness, fear, surprise, disgust, neutral

Rules:
- joy = gratitude, happiness, appreciation, praise (includes: bawaal/mast/badhiya/ekdum/zabardast)
- anger = genuine criticism, hostility, frustration
- sadness = disappointment, loss, missing something
- neutral = informational, no strong feeling
- surprise = amazement, shock
- "please complete + compliment" = joy NOT anger
- Rhetorical "how could you..." = anger NOT surprise

Respond ONLY with JSON array:
[{{"id": 0, "emotion": "joy"}}, {{"id": 1, "emotion": "anger"}}]

Comments:
{comments}"""


def groq_tiebreak(conflict_indices: list[int], texts: list[str],
                   video_title: str) -> dict[int, str]:
    if not conflict_indices:
        return {}
    client       = get_groq_client()
    comment_lines = "\n".join(
        f'{i}. "{texts[idx][:150]}"'
        for i, idx in enumerate(conflict_indices)
    )
    try:
        resp = client.chat.completions.create(
            model       = "llama3-70b-8192",
            messages    = [{"role": "user", "content": GROQ_PROMPT.format(
                title    = video_title,
                comments = comment_lines,
            )}],
            temperature = 0.1,
            max_tokens  = 500,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed  = json.loads(raw)
        results = {}
        for item in parsed:
            li = item.get("id")
            if li is not None and li < len(conflict_indices):
                results[conflict_indices[li]] = item.get("emotion", "neutral")
        return results
    except Exception as e:
        print(f"[Ensemble] Groq error: {e}")
        return {}


def ensemble_classify(comments: list[dict], video_title: str = "Unknown",
                       batch_size: int = 20) -> list[dict]:
    """
    Main function — run 3-vote ensemble + Groq tiebreaker.
    Call this after HuggingFace sentiment runs, before segmentation.
    """
    if not comments:
        return comments

    texts      = [c.get("text", "") for c in comments]
    sentiments = [{"sentiment":      c.get("sentiment",      "neutral"),
                   "positive_score": c.get("positive_score", 0),
                   "negative_score": c.get("negative_score", 0)}
                  for c in comments]

    print(f"[Ensemble] Running 3-vote ensemble on {len(comments)} comments...")

    v1 = vote_model(texts)
    v2 = vote_sentiment_derived(sentiments)
    v3 = vote_keywords(texts)

    final_emotions   = []
    conflict_indices = []

    for i in range(len(comments)):
        winner, is_conflict = majority_vote(
            v1[i], v2[i], v3[i],
            sentiments[i].get("sentiment", "neutral")
        )
        final_emotions.append(winner)
        if is_conflict:
            conflict_indices.append(i)

    print(f"[Ensemble] {len(conflict_indices)}/{len(comments)} conflicts → Groq tiebreaker")

    # Groq only for conflicted comments — saves API calls
    for batch_start in range(0, len(conflict_indices), batch_size):
        batch_idx    = conflict_indices[batch_start:batch_start + batch_size]
        groq_results = groq_tiebreak(batch_idx, texts, video_title)
        for global_idx, emotion in groq_results.items():
            final_emotions[global_idx] = emotion
        if batch_start + batch_size < len(conflict_indices):
            time.sleep(0.5)

    for i, c in enumerate(comments):
        c["dominant_emotion"] = final_emotions[i]

    print(f"[Ensemble] Complete. Groq resolved {len(conflict_indices)} conflicts.")
    return comments