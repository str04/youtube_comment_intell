"""
evaluate_models.py — Fixed version

Fixes from v1:
1. Sentiment label normalization (pos/neg/neu → positive/negative/neutral)
2. Toxicity boolean comparison fixed
3. Removed st.cache_resource (runs standalone without Streamlit)
4. Ensemble logic fixed — model vote gets higher weight than keyword vote
5. Added truncation fix for models with no max length

Run from project root:
    python evaluate_models.py
"""
# pyrefly: ignore [missing-import]
import os
# pyrefly: ignore [missing-import]
import sys
# pyrefly: ignore [missing-import]
import json
# pyrefly: ignore [missing-import]
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.path.dirname(__file__), "models")
os.environ["HF_HOME"]            = os.path.join(os.path.dirname(__file__), "models")

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report
)

# ── Test data ─────────────────────────────────────────────────────────────────
# Format: (text, true_sentiment, true_emotion, true_intent, is_toxic, is_sarcastic)
TEST_DATA = [
    # Clear positive
    ("Thank you sir this helped me a lot",                     "positive", "joy",      "praise",     False, False),
    ("Amazing explanation very good teaching",                  "positive", "joy",      "praise",     False, False),
    ("Got placed as data scientist because of this video",      "positive", "joy",      "general",    False, False),
    ("Best ML course on YouTube no doubt",                      "positive", "joy",      "praise",     False, False),
    ("bawaal video bhai ekdum mast",                            "positive", "joy",      "praise",     False, False),
    ("badhiya explanation zabardast sir",                       "positive", "joy",      "praise",     False, False),
    ("I love your teaching style so clear",                     "positive", "joy",      "praise",     False, False),
    ("This is incredible content keep it up",                   "positive", "joy",      "praise",     False, False),
    ("Got job offer after watching this playlist thank you",    "positive", "joy",      "general",    False, False),
    ("Wow I never knew this concept mind blown",                "positive", "surprise", "general",    False, False),
    ("Outstanding work sir really appreciate it",               "positive", "joy",      "praise",     False, False),
    ("I'm so excited to start this course",                     "positive", "joy",      "general",    False, False),
    ("One of the best channels for learning data science",      "positive", "joy",      "praise",     False, False),
    ("This video changed my perspective completely",            "positive", "joy",      "general",    False, False),
    ("Subscribed immediately after watching this masterpiece",  "positive", "joy",      "praise",     False, False),

    # Positive suggestions (tricky — model often labels as negative)
    ("Please complete the LangGraph playlist sir",              "neutral",  "neutral",  "suggestion", False, False),
    ("Please sir continue this series no one teaches better",   "positive", "joy",      "suggestion", False, False),
    ("Can you make a video on fine tuning LLMs please",         "neutral",  "neutral",  "suggestion", False, False),
    ("Sir please upload daily videos we are waiting",           "neutral",  "neutral",  "suggestion", False, False),
    ("Please add subtitles sir it would help a lot",            "neutral",  "neutral",  "suggestion", False, False),

    # Genuine questions
    ("What is the difference between L1 and L2 regularization","neutral",  "neutral",  "question",   False, False),
    ("How can I get the PDF notes for this lecture",            "neutral",  "neutral",  "question",   False, False),
    ("Why is the Indian flag upside down in this thumbnail",    "neutral",  "neutral",  "question",   False, False),
    ("When will the next video be uploaded",                    "neutral",  "neutral",  "question",   False, False),
    ("Which laptop should I buy for data science",              "neutral",  "neutral",  "question",   False, False),
    ("Is this course suitable for complete beginners",          "neutral",  "neutral",  "question",   False, False),

    # Clear negative
    ("This explanation is completely wrong misleading people",  "negative", "anger",    "complaint",  False, False),
    ("Worst video ever waste of time",                          "negative", "anger",    "complaint",  False, False),
    ("Shameful that our PM has to call citizens for basics",    "negative", "anger",    "complaint",  False, False),
    ("Absolutely useless content pathetic",                     "negative", "anger",    "complaint",  False, False),
    ("I am so disappointed with this channel",                  "negative", "sadness",  "complaint",  False, False),
    ("Unable to find the notes very disappointed",              "negative", "sadness",  "complaint",  False, False),
    ("The audio quality is terrible unwatchable",               "negative", "anger",    "complaint",  False, False),
    ("This is not beginner friendly at all very confusing",     "negative", "anger",    "complaint",  False, False),

    # Toxic
    ("This is fake propaganda stop spreading lies",             "negative", "anger",    "complaint",  True,  False),
    ("You are an idiot stop teaching you know nothing",         "negative", "anger",    "complaint",  True,  False),
    ("This is garbage trash content delete your channel",       "negative", "anger",    "complaint",  True,  False),

    # Sarcasm
    ("Oh great another video that explains nothing perfectly",  "negative", "anger",    "complaint",  False, True),
    ("Modi is the gift that keeps on giving hope he rules 20 years", "negative", "anger", "general", False, True),
    ("Yeah because that always works out so well",              "negative", "anger",    "general",    False, True),
    ("Wow such an original explanation never heard this before","negative", "anger",    "complaint",  False, True),

    # Neutral / informational
    ("Watching this at 2am before my exam",                     "neutral",  "neutral",  "general",    False, False),
    ("This video is 6 hours long",                              "neutral",  "neutral",  "general",    False, False),
    ("I am a working professional with 10 years experience",    "neutral",  "neutral",  "general",    False, False),
    ("Paused at 5 42 to take notes",                            "neutral",  "neutral",  "general",    False, False),

    # Hinglish
    ("Bhai ye video dekh ke meri life change ho gayi",          "positive", "joy",      "general",    False, False),
    ("Yaar kya samjhaya hai bilkul sahi baat hai",              "positive", "joy",      "praise",     False, False),
    ("Sir aap bahut acha padhate ho thank you",                 "positive", "joy",      "praise",     False, False),
    ("Nahi samjha bhai please explain karo dobara",             "neutral",  "neutral",  "question",   False, False),
    ("College mein HOD ne pucha kahan se padhte ho CampusX",    "positive", "joy",      "general",    False, False),

    # Sadness / fear
    ("I am scared I won't be able to get a job in AI",          "negative", "fear",     "general",    False, False),
    ("Missed the live session so sad",                          "negative", "sadness",  "general",    False, False),
    ("I am worried about the job market after AI",              "negative", "fear",     "general",    False, False),

    # Surprise
    ("No way I never knew Python could do this",                "positive", "surprise", "general",    False, False),
    ("Wow this analogy is absolutely incredible",               "positive", "surprise", "praise",     False, False),
]

# ── Label normalization ───────────────────────────────────────────────────────
LABEL_MAP = {
    "pos": "positive", "neg": "negative", "neu": "neutral",
    "positive": "positive", "negative": "negative", "neutral": "neutral",
    "label_0": "negative", "label_1": "neutral", "label_2": "positive",
}

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
    "joy":     ["thank","great","amazing","awesome","love","best","excellent",
                "brilliant","helped","got placed","got job","bawaal","badhiya",
                "mast","ekdum","zabardast","kamaal","shandaar","❤","🙏","😍","🥰"],
    "anger":   ["hate","worst","terrible","pathetic","useless","waste",
                "shameful","corrupt","liar","fake","disgusting","wrong","misleading"],
    "sadness": ["sad","miss","disappointed","unable","can't find","😢","😭"],
    "surprise":["wow","omg","unbelievable","mind blown","never knew","no way","😮"],
    "fear":    ["scared","worried","nervous","afraid","anxious"],
    "disgust": ["gross","revolting","vile","nasty","🤢"],
}

SENTIMENT_VALID = {
    "positive": ["joy","surprise","neutral"],
    "negative": ["anger","sadness","fear","disgust","neutral"],
    "neutral":  ["joy","anger","sadness","fear","surprise","disgust","neutral"],
}


# ── Model loaders (no st.cache_resource) ─────────────────────────────────────
def load_pipeline(model_id, task="text-classification", top_k=None):
    # pyrefly: ignore [missing-import]
    from transformers import pipeline
    cache = os.path.join(os.path.dirname(__file__), "models")
    kwargs = {"model": model_id, "model_kwargs": {"cache_dir": cache}}
    if top_k:
        kwargs["top_k"] = top_k
    return pipeline(task, **kwargs)


# ── Inference functions ───────────────────────────────────────────────────────
def run_sentiment(texts, model):
    outputs = model(
        [t[:512] for t in texts],
        batch_size=32, truncation=True, max_length=512
    )
    results = []
    for out in outputs:
        if isinstance(out, list):
            scores   = {LABEL_MAP.get(i["label"].lower(), i["label"].lower()): i["score"] for i in out}
            dominant = max(scores, key=scores.get)
        else:
            dominant = LABEL_MAP.get(out["label"].lower(), out["label"].lower())
        results.append(dominant)
    return results


def run_emotion_single(texts, model):
    outputs = model(
        [t[:512] for t in texts],
        batch_size=32, truncation=True, max_length=512
    )
    results = []
    for out in outputs:
        grouped = {}
        items   = out if isinstance(out, list) else [out]
        for item in items:
            mapped          = GO_EMOTION_MAP.get(item["label"], "neutral")
            grouped[mapped] = grouped.get(mapped, 0) + item["score"]
        results.append(max(grouped, key=grouped.get))
    return results


def run_toxicity(texts, model):
    outputs = model(
        [t[:512] for t in texts],
        batch_size=32, truncation=True, max_length=512
    )
    results = []
    for out in outputs:
        label = out["label"].lower() if isinstance(out, dict) else out[0]["label"].lower()
        results.append(label == "toxic")
    return results


def run_sarcasm(texts, model):
    outputs = model(
        [t[:512] for t in texts],
        batch_size=32, truncation=True, max_length=512
    )
    results = []
    for out in outputs:
        label = out["label"].lower() if isinstance(out, dict) else out[0]["label"].lower()
        results.append(label == "irony")
    return results


def run_intent(texts):
    from nlp.intent_classifier import classify_intent
    return [classify_intent(t) for t in texts]


def run_ensemble(texts, sentiments, single_preds):
    """
    Fixed ensemble: model vote weighted 2x, sentiment + keyword 1x each.
    Model vote is most reliable so it gets double weight.
    """
    from collections import Counter

    kw_votes = []
    for text in texts:
        t       = text.lower()
        matched = None
        for emotion, kws in KEYWORD_EMOTIONS.items():
            if any(k in t for k in kws):
                matched = emotion
                break
        kw_votes.append(matched or "neutral")

    sent_votes = []
    for s in sentiments:
        if s == "positive":   sent_votes.append("joy")
        elif s == "negative": sent_votes.append("anger")
        else:                 sent_votes.append("neutral")

    results = []
    for i in range(len(texts)):
        # Model gets 2 votes, others get 1 each
        votes  = [single_preds[i], single_preds[i], sent_votes[i], kw_votes[i]]
        counts = Counter(votes)
        winner = counts.most_common(1)[0][0]

        # Enforce sentiment validity
        valid = SENTIMENT_VALID.get(sentiments[i], list(GO_EMOTION_MAP.values()))
        if winner not in valid:
            for emotion, _ in counts.most_common():
                if emotion in valid:
                    winner = emotion
                    break
            else:
                winner = "joy" if sentiments[i] == "positive" else \
                         "anger" if sentiments[i] == "negative" else "neutral"
        results.append(winner)
    return results


# ── Report printer ─────────────────────────────────────────────────────────────
def print_report(name, y_true, y_pred, labels=None):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy:  {acc*100:.1f}%")
    if labels:
        report = classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        )
        print(f"\n  Per-class breakdown:")
        for label in labels:
            if label in report:
                r = report[label]
                print(f"  {label:<12} precision={r['precision']:.2f}  "
                      f"recall={r['recall']:.2f}  f1={r['f1-score']:.2f}  "
                      f"support={int(r['support'])}")
    else:
        p = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        r = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f = f1_score(y_true, y_pred, average="weighted", zero_division=0)
        print(f"  Precision: {p*100:.1f}%")
        print(f"  Recall:    {r*100:.1f}%")
        print(f"  F1 Score:  {f*100:.1f}%")
    return acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  YT Intel — Model Evaluation Report")
    print("="*55)
    print(f"  Test samples: {len(TEST_DATA)}")

    texts          = [d[0] for d in TEST_DATA]
    true_sentiment = [d[1] for d in TEST_DATA]
    true_emotion   = [d[2] for d in TEST_DATA]
    true_intent    = [d[3] for d in TEST_DATA]
    true_toxic     = [d[4] for d in TEST_DATA]
    true_sarcasm   = [d[5] for d in TEST_DATA]

    print("\n Loading models from cache...")
    sent_model = load_pipeline(
        "cardiffnlp/twitter-roberta-base-sentiment-latest", top_k=None
    )
    emo_model  = load_pipeline(
        "SamLowe/roberta-base-go_emotions", top_k=None
    )
    tox_model  = load_pipeline("unitary/toxic-bert")
    sar_model  = load_pipeline("cardiffnlp/twitter-roberta-base-irony")

    scores = {}

    # 1. Sentiment
    print("\n Running sentiment analysis...")
    pred_sentiment = run_sentiment(texts, sent_model)
    scores["sentiment"] = print_report(
        "1. Sentiment Model",
        true_sentiment, pred_sentiment,
        labels=["positive", "negative", "neutral"]
    )

    # 2. Emotion single
    print("\n Running emotion detection (single model)...")
    pred_emotion_single = run_emotion_single(texts, emo_model)
    scores["emotion_single"] = print_report(
        "2. Emotion — Single Model (go_emotions)",
        true_emotion, pred_emotion_single,
        labels=["joy","anger","sadness","fear","surprise","disgust","neutral"]
    )

    # 3. Ensemble
    print("\n Running ensemble emotion classifier...")
    pred_emotion_ensemble = run_ensemble(texts, pred_sentiment, pred_emotion_single)
    scores["emotion_ensemble"] = print_report(
        "3. Emotion — Ensemble (model×2 + sentiment + keywords)",
        true_emotion, pred_emotion_ensemble,
        labels=["joy","anger","sadness","fear","surprise","disgust","neutral"]
    )

    # 4. Toxicity
    print("\n Running toxicity detection...")
    pred_toxic = run_toxicity(texts, tox_model)
    scores["toxicity"] = print_report(
        "4. Toxicity Model",
        true_toxic, pred_toxic
    )

    # 5. Sarcasm
    print("\n Running sarcasm detection...")
    pred_sarcasm = run_sarcasm(texts, sar_model)
    scores["sarcasm"] = print_report(
        "5. Sarcasm Model",
        true_sarcasm, pred_sarcasm
    )

    # 6. Intent
    print("\n Running intent classification...")
    pred_intent = run_intent(texts)
    scores["intent"] = print_report(
        "6. Intent Classifier",
        true_intent, pred_intent,
        labels=["praise","complaint","question","suggestion","general"]
    )

    # Summary
    improvement = (scores["emotion_ensemble"] - scores["emotion_single"]) * 100
    print(f"\n{'='*55}")
    print(f"  FINAL SUMMARY")
    print(f"{'='*55}")
    print(f"  Sentiment accuracy:          {scores['sentiment']*100:.1f}%")
    print(f"  Emotion (single model):      {scores['emotion_single']*100:.1f}%")
    print(f"  Emotion (ensemble):          {scores['emotion_ensemble']*100:.1f}%")
    sign = '+' if improvement >= 0 else ''
    print(f"  Ensemble improvement:        {sign}{improvement:.1f}%")
    print(f"  Toxicity accuracy:           {scores['toxicity']*100:.1f}%")
    print(f"  Sarcasm accuracy:            {scores['sarcasm']*100:.1f}%")
    print(f"  Intent accuracy:             {scores['intent']*100:.1f}%")
    print(f"\n  Test set size: {len(TEST_DATA)} samples")
    print(f"{'='*55}\n")

    results = {k: round(v*100, 1) for k, v in scores.items()}
    results["ensemble_gain"]  = round(improvement, 1)
    results["test_samples"]   = len(TEST_DATA)
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("  Results saved to evaluation_results.json\n")


if __name__ == "__main__":
    main()