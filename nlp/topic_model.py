from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

_topic_model = None

# Standard English stop words + YouTube comment noise words
STOP_WORDS = [
    "you", "the", "this", "to", "and", "of", "in", "is", "it", "for",
    "that", "a", "are", "was", "be", "have", "with", "on", "at", "by",
    "from", "as", "an", "or", "but", "not", "so", "if", "we", "he",
    "she", "they", "i", "my", "your", "his", "her", "our", "their",
    "me", "him", "us", "them", "what", "which", "who", "how", "when",
    "where", "why", "can", "will", "do", "does", "did", "has", "had",
    "been", "being", "would", "could", "should", "may", "might", "must",
    "just", "also", "more", "very", "much", "about", "up", "out", "no",
    "one", "all", "there", "their", "than", "then", "now", "its", "sir",
    "bhai", "hai", "ka", "ki", "ke", "ko", "se", "ne", "ye", "yeh",
    "kya", "aur", "nahi", "hain", "tha", "thi", "ho", "hoga", "mera",
    "tera", "apna", "please", "thank", "thanks", "video", "watch",
    "watching", "watched", "see", "like", "love", "great", "good",
    "amazing", "best", "awesome", "nice", "well", "really", "actually",
    "definitely", "absolutely", "totally", "literally", "basically",
    "http", "https", "www", "com", "youtube", "channel", "subscribe",
]


def run_topic_modeling(texts: list[str]) -> dict:
    global _topic_model

    if len(texts) < 10:
        return {"topics": [], "assignments": [0] * len(texts)}

    # CountVectorizer with stop words removes noise words from topic labels
    vectorizer = CountVectorizer(
        stop_words=STOP_WORDS,
        min_df=2,
        ngram_range=(1, 2),   # allows bigrams like "decision tree", "linear regression"
    )

    _topic_model = BERTopic(
        language="multilingual",
        min_topic_size=5,
        vectorizer_model=vectorizer,
        verbose=False,
    )

    topics, _ = _topic_model.fit_transform(texts)

    topic_info = _topic_model.get_topic_info()
    topic_list = []
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:   # -1 is the outlier/noise topic, skip it
            continue
        words = _topic_model.get_topic(row["Topic"])
        # Filter out any stop words that slipped through
        clean_words = [w for w, _ in words if w.lower() not in STOP_WORDS][:8]
        if not clean_words:
            continue
        topic_list.append({
            "topic_id": int(row["Topic"]),
            "count":    int(row["Count"]),
            "label":    ", ".join(clean_words[:4]),
            "words":    clean_words,
        })

    return {
        "topics":      topic_list,
        "assignments": [int(t) for t in topics],
    }