def classify_intent(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ["?", "how", "why", "what", "when", "where", "who", "which", "can you", "could you"]):
        return "question"
    if any(w in t for w in ["should", "please", "add", "make", "improve", "fix",
                              "need", "want", "would be better", "consider", "suggest"]):
        return "suggestion"
    if any(w in t for w in ["love", "amazing", "great", "awesome", "best", "perfect",
                              "fantastic", "excellent", "brilliant", "goat", "🔥", "❤", "👏", "💯"]):
        return "praise"
    if any(w in t for w in ["hate", "worst", "bad", "boring", "waste", "dislike",
                              "trash", "terrible", "useless", "disappointed", "clickbait"]):
        return "complaint"
    return "general"

def classify_intents(texts: list[str]) -> list[str]:
    return [classify_intent(t) for t in texts]