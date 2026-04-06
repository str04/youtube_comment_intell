import re
import emoji


def clean_text(text: str) -> str:
    """Remove URLs, extra whitespace, normalize emoji."""
    text = re.sub(r"http\S+|www\S+", "", text)       # remove URLs
    text = re.sub(r"\s+", " ", text).strip()           # normalize whitespace
    text = emoji.demojize(text, delimiters=(" ", " ")) # convert emoji to text
    return text


def deduplicate(comments: list[dict]) -> list[dict]:
    """Remove exact duplicate comment texts."""
    seen = set()
    unique = []
    for c in comments:
        key = c["text"].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def filter_empty(comments: list[dict]) -> list[dict]:
    """Remove comments with no meaningful text."""
    return [c for c in comments if len(c.get("text", "").strip()) > 3]


def thread_comments(comments: list[dict]) -> dict:
    """
    Separate top-level comments from replies.
    Returns { top_level: [...], replies: [...] }
    """
    top_level = [c for c in comments if not c["is_reply"]]
    replies   = [c for c in comments if c["is_reply"]]
    return {"top_level": top_level, "replies": replies}


def parse_comments(raw_comments: list[dict]) -> list[dict]:
    """
    Full parsing pipeline:
    1. Filter empty
    2. Deduplicate
    3. Clean text (keep original too)
    """
    comments = filter_empty(raw_comments)
    comments = deduplicate(comments)
    for c in comments:
        c["clean_text"] = clean_text(c["text"])
    return comments