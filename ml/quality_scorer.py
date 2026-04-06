import time
import functools


def timer_decorator(func):
    """Decorator that times function execution and prints elapsed time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start   = time.time()
        result  = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[Timer] {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def score_quality(comment: dict) -> float:
    text         = comment.get("text", "")
    length_score = min(len(text) / 200, 1.0)
    like_score   = min(comment.get("like_count", 0) / 50, 1.0)
    toxic_pen    = -0.5 if comment.get("is_toxic") else 0
    intent_bonus = 0.2 if comment.get("intent") in ["question", "suggestion"] else 0
    pos_score    = comment.get("positive_score", 0)
    neg_score    = comment.get("negative_score", 0)

    quality = (
        0.30 * length_score +
        0.30 * like_score   +
        0.20 * pos_score    +
        0.10 * (1 - neg_score) +
        toxic_pen + intent_bonus
    )
    return round(max(0.0, min(1.0, quality)), 3)


@timer_decorator
def score_all_comments(enriched_comments: list[dict]) -> list[dict]:
    for c in enriched_comments:
        c["quality_score"] = score_quality(c)
    return enriched_comments