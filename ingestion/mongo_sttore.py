import os
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

_client = None


def get_db():
    global _client
    if _client is None:
        _client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
    return _client["yt_comment_intel"]


def save_video(metadata: dict) -> None:
    db = get_db()
    db.videos.replace_one(
        {"video_id": metadata["video_id"]}, metadata, upsert=True
    )


def save_comments(video_id: str, comments: list[dict]) -> None:
    db = get_db()
    if not comments:
        return
    for c in comments:
        c["video_id"] = video_id
    db.comments.delete_many({"video_id": video_id})
    db.comments.insert_many(comments)


def save_analysis(video_id: str, analysis: dict) -> None:
    db  = get_db()
    doc = {
        "video_id":   video_id,
        "analysis":   analysis,
        "created_at": datetime.utcnow(),
    }
    db.analyses.replace_one({"video_id": video_id}, doc, upsert=True)


def load_analysis(video_id: str) -> dict | None:
    db  = get_db()
    doc = db.analyses.find_one({"video_id": video_id}, {"_id": 0})
    return doc


def load_comments(video_id: str) -> list[dict]:
    db   = get_db()
    docs = db.comments.find({"video_id": video_id}, {"_id": 0})
    return list(docs)


def analysis_exists(video_id: str) -> bool:
    db = get_db()
    return db.analyses.count_documents({"video_id": video_id}) > 0


def list_recent(limit: int = 10) -> list[dict]:
    db   = get_db()
    docs = db.analyses.find(
        {},
        {"video_id": 1, "analysis.metadata.title": 1,
         "analysis.approval_score": 1, "created_at": 1, "_id": 0}
    ).sort("created_at", -1).limit(limit)
    return list(docs)