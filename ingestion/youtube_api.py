"""
ingestion/youtube_api.py

Fast, single-call comment + metadata fetcher using yt-dlp.

Fix: extractor_args max_comments format was breaking comment extraction
     in yt-dlp >= 2023.x. Now uses the correct approach.
"""

import re
import yt_dlp


# ── URL helpers ───────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """Pull the 11-char video ID from any YouTube URL format."""
    match = re.search(r"youtu\.be/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    match = re.search(r"[?&]v=([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    match = re.search(r"/(?:shorts|embed|live|v)/([A-Za-z0-9_-]{11})", url)
    if match:
        return match.group(1)
    return None


def clean_url(url: str) -> str:
    """
    Strip everything except ?v=VIDEO_ID so yt-dlp NEVER sees &list=
    and tries to crawl a whole playlist.
    """
    video_id = extract_video_id(url)
    if video_id:
        return f"https://www.youtube.com/watch?v={video_id}"
    return url


# ── Main fetch function ───────────────────────────────────────────────────────

def fetch_all(url: str, max_comments: int = 200) -> tuple[dict, list[dict]]:
    """
    Single yt-dlp call — returns (meta dict, raw comments list).
    """
    clean = clean_url(url)

    ydl_opts = {
        "quiet":          True,
        "no_warnings":    True,
        "skip_download":  True,
        "noplaylist":     True,         # never crawl playlists
        "socket_timeout": 30,

        # getcomments=True is the correct yt-dlp flag for comment extraction.
        # Do NOT pass max_comments inside extractor_args as a list of ints —
        # that format broke in yt-dlp ~2023.07 and returns 0 comments.
        # Instead we fetch all available top-level comments and cap below.
        "getcomments": True,

        "writeinfojson":     False,
        "writethumbnail":    False,
        "writesubtitles":    False,
        "writeautomaticsub": False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(clean, download=False)

    if info is None:
        raise ValueError(f"yt-dlp returned no info for: {clean}")

    # ── Metadata ──────────────────────────────────────────────────────────────
    meta = {
        "video_id":      info.get("id", ""),
        "title":         info.get("title", "Unknown"),
        "channel":       info.get("uploader") or info.get("channel", "Unknown"),
        "channel_id":    info.get("channel_id", ""),
        "view_count":    info.get("view_count")    or 0,
        "like_count":    info.get("like_count")    or 0,
        "comment_count": info.get("comment_count") or 0,
        "duration":      info.get("duration")      or 0,
        "upload_date":   info.get("upload_date",  ""),
        "description":   (info.get("description") or "")[:500],
        "thumbnail":     info.get("thumbnail",    ""),
        "tags":          info.get("tags")          or [],
        "url":           clean,
    }

    # ── Comments ──────────────────────────────────────────────────────────────
    raw_comments: list[dict] = info.get("comments") or []

    # Sort by like_count so we keep the most engaged comments when capping
    if len(raw_comments) > max_comments:
        raw_comments = sorted(
            raw_comments,
            key=lambda c: c.get("like_count") or 0,
            reverse=True,
        )[:max_comments]

    return meta, raw_comments