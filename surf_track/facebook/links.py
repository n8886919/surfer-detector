from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


FACEBOOK_HOSTS = {
    "facebook.com",
    "www.facebook.com",
    "m.facebook.com",
    "web.facebook.com",
}
SENSITIVE_QUERY_KEYS = {"access_token", "auth", "cookie", "password", "session", "token"}
_CONTENT_ID = re.compile(r"^[A-Za-z0-9._-]{3,}$")


class FacebookLinkError(ValueError):
    """Raised when a value is not a supported Facebook content URL."""


@dataclass(frozen=True, slots=True)
class FacebookLink:
    kind: str
    canonical_url: str
    content_id: str
    group_id: str | None
    removed_query_parameter_count: int

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "provider": "facebook",
            "kind": self.kind,
            "canonical_url": self.canonical_url,
            "content_id": self.content_id,
            "tracking_parameters_removed": self.removed_query_parameter_count,
        }
        if self.group_id:
            result["group_id"] = self.group_id
        return result


def parse_facebook_link(value: str) -> FacebookLink:
    raw = value.strip()
    if not raw:
        raise FacebookLinkError("請貼上 Facebook 貼文或影片連結。")
    if raw.startswith(("facebook.com/", "www.facebook.com/", "m.facebook.com/")):
        raw = f"https://{raw}"

    parsed = urlparse(raw)
    host = (parsed.hostname or "").lower()
    if parsed.scheme not in {"http", "https"} or host not in FACEBOOK_HOSTS:
        raise FacebookLinkError("目前只接受 facebook.com 的貼文或影片連結。")

    query_pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query_map = dict(query_pairs)
    if any(
        marker in key.lower()
        for key, _ in query_pairs
        for marker in SENSITIVE_QUERY_KEYS
    ):
        raise FacebookLinkError("連結含有疑似登入憑證，請勿貼上 token、cookie 或 session。")

    segments = [segment for segment in parsed.path.split("/") if segment]
    kind: str | None = None
    content_id: str | None = None
    group_id: str | None = None
    canonical_path: str | None = None
    kept_query: list[tuple[str, str]] = []

    if len(segments) >= 4 and segments[0] == "groups" and segments[2] in {"posts", "permalink"}:
        group_id = segments[1]
        content_id = segments[3]
        kind = "group_post"
        canonical_path = f"/groups/{group_id}/posts/{content_id}/"
    elif len(segments) >= 2 and segments[0] in {"reel", "reels"}:
        content_id = segments[1]
        kind = "reel"
        canonical_path = f"/reel/{content_id}/"
    elif segments and segments[0] == "watch" and query_map.get("v"):
        content_id = query_map["v"]
        kind = "video"
        canonical_path = "/watch/"
        kept_query = [("v", content_id)]
    elif "videos" in segments:
        index = segments.index("videos")
        if index + 1 < len(segments):
            content_id = segments[index + 1]
            kind = "video"
            canonical_path = f"/{'/'.join(segments[:index + 2])}/"
    elif "posts" in segments:
        index = segments.index("posts")
        if index + 1 < len(segments):
            content_id = segments[index + 1]
            kind = "post"
            canonical_path = f"/{'/'.join(segments[:index + 2])}/"
    elif len(segments) >= 3 and segments[0] == "share" and segments[1] in {"p", "r", "v"}:
        content_id = segments[2]
        kind = "share"
        canonical_path = f"/share/{segments[1]}/{content_id}/"
    elif parsed.path.rstrip("/") == "/story.php" and query_map.get("story_fbid"):
        content_id = query_map["story_fbid"]
        kind = "post"
        canonical_path = "/story.php"
        kept_query = [("story_fbid", content_id)]
        if query_map.get("id"):
            kept_query.append(("id", query_map["id"]))

    if not kind or not content_id or not canonical_path or not _CONTENT_ID.fullmatch(content_id):
        raise FacebookLinkError("目前無法從這個 Facebook 連結辨識貼文或影片 ID。")
    if group_id and not _CONTENT_ID.fullmatch(group_id):
        raise FacebookLinkError("Facebook 社團 ID 格式不正確。")

    kept_keys = {key for key, _ in kept_query}
    removed_count = sum(1 for key, _ in query_pairs if key not in kept_keys)
    canonical_url = urlunparse(
        ("https", "www.facebook.com", canonical_path, "", urlencode(kept_query), "")
    )
    return FacebookLink(
        kind=kind,
        canonical_url=canonical_url,
        content_id=content_id,
        group_id=group_id,
        removed_query_parameter_count=removed_count,
    )
