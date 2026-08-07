"""
Yam's media generation helpers — Pollinations.ai. These build a URL rather
than downloading bytes: fetching the URL IS the generated image/video, so
no binary handling or hosting is needed anywhere in Leo's codebase.

IMPORTANT: Pollinations now requires an API key on ALL generation requests
(this changed after this integration was first built — it used to be
truly keyless). Get a free key at https://enter.pollinations.ai and set
it as the POLLINATIONS_API_KEY env var on Leo's Railway service. Without
it, every call here will fail with a 401 "Missing or invalid API key"
error from Pollinations itself.

Docs: https://gen.pollinations.ai/docs
"""

import os
import tempfile
import requests
from urllib.parse import quote

POLLINATIONS_BASE = "https://gen.pollinations.ai"
POLLINATIONS_MEDIA_BASE = "https://media.pollinations.ai"

# Standard social formats -> (width, height), for the image path.
FORMAT_DIMENSIONS = {
    "square": (1080, 1080),      # Instagram feed
    "story": (1080, 1920),       # Stories / Reels / Shorts
    "landscape": (1200, 630),    # Facebook / LinkedIn link preview
    "portrait": (1080, 1350),    # Instagram portrait
}


def _api_key() -> str:
    return os.environ.get("POLLINATIONS_API_KEY", "")


def build_image_url(prompt: str, width: int = 1024, height: int = 1024,
                     model: str = "flux", seed: int | None = None) -> str:
    """
    Returns a direct Pollinations image URL. Fetching it returns the image
    itself — no download/re-hosting step required. Requires
    POLLINATIONS_API_KEY to be set (see module docstring).
    """
    encoded_prompt = quote(prompt)
    url = f"{POLLINATIONS_BASE}/image/{encoded_prompt}?width={width}&height={height}&model={model}&nologo=true"
    if seed is not None:
        url += f"&seed={seed}"
    key = _api_key()
    if key:
        url += f"&key={key}"
    return url


def build_video_url(prompt: str, duration: int = 5, aspect_ratio: str = "9:16",
                     model: str = "wan-2.7", image_url: str | None = None) -> str:
    """
    Returns a direct Pollinations video URL (text-to-video, or
    image-to-video if image_url is given). Same no-download pattern as
    build_image_url. Requires POLLINATIONS_API_KEY to be set (see module
    docstring) — video in particular has never worked reliably without one.
    """
    encoded_prompt = quote(prompt)
    encoded_ratio = quote(aspect_ratio, safe="")
    url = (
        f"{POLLINATIONS_BASE}/image/{encoded_prompt}"
        f"?model={model}&duration={duration}&aspectRatio={encoded_ratio}"
    )
    if image_url:
        url += f"&image={quote(image_url, safe='')}"
    key = _api_key()
    if key:
        url += f"&key={key}"
    return url


def upload_media(file_path: str, filename: str) -> dict:
    """
    Uploads a local file (e.g. a composed full_video .mp4) to
    Pollinations' own media storage (media.pollinations.ai) and returns a
    public URL — used instead of Google Drive, since Drive's Shared Drive
    requirement (service accounts have no personal storage quota) isn't
    available on a personal Gmail account. Files live 30 days, refreshed
    whenever fetched (100MB limit). Requires POLLINATIONS_API_KEY.

    Returns {"link": "..."} on success or {"error": "..."} on failure.
    """
    key = _api_key()
    if not key:
        return {"error": "POLLINATIONS_API_KEY not configured — required for media upload"}
    try:
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{POLLINATIONS_MEDIA_BASE}/upload",
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (filename, f)},
                timeout=300,
            )
        try:
            data = r.json()
        except Exception:
            return {"error": f"non-JSON response (status {r.status_code}): {(r.text or '')[:300]!r}"}
        media_id = data.get("id") or data.get("url") or ""
        if not media_id:
            return {"error": f"upload response had no id/url: {data}"}
        # The API may already return a full URL, or just an id — handle both.
        link = media_id if str(media_id).startswith("http") else f"{POLLINATIONS_MEDIA_BASE}/{media_id}"
        return {"link": link}
    except Exception as e:
        return {"error": str(e)}


def download_music(prompt: str, model: str = "elevenmusic") -> str:
    """
    Generates and downloads a background music track from Pollinations'
    audio endpoint. Returns a local file path (mp3). Raises on failure —
    caller should catch and report honestly rather than silently skip
    the music.

    "elevenmusic" is the default instrumental-music model as of this
    writing — check /audio/models on gen.pollinations.ai if it's no
    longer available; the model list has changed before.
    """
    key = _api_key()
    if not key:
        raise RuntimeError("POLLINATIONS_API_KEY not configured — required for music generation")
    encoded_prompt = quote(prompt)
    url = f"{POLLINATIONS_BASE}/audio/{encoded_prompt}?model={model}&key={key}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    fd, path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    with open(path, "wb") as f:
        f.write(response.content)
    return path
