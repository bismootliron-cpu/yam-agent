"""
Yam's voiceover helper — uses edge-tts (Microsoft Edge's public
text-to-speech service). Free, no API key, but requires the `edge-tts`
pip package to be installed (add "edge-tts" to requirements.txt).
"""

import asyncio
import os
import re
import tempfile
import edge_tts

# Yam markets in English by default — en-US-GuyNeural (male).
# en-US-JennyNeural is a solid female alternative.
DEFAULT_VOICE = "en-US-GuyNeural"

# Hebrew voices. Needed because an English voice handed Hebrew text does
# not fall back or transliterate — edge-tts returns no audio at all and
# raises "No audio was received. Please verify that your parameters are
# correct.", which reads like a bug in the call rather than a
# language/voice mismatch. This became live the moment the daily
# experiment picker started writing its briefs in Hebrew.
HEBREW_VOICE = "he-IL-AvriNeural"
HEBREW_VOICE_FEMALE = "he-IL-HilaNeural"

# Enough audio to be real speech. edge-tts can also hand back a tiny
# non-empty file when it fails partway.
MIN_AUDIO_BYTES = 2048

_HEBREW_RE = re.compile(r"[\u0590-\u05FF]")
_ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04FF]")


def detect_voice(text: str, voice: str = None) -> str:
    """
    Picks a voice that can actually speak the text.

    An explicitly passed voice always wins — the caller may know better,
    and silently overriding it would be its own surprise. Only the
    default is language-matched.
    """
    if voice and voice != DEFAULT_VOICE:
        return voice
    if _HEBREW_RE.search(text or ""):
        return HEBREW_VOICE
    if _ARABIC_RE.search(text or ""):
        return "ar-EG-ShakirNeural"
    if _CYRILLIC_RE.search(text or ""):
        return "ru-RU-DmitryNeural"
    return voice or DEFAULT_VOICE


def _validate_text(text: str) -> str:
    """Rejects input that cannot produce speech, with a reason. edge-tts
    answers empty or symbol-only input with the same generic 'no audio'
    message as a voice mismatch, so the cases are separated here."""
    cleaned = (text or "").strip()
    if not cleaned:
        raise RuntimeError("voiceover_text ריק — אין ממה לייצר קריינות")
    if not re.search(r"[^\W\d_]", cleaned, re.UNICODE):
        raise RuntimeError(
            "voiceover_text מכיל רק סימנים/מספרים ללא מילים — edge-tts "
            "לא יחזיר אודיו"
        )
    return cleaned


def _verify_audio(path: str, voice: str, text: str) -> None:
    """
    Confirms real audio landed on disk.

    The streaming path used to open a file, write whatever chunks
    arrived, and return normally even when none did — producing a
    zero-byte mp3 that ffmpeg then muxed into a silent video reported as
    ready. One path raised on failure while the other stayed quiet about
    the same failure; both check now.
    """
    size = os.path.getsize(path) if os.path.exists(path) else 0
    if size >= MIN_AUDIO_BYTES:
        return
    try:
        os.remove(path)
    except OSError:
        pass
    hint = ""
    if _HEBREW_RE.search(text) and not voice.startswith("he-"):
        hint = (
            f" — הטקסט בעברית אבל הקול הוא {voice}. "
            f"קול אנגלי לא מקריא עברית ומחזיר אפס אודיו."
        )
    raise RuntimeError(
        f"edge-tts החזיר {size} bytes של אודיו (פחות מהמינימום "
        f"{MIN_AUDIO_BYTES}) עם הקול {voice}{hint}"
    )


async def _generate(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_voiceover(text: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Generates an MP3 voiceover from text. Returns a local file path.
    Raises on failure — the caller should catch and report honestly rather
    than claim a voiceover exists when it doesn't.
    """
    text = _validate_text(text)
    voice = detect_voice(text, voice)
    fd, output_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    asyncio.run(_generate(text, voice, output_path))
    _verify_audio(output_path, voice, text)
    return output_path


async def _generate_with_timing(text: str, voice: str, output_path: str) -> list:
    """
    edge-tts reports real per-word timing (WordBoundary events) while
    streaming — this captures it instead of throwing it away, so captions
    can be genuinely synced to the spoken audio instead of just evenly
    dividing the total duration by line count.
    """
    communicate = edge_tts.Communicate(text, voice)
    word_timings = []
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # edge-tts reports offset/duration in 100-nanosecond units.
                word_timings.append({
                    "text": chunk["text"],
                    "offset": chunk["offset"] / 10_000_000,
                    "duration": chunk["duration"] / 10_000_000,
                })
    return word_timings


def generate_voiceover_with_timing(text: str, voice: str = DEFAULT_VOICE) -> tuple:
    """
    Same as generate_voiceover(), but also returns real per-word timing:
    a list of {"text", "offset", "duration"} dicts (seconds), in spoken
    order. Used to build captions that are actually synced to the
    narration instead of guessed from even division.

    Returns (audio_path, word_timings).
    """
    text = _validate_text(text)
    voice = detect_voice(text, voice)
    fd, output_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    word_timings = asyncio.run(_generate_with_timing(text, voice, output_path))
    _verify_audio(output_path, voice, text)
    if not word_timings:
        # Audio exists but no WordBoundary events arrived. Captions would
        # silently fall back to even division, so say so instead of
        # letting a worse result pass as the intended one.
        print(f"⚠️ edge-tts: לא התקבלו תזמוני מילים עבור הקול {voice} — "
              f"הכתוביות יחולקו בחלוקה שווה במקום סנכרון אמיתי")
    return output_path, word_timings
