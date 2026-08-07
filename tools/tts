"""
Yam's voiceover helper — uses edge-tts (Microsoft Edge's public
text-to-speech service). Free, no API key, but requires the `edge-tts`
pip package to be installed (add "edge-tts" to requirements.txt).
"""

import asyncio
import os
import tempfile
import edge_tts

# Yam markets in English only — en-US-GuyNeural (male) is the default.
# en-US-JennyNeural is a solid female alternative.
DEFAULT_VOICE = "en-US-GuyNeural"


async def _generate(text: str, voice: str, output_path: str):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def generate_voiceover(text: str, voice: str = DEFAULT_VOICE) -> str:
    """
    Generates an MP3 voiceover from text. Returns a local file path.
    Raises on failure — the caller should catch and report honestly rather
    than claim a voiceover exists when it doesn't.
    """
    fd, output_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    asyncio.run(_generate(text, voice, output_path))
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
    fd, output_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    word_timings = asyncio.run(_generate_with_timing(text, voice, output_path))
    return output_path, word_timings
