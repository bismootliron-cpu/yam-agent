"""
ים (Yam) — standalone creative agent. A separate service from Leo
(Liron-agent) and Hermes on purpose: Leo's BLOCKED_REPOS deliberately
forbids Leo from ever touching its own repo or Hermes's, as a safety
boundary against a runaway self-editing agent. Yam living inside Leo's
repo would have made "Yam upgrades itself" mean "Leo edits its own
code" — exactly what that boundary exists to prevent. As its own
repo/service, Yam can eventually gain a safe, explicitly-gated
self-update path without touching that boundary at all.

Talks to Hermes over the same simple HTTP task-queue contract Hermes
already uses for Leo: POST /task -> {task_id}, GET /task/<id>/status ->
{status, result|error|progress}. See main.py for the server side.
"""

import os
import re
import struct
import requests
from anthropic import Anthropic
from tools import creative
from tools import tts
from tools import video_compose

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

CREATIVE_TASK_MARKER = "__CREATIVE_TASK__"  # kept in sync with Hermes's main.py

FULL_VIDEO_OVERLAY_SECTION = "---OVERLAY LINES---"

# "ים" (Yam) — the persona/instincts used only for the caption-writing
# step in _handle_simple_media. Kept short and general on purpose: it's
# meant to travel across every venture Liron markets, not just one.
YAM_SYSTEM_PROMPT = (
    "You are 'Yam' — a creative agent skilled in writing marketing content "
    "for social media, across a range of tech and trading ventures. You "
    "think like an experienced content creator/influencer, not just a "
    "technical media producer. Principles you always apply: a sharp hook "
    "in the first 3 seconds/words of the copy; professional yet "
    "approachable tone; a clear call to action at the end; focused "
    "hashtags for the relevant niche (not too generic); matching length "
    "and tone to the target platform when specified. Concise, no preamble. "
    "ALWAYS write the caption in English, regardless of the language the "
    "request came in."
)

# ─── CAPABILITIES (self-description) ─────────────────────────────────────────
# The single source of truth for what Yam can do — used two ways:
#   1. GET /capabilities on main.py serves this directly, so Hermes (or a
#      human) can ask Yam "what do you support?" and get the REAL, current
#      answer instead of relying on a hand-maintained description in
#      Hermes's own system_prompt.txt that can silently drift out of sync.
#   2. creative_task() below cross-checks incoming fields against this and
#      warns about anything unrecognized for the given type, INSTEAD OF
#      silently ignoring it — this is the direct fix for the real bug
#      where "outro_library_tag" (resolved to outro_image_url) was sent
#      on an edit_video task, which doesn't know that field, and Yam just
#      dropped it with zero indication anything was wrong.
CAPABILITIES = {
    "image": {
        "description": "Generate a marketing image via Pollinations (Flux), or use a real image from the library as-is.",
        "fields": {"prompt", "library_tag", "format", "caption", "caption_text", "image_description"},
    },
    "video": {
        "description": "Generate a short video via Pollinations, or deliver a real video from the library as-is.",
        "fields": {"prompt", "library_tag", "aspect_ratio", "duration", "caption", "caption_text", "image_description"},
    },
    "full_video": {
        "description": "Still image (generated or real) + edge-tts voiceover + burned-in text overlays. Image source ONLY — never a video source.",
        "fields": {"image_prompt", "library_tag", "voiceover_text", "duration", "effects", "music_library_tag", "image_description", "unresolved_music_reason"},
        "optional_sections": ["---OVERLAY LINES---"],
    },
    "edit_video": {
        "description": "Trim / speed-ramp / caption / re-voice ONE real video clip.",
        "fields": {"library_tag", "trim_start", "duration", "speed_ramp_at", "speed_multiplier", "voiceover_text", "effects", "music_library_tag", "image_description", "unresolved_music_reason"},
        "optional_sections": ["---OVERLAY LINES---"],
    },
    "multi_scene_video": {
        "description": "Combine several real IMAGES as timed scenes (zoom + text each), one overall voiceover. Image tags ONLY.",
        "fields": {"voiceover_text", "effects", "disclaimer", "music_library_tag",
                   "library_tag", "duration", "zoom", "text", "text_delay", "image_description", "transition", "fit", "unresolved_music_reason"},
        "required_sections": ["---SCENE---"],
    },
    "multi_clip_video": {
        "description": "Combine several real VIDEO clips (each trimmed) into one, one continuous voiceover, optional image outro. Video tags ONLY per clip.",
        "fields": {"voiceover_text", "effects", "disclaimer", "music_library_tag",
                   "outro_library_tag", "outro_duration", "outro_text",
                   "library_tag", "trim_start", "duration", "text", "text_delay", "image_description", "unresolved_music_reason", "unresolved_outro_reason"},
        "required_sections": ["---CLIP---"],
    },
    "research_content_ideas": {
        "description": (
            "Research current social-media content trends via live web search "
            "and propose 2-3 concrete, ready-to-run content ideas — each one "
            "returned as a complete [AGENT TASK - CREATIVE] block Liron can "
            "paste straight back. Produces NO media itself; this is the "
            "'what should we post' step that comes BEFORE generation."
        ),
        "fields": {"topic", "platform", "library_context", "count"},
    },
}

# When a *_library_tag field is allowed for a type, its RESOLVED form
# (produced by Hermes before the task ever reaches here) must be allowed
# too, or every normal successful resolution would itself look "unknown".
_RESOLUTION_MAP = {
    "library_tag": {"image_url", "video_url", "unresolved_library_tag"},
    "music_library_tag": {"music_url", "unresolved_music_library_tag"},
    "outro_library_tag": {"outro_image_url", "unresolved_outro_library_tag"},
}

# Accepted on EVERY type, not per-type: which venture the asset is for.
# Without it, a generated caption is written from the playbook alone —
# which is trading-heavy — so a Seranova travel image came back tagged
# #tradingmindset #forextrader. It's universal because no media type is
# venture-agnostic.
_UNIVERSAL_FIELDS = {"project"}

# The set of names that legitimately START a new field line. Anything
# else containing a ":" is prose belonging to the PREVIOUS field — see
# the multi-line handling in creative_task().
_ALL_FIELD_NAMES = (
    set().union(*(spec["fields"] for spec in CAPABILITIES.values()))
    | {"type"}
    | _UNIVERSAL_FIELDS
    | set().union(*_RESOLUTION_MAP.values())
)


def get_capabilities() -> dict:
    """Returns the CAPABILITIES structure as-is — this IS the live answer
    to "what can Yam do", not a description that can go stale."""
    return CAPABILITIES


def _unknown_fields_warning(media_type: str, fields: dict) -> str:
    """
    Returns a "" if every field Hermes sent is recognized for this type,
    or a visible warning prefix listing whatever wasn't — e.g. a resolved
    outro_image_url sent alongside type: edit_video, which doesn't know
    that field at all. Always prepended to the result (success or
    failure) rather than silently dropped, so "Yam did nothing with a
    field I sent" is never a silent, undiagnosable outcome again.
    """
    spec = CAPABILITIES.get(media_type)
    if not spec:
        return ""  # unknown type entirely — the dispatcher below handles that separately
    allowed = set(spec["fields"]) | {"type"} | _UNIVERSAL_FIELDS
    for base_field, resolved_fields in _RESOLUTION_MAP.items():
        if base_field in spec["fields"]:
            allowed |= resolved_fields
    unknown = sorted(k for k in fields.keys() if k not in allowed)
    if not unknown:
        return ""
    return (
        f"⚠️ Note: field(s) {', '.join(unknown)} aren't recognized by "
        f"type '{media_type}' — they were ignored, not applied. Check "
        f"GET /capabilities for what this type actually accepts.\n\n"
    )


def load_playbook() -> str:
    """
    Reads creative_playbook.md — the accumulated creative principles that
    the feedback loop refines over time. Read fresh on each call rather
    than cached at import, so an approved playbook edit takes effect on
    the next generation without needing a restart.

    Returns "" if the file is missing, and callers fall back to the base
    system prompt alone — a missing playbook must degrade quality, never
    break generation.
    """
    try:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "creative_playbook.md")
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"⚠️ load_playbook error: {e}")
        return ""


def creative_system_prompt() -> str:
    """Base persona plus whatever Yam has learned so far."""
    playbook = load_playbook()
    if not playbook:
        return YAM_SYSTEM_PROMPT
    return (
        f"{YAM_SYSTEM_PROMPT}\n\n"
        f"--- YOUR CREATIVE PLAYBOOK (learned from real feedback on your own "
        f"past output — follow it) ---\n{playbook}"
    )


def _parse_effects(fields: dict) -> set:
    raw = fields.get("effects", "").strip().lower()
    if not raw:
        return set()
    return {e.strip() for e in raw.split(",") if e.strip()}


def _apply_music_if_requested(fields: dict, effects: set, final_path: str, report) -> tuple:
    """
    Adds background music to final_path if requested, PREFERRING a real
    library track (fields["music_url"], resolved by Hermes from a
    music_library_tag) over Pollinations' paid "elevenmusic" model —
    that model turned out to require paid Pollen credits (402 Payment
    Required), and no confirmed free-for-commercial-use music API was
    found, so a manually-curated library track is the default, safe
    path now. The paid model is only used as an explicit opt-in when
    "music" is in effects but no music_url was resolved.

    Returns (possibly-updated final_path, music_note) — music_note is ""
    on success, or a user-visible warning string if music was requested
    but failed.
    """
    music_url = fields.get("music_url", "").strip()
    unresolved_music_tag = fields.get("unresolved_music_library_tag", "").strip()

    if unresolved_music_tag:
        return final_path, (
            f"\n⚠️ תג המוזיקה '{unresolved_music_tag}' לא נמצא בספרייה "
            f"(או שהוא לא audio) — הוידאו מוכן בלי מוזיקה."
        )

    if music_url:
        report("מוריד ומערבב מוזיקה מהספרייה...")
        try:
            music_path = video_compose.download_library_music(music_url)
            return video_compose.mix_in_music(final_path, music_path), ""
        except Exception as e:
            return final_path, f"\n⚠️ מוזיקת הרקע מהספרייה נכשלה ({e}) — הוידאו מוכן בלעדיה."

    if "music" in effects:
        report("מייצר מוזיקת רקע דרך Pollinations (בתשלום — נסה music_library_tag לחלופה חינמית)...")
        try:
            music_path = creative.download_music("upbeat modern corporate background music")
            return video_compose.mix_in_music(final_path, music_path), ""
        except Exception as e:
            return final_path, f"\n⚠️ מוזיקת הרקע נכשלה ({e}) — הוידאו מוכן בלעדיה."

    return final_path, ""


# Timestamps that a block author appended to an overlay line, thinking
# they set the timing. They do not: timing comes from word_timings or an
# even split of the duration, so the text was simply burned into the
# frame — a video shipped reading "רעיון אחד | 00:02" on screen.
# Stripped here rather than only in the prompt that produced it, because
# a block can arrive from Hermes, from an ideas run, or pasted by hand,
# and the prompt fix only covers the first.
_OVERLAY_TIMESTAMP_RE = re.compile(
    r"\s*[|\-–—:]?\s*\(?\b\d{1,2}:\d{2}(?::\d{2})?\b\)?\s*$"
)
_OVERLAY_LEADING_INDEX_RE = re.compile(r"^\s*(?:\d{1,2}[.)]|[-•*])\s+")


def clean_overlay_line(line: str) -> str:
    """Removes trailing timestamps and leading list markers from one
    on-screen line. Returns '' if nothing is left worth drawing."""
    cleaned = _OVERLAY_TIMESTAMP_RE.sub("", line.strip())
    cleaned = _OVERLAY_LEADING_INDEX_RE.sub("", cleaned)
    return cleaned.strip()


def parse_overlay_lines(body: str, section_marker: str) -> list:
    """Shared overlay parsing, so both full_video and edit_video get the
    same cleaning — the identical block was duplicated in each."""
    if section_marker not in body:
        return []
    raw_lines = body.split(section_marker, 1)[1].strip().split("\n")
    out = []
    for raw in raw_lines:
        cleaned = clean_overlay_line(raw)
        if cleaned:
            out.append(cleaned)
    return out


def _handle_full_video(fields: dict, body: str, report) -> str:
    """
    Builds a complete marketing video: a Pollinations base image, an
    edge-tts voiceover, and ffmpeg-burned text overlays — then uploads
    the final .mp4 to media.pollinations.ai and returns that link.

    Expected fields (parsed from the __CREATIVE_TASK__ body):
        image_prompt: <visual description for the base image>
        voiceover_text: <full narration script>
        duration: <seconds, optional, default 8>
        effects: <comma list, optional: zoom, pulse_text, music>
        ---OVERLAY LINES---
        <optional — one on-screen text line per line, evenly spaced. If
        OMITTED, captions are auto-generated from the real narration
        timing instead (short synced phrases, not evenly divided).>
    """
    image_prompt = fields.get("image_prompt", "").strip()
    image_url = fields.get("image_url", "").strip()
    voiceover_text = fields.get("voiceover_text", "").strip()
    if fields.get("video_url", "").strip():
        return (
            "❌ full_video: the library_tag resolved to a VIDEO, not an image — "
            "full_video only builds from a still image. Use type: edit_video "
            "instead for a real video source (it supports voiceover_text and "
            "effects the same way)."
        )
    if not (image_prompt or image_url) or not voiceover_text:
        unresolved = fields.get("unresolved_library_tag", "")
        if unresolved:
            return f"❌ full_video: library_tag '{unresolved}' isn't in the library — check with 'lib list' for the precise stored name"
        return "❌ full_video: missing 'image_prompt:' (or resolved image_url) or 'voiceover_text:'"
    try:
        duration = int(fields.get("duration", "8"))
    except ValueError:
        duration = 8

    effects = _parse_effects(fields)

    overlay_lines = parse_overlay_lines(body, FULL_VIDEO_OVERLAY_SECTION)

    if image_url:
        report("משתמש בתמונת בסיס אמיתית מהספרייה (לא נוצרת תמונה חדשה)...")
        base_url = image_url
    else:
        report("מייצר תמונת בסיס דרך Pollinations...")
        base_url = creative.build_image_url(image_prompt, width=1080, height=1920)

    report("מייצר קריינות עם edge-tts (עם תזמון מילים לכתוביות מסונכרנות)...")
    try:
        voiceover_path, word_timings = tts.generate_voiceover_with_timing(voiceover_text)
    except Exception as e:
        return f"❌ full_video: יצירת הקריינות נכשלה ({e})"

    report("מרכיב את הוידאו הסופי (טקסטים + קריינות) עם ffmpeg...")
    try:
        final_path = video_compose.compose_video_with_overlay(
            base_url, base_is_video=False, voiceover_path=voiceover_path,
            overlay_lines=overlay_lines, duration=duration,
            word_timings=word_timings,
            pulse_text="pulse_text" in effects, zoom="zoom" in effects,
        )
    except Exception as e:
        return f"❌ full_video: הרכבת הוידאו נכשלה ({e})"

    music_note = ""
    if "music" in effects or fields.get("music_url") or fields.get("unresolved_music_library_tag"):
        final_path, music_note = _apply_music_if_requested(fields, effects, final_path, report)

    # Probe the rendered file BEFORE uploading. ffmpeg exits 0 on a
    # truncated render, which is how a clip that stopped mid-way was once
    # delivered as "✅ מוכן" — the failure that started all of this.
    report("בודק את הוידאו שנוצר (משך, ערוצים, שלמות)...")
    video_problems = video_compose.verify_video(
        final_path, expected_duration=duration, expect_audio=True
    )
    caption_problems = check_caption(fields.get("caption_text", ""))

    report("מעלה את הוידאו הסופי ל-media.pollinations.ai...")
    upload_result = creative.upload_media(final_path, os.path.basename(final_path))
    if "error" in upload_result:
        return (
            f"⚠️ הוידאו נוצר בהצלחה מקומית, אבל ההעלאה נכשלה: "
            f"{upload_result['error']}"
        )
    checks = _checks_section(video_problems, caption_problems)
    status = "⚠️" if video_problems else "✅"
    return (
        f"{status} 🎬 וידאו מלא (טקסטים על המסך + קריינות) מוכן:\n"
        f"{upload_result['link']}{music_note}{checks}"
    )


def _handle_edit_video(fields: dict, body: str, report) -> str:
    """
    Edits an EXISTING real video (from the library, resolved to
    video_url) — trims it, optionally burns in text overlays, and
    optionally replaces its audio with a new edge-tts voiceover. If no
    voiceover is given, the video's own original audio is kept untouched.

    Unlike full_video (which always builds FROM a still image), this
    starts from real, already-shot footage — for editing an uploaded
    clip, not generating a new one.

    Expected fields (parsed from the __CREATIVE_TASK__ body):
        video_url: <resolved automatically from library_tag by Hermes>
        trim_start: <seconds into the source video to start, optional, default 0>
        duration: <seconds to keep from trim_start, optional — omit to keep the rest>
        speed_ramp_at: <seconds INTO THE TRIMMED CLIP where speed increases, optional>
        speed_multiplier: <how much faster after speed_ramp_at, optional, default 2.0>
        voiceover_text: <optional — if given, REPLACES the original audio>
        effects: <comma list, optional: pulse_text, music (no zoom — real footage already moves)>
        ---OVERLAY LINES---
        <optional, one on-screen text line per line>
    """
    video_url = fields.get("video_url", "").strip()
    if not video_url:
        unresolved = fields.get("unresolved_library_tag", "")
        if unresolved:
            return f"❌ edit_video: library_tag '{unresolved}' isn't in the library — check with 'lib list' for the precise stored name"
        return "❌ edit_video: missing 'video_url' — this type requires a library_tag pointing to a real video"

    try:
        trim_start = float(fields.get("trim_start", "0") or "0")
    except ValueError:
        trim_start = 0
    duration_raw = fields.get("duration", "").strip()
    try:
        duration = float(duration_raw) if duration_raw else 0
    except ValueError:
        duration = 0

    speed_ramp_raw = fields.get("speed_ramp_at", "").strip()
    speed_ramp_at = None
    if speed_ramp_raw:
        try:
            speed_ramp_at = float(speed_ramp_raw)
        except ValueError:
            speed_ramp_at = None
    try:
        speed_multiplier = float(fields.get("speed_multiplier", "2.0") or "2.0")
    except ValueError:
        speed_multiplier = 2.0

    effects = _parse_effects(fields)

    overlay_lines = parse_overlay_lines(body, FULL_VIDEO_OVERLAY_SECTION)

    voiceover_text = fields.get("voiceover_text", "").strip()
    voiceover_path = None
    if voiceover_text:
        report("מייצר קריינות חדשה עם edge-tts (תחליף לשמע המקורי)...")
        try:
            voiceover_path = tts.generate_voiceover(voiceover_text)
        except Exception as e:
            return f"❌ edit_video: יצירת הקריינות נכשלה ({e})"

    report("עורך את הוידאו הקיים (חיתוך/מהירות/טקסט) עם ffmpeg...")
    try:
        final_path = video_compose.edit_video_with_speed_ramp(
            video_url, trim_start=trim_start, duration=duration,
            speed_ramp_at=speed_ramp_at, speed_multiplier=speed_multiplier,
            overlay_lines=overlay_lines, voiceover_path=voiceover_path,
            pulse_text="pulse_text" in effects,
        )
    except Exception as e:
        return f"❌ edit_video: העריכה נכשלה ({e})"

    music_note = ""
    if "music" in effects or fields.get("music_url") or fields.get("unresolved_music_library_tag"):
        final_path, music_note = _apply_music_if_requested(fields, effects, final_path, report)

    report("מעלה את הוידאו הערוך ל-media.pollinations.ai...")
    upload_result = creative.upload_media(final_path, os.path.basename(final_path))
    if "error" in upload_result:
        return f"⚠️ העריכה הצליחה מקומית, אבל ההעלאה נכשלה: {upload_result['error']}"
    return f"✅ 🎬 וידאו ערוך מוכן:\n{upload_result['link']}{music_note}"


MULTI_SCENE_MARKER = "---SCENE---"


def _handle_multi_scene_video(fields: dict, body: str, report) -> str:
    """
    Builds a multi-scene marketing video from several real library images,
    each shown as its own timed scene with its own zoom direction and
    optional text, concatenated together with one overall voiceover and
    an optional disclaimer burned into the final seconds.

    Expected task format (built by Hermes from an [AGENT TASK - CREATIVE]
    block):
        __CREATIVE_TASK__
        type: multi_scene_video
        voiceover_text: <optional single narration spanning the whole video — English only>
        effects: <comma list, optional: music>
        disclaimer: <optional short text burned into the last few seconds>
        ---SCENE---
        library_tag: <tag — resolved to image_url by Hermes>
        duration: <seconds this scene is shown>
        zoom: in | out | (omit for no zoom)
        text: <optional single line shown during this scene>
        text_delay: <optional seconds into the scene before the text appears>
        ---SCENE---
        library_tag: <next scene...>
        ...
    """
    if MULTI_SCENE_MARKER not in body:
        return "❌ multi_scene_video: no ---SCENE--- sections found — this type needs at least one scene"

    scene_chunks = body.split(MULTI_SCENE_MARKER)[1:]
    scenes = []
    for chunk in scene_chunks:
        scene_fields = {}
        for line in chunk.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            scene_fields[k.strip().lower()] = v.strip()

        image_url = scene_fields.get("image_url", "").strip()
        if not image_url:
            video_url = scene_fields.get("video_url", "").strip()
            unresolved_music = scene_fields.get("unresolved_music_library_tag", "").strip()
            tag = scene_fields.get("unresolved_library_tag") or scene_fields.get("library_tag", "")
            if video_url:
                return (
                    "❌ multi_scene_video: a scene resolved to a VIDEO tag — "
                    "this type only supports IMAGE tags for scenes (it builds "
                    "each scene from a still image). To use real video clips, "
                    "use edit_video instead (one clip at a time — there's no "
                    "way yet to stitch several real video clips into one "
                    "video; only multiple images can be combined this way)."
                )
            return (
                f"❌ multi_scene_video: scene with library_tag '{tag}' has no "
                f"resolved image_url — this exact tag isn't in the library "
                f"(check with 'lib list' for the precise stored name)"
            )
        try:
            duration = float(scene_fields.get("duration", "3") or "3")
        except ValueError:
            duration = 3
        try:
            text_delay = float(scene_fields.get("text_delay", "0") or "0")
        except ValueError:
            text_delay = 0
        scenes.append({
            "image_url": image_url,
            "duration": duration,
            # A scene's own fit wins; otherwise the block-level fit acts as
            # the default for every scene. Without this, writing `fit:
            # contain` once at the top did nothing at all — silently.
            "fit": (scene_fields.get("fit", "").strip().lower()
                    or fields.get("fit", "").strip().lower() or "cover"),
            "zoom": scene_fields.get("zoom", "").strip().lower(),
            "text": scene_fields.get("text", "").strip(),
            "text_delay": text_delay,
        })

    if not scenes:
        return "❌ multi_scene_video: no valid scenes parsed"

    voiceover_text = fields.get("voiceover_text", "").strip()
    voiceover_path = None
    if voiceover_text:
        report("מייצר קריינות כוללת עם edge-tts...")
        try:
            voiceover_path = tts.generate_voiceover(voiceover_text)
        except Exception as e:
            return f"❌ multi_scene_video: יצירת הקריינות נכשלה ({e})"

    disclaimer = fields.get("disclaimer", "").strip()
    effects = _parse_effects(fields)
    # Crossfade length between scenes. Hard cuts (0) stay the default so
    # existing behaviour is unchanged unless smoothness is asked for.
    try:
        transition = float(fields.get("transition", "0") or "0")
    except ValueError:
        transition = 0.5  # a non-numeric value clearly means "yes, smooth"
    if fields.get("transition", "").strip().lower() in ("yes", "true", "smooth", "fade", "כן"):
        transition = 0.5

    report(f"מרכיב וידאו רב-סצנתי ({len(scenes)} סצנות) עם ffmpeg...")
    try:
        final_path = video_compose.assemble_multi_scene_video(
            scenes, voiceover_path=voiceover_path, disclaimer=disclaimer,
            transition=transition,
        )
    except Exception as e:
        return f"❌ multi_scene_video: ההרכבה נכשלה ({e})"

    music_note = ""
    if "music" in effects or fields.get("music_url") or fields.get("unresolved_music_library_tag"):
        final_path, music_note = _apply_music_if_requested(fields, effects, final_path, report)

    report("מעלה את הוידאו הסופי ל-media.pollinations.ai...")
    upload_result = creative.upload_media(final_path, os.path.basename(final_path))
    if "error" in upload_result:
        return f"⚠️ הוידאו נוצר בהצלחה מקומית, אבל ההעלאה נכשלה: {upload_result['error']}"
    return f"✅ 🎬 וידאו רב-סצנתי ({len(scenes)} סצנות) מוכן:\n{upload_result['link']}{music_note}"


MULTI_CLIP_MARKER = "---CLIP---"


def _handle_multi_clip_video(fields: dict, body: str, report) -> str:
    """
    Combines several REAL VIDEO clips into one video with ONE continuous
    voiceover playing over the whole thing — for a script written as one
    narration spanning multiple real clips (e.g. two signal screen
    recordings back to back). Neither edit_video (one clip only) nor
    multi_scene_video (images only) can do this.

    Expected task format (built by Hermes from an [AGENT TASK - CREATIVE]
    block):
        __CREATIVE_TASK__
        type: multi_clip_video
        voiceover_text: <optional single narration spanning the WHOLE video — English only>
        effects: <comma list, optional: music>
        disclaimer: <optional short text burned into the final few seconds>
        outro_image_url: <resolved automatically from outro_library_tag by Hermes, optional>
        outro_duration: <optional seconds, default 5>
        outro_text: <optional text shown on the outro>
        ---CLIP---
        library_tag: <tag for this clip — resolved to video_url by Hermes; MUST be a [video] tag>
        trim_start: <optional seconds into this source clip, default 0>
        duration: <optional seconds to keep — omit to keep the rest of that clip>
        text: <optional single line shown during this clip>
        text_delay: <optional seconds into the clip before the text appears>
        ---CLIP---
        library_tag: <second clip...>
        ...
    """
    if MULTI_CLIP_MARKER not in body:
        return "❌ multi_clip_video: no ---CLIP--- sections found — this type needs at least one clip"

    clip_chunks = body.split(MULTI_CLIP_MARKER)[1:]
    clips = []
    for chunk in clip_chunks:
        clip_fields = {}
        for line in chunk.strip().split("\n"):
            line = line.strip()
            if not line or ":" not in line:
                continue
            k, _, v = line.partition(":")
            clip_fields[k.strip().lower()] = v.strip()

        video_url = clip_fields.get("video_url", "").strip()
        if not video_url:
            tag = clip_fields.get("unresolved_library_tag") or clip_fields.get("library_tag", "")
            image_url_mistake = clip_fields.get("image_url", "").strip()
            if image_url_mistake:
                return (
                    f"❌ multi_clip_video: a clip resolved to an IMAGE, not a video — "
                    f"this type only combines real video clips. Use multi_scene_video "
                    f"instead for image-only scenes, or check the tag is actually [video]."
                )
            return (
                f"❌ multi_clip_video: clip with library_tag '{tag}' has no resolved "
                f"video_url — check with 'lib list' for the precise stored name"
            )
        try:
            trim_start = float(clip_fields.get("trim_start", "0") or "0")
        except ValueError:
            trim_start = 0
        duration_raw = clip_fields.get("duration", "").strip()
        try:
            duration = float(duration_raw) if duration_raw else 0
        except ValueError:
            duration = 0
        try:
            text_delay = float(clip_fields.get("text_delay", "0") or "0")
        except ValueError:
            text_delay = 0
        clips.append({
            "video_url": video_url, "trim_start": trim_start, "duration": duration,
            "text": clip_fields.get("text", "").strip(), "text_delay": text_delay,
        })

    if not clips:
        return "❌ multi_clip_video: no valid clips parsed"

    voiceover_text = fields.get("voiceover_text", "").strip()
    voiceover_path = None
    if voiceover_text:
        report("מייצר קריינות כוללת עם edge-tts...")
        try:
            voiceover_path = tts.generate_voiceover(voiceover_text)
        except Exception as e:
            return f"❌ multi_clip_video: יצירת הקריינות נכשלה ({e})"

    outro_image_url = fields.get("outro_image_url", "").strip()
    try:
        outro_duration = float(fields.get("outro_duration", "5") or "5")
    except ValueError:
        outro_duration = 5
    outro_text = fields.get("outro_text", "").strip()
    disclaimer = fields.get("disclaimer", "").strip()
    effects = _parse_effects(fields)

    report(f"מרכיב וידאו מרובה-קליפים ({len(clips)} קליפים) עם ffmpeg...")
    try:
        final_path = video_compose.assemble_multi_clip_video(
            clips, voiceover_path=voiceover_path, outro_image_url=outro_image_url,
            outro_duration=outro_duration, outro_text=outro_text, disclaimer=disclaimer,
        )
    except Exception as e:
        return f"❌ multi_clip_video: ההרכבה נכשלה ({e})"

    music_note = ""
    if "music" in effects or fields.get("music_url") or fields.get("unresolved_music_library_tag"):
        final_path, music_note = _apply_music_if_requested(fields, effects, final_path, report)

    report("מעלה את הוידאו הסופי ל-media.pollinations.ai...")
    upload_result = creative.upload_media(final_path, os.path.basename(final_path))
    if "error" in upload_result:
        return f"⚠️ הוידאו נוצר בהצלחה מקומית, אבל ההעלאה נכשלה: {upload_result['error']}"
    return f"✅ 🎬 וידאו מרובה-קליפים ({len(clips)} קליפים) מוכן:\n{upload_result['link']}{music_note}"


def _handle_research_content_ideas(fields: dict, body: str, report) -> str:
    """
    Yam's first NON-MEDIA skill: instead of executing a content request
    Liron already thought of, this researches what's actually working on
    social media right now (live web search) and proposes concrete ideas
    back — each as a ready-to-paste [AGENT TASK - CREATIVE] block.

    This is deliberately the step BEFORE generation: it produces zero
    media and costs no render time, so Liron can look at several
    directions cheaply and only spend an ffmpeg render on the one he
    actually wants.

    Expected fields:
        topic: <what the content is about — defaults to the trading indicator>
        platform: <tiktok | instagram | x | general — shapes format/length advice>
        library_context: <optional, passed by Hermes: the available tag list,
            so proposals reference REAL tags instead of inventing them>
        count: <how many ideas, optional, default 3, capped at 5>
    """
    topic = fields.get("topic", "").strip() or (
        "Singularity Indicator — a TradingView trading indicator that marks "
        "entry/exit signals, sold as a paid private signal group"
    )
    platform = fields.get("platform", "").strip().lower() or "general"
    library_context = fields.get("library_context", "").strip()
    try:
        count = min(int(fields.get("count", "3") or "3"), 5)
    except ValueError:
        count = 3

    report("חוקר טרנדים עדכניים ברשתות חברתיות (web search)...")

    library_note = (
        f"\n\nAvailable media library tags (use ONLY these exact tags in any "
        f"library_tag/outro_library_tag/music_library_tag line — never invent "
        f"a tag):\n{library_context}"
        if library_context else
        "\n\nNOTE: no library tag list was provided, so prefer ideas that "
        "generate fresh media (image_prompt) over ideas needing a specific "
        "existing asset, and say plainly which ideas would need an asset "
        "Liron has to confirm exists."
    )

    capability_reference = "\n".join(
        f"- {name}: {spec['description']}\n"
        f"  fields: {', '.join(sorted(spec['fields']))}"
        + (
            f"\n  REQUIRED sections: {', '.join(spec['required_sections'])}"
            f" (repeat the marker once per item; each item's fields go under its own marker)"
            if spec.get("required_sections") else ""
        )
        + (
            f"\n  optional sections: {', '.join(spec['optional_sections'])}"
            if spec.get("optional_sections") else ""
        )
        for name, spec in CAPABILITIES.items()
        if name != "research_content_ideas"
    )

    syntax_rules = (
        "SYNTAX RULES — blocks that break these fail at runtime, so a "
        "beautiful idea in the wrong shape is worthless:\n"
        "- Every field is `key: value` on ONE line. There are NO lists and "
        "NO JSON anywhere. `library_tag: [\"a\", \"b\"]` is invalid — a "
        "library_tag is exactly ONE tag.\n"
        "- To use several images, you MUST use multi_scene_video with a "
        "repeated ---SCENE--- marker, one per image, each with its own "
        "library_tag/duration/zoom/text underneath. Same shape for "
        "multi_clip_video with ---CLIP---.\n"
        "- effects: accepts ONLY these exact words: zoom, pulse_text, "
        "music. There is no 'zoom_in', 'zoom_out' or 'fade' in effects. "
        "(Per-scene direction inside multi_scene_video does use "
        "`zoom: in` or `zoom: out`.)\n"
        "- Do NOT emit a field with an empty value. Omit the line instead.\n"
        "- `caption:` takes only yes or no. If you have written an actual "
        "caption, put it in `caption_text:` — that is used verbatim. A "
        "caption written into `caption:` is thrown away.\n"
        "- ---OVERLAY LINES--- contains PLAIN TEXT, one on-screen line per "
        "line, spaced evenly across the video. There is NO per-line timing "
        "syntax: `LINE 1:`, `TIME:`, `DURATION:`, `POSITION:` and `STYLE:` "
        "do NOT exist and would be burned into the video as literal text. "
        "If you want captions synced to the narration, just OMIT the "
        "overlay section entirely — timing is then derived from the real "
        "voiceover, which is usually better anyway.\n"
        "- `music_library_tag:` must name a tag whose kind is [audio]. An "
        "[image] tag like a logo is NOT music and will be rejected. If the "
        "library has no [audio] tag, omit music entirely rather than "
        "pointing at something that isn't audio.\n"
        "- For multi_scene_video, a top-level `duration:` means nothing — "
        "each scene carries its own duration.\n"
    )

    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2500,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            system=(
                f"{creative_system_prompt()}\n\n"
                "For this task you are NOT producing media. You are acting as "
                "a social-media strategist: research what is genuinely working "
                "RIGHT NOW (search the web — current hooks, formats, trends, "
                "posting styles in the trading/fintech/finance-creator space), "
                "then translate that into concrete content Liron can actually "
                "run today.\n\n"
                "You may ONLY propose ideas that map to a capability that "
                "actually exists. These are the real capabilities:\n"
                f"{capability_reference}\n\n"
                f"{syntax_rules}\n"
                "Output format — for EACH idea, exactly this shape:\n"
                "### <short idea name>\n"
                "<2-3 sentences in Hebrew: what the idea is, which trend or "
                "insight it's based on, and why it should work for this "
                "audience>\n"
                "```\n"
                "[AGENT TASK - CREATIVE]\n"
                "type: <one of the real types above>\n"
                "<the complete, filled-in block — no placeholders, no "
                "'FILL IN', ready to paste and run as-is>\n"
                "[/AGENT TASK]\n"
                "```\n\n"
                "Hard rules: all voiceover_text and on-screen text must be "
                "100% English (never mixed with Hebrew). Never ask the image "
                "model to render specific words/names inside an image — it is "
                "unreliable at text; use real text overlays instead. Your "
                "explanations to Liron are in Hebrew; the content itself is in "
                "English. Be specific and opinionated — no generic 'post "
                "engaging content' advice."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Topic: {topic}\n"
                    f"Target platform: {platform}\n"
                    f"Number of ideas: {count}"
                    f"{library_note}"
                )
            }]
        )
    except Exception as e:
        return f"❌ research_content_ideas: המחקר נכשל ({e})"

    text_blocks = [b.text for b in response.content if b.type == "text"]
    ideas = "\n".join(text_blocks).strip()
    if not ideas:
        return "⚠️ research_content_ideas: לא הוחזרו רעיונות — נסה שוב או צמצם את הבקשה."
    return f"💡 *רעיונות תוכן ({platform}):*\n\n{ideas}"


def creative_task(task: str, report) -> str:
    """
    Yam's execution handler — generates marketing media via Pollinations
    (image/video), OR, for type "full_video", builds a complete composed
    clip (base image + burned-in text overlays + edge-tts voiceover) via
    ffmpeg and delivers it via Pollinations' media storage.

    Rather than downloading and re-hosting the file for image/video, this
    returns a direct Pollinations URL: fetching that URL itself returns
    the generated image/video, so no binary handling or extra hosting
    step is needed in this repo for those two types. "full_video" is the
    exception — real composition needs local file processing, so that
    path downloads, processes, and uploads instead.

    Per the user's decision, Yam has autonomy to choose/use free tools for
    the creative process without an approval gate — this handler reflects
    that: no [APPROVED] check here, unlike publish_plan. The tools used
    here (Pollinations, edge-tts) are all free/no-key; if a future tool
    needs payment or a new API key/secret, that should be flagged instead
    of silently wired in.

    Expected task format (built by Hermes from an [AGENT TASK - CREATIVE]
    block):
        __CREATIVE_TASK__
        type: image | video | full_video
        prompt: <description of the visual>              (image/video only)
        format: square | story | landscape | portrait     (image only, optional)
        aspect_ratio: 9:16 | 16:9 | 1:1                    (video only, optional, default 9:16)
        duration: <seconds>                                (video/full_video, optional, default 5/8)
        caption: yes | no                                  (image/video only, optional, default yes)
        image_prompt: <visual description>                 (full_video only)
        voiceover_text: <full narration script>             (full_video only)
        ---OVERLAY LINES---                                 (full_video only, optional)
        <one on-screen text line per line>
    """
    body = task[len(CREATIVE_TASK_MARKER):].lstrip("\n")
    fields = {}
    # Multi-line values. Previously every line containing a ":" was read
    # as a new field, so a caption_text spanning several lines was cut at
    # its first line and its remaining lines became junk fields — an
    # actual caption produced "here's why seranova is different" and
    # "the math is simple" as unrecognized fields, and the real caption
    # never arrived. A line now starts a new field ONLY if its key is a
    # name Yam actually knows; anything else is prose appended to the
    # field currently being read. Blank lines inside a value are kept,
    # because paragraph breaks are meaningful in a caption.
    last_key = None
    for raw_line in body.strip().split("\n"):
        line = raw_line.strip()
        # Section markers END the top-level field block. This check MUST
        # come before the startswith("---") skip below: markers start with
        # "---" themselves, so skipping first meant the terminator was
        # never seen, and everything inside ---OVERLAY LINES--- / ---SCENE---
        # leaked in as bogus top-level fields (observed: "line 1", "time",
        # "position", "style" all showing up as unrecognised fields).
        if line.startswith("---") and line.endswith("---") and len(line) > 6:
            break
        if line.startswith("---"):
            continue
        if not line:
            if last_key:
                fields[last_key] += "\n"
            continue
        key, sep, value = line.partition(":")
        if sep and key.strip().lower() in _ALL_FIELD_NAMES:
            last_key = key.strip().lower()
            fields[last_key] = value.strip()
        elif last_key:
            fields[last_key] += "\n" + line
        # else: stray prose before any field — ignored, as before.
    fields = {k: v.strip() for k, v in fields.items()}

    media_type = fields.get("type", "image").strip().lower()

    unknown_warning = _unknown_fields_warning(media_type, fields)

    if media_type == "full_video":
        result = _handle_full_video(fields, body, report)
    elif media_type == "edit_video":
        result = _handle_edit_video(fields, body, report)
    elif media_type == "multi_scene_video":
        result = _handle_multi_scene_video(fields, body, report)
    elif media_type == "multi_clip_video":
        result = _handle_multi_clip_video(fields, body, report)
    elif media_type == "research_content_ideas":
        result = _handle_research_content_ideas(fields, body, report)
    else:
        result = _handle_simple_media(fields, media_type, report)

    return (unknown_warning + result) if unknown_warning else result


# ─── OUTPUT CHECKS ───────────────────────────────────────────────────────────
# A prompt lowers the odds of a bad output; only a check stops it from
# shipping. Everything below runs AFTER generation, on the actual artifact,
# because instructions in the playbook have already been shown to be
# insufficient on their own — the same mistake came back through multiple
# review passes before anyone counted.

# Performance numbers and group-access selling are what got four of five
# videos rejected in TikTok's ad review. The rule went into the playbook as
# an instruction, which lowers the odds but cannot catch the case where the
# model writes "+38% this week" anyway.
_BANNED_CAPTION_PATTERNS = [
    (r"[+\-−]\s?\d+(?:[.,]\d+)?\s?%", "אחוזי תשואה/ביצועים"),
    (r"\b\d+(?:[.,]\d+)?\s?%\s*(?:רווח|תשואה|profit|return|gain|win rate|winrate)", "אחוז רווח/תשואה"),
    (r"(?:רווח|הרווחתי|הכנסתי|profit|p&l|pnl)\s*(?:של\s*)?[$₪]?\s?\d", "סכום רווח"),
    (r"[$₪]\s?\d[\d,.]*\s*(?:רווח|profit|in profit|per (?:day|week|month))", "סכום רווח"),
    (r"(?:קבוצת|ערוץ|קבוצה ב)\s*(?:ה)?טלגרם|join (?:my|our|the) telegram|telegram (?:group|channel)", "מכירת גישה לטלגרם"),
    # No \b around the Hebrew stems: Python's \b is defined on ASCII word
    # characters, so "מובטחות" slipped past a \bמובטח\b pattern in testing.
    # Matching the stem catches every inflection.
    (r"מובטח|הבטחה|מבטיח|\b(?:guaranteed|guarantee)\b", "הבטחת תוצאה"),
    (r"\bwin\s?rate\b|אחוז הצלחה|אחוזי הצלחה", "אחוז הצלחה"),
]


def check_caption(caption: str) -> list:
    """Returns a list of compliance problems found in a caption, empty if clean."""
    if not caption:
        return []
    return [label for pattern, label in _BANNED_CAPTION_PATTERNS
            if re.search(pattern, caption, re.IGNORECASE)]


def _image_dimensions(head: bytes):
    """Reads width/height from the first bytes of a PNG or JPEG, without
    pulling in Pillow. Returns (w, h) or None if it can't tell."""
    try:
        if head[:8] == b"\x89PNG\r\n\x1a\n" and len(head) >= 24:
            return struct.unpack(">II", head[16:24])
        if head[:2] == b"\xff\xd8":
            i = 2
            while i < len(head) - 9:
                if head[i] != 0xFF:
                    i += 1
                    continue
                marker = head[i + 1]
                if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                              0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
                    h, w = struct.unpack(">HH", head[i + 5:i + 9])
                    return w, h
                if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
                    i += 2
                    continue
                seg = struct.unpack(">H", head[i + 2:i + 4])[0]
                if seg < 2:
                    # Malformed length would loop forever; scan onward
                    # for the next marker instead of trusting it.
                    i += 2
                    continue
                i += 2 + seg
    except Exception:
        pass
    return None


def check_media_url(url: str, expect_kind: str = "", expect_size=None) -> list:
    """
    Actually fetches the media and reports what is wrong with it.

    This exists because build_image_url()/build_video_url() only assemble
    a URL string — nothing ever opened it. A failed generation therefore
    came back as "✅ מוכן" with a dead link, indistinguishable from a
    working one. Reports problems; it does NOT block delivery, since a
    link that merely looks wrong is still worth having in hand.
    """
    problems = []
    if not url:
        return ["אין קישור מדיה"]
    try:
        r = requests.get(url, stream=True, timeout=60)
        if r.status_code != 200:
            return [f"הקישור מחזיר HTTP {r.status_code}"]
        ctype = (r.headers.get("Content-Type") or "").lower()
        head = next(r.iter_content(chunk_size=65536), b"") or b""
        r.close()
        if expect_kind and expect_kind not in ctype and not ctype.startswith(expect_kind):
            problems.append(f"סוג תוכן לא צפוי: {ctype or 'לא ידוע'}")
        if len(head) < 1024:
            problems.append("הקובץ ריק או קטן מדי")
        if expect_kind == "image" and expect_size and head:
            dims = _image_dimensions(head)
            if dims and (dims[0], dims[1]) != tuple(expect_size):
                problems.append(f"מידות בפועל {dims[0]}x{dims[1]} במקום {expect_size[0]}x{expect_size[1]}")
    except Exception as e:
        problems.append(f"הקישור לא נגיש: {e}")
    return problems


def _checks_section(media_problems: list, caption_problems: list) -> str:
    """Renders check results. A clean pass is stated explicitly rather than
    left silent, so 'no warning' can be told apart from 'never checked'."""
    if not media_problems and not caption_problems:
        return "\n\n✅ בדיקות: המדיה נגישה והקפשן נקי."
    lines = ["\n\n⚠️ *בדיקות מצאו בעיות:*"]
    for p in media_problems:
        lines.append(f"• מדיה — {p}")
    for p in caption_problems:
        lines.append(f"• קפשן — {p} (נדחה בעבר בביקורת מודעות)")
    return "\n".join(lines)


def _handle_simple_media(fields: dict, media_type: str, report) -> str:
    image_url = fields.get("image_url", "").strip()
    video_url = fields.get("video_url", "").strip()
    # image_prompt is full_video's field name, but it gets sent to
    # image/video constantly (Hermes drafts it that way, and it reads as
    # the obvious name). Accepting it as an alias turns a hard "missing
    # 'prompt:' line" failure into a successful render. The unknown-field
    # warning above still fires, so the mismatch stays visible rather
    # than being silently normalised away.
    prompt = fields.get("prompt", "").strip() or fields.get("image_prompt", "").strip()
    if not prompt and not image_url and not video_url:
        unresolved = fields.get("unresolved_library_tag", "")
        if unresolved:
            return (
                f"❌ creative task: library_tag '{unresolved}' isn't in the "
                f"library — check with 'lib list' for the precise stored name"
            )
        return "❌ creative task: missing 'prompt:' line (or a resolved image_url/video_url)"

    # Three ways a caption can be resolved, in priority order:
    #   1. caption_text: — Hermes wrote the caption itself, use it VERBATIM.
    #      This existed as a real failure: Hermes wrote a strong caption
    #      into the yes/no `caption:` field, so it was silently discarded
    #      and Yam wrote a much weaker one from scratch.
    #   2. generate one — but only with real context (see below).
    #   3. caption: no — skip entirely.
    caption_text = fields.get("caption_text", "").strip()
    want_caption = fields.get("caption", "yes").strip().lower() != "no"

    if video_url:
        # A real, already-complete video from the library (e.g. one too
        # large/long for Telegram to relay, saved via a direct link) —
        # deliver it as-is. Never fed through Pollinations as an
        # image-to-video source, since it's already a finished video.
        report("משתמש בוידאו אמיתי מהספרייה (לא נוצר וידאו חדש)...")
        media_url = video_url
        media_note = "🎬 וידאו מהספרייה"
    elif media_type == "video":
        aspect_ratio = fields.get("aspect_ratio", "9:16")
        try:
            duration = int(fields.get("duration", "5"))
        except ValueError:
            duration = 5
        report("בונה קישור וידאו דרך Pollinations (Wan 2.7)...")
        media_url = creative.build_video_url(
            prompt, duration=duration, aspect_ratio=aspect_ratio,
            image_url=image_url or None,
        )
        note_suffix = " — מתמונה אמיתית מהספרייה" if image_url else ""
        media_note = f"🎬 וידאו ({duration} שנ', {aspect_ratio}){note_suffix}"
    elif image_url:
        # A real user-provided image (resolved from a library_tag) — use it
        # as-is, no generation needed.
        report("משתמש בתמונה אמיתית מהספרייה (לא נוצרת תמונה חדשה)...")
        media_url = image_url
        media_note = "🖼️ תמונה מהספרייה"
    else:
        fmt = fields.get("format", "").strip().lower()
        width, height = creative.FORMAT_DIMENSIONS.get(fmt, (1024, 1024))
        report("בונה קישור תמונה דרך Pollinations (Flux)...")
        media_url = creative.build_image_url(prompt, width=width, height=height)
        media_note = f"🖼️ תמונה ({width}x{height})"

    caption_section = ""
    if caption_text:
        caption_section = f"\n\n📝 *Caption:*\n{caption_text}"
    elif want_caption:
        report("כותב caption שיווקי עם ים...")
        # What the media actually IS. Without this, a library_tag resolves
        # to a bare URL and Yam has no idea what it's looking at — which is
        # exactly why it once replied "tell me what's in the image" instead
        # of writing a caption.
        subject = (
            prompt
            or fields.get("image_description", "").strip()
            or "תמונה/וידאו אמיתי מהמותג — אין תיאור זמין"
        )
        # Which venture this is for. The playbook leans trading-heavy
        # because Singularity is the main venture, so with no project
        # stated the model writes a trading caption regardless of what
        # the image shows. Stating it explicitly — and saying plainly
        # not to fall back to trading — is what stops that.
        project = fields.get("project", "").strip()
        project_line = (
            f"הפרויקט: {project}. כתוב אך ורק בהקשר של הפרויקט הזה — "
            f"אל תשתמש בשפה, בהאשטגים או בזוויות של פרויקט אחר "
            f"(במיוחד לא מסחר/טריידינג, אלא אם זה הפרויקט עצמו).\n"
            if project else
            "לא צוין פרויקט — כתוב לפי מה שרואים במדיה בלבד, ואל תניח "
            "שמדובר בתוכן מסחר/טריידינג.\n"
        )
        caption_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=creative_system_prompt(),
            messages=[{
                "role": "user",
                "content": (
                    f"כתוב caption קצר לפוסט הבא. החזר את ה-caption עצמו בלבד — "
                    f"אל תשאל שאלות ואל תבקש הבהרות; אם חסר לך מידע, כתוב caption "
                    f"סביר לפי מה שיש.\n"
                    f"{project_line}"
                    f"המדיה: {media_note}\n"
                    f"מה רואים: {subject}"
                )
            }]
        )
        caption_text = caption_response.content[0].text.strip()
        caption_section = f"\n\n📝 *Caption מוצע (ים):*\n{caption_text}"

    report("בודק שהמדיה באמת נגישה ושהקפשן עומד בכללים...")
    expect_kind = "image" if media_type == "image" else "video"
    expect_size = (width, height) if media_type == "image" and not image_url else None
    media_problems = check_media_url(media_url, expect_kind, expect_size)
    caption_problems = check_caption(caption_text)
    checks = _checks_section(media_problems, caption_problems)

    status = "⚠️" if media_problems else "✅"
    return f"{status} {media_note} מוכן:\n{media_url}{caption_section}{checks}"


def execute_task(task: str, report=None) -> str:
    """
    Entry point Yam's HTTP server calls for every incoming task. Hermes's
    extract_creative_tasks() always sends task strings already prefixed
    with __CREATIVE_TASK__ (kept for compatibility — no change needed on
    Hermes's side to talk to this service instead of Leo); creative_task()
    itself expects that prefix, so this just passes through, defaulting a
    missing prefix for direct/manual testing convenience.
    """
    if report is None:
        report = lambda _msg: None
    if not task.startswith(CREATIVE_TASK_MARKER):
        task = CREATIVE_TASK_MARKER + "\n" + task
    return creative_task(task, report)
