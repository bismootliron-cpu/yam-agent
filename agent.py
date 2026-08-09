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
from anthropic import Anthropic
from tools import creative
from tools import tts
from tools import video_compose

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

CREATIVE_TASK_MARKER = "__CREATIVE_TASK__"  # kept in sync with Hermes's main.py

FULL_VIDEO_OVERLAY_SECTION = "---OVERLAY LINES---"


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

    overlay_lines = []
    if FULL_VIDEO_OVERLAY_SECTION in body:
        raw_lines = body.split(FULL_VIDEO_OVERLAY_SECTION, 1)[1].strip().split("\n")
        overlay_lines = [l.strip() for l in raw_lines if l.strip()]

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

    report("מעלה את הוידאו הסופי ל-media.pollinations.ai...")
    upload_result = creative.upload_media(final_path, os.path.basename(final_path))
    if "error" in upload_result:
        return (
            f"⚠️ הוידאו נוצר בהצלחה מקומית, אבל ההעלאה נכשלה: "
            f"{upload_result['error']}"
        )
    return f"✅ 🎬 וידאו מלא (טקסטים על המסך + קריינות) מוכן:\n{upload_result['link']}{music_note}"


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

    overlay_lines = []
    if FULL_VIDEO_OVERLAY_SECTION in body:
        raw_lines = body.split(FULL_VIDEO_OVERLAY_SECTION, 1)[1].strip().split("\n")
        overlay_lines = [l.strip() for l in raw_lines if l.strip()]

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

    report(f"מרכיב וידאו רב-סצנתי ({len(scenes)} סצנות) עם ffmpeg...")
    try:
        final_path = video_compose.assemble_multi_scene_video(
            scenes, voiceover_path=voiceover_path, disclaimer=disclaimer,
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
    for line in body.strip().split("\n"):
        line = line.strip()
        if not line or ":" not in line or line.startswith("---"):
            continue
        if FULL_VIDEO_OVERLAY_SECTION in line:
            break
        key, _, value = line.partition(":")
        fields[key.strip().lower()] = value.strip()

    media_type = fields.get("type", "image").strip().lower()

    if media_type == "full_video":
        return _handle_full_video(fields, body, report)

    if media_type == "edit_video":
        return _handle_edit_video(fields, body, report)

    if media_type == "multi_scene_video":
        return _handle_multi_scene_video(fields, body, report)

    if media_type == "multi_clip_video":
        return _handle_multi_clip_video(fields, body, report)

    image_url = fields.get("image_url", "").strip()
    video_url = fields.get("video_url", "").strip()
    prompt = fields.get("prompt", "").strip()
    if not prompt and not image_url and not video_url:
        unresolved = fields.get("unresolved_library_tag", "")
        if unresolved:
            return (
                f"❌ creative task: library_tag '{unresolved}' isn't in the "
                f"library — check with 'lib list' for the precise stored name"
            )
        return "❌ creative task: missing 'prompt:' line (or a resolved image_url/video_url)"

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
    if want_caption:
        report("כותב caption שיווקי עם ים...")
        caption_response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            system=YAM_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"כתוב caption קצר לפוסט הבא.\nהמדיה: {media_note}\nהתיאור/פרומפט: {prompt or 'תמונה אמיתית מהמותג, ללא תיאור טקסטואלי'}"
            }]
        )
        caption_section = f"\n\n📝 *Caption מוצע (ים):*\n{caption_response.content[0].text}"

    return f"✅ {media_note} מוכן:\n{media_url}{caption_section}"


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
