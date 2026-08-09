"""
Yam's video composition helper — burns text overlays onto a base
image/video and muxes in a voiceover track, via ffmpeg (subprocess).

REQUIRES the `ffmpeg` binary to be installed in the runtime environment —
this is a system dependency, not a pip package. On Railway, this needs a
Dockerfile apt-get step (see the Dockerfile in the Liron-agent repo).
This module raises a clear error if ffmpeg isn't found rather than
failing silently.
"""

import os
import shutil
import subprocess
import tempfile
import requests


def _download_to_temp(url: str, suffix: str) -> str:
    """
    Downloads a URL to a local temp file. Validates the response actually
    LOOKS like media (Content-Type starts with image/ or video/, and the
    body isn't suspiciously tiny) before handing it to ffmpeg — without
    this check, a failed/expired/rate-limited download (which often
    returns an HTML error page or a JSON error body) gets silently fed
    to ffmpeg as if it were a real image/video, producing a cryptic
    "Option loop not found" / demuxer-mismatch crash instead of a clear,
    actionable error.
    """
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    r = requests.get(url, timeout=120)
    r.raise_for_status()

    content_type = r.headers.get("Content-Type", "").lower()
    is_media_type = content_type.startswith("image/") or content_type.startswith("video/")
    if not is_media_type and len(r.content) < 1000:
        # No usable Content-Type AND suspiciously small — almost
        # certainly an error page/JSON body, not real media.
        snippet = r.content[:200]
        raise RuntimeError(
            f"download from {url} doesn't look like real media "
            f"(Content-Type: {content_type or 'missing'}, "
            f"size: {len(r.content)} bytes) — got: {snippet!r}"
        )

    with open(path, "wb") as f:
        f.write(r.content)
    return path


def _escape_drawtext(text: str) -> str:
    """
    Escapes text for safe use inside a SINGLE-QUOTED drawtext value
    (text='...'). A bare backslash before a quote does NOT work for
    apostrophes in ffmpeg's filtergraph syntax — ffmpeg requires closing
    the quote, inserting an escaped quote, and reopening it: 'It'\\''s'
    for "It's". Confirmed against a real ffmpeg failure ("No such filter"
    from a stray unescaped apostrophe splitting the filtergraph parse).
    """
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "'\\''")
    return text


def _atempo_chain(multiplier: float) -> str:
    """
    ffmpeg's atempo filter only accepts 0.5-2.0 per instance -- chain
    multiple atempo filters to reach a larger multiplier (e.g. 3.0 ->
    atempo=2.0,atempo=1.5).
    """
    if multiplier <= 0:
        raise RuntimeError("speed_multiplier must be positive")
    parts = []
    remaining = multiplier
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining}")
    return ",".join(parts)


def group_words_into_phrases(word_timings: list, words_per_phrase: int = 4) -> list:
    """
    Groups edge-tts's real per-word timing into short on-screen phrases,
    each with an ACTUAL start/end time matching the narration -- this is
    what makes captions genuinely synced instead of guessed from even
    division across the total duration.

    Returns a list of (text, start_seconds, end_seconds) tuples.
    """
    phrases = []
    for i in range(0, len(word_timings), words_per_phrase):
        chunk = word_timings[i:i + words_per_phrase]
        if not chunk:
            continue
        text = " ".join(w["text"] for w in chunk)
        start = chunk[0]["offset"]
        end = chunk[-1]["offset"] + chunk[-1]["duration"]
        phrases.append((text, start, end))
    return phrases


def _build_drawtext_filters(items: list, pulse: bool = False) -> list:
    """
    items: list of (text, start, end) tuples. pulse=True makes the text
    size gently oscillate (a subtle "pulse" effect) instead of staying a
    fixed size -- purely cosmetic, applied via a drawtext size expression.
    """
    filters = []
    for text, start, end in items:
        safe_text = _escape_drawtext(text)
        size_expr = f"'42+6*sin(2*PI*(t-{start}))'" if pulse else "42"
        filters.append(
            f"drawtext=text='{safe_text}':fontcolor=white:fontsize={size_expr}:"
            f"x=(w-text_w)/2:y=h-150:box=1:boxcolor=black@0.5:boxborderw=10:"
            f"enable='between(t,{start},{end})'"
        )
    return filters


def download_library_music(url: str) -> str:
    """
    Downloads a music track from a URL already stored in the media
    library (kind="audio") — used instead of a paid generation API when
    Hermes resolves a music_library_tag. Reuses _download_to_temp's
    validation (real Content-Type check) so a bad/expired link fails
    with a clear error instead of feeding garbage into ffmpeg.
    """
    return _download_to_temp(url, ".mp3")


def mix_in_music(video_path: str, music_path: str, music_volume: float = 0.15) -> str:
    """
    Mixes a background music track UNDER the video's existing audio
    (voiceover or original sound) at reduced volume, looping the music if
    it's shorter than the video. Runs as a separate pass after the main
    composition (re-encodes audio only -- video stream is copied, so this
    is fast and doesn't touch visual quality).

    Raises RuntimeError with ffmpeg's actual stderr on failure.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on this system")
    output_path = tempfile.mktemp(suffix=".mp4")
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-stream_loop", "-1", "-i", music_path,
        "-filter_complex",
        f"[1:a]volume={music_volume}[music];"
        f"[0:a][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg music mix failed: {result.stderr[-800:]}")
    return output_path


def edit_video_with_speed_ramp(video_url: str, trim_start: float = 0, duration: float = 0,
                                speed_ramp_at=None, speed_multiplier: float = 2.0,
                                overlay_lines=None,
                                voiceover_path=None,
                                pulse_text: bool = False) -> str:
    """
    Edits a real source video: trims to [trim_start, trim_start+duration],
    optionally speeds up everything AFTER speed_ramp_at (seconds into the
    trimmed clip) by speed_multiplier, burns in text overlays, and either
    keeps the original audio or replaces it with voiceover_path.

    overlay_lines are evenly spaced across the OUTPUT (post-speed-change)
    timeline -- NOT word-timing-synced, on purpose: during a
    fast-forwarded section, per-word captions would flash by unreadably
    fast anyway, so short manual hook/CTA lines make more sense here than
    auto-sync.

    If speed_ramp_at is None, the whole trimmed clip plays at normal speed.
    duration is REQUIRED when speed_ramp_at is given.

    Raises RuntimeError with ffmpeg's actual stderr on failure.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on this system -- it must be installed as a "
            "system dependency (not pip), e.g. via a Dockerfile apt-get step."
        )
    if speed_ramp_at is not None and not duration:
        raise RuntimeError("duration is required when using speed_ramp_at")
    if speed_ramp_at is not None and speed_ramp_at >= duration:
        raise RuntimeError("speed_ramp_at must be less than duration")

    base_path = _download_to_temp(video_url, ".mp4")
    output_path = tempfile.mktemp(suffix=".mp4")

    seg1_dur = speed_ramp_at if speed_ramp_at is not None else duration
    seg2_src_dur = (duration - speed_ramp_at) if speed_ramp_at is not None else 0
    seg2_out_dur = (seg2_src_dur / speed_multiplier) if seg2_src_dur else 0
    total_out_dur = seg1_dur + seg2_out_dur

    lines = overlay_lines or []
    slice_len = max(total_out_dur / max(len(lines), 1), 1) if total_out_dur else 1
    items = [(line, i * slice_len, i * slice_len + slice_len) for i, line in enumerate(lines)]
    drawtext_filters = _build_drawtext_filters(items, pulse=pulse_text)
    drawtext_str = "," + ",".join(drawtext_filters) if drawtext_filters else ""

    if speed_ramp_at is not None:
        atempo_str = _atempo_chain(speed_multiplier)
        filter_complex = (
            f"[0:v]trim=start={trim_start}:duration={seg1_dur},setpts=PTS-STARTPTS[v1];"
            f"[0:a]atrim=start={trim_start}:duration={seg1_dur},asetpts=PTS-STARTPTS[a1];"
            f"[0:v]trim=start={trim_start + seg1_dur}:duration={seg2_src_dur},"
            f"setpts=(PTS-STARTPTS)/{speed_multiplier}[v2];"
            f"[0:a]atrim=start={trim_start + seg1_dur}:duration={seg2_src_dur},"
            f"asetpts=PTS-STARTPTS,{atempo_str}[a2];"
            f"[v1][a1][v2][a2]concat=n=2:v=1:a=1[catv][cata];"
            f"[catv]{drawtext_str.lstrip(',') or 'null'}[outv]"
        )
        video_map, orig_audio_map = "[outv]", "[cata]"
    else:
        dur_clause = f":duration={duration}" if duration else ""
        filter_complex = (
            f"[0:v]trim=start={trim_start}{dur_clause},setpts=PTS-STARTPTS"
            f"{drawtext_str}[outv];"
            f"[0:a]atrim=start={trim_start}{dur_clause},asetpts=PTS-STARTPTS[outa]"
        )
        video_map, orig_audio_map = "[outv]", "[outa]"

    cmd = ["ffmpeg", "-y", "-i", base_path]
    if voiceover_path:
        cmd += ["-i", voiceover_path, "-filter_complex", filter_complex,
                "-map", video_map, "-map", "1:a:0", "-shortest"]
    else:
        cmd += ["-filter_complex", filter_complex, "-map", video_map, "-map", orig_audio_map]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k", "-pix_fmt", "yuv420p", output_path]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")
    return output_path


def compose_video_with_overlay(base_media_url: str, base_is_video: bool,
                                voiceover_path, overlay_lines: list,
                                duration: int = 8, trim_start: float = 0,
                                word_timings=None,
                                pulse_text: bool = False, zoom: bool = False) -> str:
    """
    Builds a final marketing video: base image/video + burned-in text
    overlays + a voiceover audio track. Returns a local path to the
    resulting .mp4.

    Captions: if word_timings is given (from
    tts.generate_voiceover_with_timing) and overlay_lines is empty,
    captions are auto-built as short phrases with REAL start/end times
    from the narration (group_words_into_phrases) -- genuinely synced,
    not evenly divided. Pass explicit overlay_lines to override with
    manual hook/CTA text instead (evenly spaced across `duration`).

    zoom=True applies a slow continuous Ken Burns zoom -- ONLY meaningful
    (and only applied) when base_is_video=False (a still image); it's a
    no-op for real video input, since zooming already-moving footage in
    ffmpeg is fragile and rarely looks intentional.

    voiceover_path may be None ONLY when base_is_video=True -- in that
    case the base video's OWN original audio is kept instead of replaced
    (real "edit an existing video" needs this).

    trim_start (seconds) only applies when base_is_video=True.

    Raises RuntimeError with ffmpeg's actual stderr if composition fails.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on this system -- it must be installed as a "
            "system dependency (not pip), e.g. via a Dockerfile apt-get step."
        )
    if voiceover_path is None and not base_is_video:
        raise RuntimeError("voiceover_path is required when composing from a still image")

    suffix = ".mp4" if base_is_video else ".jpg"
    base_path = _download_to_temp(base_media_url, suffix)
    output_path = tempfile.mktemp(suffix=".mp4")

    if overlay_lines:
        slice_len = max(duration / max(len(overlay_lines), 1), 1)
        items = [(line, i * slice_len, i * slice_len + slice_len) for i, line in enumerate(overlay_lines)]
    elif word_timings:
        items = group_words_into_phrases(word_timings)
    else:
        items = []
    drawtext_filters = _build_drawtext_filters(items, pulse=pulse_text)
    filter_str = ",".join(drawtext_filters) if drawtext_filters else "null"

    if base_is_video:
        input_args = []
        if trim_start:
            input_args += ["-ss", str(trim_start)]
        input_args += ["-i", base_path]
        if duration:
            input_args += ["-t", str(duration)]
    else:
        if zoom:
            # Slow continuous zoom-in (Ken Burns) on the still image --
            # zoompan needs an explicit frame count (25fps assumed).
            zoom_filter = (
                f"zoompan=z='min(zoom+0.0015\\,1.2)':d=1:s=1080x1920:fps=25,"
                f"trim=duration={duration}"
            )
            filter_str = f"{zoom_filter},{filter_str}" if filter_str != "null" else zoom_filter
        input_args = ["-loop", "1", "-i", base_path, "-t", str(duration)]

    if voiceover_path:
        cmd = [
            "ffmpeg", "-y", *input_args, "-i", voiceover_path,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", "-pix_fmt", "yuv420p",
            "-map", "0:v:0", "-map", "1:a:0",
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y", *input_args,
            "-vf", filter_str,
            "-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p",
            output_path,
        ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")
    return output_path


def _build_scene_clip(image_url: str, duration: float, zoom: str = "",
                       text: str = "", text_delay: float = 0) -> str:
    """
    Renders ONE scene as a silent .mp4: a still image for `duration`
    seconds, with an optional Ken Burns zoom ("in" or "out") and an
    optional single line of on-screen text appearing after text_delay
    seconds (shown for the rest of the scene). No audio track — the
    overall voiceover/music is muxed in once, after all scenes are
    concatenated.
    """
    img_path = _download_to_temp(image_url, ".jpg")
    output_path = tempfile.mktemp(suffix=".mp4")

    if zoom == "out":
        zoom_filter = f"zoompan=z='if(eq(on\\,1)\\,1.3\\,max(zoom-0.003\\,1.0))':d=1:s=1080x1920:fps=25"
    elif zoom == "in":
        zoom_filter = f"zoompan=z='min(zoom+0.002\\,1.3)':d=1:s=1080x1920:fps=25"
    else:
        zoom_filter = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"

    vf = f"{zoom_filter},trim=duration={duration}"
    if text:
        safe_text = _escape_drawtext(text)
        vf += (
            f",drawtext=text='{safe_text}':fontcolor=white:fontsize=48:"
            f"x=(w-text_w)/2:y=h-200:box=1:boxcolor=black@0.5:boxborderw=12:"
            f"enable='gte(t,{text_delay})'"
        )

    cmd = [
        "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-t", str(duration),
        "-vf", vf, "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg scene render failed: {result.stderr[-800:]}")
    return output_path


def _concat_clips(clip_paths: list) -> str:
    """Concatenates several .mp4 files (same codec/resolution) via ffmpeg's
    concat demuxer — the standard reliable way to join pre-rendered clips."""
    list_path = tempfile.mktemp(suffix=".txt")
    with open(list_path, "w") as f:
        for p in clip_paths:
            f.write(f"file '{p}'\n")
    output_path = tempfile.mktemp(suffix=".mp4")
    cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
           "-c", "copy", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr[-800:]}")
    return output_path


def add_disclaimer_overlay(video_path: str, disclaimer_text: str, seconds: float = 4) -> str:
    """Burns a small disclaimer line into the LAST `seconds` of the video."""
    output_path = tempfile.mktemp(suffix=".mp4")
    safe_text = _escape_drawtext(disclaimer_text)
    # Get total duration via ffprobe-free approach: ffmpeg -f null pass is
    # overkill here — instead just enable in the tail window relative to
    # END using a negative-from-end expression isn't native, so we probe
    # duration via ffprobe (a companion of ffmpeg, installed alongside it).
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, timeout=30,
    )
    try:
        total_duration = float(probe.stdout.strip())
    except ValueError:
        total_duration = seconds  # fallback: show for the whole tail guess
    start = max(total_duration - seconds, 0)
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf",
        f"drawtext=text='{safe_text}':fontcolor=white:fontsize=24:"
        f"x=(w-text_w)/2:y=h-60:box=1:boxcolor=black@0.6:boxborderw=8:"
        f"enable='gte(t,{start})'",
        "-c:a", "copy", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg disclaimer overlay failed: {result.stderr[-800:]}")
    return output_path


def assemble_multi_scene_video(scenes: list, voiceover_path: str | None = None,
                                disclaimer: str = "") -> str:
    """
    Builds a multi-scene marketing video from several still images, each
    shown as its own timed scene with its own zoom direction and optional
    text — then concatenates them, muxes in one overall voiceover track
    (if given), and optionally burns a disclaimer into the final seconds.

    scenes: list of dicts, each with:
        image_url (required), duration (seconds, required),
        zoom ("in" | "out" | "" , optional), text (optional),
        text_delay (seconds into the scene, optional)

    Raises RuntimeError with ffmpeg's actual stderr on failure.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on this system")
    if not scenes:
        raise RuntimeError("assemble_multi_scene_video: no scenes given")

    clip_paths = [
        _build_scene_clip(
            s["image_url"], s["duration"],
            zoom=s.get("zoom", ""), text=s.get("text", ""),
            text_delay=s.get("text_delay", 0),
        )
        for s in scenes
    ]
    combined = _concat_clips(clip_paths)

    if voiceover_path:
        with_audio = tempfile.mktemp(suffix=".mp4")
        cmd = [
            "ffmpeg", "-y", "-i", combined, "-i", voiceover_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-shortest", with_audio,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not os.path.exists(with_audio):
            raise RuntimeError(f"ffmpeg audio mux failed: {result.stderr[-800:]}")
        combined = with_audio

    if disclaimer:
        combined = add_disclaimer_overlay(combined, disclaimer)

    return combined


def _build_video_clip_scene(video_url: str, trim_start: float = 0, duration: float = 0,
                             text: str = "", text_delay: float = 0) -> str:
    """
    Trims a REAL video clip to [trim_start, trim_start+duration] (omit
    duration to keep the rest), with an optional text overlay. Audio is
    dropped (-an) — used by assemble_multi_clip_video, which mutes each
    individual clip's own sound and plays ONE continuous voiceover over
    the whole combined output instead (the point of this type: several
    real clips narrated as one continuous script, not several separate
    self-contained edits).
    """
    base_path = _download_to_temp(video_url, ".mp4")
    output_path = tempfile.mktemp(suffix=".mp4")
    dur_clause = f":duration={duration}" if duration else ""
    vf = f"trim=start={trim_start}{dur_clause},setpts=PTS-STARTPTS"
    if text:
        safe_text = _escape_drawtext(text)
        vf += (
            f",drawtext=text='{safe_text}':fontcolor=white:fontsize=42:"
            f"x=(w-text_w)/2:y=h-150:box=1:boxcolor=black@0.5:boxborderw=10:"
            f"enable='gte(t,{text_delay})'"
        )
    cmd = [
        "ffmpeg", "-y", "-i", base_path, "-vf", vf,
        "-c:v", "libx264", "-preset", "slow", "-crf", "18",
        "-pix_fmt", "yuv420p", "-an", output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0 or not os.path.exists(output_path):
        raise RuntimeError(f"ffmpeg video clip render failed: {result.stderr[-800:]}")
    return output_path


def assemble_multi_clip_video(clips: list, voiceover_path: str | None = None,
                               outro_image_url: str = "", outro_duration: float = 5,
                               outro_text: str = "", disclaimer: str = "") -> str:
    """
    Combines several REAL VIDEO clips (each trimmed) into ONE video, with
    a single continuous voiceover playing over the whole combined
    length — this is what edit_video and multi_scene_video can't do:
    edit_video only ever handles one clip, and multi_scene_video only
    accepts still images as scenes. Optionally ends on a still-image
    outro (e.g. a logo card, built the same way as multi_scene_video's
    scenes) and an optional disclaimer burned into the final seconds.

    clips: list of dicts, each with:
        video_url (required), trim_start (seconds, optional, default 0),
        duration (seconds, optional — omit to keep the rest of that
        source clip), text/text_delay (optional on-screen text).

    Raises RuntimeError with ffmpeg's actual stderr on failure.
    """
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on this system")
    if not clips:
        raise RuntimeError("assemble_multi_clip_video: no clips given")

    clip_paths = [
        _build_video_clip_scene(
            c["video_url"], trim_start=c.get("trim_start", 0),
            duration=c.get("duration", 0), text=c.get("text", ""),
            text_delay=c.get("text_delay", 0),
        )
        for c in clips
    ]
    if outro_image_url:
        clip_paths.append(_build_scene_clip(outro_image_url, outro_duration or 5, zoom="", text=outro_text))

    combined = _concat_clips(clip_paths)

    if voiceover_path:
        with_audio = tempfile.mktemp(suffix=".mp4")
        cmd = [
            "ffmpeg", "-y", "-i", combined, "-i", voiceover_path,
            "-map", "0:v", "-map", "1:a", "-c:v", "copy",
            "-c:a", "aac", "-b:a", "192k", "-shortest", with_audio,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0 or not os.path.exists(with_audio):
            raise RuntimeError(f"ffmpeg audio mux failed: {result.stderr[-800:]}")
        combined = with_audio

    if disclaimer:
        combined = add_disclaimer_overlay(combined, disclaimer)

    return combined
