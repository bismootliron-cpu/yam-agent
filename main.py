"""
ים (Yam) — standalone HTTP server. Implements the exact same task-queue
contract Hermes already uses to talk to Leo, so switching Hermes over to
this service is just pointing it at a new URL (YAM_AGENT_URL) — no change
to Hermes's delegation/polling logic itself:

    POST /task          {"task": "<task string>"}  -> {"task_id": "..."}
    GET  /task/<id>/status                          -> {"status": "running"|"done"|"failed",
                                                          "result"|"error"|"progress": ...}
    GET  /health                                     -> {"status": "ok"}
    GET  /capabilities                               -> {"<type>": {"description", "fields", ...}, ...}

Auth: same shared-secret pattern as Leo — Authorization: Bearer <AGENT_SECRET>.
"""

import os
import json
import re
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

from agent import execute_task, get_capabilities

AGENT_SECRET = os.environ.get("AGENT_SECRET", "hermes-agent-2026")

# Task state lives in Supabase when configured, with the in-memory dict as
# a cache/fallback.
#
# WHY: this used to be memory-only, and that caused a real, confusing
# failure. Deploying new code restarts the service, which wiped every
# in-flight task; Hermes then polled for task ids that no longer existed,
# got 404s, and simply never learned that four videos had been requested —
# they vanished silently. Persisting means a restart mid-render still
# leaves a record Hermes can read, and "running" tasks orphaned by a
# restart can be reported honestly instead of disappearing.
#
# Requires (same Supabase project Hermes uses):
#   create table if not exists yam_tasks (
#     task_id text primary key, status text, progress text,
#     result text, error text,
#     created_at timestamptz default now(),
#     updated_at timestamptz default now());
_tasks = {}
_tasks_lock = threading.Lock()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
_supabase = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    try:
        from supabase import create_client
        _supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("🗄️  Yam task persistence: ON")
    except Exception as e:
        print(f"⚠️ Supabase unavailable, falling back to memory-only tasks: {e}")
else:
    print("⚠️ SUPABASE_URL / SUPABASE_SERVICE_KEY not set — tasks are memory-only "
          "and will be lost on restart.")


def _persist_task(task_id: str, info: dict) -> None:
    """Best-effort write. Persistence failing must never break a render —
    the in-memory copy still serves the common case."""
    if not _supabase:
        return
    try:
        row = {
            "task_id": task_id,
            "status": info.get("status", ""),
            "progress": (info.get("progress") or "")[:1000],
            "result": (info.get("result") or "")[:4000],
            "error": (info.get("error") or "")[:2000],
            "updated_at": "now()",
        }
        _supabase.table("yam_tasks").upsert(row, on_conflict="task_id").execute()
    except Exception as e:
        print(f"⚠️ task persist error ({task_id}): {e}")


def _load_task(task_id: str):
    if not _supabase:
        return None
    try:
        result = _supabase.table("yam_tasks").select("*").eq("task_id", task_id).limit(1).execute()
        if not result.data:
            return None
        row = result.data[0]
        info = {"status": row.get("status") or "unknown"}
        for key in ("progress", "result", "error"):
            if row.get(key):
                info[key] = row[key]
        # A task still marked "running" in the DB but absent from memory
        # means the process died mid-render. Say so plainly rather than
        # leaving Hermes to poll a task nobody is working on.
        if info["status"] == "running":
            info["status"] = "failed"
            info["error"] = (
                "המשימה נקטעה כשהשירות הופעל מחדש (כנראה deploy) ולא הושלמה. "
                "צריך לשלוח אותה שוב."
            )
        return info
    except Exception as e:
        print(f"⚠️ task load error ({task_id}): {e}")
        return None


def _set_task(task_id: str, info: dict) -> None:
    with _tasks_lock:
        _tasks[task_id] = info
    _persist_task(task_id, info)


def _get_task(task_id: str):
    with _tasks_lock:
        info = _tasks.get(task_id)
    if info is not None:
        return info
    return _load_task(task_id)


def _run_task(task_id: str, task: str):
    def report(progress_msg: str):
        with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id]["progress"] = progress_msg
                snapshot = dict(_tasks[task_id])
            else:
                snapshot = {"status": "running", "progress": progress_msg}
        _persist_task(task_id, snapshot)

    try:
        result = execute_task(task, report=report)
        _set_task(task_id, {"status": "done", "result": result})
    except Exception as e:
        _set_task(task_id, {"status": "failed", "error": str(e)})


class YamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _check_auth(self) -> bool:
        return self.headers.get("Authorization", "") == f"Bearer {AGENT_SECRET}"

    def _send_json(self, status: int, payload: dict):
        # Some values in CAPABILITIES (e.g. field sets) are Python sets,
        # which json.dumps can't serialize directly — default=list covers
        # that without needing to hand-convert CAPABILITIES itself.
        body = json.dumps(payload, default=list).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "agent": "Yam"})
            return

        if self.path == "/capabilities":
            # No auth required — this is read-only self-description, not
            # an action, and it's useful to be able to check it (e.g. from
            # a browser) without needing the shared secret handy.
            self._send_json(200, get_capabilities())
            return

        match = re.match(r"^/task/([^/]+)/status$", self.path)
        if match:
            if not self._check_auth():
                self.send_response(401)
                self.end_headers()
                return
            task_id = match.group(1)
            info = _get_task(task_id)
            if info is None:
                self._send_json(404, {"status": "unknown", "error": "no such task_id"})
                return
            self._send_json(200, info)
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        if self.path == "/task":
            if not self._check_auth():
                self.send_response(401)
                self.end_headers()
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                task = body.get("task", "")
                if not task:
                    self._send_json(400, {"error": "missing 'task'"})
                    return
            except Exception as e:
                self._send_json(400, {"error": f"invalid request body: {e}"})
                return

            task_id = uuid.uuid4().hex
            _set_task(task_id, {"status": "running", "progress": "התחיל..."})
            threading.Thread(target=_run_task, args=(task_id, task), daemon=True).start()
            self._send_json(200, {"task_id": task_id})
            return

        self.send_response(404)
        self.end_headers()


def main():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(("0.0.0.0", port), YamHandler)
    print(f"🌊 Yam running on port {port} — standalone creative agent")
    server.serve_forever()


if __name__ == "__main__":
    main()
