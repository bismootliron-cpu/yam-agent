"""
ים (Yam) — standalone HTTP server. Implements the exact same task-queue
contract Hermes already uses to talk to Leo, so switching Hermes over to
this service is just pointing it at a new URL (YAM_AGENT_URL) — no change
to Hermes's delegation/polling logic itself:

    POST /task          {"task": "<task string>"}  -> {"task_id": "..."}
    GET  /task/<id>/status                          -> {"status": "running"|"done"|"failed",
                                                          "result"|"error"|"progress": ...}
    GET  /health                                     -> {"status": "ok"}

Auth: same shared-secret pattern as Leo — Authorization: Bearer <AGENT_SECRET>.
"""

import os
import json
import re
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

from agent import execute_task

AGENT_SECRET = os.environ.get("AGENT_SECRET", "hermes-agent-2026")

# In-memory task store: {task_id: {"status": ..., "result"/"error"/"progress": ...}}
# A process restart loses in-flight tasks — acceptable here since Hermes's
# own monitor already treats "no status found" as a reportable failure
# rather than hanging forever, and creative tasks are safely re-runnable
# (no partial side effects to clean up, unlike e.g. a Whop publish).
_tasks = {}
_tasks_lock = threading.Lock()


def _run_task(task_id: str, task: str):
    def report(progress_msg: str):
        with _tasks_lock:
            if task_id in _tasks:
                _tasks[task_id]["progress"] = progress_msg

    try:
        result = execute_task(task, report=report)
        with _tasks_lock:
            _tasks[task_id] = {"status": "done", "result": result}
    except Exception as e:
        with _tasks_lock:
            _tasks[task_id] = {"status": "failed", "error": str(e)}


class YamHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _check_auth(self) -> bool:
        return self.headers.get("Authorization", "") == f"Bearer {AGENT_SECRET}"

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "agent": "Yam"})
            return

        match = re.match(r"^/task/([^/]+)/status$", self.path)
        if match:
            if not self._check_auth():
                self.send_response(401)
                self.end_headers()
                return
            task_id = match.group(1)
            with _tasks_lock:
                info = _tasks.get(task_id)
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
            with _tasks_lock:
                _tasks[task_id] = {"status": "running", "progress": "התחיל..."}
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
