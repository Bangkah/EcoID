"""Local web UI server (stdlib only). Serves one HTML page and a small JSON API.

Privacy/security posture (SRS NFR-003, section 11):
- binds to 127.0.0.1 by default; Host header must be a loopback name (blocks DNS-rebinding)
- the page loads NOTHING from the internet (Content-Security-Policy enforces this in the browser)
- the server itself makes no outbound connections
"""
from __future__ import annotations

import hashlib
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from app.ai.contract.labels import CLASSES, COMMON_NAMES
from app.ai.preprocessing.preprocess import InvalidImageError
from app.observation.manager import MAX_IMAGE_BYTES, ObservationManager
from app.storage.store import NotFoundError

STATIC_DIR = Path(__file__).resolve().parent / "static"
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "[::1]"}
_ID = r"([0-9a-f]{32})"
_MIME = {".jpg": "image/jpeg", ".png": "image/png", ".webp": "image/webp", ".bmp": "image/bmp"}
CSP = ("default-src 'none'; img-src 'self' blob: data:; media-src 'self' blob:; style-src 'self' 'unsafe-inline'; "
       "script-src 'self' 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def build_model_info(config, model_path=None) -> dict:
    """SRS section 10: the user can see which model is running."""
    import onnxruntime
    info = {
        "model": config.model_name,
        "runtime": f"ONNX Runtime {onnxruntime.__version__}",
        "inference": "Local / CPU",
        "classes": f"{len(CLASSES)} plant species",
        "class_list": [{"scientific": c, "common": COMMON_NAMES[c]} for c in CLASSES],
        "threshold": config.threshold,
    }
    if model_path and Path(model_path).is_file():
        data = Path(model_path).read_bytes()
        info["file_sha256"] = hashlib.sha256(data).hexdigest()
        info["file_size_mb"] = round(len(data) / 1e6, 2)
    return info


def make_server(manager: ObservationManager, model_info: dict, host: str = "127.0.0.1", port: int = 8765,
                *, verbose: bool = False) -> ThreadingHTTPServer:
    check_host = host in ("127.0.0.1", "localhost", "::1")   # LAN mode (0.0.0.0) disables the Host check
    index_html = (STATIC_DIR / "index.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        server_version = "EcoID"
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):
            if verbose:
                super().log_message(fmt, *args)

        # ---- helpers ----
        def _send(self, code, body: bytes, ctype="application/json", extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", CSP)
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            if code in (411, 413):               # request body was not consumed: don't reuse the connection
                self.send_header("Connection", "close")
                self.close_connection = True
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code, obj):
            self._send(code, json.dumps(obj).encode())

        def _err(self, code, msg):
            self._json(code, {"error": msg})

        def _host_ok(self) -> bool:
            if not check_host:
                return True
            h = (self.headers.get("Host") or "").strip()
            name = h.rsplit(":", 1)[0] if not h.startswith("[") else h.split("]")[0] + "]"
            return name in LOOPBACK_HOSTS

        def _body(self, limit):
            try:
                n = int(self.headers.get("Content-Length", ""))
            except ValueError:
                raise _HttpError(411, "Content-Length required")
            if n > limit:
                raise _HttpError(413, "Request too large")
            return self.rfile.read(n)

        def _json_body(self):
            try:
                v = json.loads(self._body(64 * 1024) or b"{}")
            except json.JSONDecodeError:
                raise _HttpError(400, "Invalid JSON")
            if not isinstance(v, dict):
                raise _HttpError(400, "JSON object expected")
            return v

        def _dispatch(self, method):
            try:
                if not self._host_ok():
                    return self._err(403, "Forbidden host")
                url = urlparse(self.path)
                path, q = url.path, parse_qs(url.query)
                for m, pattern, fn in ROUTES:
                    if m == method and (mt := re.fullmatch(pattern, path)):
                        return fn(self, q, *mt.groups())
                self._err(404, "Not found")
            except _HttpError as e:
                self._err(e.code, e.msg)
            except NotFoundError:
                self._err(404, "Not found")
            except InvalidImageError as e:
                self._err(400, str(e))
            except ValueError as e:
                self._err(400, str(e))
            except Exception:                              # never leak internals
                self._err(500, "Internal error")

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_PATCH(self):
            self._dispatch("PATCH")

        def do_DELETE(self):
            self._dispatch("DELETE")

    # ---- routes ----
    def index(h, q):
        h._send(200, index_html, "text/html; charset=utf-8")

    def model(h, q):
        h._json(200, model_info)

    def identify(h, q):
        ctype = (h.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if not ctype.startswith("image/"):
            raise _HttpError(415, "Send the raw image bytes with an image/* Content-Type")
        draft = manager.identify(h._body(MAX_IMAGE_BYTES))
        h._json(200, draft.to_dict())

    def discard_draft(h, q, draft_id):
        manager.discard(draft_id)
        h._json(200, {"ok": True})

    def create_obs(h, q):
        b = h._json_body()
        obs = manager.save(str(b.get("draft_id", "")), b.get("verification_status"), b.get("notes", "") or "",
                           b.get("latitude"), b.get("longitude"))
        h._json(201, obs.to_dict())

    def list_obs(h, q):
        try:
            limit = int(q.get("limit", ["50"])[0])
            offset = int(q.get("offset", ["0"])[0])
        except ValueError:
            raise _HttpError(400, "limit/offset must be integers")
        items, total = manager.list((q.get("status") or [None])[0] or None, limit, offset)
        h._json(200, {"items": [o.to_dict() for o in items], "total": total})

    def get_obs(h, q, oid):
        h._json(200, manager.get(oid).to_dict())

    def patch_obs(h, q, oid):
        b = h._json_body()
        if "notes" in b and not isinstance(b["notes"], str):
            raise _HttpError(400, "notes must be a string")
        h._json(200, manager.update(oid, b.get("verification_status"), b.get("notes")).to_dict())

    def delete_obs(h, q, oid):
        manager.delete(oid)
        h._json(200, {"ok": True})

    def image(h, q, oid):
        obs = manager.get(oid)
        p = manager.store.abspath(obs.image_path)
        h._send(200, p.read_bytes(), _MIME.get(p.suffix, "application/octet-stream"))

    def thumb(h, q, oid):
        manager.get(oid)  # 404 if unknown
        p = manager.store.thumb_path(oid)
        if not p.is_file():
            raise NotFoundError(oid)
        h._send(200, p.read_bytes(), "image/jpeg")

    ROUTES = [
        ("GET", r"/", index),
        ("GET", r"/api/model", model),
        ("POST", r"/api/identify", identify),
        ("DELETE", rf"/api/drafts/{_ID}", discard_draft),
        ("POST", r"/api/observations", create_obs),
        ("GET", r"/api/observations", list_obs),
        ("GET", rf"/api/observations/{_ID}", get_obs),
        ("PATCH", rf"/api/observations/{_ID}", patch_obs),
        ("DELETE", rf"/api/observations/{_ID}", delete_obs),
        ("GET", rf"/images/{_ID}", image),
        ("GET", rf"/thumbs/{_ID}", thumb),
    ]

    srv = ThreadingHTTPServer((host, port), Handler)
    srv.daemon_threads = True
    return srv


class _HttpError(Exception):
    def __init__(self, code, msg):
        self.code, self.msg = code, msg
