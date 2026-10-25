"""Loopback-only HTTP adapter for the operator UI."""

from __future__ import annotations

import json
import re
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .service import SnapshotService


class OperatorUiServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        service: SnapshotService,
        static_dir: Path,
    ):
        host, _ = server_address
        if host != "127.0.0.1":
            raise ValueError("operator UI must bind to 127.0.0.1")
        self.service = service
        self.static_dir = static_dir.resolve(strict=True)
        super().__init__(server_address, OperatorUiHandler)


class OperatorUiHandler(BaseHTTPRequestHandler):
    server: OperatorUiServer
    server_version = "M8OperatorUI/1"
    sys_version = ""

    STATIC_ROUTES = {
    "/": (
        "index.html",
        "text/html; charset=utf-8",
    ),
    "/static/app.js": (
        "app.js",
        "text/javascript; charset=utf-8",
    ),
    "/static/styles.css": (
        "styles.css",
        "text/css; charset=utf-8",
    ),
    "/static/favicon.svg": (
        "favicon.svg",
        "image/svg+xml",
    ),
    "/favicon.ico": (
        "favicon.svg",
        "image/svg+xml",
    ),
}

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        self._route(send_body=True)

    def do_HEAD(self) -> None:  # noqa: N802
        self._route(send_body=False)

    def do_POST(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_PUT(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def do_DELETE(self) -> None:  # noqa: N802
        self._method_not_allowed()

    def _route(self, *, send_body: bool) -> None:
        path = self.path.split("?", 1)[0]
        try:
            if path in self.STATIC_ROUTES:
                filename, content_type = self.STATIC_ROUTES[path]
                self._static(filename, content_type, send_body)
            elif path == "/api/v1/snapshot":
                self._json(self.server.service.snapshot_payload(), send_body)
            elif path == "/api/v1/session":
                self._json(self.server.service.session_payload(), send_body)
            elif path == "/api/v1/health":
                self._json(self.server.service.health_payload(), send_body)
            elif path == "/overlay":
                self._overlay(send_body)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found", send_body)
        except (OSError, ValueError, RuntimeError) as exc:
            self.log_error("request failed: %s", exc)
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, "operator UI internal error", send_body)

    def _base_headers(self) -> None:
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self'; script-src 'self'; "
            "style-src 'self'; connect-src 'self'; frame-ancestors 'none'",
        )

    def _send(
        self,
        status: HTTPStatus,
        content_type: str,
        body: bytes,
        send_body: bool,
        *,
        cache_control: str = "no-store",
    ) -> None:
        self.send_response(status)
        self._base_headers()
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if send_body:
            self.wfile.write(body)

    def _json(self, value: Any, send_body: bool) -> None:
        body = json.dumps(value, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._send(HTTPStatus.OK, "application/json; charset=utf-8", body, send_body)

    def _static(self, filename: str, content_type: str, send_body: bool) -> None:
        path = self.server.static_dir / filename
        try:
            body = path.read_bytes()
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "static resource unavailable", send_body)
            return
        self._send(
            HTTPStatus.OK,
            content_type,
            body,
            send_body,
            cache_control="no-cache",
        )

    def _overlay(self, send_body: bool) -> None:
        body = self.server.service.overlay_png()
        if body is None:
            self._error(HTTPStatus.NOT_FOUND, "overlay unavailable", send_body)
            return
        self._send(HTTPStatus.OK, "image/png", body, send_body)

    def _error(self, status: HTTPStatus, message: str, send_body: bool) -> None:
        body = json.dumps({"error": message}).encode("utf-8")
        self._send(status, "application/json; charset=utf-8", body, send_body)

    def _method_not_allowed(self) -> None:
        body = b'{"error":"method not allowed"}'
        self.send_response(HTTPStatus.METHOD_NOT_ALLOWED)
        self._base_headers()
        self.send_header("Allow", "GET, HEAD")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format_string: str, *args: object) -> None:
        message = format_string % args
        message = re.sub(r"[\x00-\x1f\x7f]", "?", message)
        sys.stderr.write(f"operator-ui: {self.address_string()} {message}\n")
