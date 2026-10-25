"""Loopback HTTP contract tests for the operator UI."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from operator_ui.server import OperatorUiServer


class FakeService:
    PNG = b"\x89PNG\r\n\x1a\nhttp-test"

    def snapshot_payload(self):
        return {
            "schema_version": 1,
            "health": {"state": "LIVE"},
            "event": None,
            "counts": {},
            "holes": [],
        }

    def session_payload(self):
        return {"session_id": "test-session"}

    def health_payload(self):
        return {
            "state": "LIVE",
            "message": "",
        }

    def overlay_png(self):
        return self.PNG


class ServerTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.static_dir = Path(self.temporary_directory.name)

        (self.static_dir / "index.html").write_text(
            "<!doctype html><title>operator test</title>",
            encoding="utf-8",
        )
        (self.static_dir / "app.js").write_text(
            "console.log('operator test');",
            encoding="utf-8",
        )
        (self.static_dir / "styles.css").write_text(
            "body { color: white; }",
            encoding="utf-8",
        )
        (self.static_dir / "favicon.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"></svg>',
            encoding="utf-8",
        )

        self.server = OperatorUiServer(
            ("127.0.0.1", 0),
            FakeService(),
            self.static_dir,
        )

        self.thread = threading.Thread(
            target=self.server.serve_forever,
            kwargs={"poll_interval": 0.01},
            daemon=True,
        )
        self.thread.start()

        self.base_url = (
            f"http://127.0.0.1:{self.server.server_port}"
        )

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.temporary_directory.cleanup()

    def open(self, path, *, method="GET", data=None):
        request = Request(
            self.base_url + path,
            method=method,
            data=data,
        )
        return urlopen(request, timeout=2)

    def test_static_routes_and_favicon_alias_return_200(self):
        expected = {
            "/": "text/html",
            "/static/app.js": "text/javascript",
            "/static/styles.css": "text/css",
            "/static/favicon.svg": "image/svg+xml",
            "/favicon.ico": "image/svg+xml",
        }

        for path, content_type in expected.items():
            with self.subTest(path=path), self.open(path) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get_content_type(),
                    content_type,
                )
                self.assertTrue(response.read())

    def test_json_api_routes_return_expected_payloads(self):
        expected = {
            "/api/v1/snapshot": ("schema_version", 1),
            "/api/v1/session": (
                "session_id",
                "test-session",
            ),
            "/api/v1/health": ("state", "LIVE"),
        }

        for path, (key, value) in expected.items():
            with self.subTest(path=path), self.open(path) as response:
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.headers.get_content_type(),
                    "application/json",
                )
                self.assertEqual(
                    json.load(response)[key],
                    value,
                )

    def test_overlay_route_returns_png(self):
        with self.open("/overlay") as response:
            self.assertEqual(response.status, 200)
            self.assertEqual(
                response.headers.get_content_type(),
                "image/png",
            )
            self.assertEqual(
                response.read(),
                FakeService.PNG,
            )

    def test_head_returns_headers_without_body(self):
        with self.open(
            "/static/app.js",
            method="HEAD",
        ) as response:
            self.assertEqual(response.status, 200)
            self.assertGreater(
                int(response.headers["Content-Length"]),
                0,
            )
            self.assertEqual(response.read(), b"")

    def test_unknown_route_returns_404(self):
        with self.assertRaises(HTTPError) as caught:
            self.open("/static/not-allowlisted.txt")

        self.assertEqual(caught.exception.code, 404)
        self.assertEqual(
            json.load(caught.exception),
            {"error": "not found"},
        )

    def test_post_is_rejected_with_allow_header(self):
        with self.assertRaises(HTTPError) as caught:
            self.open(
                "/api/v1/snapshot",
                method="POST",
                data=b"{}",
            )

        self.assertEqual(caught.exception.code, 405)
        self.assertEqual(
            caught.exception.headers["Allow"],
            "GET, HEAD",
        )

    def test_security_headers_are_present(self):
        with self.open("/") as response:
            self.assertEqual(
                response.headers["X-Content-Type-Options"],
                "nosniff",
            )
            self.assertEqual(
                response.headers["X-Frame-Options"],
                "DENY",
            )
            self.assertIn(
                "connect-src 'self'",
                response.headers["Content-Security-Policy"],
            )

    def test_non_loopback_binding_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError,
            "127.0.0.1",
        ):
            OperatorUiServer(
                ("0.0.0.0", 0),
                FakeService(),
                self.static_dir,
            )


if __name__ == "__main__":
    unittest.main()