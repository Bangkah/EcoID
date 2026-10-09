"""Phase 3.4 — end-to-end, with a real HTTP server, real ONNX Runtime and the network blocked."""
import csv
import http.client
import io
import json
import re
import subprocess
import sys
import tempfile
import threading
import unittest
import uuid
import zipfile
from pathlib import Path

from app.ai.config import InferenceConfig
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier
from app.observation.manager import ObservationManager
from app.storage.store import ObservationStore
from app.ui.server import STATIC_DIR, build_model_info, make_server
from tests.helpers import block_network, exif_jpeg, image_bytes

ROOT = Path(__file__).resolve().parents[2]
FR007 = {"id", "image_path", "predicted_species", "confidence", "alternative_predictions",
         "verification_status", "timestamp", "latitude", "longitude", "notes", "model"}


class AppE2E(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmp.name)
        cls.model = cls.root / "tiny.onnx"
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_test_onnx.py"), str(cls.model)])
        cls.data = cls.root / "data"
        cfg = InferenceConfig("EcoID Vision Model", 0.65)
        cls.manager = ObservationManager(Identifier(OnnxBackend(cls.model), cfg), ObservationStore(cls.data))
        cls.srv = make_server(cls.manager, build_model_info(cfg, cls.model), "127.0.0.1", 0)
        cls.port = cls.srv.server_address[1]
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.tmp.cleanup()

    # --- tiny HTTP client ---
    def req(self, method, path, body=None, headers=None, ctype=None):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=15)
        h = dict(headers or {})
        if ctype:
            h["Content-Type"] = ctype
        c.request(method, path, body=body, headers=h)
        r = c.getresponse()
        data = r.read()
        c.close()
        return r.status, r, data

    def jreq(self, method, path, obj=None):
        body = None if obj is None else json.dumps(obj)
        s, r, d = self.req(method, path, body, ctype="application/json" if obj is not None else None)
        return s, json.loads(d) if d else None

    def identify(self, rgb=(230, 20, 20), size=(1200, 900)):
        raw = image_bytes(rgb, size)
        s, r, d = self.req("POST", "/api/identify", raw, ctype="image/jpeg")
        return raw, s, json.loads(d)

    # --- tests ---
    def test_full_offline_flow_capture_result_verify_history(self):
        with block_network() as attempts:
            raw, s, d = self.identify((230, 20, 20))                           # capture -> result
            self.assertEqual(s, 200)
            self.assertEqual(d["result"]["status"], "IDENTIFIED")
            self.assertEqual(d["result"]["candidates"][0]["label"], "Mangifera indica")
            self.assertEqual(len(d["result"]["candidates"]), 3)
            self.assertEqual(d["result"]["model"], "EcoID Vision Model")

            s, o = self.jreq("POST", "/api/observations",                      # verify -> saved
                             {"draft_id": d["draft_id"], "verification_status": "VERIFIED", "notes": "checked the leaf"})
            self.assertEqual(s, 201)
            self.assertTrue(FR007 <= set(o))
            self.assertEqual((o["verification_status"], o["notes"], o["model"]),
                             ("VERIFIED", "checked the leaf", "EcoID Vision Model"))
            self.assertIsNone(o["latitude"])

            s, lst = self.jreq("GET", "/api/observations?limit=100")           # history
            self.assertIn(o["id"], [x["id"] for x in lst["items"]])

            s, r, img = self.req("GET", f"/images/{o['id']}")                  # original photo, untouched
            self.assertEqual((s, img), (200, raw))
            self.assertEqual(r.getheader("Content-Type"), "image/jpeg")
            s, r, th = self.req("GET", f"/thumbs/{o['id']}")
            self.assertEqual((s, r.getheader("Content-Type")), (200, "image/jpeg"))

            s, o2 = self.jreq("PATCH", f"/api/observations/{o['id']}", {"verification_status": "REJECTED", "notes": "n"})
            self.assertEqual((s, o2["verification_status"], o2["notes"]), (200, "REJECTED", "n"))

            s, _ = self.jreq("DELETE", f"/api/observations/{o['id']}")
            self.assertEqual(s, 200)
            self.assertEqual(self.jreq("GET", f"/api/observations/{o['id']}")[0], 404)
            self.assertEqual(self.req("GET", f"/images/{o['id']}")[0], 404)
        self.assertEqual(attempts, [], f"outbound network attempts: {attempts}")

    def test_low_confidence_flow(self):
        _, s, d = self.identify((128, 128, 128), (200, 150))
        self.assertEqual(d["result"]["status"], "LOW_CONFIDENCE")
        s, o = self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "UNCERTAIN"})
        self.assertEqual((o["identification_status"], o["verification_status"]), ("LOW_CONFIDENCE", "UNCERTAIN"))

    def test_observations_survive_restart(self):
        _, _, d = self.identify(size=(100, 80))
        _, o = self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "VERIFIED"})
        reopened = ObservationManager(Identifier(OnnxBackend(self.model), InferenceConfig("m", 0.65)),
                                      ObservationStore(self.data))
        self.assertEqual(reopened.get(o["id"]).verification_status.value, "VERIFIED")

    def test_pagination_and_filter(self):
        for st in ("REJECTED", "REJECTED", "REJECTED"):
            _, _, d = self.identify(size=(60, 40))
            self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": st})
        s, r = self.jreq("GET", "/api/observations?status=REJECTED&limit=2&offset=0")
        self.assertEqual(len(r["items"]), 2)
        self.assertGreaterEqual(r["total"], 3)
        self.assertTrue(all(x["verification_status"] == "REJECTED" for x in r["items"]))
        self.assertEqual(self.jreq("GET", "/api/observations?limit=abc")[0], 400)
        self.assertEqual(self.jreq("GET", "/api/observations?status=MAYBE")[0], 400)

    def test_notes_are_stored_verbatim_and_served_as_json(self):
        evil = "<script>alert(1)</script>"
        _, _, d = self.identify(size=(60, 40))
        _, o = self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "VERIFIED", "notes": evil})
        self.assertEqual(o["notes"], evil)
        s, r, _ = self.req("GET", f"/api/observations/{o['id']}")
        self.assertEqual(r.getheader("Content-Type"), "application/json")

    def test_error_handling(self):
        self.assertEqual(self.req("POST", "/api/identify", b"x", ctype="text/plain")[0], 415)
        self.assertEqual(self.req("POST", "/api/identify", b"not an image", ctype="image/jpeg")[0], 400)
        self.assertEqual(self.req("POST", "/api/identify", b"", ctype="image/jpeg")[0], 400)
        self.assertEqual(self.jreq("POST", "/api/observations", {"draft_id": "0" * 32, "verification_status": "VERIFIED"})[0], 404)
        _, _, d = self.identify(size=(60, 40))
        self.assertEqual(self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "MAYBE"})[0], 400)
        self.assertEqual(self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "VERIFIED", "latitude": 5})[0], 400)
        self.assertEqual(self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "VERIFIED"})[0], 201)
        self.assertEqual(self.req("POST", "/api/observations", b"{bad", ctype="application/json")[0], 400)
        self.assertEqual(self.req("GET", "/nope")[0], 404)
        self.assertEqual(self.jreq("PATCH", "/api/observations/" + "0" * 32, {"notes": "x"})[0], 404)

    def test_field_features_over_http_with_network_blocked(self):
        with block_network() as attempts:
            photo = exif_jpeg(5.1789, 97.1421, "2026:10:05 07:30:00", "+07:00", size=(500, 400))
            s, r, d = self.req("POST", "/api/identify", photo, ctype="image/jpeg")
            d = json.loads(d)
            self.assertAlmostEqual(d["exif"]["latitude"], 5.1789, places=3)
            self.assertEqual(d["exif"]["captured_at"], "2026-10-05T07:30:00+07:00")
            tag = "fld" + d["draft_id"][:6]
            s, o = self.jreq("POST", "/api/observations", {"draft_id": d["draft_id"], "verification_status": "REJECTED",
                                                            "user_species": "rambutan", "notes": tag, "use_photo_location": True})
            self.assertEqual(s, 201)
            self.assertEqual((o["user_species"], o["human_label"], o["location_source"]), ("rambutan", "rambutan", "exif"))
            self.assertEqual(o["captured_at"], "2026-10-05T07:30:00+07:00")

            q = lambda qs: self.jreq("GET", "/api/observations?" + qs)[1]          # noqa: E731
            self.assertEqual([x["id"] for x in q(f"q={tag}")["items"]], [o["id"]])
            self.assertIn(o["id"], [x["id"] for x in q("has_location=true&limit=200")["items"]])
            self.assertIn(o["id"], [x["id"] for x in q("date_from=2026-10-05&date_to=2026-10-05&limit=200")["items"]])
            self.assertEqual(q("q=rambutan&has_location=false")["items"], [])

            s, o2 = self.jreq("PATCH", f"/api/observations/{o['id']}", {"latitude": 1.5, "longitude": 2.5, "location_source": "manual"})
            self.assertEqual((s, o2["latitude"], o2["location_source"]), (200, 1.5, "manual"))
            s, o3 = self.jreq("PATCH", f"/api/observations/{o['id']}", {"clear_location": True, "user_species": None})
            self.assertEqual((o3["latitude"], o3["user_species"], o3["location_source"]), (None, None, None))
            s, o4 = self.jreq("PATCH", f"/api/observations/{o['id']}", {"verification_status": "VERIFIED"})
            self.assertEqual(o4["human_label"], "Mangifera indica")
        self.assertEqual(attempts, [])

    def test_field_input_validation(self):
        _, _, d = self.identify(size=(60, 40))
        did = d["draft_id"]
        bad = [{"use_photo_location": "yes"}, {"use_photo_location": True},               # photo has no GPS
               {"latitude": "5"}, {"latitude": 5.0}, {"latitude": 91, "longitude": 0},
               {"latitude": 5.0, "longitude": 1.0, "location_source": "psychic"},
               {"user_species": "Citrus"}, {"user_species": 7}]
        for extra in bad:
            body = {"draft_id": did, "verification_status": "VERIFIED", **extra}
            if "user_species" in extra and extra["user_species"] == 7:
                body["verification_status"] = "REJECTED"
            self.assertEqual(self.jreq("POST", "/api/observations", body)[0], 400, extra)
        s, o = self.jreq("POST", "/api/observations", {"draft_id": did, "verification_status": "REJECTED"})   # draft survived all of it
        self.assertEqual(s, 201)
        oid = o["id"]
        for patch in ({"latitude": 5.0}, {"longitude": 5.0}, {"latitude": None, "longitude": None}, {"user_species": 5},
                      {"latitude": 5.0, "longitude": 5.0, "location_source": "x"}):
            self.assertEqual(self.jreq("PATCH", f"/api/observations/{oid}", patch)[0], 400, patch)
        for qs in ("date_from=5-10-2026", "has_location=maybe", "date_to=never"):
            self.assertEqual(self.jreq("GET", "/api/observations?" + qs)[0], 400, qs)

    def seed(self, tag, status="VERIFIED", lat=None, lon=None, rgb=(230, 20, 20), notes=None, **extra):
        _, _, d = self.identify(rgb, (200, 150))
        body = {"draft_id": d["draft_id"], "verification_status": status, "notes": notes or tag, **extra}
        if lat is not None:
            body.update(latitude=lat, longitude=lon, location_source="manual")
        s, o = self.jreq("POST", "/api/observations", body)
        self.assertEqual(s, 201, o)
        return o

    def test_stats_map_export_endpoints(self):
        tag = "eco" + uuid.uuid4().hex[:6]
        a = self.seed(tag, "VERIFIED", 5.18, 97.14, notes=f"=SUM({tag})")
        b = self.seed(tag, "REJECTED", 5.19, 97.15, rgb=(20, 20, 230), user_species="rambutan")
        c = self.seed(tag, "UNCERTAIN")
        with block_network() as attempts:
            s, st = self.jreq("GET", f"/api/stats?q={tag}")
            self.assertEqual((st["total"], st["with_location"], st["by_status"]["REJECTED"]), (3, 2, 1))
            self.assertEqual((st["field_agreement"]["k"], st["field_agreement"]["n"]), (1, 2))
            self.assertTrue(st["field_agreement"]["too_few"])
            self.assertEqual(st["corrections"][0]["actually"], "rambutan")

            s, mp = self.jreq("GET", f"/api/map?q={tag}")
            self.assertEqual(sorted(i["id"] for i in mp["items"]), sorted([a["id"], b["id"]]))   # only located ones
            self.assertEqual(set(mp["items"][0]), {"id", "latitude", "longitude", "verification_status", "predicted_species",
                                                   "user_species", "human_label", "confidence", "identification_status", "when"})
            self.assertEqual(self.jreq("GET", f"/api/map?q={tag}&status=REJECTED")[1]["total"], 1)

            s, r, body = self.req("GET", f"/api/export?format=csv&q={tag}")
            self.assertEqual(s, 200)
            self.assertIn("attachment; filename=\"ecoid-observations-", r.getheader("Content-Disposition"))
            self.assertTrue(r.getheader("Content-Type").startswith("text/csv"))
            rows = list(csv.DictReader(io.StringIO(body.decode("utf-8-sig"))))
            self.assertEqual(len(rows), 3)
            self.assertIn(f"'=SUM({tag})", [x["notes"] for x in rows])
            s, r, body = self.req("GET", f"/api/export?format=geojson&q={tag}")
            self.assertEqual(r.getheader("Content-Type"), "application/geo+json")
            self.assertEqual(len(json.loads(body)["features"]), 2)
            s, r, body = self.req("GET", f"/api/export?format=csv&q={tag}&include_location=false")
            self.assertNotIn("97.14", body.decode("utf-8-sig"))
            s, r, body = self.req("GET", f"/api/export?format=zip&q={tag}")
            self.assertEqual(r.getheader("Content-Type"), "application/zip")
            with zipfile.ZipFile(io.BytesIO(body)) as z:
                self.assertEqual(json.loads(z.read("manifest.json"))["count"], 3)
                self.assertEqual(len([n for n in z.namelist() if n.startswith("images/")]), 3)
        self.assertEqual(attempts, [])
        for qs in ("format=pdf", "format=csv&include_location=maybe", "format=geojson&include_location=false", "format=zip&date_from=x"):
            self.assertEqual(self.req("GET", "/api/export?" + qs)[0], 400, qs)
        self.assertEqual(self.req("GET", "/api/map?limit=abc")[0], 400)
        self.assertEqual(self.req("GET", "/api/stats?has_location=maybe")[0], 400)
        del c

    def test_basemap_endpoints(self):
        self.assertFalse(self.jreq("GET", "/api/basemap")[1]["available"])
        self.assertEqual(self.req("GET", "/basemap/image")[0], 404)
        (self.data / "bm.png").write_bytes(image_bytes(size=(64, 48), fmt="PNG"))
        cfg = self.data / "basemap.json"
        try:
            cfg.write_text(json.dumps({"image": "bm.png", "bounds": [5.0, 97.0, 5.4, 97.4], "attribution": "me"}), encoding="utf-8")
            s, b = self.jreq("GET", "/api/basemap")
            self.assertEqual((b["available"], b["bounds"], b["attribution"]), (True, [5.0, 97.0, 5.4, 97.4], "me"))
            s, r, body = self.req("GET", "/basemap/image")
            self.assertEqual((s, r.getheader("Content-Type")), (200, "image/png"))
            cfg.write_text(json.dumps({"image": "../escape.png", "bounds": [5.0, 97.0, 5.4, 97.4]}), encoding="utf-8")
            b = self.jreq("GET", "/api/basemap")[1]
            self.assertFalse(b["available"])
            self.assertIn("inside the data folder", b["error"])
            self.assertEqual(self.req("GET", "/basemap/image")[0], 404)
        finally:
            cfg.unlink(missing_ok=True)

    def test_oversized_upload_rejected_early(self):
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        c.putrequest("POST", "/api/identify")
        c.putheader("Content-Type", "image/jpeg")
        c.putheader("Content-Length", str(10 ** 9))
        c.endheaders()
        r = c.getresponse()
        r.read()
        self.assertEqual(r.status, 413)
        c.close()

    def test_foreign_host_header_is_refused(self):
        self.assertEqual(self.req("GET", "/api/model", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.req("GET", "/api/model", headers={"Host": f"localhost:{self.port}"})[0], 200)

    def test_path_traversal_and_unknown_ids(self):
        for p in ("/images/../../etc/passwd", "/images/" + "0" * 32, "/thumbs/..%2f..%2fetc%2fpasswd", "/images/zzz"):
            self.assertEqual(self.req("GET", p)[0], 404, p)

    def test_model_information_page_data(self):
        s, m = self.jreq("GET", "/api/model")
        self.assertEqual((m["model"], m["inference"], m["classes"]), ("EcoID Vision Model", "Local / CPU", "5 plant species"))
        self.assertTrue(m["runtime"].startswith("ONNX Runtime"))
        self.assertEqual(len(m["file_sha256"]), 64)

    def test_network_guard_really_blocks(self):
        import socket
        with block_network() as attempts:
            with self.assertRaises(OSError):
                socket.create_connection(("93.184.216.34", 80), timeout=2)
            with self.assertRaises(OSError):
                socket.getaddrinfo("example.com", 80)
        self.assertEqual(len(attempts), 2)

    def test_page_script_drives_real_server(self):
        """Runs the page's actual JavaScript (stub DOM, real HTTP) through capture -> verify -> history -> edit -> delete."""
        import os
        import shutil
        if not shutil.which("node"):
            if os.environ.get("CI"):
                self.fail("node is required in CI so the UI script test cannot be skipped silently")
            self.skipTest("node not installed")
        img = self.root / "plant.jpg"
        img.write_bytes(image_bytes((230, 20, 20), (400, 300)))
        (self.root / "gray.png").write_bytes(image_bytes((128, 128, 128), (200, 150), fmt="PNG"))
        r = subprocess.run(["node", str(ROOT / "tests" / "ui" / "smoke.js"), str(self.port), str(img),
                            str(STATIC_DIR / "index.html")], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["gray_status"], "LOW_CONFIDENCE")

    def test_page_is_self_contained(self):
        s, r, body = self.req("GET", "/")
        html = body.decode().replace("http://www.w3.org/2000/svg", "")     # an XML namespace identifier, never fetched
        self.assertEqual(s, 200)
        self.assertIn("default-src 'none'", r.getheader("Content-Security-Policy"))
        self.assertEqual(re.findall(r"(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}", html), [])   # no external hosts
        self.assertNotRegex(html, r"<script[^>]+src=|<link[^>]+href=|@import|url\(")          # nothing fetched by tag


class EntrypointTests(unittest.TestCase):
    def test_python_m_app_serves_and_is_reachable(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            model = Path(tmp) / "tiny.onnx"
            subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_test_onnx.py"), str(model)])
            p = subprocess.Popen([sys.executable, "-m", "app", "--model", str(model), "--data-dir", str(Path(tmp) / "d"),
                                  "--port", "0"], cwd=ROOT, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
            timer = threading.Timer(30, p.kill)
            timer.start()
            try:
                line = p.stdout.readline()
                m = re.search(r"http://127\.0\.0\.1:(\d+)", line)
                self.assertIsNotNone(m, line)
                c = http.client.HTTPConnection("127.0.0.1", int(m.group(1)), timeout=10)
                c.request("GET", "/api/model")
                resp = c.getresponse()
                resp.read()
                self.assertEqual(resp.status, 200)
                c.close()
            finally:
                timer.cancel()
                p.terminate()
                p.wait(10)
                p.stdout.close()

    def test_missing_model_fails_clearly(self):
        r = subprocess.run([sys.executable, "-m", "app", "--model", "/nope.onnx", "--port", "0"], cwd=ROOT,
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("Model file not found", r.stderr)


if __name__ == "__main__":
    unittest.main()
