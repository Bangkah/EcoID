"""Real-browser (Chromium via Playwright) end-to-end tests against the real server.

Opt-in: ECOID_BROWSER_TESTS=1  (needs `pip install playwright` + `playwright install chromium`).
If the flag is set but Playwright is missing, the run FAILS (never silently skipped).
Screenshots of every key state go to $ECOID_SCREENSHOTS (default: reports/screenshots).
"""
import http.client
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
import uuid
from pathlib import Path

from tests.helpers import exif_jpeg, image_bytes
from tests.ui_browser import selectors as S  # noqa: F401  (ids are validated by tests/ui/test_selectors.py)

try:
    from playwright.sync_api import expect, sync_playwright
    HAVE_PW = True
except ImportError:
    HAVE_PW = False

ROOT = Path(__file__).resolve().parents[2]
ENABLED = os.environ.get("ECOID_BROWSER_TESTS") == "1"
SHOTS = Path(os.environ.get("ECOID_SCREENSHOTS", ROOT / "reports" / "screenshots"))
STATUS_LABEL = {"VERIFIED": "Verified", "REJECTED": "Rejected", "UNCERTAIN": "Not sure"}


@unittest.skipUnless(ENABLED, "set ECOID_BROWSER_TESTS=1 (pip install playwright && playwright install chromium)")
class BrowserTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not HAVE_PW:
            raise RuntimeError("ECOID_BROWSER_TESTS=1 but the 'playwright' package is not installed")
        SHOTS.mkdir(parents=True, exist_ok=True)
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls.tmp.name)
        cls.data_dir = root / "data"
        model = root / "tiny.onnx"
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_test_onnx.py"), str(model)])
        cls.proc = subprocess.Popen([sys.executable, "-m", "app", "--model", str(model), "--data-dir", str(root / "data"),
                                     "--port", "0"], cwd=ROOT, stdout=subprocess.PIPE, text=True)
        cls._timer = threading.Timer(300, cls.proc.kill)
        cls._timer.start()
        m = re.search(r"(http://127\.0\.0\.1:\d+)", cls.proc.stdout.readline())
        if not m:
            raise RuntimeError("server did not start")
        cls.url = m.group(1)
        cls.pw = sync_playwright().start()
        cls.browser = cls.pw.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.pw.stop()
        cls._timer.cancel()
        cls.proc.terminate()
        cls.proc.wait(10)
        cls.proc.stdout.close()
        cls.tmp.cleanup()

    # ---------- helpers ----------
    def new_page(self, allow_console=(), **ctx):
        context = self.browser.new_context(**ctx)
        self.addCleanup(context.close)
        external, errors = [], []

        def gate(route):
            u = route.request.url
            if u.startswith(self.url) or u.startswith(("blob:", "data:")):
                route.continue_()
            else:
                external.append(u)
                route.abort()

        context.route("**/*", gate)
        page = context.new_page()
        page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
        page.on("console", lambda m: errors.append(f"console.error: {m.text}")
                if m.type == "error" and not any(re.search(a, m.text) for a in allow_console) else None)

        def check():
            self.assertEqual(external, [], "page tried to reach the internet")
            self.assertEqual(errors, [], "JavaScript errors in the page")

        self.addCleanup(check)
        page.goto(self.url + "/")
        return page

    def wait_js(self, page, fn, timeout=15.0):
        """Poll a JS function until truthy. (page.wait_for_function is unusable here: it needs 'unsafe-eval',
        which the app's Content-Security-Policy deliberately forbids.)"""
        end = time.time() + timeout
        while time.time() < end:
            try:
                if page.evaluate(fn):
                    return
            except Exception:                                   # DOM mid-redraw: try again
                pass
            time.sleep(0.1)
        self.fail(f"timed out waiting for: {fn}")

    @staticmethod
    def open_details(page, details_id):
        """<details> keeps its open/closed state between uses, so set it instead of toggling it."""
        page.evaluate("id => { document.getElementById(id).open = true; }", details_id)

    def shot(self, page, name):
        page.screenshot(path=str(SHOTS / f"{name}.png"), full_page=True)

    @staticmethod
    def upload(page, data, name="plant.jpg", mime="image/jpeg", input_id="in-file"):
        page.set_input_files(f"#{input_id}", files=[{"name": name, "mimeType": mime, "buffer": data}])

    def identify_and_save(self, page, rgb, status, notes, size=(640, 480)):
        self.upload(page, image_bytes(rgb, size))
        expect(page.locator("#s-result")).to_be_visible()
        page.fill("#res-notes", notes)
        page.click(f"#s-result .verify button[data-v={status}]")
        if status == "REJECTED":                                   # rejecting asks "what was it really?" first
            expect(page.locator("#fix-box")).to_be_visible()
            page.click("#btn-fix-save")
        expect(page.locator("#s-saved")).to_be_visible()
        page.click("#btn-again")

    def seed(self, tag, status="VERIFIED", lat=None, lon=None, rgb=(230, 20, 20), user_species=None, notes=None):
        """Create an observation straight through the HTTP API (fast); returns its JSON."""
        u = urllib.parse.urlparse(self.url)

        def call(method, path, body, ctype):
            c = http.client.HTTPConnection(u.hostname, u.port, timeout=20)
            c.request(method, path, body, {"Content-Type": ctype})
            r = c.getresponse()
            data = r.read()
            c.close()
            self.assertIn(r.status, (200, 201), data)
            return json.loads(data)

        d = call("POST", "/api/identify", image_bytes(rgb, (160, 120)), "image/jpeg")
        body = {"draft_id": d["draft_id"], "verification_status": status, "notes": notes or tag}
        if lat is not None:
            body.update(latitude=lat, longitude=lon, location_source="manual")
        if user_species:
            body["user_species"] = user_species
        return call("POST", "/api/observations", json.dumps(body).encode(), "application/json")

    def open_tab(self, page, name, tag=None):
        page.click(f"#tab-{name}")
        expect(page.locator(f"#view-{name}")).to_be_visible()
        if tag:
            page.fill("#q", tag)
            page.wait_for_timeout(600)       # the search box is debounced (300 ms); let its reload finish before interacting

    @staticmethod
    def marker_xs(page):
        return page.evaluate("() => [...document.querySelectorAll('#map-svg .marker')].map(e => Number(e.getAttribute('cx')))")

    def open_history(self, page):
        page.click("#tab-history")
        expect(page.locator("#view-history")).to_be_visible()

    # ---------- tests ----------
    def test_capture_result_verify_and_history_flow(self):
        page = self.new_page(viewport={"width": 1100, "height": 900})
        note = f"leaf-check-{uuid.uuid4().hex[:8]}"
        expect(page.locator("#s-capture")).to_be_visible()
        self.shot(page, "01_capture")

        self.upload(page, image_bytes((230, 20, 20), (1200, 900)))
        expect(page.locator("#s-result")).to_be_visible()
        expect(page.locator("#s-capture")).to_be_hidden()
        expect(page.locator("#res-cands .sci")).to_have_count(3)
        expect(page.locator("#res-cands .sci").first).to_have_text("Mangifera indica")
        expect(page.locator("#res-cands")).to_contain_text("Mangga")                 # Indonesian common name
        self.wait_js(page, "() => document.getElementById('res-img').naturalWidth > 0")  # photo really rendered
        ratio = page.locator("#res-cands .bar").first.evaluate(
            "b => b.firstElementChild.getBoundingClientRect().width / b.getBoundingClientRect().width")
        self.assertGreater(ratio, 0.5)                                                # confident bar is mostly filled
        expect(page.locator("#res-warn")).to_be_hidden()
        self.shot(page, "02_result_confident")

        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#s-saved")).to_be_visible()
        expect(page.locator("#s-saved")).to_contain_text("Saved")
        self.shot(page, "03_saved")
        page.click("#btn-again")
        expect(page.locator("#s-capture")).to_be_visible()

        self.open_history(page)
        item = page.locator("#hist-list .item", has_text=note)
        expect(item).to_have_count(1)
        expect(item).to_contain_text("Mangifera indica")
        expect(item.locator(".badge")).to_have_text("Verified")
        self.assertGreater(item.locator("img").evaluate(                              # thumbnail really loads
            "e => new Promise(r => e.complete ? r(e.naturalWidth) : e.addEventListener('load', () => r(e.naturalWidth)))"), 0)
        expect(page.locator("#hist-count")).to_have_text(re.compile(r"\d+ observations?"))
        self.shot(page, "04_history")

        item.click()                                                                   # detail dialog
        expect(page.locator("#dlg-detail")).to_have_attribute("open", "")
        self.wait_js(page, "() => document.getElementById('d-img').naturalWidth > 0")
        expect(page.locator("#d-notes")).to_have_value(note)
        self.shot(page, "05_detail")
        page.click("#d-verify button[data-v=REJECTED]")
        expect(page.locator("#d-sub")).to_contain_text("Rejected")
        page.fill("#d-notes", note + " (changed)")
        page.click("#d-save")
        page.click("#d-close")
        expect(page.locator("#dlg-detail")).not_to_have_attribute("open", "")
        changed = page.locator("#hist-list .item", has_text=note + " (changed)")
        expect(changed).to_have_count(1)
        expect(changed.locator(".badge")).to_have_text("Rejected")                     # edits persisted server-side

        messages = []
        page.once("dialog", lambda d: (messages.append(d.message), d.accept()))        # confirm() before delete
        changed.click()
        page.click("#d-delete")
        expect(page.locator("#hist-list .item", has_text=note)).to_have_count(0)
        self.assertEqual(len(messages), 1)
        self.assertIn("cannot be undone", messages[0])

    def test_low_confidence_banner_and_not_sure(self):
        page = self.new_page(viewport={"width": 1100, "height": 900})
        note = f"blurry-{uuid.uuid4().hex[:8]}"
        self.upload(page, image_bytes((128, 128, 128), (300, 200), "PNG"), "gray.png", "image/png")
        expect(page.locator("#s-result")).to_be_visible()
        expect(page.locator("#res-warn")).to_be_visible()
        expect(page.locator("#res-warn")).to_contain_text("Low confidence")
        expect(page.locator("#res-warn")).to_contain_text("verify manually")
        self.shot(page, "06_result_low_confidence")
        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=UNCERTAIN]")
        expect(page.locator("#s-saved")).to_be_visible()
        page.click("#btn-again")
        self.open_history(page)
        item = page.locator("#hist-list .item", has_text=note)
        expect(item).to_contain_text("low confidence")
        expect(item.locator(".badge")).to_have_text("Not sure")

    def test_each_verification_choice_saves_and_filter_works(self):
        page = self.new_page(viewport={"width": 1100, "height": 900})
        tag = uuid.uuid4().hex[:6]
        for status, rgb in (("VERIFIED", (20, 230, 20)), ("REJECTED", (20, 20, 230)), ("UNCERTAIN", (230, 20, 20))):
            self.identify_and_save(page, rgb, status, f"{status}-{tag}", (200, 150))
        self.open_history(page)
        for status, label in STATUS_LABEL.items():
            page.select_option("#flt", status)
            for other in STATUS_LABEL:                                                  # wait until the list was re-filtered
                expect(page.locator("#hist-list .item", has_text=f"{other}-{tag}")).to_have_count(1 if other == status else 0)
            badges = page.locator("#hist-list .item .badge")
            expect(badges.first).to_be_visible()
            texts = set(badges.all_inner_texts())
            self.assertEqual(texts, {label}, f"filter {status} leaked other statuses: {texts}")
        page.select_option("#flt", "")
        expect(page.locator("#hist-list .item", has_text=f"-{tag}")).to_have_count(3)
        self.shot(page, "07_history_all_statuses")

    # ---------------- Phase 4: field observation ----------------
    def test_photo_location_is_opt_in_then_saved_and_shown(self):
        page = self.new_page(viewport={"width": 1000, "height": 900})
        note = f"exif-{uuid.uuid4().hex[:6]}"
        self.upload(page, exif_jpeg(3.5948, 98.6722, "2026:10:05 07:30:00", size=(600, 450)))
        expect(page.locator("#s-result")).to_be_visible()
        expect(page.locator("#loc-status")).to_have_text("No location will be saved.")     # nothing attached by default
        page.click("#loc-box summary")
        expect(page.locator("#loc-exif")).to_be_visible()
        expect(page.locator("#loc-exif-text")).to_contain_text("3.5948")
        self.shot(page, "14_location_found_in_photo")
        page.click("#loc-use-exif")
        expect(page.locator("#loc-status")).to_contain_text("from photo")
        expect(page.locator("#loc-lat")).to_have_value("3.5948")
        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#s-saved")).to_be_visible()
        page.click("#btn-again")
        self.open_history(page)
        item = page.locator("#hist-list .item", has_text=note)
        expect(item.locator(".loc")).to_have_text("📍 3.5948, 98.6722")
        expect(item).to_contain_text("2026")                                              # shows the capture date, not the save date
        item.click()
        page.click("#d-loc-box summary")
        expect(page.locator("#d-loc-status")).to_contain_text("from photo")
        self.shot(page, "15_detail_with_location")

    def test_without_opt_in_no_location_is_stored(self):
        page = self.new_page()
        note = f"noloc-{uuid.uuid4().hex[:6]}"
        self.upload(page, exif_jpeg(3.5948, 98.6722, "2026:10:05 07:30:00"))
        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#s-saved")).to_be_visible()
        got = page.request.get(self.url + "/api/observations?q=" + note).json()["items"][0]
        self.assertIsNone(got["latitude"])
        self.assertTrue(got["captured_at"])                                               # the time is kept, the place is not

    def test_always_use_photo_location_is_remembered(self):
        page = self.new_page()
        self.upload(page, exif_jpeg(3.5948, 98.6722))
        page.click("#loc-box summary")
        page.check("#loc-always")
        page.click("#btn-discard")
        self.upload(page, exif_jpeg(-6.2, 106.8))
        expect(page.locator("#s-result")).to_be_visible()
        expect(page.locator("#loc-status")).to_contain_text("from photo")
        expect(page.locator("#loc-lat")).to_have_value("-6.2")

    def test_current_location_button_uses_the_browsers_geolocation(self):
        page = self.new_page(permissions=["geolocation"], viewport={"width": 1000, "height": 900},
                             geolocation={"latitude": 5.1801, "longitude": 97.1507, "accuracy": 25})
        note = f"gps-{uuid.uuid4().hex[:6]}"
        self.upload(page, image_bytes((20, 230, 20), (300, 200)))
        page.click("#loc-box summary")
        page.click("#loc-device")
        expect(page.locator("#loc-status")).to_contain_text("from device")
        expect(page.locator("#loc-status")).to_contain_text("±25 m")
        expect(page.locator("#loc-lat")).to_have_value("5.1801")
        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#s-saved")).to_be_visible()
        got = page.request.get(self.url + "/api/observations?q=" + note).json()["items"][0]
        self.assertEqual((got["latitude"], got["longitude"], got["location_source"], got["location_accuracy_m"]),
                         (5.1801, 97.1507, "device", 25.0))

    def test_current_location_denied_shows_a_message_and_saves_nothing(self):
        page = self.new_page()
        self.upload(page, image_bytes(size=(200, 150)))
        page.click("#loc-box summary")
        page.click("#loc-device")
        expect(page.locator("#loc-err")).to_contain_text("Could not get your position")
        expect(page.locator("#loc-status")).to_have_text("No location will be saved.")

    def test_typed_location_is_validated_and_accepts_decimal_comma(self):
        page = self.new_page()
        note = f"typed-{uuid.uuid4().hex[:6]}"
        self.upload(page, image_bytes((20, 230, 20), (300, 200)))
        page.click("#loc-box summary")
        page.fill("#res-notes", note)
        page.fill("#loc-lat", "95")
        page.fill("#loc-lon", "10")
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#loc-err")).to_contain_text("within")
        expect(page.locator("#s-result")).to_be_visible()                                 # not saved
        page.fill("#loc-lon", "")
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#loc-err")).to_contain_text("both")
        page.fill("#loc-lat", "5,18")
        page.fill("#loc-lon", "97.14")
        page.click("#s-result .verify button[data-v=VERIFIED]")
        expect(page.locator("#s-saved")).to_be_visible()
        got = page.request.get(self.url + "/api/observations?q=" + note).json()["items"][0]
        self.assertEqual((got["latitude"], got["longitude"], got["location_source"]), (5.18, 97.14, "manual"))

    def test_rejecting_asks_what_it_was_and_the_correction_is_editable(self):
        page = self.new_page(viewport={"width": 1000, "height": 1000})
        note = f"rej-{uuid.uuid4().hex[:6]}"
        self.upload(page, image_bytes((230, 20, 20), (300, 200)))
        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=REJECTED]")
        expect(page.locator("#fix-box")).to_be_visible()
        expect(page.locator("#s-result")).to_be_visible()                                 # nothing saved yet
        self.shot(page, "16_rejecting_asks_what_it_was")
        page.select_option("#fix-select", "__other__")
        expect(page.locator("#fix-other")).to_be_visible()
        page.fill("#fix-other", "Rambutan")
        page.click("#btn-fix-save")
        expect(page.locator("#s-saved")).to_be_visible()
        page.click("#btn-again")
        self.open_history(page)
        item = page.locator("#hist-list .item", has_text=note)
        expect(item.locator(".fixed")).to_have_text("Actually: Rambutan")
        item.click()
        expect(page.locator("#d-fix-wrap")).to_be_visible()
        expect(page.locator("#d-fix-select")).to_have_value("__other__")
        expect(page.locator("#d-fix-other")).to_have_value("Rambutan")
        page.select_option("#d-fix-select", "Mangifera indica")                           # change to one of the 5 classes
        page.click("#d-save")
        page.click("#d-close")
        expect(page.locator("#hist-list .item", has_text=note).locator(".fixed")).to_have_text("Actually: Mangga (Mangifera indica)")
        page.locator("#hist-list .item", has_text=note).click()
        page.click("#d-verify button[data-v=VERIFIED]")                                    # agreeing drops the correction
        expect(page.locator("#d-fix-wrap")).to_be_hidden()
        page.click("#d-close")
        expect(page.locator("#hist-list .item", has_text=note).locator(".fixed")).to_have_count(0)

    def test_rejecting_can_be_cancelled_and_skipped(self):
        page = self.new_page()
        self.upload(page, image_bytes(size=(200, 150)))
        page.click("#s-result .verify button[data-v=REJECTED]")
        page.click("#btn-fix-cancel")
        expect(page.locator("#fix-box")).to_be_hidden()
        expect(page.locator("#s-result")).to_be_visible()
        page.click("#s-result .verify button[data-v=REJECTED]")
        page.click("#btn-fix-save")                                                        # "I don't know what it is"
        expect(page.locator("#s-saved")).to_be_visible()

    def test_history_search_and_location_filter(self):
        page = self.new_page(viewport={"width": 1000, "height": 900})
        tag = uuid.uuid4().hex[:6]
        self.upload(page, exif_jpeg(3.5, 98.6, rgb=(230, 20, 20), size=(300, 200)))
        page.click("#loc-box summary")
        page.click("#loc-use-exif")
        page.fill("#res-notes", f"needle-{tag} by the river")
        page.click("#s-result .verify button[data-v=VERIFIED]")
        page.click("#btn-again")
        self.identify_and_save(page, (20, 20, 230), "UNCERTAIN", f"haystack-{tag}", (200, 150))
        self.open_history(page)
        items = page.locator("#hist-list .item")
        page.fill("#q", f"needle-{tag}")
        expect(items).to_have_count(1)
        expect(items.first).to_contain_text("by the river")
        page.fill("#q", f"-{tag}")
        expect(items).to_have_count(2)
        page.check("#has-loc")
        expect(items).to_have_count(1)
        expect(items.first).to_contain_text(f"needle-{tag}")
        page.uncheck("#has-loc")
        page.fill("#q", "Mangga")                                                           # Indonesian common name
        expect(items.first).to_contain_text("Mangifera indica")
        page.fill("#q", f"no-such-{tag}")
        expect(items).to_have_count(0)
        expect(page.locator("#hist-empty")).to_be_hidden()                                 # filtered-empty is not "no history yet"
        self.shot(page, "17_history_search")

    def test_edit_and_clear_location_in_detail(self):
        page = self.new_page(viewport={"width": 1000, "height": 1000})
        note = f"edit-{uuid.uuid4().hex[:6]}"
        self.identify_and_save(page, (230, 20, 20), "VERIFIED", note, (200, 150))
        self.open_history(page)
        page.locator("#hist-list .item", has_text=note).click()
        page.click("#d-loc-box summary")
        expect(page.locator("#d-loc-status")).to_have_text("No location will be saved.")
        page.fill("#d-loc-lat", "5.2")
        page.fill("#d-loc-lon", "97.1")
        page.click("#d-save")
        page.click("#d-close")
        expect(page.locator("#hist-list .item", has_text=note).locator(".loc")).to_have_text("📍 5.2, 97.1")
        page.locator("#hist-list .item", has_text=note).click()
        self.open_details(page, "d-loc-box")
        page.click("#d-loc-clear")
        page.click("#d-save")
        page.click("#d-close")
        expect(page.locator("#hist-list .item", has_text=note).locator(".loc")).to_have_count(0)

    # ---------------- Phase 5: Eco Mapper ----------------
    def test_map_markers_colours_popup_zoom_pan_and_details(self):
        page = self.new_page(viewport={"width": 1000, "height": 900})
        tag = f"map{uuid.uuid4().hex[:6]}"
        self.seed(tag, "VERIFIED", 5.1800, 97.1400)
        self.seed(tag, "REJECTED", 5.1900, 97.1500, rgb=(20, 20, 230))
        self.seed(tag, "UNCERTAIN", 5.2500, 97.2000, rgb=(20, 230, 20))
        self.seed(tag, "VERIFIED")                                                       # no location: not on the map
        self.open_tab(page, "map", tag)
        expect(page.locator("#hist-count")).to_have_text("3 with a location")              # the FILTERED response has arrived
        markers = page.locator("#map-svg .marker")
        expect(markers).to_have_count(3)
        self.assertEqual(page.locator("#map-svg .marker[data-status=VERIFIED]").get_attribute("fill"), "#2f9e5f")
        self.assertEqual(page.locator("#map-svg .marker[data-status=REJECTED]").get_attribute("fill"), "#d0453a")
        self.assertEqual(page.locator("#map-svg .marker[data-status=UNCERTAIN]").get_attribute("fill"), "#c9a227")
        expect(page.locator("#map-legend span")).to_have_count(3)
        expect(page.locator("#map-scale")).to_contain_text(re.compile(r"\d+ k?m"))
        expect(page.locator("#hist-count")).to_have_text("3 with a location")
        self.shot(page, "18_map")

        page.locator("#map-svg .marker[data-status=VERIFIED]").click()                   # popup -> details
        expect(page.locator("#map-popup")).to_be_visible()
        expect(page.locator("#map-popup")).to_contain_text("Mangifera indica")
        expect(page.locator("#map-popup")).to_contain_text("Verified")
        self.shot(page, "19_map_popup")
        page.click("#map-popup .popup-open")
        expect(page.locator("#dlg-detail")).to_have_attribute("open", "")
        expect(page.locator("#d-notes")).to_have_value(tag)
        page.click("#d-close")

        xs = sorted(self.marker_xs(page))
        spread, scale = xs[-1] - xs[0], page.locator("#map-scale").inner_text()
        page.click("#map-zin")
        self.wait_js(page, f"() => {{ const x = [...document.querySelectorAll('#map-svg .marker')].map(e => +e.getAttribute('cx')); return x.length > 1 ? Math.max(...x) - Math.min(...x) > {spread} * 1.5 : true; }}")
        self.assertNotEqual(page.locator("#map-scale").inner_text(), scale)               # scale bar follows the zoom
        page.click("#map-fit")
        box = page.locator("#map-svg").bounding_box()
        before = self.marker_xs(page)[0]
        page.mouse.move(box["x"] + 25, box["y"] + 25)                                      # drag on empty background
        page.mouse.down()
        page.mouse.move(box["x"] + 85, box["y"] + 25, steps=6)
        page.mouse.up()
        self.wait_js(page, f"() => Math.abs(+document.querySelector('#map-svg .marker').getAttribute('cx') - {before} - 60) < 2")
        scale2 = page.locator("#map-scale").inner_text()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)         # wheel zoom (two levels in)
        page.mouse.wheel(0, -800)
        self.wait_js(page, f"() => document.getElementById('map-scale').innerText !== {json.dumps(scale2)}")

    def test_map_color_by_species_and_legend(self):
        page = self.new_page(viewport={"width": 1000, "height": 800})
        tag = f"spc{uuid.uuid4().hex[:6]}"
        self.seed(tag, "VERIFIED", 5.18, 97.14)
        self.seed(tag, "REJECTED", 5.20, 97.16, user_species="Rambutan")
        self.open_tab(page, "map", tag)
        expect(page.locator("#hist-count")).to_have_text("2 with a location")
        expect(page.locator("#map-svg .marker")).to_have_count(2)
        page.select_option("#map-color", "species")
        fills = set(page.evaluate("() => [...document.querySelectorAll('#map-svg .marker')].map(e => e.getAttribute('fill'))"))
        self.assertEqual(fills, {"#e08a1e", "#7a8790"})                                    # mango colour + "other"
        expect(page.locator("#map-legend")).to_contain_text("Mangga")
        expect(page.locator("#map-legend")).to_contain_text("other")

    def test_map_clusters_nearby_points_then_lists_them(self):
        page = self.new_page(viewport={"width": 1000, "height": 800})
        tag = f"clu{uuid.uuid4().hex[:6]}"
        for i in range(4):
            self.seed(tag, "VERIFIED", 5.18 + i * 1e-5, 97.14)                             # ~1 m apart
        self.seed(tag, "REJECTED", 5.22, 97.18)                                            # a far one, so the view starts wide
        self.open_tab(page, "map", tag)
        expect(page.locator("#hist-count")).to_have_text("5 with a location")
        cluster = page.locator("#map-svg .cluster")
        expect(cluster).to_have_count(1)
        expect(cluster).to_have_attribute("data-count", "4")
        expect(page.locator("#map-svg .marker")).to_have_count(1)                          # the far one stays a single marker
        scale = page.locator("#map-scale").inner_text()
        cluster.click()                                                                    # zooms toward the cluster
        self.wait_js(page, f"() => document.getElementById('map-scale').innerText !== {json.dumps(scale)}")
        page.locator("#map-svg .cluster").click()                                          # same spot at max zoom: lists them
        expect(page.locator("#map-popup .popup-open")).to_have_count(4)
        self.shot(page, "20_map_cluster_list")

    def test_map_empty_state_and_offline_basemap(self):
        page = self.new_page(viewport={"width": 1000, "height": 800})
        self.open_tab(page, "map", f"nothing-{uuid.uuid4().hex[:6]}")
        expect(page.locator("#map-empty")).to_be_visible()
        tag = f"bm{uuid.uuid4().hex[:6]}"
        self.seed(tag, "VERIFIED", 5.20, 97.15)
        cfg = self.data_dir / "basemap.json"
        (self.data_dir / "bm.png").write_bytes(image_bytes((200, 210, 190), (200, 200), "PNG"))
        cfg.write_text(json.dumps({"image": "bm.png", "bounds": [5.15, 97.10, 5.25, 97.20], "attribution": "test basemap"}))
        try:
            page2 = self.new_page(viewport={"width": 1000, "height": 800})
            self.open_tab(page2, "map", tag)
            expect(page2.locator("#hist-count")).to_have_text("1 with a location")
            image = page2.locator("#map-svg image.basemap")
            expect(image).to_have_count(1)
            expect(page2.locator("#map-attrib")).to_have_text("test basemap")
            img, mk = image.bounding_box(), page2.locator("#map-svg .marker").bounding_box()
            self.assertTrue(img["x"] <= mk["x"] <= img["x"] + img["width"] and img["y"] <= mk["y"] <= img["y"] + img["height"],
                            "marker must lie inside the georeferenced picture")
            self.shot(page2, "21_map_basemap")
        finally:
            cfg.unlink(missing_ok=True)

    def test_stats_tab_agreement_confidence_corrections_and_small_n_flag(self):
        page = self.new_page(viewport={"width": 1000, "height": 1100})
        tag = f"st{uuid.uuid4().hex[:6]}"
        for _ in range(8):
            self.seed(tag, "VERIFIED")
        for _ in range(4):
            self.seed(tag, "REJECTED", user_species="Rambutan")
        self.open_tab(page, "stats", tag)
        expect(page.locator("#st-agree-rate")).to_have_text("67%")
        expect(page.locator("#st-agreement")).to_contain_text("8 of 12 decisions")
        expect(page.locator("#st-agreement .flag")).to_have_count(0)                         # n >= 10: not flagged
        body = page.locator("#stats-body")
        for text in ("Observations", "Confidence vs. what you decided", "Rambutan", "Activity", "Mangga"):
            expect(body).to_contain_text(text)
        self.shot(page, "22_stats")
        small = f"sm{uuid.uuid4().hex[:6]}"
        self.seed(small, "VERIFIED")
        self.seed(small, "REJECTED")
        page.fill("#q", small)
        expect(page.locator("#st-agree-rate")).to_have_text("50%")
        expect(page.locator("#st-agreement .flag")).to_contain_text("too few")               # honest about small samples
        page.fill("#q", f"none-{uuid.uuid4().hex[:6]}")
        expect(page.locator("#stats-body")).to_contain_text("No observations match")

    def read_download(self, page, fmt, include_location=True):
        page.click("#btn-export")
        expect(page.locator("#dlg-export")).to_have_attribute("open", "")
        page.select_option("#x-format", fmt)
        if not include_location:
            page.uncheck("#x-loc")
        with page.expect_download() as info:
            page.click("#x-go")
        dl = info.value
        self.assertTrue(dl.suggested_filename.startswith("ecoid-"), dl.suggested_filename)
        data = Path(dl.path()).read_bytes()
        page.click("#x-close")
        return dl.suggested_filename, data

    def test_export_dialog_downloads_csv_geojson_zip_and_respects_privacy_options(self):
        import io
        import zipfile
        page = self.new_page(viewport={"width": 1000, "height": 900})
        tag = f"ex{uuid.uuid4().hex[:6]}"
        self.seed(tag, "VERIFIED", 5.1234, 97.5678, notes=f"=SUM({tag})")
        self.seed(tag, "REJECTED")
        self.open_tab(page, "history", tag)
        expect(page.locator("#hist-list .item")).to_have_count(2)
        page.click("#btn-export")
        expect(page.locator("#x-scope")).to_contain_text(tag)                                 # says what will be exported
        page.click("#x-close")

        name, data = self.read_download(page, "csv")
        self.assertTrue(name.endswith(".csv"))
        text = data.decode("utf-8-sig")
        self.assertIn("5.1234", text)
        self.assertIn(f"'=SUM({tag})", text)
        self.assertEqual(text.count("\n") >= 3, True)
        name, data = self.read_download(page, "csv", include_location=False)
        self.assertNotIn("5.1234", data.decode("utf-8-sig"))
        name, data = self.read_download(page, "geojson")
        g = json.loads(data)
        self.assertEqual(g["features"][0]["geometry"]["coordinates"], [97.5678, 5.1234])     # lon, lat
        page.click("#btn-export")
        page.select_option("#x-format", "geojson")
        expect(page.locator("#x-loc")).to_be_disabled()                                        # GeoJSON without places is meaningless
        page.click("#x-close")
        name, data = self.read_download(page, "zip")
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            self.assertEqual(json.loads(z.read("manifest.json"))["count"], 2)
            self.assertEqual(len([n for n in z.namelist() if n.startswith("images/")]), 2)

    def test_discard_leaves_no_observation(self):
        page = self.new_page()
        self.upload(page, image_bytes(size=(200, 150)))
        expect(page.locator("#s-result")).to_be_visible()
        before = page.request.get(self.url + "/api/observations?limit=1").json()["total"]
        page.click("#btn-discard")
        expect(page.locator("#s-capture")).to_be_visible()
        self.assertEqual(page.request.get(self.url + "/api/observations?limit=1").json()["total"], before)

    def test_unreadable_file_shows_error_and_stays_on_capture(self):
        page = self.new_page(allow_console=[r"status of 400"])     # the browser logs the server's expected 400
        self.upload(page, b"this is not an image", "fake.jpg")
        expect(page.locator("#capture-err")).to_be_visible()
        expect(page.locator("#s-capture")).to_be_visible()
        expect(page.locator("#s-result")).to_be_hidden()
        self.shot(page, "08_error")

    def test_model_information_dialog(self):
        page = self.new_page(viewport={"width": 1100, "height": 700})
        page.click("#btn-about")
        expect(page.locator("#dlg-about")).to_have_attribute("open", "")
        body = page.locator("#about-body")
        for text in ("EcoID Vision Model", "ONNX Runtime", "Local / CPU", "5 plant species", "sha256"):
            expect(body).to_contain_text(text)
        self.shot(page, "09_about")
        page.click("#about-close")
        expect(page.locator("#dlg-about")).not_to_have_attribute("open", "")

    def test_phone_viewport_has_no_horizontal_scroll_and_buttons_fit(self):
        page = self.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2, is_mobile=True, has_touch=True)
        note = f"mobile-{uuid.uuid4().hex[:6]}"
        self.upload(page, image_bytes((230, 20, 20), (900, 600)))
        expect(page.locator("#s-result")).to_be_visible()
        self.shot(page, "10_mobile_result")
        for b in page.locator("#s-result .verify button").all():
            box = b.bounding_box()
            self.assertGreaterEqual(box["x"], 0)
            self.assertLessEqual(box["x"] + box["width"], 391)
            self.assertGreaterEqual(box["height"], 40)                                  # comfortably tappable
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), page.evaluate("window.innerWidth"))
        page.fill("#res-notes", note)
        page.click("#s-result .verify button[data-v=VERIFIED]")
        page.click("#btn-again")
        self.open_history(page)
        page.locator("#hist-list .item", has_text=note).click()
        expect(page.locator("#dlg-detail")).to_have_attribute("open", "")
        box = page.locator("#dlg-detail").bounding_box()
        self.assertLessEqual(box["x"] + box["width"], 391)
        self.shot(page, "11_mobile_detail")
        self.assertLessEqual(page.evaluate("document.documentElement.scrollWidth"), page.evaluate("window.innerWidth"))

    def test_dark_and_light_color_schemes(self):
        colors = {}
        for scheme in ("light", "dark"):
            page = self.new_page(color_scheme=scheme, viewport={"width": 900, "height": 700})
            colors[scheme] = page.evaluate("getComputedStyle(document.body).backgroundColor")
            self.shot(page, f"12_{scheme}")
        self.assertEqual(colors["light"], "rgb(246, 247, 244)")
        self.assertEqual(colors["dark"], "rgb(20, 24, 21)")

    def test_webcam_capture_with_chromium_fake_camera(self):
        """Covers the getUserMedia code path with Chromium's synthetic camera. A physical camera / phone is still manual."""
        page = self.new_page(permissions=["camera"], viewport={"width": 900, "height": 800})
        page.click("#btn-webcam")
        expect(page.locator("#webcam")).to_be_visible()
        self.wait_js(page, "() => document.getElementById('video').videoWidth > 0")
        self.shot(page, "13_webcam")
        page.click("#btn-snap")
        expect(page.locator("#s-result")).to_be_visible(timeout=15000)
        expect(page.locator("#res-cands .sci")).to_have_count(3)
        expect(page.locator("#webcam")).to_be_hidden()


if __name__ == "__main__":
    unittest.main()
