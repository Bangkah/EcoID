"""Real-browser (Chromium via Playwright) end-to-end tests against the real server.

Opt-in: ECOID_BROWSER_TESTS=1  (needs `pip install playwright` + `playwright install chromium`).
If the flag is set but Playwright is missing, the run FAILS (never silently skipped).
Screenshots of every key state go to $ECOID_SCREENSHOTS (default: reports/screenshots).
"""
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import uuid
from pathlib import Path

from tests.helpers import image_bytes
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
            if page.evaluate(fn):
                return
            time.sleep(0.1)
        self.fail(f"timed out waiting for: {fn}")

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
        expect(page.locator("#s-saved")).to_be_visible()
        page.click("#btn-again")

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
