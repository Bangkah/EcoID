import re
import unittest

from app.ui.server import STATIC_DIR
from tests.ui_browser.selectors import IDS


class SelectorTests(unittest.TestCase):
    def test_every_id_used_by_browser_tests_exists_once(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        present = re.findall(r'\bid="([^"]+)"', html)
        for i in IDS:
            self.assertEqual(present.count(i), 1, f"id={i!r} must appear exactly once in index.html")

    def test_every_getelementbyid_in_the_script_exists(self):
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        present = set(re.findall(r'\bid="([^"]+)"', html))
        script = html.split("<script>")[1]
        used = set(re.findall(r'\$\("([^"]+)"\)', script)) | set(re.findall(r'show\("([^"]+)"', script))
        used = {i for i in used if not i.endswith("-")}          # "s-" prefix of step(): s-<name>, checked below
        used |= {"s-" + s for s in ("capture", "analyzing", "result", "saved")}
        self.assertEqual(sorted(i for i in used if i not in present), [])


if __name__ == "__main__":
    unittest.main()
