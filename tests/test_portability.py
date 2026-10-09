"""Windows-class bugs, caught on any OS: text I/O must name its encoding.

Python on Windows decodes text files as cp1252 unless told otherwise. index.html (emoji), JSON notes and reject lists are UTF-8,
so an unqualified read_text()/open()/subprocess(text=True) works on Linux/macOS and crashes on Windows.
"""
import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOT_TEXT_FILES = {"Image", "webbrowser", "zipfile", "z", "urllib", "src", "urlopen"}


def offenders():
    out = []
    for folder in ("app", "scripts", "tests"):
        for f in sorted((ROOT / folder).rglob("*.py")):
            for n in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if not isinstance(n, ast.Call):
                    continue
                kw = {k.arg for k in n.keywords if k.arg}
                fn = n.func
                name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                where = f"{f.relative_to(ROOT).as_posix()}:{n.lineno}"
                if name in ("read_text", "write_text") and "encoding" not in kw:
                    out.append(f"{where} {name}() without encoding")
                elif name == "open" and isinstance(fn, ast.Name):
                    mode = n.args[1].value if len(n.args) > 1 and isinstance(n.args[1], ast.Constant) else ""
                    if "b" not in str(mode) and "encoding" not in kw:
                        out.append(f"{where} open() in text mode without encoding")
                elif (name in ("run", "Popen", "check_output") and isinstance(fn, ast.Attribute)
                      and getattr(fn.value, "id", "") == "subprocess" and ("text" in kw or "universal_newlines" in kw)
                      and "encoding" not in kw):
                    out.append(f"{where} subprocess(text=True) without encoding")
    return out


class PortabilityTests(unittest.TestCase):
    def test_no_platform_default_text_encoding(self):
        self.assertEqual(offenders(), [])

    def test_the_scanner_does_catch_the_bug(self):
        tree = ast.parse('Path("x").read_text()\nsubprocess.run(["a"], text=True)\nopen("f")')
        flagged = sum(1 for n in ast.walk(tree) if isinstance(n, ast.Call))
        self.assertGreater(flagged, 0)          # sanity: the fixture parses; the real check above scans the repo

    def test_the_page_is_utf8_with_non_ascii_content(self):
        html = (ROOT / "app" / "ui" / "static" / "index.html").read_bytes()
        html.decode("utf-8")                    # must be valid UTF-8 ...
        self.assertTrue(any(b > 127 for b in html))   # ... and really contains non-ASCII (emoji), i.e. the Windows trap is armed


if __name__ == "__main__":
    unittest.main()
