import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.ai.config import InferenceConfig
from app.ai.inference.identifier import Identifier
from app.observation import export as ex
from app.observation.manager import ObservationManager
from app.storage.store import ObservationStore
from tests.helpers import FakeBackend, image_bytes

ROOT = Path(__file__).resolve().parents[2]


class ScriptTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.model = self.root / "tiny.onnx"
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_test_onnx.py"), str(self.model)])

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "scripts" / args[0]), *args[1:]], capture_output=True, text=True)

    def test_benchmark_reports_every_stage_and_the_nfr002_verdict(self):
        out = self.root / "rep"
        r = self.run_script("benchmark.py", "--model", str(self.model), "--n", "12", "--warmup", "2", "--out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads((out / "latency.json").read_text())
        for k in ("preprocess_ms", "inference_ms", "postprocess_ms", "total_ms"):
            self.assertEqual(res[k]["n"], 12)
            self.assertGreater(res[k]["median_ms"], 0)
        parts = sum(res[k]["average_ms"] for k in ("preprocess_ms", "inference_ms", "postprocess_ms"))
        self.assertAlmostEqual(parts, res["total_ms"]["average_ms"], delta=res["total_ms"]["average_ms"] * 0.2 + 0.5)
        self.assertTrue(res["meets_target"])                                     # a 15-parameter model is far under 10 s
        text = (out / "latency.txt").read_text()
        for needle in ("decode+resize", "ONNX inference", "TOTAL", "NFR-002", "MET", "synthetic"):
            self.assertIn(needle, text)

    def test_benchmark_with_an_impossible_target_says_not_met_and_real_images_are_used(self):
        imgs = self.root / "imgs"
        imgs.mkdir()
        for i in range(3):
            (imgs / f"{i}.jpg").write_bytes(image_bytes(size=(320, 240)))
        out = self.root / "rep2"
        r = self.run_script("benchmark.py", "--model", str(self.model), "--images", str(imgs), "--n", "6", "--warmup", "1",
                            "--target-s", "0.0000001", "--out", str(out))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("NOT MET", (out / "latency.txt").read_text())
        self.assertIn("3 real images", (out / "latency.txt").read_text())
        self.assertNotEqual(self.run_script("benchmark.py", "--model", str(self.model), "--images", str(self.root / "empty")).returncode, 0)

    def test_import_backup_cli_restores_into_a_fresh_data_dir(self):
        mgr = ObservationManager(Identifier(FakeBackend([8, 0, 0, 0, 0]), InferenceConfig("m", 0.65)), ObservationStore(self.root / "a"))
        o = mgr.save(mgr.identify(image_bytes(size=(120, 90))).id, "VERIFIED", "kept")
        buf = io.BytesIO()
        ex.build_backup(mgr.store, mgr.store.list_all(), buf)
        z = self.root / "b.zip"
        z.write_bytes(buf.getvalue())
        r = self.run_script("import_backup.py", str(z), "--app-data", str(self.root / "b"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn('"imported": 1', r.stdout)
        self.assertEqual(ObservationStore(self.root / "b").get(o.id).notes, "kept")
        r2 = self.run_script("import_backup.py", str(z), "--app-data", str(self.root / "b"))
        self.assertIn('"skipped_existing": 1', r2.stdout)


if __name__ == "__main__":
    unittest.main()
