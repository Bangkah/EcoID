import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.config import InferenceConfig
from app.ai.data.checks import CLASS_DIRS
from app.ai.evaluation import metrics
from app.ai.evaluation.runner import evaluate, format_report
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier

ROOT = Path(__file__).resolve().parents[2]
RNG = np.random.default_rng(7)


def colored(path: Path, rgb, jitter=10):
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.clip(np.array(rgb) + RNG.integers(-jitter, jitter + 1, (48, 48, 3)), 0, 255).astype(np.uint8)
    Image.fromarray(a).save(path, "PNG")


class MetricsTests(unittest.TestCase):
    def setUp(self):
        # 4 positives over 3 classes; sample 3 is wrong; confidences .9 .8 .7 .5
        self.p = np.array([[.9, .05, .05], [.1, .8, .1], [.1, .2, .7], [.5, .3, .2]])
        self.y = np.array([0, 1, 1, 0])

    def test_topk(self):
        self.assertEqual(metrics.topk_accuracy(self.p, self.y, 1), 0.75)
        self.assertEqual(metrics.topk_accuracy(self.p, self.y, 2), 1.0)

    def test_confusion(self):
        m = metrics.confusion_matrix(self.p, self.y, 3)
        self.assertEqual(m.sum(), 4)
        self.assertEqual(m[1, 2], 1)

    def test_operating_point(self):
        neg = np.array([[.4, .3, .3], [.9, .05, .05]])
        op = metrics.operating_point(self.p, self.y, neg, 0.65)
        self.assertEqual(op["coverage"], 0.75)              # .9 .8 .7 accepted, .5 rejected
        self.assertAlmostEqual(op["selective_accuracy"], 2 / 3)
        self.assertEqual(op["negatives_rejected"], 1)
        self.assertEqual(op["negative_rejection_rate"], 0.5)

    def test_threshold_boundary_is_inclusive(self):
        op = metrics.operating_point(np.array([[.65, .35]]), np.array([0]), None, 0.65)
        self.assertEqual(op["coverage"], 1.0)

    def test_pick_threshold_and_infeasible(self):
        neg = np.array([[.4, .3, .3], [.45, .3, .25]])
        sweep = metrics.threshold_sweep(self.p, self.y, neg)
        best = metrics.pick_threshold(sweep, min_negative_rejection=1.0, min_selective_accuracy=0.6)
        self.assertIsNotNone(best)
        self.assertGreaterEqual(best["negative_rejection_rate"], 1.0)
        self.assertIsNone(metrics.pick_threshold(sweep, min_negative_rejection=1.0, min_selective_accuracy=1.01))

    def test_ece_perfectly_calibrated_is_small_and_overconfident_is_large(self):
        good = np.array([[1.0, 0.0]] * 5)
        self.assertAlmostEqual(metrics.expected_calibration_error(good, np.zeros(5, int))[0], 0.0)
        over = np.array([[0.99, 0.01]] * 4)
        self.assertGreater(metrics.expected_calibration_error(over, np.array([0, 1, 1, 1]))[0], 0.7)

    def test_latency_stats(self):
        s = metrics.latency_stats([10, 20, 30])
        self.assertEqual((s["average_ms"], s["median_ms"], s["n"]), (20.0, 20.0, 3))


class EndToEndEvalTests(unittest.TestCase):
    """Real pipeline + real ONNX Runtime with the synthetic model (plumbing test, NOT model quality)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls.tmp.name)
        cls.model = root / "tiny.onnx"
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_test_onnx.py"), str(cls.model)])
        colors = [(230, 20, 20), (20, 230, 20), (20, 20, 230)]    # classes 0..2 by dominant channel
        for ci, rgb in enumerate(colors):
            for i in range(10):
                colored(root / "eval" / CLASS_DIRS[ci] / f"{i}.png", rgb)
        for i in range(8):
            colored(root / "neg" / "non_plant" / f"{i}.png", (128, 128, 128))   # ambiguous -> low confidence
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_evaluate_and_report(self):
        ident = Identifier(OnnxBackend(self.model), InferenceConfig("Test Model", 0.65))
        res = evaluate(ident, self.root / "eval", self.root / "neg")
        self.assertEqual(res["eval_samples"], 30)
        self.assertEqual(res["negative_samples"], 8)
        self.assertEqual(res["top1_accuracy"], 1.0)
        self.assertEqual(res["top3_accuracy"], 1.0)
        self.assertEqual(res["operating_point"]["negative_rejection_rate"], 1.0)
        self.assertEqual(res["identified_count"], 30)          # production decision path agrees
        self.assertGreater(res["latency"]["n"], 0)
        text = format_report(res)
        for needle in ("Top-1 Accuracy: 100.0%", "Top-3 Accuracy: 100.0%", "Average inference latency",
                       "Median inference latency", "Negative samples: 8", "Correctly rejected: 8",
                       "Selective accuracy"):
            self.assertIn(needle, text)

    def test_scripts_run_and_write_reports(self):
        out = Path(self.tmp.name) / "reports"
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "evaluate.py"), "--model", str(self.model),
                               "--eval-dir", str(self.root / "eval"), "--negative-dir", str(self.root / "neg"),
                               "--out", str(out)], stdout=subprocess.DEVNULL)
        data = json.loads((out / "benchmark.json").read_text())
        self.assertEqual(len(data["model_sha256"]), 64)
        self.assertTrue((out / "benchmark.txt").read_text().startswith("EcoID Vision Model"))
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "calibrate_threshold.py"),
                               "--model", str(self.model), "--val-dir", str(self.root / "eval"),
                               "--negative-val-dir", str(self.root / "neg"), "--out", str(out)],
                              stdout=subprocess.DEVNULL)
        cal = json.loads((out / "calibration.json").read_text())
        self.assertIsNotNone(cal["recommended"])

    def test_empty_negative_dir_is_reported_not_crashed(self):
        ident = Identifier(OnnxBackend(self.model), InferenceConfig("m", 0.65))
        res = evaluate(ident, self.root / "eval", self.root / "does_not_exist")
        self.assertEqual(res["negative_samples"], 0)
        self.assertIn("Correctly rejected: n/a", format_report(res))


if __name__ == "__main__":
    unittest.main()
