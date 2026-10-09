import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "summarize_run.py"), *args], capture_output=True, text=True, encoding="utf-8", errors="replace")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_nothing_exists_is_reported_not_guessed(self):
        r = run("--data", str(self.root / "no"), "--run", str(self.root / "no"), "--onnx", str(self.root / "no.onnx"),
                "--reports", str(self.root / "no"), "--out", str(self.root / "o.md"))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.count("not found"), 5)
        self.assertNotIn("Traceback", r.stderr)

    def test_full_run_summary_contains_the_numbers_and_leaks_nothing(self):
        data = self.root / "data"
        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_dataset.py"), "--out", str(data),
                               "--train-per-class", "6", "--val-per-class", "3"], stdout=subprocess.DEVNULL)
        (data / "metadata" / "inat").mkdir(parents=True)
        (data / "metadata" / "inat" / "selection.jsonl").write_text("\n".join(json.dumps(
            {"kind": "class", "split": "train", "user_id": 100 + i % 4, "user_login": "secret_login"}) for i in range(8)), encoding="utf-8")
        run_dir, rep = self.root / "run", self.root / "rep"
        run_dir.mkdir()
        rep.mkdir()
        hist = [{"epoch": i, "stage": "head" if i < 2 else "finetune", "train_loss": 1.2 - i * .1, "train_acc": .5 + i * .05,
                 "val_loss": 1.3 - i * .1, "val_acc": .4 + i * .05, "sec": 1.0} for i in range(6)]
        (run_dir / "train_meta.json").write_text(json.dumps({
            "arch": "mobilenet_v3_small", "input_size": 224, "classes": ["a"] * 5, "args": {"seed": 42, "batch_size": 16},
            "best": {"val_acc": .65, "epoch": 5, "stage": "finetune"}, "history": hist,
            "train_counts": {"a": 6}, "val_counts": {"a": 3}, "versions": {"torch": "x"}}), encoding="utf-8")
        onnx = self.root / "m.onnx"
        Path(str(onnx) + ".json").write_text(json.dumps({"size_bytes": 4_200_000, "sha256": "ab" * 32, "opset": 17,
                                                          "parity_checks": {"tensor_max_abs_logit_diff": 1e-6}, "classes": ["a"]}), encoding="utf-8")
        (rep / "benchmark.json").write_text(json.dumps({
            "threshold": 0.65, "eval_samples": 50, "negative_samples": 110, "top1_accuracy": 0.88, "top3_accuracy": 0.98, "ece": 0.07,
            "operating_point": {"negatives_rejected": 77, "negatives_total": 110, "selective_accuracy": 0.93, "coverage": 0.8},
            "latency": {"median_ms": 31.2, "p95_ms": 44.0}, "hardware": {"processor": "cpu", "cpu_count": 4},
            "confusion_matrix": [[9, 1, 0, 0, 0], [0, 10, 0, 0, 0], [0, 0, 8, 2, 0], [0, 0, 0, 10, 0], [0, 0, 0, 0, 10]]}), encoding="utf-8")
        (rep / "calibration.json").write_text(json.dumps({"constraints": {"min_negative_rejection": .8},
                                                           "recommended": {"threshold": 0.7, "coverage": .75, "selective_accuracy": .95, "negative_rejection_rate": .82}}), encoding="utf-8")
        r = run("--data", str(data), "--run", str(run_dir), "--onnx", str(onnx), "--reports", str(rep))
        self.assertEqual(r.returncode, 0, r.stderr)
        out = r.stdout
        for needle in ("| train | 6 | 6 | 6 | 6 | 6 | 30 |", "| evaluation | 10 | 10 | 10 | 10 | 10 | 50 |", "| negative | 40 | 40 | 40 | 120 |",
                       "Licences: {'CC0': ", "| class | train | 8 | 4 | 2 |", "**0 error(s)", "best val accuracy 65.0% at epoch 5",
                       "4.2 MB", "top-1 **44/50**", "top-3 **49/50**", "negatives rejected 77/110", "recommended threshold **0.7**",
                       "  9   1   0   0   0", "Heuristic observations"):
            self.assertIn(needle, out)
        self.assertNotIn("secret_login", out)                                   # no photographer names
        self.assertTrue((rep / "run_summary.md").is_file())

    def test_heuristics_flag_chance_level_and_leakage_patterns(self):
        run_dir = self.root / "run"
        run_dir.mkdir()
        (run_dir / "train_meta.json").write_text(json.dumps({
            "arch": "m", "classes": ["a"] * 5, "args": {}, "best": {"val_acc": .2, "epoch": 3}, "train_counts": {}, "val_counts": {},
            "history": [{"epoch": i, "stage": "head", "train_loss": 1.6, "train_acc": .9, "val_loss": 1.6, "val_acc": .2, "sec": 1} for i in range(4)]}), encoding="utf-8")
        out = run("--data", str(self.root / "no"), "--run", str(run_dir), "--onnx", "x", "--reports", str(self.root / "no")).stdout
        self.assertIn("near chance level", out)
        self.assertIn("large train/val gap", out)
        (run_dir / "train_meta.json").write_text(json.dumps({
            "arch": "m", "classes": ["a"] * 5, "args": {}, "best": {"val_acc": 1.0, "epoch": 1}, "train_counts": {}, "val_counts": {},
            "history": [{"epoch": i, "stage": "head", "train_loss": .1, "train_acc": 1.0, "val_loss": .1, "val_acc": 1.0, "sec": 1} for i in range(4)]}), encoding="utf-8")
        self.assertIn("check for leakage", run("--data", str(self.root / "no"), "--run", str(run_dir), "--onnx", "x",
                                              "--reports", str(self.root / "no")).stdout)


if __name__ == "__main__":
    unittest.main()
