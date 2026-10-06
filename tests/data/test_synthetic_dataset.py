import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from app.ai.data.checks import check_dataset

ROOT = Path(__file__).resolve().parents[2]


class SyntheticDatasetTests(unittest.TestCase):
    def test_generated_dataset_passes_the_real_checker(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            subprocess.check_call([sys.executable, str(ROOT / "scripts" / "make_synthetic_dataset.py"), "--out", tmp,
                                   "--train-per-class", "6", "--val-per-class", "3"], stdout=subprocess.DEVNULL)
            r = check_dataset(Path(tmp))             # default SRS thresholds: negatives >= 100, lookalikes present, licences, no leakage
            self.assertEqual(r.errors, [])
            r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_dataset.py"), "--data", tmp],
                                capture_output=True, text=True)
            self.assertEqual(r2.returncode, 0, r2.stdout)


if __name__ == "__main__":
    unittest.main()
