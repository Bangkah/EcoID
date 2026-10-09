import csv
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.data.checks import CLASS_DIRS, check_dataset, dhash, list_images

RNG = np.random.default_rng(1)


def noise_img(path: Path, size=(64, 64)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(RNG.integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)).save(path, "JPEG")


def make_dataset(root: Path, n_train=3, n_val=2, n_eval=2, n_neg=3):
    for split, n in (("train", n_train), ("val", n_val), ("evaluation", n_eval)):
        for d in CLASS_DIRS:
            for i in range(n):
                noise_img(root / split / d / f"{i}.jpg")
    for g in ("lookalike_plants", "non_plant", "low_quality"):
        for i in range(n_neg):
            noise_img(root / "negative" / g / f"{i}.jpg")
    return root


def write_meta(root: Path, license="CC-BY"):
    rows = []
    for p in list_images(root):
        rel = p.relative_to(root).as_posix()
        rows.append({"path": rel, "split": rel.split("/")[0], "label": "", "source": "x",
                     "license": license, "author": "a", "url": "u"})
    (root / "metadata").mkdir(exist_ok=True)
    with open(root / "metadata" / "images.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["path", "split", "label", "source", "license", "author", "url"])
        w.writeheader()
        w.writerows(rows)


class DatasetCheckTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        make_dataset(self.root)
        write_meta(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def run_check(self, **kw):
        kw.setdefault("train_range", (1, 999))
        kw.setdefault("eval_per_class", 2)
        kw.setdefault("min_negative", 5)
        return check_dataset(self.root, **kw)

    def test_clean_dataset_passes(self):
        r = self.run_check()
        self.assertTrue(r.ok, r.errors)

    def test_train_eval_exact_duplicate_is_error(self):
        d = CLASS_DIRS[0]
        shutil.copy(self.root / "train" / d / "0.jpg", self.root / "evaluation" / d / "leak.jpg")
        write_meta(self.root)
        r = self.run_check()
        self.assertFalse(r.ok)
        self.assertTrue(any("exact duplicate (train vs evaluation)" in e or "exact duplicate (evaluation vs train)" in e
                            for e in r.errors), r.errors)

    def test_resized_copy_detected_as_near_duplicate(self):
        d = CLASS_DIRS[1]
        src = self.root / "train" / d / "0.jpg"
        Image.open(src).resize((200, 200)).save(self.root / "evaluation" / d / "resized.jpg", "JPEG", quality=70)
        write_meta(self.root)
        r = self.run_check()
        self.assertTrue(any("near-duplicate" in e for e in r.errors), r.errors)

    def test_corrupt_image_is_error(self):
        (self.root / "train" / CLASS_DIRS[0] / "bad.jpg").write_bytes(b"not an image")
        write_meta(self.root)
        self.assertTrue(any("corrupt" in e for e in self.run_check().errors))

    def test_missing_license_is_error(self):
        write_meta(self.root, license="")
        self.assertTrue(any("missing license" in e for e in self.run_check().errors))

    def test_noncommercial_license_warns(self):
        write_meta(self.root, license="CC-BY-NC")
        r = self.run_check()
        self.assertTrue(r.ok)
        self.assertTrue(any("non-commercial" in w for w in r.warnings))

    def test_missing_lookalike_group_is_error(self):
        shutil.rmtree(self.root / "negative" / "lookalike_plants")
        write_meta(self.root)
        self.assertTrue(any("lookalike_plants" in e for e in self.run_check().errors))

    def test_too_few_negatives_is_error(self):
        self.assertTrue(any("requires >=" in e for e in self.run_check(min_negative=1000).errors))

    def test_wrong_class_folders_is_error(self):
        shutil.rmtree(self.root / "val" / CLASS_DIRS[4])
        write_meta(self.root)
        self.assertFalse(self.run_check().ok)

    def test_srs_count_warnings(self):
        r = check_dataset(self.root, min_negative=5)  # default SRS ranges
        self.assertTrue(any("SRS expects 300-500" in w for w in r.warnings))
        self.assertTrue(any("SRS expects 10" in w for w in r.warnings))

    def test_dhash_stable_under_resize(self):
        p = self.root / "a.jpg"
        noise_img(p, (96, 96))
        q = self.root / "b.jpg"
        Image.open(p).resize((48, 48)).save(q)
        self.assertLessEqual((dhash(p) ^ dhash(q)).bit_count(), 10)


if __name__ == "__main__":
    unittest.main()
