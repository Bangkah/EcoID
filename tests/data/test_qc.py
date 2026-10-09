import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.ai.data.checks import CLASS_DIRS, check_dataset, read_metadata
from app.ai.data.qc import apply_rejections, contact_sheet, make_sheets, read_reject_ids
from tests.data.test_checks import make_dataset, write_meta

ROOT = Path(__file__).resolve().parents[2]


class QcTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        make_dataset(self.root)
        write_meta(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def test_contact_sheet_and_paging(self):
        folder = self.root / "train" / CLASS_DIRS[0]
        out = make_sheets(folder, self.root / "sheets", per_sheet=2)
        self.assertEqual(len(out), 2)                          # 3 images -> sheets of 2 + 1
        with Image.open(out[0]) as im:
            self.assertGreater(im.size[0], 500)
        self.assertEqual(contact_sheet([self.root / "missing.jpg"], self.root / "x.jpg"), 1)   # unreadable -> red tile, no crash

    def test_reject_file_parsing(self):
        f = self.root / "r.txt"
        f.write_text("123  # blurry\n\n# whole line comment\ntrain/Musa_acuminata/456.jpg\n", encoding="utf-8")
        self.assertEqual(read_reject_ids(f), {"123", "456"})

    def test_apply_rejections_moves_files_updates_metadata_and_remembers(self):
        victim = self.root / "train" / CLASS_DIRS[1] / "1.jpg"
        ids = {victim.stem + "x"}
        self.assertEqual(apply_rejections(self.root, ids), [])        # nothing matches
        dry = apply_rejections(self.root, {"1"}, dry_run=True)
        self.assertTrue(dry and victim.exists())                      # dry run changes nothing
        moved = apply_rejections(self.root, {"1"})
        self.assertIn(f"train/{CLASS_DIRS[1]}/1.jpg", moved)
        self.assertFalse(victim.exists())
        self.assertTrue((self.root / "_rejected" / "train" / CLASS_DIRS[1] / "1.jpg").exists())
        self.assertNotIn(f"train/{CLASS_DIRS[1]}/1.jpg", read_metadata(self.root / "metadata" / "images.csv"))
        self.assertIn("1", (self.root / "metadata" / "rejected_photo_ids.txt").read_text(encoding="utf-8").split())
        r = check_dataset(self.root, train_range=(1, 999), eval_per_class=2, min_negative=5)
        self.assertEqual(r.errors, [])                                # dataset still consistent afterwards

    def test_flat_images_do_not_cause_false_leakage_alarms(self):
        for split in ("train", "evaluation"):
            Image.new("RGB", (64, 64), (255, 255, 255)).save(self.root / split / CLASS_DIRS[2] / "blank.jpg")
        # two blank (different-file, identical-content) images in different splits are EXACT duplicates -> still an error
        r = check_dataset(self.root, train_range=(1, 999), eval_per_class=3, min_negative=5, require_metadata=False)
        self.assertTrue(any("exact duplicate" in e for e in r.errors))
        self.assertFalse(any("near-duplicate" in e for e in r.errors))

    def test_template_script_marks_own_photos(self):
        (self.root / "metadata" / "images.csv").unlink()
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "make_metadata_template.py"), "--data", str(self.root),
                            "--own", "negative/non_plant"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr)
        meta = read_metadata(self.root / "metadata" / "images.csv")
        own = [v for k, v in meta.items() if k.startswith("negative/non_plant/")]
        self.assertTrue(own and all(v["license"] == "own" for v in own))
        others = [v for k, v in meta.items() if k.startswith("train/")]
        self.assertTrue(all(v["license"] == "" for v in others))


if __name__ == "__main__":
    unittest.main()
