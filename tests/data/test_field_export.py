import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.ai.config import InferenceConfig
from app.ai.data.checks import check_dataset, list_images, read_metadata
from app.ai.data.field_export import export, plan_item
from app.ai.inference.identifier import Identifier
from app.observation.manager import ObservationManager
from app.storage.store import ObservationStore
from tests.helpers import FakeBackend, exif_jpeg

ROOT = Path(__file__).resolve().parents[2]


class FieldExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.tmp.name)
        self.app, self.data = root / "app", root / "data"
        self.store = ObservationStore(self.app)
        self.mgr = ObservationManager(Identifier(FakeBackend([8, 0, 0, 0, 0]), InferenceConfig("m", 0.65)), self.store)
        self.ids = {}
        specs = {"verified": ("VERIFIED", None), "rej_class": ("REJECTED", "Pisang"), "rej_other": ("REJECTED", "Rambutan"),
                 "rej_none": ("REJECTED", None), "unsure": ("UNCERTAIN", None), "verified2": ("VERIFIED", None)}
        for k, (status, fix) in specs.items():
            photo = exif_jpeg(5.18, 97.14, "2026:10:05 07:30:00", rgb=(230, 20, 20), size=(300, 200))
            self.ids[k] = self.mgr.save(self.mgr.identify(photo).id, status, user_species=fix).id

    def tearDown(self):
        self.tmp.cleanup()

    def test_what_goes_where(self):
        stats, skipped = export(self.store, self.data, "evaluation")
        self.assertEqual(stats["exported"], 4)
        self.assertEqual(len(skipped), 1)                                        # rejected without a correction
        self.assertIn("label unknown", skipped[0])
        self.assertEqual(stats["skipped_status_not_selected"], 1)                # the UNCERTAIN one is never selected
        mango = list_images(self.data / "evaluation" / "Mangifera_indica")
        self.assertEqual({p.stem for p in mango}, {self.ids["verified"], self.ids["verified2"]})
        self.assertEqual([p.stem for p in list_images(self.data / "evaluation" / "Musa_acuminata")], [self.ids["rej_class"]])
        self.assertEqual([p.stem for p in list_images(self.data / "negative" / "lookalike_plants")], [self.ids["rej_other"]])

    def test_metadata_rows_and_idempotence(self):
        export(self.store, self.data, "evaluation")
        meta = read_metadata(self.data / "metadata" / "images.csv")
        self.assertEqual(len(meta), 4)
        for r in meta.values():
            self.assertEqual((r["source"], r["license"]), ("EcoID field photo", "own"))
        self.assertEqual(meta[f"evaluation/Musa_acuminata/{self.ids['rej_class']}.jpg"]["label"], "Musa acuminata")
        again, _ = export(self.store, self.data, "evaluation")
        self.assertEqual((again["exported"], again["already_exported"]), (0, 4))

    def test_train_split_has_no_negatives(self):
        stats, skipped = export(self.store, self.data, "train")
        self.assertEqual(stats["exported"], 3)
        self.assertTrue(any("train split has no negatives" in s for s in skipped))
        self.assertFalse((self.data / "negative").exists())
        stats, _ = export(self.store, self.data, "val")
        self.assertTrue(list_images(self.data / "negative_val" / "lookalike_plants"))

    def test_per_class_cap_dry_run_and_status_filter(self):
        stats, _ = export(self.store, self.data, "evaluation", per_class=1, dry_run=True)
        self.assertEqual(stats["exported"], 3)
        self.assertEqual(list_images(self.data), [])                              # dry run wrote nothing
        stats, _ = export(self.store, self.data, "evaluation", statuses=("REJECTED",))
        self.assertEqual(stats["exported"], 2)
        with self.assertRaises(ValueError):
            export(self.store, self.data, "nonsense")

    def test_strip_exif_removes_gps_and_keeps_orientation(self):
        export(self.store, self.data, "evaluation", strip_exif=True)
        for p in list_images(self.data):
            with Image.open(p) as im:
                self.assertEqual(dict(im.getexif().get_ifd(0x8825)), {})
                self.assertEqual(im.size, (300, 200))
        export(self.store, self.data / "keep", "evaluation")
        with Image.open(list_images(self.data / "keep")[0]) as im:
            self.assertTrue(dict(im.getexif().get_ifd(0x8825)))                    # default: bytes copied untouched

    def test_plan_item_rules(self):
        o = self.store.get(self.ids["unsure"])
        self.assertIsNone(plan_item(o, "evaluation")[0])
        self.assertEqual(plan_item(self.store.get(self.ids["verified"]), "val")[0].dest_dir, "val/Mangifera_indica")

    def test_cli_and_dataset_checker_accept_the_result(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_observations.py"), "--app-data", str(self.app),
                            "--data", str(self.data), "--split", "evaluation"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("'exported': 4", r.stdout)
        res = check_dataset(self.data, train_range=(0, 999), eval_per_class=0, min_negative=0)
        # (folders other than evaluation/negative are absent in this tiny fixture; only duplicates/licences matter here)
        self.assertFalse([e for e in res.errors if "duplicate" in e or "licen" in e or "metadata" in e], res.errors)
        r2 = subprocess.run([sys.executable, str(ROOT / "scripts" / "export_observations.py"), "--app-data",
                             str(self.app / "nope"), "--split", "val"], capture_output=True, text=True, encoding="utf-8", errors="replace")
        self.assertNotEqual(r2.returncode, 0)




if __name__ == "__main__":
    unittest.main()
