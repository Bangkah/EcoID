import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from app.ai.config import InferenceConfig
from app.ai.inference.identifier import Identifier
from app.observation import export as ex
from app.observation.basemap import BasemapError, load_basemap
from app.observation.manager import ObservationManager
from app.storage.store import ObservationStore
from tests.helpers import FakeBackend, exif_jpeg, image_bytes


def make_manager(d):
    return ObservationManager(Identifier(FakeBackend([8, 0, 0, 0, 0]), InferenceConfig("Test Model", 0.65)), ObservationStore(d))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)
        self.mgr = make_manager(self.root / "a")
        self.photo = image_bytes(size=(300, 200))
        self.a = self.mgr.save(self.mgr.identify(self.photo).id, "VERIFIED", "=HYPERLINK(\"http://evil\")", 5.1789, 97.1421,
                               location_source="device", location_accuracy_m=12)
        self.b = self.mgr.save(self.mgr.identify(exif_jpeg(when="2026:10:05 07:30:00", size=(300, 200))).id, "REJECTED",
                               "- minus note, with comma\nand a newline", user_species="Rambutan")
        self.obs = self.mgr.store.list_all()

    def tearDown(self):
        self.tmp.cleanup()

    def test_csv_columns_bom_formula_guard_and_roundtrip(self):
        raw = ex.to_csv(self.obs)
        self.assertTrue(raw.startswith("\ufeff".encode()))
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        self.assertEqual(list(rows[0]), ex.CSV_FIELDS)
        by = {r["id"]: r for r in rows}
        self.assertEqual(by[self.a.id]["notes"], "'=HYPERLINK(\"http://evil\")")               # not executable in a spreadsheet
        self.assertEqual(by[self.b.id]["notes"], "'- minus note, with comma\nand a newline")   # commas/newlines survive quoting
        self.assertEqual((by[self.a.id]["common_name"], by[self.a.id]["latitude"]), ("Mangga", "5.1789"))
        self.assertEqual((by[self.b.id]["human_label"], by[self.b.id]["captured_at"]), ("Rambutan", "2026-10-05T07:30:00"))
        self.assertEqual(by[self.a.id]["photo"], Path(self.a.image_path).name)

    def test_without_location_nothing_leaks_in_any_format(self):
        for raw in (ex.to_csv(self.obs, False), ex.to_json(self.obs, False)):
            self.assertNotIn("5.1789", raw.decode("utf-8-sig"))
            self.assertNotIn("97.1421", raw.decode("utf-8-sig"))
        with self.assertRaises(ValueError):
            ex.to_geojson(self.obs, include_location=False)
        rows = json.loads(ex.to_json(self.obs, False))
        self.assertTrue(all(r["latitude"] is None and r["location_source"] is None for r in rows))

    def test_geojson_is_lon_lat_and_skips_unlocated(self):
        g = json.loads(ex.to_geojson(self.obs))
        self.assertEqual(g["type"], "FeatureCollection")
        self.assertEqual(len(g["features"]), 1)
        f = g["features"][0]
        self.assertEqual(f["geometry"], {"type": "Point", "coordinates": [97.1421, 5.1789]})     # [lon, lat] per RFC 7946
        self.assertEqual(f["properties"]["verification_status"], "VERIFIED")
        self.assertNotIn("image_path", f["properties"])

    def test_backup_roundtrip_into_a_fresh_app(self):
        buf = io.BytesIO()
        m = ex.build_backup(self.mgr.store, self.obs, buf)
        self.assertEqual((m["count"], m["format"]), (2, "ecoid-backup"))
        zpath = self.root / "b.zip"
        zpath.write_bytes(buf.getvalue())
        fresh = ObservationStore(self.root / "b")
        res = ex.import_backup(fresh, zpath)
        self.assertEqual(res["stats"], {"imported": 2})
        got = fresh.get(self.a.id)
        self.assertEqual((got.latitude, got.location_source, got.notes), (5.1789, "device", self.a.notes))
        self.assertEqual(fresh.abspath(got.image_path).read_bytes(), self.photo)                 # photo byte-identical
        self.assertTrue(fresh.thumb_path(self.a.id).is_file())
        self.assertEqual(fresh.get(self.b.id).human_label, "Rambutan")
        again = ex.import_backup(fresh, zpath)
        self.assertEqual(again["stats"], {"skipped_existing": 2})

    def test_backup_without_location(self):
        buf = io.BytesIO()
        ex.build_backup(self.mgr.store, self.obs, buf, include_location=False)
        with zipfile.ZipFile(buf) as z:
            rows = json.loads(z.read("observations.json"))
            self.assertTrue(all(r["latitude"] is None for r in rows))
            self.assertFalse(json.loads(z.read("manifest.json"))["includes_location"])

    def _tampered(self, mutate):
        buf = io.BytesIO()
        ex.build_backup(self.mgr.store, self.obs, buf)
        src = zipfile.ZipFile(io.BytesIO(buf.getvalue()))
        out = io.BytesIO()
        with zipfile.ZipFile(out, "w") as z:
            for n in src.namelist():
                z.writestr(*mutate(n, src.read(n)))
        p = self.root / "t.zip"
        p.write_bytes(out.getvalue())
        return ex.import_backup(ObservationStore(self.root / f"t{len(list(self.root.iterdir()))}"), p)

    def test_corrupted_photo_is_rejected_others_still_import(self):
        def mutate(name, data):
            if name == f"images/{self.a.id}.jpg":
                data = data[:-5] + b"XXXXX"
            return name, data
        res = self._tampered(mutate)
        self.assertEqual(res["stats"], {"imported": 1, "rejected": 1})
        self.assertIn("checksum", res["errors"][0])

    def test_hostile_archive_cannot_write_outside_the_data_dir(self):
        def mutate(name, data):
            if name == "manifest.json":
                m = json.loads(data)
                m["files"][0]["name"] = "../../evil.jpg"
                data = json.dumps(m).encode()
            if name == "observations.json":
                rows = json.loads(data)
                rows[0]["image_path"] = "../../../etc/passwd"
                data = json.dumps(rows).encode()
            return name, data
        res = self._tampered(mutate)
        self.assertEqual(res["stats"].get("rejected"), 1)
        self.assertFalse((self.root.parent / "evil.jpg").exists())

    def test_not_a_backup(self):
        p = self.root / "x.zip"
        with zipfile.ZipFile(p, "w") as z:
            z.writestr("hello.txt", "hi")
        with self.assertRaises(ValueError):
            ex.import_backup(ObservationStore(self.root / "c"), p)


class BasemapTests(unittest.TestCase):
    def test_validation(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            d = Path(d)
            self.assertIsNone(load_basemap(d))
            (d / "map.png").write_bytes(image_bytes(size=(40, 30), fmt="PNG"))
            cfg = d / "basemap.json"
            cfg.write_text(json.dumps({"image": "map.png", "bounds": [5.0, 97.0, 5.5, 97.5], "attribution": "me"}))
            b = load_basemap(d)
            self.assertEqual((b["bounds"], b["mime"]), ([5.0, 97.0, 5.5, 97.5], "image/png"))
            for bad in ({"image": "map.png", "bounds": [5.5, 97.0, 5.0, 97.5]}, {"image": "map.png", "bounds": [1, 2, 3]},
                        {"image": "../outside.png", "bounds": [1, 2, 3, 4]}, {"image": "nope.png", "bounds": [1, 2, 3, 4]},
                        {"image": "map.png", "bounds": [-89, 0, 89, 10]}, {"image": "map.txt", "bounds": [1, 2, 3, 4]}):
                cfg.write_text(json.dumps(bad))
                with self.assertRaises(BasemapError, msg=str(bad)):
                    load_basemap(d)
            cfg.write_text("{not json")
            with self.assertRaises(BasemapError):
                load_basemap(d)


if __name__ == "__main__":
    unittest.main()
