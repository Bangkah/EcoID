import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.ai.contract.result import Status
from app.observation.models import Observation
from app.observation.verification import VerificationStatus as V, parse_status
from app.storage.store import NotFoundError, ObservationStore

FR007 = {"id", "image_path", "predicted_species", "confidence", "alternative_predictions",
         "verification_status", "timestamp", "latitude", "longitude", "notes", "model"}


def obs(i=0, status=V.VERIFIED, **kw):
    base = dict(id=f"{i:032x}", image_path=f"images/2026/10/{i:032x}.jpg", predicted_species="Mangifera indica",
                confidence=0.91, alternative_predictions=({"label": "Carica papaya", "score": 0.05},),
                verification_status=status,
                timestamp=(datetime(2026, 10, 6, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat(),
                model="EcoID Vision Model", identification_status=Status.IDENTIFIED)
    base.update(kw)
    return Observation(**base)


class ModelTests(unittest.TestCase):
    def test_to_dict_has_all_fr007_fields(self):
        d = obs().to_dict()
        self.assertTrue(FR007 <= set(d))
        self.assertEqual(set(d) - FR007, {"identification_status"})
        self.assertIsNone(d["latitude"])
        self.assertEqual(d["notes"], "")

    def test_validation(self):
        for kw in ({"confidence": 1.2}, {"confidence": -0.1}, {"latitude": 10.0}, {"longitude": 10.0},
                   {"latitude": 91.0, "longitude": 0.0}, {"latitude": 0.0, "longitude": 181.0},
                   {"notes": "x" * 2001}):
            with self.assertRaises(ValueError, msg=str(kw)):
                obs(**kw)
        obs(latitude=3.59, longitude=98.67)   # valid pair

    def test_parse_status(self):
        self.assertEqual(parse_status("verified"), V.VERIFIED)
        with self.assertRaises(ValueError):
            parse_status("MAYBE")


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        self.store = ObservationStore(self.dir)

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip(self):
        o = obs(1, notes="leaf looks right", latitude=3.5, longitude=98.6)
        self.store.insert(o)
        self.assertEqual(self.store.get(o.id), o)

    def test_persists_across_instances(self):
        self.store.insert(obs(1))
        again = ObservationStore(self.dir)
        self.assertEqual(again.get(obs(1).id).predicted_species, "Mangifera indica")

    def test_list_newest_first_with_paging_and_filter(self):
        statuses = [V.VERIFIED, V.REJECTED, V.UNCERTAIN, V.VERIFIED, V.REJECTED]
        for i, s in enumerate(statuses):
            self.store.insert(obs(i, s))
        items, total = self.store.list()
        self.assertEqual(total, 5)
        self.assertEqual([o.id for o in items], [obs(i).id for i in (4, 3, 2, 1, 0)])
        page, total = self.store.list(limit=2, offset=2)
        self.assertEqual([o.id for o in page], [obs(2).id, obs(1).id])
        only, total = self.store.list(status=V.REJECTED)
        self.assertEqual((total, [o.id for o in only]), (2, [obs(4).id, obs(1).id]))

    def test_update_changes_only_status_and_notes(self):
        self.store.insert(obs(1))
        u = self.store.update(obs(1).id, verification_status=V.REJECTED, notes="actually a rambutan")
        self.assertEqual((u.verification_status, u.notes), (V.REJECTED, "actually a rambutan"))
        self.assertEqual(u.predicted_species, "Mangifera indica")
        u2 = self.store.update(obs(1).id, notes="n2")
        self.assertEqual(u2.verification_status, V.REJECTED)
        with self.assertRaises(ValueError):
            self.store.update(obs(1).id, notes="x" * 3000)
        self.assertEqual(self.store.get(obs(1).id).notes, "n2")   # failed update changed nothing

    def test_missing_ids(self):
        for fn in (lambda: self.store.get("0" * 32), lambda: self.store.delete("0" * 32),
                   lambda: self.store.update("0" * 32, notes="x")):
            with self.assertRaises(NotFoundError):
                fn()

    def test_delete_removes_row_and_files(self):
        o = obs(1)
        img = self.store.abspath(o.image_path)
        img.parent.mkdir(parents=True)
        img.write_bytes(b"x")
        self.store.thumb_path(o.id).write_bytes(b"t")
        self.store.insert(o)
        self.store.delete(o.id)
        self.assertFalse(img.exists() or self.store.thumb_path(o.id).exists())
        with self.assertRaises(NotFoundError):
            self.store.get(o.id)

    def test_database_enforces_constraints(self):
        conn = sqlite3.connect(self.store.db_path)
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO observations VALUES ('a','p','s',0.5,'[]','MAYBE','t',NULL,NULL,'','m','IDENTIFIED')")
        with self.assertRaises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO observations VALUES ('a','p','s',1.5,'[]','VERIFIED','t',NULL,NULL,'','m','IDENTIFIED')")
        conn.close()

    def test_schema_version_and_future_guard(self):
        conn = sqlite3.connect(self.store.db_path)
        self.assertEqual(conn.execute("PRAGMA user_version").fetchone()[0], 1)
        conn.execute("PRAGMA user_version = 99")
        conn.commit()
        conn.close()
        with self.assertRaises(RuntimeError):
            ObservationStore(self.dir)

    def test_abspath_blocks_escape(self):
        with self.assertRaises(ValueError):
            self.store.abspath("../outside.jpg")


if __name__ == "__main__":
    unittest.main()
