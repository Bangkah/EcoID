import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.observation.verification import VerificationStatus as V
from app.storage.store import _SCHEMA_V1, UNSET, ObservationStore
from tests.storage.test_store import obs


def at(i):
    return (datetime(2026, 10, 6, tzinfo=timezone.utc) + timedelta(minutes=i)).isoformat()


class MigrationTests(unittest.TestCase):
    def test_v1_database_is_upgraded_in_place_without_losing_data(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            for sub in ("images", "thumbs", "drafts"):
                (Path(d) / sub).mkdir()
            conn = sqlite3.connect(Path(d) / "ecoid.db")
            conn.executescript(_SCHEMA_V1)
            conn.execute("INSERT INTO observations VALUES ('old1','images/x.jpg','Cocos nucifera',0.8,'[]','VERIFIED',"
                         "'2026-09-01T00:00:00+00:00',3.5,98.6,'from phase 3','EcoID Vision Model','IDENTIFIED')")
            conn.execute("PRAGMA user_version = 1")
            conn.commit()
            conn.close()
            store = ObservationStore(d)
            old = store.get("old1")
            self.assertEqual((old.predicted_species, old.notes, old.latitude), ("Cocos nucifera", "from phase 3", 3.5))
            self.assertEqual(old.location_source, "manual")       # origin unknown in v1 -> the neutral value
            self.assertEqual((old.user_species, old.captured_at, old.location_accuracy_m), (None, None, None))
            store.update("old1", notes="edited")
            self.assertEqual(sqlite3.connect(Path(d) / "ecoid.db").execute("PRAGMA user_version").fetchone()[0], 2)
            store.insert(obs(5, V.REJECTED, user_species="Nephelium lappaceum"))
            self.assertEqual(store.get(obs(5).id).user_species, "Nephelium lappaceum")
            ObservationStore(d)                                           # opening again is a no-op


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = ObservationStore(self.tmp.name)
        mk = lambda i, **kw: self.store.insert(obs(i, timestamp=at(i), **kw))   # noqa: E731
        mk(1, notes="shady corner near the well", predicted_species="Mangifera indica")
        mk(2, notes="100% sure it is a Pisang", predicted_species="Musa acuminata", latitude=5.18, longitude=97.14,
           location_source="device")
        mk(3, notes="under_score", predicted_species="Carica papaya", verification_status=V.REJECTED,
           user_species="Nephelium lappaceum")
        mk(4, notes="old photo", predicted_species="Cocos nucifera", captured_at="2025-01-15T09:00:00")
        self.ids = [obs(i).id for i in (1, 2, 3, 4)]

    def tearDown(self):
        self.tmp.cleanup()

    def found(self, **kw):
        items, total = self.store.list(**kw)
        self.assertEqual(total, len(items))
        return {o.id for o in items}

    def test_text_search_notes_species_common_name_and_correction(self):
        i1, i2, i3, i4 = self.ids
        self.assertEqual(self.found(q="shady"), {i1})
        self.assertEqual(self.found(q="SHADY"), {i1})
        self.assertEqual(self.found(q="mangifera"), {i1})
        self.assertEqual(self.found(q="Mangga"), {i1})            # Indonesian common name of the suggestion
        self.assertEqual(self.found(q="pisang"), {i2})
        self.assertEqual(self.found(q="rambutan"), set())          # only the scientific name is stored
        self.assertEqual(self.found(q="nephelium"), {i3})          # found through the correction
        self.assertEqual(self.found(q="  "), set(self.ids))

    def test_like_wildcards_are_literal(self):
        i1, i2, i3, i4 = self.ids
        self.assertEqual(self.found(q="100%"), {i2})
        self.assertEqual(self.found(q="%"), {i2})
        self.assertEqual(self.found(q="_"), {i3})
        self.assertEqual(self.found(q="'; DROP TABLE observations;--"), set())
        self.assertEqual(len(self.found()), 4)

    def test_species_filter_matches_suggestion_or_correction(self):
        i1, i2, i3, i4 = self.ids
        self.assertEqual(self.found(species="mangifera indica"), {i1})
        self.assertEqual(self.found(species="Nephelium lappaceum"), {i3})

    def test_dates_use_capture_time_before_save_time(self):
        i1, i2, i3, i4 = self.ids
        self.assertEqual(self.found(date_to="2025-12-31"), {i4})                  # saved in 2026, photographed in 2025
        self.assertEqual(self.found(date_from="2026-01-01"), {i1, i2, i3})
        self.assertEqual(self.found(date_from="2025-01-15", date_to="2025-01-15"), {i4})

    def test_has_location_and_combined_filters_with_totals(self):
        i1, i2, i3, i4 = self.ids
        self.assertEqual(self.found(has_location=True), {i2})
        self.assertEqual(self.found(has_location=False), {i1, i3, i4})
        self.assertEqual(self.found(status=V.REJECTED, q="under"), {i3})
        self.assertEqual(self.found(status=V.VERIFIED, q="under"), set())
        items, total = self.store.list(limit=1, offset=1)
        self.assertEqual((len(items), total), (1, 4))


class UpdateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.store = ObservationStore(self.tmp.name)
        self.store.insert(obs(1, V.REJECTED, user_species="Citrus"))
        self.id = obs(1).id

    def tearDown(self):
        self.tmp.cleanup()

    def test_correction_set_clear_and_auto_dropped_on_verify(self):
        self.assertEqual(self.store.update(self.id, user_species="Annona muricata").user_species, "Annona muricata")
        self.assertIsNone(self.store.update(self.id, user_species=None).user_species)
        self.store.update(self.id, user_species="Citrus")
        v = self.store.update(self.id, verification_status=V.VERIFIED)                  # agreeing again => no correction
        self.assertIsNone(v.user_species)
        self.assertEqual(v.human_label, "Mangifera indica")

    def test_correction_with_verified_in_one_call_is_rejected_and_changes_nothing(self):
        with self.assertRaises(ValueError):
            self.store.update(self.id, verification_status=V.VERIFIED, user_species="Citrus")
        self.assertEqual(self.store.get(self.id).verification_status, V.REJECTED)

    def test_location_set_and_clear(self):
        o = self.store.update(self.id, location=(5.18, 97.14, "device", 30.0))
        self.assertEqual((o.latitude, o.longitude, o.location_source, o.location_accuracy_m), (5.18, 97.14, "device", 30.0))
        o = self.store.update(self.id, notes="n")                                       # UNSET leaves it alone
        self.assertEqual(o.latitude, 5.18)
        o = self.store.update(self.id, location=None)
        self.assertEqual((o.latitude, o.longitude, o.location_source, o.location_accuracy_m), (None, None, None, None))
        with self.assertRaises(ValueError):
            self.store.update(self.id, location=(95.0, 0.0, "manual", None))
        self.assertIsNone(self.store.get(self.id).latitude)
        self.assertIs(UNSET, UNSET)

    def test_human_label_semantics(self):
        self.assertEqual(self.store.get(self.id).human_label, "Citrus")
        self.assertIsNone(self.store.update(self.id, verification_status=V.UNCERTAIN).human_label)


if __name__ == "__main__":
    unittest.main()
