import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.ai.config import InferenceConfig
from app.ai.inference.identifier import Identifier
from app.observation.manager import ObservationManager, parse_location
from app.observation.species import normalize_species
from app.observation.verification import VerificationStatus as V
from app.storage.store import NotFoundError, ObservationStore
from tests.helpers import FakeBackend, exif_jpeg, image_bytes


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 6, 8, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


class FieldTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.dir = Path(self.tmp.name)
        self.mgr = ObservationManager(Identifier(FakeBackend([8, 0, 0, 0, 0]), InferenceConfig("Test Model", 0.65)),
                                      ObservationStore(self.dir), clock=Clock())
        self.photo = exif_jpeg(5.1789, 97.1421, "2026:10:05 07:30:00", "+07:00")

    def tearDown(self):
        self.tmp.cleanup()

    def test_draft_exposes_exif_but_saving_does_not_use_it_by_default(self):
        d = self.mgr.identify(self.photo)
        self.assertAlmostEqual(d.to_dict()["exif"]["latitude"], 5.1789, places=3)
        o = self.mgr.save(d.id, "VERIFIED")
        self.assertIsNone(o.latitude)                                   # location is opt-in
        self.assertIsNone(o.location_source)
        self.assertEqual(o.captured_at, "2026-10-05T07:30:00+07:00")    # a time, not a place: kept

    def test_opt_in_to_the_photos_own_location(self):
        o = self.mgr.save(self.mgr.identify(self.photo).id, "VERIFIED", use_photo_location=True)
        self.assertAlmostEqual(o.latitude, 5.1789, places=3)
        self.assertEqual((o.location_source, o.location_accuracy_m), ("exif", None))

    def test_photo_location_requested_but_absent_keeps_the_draft(self):
        d = self.mgr.identify(image_bytes(size=(80, 60)))
        with self.assertRaises(ValueError):
            self.mgr.save(d.id, "VERIFIED", use_photo_location=True)
        self.mgr.save(d.id, "VERIFIED")                                 # still savable

    def test_explicit_coordinates_win_and_default_to_manual(self):
        d = self.mgr.identify(self.photo)
        o = self.mgr.save(d.id, "VERIFIED", latitude=1.5, longitude=2.5, use_photo_location=True)
        self.assertEqual((o.latitude, o.longitude, o.location_source), (1.5, 2.5, "manual"))
        o = self.mgr.save(self.mgr.identify(self.photo).id, "VERIFIED", latitude=1.5, longitude=2.5,
                          location_source="device", location_accuracy_m=18.2)
        self.assertEqual((o.location_source, o.location_accuracy_m), ("device", 18.2))

    def test_malformed_location_input(self):
        for kw in ({"latitude": "5.1", "longitude": 1.0}, {"latitude": True, "longitude": 1.0}, {"latitude": 5.0},
                   {"latitude": 5.0, "longitude": 1.0, "location_source": "psychic"},
                   {"latitude": 5.0, "longitude": 1.0, "location_accuracy_m": -4}):
            d = self.mgr.identify(image_bytes(size=(80, 60)))
            with self.assertRaises(ValueError, msg=str(kw)):
                self.mgr.save(d.id, "VERIFIED", **kw)
            self.mgr.discard(d.id)
        self.assertIsNone(parse_location(None, None))

    def test_rejected_with_correction(self):
        o = self.mgr.save(self.mgr.identify(self.photo).id, "REJECTED", user_species="  rambutan ")
        self.assertEqual((o.user_species, o.human_label), ("rambutan", "rambutan"))
        o = self.mgr.save(self.mgr.identify(self.photo).id, "REJECTED", user_species="mangga")
        self.assertEqual(o.user_species, "Mangifera indica")           # canonical class name
        o = self.mgr.save(self.mgr.identify(self.photo).id, "REJECTED", user_species="")
        self.assertIsNone(o.user_species)

    def test_verified_cannot_carry_a_correction_and_draft_survives(self):
        d = self.mgr.identify(self.photo)
        with self.assertRaises(ValueError):
            self.mgr.save(d.id, "VERIFIED", user_species="Citrus")
        self.mgr.save(d.id, "VERIFIED")

    def test_normalize_species(self):
        self.assertEqual(normalize_species("Pisang"), "Musa acuminata")
        self.assertEqual(normalize_species(" cocos   NUCIFERA "), "Cocos nucifera")
        self.assertEqual(normalize_species("Citrus × limon"), "Citrus × limon")
        self.assertIsNone(normalize_species(None))
        self.assertIsNone(normalize_species("   "))
        with self.assertRaises(ValueError):
            normalize_species(5)

    def test_update_flow_correction_and_location(self):
        o = self.mgr.save(self.mgr.identify(self.photo).id, "REJECTED")
        o = self.mgr.update(o.id, user_species="Sirsak")
        self.assertEqual(o.user_species, "Sirsak")
        o = self.mgr.update(o.id, location=(5.2, 97.1, "manual", None))
        self.assertEqual(o.location_source, "manual")
        o = self.mgr.update(o.id, verification_status="VERIFIED")
        self.assertIsNone(o.user_species)
        self.assertEqual(o.latitude, 5.2)                               # status change leaves the location alone
        o = self.mgr.update(o.id, location=None)
        self.assertIsNone(o.latitude)

    def test_list_filters_and_date_validation(self):
        self.mgr.save(self.mgr.identify(self.photo).id, "VERIFIED", "needle in notes", 5.0, 97.0)
        self.mgr.save(self.mgr.identify(image_bytes(size=(80, 60))).id, "UNCERTAIN", "plain")
        self.assertEqual(self.mgr.list(q="needle")[1], 1)
        self.assertEqual(self.mgr.list(has_location=True)[1], 1)
        self.assertEqual(self.mgr.list(date_from="2026-10-05", date_to="2026-10-05")[1], 1)   # EXIF capture day
        for bad in ("05/10/2026", "2026-1-5", "tomorrow"):
            with self.assertRaises(ValueError):
                self.mgr.list(date_from=bad)

    def test_drafts_written_by_phase_3_still_load(self):
        d = self.mgr.identify(image_bytes(size=(80, 60)))
        meta = self.dir / "drafts" / f"{d.id}.json"
        m = json.loads(meta.read_text())
        m.pop("exif")
        meta.write_text(json.dumps(m))
        self.assertEqual(self.mgr.save(d.id, "VERIFIED").verification_status, V.VERIFIED)

    def test_unknown_draft(self):
        with self.assertRaises(NotFoundError):
            self.mgr.save("0" * 32, "VERIFIED", use_photo_location=True)


if __name__ == "__main__":
    unittest.main()
