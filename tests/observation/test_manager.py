import json
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from PIL import Image

from app.ai.config import InferenceConfig
from app.ai.contract.result import Status
from app.ai.inference.identifier import Identifier
from app.ai.preprocessing.preprocess import InvalidImageError
from app.observation.manager import ObservationManager
from app.observation.verification import VerificationStatus as V
from app.storage.store import NotFoundError, ObservationStore
from tests.helpers import FakeBackend, image_bytes


class Clock:
    def __init__(self):
        self.t = datetime(2026, 10, 6, 8, 0, tzinfo=timezone.utc)

    def __call__(self):
        self.t += timedelta(seconds=1)
        return self.t


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.mgr = self.make(FakeBackend([8, 0, 0, 0, 0]))

    def make(self, backend):
        return ObservationManager(Identifier(backend, InferenceConfig("Test Model", 0.65)),
                                  ObservationStore(self.dir), clock=Clock())

    def tearDown(self):
        self.tmp.cleanup()

    def test_identify_creates_draft_but_no_observation(self):
        d = self.mgr.identify(image_bytes())
        self.assertEqual(d.result.status, Status.IDENTIFIED)
        self.assertEqual(self.mgr.list()[1], 0)                # nothing saved before verification
        self.assertTrue((self.dir / "drafts" / f"{d.id}.json").exists())

    def test_save_after_verification_keeps_original_and_records_model(self):
        raw = image_bytes(size=(1200, 900))
        d = self.mgr.identify(raw)
        o = self.mgr.save(d.id, "verified", "  fruit visible  ")
        self.assertEqual(o.verification_status, V.VERIFIED)
        self.assertEqual(o.notes, "fruit visible")
        self.assertEqual(o.model, "Test Model")                # acceptance: model name on every observation
        self.assertEqual(o.predicted_species, "Mangifera indica")
        self.assertEqual(len(o.alternative_predictions), 2)    # remaining Top-3
        saved = self.mgr.store.abspath(o.image_path)
        self.assertEqual(saved.read_bytes(), raw)              # byte-identical, NOT downscaled to 224
        with Image.open(saved) as im:
            self.assertEqual(im.size, (1200, 900))
        with Image.open(self.mgr.store.thumb_path(o.id)) as t:
            self.assertLessEqual(max(t.size), 320)
        self.assertEqual(list((self.dir / "drafts").iterdir()), [])   # draft consumed

    def test_low_confidence_is_recorded(self):
        m = self.make(FakeBackend([0.5, 0.4, 0.3, 0.2, 0.1]))
        o = m.save(m.identify(image_bytes()).id, "UNCERTAIN")
        self.assertEqual(o.identification_status, Status.LOW_CONFIDENCE)
        self.assertEqual(o.verification_status, V.UNCERTAIN)

    def test_all_three_verification_choices(self):
        for s in ("VERIFIED", "REJECTED", "UNCERTAIN"):
            self.assertEqual(self.mgr.save(self.mgr.identify(image_bytes(size=(80, 60))).id, s).verification_status.value, s)

    def test_invalid_status_keeps_draft(self):
        d = self.mgr.identify(image_bytes(size=(80, 60)))
        with self.assertRaises(ValueError):
            self.mgr.save(d.id, "MAYBE")
        self.mgr.save(d.id, "VERIFIED")                        # still savable afterwards

    def test_bad_coordinates_keep_draft_and_move_no_files(self):
        d = self.mgr.identify(image_bytes(size=(80, 60)))
        with self.assertRaises(ValueError):
            self.mgr.save(d.id, "VERIFIED", latitude=5.0)      # longitude missing
        self.assertEqual(list((self.dir / "images").rglob("*.jpg")), [])
        o = self.mgr.save(d.id, "VERIFIED", latitude=3.5, longitude=98.6)
        self.assertEqual((o.latitude, o.longitude), (3.5, 98.6))

    def test_draft_cannot_be_saved_twice(self):
        d = self.mgr.identify(image_bytes(size=(80, 60)))
        self.mgr.save(d.id, "VERIFIED")
        with self.assertRaises(NotFoundError):
            self.mgr.save(d.id, "VERIFIED")

    def test_invalid_images_rejected_before_inference(self):
        for bad in (b"", b"garbage", image_bytes(fmt="JPEG")[:500]):
            with self.assertRaises(InvalidImageError):
                self.mgr.identify(bad)
        self.assertEqual(list((self.dir / "drafts").iterdir()), [])

    def test_unsupported_format_and_size_limit(self):
        import io
        buf = io.BytesIO()
        Image.new("RGB", (20, 20)).save(buf, "GIF")
        with self.assertRaises(InvalidImageError):
            self.mgr.identify(buf.getvalue())
        small = ObservationManager(Identifier(FakeBackend(), InferenceConfig("m", 0.65)),
                                   ObservationStore(self.dir), max_image_bytes=1000)
        with self.assertRaises(InvalidImageError):
            small.identify(image_bytes())

    def test_malformed_draft_ids(self):
        for bad in ("../../etc/passwd", "", "XYZ", "0" * 31):
            with self.assertRaises(NotFoundError):
                self.mgr.save(bad, "VERIFIED")

    def test_discard_and_purge(self):
        a = self.mgr.identify(image_bytes(size=(80, 60)))
        b = self.mgr.identify(image_bytes(size=(80, 60)))
        self.mgr.discard(a.id)
        self.assertFalse((self.dir / "drafts" / f"{a.id}.json").exists())
        meta = self.dir / "drafts" / f"{b.id}.json"
        m = json.loads(meta.read_text())
        m["created"] = time.time() - 3 * 86400
        meta.write_text(json.dumps(m))
        self.assertEqual(self.mgr.purge_drafts(24), 1)
        self.assertEqual(list((self.dir / "drafts").iterdir()), [])

    def test_history_update_and_delete(self):
        o = self.mgr.save(self.mgr.identify(image_bytes(size=(80, 60))).id, "VERIFIED")
        u = self.mgr.update(o.id, "rejected", " it was a different tree ")
        self.assertEqual((u.verification_status, u.notes), (V.REJECTED, "it was a different tree"))
        self.assertEqual(self.mgr.list(status="REJECTED")[1], 1)
        self.mgr.delete(o.id)
        self.assertEqual(self.mgr.list()[1], 0)


if __name__ == "__main__":
    unittest.main()
