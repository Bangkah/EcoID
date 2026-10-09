import unittest

from app.observation.exif import read_exif
from tests.helpers import exif_jpeg, image_bytes


class ExifTests(unittest.TestCase):
    def test_north_east(self):
        e = read_exif(exif_jpeg(5.1789, 97.1421, "2026:10:05 07:30:00"))
        self.assertAlmostEqual(e.latitude, 5.1789, places=3)
        self.assertAlmostEqual(e.longitude, 97.1421, places=3)
        self.assertTrue(e.has_location)

    def test_south_west_are_negative(self):
        e = read_exif(exif_jpeg(-6.2, -106.8))
        self.assertLess(e.latitude, 0)
        self.assertLess(e.longitude, 0)

    def test_capture_time_with_and_without_offset(self):
        self.assertEqual(read_exif(exif_jpeg(when="2026:10:05 07:30:00")).captured_at, "2026-10-05T07:30:00")
        self.assertEqual(read_exif(exif_jpeg(when="2026:10:05 07:30:00", offset="+07:00")).captured_at,
                         "2026-10-05T07:30:00+07:00")

    def test_time_without_gps_is_not_a_location(self):
        e = read_exif(exif_jpeg(when="2026:10:05 07:30:00"))
        self.assertFalse(e.has_location)
        self.assertIsNone(e.latitude)

    def test_null_island_and_junk_are_ignored(self):
        self.assertFalse(read_exif(exif_jpeg(0.0, 0.0)).has_location)                       # "no fix" placeholder
        self.assertIsNone(read_exif(exif_jpeg(when="0000:00:00 00:00:00")).captured_at)
        self.assertIsNone(read_exif(exif_jpeg(when="yesterday")).captured_at)

    def test_never_raises(self):
        for data in (b"", b"garbage", image_bytes(size=(20, 20)), image_bytes(size=(20, 20), fmt="PNG")):
            e = read_exif(data)
            self.assertEqual((e.latitude, e.longitude, e.captured_at), (None, None, None))

    def test_to_dict(self):
        self.assertEqual(set(read_exif(b"").to_dict()), {"latitude", "longitude", "captured_at"})


if __name__ == "__main__":
    unittest.main()
