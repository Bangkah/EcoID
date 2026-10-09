import unittest

from app.ai.contract.result import Status
from app.observation.stats import MIN_N, compute_stats, wilson
from app.observation.verification import VerificationStatus as V
from tests.storage.test_store import obs


class WilsonTests(unittest.TestCase):
    def test_known_values(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        lo, hi = wilson(0, 10)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, 0.2775, places=3)
        lo, hi = wilson(10, 10)
        self.assertAlmostEqual(lo, 0.7225, places=3)
        self.assertEqual(hi, 1.0)

    def test_empty_and_bounds(self):
        self.assertIsNone(wilson(0, 0))
        for k, n in ((1, 3), (7, 9), (50, 50), (0, 1)):
            lo, hi = wilson(k, n)
            self.assertTrue(0 <= lo <= k / n <= hi <= 1)


class StatsTests(unittest.TestCase):
    def build(self):
        o, i = [], 0

        def add(**kw):
            nonlocal i
            i += 1
            o.append(obs(i, **kw))
        for _ in range(6):
            add(confidence=0.95)                                                       # verified, high confidence
        for _ in range(2):
            add(confidence=0.95, verification_status=V.REJECTED)
        add(confidence=0.4, verification_status=V.REJECTED, user_species="Nephelium lappaceum")
        add(confidence=0.4, verification_status=V.REJECTED, user_species="Nephelium lappaceum")
        add(confidence=0.7, verification_status=V.UNCERTAIN)
        add(confidence=0.55, latitude=5.0, longitude=97.0, location_source="manual", predicted_species="Musa acuminata",
            identification_status=Status.LOW_CONFIDENCE)
        return o

    def test_counts_and_agreement(self):
        s = compute_stats(self.build())
        self.assertEqual(s["total"], 12)
        self.assertEqual(s["by_status"], {"VERIFIED": 7, "REJECTED": 4, "UNCERTAIN": 1})
        fa = s["field_agreement"]
        self.assertEqual((fa["k"], fa["n"]), (7, 11))
        self.assertFalse(fa["too_few"])
        self.assertTrue(fa["ci_low"] < 7 / 11 < fa["ci_high"])
        self.assertEqual(s["with_location"], 1)
        self.assertEqual(s["low_confidence"], 1)

    def test_confidence_bins_show_the_reliability_of_high_vs_low_scores(self):
        rows = {r["label"]: r for r in compute_stats(self.build())["by_confidence"]}
        hi = rows["0.90–1.00"]
        self.assertEqual((hi["k"], hi["n"]), (6, 8))
        lo = rows["0.00–0.50"]
        self.assertEqual((lo["k"], lo["n"], lo["rate"]), (0, 2, 0.0))
        self.assertTrue(lo["too_few"] and hi["too_few"])                       # n < 10: flagged, not hidden

    def test_corrections_species_and_days(self):
        s = compute_stats(self.build())
        self.assertEqual(s["corrections"], [{"suggested": "Mangifera indica", "actually": "Nephelium lappaceum", "count": 2}])
        self.assertEqual(s["suggested"]["Mangifera indica"], 11)
        self.assertEqual(s["human_labelled"], {"Mangifera indica": 6, "Musa acuminata": 1, "Nephelium lappaceum": 2})
        self.assertEqual(sum(s["per_day"].values()), 12)
        self.assertEqual(s["by_identification_status"]["LOW_CONFIDENCE"]["n"], 1)

    def test_empty(self):
        s = compute_stats([])
        self.assertEqual((s["total"], s["field_agreement"]["n"], s["field_agreement"]["rate"]), (0, 0, None))
        self.assertTrue(s["field_agreement"]["too_few"])
        self.assertEqual(MIN_N, 10)


if __name__ == "__main__":
    unittest.main()
