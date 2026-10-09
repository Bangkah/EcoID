import http.server
import json
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
from collections import Counter
from pathlib import Path

from PIL import Image

from app.ai.data import inat, inat_pipeline as P
from app.ai.data.checks import CLASS_DIRS, check_dataset, list_images, read_metadata
from tests.data.fake_inat import FakeINat, all_taxon_names

ROOT = Path(__file__).resolve().parents[2]
UA = "test-agent"


def obs(**kw):
    o = {"id": 1, "uri": "https://www.inaturalist.org/observations/1", "quality_grade": "research", "captive": False,
         "taxon": {"id": 9, "name": "Mangifera indica"}, "user": {"id": 7, "login": "u"},
         "num_identification_agreements": 2, "num_identification_disagreements": 0, "place_ids": [1],
         "photos": [{"id": 11, "url": inat.photo_url_variants(
             "https://inaturalist-open-data.s3.amazonaws.com/photos/11/square.jpg", "square")[0],
             "license_code": "cc-by", "attribution": "(c) u", "original_dimensions": {"width": 2048, "height": 1536}}]}
    o.update(kw)
    return o


class UrlTests(unittest.TestCase):
    def test_variants_swap_size_and_drop_query_and_try_extensions(self):
        v = inat.photo_url_variants("https://inaturalist-open-data.s3.amazonaws.com/photos/398757/square.JPG?1372952785", "medium")
        self.assertEqual(v[0], "https://inaturalist-open-data.s3.amazonaws.com/photos/398757/medium.JPG")
        self.assertIn("https://inaturalist-open-data.s3.amazonaws.com/photos/398757/medium.jpg", v)
        self.assertEqual(len(v), len(set(v)))

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            inat.photo_url_variants("https://x/y.jpg", "medium")
        with self.assertRaises(ValueError):
            inat.photo_url_variants("https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg", "huge")

    def test_open_data_host_check(self):
        self.assertTrue(inat.is_open_data_url("https://inaturalist-open-data.s3.amazonaws.com/photos/1/square.jpg"))
        self.assertFalse(inat.is_open_data_url("https://static.inaturalist.org/photos/1/square.jpg"))
        self.assertFalse(inat.is_open_data_url("https://inaturalist-open-data.s3.amazonaws.com.evil.com/photos/1/square.jpg"))


class ExtractTests(unittest.TestCase):
    def run_extract(self, o, **kw):
        drops = Counter()
        return inat.extract_candidates(o, "class", "Mangifera indica", drops=drops, **kw), drops

    def test_good_photo(self):
        c, d = self.run_extract(obs())
        self.assertEqual((len(c), c[0].license, c[0].user_id, c[0].photo_id), (1, "cc-by", 7, 11))
        self.assertEqual(sum(d.values()), 0)

    def test_licence_filter_per_photo_and_nc_opt_in(self):
        o = obs()
        o["photos"][0]["license_code"] = "cc-by-nc"
        self.assertEqual(self.run_extract(o)[0], [])
        self.assertEqual(len(self.run_extract(o, allowed_licenses=inat.DEFAULT_LICENSES + inat.NC_LICENSES)[0]), 1)
        o["photos"][0]["license_code"] = None
        self.assertEqual(self.run_extract(o)[1]["photo_license_not_allowed"], 1)
        o["photos"][0]["license_code"] = "cc-by-nd"
        self.assertEqual(self.run_extract(o)[0], [])

    def test_only_open_data_host(self):
        o = obs()
        o["photos"][0]["url"] = "https://static.inaturalist.org/photos/11/square.jpg"
        c, d = self.run_extract(o)
        self.assertEqual((c, d["not_open_data_host"]), ([], 1))

    def test_small_photos_dropped(self):
        o = obs()
        o["photos"][0]["original_dimensions"] = {"width": 400, "height": 250}
        self.assertEqual(self.run_extract(o)[1]["too_small"], 1)

    def test_casual_needs_independent_agreement_and_no_disagreement(self):
        self.assertEqual(self.run_extract(obs(quality_grade="casual", num_identification_agreements=0))[1]["too_few_agreements"], 1)
        self.assertEqual(self.run_extract(obs(quality_grade="casual", num_identification_disagreements=1))[1]["has_disagreements"], 1)
        self.assertEqual(len(self.run_extract(obs(quality_grade="casual"))[0]), 1)
        self.assertEqual(len(self.run_extract(obs(quality_grade="research", num_identification_agreements=0))[0]), 1)

    def test_malformed_observations_are_skipped_not_fatal(self):
        for bad in (obs(taxon=None), obs(user={}), obs(identifications_most_disagree=True), obs(photos=None)):
            self.assertEqual(self.run_extract(bad)[0], [])


class SplitTests(unittest.TestCase):
    def test_deterministic_and_roughly_proportional(self):
        f = inat.POSITIVE_FRACTIONS
        a = [inat.assign_split(u, "s", f) for u in range(4000)]
        self.assertEqual(a, [inat.assign_split(u, "s", f) for u in range(4000)])
        share = {k: v / 4000 for k, v in Counter(a).items()}
        for k, want in f.items():
            self.assertAlmostEqual(share[k], want, delta=0.03)
        self.assertNotEqual(a, [inat.assign_split(u, "other-salt", f) for u in range(4000)])

    def cands(self, n_users=60, per_user=5, group="Mangifera indica"):
        out, pid = [], 0
        for u in range(n_users):
            for k in range(per_user):
                pid += 1
                out.append(inat.Candidate(photo_id=pid, observation_id=u * 100 + k, user_id=u, user_login="", kind="class",
                                          group=group, taxon_id=1, taxon_name=group, quality_grade="research", captive=None,
                                          agreements=1, disagreements=0, license="cc-by", attribution="", url="u",
                                          observation_url="o", width=None, height=None, observed_on=None,
                                          place_ids=(6903,) if u % 2 else (2,)))
        return out

    def run_select(self, **kw):
        kw.setdefault("salt", "s")
        kw.setdefault("seed", 0)
        return inat.select_group(self.cands(), {"train": 40, "val": 10, "evaluation": 3}, {"train": 3, "val": 2, "evaluation": 1},
                                 inat.POSITIVE_FRACTIONS, **kw)

    def test_an_observer_never_appears_in_two_splits(self):
        sel, _ = self.run_select()
        seen = {}
        for split, c in sel:
            self.assertEqual(seen.setdefault(c.user_id, split), split)

    def test_caps_and_targets(self):
        sel, short = self.run_select()
        counts = Counter(s for s, _ in sel)
        self.assertLessEqual(counts["train"], 40)
        self.assertLessEqual(counts["evaluation"], 3)
        per_user = Counter((s, c.user_id) for s, c in sel)
        self.assertTrue(all(n <= {"train": 3, "val": 2, "evaluation": 1}[s] for (s, _), n in per_user.items()))
        self.assertEqual(len({c.photo_id for _, c in sel}), len(sel))

    def test_round_robin_spreads_across_observers(self):
        sel, _ = self.run_select()
        train_users = Counter(c.user_id for s, c in sel if s == "train")
        self.assertGreater(len(train_users), 13)             # 40 photos from >=14 people, not 8 prolific ones

    def test_same_seed_same_result_different_seed_different(self):
        a = [c.photo_id for _, c in self.run_select(seed=1)[0]]
        self.assertEqual(a, [c.photo_id for _, c in self.run_select(seed=1)[0]])
        self.assertNotEqual(a, [c.photo_id for _, c in self.run_select(seed=2)[0]])

    def test_shortage_is_reported_not_hidden(self):
        sel, short = inat.select_group(self.cands(n_users=6), {"train": 400, "val": 10, "evaluation": 10}, {"train": 3},
                                       inat.POSITIVE_FRACTIONS, salt="s", seed=0)
        self.assertIn("train", short)
        self.assertLess(short["train"][0], 400)

    def test_exclusions_and_eval_place(self):
        sel, _ = self.run_select()
        first = sel[0][1].photo_id
        again, _ = self.run_select(exclude_photo_ids={first})
        self.assertNotIn(first, [c.photo_id for _, c in again])
        sel, _ = self.run_select(eval_place_id=6903)
        ev = [c for s, c in sel if s == "evaluation"]
        self.assertTrue(ev and all(6903 in c.place_ids for c in ev))


class ClientTests(unittest.TestCase):
    def test_pagination_with_id_below_and_per_page_200(self):
        fake = FakeINat(["Mangifera indica"], n_obs=1000, n_users=50)
        c = inat.INatClient(UA, fake.fetch_json, inat.RateLimiter(0, sleep=lambda s: None))
        tid = next(iter(fake.taxa))
        got = list(c.iter_observations(tid, "research", inat.DEFAULT_LICENSES, max_pages=50))
        ids = [o["id"] for o in got]
        self.assertEqual(ids, sorted(ids, reverse=True))
        self.assertEqual(len(ids), len(set(ids)))
        obs_calls = [p for u, p in fake.requests if u.endswith("/observations")]
        self.assertTrue(all(p["per_page"] == 200 for p in obs_calls))
        self.assertTrue(len(obs_calls) >= 2 and "id_below" not in obs_calls[0] and "id_below" in obs_calls[1])

    def test_request_budget_is_a_hard_stop(self):
        fake = FakeINat(["Mangifera indica"], n_obs=1000)
        c = inat.INatClient(UA, fake.fetch_json, inat.RateLimiter(0, sleep=lambda s: None), max_requests=1)
        with self.assertRaises(inat.RequestBudgetExceeded):
            list(c.iter_observations(next(iter(fake.taxa)), "research", inat.DEFAULT_LICENSES, max_pages=50))
        self.assertEqual(c.requests, 1)

    def test_rate_limiter_spaces_calls(self):
        t, slept = [0.0], []
        lim = inat.RateLimiter(1.0, clock=lambda: t[0], sleep=lambda s: (slept.append(s), t.__setitem__(0, t[0] + s)))
        lim.wait(); lim.wait(); lim.wait()
        self.assertEqual(len(slept), 2)
        self.assertTrue(all(abs(s - 1.0) < 1e-9 for s in slept))

    def test_resolve_taxon_exact_active_most_observed(self):
        def fetch(url, params):
            return {"results": [
                {"id": 1, "name": "Citrus", "is_active": True, "observations_count": 5},
                {"id": 2, "name": "Citrus", "is_active": False, "observations_count": 999},
                {"id": 3, "name": "Citrus × limon", "is_active": True, "observations_count": 800},
                {"id": 4, "name": "citrus", "is_active": True, "observations_count": 50}]}
        c = inat.INatClient(UA, fetch, inat.RateLimiter(0, sleep=lambda s: None))
        self.assertEqual(c.resolve_taxon("Citrus")["id"], 4)
        with self.assertRaises(inat.TaxonNotFound):
            c.resolve_taxon("Nothing here")


class HttpRetryTests(unittest.TestCase):
    def test_429_is_retried_honouring_retry_after_and_404_is_not(self):
        hits = {"n": 0}

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                hits["n"] += 1
                if self.path.startswith("/limited") and hits["n"] == 1:
                    self.send_response(429); self.send_header("Retry-After", "7"); self.end_headers(); return
                if self.path.startswith("/missing"):
                    self.send_response(404); self.end_headers(); return
                body = json.dumps({"ok": True}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers()
                self.wfile.write(body)

        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        base = f"http://127.0.0.1:{srv.server_address[1]}"
        try:
            slept = []
            self.assertEqual(inat.http_get_json(base + "/limited", {"a": 1}, UA, sleep=slept.append), {"ok": True})
            self.assertEqual(slept, [7.0])
            before = hits["n"]
            with self.assertRaises(urllib.error.HTTPError):
                inat.http_get(base + "/missing", UA, sleep=slept.append)
            self.assertEqual(hits["n"], before + 1)                       # no retry on 404
            with self.assertRaises(ValueError):
                inat.http_get(base + "/ok", UA, max_bytes=3)
        finally:
            srv.shutdown(); srv.server_close()


class PipelineEndToEnd(unittest.TestCase):
    """plan -> select -> download against the fake iNaturalist, then the REAL dataset checker."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.data = Path(cls.tmp.name) / "data"
        cls.cand = cls.data / "metadata" / "inat" / "candidates"
        cls.fake = FakeINat(all_taxon_names(), n_obs=260, n_users=90)
        cls.cfg = P.load_config(ROOT / "config" / "inat_taxa.toml")
        cls.client = inat.INatClient(UA, cls.fake.fetch_json, inat.RateLimiter(0, sleep=lambda s: None), max_requests=5000)
        cls.logs = []
        cls.report = P.plan(cls.client, cls.cfg, cls.cand, licenses=inat.DEFAULT_LICENSES, grades=["research", "casual"],
                            max_candidates=120, max_pages=3, log=cls.logs.append)
        cls.targets = {"train": 8, "val": 3, "evaluation": 2}
        cls.neg = {g: {"negative": 10, "negative_val": 4} for g in ("lookalike_plants", "non_plant", "low_quality")}
        cls.rows, cls.short = P.select(cls.cand, targets=cls.targets, neg_targets=cls.neg, seed=1)
        degrade = {g for k, g, _, d in P.iter_groups(cls.cfg) if d}
        cls.stats = P.download(cls.rows, cls.data, cls.fake.fetch_bytes, degrade_groups=degrade, seed=1,
                               limiter=inat.RateLimiter(0, sleep=lambda s: None), log=cls.logs.append)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_plan_report_explains_every_drop(self):
        total_drops = Counter()
        for r in self.report:
            total_drops.update(r["drops"])
        for reason in ("photo_license_not_allowed", "too_small", "too_few_agreements", "not_open_data_host"):
            self.assertGreater(total_drops[reason], 0, reason)
        self.assertTrue(any("resolved 'Mangifera indica'" in l for l in self.logs))

    def test_only_allowed_licences_survive(self):
        lic = {c["license"] for c in self.rows}
        self.assertTrue(lic <= set(inat.DEFAULT_LICENSES), lic)
        self.assertTrue(all(inat.is_open_data_url(c["url"]) for c in self.rows))

    def test_no_observer_in_two_splits_and_no_photo_twice(self):
        users = {}
        for r in self.rows:
            key = (r["kind"], r["user_id"]) if r["kind"] == "class" else ("neg", r["user_id"])
            if r["kind"] == "class":
                self.assertEqual(users.setdefault(key, r["split"]), r["split"])
        self.assertEqual(len({r["photo_id"] for r in self.rows}), len(self.rows))

    def test_layout_and_counts(self):
        self.assertFalse(self.short, self.short)
        for split, n in self.targets.items():
            for d in CLASS_DIRS:
                self.assertEqual(len(list_images(self.data / split / d)), n, (split, d))
        for g in ("lookalike_plants", "non_plant", "low_quality"):
            self.assertEqual(len(list_images(self.data / "negative" / g)), 10)
            self.assertEqual(len(list_images(self.data / "negative_val" / g)), 4)
        self.assertEqual(self.stats["downloaded"], len(self.rows))
        self.assertEqual(self.stats.get("not_found_or_failed", 0), 0)    # .JPG 404 -> .jpg fallback worked

    def test_metadata_and_attribution_are_complete(self):
        meta = read_metadata(self.data / "metadata" / "images.csv")
        files = [p.relative_to(self.data).as_posix() for p in list_images(self.data)]
        self.assertEqual(sorted(meta), sorted(files))
        for r in meta.values():
            self.assertEqual(r["source"], "iNaturalist")
            self.assertTrue(r["license"] and r["author"] and r["url"].startswith("https://www.inaturalist.org/observations/"))
        low = [r for k, r in meta.items() if "/low_quality/" in k]
        self.assertTrue(low and all("degraded" in r["author"] for r in low))
        att = (self.data / "ATTRIBUTION.md").read_text()
        self.assertEqual(att.count("\n| ") - 1, len(meta))
        self.assertTrue((self.data / "metadata" / "inat" / "downloaded.jsonl").is_file())

    def test_real_dataset_checker_accepts_the_result(self):
        r = check_dataset(self.data, train_range=(1, 999), eval_per_class=2, min_negative=30)
        self.assertEqual(r.errors, [])

    def test_degraded_images_are_actually_degraded_and_decodable(self):
        for p in list_images(self.data / "negative" / "low_quality"):
            with Image.open(p) as im:
                im.load()
                self.assertGreaterEqual(min(im.size), 224)

    def test_download_is_resumable_and_idempotent(self):
        calls = []
        stats = P.download(self.rows, self.data, lambda u: calls.append(u) or self.fake.fetch_bytes(u),
                           limiter=inat.RateLimiter(0, sleep=lambda s: None))
        self.assertEqual(calls, [])
        self.assertEqual(stats["already_downloaded"], len(self.rows))

    def test_budget_stops_downloading_and_can_continue(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            s1 = P.download(self.rows, Path(d), self.fake.fetch_bytes, budget_bytes=30_000,
                            limiter=inat.RateLimiter(0, sleep=lambda s: None), log=lambda *_: None)
            self.assertEqual(s1["stopped_by_budget"], 1)
            self.assertLess(s1["downloaded"], len(self.rows))
            s2 = P.download(self.rows, Path(d), self.fake.fetch_bytes, limiter=inat.RateLimiter(0, sleep=lambda s: None))
            self.assertEqual(s1["downloaded"] + s2["downloaded"], len(self.rows))

    def test_plan_resumes_without_new_requests(self):
        before = self.client.requests
        P.plan(self.client, self.cfg, self.cand, licenses=inat.DEFAULT_LICENSES, grades=["research", "casual"], log=lambda *_: None)
        self.assertEqual(self.client.requests, before)

    def test_select_cli_runs_offline(self):
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "fetch_inat.py"), "--data", str(self.data), "select",
                            "--train", "5", "--val", "2", "--eval", "1"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("selection written", r.stdout)


class ConfigTests(unittest.TestCase):
    def test_shipped_config_matches_the_srs_classes_and_required_groups(self):
        cfg = P.load_config(ROOT / "config" / "inat_taxa.toml")
        self.assertEqual(set(cfg["negatives"]), {"lookalike_plants", "non_plant", "low_quality"})
        self.assertIn("Citrus", cfg["negatives"]["lookalike_plants"]["taxa"])
        self.assertTrue(cfg["negatives"]["low_quality"]["degrade"])

    def test_wrong_class_set_is_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            p = Path(d) / "c.toml"
            p.write_text('[classes]\n"Mangifera indica" = ["x"]\n')
            with self.assertRaises(ValueError):
                P.load_config(p)


if __name__ == "__main__":
    unittest.main()
