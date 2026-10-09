"""A tiny fake iNaturalist (API + image host) with the response shapes documented for GET /observations."""
import io
import random
import tomllib
import urllib.error
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OPEN = "https://inaturalist-open-data.s3.amazonaws.com/photos/{pid}/square.{ext}?1372952785"
STATIC = "https://static.inaturalist.org/photos/{pid}/square.jpg"


def all_taxon_names():
    cfg = tomllib.loads((ROOT / "config" / "inat_taxa.toml").read_text(encoding="utf-8"))
    names = {n for v in cfg["classes"].values() for n in v}
    for g in cfg["negatives"].values():
        names |= set(g["taxa"])
    return sorted(names)


class FakeINat:
    def __init__(self, names, n_obs=300, n_users=90, seed=0):
        rng = random.Random(seed)
        self.taxa, self.obs, self.requests = {}, {}, []
        oid, pid = 1_000_000, 5_000_000
        for i, name in enumerate(names):
            tid = 100 + i
            self.taxa[tid] = {"id": tid, "name": name, "rank": "species", "is_active": True, "observations_count": 1000}
            rows = []
            for _ in range(n_obs):
                oid += 1
                uid = rng.randrange(n_users)
                grade = rng.choice(["research", "casual"])
                photos = []
                for _ in range(rng.choice([1, 1, 2, 3])):
                    pid += 1
                    lic = rng.choice(["cc-by", "cc0", "cc-by-sa", "cc-by-nc", None, "cc-by-nd"])
                    url = (STATIC if rng.random() < 0.05 else OPEN).format(pid=pid, ext=rng.choice(["jpg", "JPG"]))
                    side = rng.choice([200, 1024, 2048, 2048, 2048])
                    photos.append({"id": pid, "url": url, "license_code": lic,
                                   "attribution": f"(c) user{uid}, some rights reserved ({(lic or 'none').upper()})",
                                   "original_dimensions": {"width": side * 4 // 3, "height": side}})
                rows.append({"id": oid, "uri": f"https://www.inaturalist.org/observations/{oid}", "quality_grade": grade,
                             "captive": grade == "casual", "taxon": {"id": tid, "name": name},
                             "user": {"id": uid, "login": f"user{uid}"}, "photos": photos,
                             "num_identification_agreements": rng.choice([0, 1, 2]),
                             "num_identification_disagreements": rng.choice([0, 0, 0, 1]),
                             "place_ids": rng.choice([[1, 6903], [1, 2]]), "observed_on": "2024-05-01"})
            self.obs[tid] = sorted(rows, key=lambda o: -o["id"])

    def fetch_json(self, url, params):
        self.requests.append((url, dict(params)))
        if url.endswith("/taxa"):
            q = params["q"].lower()
            return {"results": [t for t in self.taxa.values() if q in t["name"].lower()]}
        if url.endswith("/places/autocomplete"):
            return {"results": [{"id": 6903, "name": "Indonesia", "display_name": "Indonesia"}]}
        if url.endswith("/observations"):
            rows = self.obs[int(params["taxon_id"])]
            lic = set(params["photo_license"].split(","))
            rows = [o for o in rows if o["quality_grade"] == params["quality_grade"]
                    and any((p["license_code"] or "") in lic for p in o["photos"])]
            if params.get("id_below"):
                rows = [o for o in rows if o["id"] < int(params["id_below"])]
            if params.get("place_id"):
                rows = [o for o in rows if int(params["place_id"]) in o["place_ids"]]
            return {"total_results": len(rows), "results": rows[: int(params["per_page"])]}
        raise AssertionError(f"unexpected url {url}")

    def fetch_bytes(self, url):
        """Files exist only as <size>.jpg (so the .JPG variant must 404, like the real bucket's case-sensitive keys)."""
        if not url.endswith(".jpg"):
            raise urllib.error.HTTPError(url, 404, "not found", {}, None)
        pid = int(url.split("/photos/")[1].split("/")[0])
        rng = np.random.default_rng(pid)
        base = rng.integers(40, 215, 3)
        a = np.clip(base + rng.normal(0, 40, (300, 400, 3)), 0, 255).astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(a).save(buf, "JPEG", quality=90)
        return buf.getvalue()
