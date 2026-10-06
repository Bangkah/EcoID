"""iNaturalist data collection: pure logic + a thin, injectable network layer.

Follows iNaturalist's published API practices (https://www.inaturalist.org/pages/api+recommended+practices):
about 1 request/second, <= ~10k requests/day, per_page 200, a descriptive User-Agent, media well under 5 GB/hour.
Only photos hosted on the open-data bucket with an explicitly allowed licence are ever selected.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

API_BASE = "https://api.inaturalist.org/v1"
OPEN_DATA_HOST = "inaturalist-open-data.s3.amazonaws.com"
PHOTO_SIZES = ("square", "small", "medium", "large", "original")
DEFAULT_LICENSES = ("cc0", "cc-by", "cc-by-sa")          # commercial use + derivatives allowed
NC_LICENSES = ("cc-by-nc", "cc-by-nc-sa")                # opt-in only (--allow-nc)


# ----------------------------------------------------------------------------- URLs
def is_open_data_url(url: str) -> bool:
    return urllib.parse.urlparse(url or "").netloc == OPEN_DATA_HOST


def photo_url_variants(url: str, size: str = "medium") -> list[str]:
    """Same photo at another size. The file extension is not always the same, so return likely candidates."""
    if size not in PHOTO_SIZES:
        raise ValueError(f"size must be one of {PHOTO_SIZES}")
    u = urllib.parse.urlparse(url)
    m = re.match(r"^(.*/photos/\d+)/(?:square|small|medium|large|original)\.([A-Za-z0-9]+)$", u.path)
    if not m:
        raise ValueError(f"unrecognised photo url: {url}")
    base, ext = m.groups()
    exts = [ext, "jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
    seen, out = set(), []
    for e in exts:
        if e not in seen:
            seen.add(e)
            out.append(urllib.parse.urlunparse((u.scheme, u.netloc, f"{base}/{size}.{e}", "", "", "")))
    return out


# ----------------------------------------------------------------------------- network layer
class RateLimiter:
    def __init__(self, min_interval: float = 1.1, clock=time.monotonic, sleep=time.sleep):
        self.min_interval, self._clock, self._sleep, self._last = min_interval, clock, sleep, None

    def wait(self):
        now = self._clock()
        if self._last is not None and now - self._last < self.min_interval:
            self._sleep(self.min_interval - (now - self._last))
        self._last = self._clock()


def _retry_wait(err, attempt):
    ra = getattr(err, "headers", None) and err.headers.get("Retry-After")
    try:
        return float(ra)
    except (TypeError, ValueError):
        return min(120.0, 5.0 * 2 ** attempt)


def http_get(url: str, user_agent: str, *, params=None, timeout=30, retries=5, sleep=time.sleep,
             max_bytes: int | None = None, accept="*/*") -> bytes:
    """GET with retry/backoff on 429 and 5xx (honouring Retry-After). 4xx other than 429 raises immediately."""
    full = url + ("?" + urllib.parse.urlencode(params, doseq=True) if params else "")
    for attempt in range(retries + 1):
        req = urllib.request.Request(full, headers={"User-Agent": user_agent, "Accept": accept})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read(max_bytes + 1) if max_bytes else r.read()
            if max_bytes and len(data) > max_bytes:
                raise ValueError(f"response larger than {max_bytes} bytes")
            return data
        except urllib.error.HTTPError as e:
            if (e.code == 429 or e.code >= 500) and attempt < retries:
                sleep(_retry_wait(e, attempt))
                continue
            raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt >= retries:
                raise
            sleep(min(60.0, 2.0 * 2 ** attempt))
    raise RuntimeError("unreachable")


def http_get_json(url, params, user_agent, **kw) -> dict:
    return json.loads(http_get(url, user_agent, params=params, accept="application/json", **kw))


class TaxonNotFound(LookupError):
    pass


class RequestBudgetExceeded(RuntimeError):
    pass


class INatClient:
    def __init__(self, user_agent: str, fetch_json: Optional[Callable[[str, dict], dict]] = None,
                 limiter: Optional[RateLimiter] = None, max_requests: int = 2000):
        self.user_agent = user_agent
        self._fetch = fetch_json or (lambda url, params: http_get_json(url, params, user_agent))
        self._limiter = limiter or RateLimiter()
        self.max_requests, self.requests = max_requests, 0

    def get(self, path: str, params: dict | None = None) -> dict:
        if self.requests >= self.max_requests:
            raise RequestBudgetExceeded(f"stopped after {self.requests} requests (--max-requests)")
        self._limiter.wait()
        self.requests += 1
        return self._fetch(API_BASE + path, dict(params or {}))

    def resolve_taxon(self, name: str) -> dict:
        data = self.get("/taxa", {"q": name, "per_page": 30})
        exact = [t for t in data.get("results", [])
                 if str(t.get("name", "")).lower() == name.lower() and t.get("is_active", True)]
        if not exact:
            raise TaxonNotFound(f"no active iNaturalist taxon named {name!r}")
        return max(exact, key=lambda t: t.get("observations_count", 0))

    def search_places(self, q: str) -> list[dict]:
        return self.get("/places/autocomplete", {"q": q, "per_page": 5}).get("results", [])

    def iter_observations(self, taxon_id: int, quality_grade: str, licenses, *, place_id=None, max_pages=40,
                          per_page=200) -> Iterator[dict]:
        """Newest first (id desc), paging with id_below (the pattern iNaturalist recommends beyond 10k results)."""
        id_below = None
        for _ in range(max_pages):
            params = {"taxon_id": taxon_id, "quality_grade": quality_grade, "photos": "true",
                      "photo_license": ",".join(licenses), "order_by": "id", "order": "desc", "per_page": per_page}
            if place_id:
                params["place_id"] = place_id
            if id_below:
                params["id_below"] = id_below
            results = self.get("/observations", params).get("results", [])
            yield from results
            if len(results) < per_page:
                return
            id_below = results[-1]["id"]


# ----------------------------------------------------------------------------- candidates
@dataclass(frozen=True)
class Candidate:
    photo_id: int
    observation_id: int
    user_id: int
    user_login: str
    kind: str                 # "class" | "negative"
    group: str                # class scientific name, or negative group name
    taxon_id: int
    taxon_name: str
    quality_grade: str
    captive: Optional[bool]
    agreements: int
    disagreements: int
    license: str
    attribution: str
    url: str
    observation_url: str
    width: Optional[int]
    height: Optional[int]
    observed_on: Optional[str]
    place_ids: tuple = ()

    def to_json(self) -> dict:
        d = asdict(self)
        d["place_ids"] = list(self.place_ids)
        return d

    @staticmethod
    def from_json(d: dict) -> "Candidate":
        d = dict(d)
        d["place_ids"] = tuple(d.get("place_ids") or ())
        return Candidate(**d)


def extract_candidates(obs: dict, kind: str, group: str, *, allowed_licenses=DEFAULT_LICENSES, min_side=300,
                       min_agreements=1, drops: Counter | None = None) -> list[Candidate]:
    """One observation -> usable photo candidates. Every rejection is counted in `drops` (shown in the plan report)."""
    drops = drops if drops is not None else Counter()
    taxon = obs.get("taxon") or {}
    uid = (obs.get("user") or {}).get("id")
    if not taxon.get("id"):
        drops["no_taxon"] += 1
        return []
    if uid is None:
        drops["no_user"] += 1
        return []
    if obs.get("identifications_most_disagree"):
        drops["community_disagrees"] += 1
        return []
    grade = obs.get("quality_grade") or ""
    agree = obs.get("num_identification_agreements") or 0
    dis = obs.get("num_identification_disagreements") or 0
    if grade != "research":                     # e.g. cultivated plants are 'casual': require independent agreement
        if agree < min_agreements:
            drops["too_few_agreements"] += 1
            return []
        if dis > 0:
            drops["has_disagreements"] += 1
            return []
    allowed = {x.lower() for x in allowed_licenses}
    out = []
    for p in obs.get("photos") or []:
        lic = (p.get("license_code") or "").lower()
        url = p.get("url") or ""
        if lic not in allowed:
            drops["photo_license_not_allowed"] += 1
        elif not is_open_data_url(url):
            drops["not_open_data_host"] += 1
        else:
            dims = p.get("original_dimensions") or {}
            w, h = dims.get("width"), dims.get("height")
            if w and h and min(w, h) < min_side:
                drops["too_small"] += 1
                continue
            out.append(Candidate(
                photo_id=int(p["id"]), observation_id=int(obs["id"]), user_id=int(uid),
                user_login=str((obs.get("user") or {}).get("login") or ""), kind=kind, group=group,
                taxon_id=int(taxon["id"]), taxon_name=str(taxon.get("name") or ""), quality_grade=grade,
                captive=obs.get("captive"), agreements=int(agree), disagreements=int(dis), license=lic,
                attribution=str(p.get("attribution") or ""), url=url,
                observation_url=str(obs.get("uri") or f"https://www.inaturalist.org/observations/{obs['id']}"),
                width=w, height=h, observed_on=obs.get("observed_on"), place_ids=tuple(obs.get("place_ids") or ())))
    return out


# ----------------------------------------------------------------------------- splitting (by OBSERVER, not by image)
def bucket(user_id: int, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{user_id}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64


def assign_split(user_id: int, salt: str, fractions: dict[str, float]) -> str:
    """Deterministic: every photographer belongs to exactly ONE split, so no observer's style/location leaks across."""
    total = sum(fractions.values())
    x, acc = bucket(user_id, salt) * total, 0.0
    for name, f in fractions.items():
        acc += f
        if x < acc:
            return name
    return list(fractions)[-1]


POSITIVE_FRACTIONS = {"train": 0.75, "val": 0.15, "evaluation": 0.10}
NEGATIVE_FRACTIONS = {"negative": 0.70, "negative_val": 0.30}


def select_group(cands: list[Candidate], targets: dict[str, int], per_observer: dict[str, int], fractions: dict[str, float],
                 *, salt: str, seed: int, exclude_photo_ids=frozenset(), eval_place_id: Optional[int] = None):
    """Pick up to targets[split] photos per split, round-robin across observers so no single person dominates.
    Returns (selected: list[(split, Candidate)], shortages: dict[split, (got, wanted)])."""
    by_split: dict[str, dict[int, list[Candidate]]] = {s: defaultdict(list) for s in fractions}
    for c in cands:
        if c.photo_id in exclude_photo_ids:
            continue
        s = assign_split(c.user_id, salt, fractions)
        if s == "evaluation" and eval_place_id and eval_place_id not in c.place_ids:
            continue
        by_split[s][c.user_id].append(c)
    selected, shortage = [], {}
    for split, want in targets.items():
        pool = by_split.get(split, {})
        rng = random.Random(f"{seed}:{salt}:{split}")
        users = sorted(pool)
        rng.shuffle(users)
        lists = {}
        for u in users:                         # one photo per observation first, so observations aren't over-represented
            per_obs: dict[int, list[Candidate]] = defaultdict(list)
            for c in pool[u]:
                per_obs[c.observation_id].append(c)
            ordered = []
            for photos in per_obs.values():
                rng.shuffle(photos)
            obs_ids = sorted(per_obs)
            rng.shuffle(obs_ids)
            depth = max(len(v) for v in per_obs.values())
            for d in range(depth):
                ordered += [per_obs[o][d] for o in obs_ids if len(per_obs[o]) > d]
            lists[u] = ordered
        got = []
        for r in range(per_observer.get(split, 1)):
            for u in users:
                if len(got) >= want:
                    break
                if len(lists[u]) > r:
                    got.append(lists[u][r])
        selected += [(split, c) for c in got]
        if len(got) < want:
            shortage[split] = (len(got), want)
    return selected, shortage


# ----------------------------------------------------------------------------- manifests
def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
