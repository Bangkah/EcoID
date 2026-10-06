"""plan -> select -> download. Each stage reads/writes plain files so it can be stopped, inspected and resumed."""
from __future__ import annotations

import hashlib
import io
import math
import random
import tomllib
import urllib.error
from collections import Counter
from pathlib import Path
from typing import Callable

from PIL import Image

from app.ai.data import inat
from app.ai.data.checks import read_metadata, write_metadata
from app.ai.data.degrade import degrade_image
from app.ai.contract.labels import CLASSES

_EXT = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}


def load_config(path) -> dict:
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    unknown = [c for c in cfg.get("classes", {}) if c not in CLASSES]
    missing = [c for c in CLASSES if c not in cfg.get("classes", {})]
    if unknown or missing:
        raise ValueError(f"[classes] must match the SRS classes exactly (unknown={unknown}, missing={missing})")
    return cfg


def slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_")


def group_dir_name(kind: str, group: str) -> str:
    return group.replace(" ", "_") if kind == "class" else group


# ----------------------------------------------------------------------------- plan
def iter_groups(cfg):
    for label, specs in cfg["classes"].items():
        yield "class", label, specs, False
    for name, spec in cfg.get("negatives", {}).items():
        yield "negative", name, spec["taxa"], bool(spec.get("degrade", False))


def resolve_specs(client, specs, log=print):
    out = []
    for s in specs:
        if isinstance(s, dict):
            out.append({"id": int(s["id"]), "name": s.get("name", str(s["id"]))})
        else:
            t = client.resolve_taxon(s)
            log(f"    resolved {s!r} -> id {t['id']} ({t.get('rank', '?')}, {t.get('observations_count', '?')} observations on iNaturalist)")
            out.append({"id": int(t["id"]), "name": t.get("name", s)})
    return out


def plan(client, cfg, cand_dir: Path, *, licenses, grades, place_id=None, max_candidates=2500, max_pages=20,
         min_side=300, min_agreements=1, per_observer_candidates=30, refresh=False, log=print) -> list[dict]:
    cand_dir = Path(cand_dir)
    report = []
    for kind, group, specs, _ in iter_groups(cfg):
        path = cand_dir / f"{kind}__{slug(group)}.jsonl"
        if path.exists() and not refresh:
            log(f"[{kind}:{group}] already planned ({path.name}); use --refresh to redo")
            continue
        log(f"[{kind}:{group}]")
        taxa = resolve_specs(client, specs, log)
        cap_taxon = math.ceil(max_candidates / len(taxa))
        cap_grade = math.ceil(cap_taxon / len(grades))
        seen, cands = set(), []
        for tx in taxa:
            for grade in grades:
                drops, users, kept, fetched = Counter(), Counter(), 0, 0
                for obs in client.iter_observations(tx["id"], grade, licenses, place_id=place_id, max_pages=max_pages):
                    fetched += 1
                    for c in inat.extract_candidates(obs, kind, group, allowed_licenses=licenses, min_side=min_side,
                                                     min_agreements=min_agreements, drops=drops):
                        if c.photo_id in seen:
                            continue
                        if users[c.user_id] >= per_observer_candidates:
                            drops["observer_cap"] += 1
                            continue
                        seen.add(c.photo_id)
                        users[c.user_id] += 1
                        cands.append(c)
                        kept += 1
                    if kept >= cap_grade:
                        break
                row = {"kind": kind, "group": group, "taxon": tx["name"], "taxon_id": tx["id"], "grade": grade,
                       "observations_fetched": fetched, "photos_kept": kept, "observers": len(users),
                       "drops": dict(drops)}
                report.append(row)
                log(f"    {tx['name']} / {grade}: fetched {fetched} observations -> kept {kept} photos from {len(users)} observers; drops {dict(drops)}")
        inat.write_jsonl(path, (c.to_json() for c in cands))
    return report


# ----------------------------------------------------------------------------- select
DEFAULT_TARGETS = {"train": 400, "val": 60, "evaluation": 10}
DEFAULT_NEG_TARGETS = {"lookalike_plants": {"negative": 50, "negative_val": 20},
                       "non_plant": {"negative": 30, "negative_val": 10},
                       "low_quality": {"negative": 30, "negative_val": 10}}
PER_OBSERVER = {"train": 6, "val": 2, "evaluation": 1, "negative": 3, "negative_val": 2}


def select(cand_dir: Path, *, targets=None, neg_targets=None, salt="ecoid-v1", seed=0, exclude_ids=frozenset(),
           eval_place_id=None):
    targets, neg_targets = targets or DEFAULT_TARGETS, neg_targets or DEFAULT_NEG_TARGETS
    rows, shortages = [], []
    used = set(exclude_ids)
    for path in sorted(Path(cand_dir).glob("*.jsonl")):
        cands = [inat.Candidate.from_json(d) for d in inat.read_jsonl(path)]
        if not cands:
            continue
        kind, group = cands[0].kind, cands[0].group
        if kind == "class":
            tg, fr = targets, inat.POSITIVE_FRACTIONS
        else:
            tg, fr = neg_targets.get(group, {}), inat.NEGATIVE_FRACTIONS
        if not tg:
            continue
        sel, short = inat.select_group(cands, tg, PER_OBSERVER, fr, salt=salt, seed=seed, exclude_photo_ids=used,
                                       eval_place_id=eval_place_id)
        for split, c in sel:
            used.add(c.photo_id)
            rows.append({**c.to_json(), "split": split})
        for split, (got, want) in short.items():
            shortages.append(f"{kind}:{group} {split}: only {got} of {want} available")
    return rows, shortages


# ----------------------------------------------------------------------------- download
def default_fetch_bytes(user_agent):
    return lambda url: inat.http_get(url, user_agent, max_bytes=25 * 1024 * 1024, retries=3)


def download(rows, data_dir: Path, fetch_bytes: Callable[[str], bytes], *, size="medium", min_side=224,
             budget_bytes=2 * 1024 ** 3, limiter=None, degrade_groups=frozenset(), seed=0, log=print) -> dict:
    data_dir = Path(data_dir)
    limiter = limiter or inat.RateLimiter(0.25)
    stats, total_bytes = Counter(), 0
    meta = read_metadata(data_dir / "metadata" / "images.csv")
    ledger = []
    for row in rows:
        c = inat.Candidate.from_json({k: v for k, v in row.items() if k != "split"})
        split = row["split"]
        dest_dir = data_dir / split / group_dir_name(c.kind, c.group)
        existing = list(dest_dir.glob(f"{c.photo_id}.*"))
        if existing:
            try:
                with Image.open(existing[0]) as im:
                    im.load()
                stats["already_downloaded"] += 1
                continue
            except Exception:
                existing[0].unlink()
        if total_bytes >= budget_bytes:
            stats["stopped_by_budget"] = 1
            log(f"media budget of {budget_bytes / 1e9:.1f} GB reached; re-run `download` to continue later")
            break
        data = None
        for url in inat.photo_url_variants(c.url, size):
            limiter.wait()
            try:
                data = fetch_bytes(url)
                break
            except urllib.error.HTTPError as e:
                if e.code in (403, 404):
                    continue
                stats["failed_http"] += 1
                break
            except Exception:
                stats["failed_other"] += 1
                break
        if data is None:
            stats["not_found_or_failed"] += 1
            continue
        total_bytes += len(data)
        try:
            with Image.open(io.BytesIO(data)) as im:
                fmt = im.format
                im.load()
                img = im.convert("RGB")
        except Exception:
            stats["undecodable"] += 1
            continue
        if fmt not in _EXT or min(img.size) < min_side:
            stats["rejected_format_or_size"] += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        degraded = c.group in degrade_groups
        author = c.attribution
        if degraded:
            img = degrade_image(img, random.Random(f"{seed}:{c.photo_id}"))
            dest = dest_dir / f"{c.photo_id}.jpg"
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            payload = buf.getvalue()
            author += " [modified: synthetically degraded]"
        else:
            dest = dest_dir / f"{c.photo_id}{_EXT[fmt]}"
            payload = data
        tmp = dest.with_suffix(dest.suffix + ".part")
        tmp.write_bytes(payload)
        tmp.replace(dest)
        rel = dest.relative_to(data_dir).as_posix()
        meta[rel] = {"path": rel, "split": split, "label": c.group if c.kind == "class" else "", "source": "iNaturalist",
                     "license": c.license.upper(), "author": author, "url": c.observation_url}
        ledger.append({"photo_id": c.photo_id, "path": rel, "sha256": hashlib.sha256(payload).hexdigest(),
                       "bytes": len(payload), "degraded": degraded})
        stats["downloaded"] += 1
        if stats["downloaded"] % 25 == 0:
            write_metadata(data_dir / "metadata" / "images.csv", meta.values())   # checkpoint: safe to interrupt
    write_metadata(data_dir / "metadata" / "images.csv", meta.values())
    if ledger:
        prev = inat.read_jsonl(data_dir / "metadata" / "inat" / "downloaded.jsonl")
        inat.write_jsonl(data_dir / "metadata" / "inat" / "downloaded.jsonl", prev + ledger)
    write_attribution(data_dir, meta)
    stats["bytes"] = total_bytes
    return dict(stats)


def write_attribution(data_dir: Path, meta: dict) -> None:
    lines = ["# Image attribution", "",
             "Photos come from iNaturalist observers under the licences shown. Keep this file with any copy of the images.",
             "Do not redistribute the dataset without honouring each licence (attribution; share-alike where applicable).",
             "", "| file | licence | credit | observation |", "|---|---|---|---|"]
    for r in sorted(meta.values(), key=lambda r: r["path"]):
        if r.get("source") == "iNaturalist":
            lines.append(f"| {r['path']} | {r['license']} | {r['author'].replace('|', '/')} | {r['url']} |")
    (data_dir / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def availability_table(rows, targets=None, neg_targets=None) -> str:
    c = Counter((r["kind"], r["group"], r["split"]) for r in rows)
    out = [f"{'group':<34}{'split':<14}{'selected':>9}"]
    for (kind, group, split), n in sorted(c.items()):
        out.append(f"{kind + ':' + group:<34}{split:<14}{n:>9}")
    return "\n".join(out)
