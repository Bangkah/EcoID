"""Build the real dataset from iNaturalist (licensed photos only), in four resumable steps.

  python scripts/fetch_inat.py probe    --contact you@example.com        # 1 request: check API field names
  python scripts/fetch_inat.py plan     --contact you@example.com        # ~hundreds of API requests, no images
  python scripts/fetch_inat.py select                                    # offline: choose photos, split BY OBSERVER
  python scripts/fetch_inat.py download --contact you@example.com        # images + metadata + ATTRIBUTION.md

Read docs/data_collection.md first. iNaturalist asks for ~1 request/second and says the API is not for bulk
scraping; this tool stays far below its limits and refuses to exceed --max-requests / --budget-gb.
"""
import argparse
import sys
from pathlib import Path

import _common  # noqa: F401
from app.ai.data import inat, inat_pipeline as P

ROOT = Path(__file__).resolve().parents[1]


def user_agent(contact):
    return f"EcoID-dataset-builder/0.1 (student research project; contact: {contact})"


def paths(a):
    data = Path(a.data)
    return data, data / "metadata" / "inat" / "candidates", data / "metadata" / "inat" / "selection.jsonl"


def cmd_probe(a):
    c = inat.INatClient(user_agent(a.contact), max_requests=5)
    t = c.resolve_taxon("Mangifera indica")
    print("taxon:", t["id"], t.get("name"), t.get("rank"), t.get("observations_count"))
    res = c.get("/observations", {"taxon_id": t["id"], "photos": "true", "photo_license": "cc-by,cc0,cc-by-sa",
                                  "per_page": 1, "quality_grade": "research"}).get("results", [])
    if not res:
        sys.exit("no observation returned -- check connectivity")
    o = res[0]
    need = ["id", "quality_grade", "user", "taxon", "photos", "uri", "place_ids", "num_identification_agreements",
            "num_identification_disagreements", "captive"]
    print("observation keys present:", {k: (k in o) for k in need})
    print("first photo:", {k: o["photos"][0].get(k) for k in ("id", "url", "license_code", "attribution", "original_dimensions")})
    print("\nIf any key above is False/None where you expect data, the API changed: tell me before running `plan`.")


def cmd_plan(a):
    data, cand_dir, _ = paths(a)
    cfg = P.load_config(a.taxa)
    c = inat.INatClient(user_agent(a.contact), max_requests=a.max_requests)
    place_id = a.place_id
    if a.place and not place_id:
        found = c.search_places(a.place)
        for p in found:
            print(f"  place {p['id']}: {p.get('display_name') or p.get('name')}")
        sys.exit("Pick the right id above and re-run with --place-id ID")
    licenses = inat.DEFAULT_LICENSES + (inat.NC_LICENSES if a.allow_nc else ())
    groups = sum(1 for _ in P.iter_groups(cfg))
    print(f"licences: {licenses}\ngrades: {a.grades}\nplace_id: {place_id}\nrequest budget: {a.max_requests} "
          f"({groups} groups, <= {a.max_pages} pages of 200 per taxon+grade), ~1 request/second\n")
    try:
        P.plan(c, cfg, cand_dir, licenses=licenses, grades=a.grades.split(","), place_id=place_id,
               max_candidates=a.max_candidates, max_pages=a.max_pages, min_side=a.min_side,
               min_agreements=a.min_agreements, refresh=a.refresh)
    except inat.RequestBudgetExceeded as e:
        print(f"\n{e}. Progress is saved per group; re-run `plan` to continue the remaining groups.")
    print(f"\nused {c.requests} API requests. Candidate lists are in {cand_dir}. Next: `select`.")


def cmd_select(a):
    data, cand_dir, sel_path = paths(a)
    rej = data / "metadata" / "rejected_photo_ids.txt"
    exclude = {int(x) for x in rej.read_text().split() if x.isdigit()} if rej.exists() else set()
    targets = {"train": a.train, "val": a.val, "evaluation": a.eval}
    neg = {"lookalike_plants": {"negative": a.neg_lookalike, "negative_val": a.negval_lookalike},
           "non_plant": {"negative": a.neg_non_plant, "negative_val": a.negval_non_plant},
           "low_quality": {"negative": a.neg_low_quality, "negative_val": a.negval_low_quality}}
    rows, shortages = P.select(cand_dir, targets=targets, neg_targets=neg, seed=a.seed, exclude_ids=exclude,
                               eval_place_id=a.eval_place_id)
    inat.write_jsonl(sel_path, rows)
    print(P.availability_table(rows))
    if shortages:
        print("\nSHORTAGES (not enough photographers/photos for the target):")
        for s in shortages:
            print("  -", s)
        print("Fix: raise --max-candidates / --max-pages in `plan --refresh`, widen the place, add taxa, or lower the target.")
    print(f"\nselection written to {sel_path} ({len(rows)} photos). Next: `download`.")


def cmd_download(a):
    data, _, sel_path = paths(a)
    rows = inat.read_jsonl(sel_path)
    if not rows:
        sys.exit("empty selection: run `select` first")
    cfg = P.load_config(a.taxa)
    degrade = {g for k, g, _, d in P.iter_groups(cfg) if d}
    stats = P.download(rows, data, P.default_fetch_bytes(user_agent(a.contact)), size=a.size,
                       budget_bytes=int(a.budget_gb * 1024 ** 3), degrade_groups=degrade, seed=a.seed)
    print("download stats:", stats)
    print("\nNext: python scripts/check_dataset.py  ->  python scripts/make_contact_sheets.py  (look at every sheet!)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--taxa", default=str(ROOT / "config" / "inat_taxa.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)

    def contact(p):
        p.add_argument("--contact", required=True, help="your email or iNaturalist username (sent in the User-Agent)")

    p = sub.add_parser("probe"); contact(p); p.set_defaults(fn=cmd_probe)
    p = sub.add_parser("plan"); contact(p)
    p.add_argument("--grades", default="research,casual", help="cultivated plants are 'casual'; research grade = wild only")
    p.add_argument("--allow-nc", action="store_true", help="also allow CC BY-NC / BY-NC-SA photos (non-commercial)")
    p.add_argument("--place-id", type=int); p.add_argument("--place", help="search a place name, prints ids")
    p.add_argument("--max-candidates", type=int, default=2500, help="per group")
    p.add_argument("--max-pages", type=int, default=20); p.add_argument("--max-requests", type=int, default=1500)
    p.add_argument("--min-side", type=int, default=300); p.add_argument("--min-agreements", type=int, default=1)
    p.add_argument("--refresh", action="store_true"); p.set_defaults(fn=cmd_plan)
    p = sub.add_parser("select")
    p.add_argument("--train", type=int, default=400); p.add_argument("--val", type=int, default=60)
    p.add_argument("--eval", type=int, default=10)
    for g, (n, v) in {"lookalike": (50, 20), "non-plant": (30, 10), "low-quality": (30, 10)}.items():
        p.add_argument(f"--neg-{g}", type=int, default=n); p.add_argument(f"--negval-{g}", type=int, default=v)
    p.add_argument("--eval-place-id", type=int, help="evaluation photos must lie in this place (e.g. Indonesia)")
    p.add_argument("--seed", type=int, default=0); p.set_defaults(fn=cmd_select)
    p = sub.add_parser("download"); contact(p)
    p.add_argument("--size", choices=inat.PHOTO_SIZES, default="medium")
    p.add_argument("--budget-gb", type=float, default=2.0); p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_download)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
