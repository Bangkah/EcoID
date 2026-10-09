"""One paste-able summary of a run: dataset structure, training log, export parity, calibration, benchmark.

  python scripts/summarize_run.py            # -> reports/run_summary.md  (also printed)

Contains NO photos, file names, photographer names/ids or coordinates: only counts and numbers. Safe to paste in a chat.
Anything missing is reported as "not found", never guessed.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import _common  # noqa: F401
from app.ai.data.checks import CLASS_DIRS, REQUIRED_NEGATIVE_GROUPS, check_dataset, list_images, read_metadata


def load(path):
    p = Path(path)
    try:
        return json.loads(p.read_text()) if p.is_file() else None
    except (OSError, ValueError):
        return None


def jsonl(path):
    p = Path(path)
    if not p.is_file():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def pct(x):
    return "n/a" if x is None else f"{x * 100:.1f}%"


def dataset_section(data: Path) -> list[str]:
    out = ["## 1. Dataset structure", ""]
    if not data.is_dir():
        return out + [f"`{data}` not found.", ""]
    out += ["| split | " + " | ".join(CLASS_DIRS) + " | total |", "|---|" + "---|" * (len(CLASS_DIRS) + 1)]
    for split in ("train", "val", "evaluation"):
        counts = [len(list_images(data / split / d)) for d in CLASS_DIRS]
        out.append(f"| {split} | " + " | ".join(map(str, counts)) + f" | {sum(counts)} |")
    out += ["", "| negatives | " + " | ".join(REQUIRED_NEGATIVE_GROUPS) + " | total |", "|---|" + "---|" * (len(REQUIRED_NEGATIVE_GROUPS) + 1)]
    for split in ("negative", "negative_val"):
        counts = [len(list_images(data / split / g)) for g in REQUIRED_NEGATIVE_GROUPS]
        out.append(f"| {split} | " + " | ".join(map(str, counts)) + f" | {len(list_images(data / split))} |")

    meta = read_metadata(data / "metadata" / "images.csv")
    if meta:
        lic, src = Counter(r.get("license", "") or "(blank)" for r in meta.values()), Counter(r.get("source", "") or "(blank)" for r in meta.values())
        out += ["", f"Licences: {dict(lic)}", f"Sources: {dict(src)}"]
    sel = jsonl(data / "metadata" / "inat" / "selection.jsonl")
    if sel:
        out += ["", "Distinct photographers per split (iNaturalist selection; ids are not shown):", "",
                "| group kind | split | photos | photographers | max photos from one person |", "|---|---|---|---|---|"]
        by = {}
        for r in sel:
            by.setdefault((r["kind"], r["split"]), []).append(r["user_id"])
        for (kind, split), users in sorted(by.items()):
            out.append(f"| {kind} | {split} | {len(users)} | {len(set(users))} | {max(Counter(users).values())} |")
    res = check_dataset(data, require_metadata=bool(meta))
    out += ["", f"`check_dataset`: **{len(res.errors)} error(s), {len(res.warnings)} warning(s)**"]
    for line in (res.errors[:10] + res.warnings[:6]):
        out.append(f"- {line}")
    return out + [""]


def training_section(run: Path) -> list[str]:
    out = ["## 2. Training", ""]
    meta = load(run / "train_meta.json")
    if not meta:
        return out + [f"`{run}/train_meta.json` not found.", ""]
    a, best, hist = meta.get("args", {}), meta.get("best", {}), meta.get("history", [])
    out += [f"arch `{meta.get('arch')}`, input {meta.get('input_size')}, seed {a.get('seed')}, batch {a.get('batch_size')}, "
            f"epochs head/finetune {a.get('epochs_head')}/{a.get('epochs_finetune')}, unfreeze {a.get('unfreeze_blocks')} blocks, "
            f"lr {a.get('lr_head')}/{a.get('lr_finetune')}",
            f"train images {sum((meta.get('train_counts') or {}).values())} {meta.get('train_counts')}",
            f"val images {sum((meta.get('val_counts') or {}).values())} {meta.get('val_counts')}",
            f"versions {meta.get('versions')}", "",
            f"**best val accuracy {pct(best.get('val_acc'))} at epoch {best.get('epoch')} ({best.get('stage')})**", "",
            "| epoch | stage | train loss | train acc | val loss | val acc |", "|---|---|---|---|---|---|"]
    shown = hist if len(hist) <= 14 else hist[:4] + hist[-8:]
    for h in shown:
        out.append(f"| {h['epoch']} | {h['stage']} | {h['train_loss']:.3f} | {pct(h['train_acc'])} | {h['val_loss']:.3f} | {pct(h['val_acc'])} |")
    notes = []
    if hist:
        last, top = hist[-1], max(h["val_acc"] for h in hist)
        n_cls = len(meta.get("classes") or CLASS_DIRS)
        if top <= 1.5 / n_cls:
            notes.append("best val accuracy is near chance level: suspect labels, class order or preprocessing, not tuning")
        if last["train_acc"] - last["val_acc"] > 0.25:
            notes.append("large train/val gap: more varied photographers beat more epochs")
        if len(hist) >= 2 and hist[1]["val_acc"] >= 0.999 and top >= 0.999:
            notes.append("100% val accuracy within 2 epochs: check for leakage between train and val")
    out += ["", "Heuristic observations (not verdicts): " + ("; ".join(notes) if notes else "none triggered")]
    return out + [""]


def export_section(onnx: Path) -> list[str]:
    out = ["## 3. ONNX export", ""]
    side = load(str(onnx) + ".json")
    if not side:
        return out + [f"`{onnx}.json` not found.", ""]
    out += [f"size {side.get('size_bytes', 0) / 1e6:.1f} MB, sha256 `{str(side.get('sha256'))[:16]}…`, opset {side.get('opset')}",
            f"parity checks: `{side.get('parity_checks')}`", f"classes (order): {side.get('classes')}"]
    return out + [""]


def eval_section(reports: Path) -> list[str]:
    out = ["## 4. Threshold calibration (validation data)", ""]
    cal = load(reports / "calibration.json")
    if not cal:
        out.append("`calibration.json` not found.")
    else:
        rec = cal.get("recommended")
        out.append(f"constraints {cal.get('constraints')}")
        out.append("no threshold satisfies the constraints" if rec is None else
                   f"recommended threshold **{rec['threshold']}** (coverage {pct(rec['coverage'])}, selective accuracy {pct(rec['selective_accuracy'])}, "
                   f"negative rejection {pct(rec.get('negative_rejection_rate'))})")
    out += ["", "## 5. Benchmark (evaluation data)", ""]
    b = load(reports / "benchmark.json")
    if not b:
        out.append("`benchmark.json` not found.")
    else:
        n, op = b.get("eval_samples", 0), b.get("operating_point", {})
        k1, k3 = round((b.get("top1_accuracy") or 0) * n), round((b.get("top3_accuracy") or 0) * n)
        out += [f"threshold {b.get('threshold')}, evaluation samples {n}, negatives {b.get('negative_samples')}",
                f"top-1 **{k1}/{n}** ({pct(b.get('top1_accuracy'))}), top-3 **{k3}/{n}** ({pct(b.get('top3_accuracy'))})",
                f"negatives rejected {op.get('negatives_rejected', 'n/a')}/{op.get('negatives_total', 'n/a')}, selective accuracy {pct(op.get('selective_accuracy'))}, "
                f"coverage {pct(op.get('coverage'))}, ECE {b.get('ece', float('nan')):.3f}",
                f"latency (preprocess+inference+softmax) median {b['latency']['median_ms']:.1f} ms, p95 {b['latency']['p95_ms']:.1f} ms; hardware {b['hardware'].get('processor')} / {b['hardware'].get('cpu_count')} CPUs"]
        cm = b.get("confusion_matrix")
        if cm:
            out += ["", "Confusion matrix (rows = true, cols = predicted, class order as in `labels.py`):", "```"] + [" ".join(f"{v:3d}" for v in row) for row in cm] + ["```"]
    lat = load(reports / "latency.json")
    if lat:
        out += ["", f"`benchmark.py`: total median {lat['total_ms']['median_ms']:.1f} ms, p95 {lat['total_ms']['p95_ms']:.1f} ms, NFR-002 {'met' if lat['meets_target'] else 'NOT met'} ({lat['source']})"]
    return out + [""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--run", default="models/checkpoints/run1")
    ap.add_argument("--onnx", default="models/ecoid.onnx")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    lines = ["# EcoID run summary", "", "Counts and numbers only (no photos, names, ids or coordinates).", ""]
    lines += dataset_section(Path(a.data)) + training_section(Path(a.run)) + export_section(Path(a.onnx)) + eval_section(Path(a.reports))
    text = "\n".join(lines)
    out = Path(a.out or Path(a.reports) / "run_summary.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text + "\n", encoding="utf-8")
    print(text)
    print(f"\n(written to {out})")


if __name__ == "__main__":
    main()
