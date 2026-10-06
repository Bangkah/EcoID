"""Manual label QC (SRS: web-scraped data needs hand QC). Contact sheets to eyeball, a reject list to act on."""
from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from app.ai.data.checks import list_images, read_metadata, write_metadata


def contact_sheet(paths: list[Path], out_path: Path, *, cols=6, rows=5, thumb=170, title="") -> int:
    """Grid of thumbnails, each captioned with the file's id (its stem). Returns how many were drawn."""
    paths = paths[: cols * rows]
    cap, head = 16, 22 if title else 0
    sheet = Image.new("RGB", (cols * thumb, head + rows * (thumb + cap)), "white")
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((4, 4), title, fill="black")
    for i, p in enumerate(paths):
        x, y = (i % cols) * thumb, head + (i // cols) * (thumb + cap)
        try:
            with Image.open(p) as im:
                t = ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (thumb - 4, thumb - 4))
        except Exception:
            t = Image.new("RGB", (thumb - 4, thumb - 4), "red")
        sheet.paste(t, (x + 2, y + 2))
        d.text((x + 3, y + thumb - 1), p.stem[:24], fill="black")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=88)
    return len(paths)


def make_sheets(folder: Path, out_dir: Path, per_sheet=30, **kw) -> list[Path]:
    files = list_images(folder)
    out = []
    for i in range(0, len(files), per_sheet):
        p = out_dir / f"{folder.parent.name}__{folder.name}__{i // per_sheet + 1:02d}.jpg"
        contact_sheet(files[i:i + per_sheet], p, title=f"{folder.parent.name}/{folder.name}  #{i // per_sheet + 1}", **kw)
        out.append(p)
    return out


def read_reject_ids(path: Path) -> set[str]:
    """One id (the file name without extension) per line; '#' starts a comment."""
    ids = set()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        tok = line.split("#", 1)[0].strip()
        if tok:
            ids.add(Path(tok).stem)
    return ids


def apply_rejections(data_dir: Path, ids: set[str], dry_run=False) -> list[str]:
    """Move rejected images to data/_rejected/, drop their metadata rows, remember ids so `select` never picks them again."""
    data_dir = Path(data_dir)
    moved = []
    meta = read_metadata(data_dir / "metadata" / "images.csv")
    for split_dir in sorted(p for p in data_dir.iterdir() if p.is_dir() and not p.name.startswith(("_", "metadata"))):
        for f in list_images(split_dir):
            if f.stem in ids:
                rel = f.relative_to(data_dir)
                moved.append(rel.as_posix())
                if not dry_run:
                    dest = data_dir / "_rejected" / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(f), str(dest))
                    meta.pop(rel.as_posix(), None)
    if moved and not dry_run:
        write_metadata(data_dir / "metadata" / "images.csv", meta.values())
        log = data_dir / "metadata" / "rejected_photo_ids.txt"
        known = set(log.read_text().split()) if log.exists() else set()
        log.write_text("\n".join(sorted(known | ids)) + "\n")
    return moved
