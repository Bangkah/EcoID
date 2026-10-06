"""Synthetic low-quality images for the SRS 'blur / low-quality' negative group."""
from __future__ import annotations

import io
import random

from PIL import Image, ImageEnhance, ImageFilter


def degrade_image(img: Image.Image, rng: random.Random) -> Image.Image:
    img = img.convert("RGB")
    ops = rng.sample(["blur", "pixelate", "dark", "bright", "tiny_crop", "jpeg"], k=rng.choice([1, 2]))
    for op in ops:
        w, h = img.size
        if op == "blur":
            img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(5, 12)))
        elif op == "pixelate":
            f = rng.randint(14, 30)
            img = img.resize((max(2, w // f), max(2, h // f)), Image.BILINEAR).resize((w, h), Image.NEAREST)
        elif op == "dark":
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.08, 0.3))
        elif op == "bright":
            img = ImageEnhance.Brightness(img).enhance(rng.uniform(2.2, 3.2))
        elif op == "tiny_crop":
            s = rng.uniform(0.06, 0.15)
            cw, ch = max(4, int(w * s)), max(4, int(h * s))
            x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch)).resize((w, h), Image.BILINEAR)
        elif op == "jpeg":
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=rng.randint(3, 10))
            img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img
