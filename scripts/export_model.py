"""Export a checkpoint to ONNX (logits output) and PROVE it matches PyTorch + the serving pipeline.

  python scripts/export_model.py --checkpoint models/checkpoints/run1 \
      --out models/ecoid.onnx --sample-dir data/val
Fails (exit 1) if parity checks fail.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

import _torch_common as C
from app.ai.data.checks import list_images
from app.ai.preprocessing.preprocess import load_image, preprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="run dir containing best.pt and train_meta.json")
    ap.add_argument("--out", default="models/ecoid.onnx")
    ap.add_argument("--sample-dir", default=None, help="folder of real images for preprocessing parity")
    ap.add_argument("--n-samples", type=int, default=20)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    run = Path(a.checkpoint)
    meta = json.loads((run / "train_meta.json").read_text(encoding="utf-8"))
    arch = meta["arch"].split("_")[-1]
    model = C.build_model(arch, pretrained=False)           # loading OUR fine-tuned weights, not scratch training
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, 224, 224)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=a.opset,
              do_constant_folding=True, dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(out), dynamo=False, **kw)   # stable legacy exporter
    except TypeError:                                                     # older torch without `dynamo`
        torch.onnx.export(model, dummy, str(out), **kw)
    onnx.checker.check_model(onnx.load(str(out)))

    failures, checks = [], {}
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])

    # 1) raw-tensor parity torch vs ONNX Runtime
    rng = np.random.default_rng(0)
    x = rng.normal(size=(8, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        t_logits = model(torch.from_numpy(x)).numpy()
    o_logits = sess.run(None, {"input": x})[0]
    diff = float(np.abs(t_logits - o_logits).max())
    checks["tensor_max_abs_logit_diff"] = diff
    if diff > 1e-3 or o_logits.shape != (8, 5) or not (t_logits.argmax(1) == o_logits.argmax(1)).all():
        failures.append(f"torch vs onnx mismatch (max diff {diff})")

    # 2) preprocessing parity: torch eval transform vs app.ai.preprocessing on REAL images
    if a.sample_dir:
        paths = list_images(Path(a.sample_dir))[: a.n_samples]
        pre, agree = [], 0
        for p in paths:
            t_in = C.EVAL_TF(load_image(p)).unsqueeze(0).numpy()
            s_in = preprocess(p)
            pre.append(float(np.abs(t_in - s_in).max()))
            with torch.no_grad():
                agree += int(model(torch.from_numpy(t_in)).argmax(1).item()
                             == int(sess.run(None, {"input": s_in})[0].argmax(1)[0]))
        checks["preprocess_max_abs_diff"] = max(pre) if pre else None
        checks["end_to_end_top1_agreement"] = f"{agree}/{len(paths)}"
        if not pre:
            failures.append("sample-dir has no images")
        elif max(pre) > 1e-4 or agree != len(paths):
            failures.append(f"preprocessing parity failed (max diff {max(pre):.2e}, agree {agree}/{len(paths)})")
    else:
        checks["preprocess_parity"] = "SKIPPED (pass --sample-dir to verify)"

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    sidecar = {"file": out.name, "sha256": sha, "size_bytes": out.stat().st_size,
               "classes": meta["classes"], "input": "float32 [N,3,224,224], RGB, /255, ImageNet mean/std, direct resize",
               "output": "raw logits [N,5] (softmax applied in post-processing)",
               "source_checkpoint": str(run), "train_fingerprint": meta.get("train_fingerprint"),
               "versions": meta.get("versions"), "parity_checks": checks, "opset": a.opset}
    Path(str(out) + ".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    print(json.dumps(sidecar["parity_checks"], indent=2))
    print(f"size {out.stat().st_size/1e6:.1f} MB  sha256 {sha[:16]}...")
    if failures:
        print("EXPORT FAILED:", *failures, sep="\n  ")
        sys.exit(1)
    print("Export OK")


if __name__ == "__main__":
    main()
