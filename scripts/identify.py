"""CLI smoke test:  python scripts/identify.py IMAGE --model models/x.onnx"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ai.config import DEFAULT_CONFIG_PATH, InferenceConfig
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--model", required=True)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    a = ap.parse_args()
    result = Identifier(OnnxBackend(a.model), InferenceConfig.load(a.config)).identify(a.image, top_k=a.top_k)
    print(json.dumps(result.to_dict(), indent=2))
    if result.status.value == "LOW_CONFIDENCE":
        print("\n⚠ Low confidence — manual verification required", file=sys.stderr)


if __name__ == "__main__":
    main()
