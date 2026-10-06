"""python -m app --model models/ecoid.onnx   ->   http://127.0.0.1:8765"""
from __future__ import annotations

import argparse
import sys
import webbrowser
from pathlib import Path

from app.ai.config import DEFAULT_CONFIG_PATH, InferenceConfig
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier
from app.observation.manager import ObservationManager
from app.storage.store import ObservationStore
from app.ui.server import build_model_info, make_server


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m app", description="EcoID — offline AI field mapper")
    ap.add_argument("--model", required=True, help="path to the .onnx model")
    ap.add_argument("--data-dir", default=str(Path.home() / ".ecoid"), help="where photos + observations are stored")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--host", default="127.0.0.1",
                    help="default is local-only; 0.0.0.0 exposes the app (no login!) to your network")
    ap.add_argument("--port", type=int, default=8765, help="0 = pick a free port")
    ap.add_argument("--open", action="store_true", help="open the browser")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args(argv)

    config = InferenceConfig.load(a.config)
    try:
        backend = OnnxBackend(a.model)
    except FileNotFoundError as e:
        sys.exit(f"error: {e}")
    manager = ObservationManager(Identifier(backend, config), ObservationStore(a.data_dir))
    purged = manager.purge_drafts()
    srv = make_server(manager, build_model_info(config, a.model), a.host, a.port, verbose=a.verbose)

    host, port = srv.server_address[:2]
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: listening on a non-loopback address. EcoID has no login; anyone on this network can use it.",
              file=sys.stderr)
    print(f"EcoID serving on {url}  (data: {a.data_dir}; abandoned drafts removed: {purged})", flush=True)
    if a.open:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
