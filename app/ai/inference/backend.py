"""Model backends. The Identifier only sees `ModelBackend`, so the model is
swappable (NFR-005) and tests can use a fake without any model file."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np


class ModelBackend(Protocol):
    def run(self, tensor: np.ndarray) -> np.ndarray:
        """(1,3,224,224) float32 -> raw logits of shape (1,N) or (N,)."""
        ...


class OnnxBackend:
    """CPU-only ONNX Runtime backend (SRS FR-003). Model must output LOGITS."""

    def __init__(self, model_path: str | Path, num_threads: int | None = None):
        import onnxruntime as ort

        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")
        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
        self._session = ort.InferenceSession(
            str(path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        inp = self._session.get_inputs()[0]
        self._input_name = inp.name
        self.input_shape = inp.shape

    def run(self, tensor: np.ndarray) -> np.ndarray:
        out = self._session.run(None, {self._input_name: tensor})[0]
        return np.asarray(out)
