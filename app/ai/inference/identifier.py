"""identify(image) -> IdentificationResult  (SRS section 7.1)."""
from __future__ import annotations

from app.ai.config import InferenceConfig
from app.ai.contract.labels import CLASSES
from app.ai.contract.result import IdentificationResult
from app.ai.inference.backend import ModelBackend
from app.ai.inference.postprocess import build_result
from app.ai.preprocessing.preprocess import ImageSource, preprocess


class Identifier:
    def __init__(
        self,
        backend: ModelBackend,
        config: InferenceConfig | None = None,
        labels: tuple[str, ...] = CLASSES,
    ):
        self._backend = backend
        self._config = config or InferenceConfig.load()
        self._labels = labels

    @property
    def config(self) -> InferenceConfig:
        return self._config

    def logits(self, image: ImageSource):
        """Raw model output for one image (used by evaluation; same path as identify)."""
        return self._backend.run(preprocess(image))

    def identify(self, image: ImageSource, top_k: int | None = None) -> IdentificationResult:
        logits = self.logits(image)
        return build_result(
            logits, self._labels,
            threshold=self._config.threshold,
            model_name=self._config.model_name,
            top_k=top_k or self._config.top_k,
        )
