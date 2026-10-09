import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.ai.config import InferenceConfig
from app.ai.contract.labels import CLASSES
from app.ai.contract.result import Status
from app.ai.inference.backend import OnnxBackend
from app.ai.inference.identifier import Identifier
from app.ai.inference.postprocess import build_result as _build_result, softmax
from app.ai.preprocessing.preprocess import (
    InvalidImageError, load_image, preprocess,
)

ROOT = Path(__file__).resolve().parents[2]


def build_result(logits, labels, threshold=0.65, model_name="test-model", **kw):
    return _build_result(logits, labels, threshold=threshold, model_name=model_name, **kw)


def solid(color, size=(300, 200), mode="RGB"):
    return Image.new(mode, size, color)


class FakeBackend:
    def __init__(self, logits):
        self.logits = np.array(logits, dtype=np.float32)
        self.calls = 0

    def run(self, tensor):
        assert tensor.shape == (1, 3, 224, 224) and tensor.dtype == np.float32
        self.calls += 1
        return self.logits[np.newaxis, :]


class PreprocessTests(unittest.TestCase):
    def test_shape_dtype(self):
        t = preprocess(solid((10, 200, 30)))
        self.assertEqual(t.shape, (1, 3, 224, 224))
        self.assertEqual(t.dtype, np.float32)
        self.assertTrue(t.flags["C_CONTIGUOUS"])

    def test_normalization_values(self):
        t = preprocess(solid((255, 255, 255)))
        expected = (1.0 - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        for c in range(3):
            self.assertTrue(np.allclose(t[0, c], expected[c], atol=1e-5))

    def test_channel_order_is_rgb(self):
        t = preprocess(solid((255, 0, 0)))
        self.assertGreater(t[0, 0].mean(), t[0, 1].mean())  # R high, G low

    def test_grayscale_and_rgba_and_palette(self):
        self.assertEqual(preprocess(solid(128, mode="L")).shape, (1, 3, 224, 224))
        self.assertEqual(preprocess(solid((1, 2, 3, 128), mode="RGBA")).shape, (1, 3, 224, 224))
        self.assertEqual(preprocess(solid(5, mode="P")).shape, (1, 3, 224, 224))

    def test_transparent_flattens_to_white(self):
        img = load_image(solid((0, 0, 0, 0), mode="RGBA"))
        self.assertEqual(img.getpixel((0, 0)), (255, 255, 255))

    def test_exif_orientation_applied(self):
        img = Image.new("RGB", (40, 20), (0, 0, 0))
        exif = Image.Exif()
        exif[0x0112] = 6  # rotate 90 CW when displayed
        buf = io.BytesIO()
        img.save(buf, "JPEG", exif=exif)
        out = load_image(buf.getvalue())
        self.assertEqual(out.size, (20, 40))

    def test_bytes_and_path_inputs(self):
        buf = io.BytesIO()
        solid((9, 9, 9)).save(buf, "PNG")
        a = preprocess(buf.getvalue())
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            p = Path(d) / "x.png"
            p.write_bytes(buf.getvalue())
            b = preprocess(p)
            c = preprocess(str(p))
        self.assertTrue(np.array_equal(a, b) and np.array_equal(b, c))

    def test_deterministic(self):
        img = solid((50, 60, 70))
        self.assertTrue(np.array_equal(preprocess(img), preprocess(img)))

    def test_invalid_inputs(self):
        with self.assertRaises(InvalidImageError):
            preprocess(b"not an image")
        with self.assertRaises(InvalidImageError):
            preprocess(b"")
        with self.assertRaises((InvalidImageError, FileNotFoundError, OSError)):
            preprocess("/nonexistent/file.jpg")

    def test_truncated_file(self):
        buf = io.BytesIO()
        Image.effect_noise((200, 200), 80).convert("RGB").save(buf, "JPEG")
        with self.assertRaises(InvalidImageError):
            preprocess(buf.getvalue()[: len(buf.getvalue()) // 3])


class PostprocessTests(unittest.TestCase):
    def test_softmax_sums_to_one_and_stable(self):
        p = softmax(np.array([1000.0, 1001.0, 999.0]))
        self.assertAlmostEqual(p.sum(), 1.0)
        self.assertTrue(np.all(np.isfinite(p)))

    def test_nan_rejected(self):
        with self.assertRaises(ValueError):
            softmax(np.array([1.0, np.nan]))

    def test_topk_sorted_and_labeled(self):
        r = build_result(np.array([0.1, 3.0, 0.2, 1.0, 0.0]), CLASSES, top_k=3)
        self.assertEqual(len(r.candidates), 3)
        self.assertEqual(r.candidates[0].label, CLASSES[1])
        scores = [c.score for c in r.candidates]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_capped_to_num_classes(self):
        r = build_result(np.zeros(5), CLASSES, top_k=99)
        self.assertEqual(len(r.candidates), 5)

    def test_tie_break_is_stable(self):
        r = build_result(np.zeros(5), CLASSES)
        self.assertEqual([c.label for c in r.candidates], list(CLASSES[:3]))

    def test_threshold_boundary_inclusive(self):
        # 2 classes, p(top)=0.65 exactly -> IDENTIFIED (>=)
        logit = np.log(0.65 / 0.35)
        r = build_result(np.array([logit, 0.0]), ("a", "b"), )
        self.assertEqual(r.status, Status.IDENTIFIED)
        r2 = build_result(np.array([logit - 0.01, 0.0]), ("a", "b"))
        self.assertEqual(r2.status, Status.LOW_CONFIDENCE)

    def test_uniform_output_is_low_confidence(self):
        self.assertEqual(build_result(np.zeros(5), CLASSES).status, Status.LOW_CONFIDENCE)

    def test_wrong_output_size_fails_loudly(self):
        with self.assertRaises(ValueError):
            build_result(np.zeros(7), CLASSES)

    def test_batch_dim_flattened(self):
        r = build_result(np.array([[9.0, 0, 0, 0, 0]]), CLASSES)
        self.assertEqual(r.top1.label, CLASSES[0])

    def test_to_dict_matches_contract(self):
        d = build_result(np.array([9.0, 0, 0, 0, 0]), CLASSES).to_dict()
        self.assertEqual(set(d), {"candidates", "status", "model"})
        self.assertEqual(d["status"], "IDENTIFIED")
        self.assertEqual(set(d["candidates"][0]), {"label", "score"})


class IdentifierTests(unittest.TestCase):
    def test_identified_flow(self):
        be = FakeBackend([8, 0, 0, 0, 0])
        r = Identifier(be).identify(solid((0, 128, 0)))
        self.assertEqual(r.status, Status.IDENTIFIED)
        self.assertEqual(r.top1.label, "Mangifera indica")
        self.assertEqual(be.calls, 1)

    def test_low_confidence_flow(self):
        r = Identifier(FakeBackend([0.5, 0.4, 0.3, 0.2, 0.1])).identify(solid((1, 1, 1)))
        self.assertEqual(r.status, Status.LOW_CONFIDENCE)

    def test_invalid_image_never_reaches_model(self):
        be = FakeBackend([1, 0, 0, 0, 0])
        with self.assertRaises(InvalidImageError):
            Identifier(be).identify(b"garbage")
        self.assertEqual(be.calls, 0)

    def test_backend_is_swappable(self):
        a = Identifier(FakeBackend([9, 0, 0, 0, 0])).identify(solid((1, 1, 1)))
        b = Identifier(FakeBackend([0, 0, 0, 0, 9])).identify(solid((1, 1, 1)))
        self.assertNotEqual(a.top1.label, b.top1.label)


class ConfigTests(unittest.TestCase):
    def _write(self, text):
        d = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(d.cleanup)
        p = Path(d.name) / "c.toml"
        p.write_text(text)
        return p

    def test_default_config_loads(self):
        c = InferenceConfig.load()
        self.assertEqual(c.threshold, 0.65)
        self.assertEqual(c.model_name, "EcoID Vision Model")

    def test_invalid_values_rejected(self):
        for bad in ('model_name="x"\nthreshold=0\n', 'model_name="x"\nthreshold=1.5\n',
                    'model_name=""\nthreshold=0.5\n', 'model_name="x"\nthreshold=0.5\ntop_k=0\n'):
            with self.assertRaises(ValueError):
                InferenceConfig.load(self._write(bad))

    def test_unknown_key_rejected(self):
        with self.assertRaises(ValueError):
            InferenceConfig.load(self._write('model_name="x"\nthreshold=0.5\ntreshold=0.9\n'))

    def test_threshold_and_name_flow_from_config(self):
        logits = [2.0, 0, 0, 0, 0]  # top score ~0.65-0.7
        strict = Identifier(FakeBackend(logits), InferenceConfig("strict", 0.99))
        loose = Identifier(FakeBackend(logits), InferenceConfig("loose", 0.30))
        a, b = strict.identify(solid((1, 2, 3))), loose.identify(solid((1, 2, 3)))
        self.assertEqual(a.status, Status.LOW_CONFIDENCE)
        self.assertEqual(b.status, Status.IDENTIFIED)
        self.assertEqual((a.model, b.model), ("strict", "loose"))
        self.assertEqual(a.threshold, 0.99)


class OnnxIntegrationTests(unittest.TestCase):
    """Real ONNX Runtime path using the tiny synthetic model (not a plant model)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.model = Path(cls.tmp.name) / "tiny.onnx"
        subprocess.check_call(
            [sys.executable, str(ROOT / "scripts" / "make_test_onnx.py"), str(cls.model)]
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_runs_on_cpu_end_to_end(self):
        ident = Identifier(OnnxBackend(self.model))
        red = ident.identify(solid((255, 0, 0)))
        blue = ident.identify(solid((0, 0, 255)))
        self.assertEqual(red.top1.label, CLASSES[0])
        self.assertEqual(blue.top1.label, CLASSES[2])
        self.assertAlmostEqual(sum(c.score for c in red.candidates) <= 1.0 + 1e-9, True)

    def test_repeatable(self):
        ident = Identifier(OnnxBackend(self.model))
        img = solid((20, 90, 200))
        self.assertEqual(ident.identify(img), ident.identify(img))

    def test_missing_model_file(self):
        with self.assertRaises(FileNotFoundError):
            OnnxBackend("/no/such/model.onnx")



if __name__ == "__main__":
    unittest.main()
