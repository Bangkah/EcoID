"""Writes a tiny deterministic ONNX model (no `onnx` package needed).

x[1,3,224,224] -> GlobalAveragePool -> Flatten -> MatMul(W[3,5]) -> logits[1,5]
Used ONLY to test the real ONNX Runtime path in Phase 1 (not a plant model).
"""
import struct
import sys


def _varint(n):
    out = b""
    while True:
        b = n & 0x7F
        n >>= 7
        out += bytes([b | (0x80 if n else 0)])
        if not n:
            return out


def _key(field, wt):
    return _varint((field << 3) | wt)


def _ld(field, payload):  # length-delimited
    return _key(field, 2) + _varint(len(payload)) + payload


def _s(field, text):
    return _ld(field, text.encode())


def _i(field, v):
    return _key(field, 0) + _varint(v)


def _value_info(name, dims):
    shape = b"".join(_ld(1, _i(1, d)) for d in dims)
    tensor_t = _i(1, 1) + _ld(2, shape)  # elem_type=FLOAT, shape
    return _s(1, name) + _ld(2, _ld(1, tensor_t))


def _node(op, ins, outs, attrs=b""):
    p = b"".join(_s(1, i) for i in ins) + b"".join(_s(2, o) for o in outs)
    return p + _s(4, op) + attrs


def build(path):
    # W: 3 channels -> 5 classes (each class "prefers" a different channel mix)
    W = [
        2.0, -1.0, -1.0, 0.5, 0.0,
        -1.0, 2.0, -1.0, 0.5, 0.0,
        -1.0, -1.0, 2.0, 0.5, 0.0,
    ]
    init = (
        _i(1, 3) + _i(1, 5) + _i(2, 1)
        + _ld(4, struct.pack("<15f", *W)) + _s(8, "W")
    )
    flatten_attr = _ld(5, _s(1, "axis") + _i(3, 1) + _i(20, 2))
    graph = (
        _ld(1, _node("GlobalAveragePool", ["x"], ["g"]))
        + _ld(1, _node("Flatten", ["g"], ["f"], flatten_attr))
        + _ld(1, _node("MatMul", ["f", "W"], ["logits"]))
        + _s(2, "tiny")
        + _ld(5, init)
        + _ld(11, _value_info("x", [1, 3, 224, 224]))
        + _ld(12, _value_info("logits", [1, 5]))
    )
    model = _i(1, 8) + _ld(8, _s(1, "") + _i(2, 13)) + _ld(7, graph)
    with open(path, "wb") as f:
        f.write(model)


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "models/tiny_test.onnx")
