"""Shared test helpers (images, fake backend, offline guard)."""
import io
import socket
from contextlib import contextmanager

import numpy as np
from PIL import Image

_RNG = np.random.default_rng(3)


def image_bytes(rgb=(230, 20, 20), size=(1200, 900), fmt="JPEG", jitter=8):
    a = np.clip(np.array(rgb) + _RNG.integers(-jitter, jitter + 1, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, fmt)
    return buf.getvalue()


class FakeBackend:
    def __init__(self, logits=(8, 0, 0, 0, 0)):
        self.logits = np.array(logits, dtype=np.float32)

    def run(self, tensor):
        return self.logits[np.newaxis, :]


@contextmanager
def block_network():
    """Fail (and record) any socket connection that is not loopback / unix-socket / the test server."""
    attempts = []
    real_connect, real_connect_ex = socket.socket.connect, socket.socket.connect_ex
    real_gai = socket.getaddrinfo

    def is_local(addr):
        if isinstance(addr, (str, bytes)):          # AF_UNIX
            return True
        return str(addr[0]) in ("127.0.0.1", "::1", "localhost")

    def connect(self, addr):
        if not is_local(addr):
            attempts.append(addr)
            raise OSError("network blocked by test")
        return real_connect(self, addr)

    def connect_ex(self, addr):
        if not is_local(addr):
            attempts.append(addr)
            return 111
        return real_connect_ex(self, addr)

    def gai(host, *a, **k):
        if host not in ("127.0.0.1", "::1", "localhost", None, ""):
            attempts.append(("dns", host))
            raise socket.gaierror("DNS blocked by test")
        return real_gai(host, *a, **k)

    socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = connect, connect_ex, gai
    try:
        yield attempts
    finally:
        socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo = real_connect, real_connect_ex, real_gai


def exif_jpeg(lat=None, lon=None, when=None, offset=None, rgb=(230, 20, 20), size=(400, 300), lat_ref=None, lon_ref=None):
    """A JPEG carrying EXIF GPS and/or DateTimeOriginal. lat/lon are signed decimal degrees."""
    from PIL import Image
    a = np.clip(np.array(rgb) + _RNG.integers(-8, 9, (size[1], size[0], 3)), 0, 255).astype(np.uint8)
    ex = Image.Exif()
    if lat is not None:
        def dms(v):
            v = abs(v)
            d = int(v)
            m = int((v - d) * 60)
            return (float(d), float(m), round((v - d - m / 60) * 3600, 4))
        ex.get_ifd(0x8825).update({1: lat_ref or ("N" if lat >= 0 else "S"), 2: dms(lat),
                                   3: lon_ref or ("E" if lon >= 0 else "W"), 4: dms(lon)})
    if when:
        ex.get_ifd(0x8769).update({0x9003: when})
        if offset:
            ex.get_ifd(0x8769).update({0x9011: offset})
    buf = io.BytesIO()
    Image.fromarray(a).save(buf, "JPEG", exif=ex)
    return buf.getvalue()
