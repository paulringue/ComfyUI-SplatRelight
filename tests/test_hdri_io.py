"""Round-trip tests for the HDRI readers.

There is no EXR/HDR writer available in this environment, so the tests encode
the files themselves. That is the point: the encoders are written straight from
the format specs, independently of the readers, so a matching round-trip is real
evidence both directions are right rather than two copies of the same mistake.
"""

import os
import struct
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import hdri_io as hio  # noqa: E402


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def make_image(h=16, w=32, seed=3):
    """An HDR-ish test image: mostly dim, with a very bright 'sun'."""
    g = np.random.default_rng(seed)
    img = (g.random((h, w, 3)) * 0.8).astype(np.float32)
    img[2:4, 5:8] = [120.0, 90.0, 40.0]       # values well above 1.0
    img[10, 20] = [0.0, 0.0, 0.0]
    return img


# ---------------------------------------------------------------- encoders

def encode_radiance(img: np.ndarray) -> bytes:
    """Write flat (uncompressed) RGBE scanlines."""
    h, w, _ = img.shape
    m = img.max(-1)
    exp = np.zeros_like(m, dtype=np.int32)
    nz = m > 1e-32
    mant, e = np.frexp(m[nz])
    exp[nz] = e + 128
    scale = np.zeros_like(m)
    scale[nz] = mant * 256.0 / m[nz]
    rgb = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    rgbe = np.concatenate([rgb, exp.clip(0, 255).astype(np.uint8)[..., None]], -1)
    head = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n" + f"-Y {h} +X {w}\n".encode()
    return head + rgbe.astype(np.uint8).tobytes()


def _exr_predictor(b: bytes) -> bytes:
    """Inverse of hdri_io._exr_reconstruct: de-interleave then delta-encode."""
    n = len(b)
    half = (n + 1) // 2
    inter = bytearray(n)
    inter[:half] = b[0::2]
    inter[half:] = b[1::2]
    out = bytearray(n)
    if n:
        out[0] = inter[0]
        prev = inter[0]
        for i in range(1, n):
            d = (inter[i] - prev + 128) & 0xFF
            out[i] = d
            prev = inter[i]
    return bytes(out)


def encode_exr(img: np.ndarray, compression=0, half=False) -> bytes:
    """Write a scanline EXR with NONE (0) or ZIPS (2) / ZIP (3) compression."""
    h, w, _ = img.shape
    dt = np.float16 if half else np.float32
    ptype = 1 if half else 2
    rows_per_block = 16 if compression == 3 else 1

    out = bytearray(struct.pack("<ii", hio._EXR_MAGIC, 2))

    def attr(name, atype, payload):
        out.extend(name.encode() + b"\x00" + atype.encode() + b"\x00")
        out.extend(struct.pack("<i", len(payload)))
        out.extend(payload)

    chans = bytearray()
    for c in ("B", "G", "R"):                       # alphabetical, as the spec requires
        chans.extend(c.encode() + b"\x00")
        chans.extend(struct.pack("<iiii", ptype, 0, 1, 1))
    chans.append(0)
    attr("channels", "chlist", bytes(chans))
    attr("compression", "compression", bytes([compression]))
    attr("dataWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
    attr("displayWindow", "box2i", struct.pack("<iiii", 0, 0, w - 1, h - 1))
    attr("lineOrder", "lineOrder", bytes([0]))
    attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    attr("screenWindowCenter", "v2f", struct.pack("<ff", 0.0, 0.0))
    attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    out.append(0)

    n_blocks = (h + rows_per_block - 1) // rows_per_block
    offset_pos = len(out)
    out.extend(b"\x00" * (8 * n_blocks))

    offsets = []
    for bi in range(n_blocks):
        y = bi * rows_per_block
        rows = min(rows_per_block, h - y)
        body = bytearray()
        for r in range(rows):
            for ci, cname in enumerate(("B", "G", "R")):
                idx = {"R": 0, "G": 1, "B": 2}[cname]
                body.extend(img[y + r, :, idx].astype(dt).tobytes())
        payload = bytes(body)
        if compression in (2, 3):
            packed = zlib.compress(_exr_predictor(payload), 6)
            if len(packed) >= len(payload):
                packed = payload                    # spec: store raw if not smaller
        else:
            packed = payload
        offsets.append(len(out))
        out.extend(struct.pack("<ii", y, len(packed)))
        out.extend(packed)

    struct.pack_into(f"<{n_blocks}Q", out, offset_pos, *offsets)
    return bytes(out)


# ---------------------------------------------------------------- tests

def test_radiance():
    print("radiance .hdr")
    img = make_image()
    got = hio.read_radiance(encode_radiance(img))
    check(got.shape == img.shape, f"shape round-trips {got.shape}")
    # RGBE is an 8-bit mantissa with a shared exponent, so ~0.5% relative error
    m = img.max(-1) > 1e-6
    rel = np.abs(got[m] - img[m]) / np.maximum(img[m].max(-1, keepdims=True), 1e-6)
    check(rel.max() < 0.01, f"values round-trip within RGBE precision (max rel err {rel.max():.4f})")
    check(got[2, 5, 0] > 100.0, f"above-1.0 sun survives (got {got[2,5,0]:.1f}, want ~120)")
    check(np.all(got[10, 20] == 0), "black pixel stays black")


def test_exr_uncompressed():
    print("exr NONE")
    img = make_image()
    got = hio.read_exr(encode_exr(img, compression=0))
    check(got.shape == img.shape, f"shape round-trips {got.shape}")
    check(np.allclose(got, img, atol=1e-6), "float32 values are exact")
    check(got[2, 5, 0] > 100.0, f"sun survives ({got[2,5,0]:.1f})")


def test_exr_zip():
    print("exr ZIP / ZIPS")
    img = make_image(h=40, w=24)
    for comp, name in ((2, "ZIPS"), (3, "ZIP")):
        got = hio.read_exr(encode_exr(img, compression=comp))
        check(got.shape == img.shape, f"{name}: shape {got.shape}")
        check(np.allclose(got, img, atol=1e-6), f"{name}: values exact through zlib + predictor")


def test_exr_half():
    print("exr HALF channels")
    img = make_image()
    got = hio.read_exr(encode_exr(img, compression=3, half=True))
    check(np.allclose(got, img.astype(np.float16).astype(np.float32), atol=1e-3),
          "half-float channels decode correctly")
    check(got[2, 5, 0] > 100.0, f"sun survives half precision ({got[2,5,0]:.1f})")


def test_channel_order():
    print("channel order")
    img = np.zeros((4, 4, 3), np.float32)
    img[..., 0] = 1.0          # pure red
    got = hio.read_exr(encode_exr(img, compression=0))
    check(got[0, 0, 0] > 0.99 and got[0, 0, 1] < 1e-6 and got[0, 0, 2] < 1e-6,
          "R/G/B are not transposed (pure red stays red)")


def test_errors():
    print("error handling")
    try:
        hio.read_exr(b"not an exr at all............")
        check(False, "should have rejected a non-EXR")
    except ValueError as e:
        check("magic" in str(e), f"non-EXR rejected: {e}")
    try:
        hio.read_radiance(b"nope")
        check(False, "should have rejected a non-HDR")
    except ValueError as e:
        check("RADIANCE" in str(e), f"non-HDR rejected: {e}")

    # an unsupported compression must name itself rather than fail obscurely
    data = bytearray(encode_exr(make_image(), compression=0))
    i = data.index(b"compression\x00compression\x00")
    data[i + len("compression\x00compression\x00") + 4] = 4          # PIZ
    try:
        hio.read_exr(bytes(data))
        check(False, "should have rejected PIZ")
    except ValueError as e:
        check("PIZ" in str(e), f"unsupported compression names itself: {str(e)[:60]}...")


def test_dispatch(tmpdir=None):
    print("load_hdri dispatch")
    import tempfile
    img = make_image()
    d = tempfile.mkdtemp()
    hp = os.path.join(d, "sky.hdr")
    ep = os.path.join(d, "sky.exr")
    open(hp, "wb").write(encode_radiance(img))
    open(ep, "wb").write(encode_exr(img, compression=3))
    a = hio.load_hdri(hp)
    b = hio.load_hdri(ep)
    check(a.shape == img.shape and b.shape == img.shape, "both formats dispatch by content")
    check(np.allclose(b, img, atol=1e-6), "exr path is exact through load_hdri")
    check(a.dtype == np.float32 and b.dtype == np.float32, "both return float32")


def main():
    for fn in [test_radiance, test_exr_uncompressed, test_exr_zip, test_exr_half,
               test_channel_order, test_errors, test_dispatch]:
        fn()
    print("\nall hdri_io tests passed")


if __name__ == "__main__":
    main()
