"""Dependency-free readers for HDR panorama formats.

ComfyUI's Load Image goes through Pillow, which reads neither Radiance `.hdr`
nor OpenEXR `.exr`. That matters for image-based lighting specifically: an HDRI's
useful information lives *above* 1.0 - the sun, the sky, a window - and an 8-bit
PNG clips exactly the part that drives the lighting.

So this module reads both formats directly into linear float32:

  * `.hdr` / `.pic` - Radiance RGBE, including new- and old-style RLE. Complete.
  * `.exr`  - OpenEXR scanline images with NONE / RLE / ZIP / ZIPS compression,
              HALF or FLOAT channels. That covers the overwhelming majority of
              HDRIs in the wild. PIZ / PXR24 / B44 / DWAA need a real codec, so
              those fall through to an installed library if there is one, and
              otherwise raise a message naming the compression.

No ComfyUI imports here, so it can be tested standalone.
"""

import struct
import zlib

import numpy as np

# ---------------------------------------------------------------------------
# Radiance .hdr (RGBE)
# ---------------------------------------------------------------------------

def read_radiance(data: bytes) -> np.ndarray:
    """Radiance RGBE bytes -> (H, W, 3) float32 linear."""
    if not (data.startswith(b"#?RADIANCE") or data.startswith(b"#?RGBE")):
        raise ValueError("not a Radiance .hdr file (missing #?RADIANCE signature)")

    pos = data.index(b"\n") + 1
    while True:                                   # header lines until a blank one
        end = data.index(b"\n", pos)
        line = data[pos:end].strip()
        pos = end + 1
        if not line:
            break
        if line.startswith(b"FORMAT=") and b"32-bit_rle_rgbe" not in line:
            raise ValueError(f"unsupported Radiance FORMAT: {line.decode('ascii', 'replace')}")

    end = data.index(b"\n", pos)
    dims = data[pos:end].split()
    pos = end + 1
    if len(dims) != 4 or dims[0] not in (b"-Y", b"+Y"):
        raise ValueError(f"unsupported Radiance resolution line: {b' '.join(dims)!r}")
    height, width = int(dims[1]), int(dims[3])
    flip_y = dims[0] == b"+Y"

    rgbe = np.empty((height, width, 4), np.uint8)
    buf = memoryview(data)
    for y in range(height):
        if pos + 4 > len(buf):
            raise ValueError("truncated Radiance data")
        b0, b1, b2, b3 = buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]
        if b0 == 2 and b1 == 2 and ((b2 << 8) | b3) == width and width >= 8:
            pos += 4                              # new-style, per-component RLE
            for c in range(4):
                x = 0
                while x < width:
                    n = buf[pos]; pos += 1
                    if n > 128:                   # a run of one repeated value
                        rgbe[y, x:x + (n - 128), c] = buf[pos]; pos += 1
                        x += n - 128
                    else:                         # a literal span
                        rgbe[y, x:x + n, c] = np.frombuffer(data, np.uint8, n, pos)
                        pos += n
                        x += n
        else:                                     # flat, or old-style RLE
            x, shift = 0, 0
            while x < width:
                r, g, b, e = buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]
                pos += 4
                if r == 1 and g == 1 and b == 1:  # repeat the previous pixel
                    count = e << shift
                    rgbe[y, x:x + count] = rgbe[y, x - 1]
                    x += count
                    shift += 8
                else:
                    rgbe[y, x] = (r, g, b, e)
                    x += 1
                    shift = 0

    exp = rgbe[..., 3].astype(np.int32)
    scale = np.where(exp == 0, 0.0, np.ldexp(1.0, exp - (128 + 8))).astype(np.float32)
    out = rgbe[..., :3].astype(np.float32) * scale[..., None]
    return out[::-1] if flip_y else out


# ---------------------------------------------------------------------------
# OpenEXR (scanline, NONE / RLE / ZIPS / ZIP)
# ---------------------------------------------------------------------------

_EXR_MAGIC = 0x01312F76
_COMP_NAMES = {0: "NONE", 1: "RLE", 2: "ZIPS", 3: "ZIP", 4: "PIZ",
               5: "PXR24", 6: "B44", 7: "B44A", 8: "DWAA", 9: "DWAB"}
_ROWS_PER_BLOCK = {0: 1, 1: 1, 2: 1, 3: 16, 4: 32, 5: 16, 6: 32, 7: 32, 8: 32, 9: 256}
_PIX_DTYPE = {0: np.uint32, 1: np.float16, 2: np.float32}


def _exr_reconstruct(b: bytes) -> bytes:
    """Undo the delta predictor ZIP/RLE blocks are stored with.

    Vectorised: the loop form is byte-at-a-time over megabytes, which is far too
    slow in Python for a 4k panorama. Each byte is the running sum of the deltas
    modulo 256, so a cumulative sum gives the same answer in one pass.
    """
    a = np.frombuffer(b, np.uint8).astype(np.int64)
    if a.size == 0:
        return b
    out = np.empty(a.size, np.int64)
    out[0] = a[0]
    out[1:] = a[1:] - 128
    return (np.cumsum(out) & 0xFF).astype(np.uint8).tobytes()


def _exr_interleave(b: bytes) -> bytes:
    n = len(b)
    half = (n + 1) // 2
    out = bytearray(n)
    out[0::2] = b[:half]
    out[1::2] = b[half:]
    return bytes(out)


def _exr_rle_decode(b: bytes, expected: int) -> bytes:
    out = bytearray()
    i = 0
    while i < len(b) and len(out) < expected:
        n = b[i]
        i += 1
        if n > 127:                     # signed byte < 0 -> run
            count = 257 - n
            out.extend(b[i:i + 1] * count)
            i += 1
        else:
            count = n + 1
            out.extend(b[i:i + count])
            i += count
    return bytes(out)


def _exr_decompress(comp: int, raw: bytes, expected: int) -> bytes:
    if comp == 0 or len(raw) >= expected:
        return raw[:expected] if len(raw) >= expected else raw
    if comp in (2, 3):
        data = zlib.decompress(raw)
    elif comp == 1:
        data = _exr_rle_decode(raw, expected)
    else:
        raise ValueError(f"EXR compression {_COMP_NAMES.get(comp, comp)} is not supported natively")
    return _exr_interleave(_exr_reconstruct(data))


def _exr_read_attr(data: bytes, pos: int):
    end = data.index(b"\x00", pos)
    name = data[pos:end].decode("ascii", "replace")
    pos = end + 1
    if not name:
        return None, None, None, pos
    end = data.index(b"\x00", pos)
    atype = data[pos:end].decode("ascii", "replace")
    pos = end + 1
    (size,) = struct.unpack_from("<i", data, pos)
    pos += 4
    value = data[pos:pos + size]
    return name, atype, value, pos + size


def read_exr(data: bytes) -> np.ndarray:
    """OpenEXR bytes -> (H, W, 3) float32 linear. Scanline images only."""
    magic, version = struct.unpack_from("<ii", data, 0)
    if magic != _EXR_MAGIC:
        raise ValueError("not an OpenEXR file (bad magic)")
    if version & (1 << 9):
        raise ValueError("tiled EXR images are not supported; re-save as scanline")
    if version & (1 << 11) or version & (1 << 12):
        raise ValueError("deep or multi-part EXR files are not supported")

    pos = 8
    channels, box, comp = [], None, 0
    while True:
        name, atype, value, pos = _exr_read_attr(data, pos)
        if name is None:
            break
        if name == "channels":
            p = 0
            while p < len(value) and value[p] != 0:
                e = value.index(b"\x00", p)
                cname = value[p:e].decode("ascii", "replace")
                p = e + 1
                (ptype,) = struct.unpack_from("<i", value, p)
                channels.append((cname, ptype))
                p += 16                       # pixelType, pLinear+reserved, xSampling, ySampling
        elif name == "dataWindow":
            box = struct.unpack_from("<iiii", value, 0)
        elif name == "compression":
            comp = value[0]

    if box is None or not channels:
        raise ValueError("EXR header is missing dataWindow or channels")
    x0, y0, x1, y1 = box
    width, height = x1 - x0 + 1, y1 - y0 + 1

    if comp not in (0, 1, 2, 3):
        raise ValueError(
            f"EXR uses {_COMP_NAMES.get(comp, comp)} compression, which needs a full codec. "
            "Re-save the HDRI as ZIP (or use .hdr), or install one of "
            "'OpenEXR', 'opencv-python' or 'imageio' into the ComfyUI environment.")

    channels.sort(key=lambda c: c[0])          # EXR stores channels alphabetically
    rows_per_block = _ROWS_PER_BLOCK[comp]
    n_blocks = (height + rows_per_block - 1) // rows_per_block
    offsets = struct.unpack_from(f"<{n_blocks}Q", data, pos)

    planes = {c: np.zeros((height, width), np.float32) for c, _ in channels}
    for off in offsets:
        y, size = struct.unpack_from("<ii", data, off)
        raw = data[off + 8: off + 8 + size]
        rows = min(rows_per_block, height - (y - y0))
        row_bytes = sum(width * np.dtype(_PIX_DTYPE[t]).itemsize for _, t in channels)
        block = _exr_decompress(comp, raw, row_bytes * rows)
        p = 0
        for r in range(rows):
            for cname, ptype in channels:
                dt = np.dtype(_PIX_DTYPE[ptype])
                n = width * dt.itemsize
                planes[cname][y - y0 + r] = np.frombuffer(block, dt, width, p).astype(np.float32)
                p += n

    def pick(*names):
        for n in names:
            if n in planes:
                return planes[n]
        return None

    r = pick("R", "r", "Y")
    g = pick("G", "g", "Y")
    b = pick("B", "b", "Y")
    if r is None or g is None or b is None:
        avail = ", ".join(sorted(planes))
        raise ValueError(f"EXR has no RGB channels (found: {avail})")
    return np.stack([r, g, b], -1).astype(np.float32)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

def _try_libraries(path: str):
    """Fall back to whatever HDR-capable library happens to be installed."""
    try:
        import cv2  # noqa
        import os
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
        if img is not None:
            return np.ascontiguousarray(img[..., ::-1].astype(np.float32))   # BGR -> RGB
    except Exception:
        pass
    try:
        import imageio.v3 as iio  # noqa
        return np.asarray(iio.imread(path)).astype(np.float32)[..., :3]
    except Exception:
        pass
    return None


def load_hdri(path: str) -> np.ndarray:
    """Read a `.hdr` or `.exr` panorama as (H, W, 3) float32 linear radiance."""
    with open(path, "rb") as fh:
        data = fh.read()
    lower = path.lower()
    try:
        if data.startswith(b"#?"):
            return read_radiance(data)
        if struct.unpack_from("<i", data, 0)[0] == _EXR_MAGIC:
            return read_exr(data)
    except Exception as native_error:
        img = _try_libraries(path)
        if img is not None:
            return img[..., :3]
        raise native_error
    if lower.endswith((".hdr", ".pic")):
        return read_radiance(data)
    img = _try_libraries(path)
    if img is not None:
        return img[..., :3]
    raise ValueError(f"unrecognised HDRI format: {path}")
