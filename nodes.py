"""ComfyUI nodes for relighting animated 3D gaussian splats.

The graph shape this is built for:

    Load Splat Sequence -> Delight Splat -> Relight Splat -> Render Splat -> Save Video
                                                ^
                                   Splat Light / Environment Light

Relighting happens in *splat space*: we rewrite each gaussian's SH DC term from
a physically-based shade of its own surfel normal. That is deliberate. The
alternative - deferred shading of the rendered 2D buffers - would re-derive
lighting per frame and per camera, so a turntable would swim and shadows would
detach. Relighting the primitives themselves means one lit asset that stays
consistent from every angle and can be exported back out to .ply.
"""

import glob as _glob
import json
import logging
import math
import os
import uuid

import torch
from typing_extensions import override

import comfy.model_management
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, IO, Types

from . import relight_core as rc

try:  # reuse ComfyUI's own splat parsers rather than shipping a second one
    from comfy_extras.nodes_gaussian_splat import _GAUSSIAN_PARSERS, _detect_splat_format
except Exception as e:  # pragma: no cover - only trips on an unexpected core refactor
    _GAUSSIAN_PARSERS, _detect_splat_format = None, None
    _IMPORT_ERROR = e

LightRig = IO.Custom("SPLAT_LIGHT_RIG")
# A crop region authored in the viewport: {"shape", "centre" (viewer space), "size"}
CropRegion = IO.Custom("SPLAT_CROP")

_SEQ_EXTS = (".ply", ".splat", ".ksplat", ".spz")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _hex_to_rgb(h: str):
    h = (h or "").lstrip("#")
    if len(h) != 6:
        return (1.0, 1.0, 1.0)
    return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def _real_len(splat, i):
    return int(splat.counts[i].item()) if getattr(splat, "counts", None) is not None \
        else splat.positions.shape[1]


def _yaw_pitch_to_dir(yaw_deg, pitch_deg):
    """Artist-facing azimuth/elevation -> the direction the light travels along.

    Deliberately the same yaw/pitch convention as Create Camera Info's orbit
    mode, so a light at yaw 35 / pitch 30 sits exactly where a camera at yaw 35 /
    pitch 30 would be. Positive pitch raises the light. Viewer space (Y-up); the
    conversion to splat space happens later, in _to_splat_frame.
    """
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return [-math.cos(p) * math.sin(y), -math.sin(p), -math.cos(p) * math.cos(y)]


def _to_splat_frame(light):
    """Move a viewer-space light into the splat's Y-down frame.

    Environment lights are already projected into splat space when their SH is
    built, so they pass through untouched.
    """
    if light["type"] == "environment":
        return light
    lt = dict(light)
    if lt.get("direction") is not None:
        lt["direction"] = rc.viewer_to_splat(lt["direction"])
    if lt.get("position") is not None:
        lt["position"] = rc.viewer_to_splat(lt["position"])
    return lt


def _rotate_y(vec, deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    x, y, z = vec
    return [c * x + s * z, y, -s * x + c * z]


def _animate(light, frame):
    """Apply a light's per-frame orbit. Returns a shallow copy for that frame."""
    speed = float(light.get("orbit_speed", 0.0))
    if speed == 0.0 or frame == 0:
        return light
    lt = dict(light)
    ang = speed * frame
    if "direction" in lt and lt["direction"] is not None:
        lt["direction"] = _rotate_y(lt["direction"], ang)
    if "position" in lt and lt["position"] is not None:
        lt["position"] = _rotate_y(lt["position"], ang)
    return lt


def _splat_normals(splat, i, positions, scales, rotations, opacities,
                   orientation, grid, sigma, smoothing=0.0, smooth_radius=1):
    """Surfel normals for batch item i, cached on the SPLAT object.

    Delight and Relight both need the same normals, and the visibility pass is
    the expensive part, so cache it. A new SPLAT object (e.g. after Transform
    Splat) correctly starts with a cold cache.
    """
    key = "_splatrelight_normals"
    sig = (orientation, round(float(smoothing), 4), int(smooth_radius))
    cache = getattr(splat, key, None)
    if isinstance(cache, dict):
        hit = cache.get(i)
        if hit is not None and hit[0] == sig and hit[1].shape[0] == positions.shape[0]:
            return hit[1].to(positions.device)
    else:
        cache = {}
    n = rc.surfel_normals(scales, rotations)
    n = rc.orient_normals(n, positions, mode=orientation, grid=grid, sigma=sigma)
    if smoothing > 0 and grid is not None:
        # weight by flatness x opacity so confident gaussians drive the plane
        w = rc.anisotropy(scales) * opacities.reshape(-1)
        n = rc.smooth_normals(grid, positions, n, weights=w,
                              radius=smooth_radius, strength=smoothing)
    cache[i] = (sig, n)
    try:
        setattr(splat, key, cache)
    except Exception:
        pass
    return n


def _clone_sh(splat):
    return splat.sh.clone()


def _out_splat(splat, sh):
    return Types.SPLAT(splat.positions, splat.scales, splat.rotations, splat.opacities, sh,
                       counts=getattr(splat, "counts", None))


# --------------------------------------------------------------------------
# sequence loading
# --------------------------------------------------------------------------

class LoadSplatSequence(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LoadSplatSequence",
            display_name="Load Splat Sequence",
            search_aliases=["animated splat", "4d gaussian", "splat sequence", "load ply sequence"],
            category="3d/splat/relight",
            description="Load a folder of per-frame gaussian splat files (.ply/.splat/.ksplat/.spz) "
                        "as one animated splat. Files are sorted by name; each becomes a frame in the "
                        "splat batch, which Render Splat then renders as an image sequence. Frames may "
                        "have different gaussian counts - they are padded and tracked automatically.",
            inputs=[
                IO.String.Input("directory", default="",
                                tooltip="Folder holding the per-frame splat files. Absolute path, or a "
                                        "path relative to the ComfyUI input directory."),
                IO.String.Input("pattern", default="*.ply",
                                tooltip="Glob for the frame files, e.g. *.ply or frame_*.spz"),
                IO.Int.Input("start_frame", default=0, min=0, max=100000),
                IO.Int.Input("frame_count", default=0, min=0, max=100000,
                             tooltip="How many frames to load. 0 = every frame found."),
                IO.Int.Input("step", default=1, min=1, max=100,
                             tooltip="Load every Nth file - useful for previewing a long take cheaply."),
            ],
            outputs=[IO.Splat.Output(display_name="splat"),
                     IO.Int.Output(display_name="frame_count")],
        )

    @classmethod
    def execute(cls, directory, pattern, start_frame, frame_count, step) -> IO.NodeOutput:
        if _GAUSSIAN_PARSERS is None:
            raise RuntimeError(
                "Could not import ComfyUI's gaussian splat parsers "
                f"(comfy_extras.nodes_gaussian_splat): {_IMPORT_ERROR}. "
                "ComfyUI 0.23.0 or newer is required.")

        base = directory.strip().strip('"')
        if not base:
            raise ValueError("Load Splat Sequence: 'directory' is empty.")
        if not os.path.isabs(base):
            base = os.path.join(folder_paths.get_input_directory(), base)
        if not os.path.isdir(base):
            raise FileNotFoundError(f"Load Splat Sequence: no such folder: {base}")

        files = sorted(_glob.glob(os.path.join(base, pattern or "*.ply")))
        files = [f for f in files if os.path.splitext(f)[1].lower() in _SEQ_EXTS]
        if not files:
            raise FileNotFoundError(
                f"Load Splat Sequence: no splat files matching '{pattern}' in {base}. "
                f"Supported extensions: {', '.join(_SEQ_EXTS)}")

        files = files[start_frame::max(1, step)]
        if frame_count > 0:
            files = files[:frame_count]
        if not files:
            raise ValueError("Load Splat Sequence: start_frame/step skipped past every file.")

        pbar = comfy.utils.ProgressBar(len(files))
        frames, counts = [], []
        for path in files:
            with open(path, "rb") as fh:
                data = fh.read()
            ext = os.path.splitext(path)[1].lstrip(".").lower()
            parser = _GAUSSIAN_PARSERS.get(ext) or _GAUSSIAN_PARSERS[_detect_splat_format(data)]
            xyz, scale, rot, opacity, sh = parser(data)
            t = lambda a: torch.from_numpy(a.copy()).float() if not torch.is_tensor(a) else a.float()
            frames.append((t(xyz), t(scale), t(rot), t(opacity).reshape(-1, 1), t(sh)))
            counts.append(frames[-1][0].shape[0])
            pbar.update(1)

        n_max = max(counts)
        k_max = max(f[4].shape[1] for f in frames)

        def stack(idx):
            out = []
            for f in frames:
                a = f[idx]
                if idx == 4 and a.shape[1] < k_max:      # pad SH bands to a common degree
                    pad = torch.zeros(a.shape[0], k_max - a.shape[1], 3, dtype=a.dtype)
                    a = torch.cat([a, pad], dim=1)
                if a.shape[0] < n_max:                   # pad the gaussian count
                    shape = (n_max - a.shape[0],) + tuple(a.shape[1:])
                    a = torch.cat([a, torch.zeros(shape, dtype=a.dtype)], dim=0)
                out.append(a)
            return torch.stack(out)

        splat = Types.SPLAT(
            stack(0), stack(1), stack(2), stack(3), stack(4),
            counts=(torch.tensor(counts, dtype=torch.long)
                    if len(set(counts)) > 1 else None),
        )
        return IO.NodeOutput(splat, len(files))


# --------------------------------------------------------------------------
# lights
# --------------------------------------------------------------------------

class SplatLight(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatLight",
            display_name="Splat Light",
            search_aliases=["add light", "key light", "point light", "spot light", "relight"],
            category="3d/splat/relight",
            description="Add one light to a lighting rig. Chain several of these to build a key/fill/rim "
                        "setup - each node takes the previous rig and appends to it. Set orbit_speed to "
                        "make the light travel around the subject over an animated sequence.",
            inputs=[
                IO.DynamicCombo.Input("light", options=[
                    IO.DynamicCombo.Option("directional", [
                        IO.Float.Input("yaw", default=40.0, min=-360.0, max=360.0, step=1.0,
                                       tooltip="Compass angle of the light around the subject, degrees."),
                        IO.Float.Input("pitch", default=35.0, min=-89.0, max=89.0, step=1.0,
                                       tooltip="Height of the light. Positive is above the subject."),
                    ]),
                    IO.DynamicCombo.Option("point", [
                        IO.Float.Input("position_x", default=2.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_y", default=2.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_z", default=2.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("radius", default=0.0, min=0.0, max=100.0, step=0.01,
                                       tooltip="Softens the inverse-square falloff near the source. "
                                               "0 = pure 1/d^2 (physically correct, can blow out up close)."),
                    ]),
                    IO.DynamicCombo.Option("spot", [
                        IO.Float.Input("position_x", default=2.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_y", default=3.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("position_z", default=2.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("target_x", default=0.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("target_y", default=0.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("target_z", default=0.0, min=-1000.0, max=1000.0, step=0.01),
                        IO.Float.Input("cone_angle", default=45.0, min=1.0, max=179.0, step=1.0,
                                       tooltip="Full width of the bright core, degrees."),
                        IO.Float.Input("cone_softness", default=10.0, min=0.0, max=90.0, step=1.0,
                                       tooltip="Extra degrees of soft falloff outside the core."),
                        IO.Float.Input("radius", default=0.0, min=0.0, max=100.0, step=0.01),
                    ]),
                ], tooltip="Light type. Directional acts like the sun (parallel rays, no falloff); "
                           "point and spot are placed in the scene and fall off with distance."),
                IO.Color.Input("color", default="#FFFFFF"),
                IO.Float.Input("intensity", default=3.0, min=0.0, max=1000.0, step=0.1,
                               tooltip="Brightness. A directional light at pi renders a white surface "
                                       "facing it at full albedo."),
                IO.Boolean.Input("cast_shadow", default=True,
                                 tooltip="Let this light be occluded by the splat. Requires shadows to be "
                                         "enabled on the Relight Splat node."),
                IO.Float.Input("orbit_speed", default=0.0, min=-45.0, max=45.0, step=0.1,
                               tooltip="Degrees the light rotates around the world Y axis per frame. "
                                       "Non-zero animates the lighting across a splat sequence."),
                LightRig.Input("rig", optional=True,
                               tooltip="Rig to add this light to. Leave empty to start a new rig."),
            ],
            outputs=[LightRig.Output(display_name="rig")],
        )

    @classmethod
    def execute(cls, light, color, intensity, cast_shadow, orbit_speed, rig=None) -> IO.NodeOutput:
        kind = light["light"]
        lt = {"type": kind, "color": list(_hex_to_rgb(color)), "intensity": float(intensity),
              "cast_shadow": bool(cast_shadow), "orbit_speed": float(orbit_speed)}

        if kind == "directional":
            lt["direction"] = _yaw_pitch_to_dir(light["yaw"], light["pitch"])
            lt["position"] = None
        else:
            pos = [light["position_x"], light["position_y"], light["position_z"]]
            lt["position"] = pos
            lt["radius"] = float(light.get("radius", 0.0))
            if kind == "spot":
                tgt = [light["target_x"], light["target_y"], light["target_z"]]
                d = [tgt[j] - pos[j] for j in range(3)]
                ln = math.sqrt(sum(c * c for c in d)) or 1.0
                lt["direction"] = [c / ln for c in d]
                lt["cone_angle"] = float(light["cone_angle"])
                lt["cone_softness"] = float(light["cone_softness"])
            else:
                lt["direction"] = None

        return IO.NodeOutput(list(rig or []) + [lt])


class LoadHDRI(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="LoadHDRI",
            display_name="Load HDRI (EXR / HDR)",
            search_aliases=["load exr", "load hdr", "hdri", "open exr", "radiance",
                            "load panorama", "environment map"],
            category="3d/splat/relight",
            description="Load an OpenEXR (.exr) or Radiance (.hdr) panorama as a linear HDR image. "
                        "ComfyUI's Load Image goes through Pillow, which reads neither - and for "
                        "image-based lighting that matters, because an HDRI's useful information "
                        "(sun, sky, windows) lives above 1.0 and an 8-bit PNG clips exactly the part "
                        "that drives the lighting. Feed 'image' to Splat Environment Light with "
                        "input_is_srgb OFF.",
            inputs=[
                IO.String.Input("path", default="",
                                tooltip="Path to the .exr / .hdr panorama. Absolute, or relative to "
                                        "the ComfyUI input directory."),
                IO.Float.Input("exposure", default=1.0, min=0.0001, max=10000.0, step=0.1,
                               tooltip="Linear multiplier applied on load. The lighting is driven by "
                                       "absolute radiance, so this is the main brightness control."),
                IO.Float.Input("rotation", default=0.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Spin the panorama around the vertical axis, degrees."),
                IO.Boolean.Input("flip_vertical", default=False,
                                 tooltip="Turn on if the environment appears upside down - some "
                                         "exporters write panoramas bottom-up."),
                IO.Float.Input("clamp_max", default=0.0, min=0.0, max=100000.0, step=1.0,
                               advanced=True,
                               tooltip="Clip radiance above this before use. 0 = no clamping. Useful "
                                       "when a tiny blown-out sun makes the SH fit ring."),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.Image.Output(display_name="preview"),
            ],
        )

    @classmethod
    def execute(cls, path, exposure, rotation, flip_vertical, clamp_max) -> IO.NodeOutput:
        from . import hdri_io

        p = (path or "").strip().strip('"')
        if not p:
            raise ValueError("Load HDRI: 'path' is empty.")
        if not os.path.isabs(p):
            p = os.path.join(folder_paths.get_input_directory(), p)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Load HDRI: no such file: {p}")

        arr = hdri_io.load_hdri(p)                       # (H, W, 3) float32 linear
        img = torch.from_numpy(arr).float()
        if flip_vertical:
            img = torch.flip(img, dims=[0])
        if rotation:
            shift = int(round((rotation % 360.0) / 360.0 * img.shape[1]))
            if shift:
                img = torch.roll(img, shifts=shift, dims=1)
        img = img * float(exposure)
        if clamp_max > 0:
            img = img.clamp(max=float(clamp_max))
        img = torch.nan_to_num(img, nan=0.0, posinf=float(clamp_max or 1e6), neginf=0.0).clamp_min(0.0)

        # second output is display-safe: tone mapped and gamma encoded, so a
        # Preview Image node shows the panorama instead of a clipped white sheet
        preview = rc.linear_to_srgb(rc.tonemap(img, exposure=1.0, mode="aces")).clamp(0, 1)
        return IO.NodeOutput(img.unsqueeze(0), preview.unsqueeze(0))


def _camera_vectors(camera_info):
    """(position, target) in viewer space from a camera_info dict."""
    p = (camera_info or {}).get("position", {}) or {}
    t = (camera_info or {}).get("target", {}) or {}
    pos = [float(p.get("x", 0.0)), float(p.get("y", 0.0)), float(p.get("z", 0.0))]
    tgt = [float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("z", 0.0))]
    return pos, tgt


def _viewport_blob(splat, index, max_points):
    """Decimate a splat to a compact binary the browser can draw at 60fps.

    Layout: 'SRV1' | uint32 count | then per point 3x float32 position (viewer
    space) and 4x uint8 rgba. 16 bytes a point, so 200k points is ~3 MB - small
    enough to fetch instantly, dense enough to aim lights against.

    Positions are converted to viewer space here, so the camera and light
    directions the viewport hands back need no further conversion.
    """
    end = _real_len(splat, index)
    if end == 0:
        raise ValueError("Splat Viewport: the splat is empty.")
    step = max(1, math.ceil(end / max(int(max_points), 1)))
    sel = slice(0, end, step)

    pos = splat.positions[index, sel].float()
    rgb = rc.dc_to_rgb(splat.sh[index, sel, 0, :].float())
    opa = splat.opacities[index, sel].float().reshape(-1)

    pos = rc.flip_frame(pos)                       # splat (Y-down) -> viewer (Y-up)
    rgb8 = (rc.linear_to_srgb(rgb.clamp(0, 1)) * 255.0).clamp(0, 255).to(torch.uint8)
    a8 = (opa.clamp(0, 1) * 255.0).to(torch.uint8).reshape(-1, 1)

    import struct as _struct
    n = pos.shape[0]
    head = b"SRV1" + _struct.pack("<I", n)
    body = torch.cat([rgb8, a8], dim=1).cpu().numpy().tobytes()
    return head + pos.cpu().numpy().astype("<f4").tobytes() + body, n


class CropSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="CropSplat",
            display_name="Crop Splat",
            search_aliases=["crop", "trim splat", "isolate", "region", "cut out", "close up"],
            category="3d/splat/relight",
            description="Keep only the gaussians inside a box or sphere. The practical use is getting "
                        "close to something: ComfyUI's rasteriser renders nothing from inside geometry, "
                        "so on a room capture you can never get near a detail. Crop to that detail and "
                        "the shell you have to stay outside becomes the detail itself, not the room. "
                        "It also makes everything downstream faster. Connect a camera to centre the crop "
                        "on whatever you framed in the viewport.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Combo.Input("shape", options=["box", "sphere"]),
                IO.Float.Input("centre_x", default=0.0, min=-1000.0, max=1000.0, step=0.01,
                               tooltip="Crop centre in viewer space (Y-up), ignored when a camera is "
                                       "connected."),
                IO.Float.Input("centre_y", default=0.0, min=-1000.0, max=1000.0, step=0.01),
                IO.Float.Input("centre_z", default=0.0, min=-1000.0, max=1000.0, step=0.01),
                IO.Float.Input("size_x", default=1.0, min=0.01, max=1000.0, step=0.01,
                               tooltip="Full width of the box. For a sphere, this is the diameter and "
                                       "the other two are ignored."),
                IO.Float.Input("size_y", default=1.0, min=0.01, max=1000.0, step=0.01),
                IO.Float.Input("size_z", default=1.0, min=0.01, max=1000.0, step=0.01),
                IO.Boolean.Input("invert", default=False,
                                 tooltip="Keep what is OUTSIDE the region instead - handy for removing "
                                         "a floor or a wall that blocks the view."),
                CropRegion.Input("crop", optional=True,
                                 tooltip="A region dragged in Splat Viewport. When connected it "
                                         "overrides the shape, centre and size below, so you place the "
                                         "crop by eye instead of typing numbers."),
                IO.Load3DCamera.Input("camera_info", optional=True,
                                      tooltip="If connected, the crop centres on this camera's look-at "
                                              "point. Ignored when a crop region is connected."),
            ],
            outputs=[IO.Splat.Output(display_name="splat"),
                     IO.Int.Output(display_name="kept")],
        )

    @classmethod
    def execute(cls, splat, shape, centre_x, centre_y, centre_z, size_x, size_y, size_z,
                invert, crop=None, camera_info=None) -> IO.NodeOutput:
        if crop:
            shape = str(crop.get("shape", shape))
            c = crop.get("centre") or [centre_x, centre_y, centre_z]
            centre_x, centre_y, centre_z = (float(c[0]), float(c[1]), float(c[2]))
            s = crop.get("size")
            if s is not None:
                if isinstance(s, (list, tuple)):
                    size_x, size_y, size_z = (float(s[0]), float(s[1]), float(s[2]))
                else:
                    size_x = size_y = size_z = float(s)
        if camera_info and not crop:
            t = camera_info.get("target", {}) or {}
            centre_view = [float(t.get("x", 0.0)), float(t.get("y", 0.0)), float(t.get("z", 0.0))]
        else:
            centre_view = [float(centre_x), float(centre_y), float(centre_z)]
        centre = rc.viewer_to_splat(centre_view)          # the splat lives Y-down
        half = [abs(size_x) * 0.5, abs(size_y) * 0.5, abs(size_z) * 0.5]

        b = splat.positions.shape[0]
        keeps, total = [], 0
        for i in range(b):
            end = _real_len(splat, i)
            pos = splat.positions[i, :end]
            m = rc.crop_mask(pos, centre, half, shape)
            if invert:
                m = ~m
            idx = m.nonzero(as_tuple=True)[0]
            keeps.append(idx)
            total += int(idx.numel())

        if total == 0:
            raise ValueError(
                "Crop Splat: the region is empty - every gaussian was cut. Check the centre is in "
                "viewer space (Y-up) and that the size covers something; Splat Viewport's readout "
                "shows the scene centre and radius.")

        n_max = max(int(k.numel()) for k in keeps)
        def gather(src, kept, width):
            out = torch.zeros((b, n_max) + tuple(src.shape[2:]), dtype=src.dtype, device=src.device)
            for i, idx in enumerate(kept):
                if idx.numel():
                    out[i, :idx.numel()] = src[i, idx]
            return out

        counts = torch.tensor([int(k.numel()) for k in keeps], dtype=torch.long)
        out = Types.SPLAT(gather(splat.positions, keeps, 3), gather(splat.scales, keeps, 3),
                          gather(splat.rotations, keeps, 4), gather(splat.opacities, keeps, 1),
                          gather(splat.sh, keeps, None),
                          counts=counts if b > 1 or int(counts[0]) != n_max else None)
        logging.info(f"Crop Splat: kept {total:,} of "
                     f"{sum(_real_len(splat, i) for i in range(b)):,} gaussians")
        return IO.NodeOutput(out, total)


class SplatViewport(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatViewport",
            display_name="Splat Viewport",
            search_aliases=["viewer", "splat viewer", "3d view", "orbit", "interactive",
                            "move lights", "set camera", "gaussian viewer"],
            category="3d/splat/relight",
            description="An interactive viewport: see the splat, orbit to set the camera, and drag the "
                        "lights around inside the 3D view. Left-drag orbits, right-drag pans, scroll "
                        "zooms; drag a coloured light handle to move that light, and it updates live. "
                        "Outputs the camera you framed and the rig you placed, so the rest of the graph "
                        "follows what you set up here.",
            is_output_node=True,
            inputs=[
                IO.Splat.Input("splat", tooltip="Usually the relit splat, so you light what you see."),
                IO.Int.Input("max_points", default=180000, min=5000, max=1500000, step=5000,
                             tooltip="Points sent to the browser. The viewport decimates for speed; "
                                     "the final render always uses every gaussian."),
                IO.Float.Input("point_size", default=2.6, min=0.5, max=20.0, step=0.1),
                IO.Color.Input("background", default="#111318"),
                IO.String.Input("view_state", default="", multiline=True,
                                tooltip="Set by the viewport: camera and light placement. Leave alone."),
                IO.Float.Input("key_intensity", default=4.0, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("fill_intensity", default=1.5, min=0.0, max=1000.0, step=0.1),
                IO.Float.Input("rim_intensity", default=2.0, min=0.0, max=1000.0, step=0.1),
                IO.Boolean.Input("cast_shadow", default=True),
                IO.Boolean.Input("show_crop", default=False,
                                 tooltip="Draw the crop region in the viewport and give it a drag "
                                         "gizmo. Wire the crop output into Crop Splat to use it."),
                IO.Combo.Input("crop_shape", options=["box", "sphere"]),
                IO.Float.Input("crop_size_x", default=1.6, min=0.02, max=1000.0, step=0.05,
                               tooltip="Full width of the crop box along X. For a sphere this is the "
                                       "diameter and the other two are ignored. Dragging the square "
                                       "handles in the viewport overwrites these."),
                IO.Float.Input("crop_size_y", default=1.6, min=0.02, max=1000.0, step=0.05),
                IO.Float.Input("crop_size_z", default=1.6, min=0.02, max=1000.0, step=0.05),
                LightRig.Input("rig", optional=True),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info"),
                     LightRig.Output(display_name="rig"),
                     CropRegion.Output(display_name="crop")],
        )

    DEFAULT_LIGHTS = [
        {"on": True, "yaw": 55.0, "pitch": 38.0, "color": "#FFC98F"},
        {"on": True, "yaw": -70.0, "pitch": 8.0, "color": "#6E9CFF"},
        {"on": False, "yaw": 175.0, "pitch": 25.0, "color": "#FFFFFF"},
    ]

    @classmethod
    def execute(cls, splat, max_points, point_size, background, view_state,
                key_intensity, fill_intensity, rim_intensity, cast_shadow,
                show_crop, crop_shape, crop_size_x, crop_size_y, crop_size_z,
                rig=None) -> IO.NodeOutput:
        blob, n = _viewport_blob(splat, 0, max_points)
        name = f"splat_viewport_{uuid.uuid4().hex}.bin"
        path = os.path.join(folder_paths.get_temp_directory(), name)
        os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(blob)

        centre_splat, radius = rc.scene_bounds(splat.positions[0, :_real_len(splat, 0)].float())
        centre = rc.viewer_to_splat(centre_splat)

        try:
            state = json.loads(view_state) if view_state.strip() else {}
        except (ValueError, TypeError):
            state = {}
        lights = state.get("lights") or [dict(d) for d in cls.DEFAULT_LIGHTS]
        cam = state.get("camera") or {}

        # camera: whatever the viewport was left at, else a sane 3/4 framing
        if cam.get("position") and cam.get("target"):
            cam_pos = [float(v) for v in cam["position"]]
            cam_tgt = [float(v) for v in cam["target"]]
        else:
            d = _yaw_pitch_to_dir(40.0, 22.0)
            dist = rc.frame_distance(radius, 40.0, 1.3)
            cam_pos = [centre[i] - d[i] * dist for i in range(3)]
            cam_tgt = list(centre)
        q = rc.lookat_quat_xyzw(cam_pos, cam_tgt)
        xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
        camera_info = {"position": xyz(cam_pos), "target": xyz(cam_tgt),
                       "quaternion": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
                       "fov": float(cam.get("fov", 40.0)), "cameraType": "perspective", "zoom": 1.0}

        intens = [key_intensity, fill_intensity, rim_intensity]
        out_rig = list(rig or [])
        for i, lt in enumerate(lights[:3]):
            if not lt.get("on") or float(intens[i]) <= 0:
                continue
            out_rig.append({
                "type": "directional",
                "direction": _yaw_pitch_to_dir(float(lt.get("yaw", 0.0)), float(lt.get("pitch", 0.0))),
                "position": None,
                "color": list(_hex_to_rgb(lt.get("color", cls.DEFAULT_LIGHTS[i]["color"]))),
                "intensity": float(intens[i]),
                "cast_shadow": bool(cast_shadow),
                "orbit_speed": 0.0,
            })

        # the crop region: dragged in the viewport, falling back to the widgets.
        # A size dragged in the viewport wins, so the handles are not fighting
        # numbers typed on the node.
        saved_crop = state.get("crop", {}) or {}
        crop_centre = saved_crop.get("centre") or list(centre)
        size = saved_crop.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 3:
            crop_size = [float(v) for v in size]
        else:
            crop_size = [float(crop_size_x), float(crop_size_y), float(crop_size_z)]
        crop = {"shape": str(crop_shape), "centre": [float(v) for v in crop_centre],
                "size": crop_size, "enabled": bool(show_crop)}

        ui = {"splat_viewport": [{
            "file": name, "count": n, "point_size": float(point_size),
            "background": background, "centre": centre, "radius": radius,
            "lights": lights,
            "camera": {"position": cam_pos, "target": cam_tgt, "fov": camera_info["fov"]},
            "crop": crop, "show_crop": bool(show_crop),
        }]}
        return IO.NodeOutput(camera_info, out_rig, crop, ui=ui)


class SplatLightGizmo(IO.ComfyNode):
    SLOTS = ("key", "fill", "rim")
    DEFAULTS = [{"on": True, "yaw": 40.0, "pitch": 35.0},
                {"on": True, "yaw": -60.0, "pitch": 10.0},
                {"on": False, "yaw": 180.0, "pitch": 25.0}]

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatLightGizmo",
            display_name="Splat Light Gizmo",
            search_aliases=["light gizmo", "drag lights", "lighting ball", "three point",
                            "key fill rim", "move lights", "interactive lights"],
            category="3d/splat/relight",
            description="A three-point rig you aim by dragging. The node draws a lighting ball: drag "
                        "each handle to point that light, drag outside the ring to swing it behind the "
                        "subject, double-click a handle to switch it off. The ball is shaded live by the "
                        "colours and intensities below, so you can see the rig before you render. "
                        "Outputs the same rig as Splat Light, and chains with it.",
            inputs=[
                IO.String.Input("lights", default=json.dumps(cls.DEFAULTS), multiline=True,
                                tooltip="Set by the gizmo. Editable JSON if the UI is unavailable: a "
                                        "list of three {on, yaw, pitch} entries for key, fill and rim."),
                IO.Color.Input("key_color", default="#FFD2A8"),
                IO.Float.Input("key_intensity", default=3.5, min=0.0, max=1000.0, step=0.1),
                IO.Color.Input("fill_color", default="#9FB6FF"),
                IO.Float.Input("fill_intensity", default=1.2, min=0.0, max=1000.0, step=0.1),
                IO.Color.Input("rim_color", default="#FFFFFF"),
                IO.Float.Input("rim_intensity", default=2.0, min=0.0, max=1000.0, step=0.1),
                IO.Boolean.Input("cast_shadow", default=True),
                IO.Float.Input("orbit_speed", default=0.0, min=-45.0, max=45.0, step=0.1,
                               tooltip="Degrees per frame around world Y, applied to every light in "
                                       "this rig, for animated sequences."),
                LightRig.Input("rig", optional=True),
            ],
            outputs=[LightRig.Output(display_name="rig")],
        )

    @classmethod
    def execute(cls, lights, key_color, key_intensity, fill_color, fill_intensity,
                rim_color, rim_intensity, cast_shadow, orbit_speed, rig=None) -> IO.NodeOutput:
        try:
            state = json.loads(lights) if lights else []
        except (ValueError, TypeError) as e:
            raise ValueError(f"Splat Light Gizmo: 'lights' is not valid JSON ({e}). Expected a list "
                             "of three {{on, yaw, pitch}} entries.")
        if not isinstance(state, list) or len(state) != 3:
            state = [dict(d) for d in cls.DEFAULTS]

        styles = [(key_color, key_intensity), (fill_color, fill_intensity), (rim_color, rim_intensity)]
        out = list(rig or [])
        for i, entry in enumerate(state):
            if not isinstance(entry, dict) or not entry.get("on"):
                continue
            colour, intensity = styles[i]
            if float(intensity) <= 0.0:
                continue
            yaw = float(entry.get("yaw", cls.DEFAULTS[i]["yaw"]))
            pitch = float(entry.get("pitch", cls.DEFAULTS[i]["pitch"]))
            out.append({"type": "directional",
                        "direction": _yaw_pitch_to_dir(yaw, pitch),
                        "position": None,
                        "color": list(_hex_to_rgb(colour)),
                        "intensity": float(intensity),
                        "cast_shadow": bool(cast_shadow),
                        "orbit_speed": float(orbit_speed)})
        return IO.NodeOutput(out)


class FrameSplatCamera(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="FrameSplatCamera",
            display_name="Frame Splat Camera",
            search_aliases=["fit camera", "frame camera", "camera guard", "safe camera",
                            "black frames", "empty render"],
            category="3d/splat/relight",
            description="Keep a camera outside the splat so Render Splat can actually see it. "
                        "ComfyUI's rasteriser returns an EMPTY frame whenever the camera sits inside "
                        "the geometry, and an interactive viewer will happily auto-frame inside a room "
                        "capture - so a camera taken straight from a viewport can render nothing. This "
                        "keeps the angle you picked and only pushes the distance out far enough to clear "
                        "the scene. Put it between a Preview Splat's camera_info output and Render Splat.",
            inputs=[
                IO.Splat.Input("splat", tooltip="The splat being framed - used for its bounds."),
                IO.Load3DCamera.Input("camera_info", optional=True,
                                      tooltip="Camera to guard. Leave empty for an auto 3/4 view."),
                IO.Float.Input("margin", default=1.15, min=1.0, max=10.0, step=0.05,
                               tooltip="Multiples of the scene radius to keep the camera at, at "
                                       "minimum. 1.0 sits right on the shell; raise it to pull back."),
                IO.Boolean.Input("respect_camera", default=True,
                                 tooltip="When a camera is connected, keep the distance and aim you "
                                         "chose and only push out far enough to clear the geometry. "
                                         "Turn this OFF to have the node frame the whole splat "
                                         "instead, which overrides your composition."),
                IO.Boolean.Input("recentre_target", default=False,
                                 tooltip="Re-aim at the splat's centre, discarding the look-at point "
                                         "you framed. Off by default so a close-up survives."),
                IO.Float.Input("fallback_yaw", default=40.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Used only when no camera_info is connected."),
                IO.Float.Input("fallback_pitch", default=22.0, min=-89.0, max=89.0, step=1.0),
                IO.Float.Input("fov", default=40.0, min=1.0, max=120.0, step=1.0),
            ],
            outputs=[IO.Load3DCamera.Output(display_name="camera_info"),
                     IO.Float.Output(display_name="distance")],
        )

    @classmethod
    def execute(cls, splat, margin, respect_camera, recentre_target, fallback_yaw, fallback_pitch,
                fov, camera_info=None) -> IO.NodeOutput:
        end = _real_len(splat, 0)
        pos = splat.positions[0, :end].float()
        centre_splat, radius = rc.scene_bounds(pos)
        centre = rc.viewer_to_splat(centre_splat)      # involution: splat -> viewer

        if camera_info:
            cam_pos, cam_tgt = _camera_vectors(camera_info)
            fov_out = float(camera_info.get("fov", fov) or fov)
        else:                                          # nothing connected: a plain 3/4 view
            d = _yaw_pitch_to_dir(fallback_yaw, fallback_pitch)
            cam_pos = [centre[i] - d[i] * radius * 2.5 for i in range(3)]
            cam_tgt = list(centre)
            fov_out = float(fov)

        # With a camera connected, this is a safety net, not a framing tool: only
        # push out far enough to clear the geometry, because the fit distance
        # would throw away a close-up the user deliberately composed. Without
        # one, there is no composition to protect, so frame the whole splat.
        if camera_info and respect_camera:
            # clearance must use the radius that actually encloses the splat, not
            # the typical one - the long axis of a room is what you end up inside
            enclosing = rc.enclosing_radius(pos, centre_splat)
            need = max(enclosing, radius) * float(margin)
        else:
            need = rc.frame_distance(radius, fov_out, margin)
        new_pos, moved = rc.push_camera_outside(cam_pos, centre, need)
        target = list(centre) if recentre_target else cam_tgt
        dist = math.sqrt(sum((new_pos[i] - target[i]) ** 2 for i in range(3)))
        if moved:
            logging.info(f"Frame Splat Camera: camera was inside a radius-{radius:.2f} splat "
                         f"(ComfyUI's rasteriser renders nothing from in there); pushed out to "
                         f"{dist:.2f}. Raise margin if it still clips.")

        xyz = lambda v: {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
        # the quaternion is not optional - see relight_core.lookat_quat_xyzw
        q = rc.lookat_quat_xyzw(new_pos, target)
        out = {"position": xyz(new_pos), "target": xyz(target), "fov": fov_out,
               "quaternion": {"x": q[0], "y": q[1], "z": q[2], "w": q[3]},
               "cameraType": str(camera_info.get("cameraType", "perspective"))
                             if camera_info else "perspective",
               "zoom": float(camera_info.get("zoom", 1.0) or 1.0) if camera_info else 1.0}
        return IO.NodeOutput(out, dist)


class SplatLightFromCamera(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatLightFromCamera",
            display_name="Splat Light from Camera",
            search_aliases=["light from view", "aim light", "interactive light",
                            "place light", "light gizmo", "viewport light"],
            category="3d/splat/relight",
            description="Turn a camera into a light: the light is placed where the camera is and aimed "
                        "at what it looks at. Wire a Preview Splat viewer's camera_info output into "
                        "this and you can drag the viewport to aim the light and watch the result, "
                        "instead of guessing yaw and pitch numbers. Works with Create Camera Info too.",
            inputs=[
                IO.Load3DCamera.Input("camera_info", optional=True,
                                      tooltip="Usually a Preview Splat viewer's camera_info OUTPUT - "
                                              "orbit that viewport to move this light. Until the "
                                              "viewport has been touched it reports nothing, so the "
                                              "fallback yaw/pitch below is used instead."),
                IO.Float.Input("fallback_yaw", default=35.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Used only when camera_info is empty - i.e. on the very first "
                                       "run, before anyone has orbited the viewer."),
                IO.Float.Input("fallback_pitch", default=30.0, min=-89.0, max=89.0, step=1.0),
                IO.Combo.Input("light_type", options=["directional", "point", "spot"],
                               tooltip="directional ignores the camera's distance and uses only its "
                                       "direction; point and spot are placed at the camera position."),
                IO.Color.Input("color", default="#FFD2A8"),
                IO.Float.Input("intensity", default=3.5, min=0.0, max=1000.0, step=0.1),
                IO.Boolean.Input("cast_shadow", default=True),
                IO.Float.Input("cone_angle", default=45.0, min=1.0, max=179.0, step=1.0,
                               tooltip="Spot only: full width of the bright core, degrees."),
                IO.Float.Input("cone_softness", default=10.0, min=0.0, max=90.0, step=1.0,
                               tooltip="Spot only: extra degrees of soft falloff."),
                IO.Float.Input("distance_scale", default=1.0, min=0.05, max=20.0, step=0.05,
                               advanced=True,
                               tooltip="Push the light nearer (<1) or further (>1) along the view "
                                       "direction. Point/spot only."),
                IO.Float.Input("orbit_speed", default=0.0, min=-45.0, max=45.0, step=0.1,
                               tooltip="Degrees per frame around world Y, for animated sequences."),
                LightRig.Input("rig", optional=True),
            ],
            outputs=[LightRig.Output(display_name="rig")],
        )

    @classmethod
    def execute(cls, light_type, color, intensity, cast_shadow, cone_angle, cone_softness,
                distance_scale, orbit_speed, fallback_yaw, fallback_pitch,
                camera_info=None, rig=None) -> IO.NodeOutput:
        pos, tgt = _camera_vectors(camera_info) if camera_info else (None, None)
        if not camera_info or pos == tgt:
            # No viewport camera yet (nobody has orbited, or it sits on its target),
            # so fall back to plain yaw/pitch rather than failing the whole graph.
            direction = _yaw_pitch_to_dir(fallback_yaw, fallback_pitch)
            light_pos = [-d * 4.0 * float(distance_scale) for d in direction]
        else:
            direction, light_pos = rc.light_from_camera(pos, tgt, distance_scale)
        lt = {"type": light_type, "color": list(_hex_to_rgb(color)), "intensity": float(intensity),
              "cast_shadow": bool(cast_shadow), "orbit_speed": float(orbit_speed),
              "direction": direction, "position": None}
        if light_type != "directional":
            lt["position"] = light_pos
            lt["radius"] = 0.0
            if light_type == "spot":
                lt["cone_angle"] = float(cone_angle)
                lt["cone_softness"] = float(cone_softness)
        return IO.NodeOutput(list(rig or []) + [lt])


class SplatEnvironmentLight(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatEnvironmentLight",
            display_name="Splat Environment Light",
            search_aliases=["hdri", "image based lighting", "ibl", "environment map"],
            category="3d/splat/relight",
            description="Light the splat with an equirectangular environment image (HDRI). The image is "
                        "projected onto order-2 spherical harmonics, which captures essentially all of a "
                        "diffuse irradiance field, so this is fast and smooth rather than a ray-traced "
                        "lookup. Feed it any panoramic image - including one generated upstream in ComfyUI.",
            inputs=[
                IO.Image.Input("environment",
                               tooltip="Equirectangular (2:1) panorama. Rows map to elevation, columns to azimuth."),
                IO.Float.Input("intensity", default=1.0, min=0.0, max=100.0, step=0.05),
                IO.Color.Input("tint", default="#FFFFFF"),
                IO.Float.Input("rotation", default=0.0, min=-360.0, max=360.0, step=1.0,
                               tooltip="Spin the environment around the vertical axis, degrees."),
                IO.Boolean.Input("input_is_srgb", default=True,
                                 tooltip="Linearise the image before projecting. Leave ON for ordinary "
                                         "PNG/JPG panoramas from Load Image. Turn OFF when the source "
                                         "is Load HDRI (EXR / HDR), which already gives linear "
                                         "radiance - leaving it on would darken the environment twice."),
                LightRig.Input("rig", optional=True),
            ],
            outputs=[LightRig.Output(display_name="rig")],
        )

    @classmethod
    def execute(cls, environment, intensity, tint, rotation, input_is_srgb, rig=None) -> IO.NodeOutput:
        img = environment[0, ..., :3].float()
        if input_is_srgb:
            img = rc.srgb_to_linear(img)
        if rotation:
            shift = int(round((rotation % 360.0) / 360.0 * img.shape[1]))
            if shift:
                img = torch.roll(img, shifts=shift, dims=1)
        # project straight into splat space so it evaluates against splat normals
        coeff = rc.equirect_to_sh9(img, to_splat_frame=True)
        lt = {"type": "environment", "sh": coeff.cpu(), "color": list(_hex_to_rgb(tint)),
              "intensity": float(intensity), "cast_shadow": False, "orbit_speed": 0.0}
        return IO.NodeOutput(list(rig or []) + [lt])


# --------------------------------------------------------------------------
# delight
# --------------------------------------------------------------------------

class DelightSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="DelightSplat",
            display_name="Delight Splat",
            search_aliases=["remove lighting", "albedo", "flatten lighting", "unbake"],
            category="3d/splat/relight",
            description="Estimate and remove the lighting baked into a captured splat, leaving an albedo "
                        "estimate to relight. A capture records colour *and* the light it was shot under; "
                        "without this step new lights stack on top of the original ones and the result "
                        "looks doubly lit. Whatever part of the colour is explained by surface orientation "
                        "is treated as illumination and divided out.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Float.Input("amount", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="0 keeps the original bake, 1 removes all estimated shading. "
                                       "Back off if the result looks flat or grey."),
                IO.Float.Input("shadow_floor", default=0.15, min=0.01, max=1.0, step=0.01,
                               tooltip="Lower bound on the divisor. Raise it if deep shadows come back "
                                       "as blown-out noise."),
                IO.Float.Input("strip_view_dependent", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Fade out the higher spherical-harmonic bands, which hold the "
                                       "baked specular highlights of the original lighting."),
                IO.Combo.Input("orient_normals", options=["visibility", "outward", "keep"],
                               tooltip="How to decide which way each surfel faces. 'visibility' ray-marches "
                                       "and is the reliable choice; 'outward' is a fast centroid test that "
                                       "suits convex objects; 'keep' trusts the file."),
                IO.Float.Input("normal_smoothing", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Average each normal with its spatial neighbours. Real captures have "
                                       "round gaussians whose individual normals are close to arbitrary; "
                                       "neighbours on the same surface agree, so this recovers the plane. "
                                       "Raise it if the normals debug view looks like coloured noise."),
                IO.Int.Input("smooth_radius", default=1, min=1, max=4, step=1, advanced=True,
                             tooltip="Neighbourhood radius in grid voxels. Larger smooths harder and "
                                     "flattens genuine detail."),
                IO.Boolean.Input("weight_by_flatness", default=True,
                                 tooltip="Trust flat gaussians more than round ones when fitting the "
                                         "lighting - round blobs have no meaningful normal."),
                IO.Float.Input("shadow_lift", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="Last resort for baked shadows the physical fit cannot touch. "
                                       "'amount' only removes shading explained by surface orientation, "
                                       "so on a diffusely lit capture it removes almost nothing and cast "
                                       "shadows survive. This squashes luminance contrast directly: a "
                                       "ratio r becomes r^(1-lift), so 0.5 turns 4x into 2x. It cannot "
                                       "tell a shadow from dark paint and lightens both - raise it until "
                                       "the shadows stop reading, then stop."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, amount, shadow_floor, strip_view_dependent, orient_normals,
                normal_smoothing, smooth_radius, weight_by_flatness,
                shadow_lift=0.0) -> IO.NodeOutput:
        device = comfy.model_management.get_torch_device()
        sh = _clone_sh(splat)
        b = splat.positions.shape[0]
        pbar = comfy.utils.ProgressBar(b) if b > 1 else None

        for i in range(b):
            end = _real_len(splat, i)
            if end == 0:
                continue
            pos = splat.positions[i, :end].to(device).float()
            sc = splat.scales[i, :end].to(device).float()
            rot = splat.rotations[i, :end].to(device).float()
            op = splat.opacities[i, :end].to(device).float().reshape(-1)
            rgb = rc.dc_to_rgb(sh[i, :end, 0, :].to(device).float())

            need_grid = orient_normals == "visibility" or normal_smoothing > 0
            grid = rc.DensityGrid(pos, op, sc) if need_grid else None
            n = _splat_normals(splat, i, pos, sc, rot, op, orient_normals, grid, 1.0,
                               smoothing=normal_smoothing, smooth_radius=smooth_radius)
            w = rc.anisotropy(sc) * op if weight_by_flatness else None

            albedo, _ = rc.delight(rgb, n, amount=amount, weights=w, floor=shadow_floor,
                                   shadow_lift=shadow_lift)
            sh[i, :end, 0, :] = rc.rgb_to_dc(albedo).to(dtype=sh.dtype, device=sh.device)
            if sh.shape[2] > 1 and strip_view_dependent > 0:
                sh[i, :end, 1:, :] *= (1.0 - float(strip_view_dependent))
            if pbar:
                pbar.update(1)

        return IO.NodeOutput(_out_splat(splat, sh))


# --------------------------------------------------------------------------
# relight
# --------------------------------------------------------------------------

class RelightSplat(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="RelightSplat",
            display_name="Relight Splat",
            search_aliases=["relight", "pbr splat", "shade splat", "light gaussian"],
            category="3d/splat/relight",
            description="Shade every gaussian with a lighting rig and write the result back into the splat. "
                        "Because the primitives themselves are relit, the lighting is consistent from every "
                        "camera angle and survives export to .ply. Works per frame on an animated splat, so "
                        "moving geometry casts moving shadows.",
            inputs=[
                IO.Splat.Input("splat", tooltip="Ideally the output of Delight Splat."),
                LightRig.Input("rig"),
                IO.Float.Input("roughness", default=0.55, min=0.02, max=1.0, step=0.01,
                               tooltip="Surface microroughness. Low = tight glossy highlight, high = matte."),
                IO.Float.Input("metallic", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Metals tint their highlight with the albedo and lose diffuse response."),
                IO.Float.Input("specular", default=0.5, min=0.0, max=4.0, step=0.05,
                               tooltip="Strength of the specular lobe. 0 gives pure matte diffuse."),
                IO.Color.Input("ambient_color", default="#20242C"),
                IO.Float.Input("ambient_intensity", default=1.0, min=0.0, max=20.0, step=0.05,
                               tooltip="Uniform fill so unlit sides do not go pure black."),
                IO.Float.Input("exposure", default=1.0, min=0.01, max=20.0, step=0.05),
                IO.Combo.Input("tonemap", options=["aces", "reinhard", "none"],
                               tooltip="How to fit the lit result back into displayable range. "
                                       "'none' hard-clips and will posterise bright highlights."),
                IO.Combo.Input("orient_normals", options=["visibility", "outward", "keep"]),
                IO.Float.Input("normal_smoothing", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Average each normal with its spatial neighbours. Real captures have "
                                       "round gaussians whose individual normals are close to arbitrary; "
                                       "neighbours on the same surface agree, so this recovers the plane. "
                                       "Raise it if the relight looks speckled."),
                IO.Int.Input("smooth_radius", default=1, min=1, max=4, step=1, advanced=True,
                             tooltip="Neighbourhood radius in grid voxels. Larger smooths harder and "
                                     "flattens genuine detail."),
                IO.Boolean.Input("cast_shadows", default=True,
                                 tooltip="Ray-march the splat's own density so it shadows itself. "
                                         "This is what makes relighting read as three-dimensional."),
                IO.Int.Input("shadow_quality", default=24, min=4, max=128, step=4, advanced=True,
                             tooltip="Ray-march steps per shadow ray. Higher is cleaner and slower."),
                IO.Float.Input("shadow_density", default=1.0, min=0.05, max=20.0, step=0.05, advanced=True,
                               tooltip="How opaque the splat is to shadow rays. Raise for darker, "
                                       "more contacty shadows."),
                IO.Float.Input("ao_strength", default=0.6, min=0.0, max=1.0, step=0.05,
                               tooltip="Ambient occlusion applied to the ambient and environment terms, "
                                       "which is what grounds crevices and contact points."),
                IO.Float.Input("ao_radius", default=0.25, min=0.01, max=10.0, step=0.01,
                               tooltip="World-space radius the occlusion test samples over."),
                IO.Int.Input("grid_resolution", default=128, min=32, max=384, step=16, advanced=True,
                             tooltip="Occupancy grid resolution used for shadows, AO and normal "
                                     "orientation. Higher resolves finer contact, costs memory."),
                IO.Combo.Input("view_vector", options=["normal", "camera"], advanced=True,
                               tooltip="Which view direction the specular highlight uses. 'normal' is "
                                       "view-independent and stays stable on a turntable; 'camera' is "
                                       "physically correct for one fixed viewpoint (needs camera_info)."),
                IO.Load3DCamera.Input("camera_info", optional=True,
                                      tooltip="Only used when view_vector is 'camera'."),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, rig, roughness, metallic, specular, ambient_color, ambient_intensity,
                exposure, tonemap, orient_normals, normal_smoothing, smooth_radius, cast_shadows,
                shadow_quality, shadow_density, ao_strength, ao_radius, grid_resolution,
                view_vector, camera_info=None) -> IO.NodeOutput:
        if not rig:
            raise ValueError("Relight Splat: the rig is empty. Connect at least one Splat Light "
                             "or Splat Environment Light.")

        device = comfy.model_management.get_torch_device()
        sh = _clone_sh(splat)
        b = splat.positions.shape[0]
        amb = torch.tensor(_hex_to_rgb(ambient_color), device=device) * float(ambient_intensity)

        cam_pos = None
        if view_vector == "camera" and camera_info is not None:
            p = camera_info.get("position", {})
            # camera_info is viewer space; the gaussians are not
            cam_pos = torch.tensor([rc.viewer_to_splat(
                [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)])],
                device=device, dtype=torch.float32)

        pbar = comfy.utils.ProgressBar(b) if b > 1 else None
        for i in range(b):
            end = _real_len(splat, i)
            if end == 0:
                continue
            pos = splat.positions[i, :end].to(device).float()
            sc = splat.scales[i, :end].to(device).float()
            rot = splat.rotations[i, :end].to(device).float()
            op = splat.opacities[i, :end].to(device).float().reshape(-1)
            albedo = rc.dc_to_rgb(sh[i, :end, 0, :].to(device).float())

            need_grid = (cast_shadows or ao_strength > 0 or orient_normals == "visibility"
                         or normal_smoothing > 0)
            grid = rc.DensityGrid(pos, op, sc, resolution=grid_resolution) if need_grid else None
            n = _splat_normals(splat, i, pos, sc, rot, op, orient_normals, grid, shadow_density,
                               smoothing=normal_smoothing, smooth_radius=smooth_radius)

            if cam_pos is not None:
                v = cam_pos - pos
                v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            else:
                v = n

            ao = None
            if ao_strength > 0 and grid is not None:
                vis = rc.ambient_occlusion(grid, pos, n, radius=ao_radius, sigma=shadow_density)
                ao = torch.lerp(torch.ones_like(vis), vis, float(ao_strength))

            # animate in viewer space (so orbit_speed reads as the artist expects),
            # then convert the result into the splat's frame for shading
            lights = [_to_splat_frame(_animate(lt, i)) for lt in rig]
            lit = rc.shade(albedo, n, pos, lights, v,
                           roughness=roughness, metallic=metallic,
                           ambient_color=amb, ao=ao,
                           grid=grid if cast_shadows else None,
                           shadow_steps=shadow_quality, shadow_sigma=shadow_density,
                           specular_strength=specular)
            out = rc.tonemap(lit, exposure=exposure, mode=tonemap)

            dc = rc.rgb_to_dc(out).to(dtype=sh.dtype, device=sh.device)
            sh[i, :end, 0, :] = dc
            if sh.shape[2] > 1:
                # the old higher-order bands describe the *previous* lighting's
                # view-dependence, so keeping them would contradict the new rig
                sh[i, :end, 1:, :] = 0.0
            if pbar:
                pbar.update(1)

        return IO.NodeOutput(_out_splat(splat, sh))


# --------------------------------------------------------------------------
# inspection
# --------------------------------------------------------------------------

class SplatRelightDebug(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="SplatRelightDebug",
            display_name="Splat Relight Debug",
            search_aliases=["show normals", "debug splat", "visualise normals", "ao preview"],
            category="3d/splat/relight",
            description="Bake a diagnostic channel into the splat's colour so you can look at it with "
                        "Render Splat. Check this first when a relight looks wrong: flipped normals are "
                        "the usual culprit, and in 'normals' mode a correctly oriented capture shows a "
                        "smooth colour gradient across the surface rather than a noisy patchwork.",
            inputs=[
                IO.Splat.Input("splat"),
                IO.Combo.Input("channel", options=["normals", "ambient_occlusion", "estimated_shading",
                                                   "albedo", "flatness"]),
                IO.Combo.Input("orient_normals", options=["visibility", "outward", "keep"]),
                IO.Float.Input("normal_smoothing", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="Average each normal with its spatial neighbours. Compare 0 and 1 "
                                       "here to judge how trustworthy this capture's normals are."),
                IO.Int.Input("smooth_radius", default=1, min=1, max=4, step=1, advanced=True),
                IO.Float.Input("ao_radius", default=0.25, min=0.01, max=10.0, step=0.01),
                IO.Int.Input("grid_resolution", default=128, min=32, max=384, step=16, advanced=True),
            ],
            outputs=[IO.Splat.Output(display_name="splat")],
        )

    @classmethod
    def execute(cls, splat, channel, orient_normals, normal_smoothing, smooth_radius,
                ao_radius, grid_resolution) -> IO.NodeOutput:
        device = comfy.model_management.get_torch_device()
        sh = _clone_sh(splat)
        b = splat.positions.shape[0]
        pbar = comfy.utils.ProgressBar(b) if b > 1 else None

        for i in range(b):
            end = _real_len(splat, i)
            if end == 0:
                continue
            pos = splat.positions[i, :end].to(device).float()
            sc = splat.scales[i, :end].to(device).float()
            rot = splat.rotations[i, :end].to(device).float()
            op = splat.opacities[i, :end].to(device).float().reshape(-1)
            rgb = rc.dc_to_rgb(sh[i, :end, 0, :].to(device).float())

            need_grid = (channel == "ambient_occlusion" or orient_normals == "visibility"
                         or normal_smoothing > 0)
            grid = rc.DensityGrid(pos, op, sc, resolution=grid_resolution) if need_grid else None
            n = _splat_normals(splat, i, pos, sc, rot, op, orient_normals, grid, 1.0,
                               smoothing=normal_smoothing, smooth_radius=smooth_radius)

            if channel == "normals":
                # encode into viewer space so this reads the same way as
                # Render Splat's own "normal" style
                col = rc.flip_frame(n) * 0.5 + 0.5
            elif channel == "ambient_occlusion":
                col = rc.ambient_occlusion(grid, pos, n, radius=ao_radius)[:, None].expand(-1, 3)
            elif channel == "estimated_shading":
                s, _ = rc.estimate_shading(n, rgb, rc.anisotropy(sc) * op)
                col = (s * 0.5)[:, None].expand(-1, 3).clamp(0, 1)
            elif channel == "flatness":
                col = rc.anisotropy(sc)[:, None].expand(-1, 3)
            else:  # albedo
                col = rgb

            sh[i, :end, 0, :] = rc.rgb_to_dc(col.contiguous()).to(dtype=sh.dtype, device=sh.device)
            if sh.shape[2] > 1:
                sh[i, :end, 1:, :] = 0.0
            if pbar:
                pbar.update(1)

        return IO.NodeOutput(_out_splat(splat, sh))


class SplatRelightExtension(ComfyExtension):
    @override
    async def get_node_list(self):
        return [LoadSplatSequence, LoadHDRI, CropSplat, SplatViewport, SplatLight, SplatLightGizmo, SplatLightFromCamera,
                SplatEnvironmentLight, DelightSplat, RelightSplat, FrameSplatCamera,
                SplatRelightDebug]


async def comfy_entrypoint() -> ComfyExtension:
    return SplatRelightExtension()
