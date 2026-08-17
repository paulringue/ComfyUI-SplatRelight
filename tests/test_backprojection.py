"""Unit tests for multi-view back-projection.

Run with:  python -m pytest tests/ -q   (or plain `python tests/test_backprojection.py`)

Back-projection is the step that lets a 2D image model write onto 3D gaussians,
so a silent bug here would quietly corrupt every gaussian in a capture rather
than throw. These tests pin the geometry against hand-computable answers and
the accumulation against invariants that must hold whatever the weights are.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import relight_core as rc  # noqa: E402

PASS = FAIL = 0


def check(cond, msg):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


def test_projection_geometry():
    print("projection geometry")
    W, H, FOV = 200, 100, 90.0
    cam, tgt = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]          # looking down +Z

    pts = torch.tensor([
        [0.0, 0.0, 5.0],      # dead ahead   -> centre
        [0.0, 0.0, -5.0],     # behind       -> rejected
        [0.0, 0.0, 1.0],      # ahead, close -> centre
    ])
    px, py, z, inside = rc.project(pts, cam, tgt, FOV, W, H)
    check(abs(px[0].item() - W / 2) < 1e-3 and abs(py[0].item() - H / 2) < 1e-3,
          f"point dead ahead lands at frame centre ({px[0]:.1f},{py[0]:.1f})")
    check(not bool(inside[1]), "point behind the camera is rejected")
    check(bool(inside[0]) and bool(inside[2]), "points in front are accepted")
    check(abs(z[0].item() - 5.0) < 1e-5, "depth equals distance along the view axis")

    # at fov 90 the half-height spans exactly `depth` in world units
    focal = (H / 2) / math.tan(math.radians(FOV) / 2)
    edge = torch.tensor([[0.0, -1.0 * 5.0 * (H / 2) / focal / (H / 2) * (H / 2), 5.0]])
    # simpler: a point one focal-height up should sit at the top edge
    up_pt = torch.tensor([[0.0, -5.0 * math.tan(math.radians(FOV) / 2), 5.0]])
    _, upy, _, _ = rc.project(up_pt, cam, tgt, FOV, W, H)
    check(abs(upy[0].item() - 0.0) < 1.0,
          f"a point at the top of the frustum maps to the top edge (y={upy[0]:.2f})")

    # splat frame is Y-down, so -Y is up: verify the sign convention explicitly
    check(upy[0].item() < H / 2, "negative Y projects to the upper half of the image")


def test_sample_image():
    print("image sampling")
    img = torch.zeros(2, 2, 1)
    img[0, 0, 0] = 0.0
    img[0, 1, 0] = 1.0
    img[1, 0, 0] = 2.0
    img[1, 1, 0] = 3.0
    at = lambda x, y: rc.sample_image(img, torch.tensor([x]), torch.tensor([y]))[0, 0].item()
    check(abs(at(0.0, 0.0) - 0.0) < 1e-6, "exact corner sample returns that texel")
    check(abs(at(1.0, 0.0) - 1.0) < 1e-6, "exact texel to the right returns its value")
    check(abs(at(0.5, 0.0) - 0.5) < 1e-6, "horizontal midpoint interpolates")
    check(abs(at(0.0, 0.5) - 1.0) < 1e-6, "vertical midpoint interpolates")
    check(abs(at(0.5, 0.5) - 1.5) < 1e-6, "centre averages all four texels")
    # out of range must clamp rather than wrap or crash
    check(abs(at(-5.0, -5.0) - 0.0) < 1e-6, "out-of-range sample clamps to the edge")
    check(abs(at(99.0, 99.0) - 3.0) < 1e-6, "far out-of-range clamps to the far corner")


def test_near_plane_cut():
    print("near-plane cut")
    cam, tgt = [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]
    pts = torch.tensor([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0], [0.0, 0.0, 0.0], [3.0, 3.0, 2.0]])
    m = rc.near_plane_mask(pts, cam, tgt, offset=0.1)
    check(bool(m[0]), "geometry in front of the camera survives")
    check(not bool(m[1]), "geometry behind the camera is cut")
    check(not bool(m[2]), "geometry at the camera itself is cut")
    check(bool(m[3]), "off-axis geometry in front survives")

    # the cut must follow the view direction, not an axis
    cam2, tgt2 = [0.0, 0.0, 0.0], [1.0, 0.0, 1.0]        # 45 degrees
    diag = torch.tensor([[1.0, 0.0, 1.0], [-1.0, 0.0, -1.0], [1.0, 0.0, -1.0]])
    m2 = rc.near_plane_mask(diag, cam2, tgt2, offset=0.0)
    check(bool(m2[0]) and not bool(m2[1]),
          "an oblique view direction cuts along the view plane, not an axis")
    check(not bool(m2[2]), "a point perpendicular-behind the oblique plane is cut")


def _sphere(n=800, radius=1.0, device="cpu"):
    i = torch.arange(n, dtype=torch.float32, device=device) + 0.5
    z = 1 - 2 * i / n
    r = torch.sqrt((1 - z * z).clamp_min(0))
    ph = i * math.pi * (3 - math.sqrt(5.0))
    dirs = torch.stack([r * torch.cos(ph), r * torch.sin(ph), z], -1)
    return dirs * radius, dirs


def test_backproject_constant_recovery():
    """Whatever the weights, constant input must come back exactly constant."""
    print("back-projection: constant recovery")
    pos, nrm = _sphere(600, 1.0)
    cams = rc.orbit_cameras([0.0, 0.0, 0.0], 4.0, count=8, pitch_deg=15.0)
    VAL = torch.tensor([0.25, 0.5, 0.75])
    images = [VAL.expand(64, 96, 3).clone() for _ in cams]

    vals, cov = rc.backproject(images, cams, pos, normals=nrm, grid=None, fov_deg=60.0)
    seen = cov > 1e-6
    check(int(seen.sum()) > 400, f"most gaussians are seen by at least one view ({int(seen.sum())}/600)")
    err = (vals[seen] - VAL).abs().max().item()
    check(err < 1e-5, f"every seen gaussian recovers the constant exactly (max err {err:.2e})")
    check(torch.allclose(vals[~seen], torch.zeros(3)) if (~seen).any() else True,
          "unseen gaussians are left at zero rather than given a wrong value")
    check(bool((cov >= 0).all()), "coverage is never negative")


def test_backproject_weighting():
    """A view that cannot see a gaussian must not colour it."""
    print("back-projection: weighting")
    pos = torch.tensor([[0.0, 0.0, 0.0]])
    nrm = torch.tensor([[0.0, 0.0, -1.0]])              # faces -Z
    front = ([0.0, 0.0, -3.0], [0.0, 0.0, 0.0])         # sees the front face
    back = ([0.0, 0.0, 3.0], [0.0, 0.0, 0.0])           # sees the back

    red = torch.tensor([1.0, 0.0, 0.0]).expand(32, 32, 3).clone()
    blue = torch.tensor([0.0, 0.0, 1.0]).expand(32, 32, 3).clone()

    # facing term uses |cos|, so both views count; the average must be the midpoint
    v, _ = rc.backproject([red, blue], [front, back], pos, normals=nrm, fov_deg=60.0)
    check(abs(v[0, 0].item() - 0.5) < 1e-4 and abs(v[0, 2].item() - 0.5) < 1e-4,
          f"two equally-weighted views average ({v[0,0]:.3f},{v[0,2]:.3f})")

    # a gaussian outside the frustum of one view takes the other view only
    off = torch.tensor([[0.0, 0.0, 0.0]])
    near_miss = ([0.0, 0.0, -3.0], [0.0, 0.0, 0.0])
    aimed_away = ([0.0, 0.0, -3.0], [0.0, 0.0, -6.0])   # looks away from the point
    v2, c2 = rc.backproject([red, blue], [near_miss, aimed_away], off, normals=None, fov_deg=40.0)
    check(abs(v2[0, 0].item() - 1.0) < 1e-4 and v2[0, 2].item() < 1e-4,
          f"a view aimed away contributes nothing ({v2[0,0]:.3f},{v2[0,2]:.3f})")
    check(c2[0].item() > 0, "the remaining view still yields coverage")


def test_backproject_occlusion():
    """A wall between gaussian and camera must suppress that view."""
    print("back-projection: occlusion")
    # a dense slab at z=0, and a lone gaussian behind it at z=+1
    g = torch.rand(20000, 3) * torch.tensor([4.0, 4.0, 0.08]) - torch.tensor([2.0, 2.0, 0.04])
    target = torch.tensor([[0.0, 0.0, 1.0]])
    pos = torch.cat([g, target], 0)
    op = torch.ones(pos.shape[0])
    sc = torch.full((pos.shape[0], 3), 0.03)
    grid = rc.DensityGrid(pos, op, sc, resolution=64)

    cam = ([0.0, 0.0, -4.0], [0.0, 0.0, 1.0])           # slab sits between camera and target
    # the depth buffer has to be built from the WHOLE scene, not the query subset
    buf = rc.depth_buffer(pos, cam[0], cam[1], 60.0, 64, 64, op)
    w = rc.view_weight(target, None, cam[0], cam[1], 60.0, 64, 64, depth_buf=buf)
    w_off = rc.view_weight(target, None, cam[0], cam[1], 60.0, 64, 64, use_zbuffer=False)
    check(w_off[0].item() > 0.5, "with occlusion disabled the hidden gaussian is naively weighted")
    check(w[0].item() < 1e-6,
          f"the occluding slab rejects it outright ({w_off[0]:.3f} -> {w[0]:.3f})")

    behind = ([0.0, 0.0, 4.0], [0.0, 0.0, 1.0])         # same side, nothing in the way
    buf2 = rc.depth_buffer(pos, behind[0], behind[1], 60.0, 64, 64, op)
    w2 = rc.view_weight(target, None, behind[0], behind[1], 60.0, 64, 64, depth_buf=buf2)
    check(w2[0].item() > w[0].item(),
          f"an unobstructed camera keeps the weight ({w[0]:.3f} vs {w2[0]:.3f})")

    # a gaussian buried *inside* the slab must also be rejected from the front
    inside_slab = torch.tensor([[0.0, 0.0, 0.30]])
    w3 = rc.view_weight(inside_slab, None, cam[0], cam[1], 60.0, 64, 64, depth_buf=buf)
    check(w3[0].item() < 1e-6, "geometry behind the first surface is rejected")


def test_orbit_cameras():
    print("orbit cameras")
    centre, radius = [1.0, -2.0, 0.5], 3.0
    cams = rc.orbit_cameras(centre, radius, count=12, pitch_deg=20.0)
    check(len(cams) == 12, "returns the requested number of cameras")
    c = torch.tensor(centre)
    d = [(torch.tensor(p) - c).norm().item() for p, _ in cams]
    check(max(abs(x - radius) for x in d) < 1e-4,
          f"every camera sits at the requested radius ({min(d):.4f}..{max(d):.4f})")
    check(all(t == centre for _, t in cams), "every camera targets the centre")
    ys = [p[1] for p, _ in cams]
    check(max(ys) - min(ys) < 1e-6, "a single ring keeps a constant height")
    check(ys[0] < centre[1], "positive pitch lifts the camera (splat frame is Y-down)")
    az = {round(math.atan2(p[0] - centre[0], p[2] - centre[2]), 4) for p, _ in cams}
    check(len(az) == 12, "cameras are at distinct azimuths")


def test_shadow_mask():
    """The mask must select relative darkness, not dark material."""
    print("shadow mask")
    H, W = 128, 128
    # uniform shading with one darkened patch = a shadow
    s = torch.ones(H, W)
    s[40:70, 30:80] = 0.35
    m, rel = rc.shadow_mask(s, ratio=0.62, context=25.0, feather=0.0)
    inside = m[48:62, 40:70].mean().item()
    outside = torch.cat([m[:30].reshape(-1), m[100:].reshape(-1)]).mean().item()
    check(inside > 0.9, f"the darkened patch is selected ({inside:.2f})")
    check(outside < 0.02, f"unshadowed area is not selected ({outside:.3f})")

    # a uniformly dark image has no *relative* shadow - nothing may be selected
    flat = torch.full((H, W), 0.15)
    m2, _ = rc.shadow_mask(flat, ratio=0.62, context=25.0, feather=0.0)
    check(m2.mean().item() < 1e-6,
          f"uniformly dark shading selects nothing ({m2.mean():.4f}) - dark paint is not shadow")

    # scale invariance: shading is up_to_scale, so 10x input must give the same mask
    m3, _ = rc.shadow_mask(s * 10.0, ratio=0.62, context=25.0, feather=0.0)
    check(torch.equal(m2 * 0 + m3, m), "mask is invariant to the shading's arbitrary scale")

    # a (H,W,3) input must behave like its luminance
    m4, _ = rc.shadow_mask(s[..., None].expand(-1, -1, 3).contiguous(),
                           ratio=0.62, context=25.0, feather=0.0)
    check(torch.equal(m4, m), "colour shading input matches the luminance path")

    # feathering must soften the edge without moving the interior
    mf, _ = rc.shadow_mask(s, ratio=0.62, context=25.0, feather=3.0)
    check(mf[48:62, 40:70].mean().item() > 0.9, "feathering leaves the interior filled")
    check(float(((mf > 0.01) & (mf < 0.99)).sum()) > float(((m > 0.01) & (m < 0.99)).sum()),
          "feathering creates a soft border")

    check(rc.blur2d(torch.ones(16, 16), 2.0).std().item() < 1e-5,
          "blurring a constant field leaves it constant")

    # growing must strictly expand, never shrink, and must reach the penumbra
    base, _ = rc.shadow_mask(s, ratio=0.62, context=25.0, feather=0.0, grow=0)
    g8, _ = rc.shadow_mask(s, ratio=0.62, context=25.0, feather=0.0, grow=8)
    check(g8.sum().item() > base.sum().item(), "grow expands the mask")
    check(bool((g8 >= base - 1e-6).all()), "grow never removes anything already masked")
    a = float(base.sum()); b = float(g8.sum())
    check(b > a * 1.2, f"grow=8 meaningfully widens the region ({a:.0f} -> {b:.0f} px)")
    # the grown mask must extend beyond the original edge on every side
    ys, xs = torch.where(base > 0.5)
    ys2, xs2 = torch.where(g8 > 0.5)
    check(int(xs2.min()) < int(xs.min()) and int(xs2.max()) > int(xs.max())
          and int(ys2.min()) < int(ys.min()) and int(ys2.max()) > int(ys.max()),
          "grow extends past the original boundary on all four sides")


def test_flat_region_mask():
    """A fill must be kept away from object silhouettes."""
    print("flat region mask")
    H, W = 96, 96
    d = torch.full((H, W), 3.0)
    d[:, 48:] = 3.6                      # a step: two surfaces 0.6 apart
    e = rc.edge_map(d)
    check(e[:, 48].mean().item() > 0.9, "the depth step is detected as an edge")
    check(e[:, :40].mean().item() < 1e-6, "flat area away from the step has no edge")

    flat = rc.flat_region_mask(d, guard=7, smooth=0)
    check(flat[:, 48].mean().item() < 1e-6, "the silhouette itself is excluded")
    check(flat[:, 44:53].mean().item() < 1e-6, "a guard band either side is excluded")
    check(flat[:, :35].mean().item() > 0.99, "surface interior stays available")
    check(flat[:, 60:].mean().item() > 0.99, "the far surface interior stays available")

    # empty pixels (inf) must be treated as edges, not as smooth surface
    d2 = torch.full((H, W), 3.0)
    d2[20:30, 20:30] = float("inf")
    f2 = rc.flat_region_mask(d2, guard=3, smooth=0)
    check(f2[24, 24].item() < 1e-6, "an empty hole is excluded rather than trusted")
    check(f2[80, 80].item() > 0.99, "distant flat area is unaffected by the hole")

    # combining with a shadow mask must only ever shrink it
    sh = torch.ones(H, W)
    both = sh * flat
    check(bool((both <= sh + 1e-6).all()), "combining with flatness can only shrink a mask")

    # smoothing is the default because a splatted depth buffer is full of holes
    holey = torch.full((H, W), 3.0)
    holey[::3, ::3] = float("inf")                 # scatter, not real structure
    raw_flat = rc.flat_region_mask(holey, guard=3, smooth=0)
    sm_flat = rc.flat_region_mask(holey, guard=3, smooth=2)
    check(raw_flat.mean().item() < 0.05,
          f"unsmoothed, a holey buffer rejects nearly everything ({raw_flat.mean():.3f})")
    check(sm_flat.mean().item() > 0.8,
          f"smoothing recovers the flat surface ({sm_flat.mean():.3f})")
    step = torch.full((H, W), 3.0); step[:, 48:] = 3.6
    check(rc.flat_region_mask(step, guard=5, smooth=2)[:, 48].mean().item() < 1e-6,
          "smoothing still rejects a genuine depth step")


def test_split_fill_mask():
    """Generate and blend masks must be decoupled, with the ramp contained."""
    print("split fill mask")
    H, W = 160, 160
    core = torch.zeros(H, W)
    core[60:100, 60:100] = 1.0
    gen, blend = rc.split_fill_mask(core, blend_sigma=6.0)

    check(gen.sum().item() > core.sum().item(),
          "the generate mask is larger than the core region")
    check(blend.max().item() > 0.95, "the blend mask still reaches full strength inside")
    check(blend[78, 78].item() > 0.95, "core centre blends at full strength")

    # THE critical invariant: the ramp must never exceed the generated area,
    # or the composite pulls in the model's reconstruction of untouched surface
    leak = float(((blend > 1e-4) & (gen < 0.5)).sum())
    check(leak == 0, f"blend ramp is fully contained by the generate mask (leak {leak:.0f} px)")

    # there must actually be a soft ramp, not a step
    soft = float(((blend > 0.05) & (blend < 0.95)).sum())
    check(soft > 500, f"a genuine soft ramp exists ({soft:.0f} px in transition)")

    # a wider sigma must widen the ramp and the generated area together
    g2, b2 = rc.split_fill_mask(core, blend_sigma=14.0)
    soft2 = float(((b2 > 0.05) & (b2 < 0.95)).sum())
    check(soft2 > soft, "a larger sigma widens the transition")
    check(g2.sum().item() > gen.sum().item(), "a larger sigma also widens the generate mask")
    check(float(((b2 > 1e-4) & (g2 < 0.5)).sum()) == 0,
          "containment still holds at the larger sigma")

    # empty input must not blow up
    ge, be = rc.split_fill_mask(torch.zeros(H, W), blend_sigma=6.0)
    check(float(ge.sum()) == 0 and float(be.sum()) == 0, "an empty region yields empty masks")


def test_soft_shadow_mask():
    """Softness must come from the shadow's own gradient, not a chosen radius."""
    print("soft shadow mask")
    H, W = 200, 400
    CTX = 40.0
    # Shadows must be SMALL relative to the context blur, or each one drags down
    # its own reference and never reaches full alpha. Two narrow bars, far apart,
    # so each ramp can be measured against clean surface on one side only.
    s = torch.ones(H, W)
    s[80:120, 60:80] = 0.30                                    # crisp step
    ramp = torch.linspace(0.30, 1.0, 40)                       # 40px falloff
    s[80:120, 260:300] = ramp[None, :].expand(40, -1)
    a, rel = rc.soft_shadow_mask(s, ratio_full=0.50, ratio_none=0.82, context=CTX)

    check(a[100, 68].item() > 0.9,
          f"the crisp shadow core is fully masked (alpha {a[100,68]:.2f})")
    check(a[10, 10].item() < 0.05, "clean surface away from any shadow is untouched")
    check(a[100, 200].item() < 0.05, "clean surface between the two shadows is untouched")
    check(bool(((a >= 0) & (a <= 1)).all()), "alpha stays within [0,1]")

    # ramp width measured outward from each shadow's right edge only
    crisp = int(((a[100, 80:120] > 0.05) & (a[100, 80:120] < 0.95)).sum())
    soft = int(((a[100, 262:340] > 0.05) & (a[100, 262:340] < 0.95)).sum())
    check(soft > crisp,
          f"the gradual shadow gets a wider ramp than the crisp one ({soft} vs {crisp} px)")
    check(crisp < 15, f"a crisp edge is not expanded arbitrarily ({crisp} px of ramp)")

    # widening the ratio band widens the ramp without unseating the core
    a2, _ = rc.soft_shadow_mask(s, ratio_full=0.50, ratio_none=0.95, context=CTX)
    soft2 = int(((a2[100, 262:340] > 0.05) & (a2[100, 262:340] < 0.95)).sum())
    check(soft2 >= soft, f"a wider ratio band widens the transition ({soft2} vs {soft})")
    check(a2[100, 68].item() > 0.9, "the core stays fully masked regardless")

    # uniformly dark shading is not a shadow at all
    flat, _ = rc.soft_shadow_mask(torch.full((H, W), 0.2), context=CTX)
    check(float((flat > 0.5).float().mean()) < 1e-6,
          "uniformly dark shading yields no shadow")

    # a shadow wide against the context blur cannot reach full alpha - the
    # documented failure mode, pinned so it is a known property not a surprise
    wide = torch.ones(H, W)
    wide[:, 100:300] = 0.30
    aw, _ = rc.soft_shadow_mask(wide, ratio_full=0.50, ratio_none=0.82, context=CTX)
    check(aw[100, 200].item() < 0.9,
          f"a shadow wider than the context blur under-masks its core "
          f"(alpha {aw[100,200]:.2f}) - context must exceed the shadow")


def test_composite_fill():
    """Frequency-split compositing must hide the seam without losing texture."""
    print("frequency-split composite")
    torch.manual_seed(3)
    H, W = 200, 200
    tex = torch.rand(H, W, 3) * 0.25 + 0.4                 # detailed surface
    orig = tex.clone()
    orig[:, :100] *= 0.35                                   # left half in shadow
    core = torch.zeros(H, W)
    core[:, 20:80] = 1.0                                    # fill only part of it
    fill = tex.clone() * 1.02                               # a good, brighter fill

    out = rc.composite_fill(orig, fill, core, ramp_sigma=12.0)
    check(tuple(out.shape) == (H, W, 3), "shape is preserved")
    check(bool(torch.isfinite(out).all()), "output is finite")

    hp = lambda a: (rc.luminance(a) - rc.blur2d(rc.luminance(a), 2.0))
    far = (slice(None), slice(140, 200))
    check(torch.allclose(hp(out)[far], hp(orig)[far], atol=1e-5),
          "high frequencies far from the core are identical to the original")

    # the whole point: brightness must be corrected inside the core
    lit = rc.luminance(orig)[:, 150:].mean()
    before = rc.luminance(orig)[:, 40:60].mean()
    after = rc.luminance(out)[:, 40:60].mean()
    check(after > before * 1.5, f"the core is brightened ({before:.3f} -> {after:.3f})")

    # texture inside the core must not collapse
    t_in_o = hp(orig)[:, 40:60].std()
    t_in_n = hp(out)[:, 40:60].std()
    check(t_in_n > t_in_o * 0.8,
          f"texture survives inside the core ({t_in_o:.4f} -> {t_in_n:.4f})")

    # a WIDER ramp must not cost any texture outside the core - the property
    # the naive alpha blend failed, at 27.6% detail loss
    # A gaussian is not a perfect low-pass, so a very wide ramp leaks a trace of
    # high frequency. Assert it stays negligible against the texture it sits on,
    # rather than bit-exact: measured 0.25% at sigma 40, against the 27.6% loss
    # the naive alpha blend caused.
    wide = rc.composite_fill(orig, fill, core, ramp_sigma=40.0)
    leak = float((hp(wide)[far] - hp(orig)[far]).abs().max())
    scale = float(hp(orig)[far].std())
    check(leak < scale * 0.01,
          f"a wide lighting ramp leaks <1% of texture magnitude ({100*leak/scale:.3f}%)")
    e12 = float((rc.luminance(out)[:, 95:105] - rc.luminance(out)[:, 85:95]).abs().mean())
    e40 = float((rc.luminance(wide)[:, 95:105] - rc.luminance(wide)[:, 85:95]).abs().mean())
    check(e40 <= e12 + 1e-4, f"a wider ramp softens the step at least as much ({e12:.4f} -> {e40:.4f})")

    # A full core with a full ramp must reproduce the fill EXACTLY. This is what
    # catches an inconsistent frequency split: with two different sigmas the
    # mid-band is double-counted and this identity quietly fails.
    allc = torch.ones(H, W)
    exact = rc.composite_fill(orig, fill, allc, ramp_sigma=4.0)
    err = float((exact - fill).abs().max())
    check(err < 1e-4, f"a full core reproduces the fill exactly (max err {err:.2e})")

    # A NARROW core must still receive the full lighting correction. A plain
    # blurred ramp peaks well below 1 on a thin region, leaving it darker than
    # the fill it was handed.
    thin = torch.zeros(H, W)
    thin[:, 96:104] = 1.0                                   # 8px wide
    tout = rc.composite_fill(orig, fill, thin, ramp_sigma=20.0)
    want = rc.luminance(fill)[:, 98:102].mean()
    got = rc.luminance(tout)[:, 98:102].mean()
    check(abs(float(got - want)) < 0.02,
          f"a thin core still reaches the fill's brightness ({got:.3f} vs {want:.3f})")

    # an all-zero core must be a no-op
    noop = rc.composite_fill(orig, fill, torch.zeros(H, W), ramp_sigma=12.0)
    check(torch.allclose(noop, orig, atol=1e-5), "an empty core leaves the original untouched")


def test_contact_taper():
    """Contact occlusion near objects must be preserved, not filled."""
    print("contact taper")
    H, W = 160, 160
    obj = torch.zeros(H, W)
    obj[:, 100:] = 1.0                       # an object occupying the right half
    core = torch.ones(H, W)                  # naively fill everything

    t = rc.contact_taper(core, obj, taper_px=20)
    check(t[80, 10].item() > 0.95, "far from the object the fill is untouched")
    check(t[80, 95].item() < 0.5, f"right at the object the fill is suppressed ({t[80,95]:.2f})")
    check(bool((t <= core + 1e-6).all()), "tapering can only reduce a mask, never grow it")
    check(bool(((t >= 0) & (t <= 1)).all()), "output stays in [0,1]")

    # monotonic: further from the object means more fill
    prof = [t[80, x].item() for x in (60, 70, 80, 90)]
    check(all(prof[i] >= prof[i+1] - 1e-6 for i in range(len(prof)-1)),
          f"suppression increases monotonically toward the object ({[round(v,2) for v in prof]})")

    # a wider taper protects more
    t2 = rc.contact_taper(core, obj, taper_px=40)
    check(t2.sum().item() < t.sum().item(), "a wider taper protects a wider band")
    check(t2[80, 10].item() > 0.9, "a wider taper still leaves distant fill alone")

    # zero taper is a no-op, and no object means nothing to protect
    check(torch.equal(rc.contact_taper(core, obj, taper_px=0), core),
          "taper 0 is an exact no-op")
    none = rc.contact_taper(core, torch.zeros(H, W), taper_px=20)
    check(torch.allclose(none, core, atol=1e-6), "with no objects the mask is unchanged")


def main():
    for fn in [test_projection_geometry, test_sample_image, test_near_plane_cut,
               test_backproject_constant_recovery, test_backproject_weighting,
               test_backproject_occlusion, test_orbit_cameras, test_shadow_mask, test_flat_region_mask, test_split_fill_mask, test_soft_shadow_mask, test_composite_fill, test_contact_taper]:
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)
    print("all back-projection tests passed")


if __name__ == "__main__":
    main()
