"""Unit tests for the relighting math.

Run with:  python -m pytest tests/ -q     (or plain `python tests/test_relight_core.py`)

These build a synthetic splat whose ground truth we know - a sphere of flat,
outward-facing gaussians - so we can assert that normals, delighting, shadows
and shading all behave the way the physics says they should.
"""

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import relight_core as rc  # noqa: E402


def _quat_from_z_to(n):
    """Quaternion (wxyz) rotating +Z onto each unit vector n (N,3)."""
    z = torch.zeros_like(n)
    z[:, 2] = 1.0
    v = torch.cross(z, n, dim=-1)
    w = (1.0 + (z * n).sum(-1, keepdim=True))
    q = torch.cat([w, v], dim=-1)
    # antipodal case: rotate 180 deg about X
    deg = w.reshape(-1) < 1e-6
    if deg.any():
        q[deg] = torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=n.dtype)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def make_sphere(n=2000, radius=1.0, thin=0.01, wide=0.05, seed=0):
    """A hollow sphere of flat gaussians tangent to the surface."""
    g = torch.Generator().manual_seed(seed)
    i = torch.arange(n, dtype=torch.float64) + 0.5
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    y = 1.0 - 2.0 * i / n
    r = (1.0 - y * y).clamp_min(0).sqrt()
    dirs = torch.stack([r * torch.cos(phi), y, r * torch.sin(phi)], dim=-1).float()
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    positions = dirs * radius
    # thinnest axis (index 2) aligned with the true surface normal
    scales = torch.tensor([wide, wide, thin]).expand(n, 3).contiguous()
    rotations = _quat_from_z_to(dirs)
    opacities = torch.ones(n)
    return positions, scales, rotations, opacities, dirs


def check(cond, msg):
    if not cond:
        raise AssertionError(msg)
    print(f"  ok  {msg}")


def test_quat_and_normals():
    print("normals")
    pos, sc, rot, op, truth = make_sphere()
    n = rc.surfel_normals(sc, rot)
    # sign is unresolved, so compare absolute alignment
    align = (n * truth).sum(-1).abs()
    check(align.min() > 0.99, f"thinnest-axis normal matches truth (min |cos|={align.min():.4f})")

    oriented = rc.orient_normals(n, pos, mode="outward")
    signed = (oriented * truth).sum(-1)
    check(signed.min() > 0.99, f"outward orientation fixes the sign (min cos={signed.min():.4f})")

    # visibility-based orientation must also work on a hollow shell, where a
    # local density probe cannot distinguish the inside from the outside.
    grid = rc.DensityGrid(pos, op, sc, resolution=64)
    og = rc.orient_normals(n, pos, mode="visibility", grid=grid, sigma=4.0)
    frac = ((og * truth).sum(-1) > 0).float().mean()
    check(frac > 0.95, f"visibility orientation agrees {frac * 100:.1f}% of the time")


def test_anisotropy():
    print("anisotropy")
    flat = torch.tensor([[0.05, 0.05, 0.001]])
    blob = torch.tensor([[0.05, 0.05, 0.05]])
    check(rc.anisotropy(flat).item() > 0.97, "flat gaussian reports high anisotropy")
    check(rc.anisotropy(blob).item() < 0.01, "isotropic gaussian reports ~0 anisotropy")


def test_sh_roundtrip():
    print("spherical harmonics")
    rgb = torch.rand(500, 3)
    back = rc.dc_to_rgb(rc.rgb_to_dc(rgb))
    check(torch.allclose(rgb, back, atol=1e-5), "dc <-> rgb round-trips")

    # a constant environment must give constant irradiance equal to its radiance
    env = torch.ones(64, 128, 3) * 0.5
    coeff = rc.equirect_to_sh9(env)
    n = torch.nn.functional.normalize(torch.randn(200, 3), dim=-1)
    irr = rc.sh9_irradiance(coeff, n)
    check(irr.std() < 1e-3, f"uniform env -> direction-independent irradiance (std={irr.std():.2e})")
    check(abs(irr.mean().item() - 0.5) < 0.02,
          f"uniform env irradiance preserves albedo scale (got {irr.mean().item():.4f}, want 0.5)")


def test_env_directionality():
    print("environment directionality")
    # bright band only in the upper hemisphere -> up-facing normals get more light
    env = torch.zeros(64, 128, 3)
    env[:16] = 3.0                       # rows near theta=0 are "up" (y=+1)
    coeff = rc.equirect_to_sh9(env)
    up = torch.tensor([[0.0, 1.0, 0.0]])
    down = torch.tensor([[0.0, -1.0, 0.0]])
    i_up = rc.sh9_irradiance(coeff, up).mean()
    i_dn = rc.sh9_irradiance(coeff, down).mean()
    check(i_up > i_dn * 3, f"top-lit env lights up-normals far more ({i_up:.3f} vs {i_dn:.3f})")


def test_delight_recovers_albedo():
    print("delighting")
    pos, sc, rot, op, truth = make_sphere(n=4000)
    albedo = torch.full((truth.shape[0], 3), 0.55)
    light = torch.nn.functional.normalize(torch.tensor([[0.4, 1.0, 0.3]]), dim=-1)
    shading = (truth * light).sum(-1).clamp_min(0.0) + 0.25
    baked = (albedo * shading[:, None]).clamp(0, 1)

    spread_before = baked.std(0).mean()
    recovered, est = rc.delight(baked, truth, amount=1.0)
    spread_after = recovered.std(0).mean()
    check(spread_after < spread_before * 0.5,
          f"delighting flattens baked shading ({spread_before:.4f} -> {spread_after:.4f})")

    corr = torch.corrcoef(torch.stack([est, shading]))[0, 1]
    check(corr > 0.95, f"estimated shading correlates with truth (r={corr:.4f})")

    half = rc.delight(baked, truth, amount=0.0)[0]
    check(torch.allclose(half, baked, atol=1e-5), "amount=0 is a no-op")


def test_shadow_and_ao():
    print("occlusion")
    pos, sc, rot, op, truth = make_sphere(n=6000)
    grid = rc.DensityGrid(pos, op, sc, resolution=96)

    # a ray from the sphere's centre outward must be blocked by the shell
    centre = torch.zeros(1, 3)
    d = torch.tensor([[0.0, 1.0, 0.0]])
    t_out = grid.transmittance(centre, d, max_dist=2.0, steps=64, sigma=5.0)
    # a ray in empty space far outside must not be
    far = torch.tensor([[5.0, 5.0, 5.0]])
    t_free = grid.transmittance(far, d, max_dist=2.0, steps=64, sigma=5.0)
    check(t_out.item() < t_free.item(),
          f"shell occludes an interior ray ({t_out.item():.4f} < {t_free.item():.4f})")
    check(t_free.item() > 0.99, "empty space is fully transmissive")

    ao = rc.ambient_occlusion(grid, pos, truth, radius=0.5, n_dirs=6, steps=8, sigma=2.0)
    check(ao.shape == (pos.shape[0],), "AO returns one value per gaussian")
    check(float(ao.min()) >= 0.0 and float(ao.max()) <= 1.0, "AO stays within [0,1]")

    inward = rc.ambient_occlusion(grid, pos, -truth, radius=0.5, n_dirs=6, steps=8, sigma=2.0)
    check(inward.mean() < ao.mean(),
          f"inward-facing normals are more occluded ({inward.mean():.4f} < {ao.mean():.4f})")


def test_shading_physics():
    print("shading")
    n = torch.tensor([[0.0, 1.0, 0.0]])
    p = torch.zeros(1, 3)
    v = torch.tensor([[0.0, 1.0, 0.0]])
    albedo = torch.full((1, 3), 0.8)

    top = [{"type": "directional", "direction": [0.0, -1.0, 0.0], "color": [1.0] * 3,
            "intensity": math.pi, "cast_shadow": False}]
    lit = rc.shade(albedo, n, p, top, v, roughness=1.0, metallic=0.0, specular_strength=0.0)
    check(torch.allclose(lit, albedo, atol=1e-3),
          f"head-on lambert returns albedo at intensity pi (got {lit[0,0]:.4f})")

    back = [{"type": "directional", "direction": [0.0, 1.0, 0.0], "color": [1.0] * 3,
             "intensity": math.pi, "cast_shadow": False}]
    dark = rc.shade(albedo, n, p, back, v, roughness=1.0, metallic=0.0, specular_strength=0.0)
    check(float(dark.max()) < 1e-6, "back-facing light contributes nothing")

    # inverse-square falloff for point lights
    near = [{"type": "point", "position": [0.0, 1.0, 0.0], "color": [1.0] * 3,
             "intensity": 1.0, "radius": 0.0, "cast_shadow": False}]
    far = [{"type": "point", "position": [0.0, 2.0, 0.0], "color": [1.0] * 3,
            "intensity": 1.0, "radius": 0.0, "cast_shadow": False}]
    a = rc.shade(albedo, n, p, near, v, 1.0, 0.0, specular_strength=0.0).mean()
    b = rc.shade(albedo, n, p, far, v, 1.0, 0.0, specular_strength=0.0).mean()
    check(abs((a / b).item() - 4.0) < 0.05, f"point light falls off as 1/d^2 (ratio={a/b:.3f})")

    # metals kill diffuse response
    met = rc.shade(albedo, n, p, top, v, roughness=1.0, metallic=1.0, specular_strength=0.0)
    check(float(met.max()) < 1e-6, "fully metallic surface has no diffuse term")

    # coloured lights tint the result
    red = [{"type": "directional", "direction": [0.0, -1.0, 0.0], "color": [1.0, 0.0, 0.0],
            "intensity": math.pi, "cast_shadow": False}]
    r = rc.shade(albedo, n, p, red, v, 1.0, 0.0, specular_strength=0.0)
    check(r[0, 0] > 0.5 and r[0, 1] < 1e-6, "red light produces a red-only response")


def test_spot_cone():
    print("spot light")
    n = torch.tensor([[0.0, 1.0, 0.0]])
    p = torch.zeros(1, 3)
    v = torch.tensor([[0.0, 1.0, 0.0]])
    albedo = torch.full((1, 3), 0.8)
    lt = {"type": "spot", "position": [0.0, 2.0, 0.0], "direction": [0.0, -1.0, 0.0],
          "color": [1.0] * 3, "intensity": 10.0, "radius": 0.0,
          "cone_angle": 30.0, "cone_softness": 5.0, "cast_shadow": False}
    inside = rc.shade(albedo, n, p, [lt], v, 1.0, 0.0, specular_strength=0.0).mean()
    off = dict(lt, position=[3.0, 2.0, 0.0])
    outside = rc.shade(albedo, n, p, [off], v, 1.0, 0.0, specular_strength=0.0).mean()
    check(inside > 0, "surface inside the cone is lit")
    check(float(outside) < 1e-6, "surface outside the cone is dark")


def test_specular_and_tonemap():
    print("specular / tonemap")
    n = torch.tensor([[0.0, 1.0, 0.0]])
    p = torch.zeros(1, 3)
    v = torch.tensor([[0.0, 1.0, 0.0]])
    albedo = torch.full((1, 3), 0.5)
    lt = [{"type": "directional", "direction": [0.0, -1.0, 0.0], "color": [1.0] * 3,
           "intensity": math.pi, "cast_shadow": False}]
    sharp = rc.shade(albedo, n, p, lt, v, roughness=0.05, metallic=0.0, specular_strength=1.0)
    dull = rc.shade(albedo, n, p, lt, v, roughness=0.9, metallic=0.0, specular_strength=1.0)
    check(sharp.max() > dull.max(), "low roughness gives a stronger mirror highlight")

    hdr = torch.tensor([[0.0, 1.0, 40.0]])
    for mode in ("aces", "reinhard", "none"):
        out = rc.tonemap(hdr, 1.0, mode)
        check(float(out.max()) <= 1.0 and float(out.min()) >= 0.0, f"tonemap '{mode}' stays in gamut")
    check(rc.tonemap(hdr, 1.0, "aces")[0, 2] > rc.tonemap(hdr, 1.0, "aces")[0, 1],
          "tonemap preserves ordering")

    # shape regression: with N=1 a mis-broadcast specular term collapses back to
    # the right shape by accident, so exercise many gaussians at once.
    pos, sc, rot, op, truth = make_sphere(n=512)
    alb = torch.full((512, 3), 0.5)
    many = rc.shade(alb, truth, pos, lt, truth, roughness=0.3, metallic=0.2,
                    specular_strength=1.0)
    check(many.shape == (512, 3), f"specular keeps (N,3) for N>1 (got {tuple(many.shape)})")
    check(torch.isfinite(many).all(), "batched specular stays finite")


def test_normal_smoothing():
    print("normal smoothing")
    # a flat wall whose per-gaussian normals have been badly corrupted, which is
    # what a real capture's round gaussians actually look like
    g = torch.Generator().manual_seed(7)
    n_pts = 20000
    xy = torch.rand(n_pts, 2, generator=g) * 2.0 - 1.0
    pos = torch.stack([xy[:, 0], xy[:, 1], torch.zeros(n_pts)], -1)
    truth = torch.zeros(n_pts, 3); truth[:, 2] = 1.0
    noisy = truth + torch.randn(n_pts, 3, generator=g) * 0.75
    noisy = noisy / noisy.norm(dim=-1, keepdim=True)
    noisy = noisy * torch.sign((noisy * truth).sum(-1, keepdim=True))  # same hemisphere

    sc = torch.full((n_pts, 3), 0.01)
    grid = rc.DensityGrid(pos, torch.ones(n_pts), sc, resolution=48)
    before = (noisy * truth).sum(-1).mean()
    sm = rc.smooth_normals(grid, pos, noisy, radius=1, strength=1.0)
    after = (sm * truth).sum(-1).mean()
    check(after > before + 0.15,
          f"smoothing recovers the plane (mean cos {before:.3f} -> {after:.3f})")
    check(torch.allclose(sm.norm(dim=-1), torch.ones(n_pts), atol=1e-4),
          "smoothed normals stay unit length")

    same = rc.smooth_normals(grid, pos, noisy, radius=1, strength=0.0)
    check(torch.equal(same, noisy), "strength=0 is a no-op")

    # must not destroy genuine curvature: a sphere stays recognisably a sphere
    spos, ssc, srot, sop, struth = make_sphere(n=8000)
    sgrid = rc.DensityGrid(spos, sop, ssc, resolution=48)
    ssm = rc.smooth_normals(sgrid, spos, struth, radius=1, strength=1.0)
    check((ssm * struth).sum(-1).mean() > 0.95,
          f"curvature survives smoothing (mean cos {(ssm*struth).sum(-1).mean():.3f})")


def test_frame_conversion():
    print("coordinate frames")
    check(rc.viewer_to_splat([1.0, 2.0, 3.0]) == [1.0, -2.0, -3.0],
          "viewer->splat negates Y and Z")
    v = torch.randn(50, 3)
    check(torch.allclose(rc.flip_frame(rc.flip_frame(v)), v), "the frame flip is its own inverse")

    # A light authored at the camera's own yaw/pitch must land on the surfaces
    # facing that camera. This is the bug that a purely numeric test misses:
    # every term stays finite and in range while the key light lights the back.
    for yaw, pitch in [(0.0, 0.0), (35.0, 15.0), (-70.0, 45.0), (150.0, -20.0)]:
        y, p = math.radians(yaw), math.radians(pitch)
        # where Create Camera Info puts a camera, in viewer space
        cam_view = torch.tensor([math.cos(p) * math.sin(y), math.sin(p), math.cos(p) * math.cos(y)])
        cam_splat = torch.tensor(rc.viewer_to_splat(cam_view.tolist()))
        # the surfel that faces that camera
        n = (cam_splat / cam_splat.norm())[None, :]

        travel = _yaw_pitch_to_dir_ref(yaw, pitch)
        l = -torch.tensor(rc.viewer_to_splat(travel))[None, :]
        n_dot_l = float((n * l).sum())
        check(n_dot_l > 0.999,
              f"light at yaw={yaw:.0f} pitch={pitch:.0f} lights the matching camera view (n.l={n_dot_l:.4f})")


def _yaw_pitch_to_dir_ref(yaw_deg, pitch_deg):
    """Mirror of nodes._yaw_pitch_to_dir, kept here so the core tests stay
    importable without ComfyUI on the path."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    return [-math.cos(p) * math.sin(y), -math.sin(p), -math.cos(p) * math.cos(y)]


def test_light_from_camera():
    print("light from a camera pose")
    # a camera directly above, looking down -> the light travels straight down
    d, p = rc.light_from_camera([0, 5, 0], [0, 0, 0])
    check(abs(d[1] + 1) < 1e-9 and abs(d[0]) < 1e-9 and abs(d[2]) < 1e-9,
          f"camera above gives a downward light ({[round(v,3) for v in d]})")
    check(all(abs(p[i] - [0, 5, 0][i]) < 1e-9 for i in range(3)),
          "the light lands at the camera position")

    # placing a camera where yaw/pitch would put it must give the same direction
    for yaw, pitch in [(0.0, 0.0), (35.0, 30.0), (-120.0, 55.0), (150.0, -20.0)]:
        y, pch = math.radians(yaw), math.radians(pitch)
        cam = [math.cos(pch) * math.sin(y), math.sin(pch), math.cos(pch) * math.cos(y)]
        got, _ = rc.light_from_camera(cam, [0, 0, 0])
        want = _yaw_pitch_to_dir_ref(yaw, pitch)
        diff = max(abs(got[i] - want[i]) for i in range(3))
        check(diff < 1e-9,
              f"camera at yaw={yaw:.0f} pitch={pitch:.0f} matches the yaw/pitch light ({diff:.1e})")

    _, half = rc.light_from_camera([0, 4, 0], [0, 0, 0], distance_scale=0.5)
    check(abs(half[1] - 2.0) < 1e-9, f"distance_scale halves the distance (y={half[1]:.2f})")

    try:
        rc.light_from_camera([1, 1, 1], [1, 1, 1])
        check(False, "should reject a camera sitting on its target")
    except ValueError:
        check(True, "rejects a camera sitting on its own target")


def test_camera_guard():
    print("camera framing guard")
    pos, sc, rot, op, truth = make_sphere(n=4000, radius=1.85)     # a room-sized shell
    centre, radius = rc.scene_bounds(pos)
    check(all(abs(c) < 0.05 for c in centre), f"centre found at origin ({[round(c,3) for c in centre]})")
    check(abs(radius - 1.85) < 0.08, f"radius recovered (~1.85, got {radius:.2f})")

    # framing must dominate bare clearance for a normal lens, or the camera ends
    # up skimming the walls of a room capture
    need = rc.frame_distance(radius, 40.0, margin=1.25)
    check(need > radius * 1.25 + 1e-6,
          f"a 40 deg view needs more than clearance ({need:.2f} vs {radius*1.25:.2f})")
    check(abs(need - radius / math.tan(math.radians(20.0))) < 1e-6,
          "framing distance is radius / tan(fov/2)")
    check(rc.frame_distance(radius, 120.0, margin=2.0) == radius * 2.0,
          "a very wide lens falls back to the clearance floor")

    # a camera inside the shell must be pushed out past it. Distances are from
    # the recovered centre, which is not exactly the origin.
    orig = [0.5, 0.2, 0.3]
    frm = lambda v: math.sqrt(sum((v[i] - centre[i]) ** 2 for i in range(3)))
    inside, moved = rc.push_camera_outside(orig, centre, need)
    d_in = frm(inside)
    check(moved and abs(d_in - need) < 1e-6, f"inside camera pushed out to {d_in:.3f} (want {need:.3f})")

    # direction must be preserved - only the distance changes
    a = [inside[i] - centre[i] for i in range(3)]
    b = [orig[i] - centre[i] for i in range(3)]
    nb = math.sqrt(sum(c * c for c in b))
    cos = sum(a[i] * b[i] for i in range(3)) / (d_in * nb)
    check(cos > 0.9999, f"view direction preserved (cos={cos:.6f})")

    # a camera already far enough must be left alone
    out, moved2 = rc.push_camera_outside([0, 0, 12.0], centre, need)
    check(not moved2 and abs(out[2] - 12.0) < 1e-9, "camera already far enough is untouched")

    # degenerate: camera exactly at the centre still yields something renderable
    deg, _ = rc.push_camera_outside([0, 0, 0], centre, need)
    check(math.sqrt(sum(v * v for v in deg)) > radius, "camera at the centre is still pushed out")

    # and the quaternion we emit must match ComfyUI's, or turntables break
    q = rc.lookat_quat_xyzw([3, 3, 3], [0, 0, 0])
    check(abs(sum(c * c for c in q) - 1.0) < 1e-6, "look-at quaternion is normalised")
    qd = rc.lookat_quat_xyzw([0, 5, 0], [0, 0, 0])
    check(all(math.isfinite(c) for c in qd), "straight-down look-at is finite (degenerate up vector)")


def test_crop_and_enclosing():
    print("crop / enclosing radius")
    # a room-shaped shell: 3 wide, 3 tall, 4 deep. The long axis is what you end
    # up inside if clearance is measured with the quantile radius.
    g = torch.Generator().manual_seed(5)
    p = torch.randn(30000, 3, generator=g)
    p = p / p.norm(dim=-1, keepdim=True) * torch.tensor([1.52, 1.56, 2.0])
    centre, radius = rc.scene_bounds(p)
    enc = rc.enclosing_radius(p, centre)
    check(enc > radius,
          f"enclosing radius exceeds the quantile one ({enc:.3f} > {radius:.3f})")
    check(abs(enc - 2.0) < 0.12, f"enclosing radius finds the long axis (~2.0, got {enc:.3f})")

    # crop to a corner box
    m = rc.crop_mask(p, [1.0, 0.0, 0.0], [0.7, 0.7, 0.7], "box")
    kept = p[m]
    check(0 < kept.shape[0] < p.shape[0], f"box keeps a subset ({kept.shape[0]} of {p.shape[0]})")
    check(bool(((kept - torch.tensor([1.0, 0.0, 0.0])).abs() <= 0.7 + 1e-6).all()),
          "everything kept really is inside the box")
    _, r2 = rc.scene_bounds(kept)
    check(r2 < radius, f"cropping shrinks the shell you must clear ({r2:.3f} < {radius:.3f})")

    s = rc.crop_mask(p, [0.0, 0.0, 0.0], [1.6], "sphere")
    d = (p[s]).norm(dim=-1)
    check(bool((d <= 1.6 + 1e-6).all()), "sphere crop respects its radius")
    check(int(m.sum()) + int((~m).sum()) == p.shape[0], "invert is the exact complement")


def test_env_frame():
    print("environment frame")
    # a bright top band must light viewer-up, which is splat -Y
    env = torch.zeros(64, 128, 3)
    env[:16] = 3.0
    coeff = rc.equirect_to_sh9(env, to_splat_frame=True)
    splat_up = torch.tensor([[0.0, -1.0, 0.0]])     # viewer +Y in splat space
    splat_dn = torch.tensor([[0.0, 1.0, 0.0]])
    check(rc.sh9_irradiance(coeff, splat_up).mean() > rc.sh9_irradiance(coeff, splat_dn).mean() * 3,
          "splat-frame env lights the viewer-up direction")


def test_variable_batch_safety():
    print("robustness")
    # degenerate input: all gaussians at one point must not produce NaNs
    pos = torch.zeros(64, 3)
    sc = torch.full((64, 3), 0.01)
    rot = torch.zeros(64, 4)
    rot[:, 0] = 1.0
    op = torch.ones(64)
    nrm = rc.surfel_normals(sc, rot)
    check(torch.isfinite(nrm).all(), "degenerate isotropic gaussians yield finite normals")
    grid = rc.DensityGrid(pos, op, sc, resolution=32)
    check(torch.isfinite(grid.grid).all(), "degenerate splat yields a finite density grid")
    alb, sh = rc.delight(torch.rand(64, 3), nrm, amount=1.0)
    check(torch.isfinite(alb).all(), "delighting stays finite on degenerate normals")


def test_shadow_lift():
    """Tonal lift must hit its stated contrast law and preserve mean brightness."""
    torch.manual_seed(7)
    # two populations of identical albedo, one darkened 5x as if in shadow
    base = torch.full((512, 3), 0.6)
    rgb = base.clone()
    rgb[:256] *= 0.2                                   # 5x contrast, purely spatial
    lo = lambda c: rc.luminance(c)[:256].mean()
    hi = lambda c: rc.luminance(c)[256:].mean()

    r0 = (hi(rgb) / lo(rgb)).item()
    check(abs(r0 - 5.0) < 1e-3, f"synthetic contrast is 5x (got {r0:.3f})")

    check(torch.equal(rc.lift_shadows(rgb, 0.0), rgb), "shadow_lift 0 is exactly a no-op")

    for k in (0.25, 0.5, 0.75):
        out = rc.lift_shadows(rgb, k)
        r = (hi(out) / lo(out)).item()
        want = r0 ** (1.0 - k)
        check(abs(r - want) / want < 0.02,
              f"lift {k}: contrast {r:.3f} matches predicted r^(1-k) = {want:.3f}")

    flat = rc.lift_shadows(rgb, 1.0)
    r = (hi(flat) / lo(flat)).item()
    check(abs(r - 1.0) < 0.02, f"lift 1.0 flattens luminance entirely (got {r:.3f})")

    # mean luminance must not drift - this compresses, it does not brighten
    m0 = rc.luminance(rgb).mean()
    m1 = rc.luminance(rc.lift_shadows(rgb, 0.5)).mean()
    check(abs((m1 - m0) / m0) < 0.15, f"mean luminance preserved ({m0:.3f} -> {m1:.3f})")

    # and it must be reachable through delight() without disturbing amount=0
    nrm = torch.nn.functional.normalize(torch.randn(512, 3), dim=-1)
    a0, _ = rc.delight(rgb, nrm, amount=0.0, shadow_lift=0.0)
    check(torch.allclose(a0, rgb, atol=1e-6), "delight(amount=0, lift=0) returns the input")
    a1, _ = rc.delight(rgb, nrm, amount=0.0, shadow_lift=0.6)
    r1 = (hi(a1) / lo(a1)).item()
    check(r1 < r0 * 0.6, f"delight passes shadow_lift through ({r0:.2f}x -> {r1:.2f}x)")
    check(torch.isfinite(a1).all(), "lifted albedo stays finite")

    # pure black must not explode
    blk = torch.zeros(16, 3)
    check(torch.isfinite(rc.lift_shadows(blk, 1.0)).all(), "all-black input stays finite")


def main():
    for fn in [test_quat_and_normals, test_anisotropy, test_sh_roundtrip, test_env_directionality,
               test_delight_recovers_albedo, test_shadow_and_ao, test_shading_physics,
               test_spot_cone, test_specular_and_tonemap, test_normal_smoothing, test_frame_conversion,
               test_light_from_camera, test_camera_guard, test_crop_and_enclosing, test_env_frame,
               test_variable_batch_safety, test_shadow_lift]:
        fn()
    print("\nall relight_core tests passed")


if __name__ == "__main__":
    main()
