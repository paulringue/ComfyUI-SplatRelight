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
    w = rc.view_weight(target, None, cam[0], cam[1], 60.0, 64, 64, grid=grid, sigma=40.0)
    w_nogrid = rc.view_weight(target, None, cam[0], cam[1], 60.0, 64, 64, grid=None)
    check(w_nogrid[0].item() > 0.5, "without a grid the hidden gaussian is naively weighted")
    check(w[0].item() < w_nogrid[0].item() * 0.5,
          f"the occluding slab cuts the weight ({w_nogrid[0]:.3f} -> {w[0]:.3f})")

    behind = ([0.0, 0.0, 4.0], [0.0, 0.0, 1.0])         # same side, nothing in the way
    w2 = rc.view_weight(target, None, behind[0], behind[1], 60.0, 64, 64, grid=grid, sigma=40.0)
    check(w2[0].item() > w[0].item(),
          f"an unobstructed camera keeps far more weight ({w[0]:.3f} vs {w2[0]:.3f})")


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


def main():
    for fn in [test_projection_geometry, test_sample_image, test_near_plane_cut,
               test_backproject_constant_recovery, test_backproject_weighting,
               test_backproject_occlusion, test_orbit_cameras]:
        fn()
    print(f"\n{PASS} passed, {FAIL} failed")
    if FAIL:
        raise SystemExit(1)
    print("all back-projection tests passed")


if __name__ == "__main__":
    main()
