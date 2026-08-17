"""Pure-torch relighting math for 3D gaussian splats.

Deliberately free of any ComfyUI imports so it can be unit-tested standalone.
Everything here works on flat per-gaussian tensors (N, ...) for a single frame;
the node layer in nodes.py handles batching over an animated sequence.

Conventions match comfy_extras/nodes_gaussian_splat.py:
  * positions (N,3) world space, right-handed, Y-up
  * scales    (N,3) linear per-axis std (already activated)
  * rotations (N,4) quaternion wxyz, normalized
  * opacities (N,) in [0,1]
  * sh        (N,K,3) spherical harmonics, K=(deg+1)^2, DC term is sh[:,0,:]
"""

import math

import torch

# SH band-0 constant: rgb = dc * C0 + 0.5  (same constant ComfyUI uses)
C0 = 0.28209479177387814

# ComfyUI has two coordinate frames in play, and lighting has to bridge them.
# Load3D / camera_info / CreateCameraInfo are three.js: right-handed, Y-up.
# The gaussian data itself is 3DGS: Y-down, Z-forward. The two differ by a
# 180-degree rotation about X, i.e. negating Y and Z. ComfyUI's own rasteriser
# documents this (comfy_extras/nodes_gaussian_splat.py) and applies the same
# [1,-1,-1] flip when it exports normals and meshes.
#
# Everything in this module works in *splat* space. Lights are authored in
# viewer space - so that "yaw 35" puts the light where a camera at yaw 35 sits -
# and converted on the way in.
FRAME_FLIP = (1.0, -1.0, -1.0)


def viewer_to_splat(v):
    """Convert a viewer-space (Y-up) vector to splat space (Y-down)."""
    return [float(v[0]) * FRAME_FLIP[0], float(v[1]) * FRAME_FLIP[1], float(v[2]) * FRAME_FLIP[2]]


def scene_bounds(positions: torch.Tensor, q: float = 0.02):
    """Robust centre and radius of a splat, ignoring floaters via quantiles."""
    p = positions if positions.shape[0] < 200000 else positions[::53]
    lo = torch.quantile(p.float(), q, dim=0)
    hi = torch.quantile(p.float(), 1.0 - q, dim=0)
    centre = ((lo + hi) * 0.5).tolist()
    radius = float((hi - lo).max().item()) * 0.5
    return centre, max(radius, 1e-4)


def crop_mask(positions: torch.Tensor, centre, half, mode: str = "box") -> torch.Tensor:
    """Which gaussians fall inside a box or sphere. positions (N,3) -> (N,) bool.

    `centre` and `half` are in the same frame as `positions`. For a sphere only
    half[0] is used as the radius.
    """
    c = torch.as_tensor(centre, device=positions.device, dtype=positions.dtype)
    d = positions - c
    if mode == "sphere":
        return d.norm(dim=-1) <= float(half[0])
    h = torch.as_tensor(half, device=positions.device, dtype=positions.dtype).abs()
    return (d.abs() <= h).all(dim=-1)


def near_plane_mask(positions: torch.Tensor, cam_pos, cam_target, offset: float = 0.05
                    ) -> torch.Tensor:
    """Keep only gaussians in front of a plane just ahead of the camera. -> (N,) bool.

    A camera inside a room is enclosed by the very surfaces it is looking
    through, and the rasteriser returns an empty or wall-filled frame. Cutting
    a plane perpendicular to the view direction removes the near wall and
    nothing else. An axis-aligned box cannot do this: it leaves fragments
    wherever a wall runs oblique to the axes, which is most real captures.

    Everything is in the splat frame. `offset` pushes the plane slightly ahead
    of the camera so gaussians straddling it do not clip into the lens.
    """
    p = torch.as_tensor(cam_pos, device=positions.device, dtype=positions.dtype)
    t = torch.as_tensor(cam_target, device=positions.device, dtype=positions.dtype)
    d = t - p
    d = d / d.norm().clamp_min(1e-8)
    return (positions - (p + d * float(offset))) @ d > 0


def enclosing_radius(positions: torch.Tensor, centre, q: float = 0.995):
    """Distance from `centre` that actually contains the splat.

    The quantile radius from scene_bounds is a *typical* half-extent and is the
    right thing for framing, but it under-reports the long axis - a room 4 deep
    and 3 wide reports ~1.85 when its far wall is at 2.0. Using it for clearance
    leaves the camera inside the geometry, where ComfyUI's rasteriser renders
    nothing at all. This measures the real thing, still trimming floaters.
    """
    p = positions if positions.shape[0] < 200000 else positions[::53]
    c = torch.tensor(centre, device=p.device, dtype=p.dtype)
    d = (p.float() - c).norm(dim=-1)
    return float(_quantile_1d(d, q))


def _quantile_1d(x: torch.Tensor, q: float) -> float:
    lim = 1 << 24
    if x.numel() > lim:
        x = x[:: x.numel() // lim + 1]
    return float(torch.quantile(x, q).item())


def frame_distance(radius, fov_deg, margin=1.25):
    """How far a camera must sit to both clear the scene and actually frame it.

    Two separate requirements, and the larger wins:
      * clearance - ComfyUI's rasteriser returns an empty frame when the camera
        is inside the geometry, so we must be past `radius * margin`;
      * framing   - to fit a sphere of `radius` in a `fov_deg` view you need
        radius / tan(fov/2), which for a normal lens is a good deal further out.
    Clearance alone leaves the camera skimming the walls of a room capture.
    """
    half = math.radians(max(float(fov_deg), 1.0) * 0.5)
    fit = float(radius) / max(math.tan(half), 1e-6)
    return max(float(radius) * float(margin), fit)


def push_camera_outside(cam_pos, centre, min_distance):
    """Move a camera out along its own view axis to at least `min_distance`.

    Keeps the direction the user chose - only the distance changes - and leaves
    a camera that is already far enough completely alone.

    Returns (position, moved).
    """
    d = [float(cam_pos[i]) - float(centre[i]) for i in range(3)]
    ln = math.sqrt(sum(c * c for c in d))
    if ln < 1e-9:
        d, ln = [0.0, 0.0, 1.0], 1.0
    need = float(min_distance)
    if ln >= need:
        return [float(v) for v in cam_pos], False
    return [centre[i] + d[i] / ln * need for i in range(3)], True


def lookat_quat_xyzw(position, target):
    """three.js look-at rotation for a camera, returned as (x, y, z, w).

    A camera_info without a quaternion is a trap: ComfyUI's turntable helper
    defaults a missing one to *identity* and then writes it into the camera it
    returns, so the renderer takes the quaternion path with a rotation that has
    nothing to do with the look-at. Every frame but one comes out empty. Any
    camera_info we emit therefore carries a real quaternion.

    Mirrors comfy_extras/nodes_gaussian_splat.py: camera local +Z = (eye - target),
    up = world +Y.
    """
    z = [float(position[i]) - float(target[i]) for i in range(3)]
    n = math.sqrt(sum(c * c for c in z)) or 1.0
    z = [c / n for c in z]
    up = [0.0, 1.0, 0.0]
    if abs(z[1]) > 0.999:                       # looking straight up or down
        up = [0.0, 0.0, 1.0]
    cross = lambda a, b: [a[1] * b[2] - a[2] * b[1],
                          a[2] * b[0] - a[0] * b[2],
                          a[0] * b[1] - a[1] * b[0]]
    x = cross(up, z)
    nx = math.sqrt(sum(c * c for c in x)) or 1.0
    x = [c / nx for c in x]
    y = cross(z, x)

    # rotation matrix with columns x, y, z -> quaternion
    m = [[x[0], y[0], z[0]], [x[1], y[1], z[1]], [x[2], y[2], z[2]]]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        qw, qx = 0.25 * s, (m[2][1] - m[1][2]) / s
        qy, qz = (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        qw, qx = (m[2][1] - m[1][2]) / s, 0.25 * s
        qy, qz = (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        qw, qx = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s
        qy, qz = 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        qw, qx = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s
        qy, qz = (m[1][2] + m[2][1]) / s, 0.25 * s
    ln = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) or 1.0
    return [qx / ln, qy / ln, qz / ln, qw / ln]


def light_from_camera(position, target, distance_scale=1.0):
    """A camera pose -> (travel direction, light position), both viewer space.

    The light is placed where the camera is and aimed at what it looks at, which
    is what makes an interactive viewport usable as a lighting gizmo: orbit until
    the view looks like where you want the light, and that *is* the light.

    Raises ValueError if the camera sits on its own target, where there is no
    direction to take.
    """
    d = [float(target[i]) - float(position[i]) for i in range(3)]
    ln = math.sqrt(sum(c * c for c in d))
    if ln < 1e-9:
        raise ValueError("camera position and target coincide, so there is no view direction")
    direction = [c / ln for c in d]
    s = float(distance_scale)
    light_pos = [float(target[i]) - direction[i] * ln * s for i in range(3)]
    return direction, light_pos


def flip_frame(t: torch.Tensor) -> torch.Tensor:
    """Same conversion for an (...,3) tensor. It is its own inverse."""
    f = torch.tensor(FRAME_FLIP, device=t.device, dtype=t.dtype)
    return t * f

# Ramamoorthi & Hanrahan irradiance convolution constants for order-2 SH.
_A0, _A1, _A2 = 3.141593, 2.094395, 0.785398


# --------------------------------------------------------------------------
# basic conversions
# --------------------------------------------------------------------------

def quat_to_mat(q: torch.Tensor) -> torch.Tensor:
    """(N,4) wxyz -> (N,3,3) rotation matrices. Mirrors ComfyUI's _quat_to_mat."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    return torch.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], dim=-1).reshape(-1, 3, 3)


def dc_to_rgb(dc: torch.Tensor) -> torch.Tensor:
    """SH DC coefficients (N,3) -> linear-ish base colour in [0,1]."""
    return (dc * C0 + 0.5).clamp(0.0, 1.0)


def rgb_to_dc(rgb: torch.Tensor) -> torch.Tensor:
    """Inverse of dc_to_rgb; no clamping so HDR highlights survive to export."""
    return (rgb - 0.5) / C0


def srgb_to_linear(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.04045, c / 12.92, ((c.clamp_min(0) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: torch.Tensor) -> torch.Tensor:
    return torch.where(c <= 0.0031308, c * 12.92, 1.055 * c.clamp_min(0) ** (1 / 2.4) - 0.055)


def luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Rec.709 luma, (N,3) -> (N,)."""
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=rgb.device, dtype=rgb.dtype)
    return (rgb * w).sum(-1)


# --------------------------------------------------------------------------
# surfel normals
# --------------------------------------------------------------------------

def surfel_normals(scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
    """Per-gaussian surface normal = the thinnest principal axis.

    A well-converged 3DGS reconstruction produces strongly oblate ("flat")
    gaussians that lie tangent to the surface, so the shortest axis of the
    covariance is a good surface normal. This is the same estimator ComfyUI's
    own rasteriser uses for its headlight/clay shading.

    Returns (N,3) unit normals with an arbitrary (unresolved) sign.
    """
    R = quat_to_mat(rotations)                      # (N,3,3)
    thin = scales.argmin(dim=-1)                    # (N,) index of smallest std
    idx = torch.arange(R.shape[0], device=R.device)
    n = R[idx, :, thin]                             # column `thin` = that axis
    return n / n.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def anisotropy(scales: torch.Tensor) -> torch.Tensor:
    """How flat each gaussian is, in [0,1]. 0 = isotropic blob (normal is
    meaningless), 1 = perfectly flat disc (normal is trustworthy)."""
    s = scales.clamp_min(1e-12)
    smin = s.min(dim=-1).values
    smax = s.max(dim=-1).values
    return (1.0 - smin / smax.clamp_min(1e-12)).clamp(0.0, 1.0)


def orient_normals(normals, positions, mode="visibility", grid=None, steps=16, sigma=1.0):
    """Resolve the sign ambiguity of surfel normals so they point *outward*.

    visibility : march the occupancy grid along +n and -n and keep whichever
                 direction escapes the scene more easily. This is the reliable
                 choice: a 3DGS reconstruction is a hollow *shell*, so a purely
                 local density probe sees empty space on both sides of the
                 surface and cannot tell inside from outside. Integrating over a
                 long ray can - the inward ray eventually hits the far wall.
    outward    : flip so the normal points away from the scene centroid. Cheap,
                 and correct for star-shaped (convex-ish) object captures, but
                 wrong on interiors and concavities.
    keep       : leave the raw sign alone.
    """
    if mode == "keep":
        return normals
    if mode == "outward" or grid is None:
        centre = positions.mean(0, keepdim=True)
        sign = torch.sign(((positions - centre) * normals).sum(-1, keepdim=True))
        sign = torch.where(sign == 0, torch.ones_like(sign), sign)
        return normals * sign
    # visibility: whichever side is less occluded is the outside.
    bias = 1.5 * grid.voxel_size
    reach = float(grid.extent.max().item()) * 0.5
    t_pos = grid.transmittance(positions + normals * bias, normals, reach, steps=steps, sigma=sigma)
    t_neg = grid.transmittance(positions - normals * bias, -normals, reach, steps=steps, sigma=sigma)
    sign = torch.where((t_neg > t_pos).unsqueeze(-1), -1.0, 1.0)
    return normals * sign


# --------------------------------------------------------------------------
# occupancy grid: shared machinery for shadows and ambient occlusion
# --------------------------------------------------------------------------

class DensityGrid:
    """Coarse voxel occupancy built from the splat, used to ray-march visibility.

    Gaussians are accumulated into a dense grid weighted by opacity and by their
    footprint, giving an optical-density field we can integrate along a ray.
    Cheap and fully vectorised - good enough for soft shadows and AO, which is
    all we need since the splat itself is the high-frequency detail.
    """

    def __init__(self, positions, opacities, scales, resolution=128, padding=0.05):
        dev, dt = positions.device, positions.dtype
        lo = positions.min(0).values
        hi = positions.max(0).values
        span = (hi - lo).clamp_min(1e-6)
        lo = lo - span * padding
        hi = hi + span * padding
        self.lo, self.hi = lo, hi
        self.res = int(resolution)
        self.extent = (hi - lo).clamp_min(1e-6)
        self.voxel_size = float(self.extent.max().item()) / self.res

        # normalised [0, res) voxel coordinates
        vc = (positions - lo) / self.extent * self.res
        vi = vc.long().clamp(0, self.res - 1)
        flat = (vi[:, 0] * self.res + vi[:, 1]) * self.res + vi[:, 2]

        # weight by opacity and by how much of a voxel the gaussian covers
        vol = (scales.mean(-1) / max(self.voxel_size, 1e-9)).clamp(0.05, 4.0)
        w = (opacities.reshape(-1) * vol).to(dt)

        grid = torch.zeros(self.res ** 3, device=dev, dtype=dt)
        grid.index_add_(0, flat, w)
        self.grid = grid.reshape(1, 1, self.res, self.res, self.res)

    def _coords(self, pts: torch.Tensor) -> torch.Tensor:
        g = (pts - self.lo) / self.extent          # [0,1]
        g = g * 2.0 - 1.0                          # [-1,1]
        # grid_sample expects (x,y,z) ordered as the *last* dim indexes width;
        # our grid is [D=x, H=y, W=z], so flip to (z,y,x).
        return g.flip(-1).reshape(1, 1, 1, -1, 3)

    def sample(self, pts: torch.Tensor) -> torch.Tensor:
        """Trilinearly sample density at world points (M,3) -> (M,)."""
        out = torch.nn.functional.grid_sample(
            self.grid, self._coords(pts), mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.reshape(-1)

    def sample_volume(self, vol: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """Trilinearly sample an arbitrary (1,C,R,R,R) volume on this grid -> (M,C)."""
        out = torch.nn.functional.grid_sample(
            vol, self._coords(pts), mode="bilinear", padding_mode="zeros", align_corners=False)
        return out.reshape(vol.shape[1], -1).T

    def voxel_index(self, pts: torch.Tensor) -> torch.Tensor:
        vi = ((pts - self.lo) / self.extent * self.res).long().clamp(0, self.res - 1)
        return (vi[:, 0] * self.res + vi[:, 1]) * self.res + vi[:, 2]

    def transmittance(self, origins, directions, max_dist, steps=24, sigma=1.0, chunk=1 << 20):
        """Fraction of light surviving from `origins` along `directions`.

        Beer-Lambert integration of the density field. Returns (M,) in [0,1],
        1 = fully lit, 0 = fully occluded.
        """
        m = origins.shape[0]
        dt = max_dist / steps
        out = torch.ones(m, device=origins.device, dtype=origins.dtype)
        # chunk over gaussians to bound peak memory on large splats
        per = max(1, chunk // max(steps, 1))
        for s in range(0, m, per):
            o = origins[s:s + per]
            d = directions[s:s + per]
            t = (torch.arange(steps, device=o.device, dtype=o.dtype) + 0.5) * dt
            pts = o[:, None, :] + d[:, None, :] * t[None, :, None]     # (m,steps,3)
            dens = self.sample(pts.reshape(-1, 3)).reshape(o.shape[0], steps)
            out[s:s + per] = torch.exp(-sigma * dens.sum(-1) * dt)
        return out.clamp(0.0, 1.0)


def smooth_normals(grid, positions, normals, weights=None, radius=1, strength=1.0):
    """Spatially average surfel normals across neighbouring gaussians.

    On a real capture the per-gaussian normal is noisy: reconstructions are
    optimised to reproduce images, not geometry, so many gaussians end up round
    enough that their shortest axis is close to arbitrary. But neighbouring
    gaussians on the same wall *do* share a surface, so averaging over a small
    spatial neighbourhood recovers the plane and cancels the noise.

    Normals are accumulated into the occupancy grid (weighted, so flat and
    confident gaussians dominate), box-blurred over a (2*radius+1)^3
    neighbourhood, then sampled back trilinearly.

    Must run *after* orientation - averaging normals whose signs disagree would
    cancel them to zero rather than agree.
    """
    if strength <= 0:
        return normals
    res = grid.res
    dev, dt = normals.device, normals.dtype
    w = torch.ones(normals.shape[0], device=dev, dtype=dt) if weights is None else weights.clamp_min(0).to(dt)
    flat = grid.voxel_index(positions)

    acc_n = torch.zeros(res ** 3, 3, device=dev, dtype=dt)
    acc_n.index_add_(0, flat, normals * w[:, None])
    acc_w = torch.zeros(res ** 3, device=dev, dtype=dt)
    acc_w.index_add_(0, flat, w)

    k = 2 * int(radius) + 1
    pool = torch.nn.functional.avg_pool3d
    n_vol = acc_n.reshape(res, res, res, 3).permute(3, 0, 1, 2)[None]      # (1,3,R,R,R)
    w_vol = acc_w.reshape(1, 1, res, res, res)
    n_vol = pool(n_vol, k, stride=1, padding=int(radius), count_include_pad=True)
    w_vol = pool(w_vol, k, stride=1, padding=int(radius), count_include_pad=True)

    n_s = grid.sample_volume(n_vol, positions)                              # (M,3)
    w_s = grid.sample_volume(w_vol, positions)                              # (M,1)
    n_s = n_s / w_s.clamp_min(1e-8)
    # where the neighbourhood carried no usable signal, keep the original
    bad = n_s.norm(dim=-1, keepdim=True) < 1e-6
    n_s = torch.where(bad, normals, n_s)
    out = torch.lerp(normals, n_s, float(strength))
    return out / out.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def _hemisphere_dirs(n_dirs, device, dtype):
    """Deterministic Fibonacci hemisphere directions around +Z, (R,3)."""
    i = torch.arange(n_dirs, device=device, dtype=dtype) + 0.5
    phi = i * math.pi * (3.0 - math.sqrt(5.0))
    z = 1.0 - i / n_dirs                       # (0,1], biased toward the pole
    r = (1.0 - z * z).clamp_min(0).sqrt()
    return torch.stack([r * torch.cos(phi), r * torch.sin(phi), z], dim=-1)


def _basis_from_normal(n):
    """Build an orthonormal frame whose +Z is n. Returns (N,3,3)."""
    up = torch.where(
        n[:, 2:3].abs() < 0.9,
        torch.tensor([0.0, 0.0, 1.0], device=n.device, dtype=n.dtype).expand_as(n),
        torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n))
    t = torch.cross(up, n, dim=-1)
    t = t / t.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    b = torch.cross(n, t, dim=-1)
    return torch.stack([t, b, n], dim=-1)      # columns t,b,n


def ambient_occlusion(grid, positions, normals, radius, n_dirs=6, steps=8, sigma=1.0):
    """Hemispherical AO in [0,1]; 1 = fully open, 0 = fully enclosed."""
    frame = _basis_from_normal(normals)                       # (N,3,3)
    dirs = _hemisphere_dirs(n_dirs, positions.device, positions.dtype)   # (R,3)
    bias = 1.5 * grid.voxel_size
    acc = torch.zeros(positions.shape[0], device=positions.device, dtype=positions.dtype)
    for r in range(n_dirs):
        d = (frame @ dirs[r]).contiguous()                    # (N,3) world dir
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        acc += grid.transmittance(positions + normals * bias, d, radius, steps=steps, sigma=sigma)
    return (acc / n_dirs).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# delighting: strip baked illumination to recover an albedo estimate
# --------------------------------------------------------------------------

def estimate_shading(normals, rgb, weights=None, order=2):
    """Fit a low-frequency shading field S(n) to luminance via spherical harmonics.

    Under the standard "albedo is uncorrelated with surface orientation" prior,
    whatever part of the observed luminance is explained by normal direction is
    illumination rather than albedo.

    `order` 2 (9 coefficients) is the default and matters: real baked lighting is
    a sum of *clamped* cosine lobes, and an order-1 fit cannot represent the
    clamp - it undershoots into negatives on the unlit side, which then explodes
    when divided out. Order-2 SH captures ~99% of any diffuse irradiance field
    (Ramamoorthi & Hanrahan 2001), so the residual is genuinely albedo.

    Returns (S (N,) normalised to mean 1, coefficients).
    """
    lum = luminance(rgb)
    basis = (sh9_basis(normals) if order >= 2
             else torch.cat([torch.ones_like(normals[:, :1]), normals], dim=-1))
    w = torch.ones_like(lum) if weights is None else weights.clamp_min(0)
    wb = basis * w[:, None]
    ata = basis.T @ wb
    atb = wb.T @ lum
    k = basis.shape[1]
    ata = ata + torch.eye(k, device=rgb.device, dtype=rgb.dtype) * 1e-4 * ata.diagonal().mean().clamp_min(1e-8)
    coeff = torch.linalg.solve(ata, atb)
    shading = basis @ coeff
    mean = shading.mean().clamp_min(1e-6)
    return (shading / mean).clamp_min(1e-3), coeff


def lift_shadows(rgb, strength, weights=None, floor=0.05):
    """Compress luminance contrast toward the scene mean.

    This is a *tonal* operation, not an inverse-lighting one, and it exists
    because on many real captures the physical route cannot work. `delight`
    removes shading it can attribute to surface orientation; when a scene was
    lit diffusely there is no such attribution to make - measured on an indoor
    capture the entire fitted field spanned 1.1x while the visible light/shadow
    contrast was 4.6x. That residual is inseparable from albedo without extra
    information (a second lighting condition, or a learned prior).

    So rather than pretend, this squashes the ratio directly: a contrast of r
    becomes r ** (1 - strength). 0 is a no-op, 1 flattens luminance entirely.
    It cannot tell a shadow from genuinely dark paint, and will lighten both -
    that trade is the whole reason it is a separate, off-by-default control.
    """
    if float(strength) <= 0.0:
        return rgb
    lum = luminance(rgb)
    w = torch.ones_like(lum) if weights is None else weights.clamp_min(0)
    ref = ((lum * w).sum() / w.sum().clamp_min(1e-8)).clamp_min(1e-4)
    ratio = (lum / ref).clamp_min(float(floor))
    return (rgb * ratio.pow(-float(strength))[:, None]).clamp(0.0, 1.0)


def delight(rgb, normals, amount=1.0, weights=None, floor=0.15, order=2, shadow_lift=0.0):
    """Divide out estimated baked shading to approximate albedo.

    `amount` blends between the original colour (0) and fully flattened albedo
    (1). `floor` bounds the divisor so deeply shadowed gaussians do not blow up.
    `shadow_lift` then applies `lift_shadows` to whatever contrast the physical
    fit could not explain - see that function for why it is sometimes the only
    thing that will move a baked shadow.
    Returns (albedo (N,3), estimated shading (N,)).
    """
    shading, _ = estimate_shading(normals, rgb, weights, order=order)
    s = torch.lerp(torch.ones_like(shading), shading.clamp_min(floor), float(amount))
    albedo = (rgb / s[:, None]).clamp(0.0, 1.0)
    return lift_shadows(albedo, shadow_lift, weights), shading


# --------------------------------------------------------------------------
# environment lighting via order-2 spherical harmonics
# --------------------------------------------------------------------------

def sh9_basis(d: torch.Tensor) -> torch.Tensor:
    """Order-2 real SH basis for directions (N,3) -> (N,9)."""
    x, y, z = d.unbind(-1)
    o = torch.ones_like(x)
    return torch.stack([
        0.282095 * o,
        0.488603 * y, 0.488603 * z, 0.488603 * x,
        1.092548 * x * y, 1.092548 * y * z,
        0.315392 * (3.0 * z * z - 1.0),
        1.092548 * x * z,
        0.546274 * (x * x - y * y),
    ], dim=-1)


def equirect_to_sh9(env: torch.Tensor, to_splat_frame: bool = False) -> torch.Tensor:
    """Project an equirectangular HDRI (H,W,3) into 9 SH coefficients (9,3).

    Rows map to polar angle theta in [0,pi], columns to azimuth phi in [0,2pi),
    with the solid-angle sin(theta) weighting applied. The top row of the image
    is "up" in viewer space.

    Set `to_splat_frame` to build the coefficients directly in splat space, so
    they can be evaluated against splat-space normals without rotating the SH
    basis afterwards.
    """
    h, w, _ = env.shape
    dev, dt = env.device, env.dtype
    theta = (torch.arange(h, device=dev, dtype=dt) + 0.5) / h * math.pi
    phi = (torch.arange(w, device=dev, dtype=dt) + 0.5) / w * (2 * math.pi)
    st, ct = theta.sin(), theta.cos()
    # Y-up world: y = cos(theta), and phi sweeps the XZ plane.
    dx = (st[:, None] * phi.sin()[None, :])
    dy = ct[:, None].expand(h, w)
    dz = (st[:, None] * phi.cos()[None, :])
    dirs = torch.stack([dx, dy, dz], dim=-1).reshape(-1, 3)
    if to_splat_frame:
        dirs = flip_frame(dirs)
    basis = sh9_basis(dirs)                                     # (H*W, 9)
    dw = (st[:, None].expand(h, w).reshape(-1) * (math.pi / h) * (2 * math.pi / w))
    return basis.T @ (env.reshape(-1, 3) * dw[:, None])         # (9,3)


def sh9_irradiance(coeff: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Diffuse irradiance from SH coefficients (9,3) at normals (N,3) -> (N,3).

    Applies the Lambertian convolution factors, so this is cosine-convolved
    irradiance rather than raw radiance.
    """
    conv = torch.tensor([_A0, _A1, _A1, _A1, _A2, _A2, _A2, _A2, _A2],
                        device=coeff.device, dtype=coeff.dtype)
    basis = sh9_basis(normals) * conv                            # (N,9)
    return (basis @ coeff) / math.pi                             # (N,3)


def sh9_radiance(coeff: torch.Tensor, dirs: torch.Tensor) -> torch.Tensor:
    """Raw (unconvolved) radiance from SH in a direction - used for specular."""
    return sh9_basis(dirs) @ coeff


# --------------------------------------------------------------------------
# PBR shading
# --------------------------------------------------------------------------

def _ggx_specular(n_dot_l, n_dot_v, n_dot_h, v_dot_h, roughness, f0):
    """Cook-Torrance GGX: D * G * F / (4 NdotL NdotV), returned per-channel.

    Every scalar term arrives as (N,1) and f0 as (N,3), so the product
    broadcasts straight to (N,3) - no reshaping.
    """
    a = (roughness * roughness).clamp_min(1e-3)
    a2 = a * a
    d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0
    D = a2 / (math.pi * d * d).clamp_min(1e-8)
    k = a / 2.0
    gv = n_dot_v / (n_dot_v * (1.0 - k) + k).clamp_min(1e-8)
    gl = n_dot_l / (n_dot_l * (1.0 - k) + k).clamp_min(1e-8)
    G = gv * gl
    F = f0 + (1.0 - f0) * (1.0 - v_dot_h).clamp(0, 1) ** 5
    return (D * G / (4.0 * n_dot_v * n_dot_l).clamp_min(1e-6)) * F


def shade(albedo, normals, positions, lights, view_dirs, roughness, metallic,
          ambient_color=None, ao=None, grid=None, shadow_steps=24, shadow_sigma=1.0,
          shadow_bias=1.5, specular_strength=1.0):
    """Evaluate the lighting rig over every gaussian.

    lights      : list of dicts (see nodes.SplatLight for the schema)
    view_dirs   : (N,3) unit vectors from surface toward the viewer
    ao          : optional (N,) ambient occlusion, applied to ambient/environment
    grid        : optional DensityGrid enabling cast shadows

    Returns linear RGB (N,3), unclamped so downstream tone mapping can work.
    """
    dev, dt = albedo.device, albedo.dtype
    n = normals
    v = view_dirs
    n_dot_v = (n * v).sum(-1, keepdim=True).clamp_min(1e-4)

    rough = roughness if torch.is_tensor(roughness) else torch.full((albedo.shape[0], 1), float(roughness), device=dev, dtype=dt)
    if rough.dim() == 1:
        rough = rough[:, None]
    met = metallic if torch.is_tensor(metallic) else torch.full((albedo.shape[0], 1), float(metallic), device=dev, dtype=dt)
    if met.dim() == 1:
        met = met[:, None]

    # metals take their specular tint from albedo and lose diffuse response
    f0 = torch.lerp(torch.full_like(albedo, 0.04), albedo, met)
    diffuse_albedo = albedo * (1.0 - met)

    out = torch.zeros_like(albedo)
    ao_t = torch.ones(albedo.shape[0], 1, device=dev, dtype=dt) if ao is None else ao.reshape(-1, 1)

    if ambient_color is not None:
        out = out + diffuse_albedo * ambient_color.reshape(1, 3) * ao_t

    for lt in lights:
        colour = torch.as_tensor(lt["color"], device=dev, dtype=dt).reshape(1, 3) * float(lt["intensity"])
        kind = lt["type"]

        if kind == "environment":
            coeff = lt["sh"].to(device=dev, dtype=dt)
            irr = sh9_irradiance(coeff, n).clamp_min(0.0)
            out = out + diffuse_albedo * irr * colour * ao_t
            if specular_strength > 0:
                refl = (2.0 * n_dot_v * n - v)
                refl = refl / refl.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                spec_env = sh9_radiance(coeff, refl).clamp_min(0.0)
                fres = f0 + (1.0 - f0) * (1.0 - n_dot_v).clamp(0, 1) ** 5
                gloss = (1.0 - rough)
                out = out + spec_env * fres * gloss * colour * ao_t * specular_strength
            continue

        if kind == "directional":
            l = -torch.as_tensor(lt["direction"], device=dev, dtype=dt).reshape(1, 3)
            l = (l / l.norm(dim=-1, keepdim=True).clamp_min(1e-12)).expand_as(n)
            atten = torch.ones(n.shape[0], 1, device=dev, dtype=dt)
            dist = None
        else:  # point / spot
            p = torch.as_tensor(lt["position"], device=dev, dtype=dt).reshape(1, 3)
            delta = p - positions
            dist = delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            l = delta / dist
            rad = float(lt.get("radius", 0.0))
            atten = 1.0 / (1.0 + (dist / max(rad, 1e-6)) ** 2) if rad > 0 else 1.0 / (dist * dist)
            if kind == "spot":
                axis = torch.as_tensor(lt["direction"], device=dev, dtype=dt).reshape(1, 3)
                axis = axis / axis.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                cosang = (-l * axis).sum(-1, keepdim=True)
                inner = math.cos(math.radians(float(lt.get("cone_angle", 40.0)) * 0.5))
                outer = math.cos(math.radians(float(lt.get("cone_angle", 40.0)) * 0.5 +
                                              float(lt.get("cone_softness", 10.0))))
                atten = atten * ((cosang - outer) / max(inner - outer, 1e-6)).clamp(0, 1)

        n_dot_l = (n * l).sum(-1, keepdim=True).clamp_min(0.0)

        vis = torch.ones_like(n_dot_l)
        if grid is not None and lt.get("cast_shadow", True):
            lit = (n_dot_l.reshape(-1) > 0)
            if lit.any():
                origin = positions[lit] + n[lit] * (shadow_bias * grid.voxel_size)
                ray_len = (float(dist[lit].max().item()) if dist is not None
                           else float(grid.extent.max().item()))
                t = grid.transmittance(origin, l[lit], ray_len, steps=shadow_steps, sigma=shadow_sigma)
                vis = vis.clone()
                vis[lit] = t.reshape(-1, 1)
                vis[~lit] = 0.0

        radiance = colour * n_dot_l * atten * vis
        out = out + diffuse_albedo * radiance / math.pi

        if specular_strength > 0:
            h = l + v
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-12)
            n_dot_h = (n * h).sum(-1, keepdim=True).clamp(0, 1)
            v_dot_h = (v * h).sum(-1, keepdim=True).clamp(0, 1)
            spec = _ggx_specular(n_dot_l, n_dot_v, n_dot_h, v_dot_h, rough, f0)
            out = out + spec * radiance * specular_strength

    return out


def tonemap(rgb: torch.Tensor, exposure: float = 1.0, mode: str = "aces") -> torch.Tensor:
    """Map HDR shading results back into the [0,1] range the splat DC expects."""
    c = rgb * float(exposure)
    if mode == "none":
        return c.clamp(0.0, 1.0)
    if mode == "reinhard":
        return (c / (1.0 + c)).clamp(0.0, 1.0)
    # Narkowicz ACES approximation
    a, b, cc, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
    return ((c * (a * c + b)) / (c * (cc * c + d) + e)).clamp(0.0, 1.0)


# --------------------------------------------------------------------------
# multi-view back-projection: writing 2D image results onto 3D gaussians
# --------------------------------------------------------------------------
#
# A splat has no UV atlas, so an image model's output cannot be "unwrapped"
# onto it. It does not need to be: every gaussian is seen from many cameras,
# so the gaussians themselves are the atlas. Render N views, run the model per
# view, then for each gaussian gather the pixel it lands on in every view and
# average - weighted by whether it was actually visible there and how squarely
# it faced that camera.
#
# The weighting is what makes this work rather than smear. A gaussian hidden
# behind a wall in view 7 still projects to some pixel in view 7; without a
# visibility term it would collect that unrelated colour.

def camera_basis(cam_pos, cam_target, device=None, dtype=torch.float32):
    """Orthonormal (right, up, forward) plus position for a camera, splat frame.

    The splat frame is Y-down, so the world-up hint is -Y.
    """
    p = torch.as_tensor(cam_pos, device=device, dtype=dtype)
    t = torch.as_tensor(cam_target, device=device, dtype=dtype)
    fwd = t - p
    fwd = fwd / fwd.norm().clamp_min(1e-8)
    up_hint = torch.tensor([0.0, -1.0, 0.0], device=fwd.device, dtype=dtype)
    if (fwd @ up_hint).abs() > 0.999:                   # looking straight up or down
        up_hint = torch.tensor([0.0, 0.0, 1.0], device=fwd.device, dtype=dtype)
    right = torch.cross(fwd, up_hint, dim=-1)
    right = right / right.norm().clamp_min(1e-8)
    up = torch.cross(right, fwd, dim=-1)
    return right, up, fwd, p


def project(positions, cam_pos, cam_target, fov_deg, width, height):
    """Perspective-project gaussians into a camera.

    Returns (px, py, depth, inside) with px/py in float pixel coordinates and
    `inside` true where the gaussian is in front of the camera and on screen.
    """
    right, up, fwd, p = camera_basis(cam_pos, cam_target, positions.device, positions.dtype)
    d = positions - p
    z = d @ fwd
    focal = (float(height) * 0.5) / math.tan(math.radians(float(fov_deg)) * 0.5)
    safe = z.clamp_min(1e-6)
    px = (d @ right) / safe * focal + float(width) * 0.5
    py = -(d @ up) / safe * focal + float(height) * 0.5
    inside = (z > 1e-3) & (px >= 0) & (px <= width - 1) & (py >= 0) & (py <= height - 1)
    return px, py, z, inside


def sample_image(image, px, py):
    """Bilinearly sample an (H,W,C) image at float pixel coords -> (N,C)."""
    h, w = image.shape[0], image.shape[1]
    x = px.clamp(0, w - 1)
    y = py.clamp(0, h - 1)
    x0 = x.floor().long().clamp(0, w - 1); x1 = (x0 + 1).clamp(0, w - 1)
    y0 = y.floor().long().clamp(0, h - 1); y1 = (y0 + 1).clamp(0, h - 1)
    fx = (x - x0.to(x.dtype))[:, None]
    fy = (y - y0.to(y.dtype))[:, None]
    flat = image.reshape(-1, image.shape[2])
    g = lambda yy, xx: flat[yy * w + xx]
    top = g(y0, x0) * (1 - fx) + g(y0, x1) * fx
    bot = g(y1, x0) * (1 - fx) + g(y1, x1) * fx
    return top * (1 - fy) + bot * fy


def depth_buffer(positions, cam_pos, cam_target, fov_deg, width, height, opacities=None,
                 min_opacity=0.25):
    """Nearest depth per pixel, splatted from the gaussians. -> (H*W,) float

    Empty pixels hold +inf. This is an exact occlusion test, unlike marching the
    density grid, which answers "roughly how much stuff is in the way" and on a
    real capture passes surfaces several metres apart as equally visible.
    """
    px, py, z, inside = project(positions, cam_pos, cam_target, fov_deg, width, height)
    keep = inside
    if opacities is not None:
        keep = keep & (opacities.reshape(-1) > float(min_opacity))
    buf = torch.full((height * width,), float("inf"), device=positions.device,
                     dtype=positions.dtype)
    if not bool(keep.any()):
        return buf
    idx = py[keep].long() * width + px[keep].long()
    buf.scatter_reduce_(0, idx, z[keep], reduce="amin", include_self=True)
    return buf


def surface_mask(positions, cam_pos, cam_target, fov_deg, width, height, opacities=None,
                 tol=0.05, rel_tol=0.02, min_opacity=0.25, buf=None):
    """Which gaussians form the first visible surface from this camera. -> (N,) bool

    A gaussian passes if its depth is within tolerance of the nearest depth
    recorded at the pixel it lands on. Everything behind that - the far wall
    seen "through" the near one, geometry inside a cupboard - is rejected.

    Tolerance is absolute plus a fraction of depth, because a splat's surface
    has real thickness and projected error grows with distance.
    """
    px, py, z, inside = project(positions, cam_pos, cam_target, fov_deg, width, height)
    if buf is None:
        buf = depth_buffer(positions, cam_pos, cam_target, fov_deg, width, height,
                           opacities, min_opacity)
    out = torch.zeros(positions.shape[0], dtype=torch.bool, device=positions.device)
    if not bool(inside.any()):
        return out
    idx = py[inside].long() * width + px[inside].long()
    near = buf[idx]
    lim = float(tol) + float(rel_tol) * z[inside].abs()
    out[inside] = z[inside] <= near + lim
    return out


def view_weight(positions, normals, cam_pos, cam_target, fov_deg, width, height,
                grid=None, sigma=8.0, steps=24, min_facing=0.05, edge_fade=0.06,
                opacities=None, use_zbuffer=True, depth_tol=0.05, depth_buf=None):
    """How far a gaussian's sample from one camera should be trusted. -> (N,) >= 0

    Three independent terms, each a real failure mode:
      * on screen and in front of the camera, with a soft fade at the frame
        edge so a gaussian does not pop as it crosses between views
      * facing the camera - a surfel seen edge-on covers almost no pixels, so
        its sample is unreliable
      * visible - ray-march the density field from the surface to the camera.
        Without this, geometry hidden behind a wall still projects to a pixel
        and would collect a completely unrelated colour.
    """
    px, py, z, inside = project(positions, cam_pos, cam_target, fov_deg, width, height)
    w = inside.to(positions.dtype)

    if edge_fade > 0:
        fx = (px / max(width - 1, 1)).clamp(0, 1)
        fy = (py / max(height - 1, 1)).clamp(0, 1)
        m = float(edge_fade)
        fade = (fx.clamp(0, m) / m) * ((1 - fx).clamp(0, m) / m) \
             * (fy.clamp(0, m) / m) * ((1 - fy).clamp(0, m) / m)
        w = w * fade.clamp(0, 1)

    right, up, fwd, p = camera_basis(cam_pos, cam_target, positions.device, positions.dtype)
    to_cam = p - positions
    dist = to_cam.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    dir_cam = to_cam / dist
    if normals is not None:
        facing = (normals * dir_cam).sum(-1).abs()      # abs: surfel orientation is arbitrary
        w = w * (facing - min_facing).clamp_min(0.0) / max(1.0 - min_facing, 1e-6)

    if use_zbuffer:
        # exact first-surface test; strictly better than marching the density
        # field, which passed surfaces metres apart as equally visible
        # `depth_buf` must be built from the WHOLE scene. Building it from the
        # queried subset makes every queried gaussian trivially the nearest,
        # which silently disables occlusion instead of erroring.
        w = w * surface_mask(positions, cam_pos, cam_target, fov_deg, width, height,
                             opacities=opacities, tol=depth_tol,
                             buf=depth_buf).to(w.dtype)
    elif grid is not None:
        eps = grid.voxel_size * 1.5
        origin = positions + dir_cam * eps
        reach = (dist.reshape(-1) - eps).clamp_min(0.0)
        vis = torch.ones_like(w)
        far = reach > 1e-4
        if far.any():
            vis[far] = grid.transmittance(origin[far], dir_cam[far],
                                          float(reach[far].max()), steps=steps, sigma=sigma)
        w = w * vis
    return w.clamp_min(0.0)


def backproject(images, cameras, positions, normals=None, grid=None, fov_deg=55.0,
                sigma=8.0, steps=24, min_facing=0.05, edge_fade=0.06, masks=None):
    """Gather per-gaussian values from a set of rendered views.

    `images`  : list of (H,W,C) tensors, one per view, all sharing C
    `cameras` : list of (cam_pos, cam_target) in the splat frame
    `masks`   : optional list of (N,) bool, which gaussians each view actually
                drew. An interior view is rendered with the near wall cut away,
                and the cut gaussians must not then sample the image they were
                excluded from - they would collect whatever showed through the
                hole they left.

    Returns (values (N,C), coverage (N,)). Coverage is the summed weight, so
    zero means the gaussian was never reliably seen and its value is a fallback
    rather than a measurement - callers must not treat those as data.
    """
    n = positions.shape[0]
    c = images[0].shape[2]
    num = torch.zeros(n, c, device=positions.device, dtype=positions.dtype)
    den = torch.zeros(n, device=positions.device, dtype=positions.dtype)
    for i, (img, (cp, ct)) in enumerate(zip(images, cameras)):
        h, w = img.shape[0], img.shape[1]
        wt = view_weight(positions, normals, cp, ct, fov_deg, w, h,
                         grid=grid, sigma=sigma, steps=steps,
                         min_facing=min_facing, edge_fade=edge_fade)
        if masks is not None and masks[i] is not None:
            wt = wt * masks[i].to(wt.dtype)
        hit = wt > 1e-6
        if not bool(hit.any()):
            continue
        px, py, _, _ = project(positions, cp, ct, fov_deg, w, h)
        vals = sample_image(img.to(positions.dtype), px[hit], py[hit])
        num[hit] += vals * wt[hit][:, None]
        den[hit] += wt[hit]
    good = den > 1e-6
    out = torch.zeros_like(num)
    out[good] = num[good] / den[good][:, None]
    return out, den


def orbit_cameras(centre, radius, count=24, pitch_deg=18.0):
    """Cameras evenly spaced on a ring around `centre`, in the splat frame.

    Returns a list of (position, target).
    """
    out = []
    c = [float(v) for v in centre]
    pitch = math.radians(float(pitch_deg))
    for i in range(int(count)):
        a = 2.0 * math.pi * i / float(count)
        # splat frame is Y-down, so a positive pitch lifts the camera toward -Y
        pos = [c[0] + radius * math.cos(pitch) * math.sin(a),
               c[1] - radius * math.sin(pitch),
               c[2] + radius * math.cos(pitch) * math.cos(a)]
        out.append((pos, list(c)))
    return out


# --------------------------------------------------------------------------
# shadow masking: pointing a generative fill at shading rather than albedo
# --------------------------------------------------------------------------
#
# Lifting a baked shadow cannot work where the shadow is deep: the capture
# recorded almost no signal there, and dividing near-zero by anything is still
# near-zero. Regenerating that region is the only route that can put detail
# back, which makes the mask the critical part - point a generative fill at
# dark *paint* instead of dark *shading* and it erases real material.
#
# Marigold's diffuse shading is the right source: measured on a real capture it
# accounted for ~95% of the shadow in image space while carrying no albedo, so
# a grimy dark floor stain is correctly ignored while the shadow beside it is
# selected.

def _gaussian_kernel1d(sigma, device, dtype=torch.float32):
    r = int(float(sigma) * 3) | 1
    x = torch.arange(-(r // 2), r // 2 + 1, device=device, dtype=dtype)
    k = torch.exp(-x * x / (2.0 * float(sigma) ** 2))
    return k / k.sum(), r // 2


def blur2d(x, sigma):
    """Separable gaussian blur of a (H,W) tensor."""
    k, p = _gaussian_kernel1d(sigma, x.device, x.dtype)
    t = x[None, None]
    t = torch.nn.functional.conv2d(
        torch.nn.functional.pad(t, (p, p, 0, 0), mode="replicate"), k.view(1, 1, 1, -1))
    t = torch.nn.functional.conv2d(
        torch.nn.functional.pad(t, (0, 0, p, p), mode="replicate"), k.view(1, 1, -1, 1))
    return t[0, 0]


def _dilate2d(m, r):
    return torch.nn.functional.max_pool2d(m[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]


def _erode2d(m, r):
    return -torch.nn.functional.max_pool2d(-m[None, None], 2 * r + 1, stride=1, padding=r)[0, 0]


def shadow_mask(shading, ratio=0.62, context=45.0, open_r=2, close_r=6, feather=3.0,
                min_frac=0.001, grow=0):
    """Where is a pixel shadowed *relative to its surroundings*? -> ((H,W) in [0,1], ratio map)

    `shading` : (H,W) or (H,W,3) diffuse shading, any scale - it is normalised here
    `ratio`   : below this fraction of the local mean counts as shadow
    `context` : blur sigma defining "local". Must be much wider than the shadow,
                or the shadow drags down its own reference and hides itself.

    Relative rather than absolute, because a genuinely dark surface in full
    light is not a shadow and must not be regenerated. Open then close removes
    speckle and fills pinholes; the final blur feathers the edge so a fill
    blends instead of seaming.
    """
    s = shading if shading.dim() == 2 else luminance(shading)
    s = s.float().clamp_min(1e-4)
    rel = s / blur2d(s, context).clamp_min(1e-4)
    m = (rel < float(ratio)).float()
    if open_r:
        m = _dilate2d(_erode2d(m, int(open_r)), int(open_r))
    if close_r:
        m = _erode2d(_dilate2d(m, int(close_r)), int(close_r))
    if grow:
        # A cast shadow has a penumbra: the threshold catches the core but leaves
        # a soft rim just outside it, which survives the fill and reads as a
        # residual shadow outline. Growing the mask past that rim removes it.
        #
        # Keep this SMALL. Measured on a 1024x768 render, 4px and 14px removed
        # the same amount of shadow (41.2% vs 41.6%), but 4px repainted 3.19% of
        # the frame against 5.95% and left chroma noise unchanged (-0.1%) where
        # 14px raised it 6.3%. Over-growing buys nothing and costs real surface.
        # 3-5px is the useful range at that resolution.
        m = _dilate2d(m, int(grow))
    if float(m.mean()) < float(min_frac):
        return torch.zeros_like(s), rel
    if feather > 0:
        m = blur2d(m, feather).clamp(0, 1)
    return m, rel


def edge_map(depth, rel_thresh=0.02, abs_thresh=0.02):
    """Where a depth buffer breaks - object silhouettes. -> (H,W) float in {0,1}

    Infinities (empty pixels) count as edges.
    """
    d = depth.clone()
    finite = torch.isfinite(d)
    fill = d[finite].max() if bool(finite.any()) else torch.tensor(1.0, device=d.device)
    d[~finite] = fill
    gx = torch.zeros_like(d); gy = torch.zeros_like(d)
    gx[:, 1:] = (d[:, 1:] - d[:, :-1]).abs()
    gy[1:, :] = (d[1:, :] - d[:-1, :]).abs()
    g = torch.maximum(gx, gy)
    lim = float(abs_thresh) + float(rel_thresh) * d.abs()
    e = (g > lim).float()
    e = torch.maximum(e, (~finite).float())
    return e


def flat_region_mask(depth, guard=7, rel_thresh=0.02, abs_thresh=0.02, smooth=2):
    """Pixels safely inside a continuous surface, away from any silhouette.

    A generative fill is reliable when continuing a flat surface - floor, wall -
    and destructive when it crosses an object boundary, because it has no idea
    the shape it is painting over was a sink pedestal. Excluding a band around
    every depth discontinuity confines the fill to where it can only invent
    texture, not structure.

    MEASURED LIMITATION: on the real capture this is the wrong tool for keeping
    a generative fill off objects. Even smoothed, the buffer's internal scatter
    punches holes through the middle of a shadow region, fragmenting the mask
    into islands - and inpainting a fragmented mask produces visible patchwork,
    worse than no guard at all. Use it for genuine large depth steps; for
    excluding a sink or a mirror, segment them semantically instead.

    `smooth` matters and defaults on. A depth buffer splatted from gaussians is
    not a mesh depth buffer: on a real capture only ~52% of pixels received a
    point at 1024x768, so the raw buffer is riddled with holes and scatter and
    68% of it reads as discontinuity - which rejected the entire mask. A
    min-filter closes the holes toward the nearest surface and a light blur
    removes point scatter, taking that to 12% while keeping genuine steps.
    """
    if smooth:
        r = int(smooth)
        d = depth.clone()
        finite = torch.isfinite(d)
        if bool(finite.any()):
            d[~finite] = d[finite].max()
        # min-filter (erode) closes holes toward the nearer surface
        d = -torch.nn.functional.max_pool2d(-d[None, None], 2 * r + 1, 1, r)[0, 0]
        depth = blur2d(d, float(r))
    e = edge_map(depth, rel_thresh, abs_thresh)
    grown = torch.nn.functional.max_pool2d(e[None, None], 2 * int(guard) + 1, 1, int(guard))[0, 0]
    return 1.0 - grown
