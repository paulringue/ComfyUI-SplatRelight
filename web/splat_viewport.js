/**
 * Interactive splat viewport for ComfyUI-SplatRelight.
 *
 * Shows the splat, lets you orbit to set the render camera, and lets you drag
 * the lights around *inside* the 3D view. Camera and rig both come back out of
 * the node, so the rest of the graph follows whatever you set up here.
 *
 * Written from first principles on WebGL2 - no third-party viewer, no bundled
 * renderer. Points are drawn as depth-tested round sprites rather than
 * alpha-blended splats: for a viewport whose job is aiming lights that reads
 * just as well, costs one draw call, and sidesteps per-frame depth sorting
 * entirely. The final image always comes from ComfyUI's real rasteriser.
 *
 * Everything here is viewer space (Y-up); the Python side converts.
 */

import { app } from "../../scripts/app.js";

const VS = `#version 300 es
layout(location=0) in vec3 aPos;
layout(location=1) in vec4 aCol;
uniform mat4 uViewProj;
uniform float uPointSize;
uniform float uViewportH;
out vec4 vCol;
void main() {
  vec4 clip = uViewProj * vec4(aPos, 1.0);
  gl_Position = clip;
  // keep sprites a constant world size: shrink with distance
  float w = max(clip.w, 0.001);
  gl_PointSize = clamp(uPointSize * uViewportH / (w * 400.0), 1.0, 64.0);
  vCol = aCol;
}`;

const FS = `#version 300 es
precision mediump float;
in vec4 vCol;
out vec4 frag;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  if (dot(d, d) > 0.25) discard;          // round sprite
  frag = vec4(vCol.rgb, 1.0);
}`;

// ---------------------------------------------------------------- math
const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
const add = (a, b) => [a[0]+b[0], a[1]+b[1], a[2]+b[2]];
const mul = (a, s) => [a[0]*s, a[1]*s, a[2]*s];
const dot = (a, b) => a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
const len = (a) => Math.hypot(a[0], a[1], a[2]);
const norm = (a) => { const l = len(a) || 1; return [a[0]/l, a[1]/l, a[2]/l]; };
const cross = (a, b) => [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];

function lookAt(eye, target, up) {
  const z = norm(sub(eye, target));
  let x = cross(up, z);
  if (len(x) < 1e-6) x = cross([0, 0, 1], z);
  x = norm(x);
  const y = cross(z, x);
  return [
    x[0], y[0], z[0], 0,
    x[1], y[1], z[1], 0,
    x[2], y[2], z[2], 0,
    -dot(x, eye), -dot(y, eye), -dot(z, eye), 1,
  ];
}

function perspective(fovDeg, aspect, near, far) {
  const f = 1 / Math.tan((fovDeg * Math.PI) / 360);
  return [
    f / aspect, 0, 0, 0,
    0, f, 0, 0,
    0, 0, (far + near) / (near - far), -1,
    0, 0, (2 * far * near) / (near - far), 0,
  ];
}

function mat4mul(a, b) {
  const o = new Array(16).fill(0);
  for (let c = 0; c < 4; c++)
    for (let r = 0; r < 4; r++)
      for (let k = 0; k < 4; k++) o[c * 4 + r] += a[k * 4 + r] * b[c * 4 + k];
  return o;
}

function project(m, p) {
  const x = m[0]*p[0] + m[4]*p[1] + m[8]*p[2] + m[12];
  const y = m[1]*p[0] + m[5]*p[1] + m[9]*p[2] + m[13];
  const w = m[3]*p[0] + m[7]*p[1] + m[11]*p[2] + m[15];
  return { x, y, w };
}

const dirFromYawPitch = (yaw, pitch) => {
  const y = (yaw * Math.PI) / 180, p = (pitch * Math.PI) / 180;
  return [Math.cos(p) * Math.sin(y), Math.sin(p), Math.cos(p) * Math.cos(y)];
};
const yawPitchFromDir = (d) => {
  const v = norm(d);
  return { yaw: (Math.atan2(v[0], v[2]) * 180) / Math.PI,
           pitch: (Math.asin(Math.max(-1, Math.min(1, v[1]))) * 180) / Math.PI };
};
const hexToRgb = (h) => {
  const s = (h || "#fff").replace("#", "");
  return s.length === 6
    ? [parseInt(s.slice(0,2),16)/255, parseInt(s.slice(2,4),16)/255, parseInt(s.slice(4,6),16)/255]
    : [1, 1, 1];
};

const SLOTS = ["key", "fill", "rim"];
// Translate gizmo, same colour convention as every 3D package: X red, Y green,
// Z blue. Dragging an arrow moves along that world axis only.
const AXES = [
  { v: [1, 0, 0], col: "#e04a4a", name: "X" },
  { v: [0, 1, 0], col: "#3fbf50", name: "Y" },
  { v: [0, 0, 1], col: "#3b7fe0", name: "Z" },
];

function distToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const l2 = dx * dx + dy * dy;
  if (l2 < 1e-9) return Math.hypot(px - ax, py - ay);
  let t = ((px - ax) * dx + (py - ay) * dy) / l2;
  t = Math.max(0, Math.min(1, t));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}
const DEFAULT_LIGHTS = [
  { on: true, yaw: 55, pitch: 38, color: "#FFC98F" },
  { on: true, yaw: -70, pitch: 8, color: "#6E9CFF" },
  { on: false, yaw: 175, pitch: 25, color: "#FFFFFF" },
];

// ---------------------------------------------------------------- viewport
class Viewport {
  constructor(node, stateWidget) {
    this.node = node;
    this.stateWidget = stateWidget;
    this.lights = DEFAULT_LIGHTS.map((l) => ({ ...l }));
    // Two distinct points, deliberately. `centre` is the camera's orbit pivot
    // and moves when you pan; `sceneCentre` is the splat itself and never does.
    // Conflating them makes the lights slide off the subject the moment you pan.
    this.centre = [0, 0, 0];
    this.sceneCentre = [0, 0, 0];
    this.radius = 1;
    this.yaw = 40; this.pitch = 22; this.dist = 4;
    this.fov = 40;
    this.pointSize = 2.6;
    this.bg = [0.07, 0.075, 0.095];
    this.count = 0;
    this.drag = null;
    this.hover = -1;
    this.hoverAxis = false;
    this.selected = 0;          // key light selected by default, so its arrows show
    this.crop = null;           // {centre:[x,y,z]} when the crop region is shown

    const wrap = document.createElement("div");
    wrap.style.cssText = "display:flex;flex-direction:column;gap:3px;padding:2px 0;width:100%;";
    const canvas = document.createElement("canvas");
    // Explicit pixel size, computed from the node's own width in resize().
    // width:100% is not usable here: ComfyUI's DOM-widget container has no
    // definite width, so a percentage collapses and the canvas ends up a small
    // box in the corner of a large node.
    canvas.style.cssText =
      "display:block;border-radius:6px;touch-action:none;cursor:grab;background:#111318;";
    const row = document.createElement("div");
    row.style.cssText = "display:flex;gap:6px;justify-content:center;align-items:center;";
    const mkBtn = (label, title) => {
      const b = document.createElement("button");
      b.textContent = label;
      b.title = title;
      b.style.cssText =
        "font-size:11px;font-family:sans-serif;padding:3px 10px;border-radius:5px;" +
        "border:1px solid rgba(255,255,255,.18);background:#2a2d36;color:#dfe2ea;cursor:pointer;";
      b.addEventListener("pointerdown", (e) => e.stopPropagation());
      return b;
    };
    this.setBtn = mkBtn("Set Camera", "Commit this view as the render camera");
    this.recallBtn = mkBtn("Recall", "Snap the view back to the committed camera");
    // Recall only helps once a camera has been committed, and it returns you to
    // whatever that camera was - which is no use if the committed shot is itself
    // a tight close-up. Fit is the unconditional way out of a lost view: it
    // re-centres on the splat and backs off to a framing distance, undoing
    // exactly the pan and zoom that stranded you, while keeping the orbit angle
    // you had chosen.
    this.fitBtn = mkBtn("Fit", "Re-centre on the splat and back off to frame it (undo pan/zoom)");
    row.appendChild(this.setBtn);
    row.appendChild(this.recallBtn);
    row.appendChild(this.fitBtn);

    const bar = document.createElement("div");
    bar.style.cssText = "font-size:10px;opacity:.6;text-align:center;font-family:sans-serif;line-height:1.3;";
    // The viewport is fed by the node's own output, so there is nothing to show
    // - and nothing to drag - until the graph has run once. Say so, rather than
    // presenting an empty box that looks broken.
    bar.textContent = "Run the graph once to load the splat, then drag here";
    // The GL canvas and its 2D overlay must share one positioned box of their
    // own. Parenting the overlay to the whole wrapper stretches it over the
    // buttons and status line too, which throws the handles out of alignment
    // with the splat by exactly the height of that furniture.
    const stage = document.createElement("div");
    stage.style.cssText = "position:relative;width:100%;line-height:0;";
    stage.appendChild(canvas);
    wrap.appendChild(stage); wrap.appendChild(row); wrap.appendChild(bar);
    this.el = wrap; this.stage = stage; this.canvas = canvas; this.status = bar;

    // The committed camera is what the graph renders from. Orbiting only moves
    // the preview, so you can look around without disturbing a framing you
    // already chose - press Set Camera to adopt the current view.
    this.committed = null;
    this.setBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.commitCamera();
    });
    this.recallBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.recallCamera();
    });
    this.fitBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      this.fitView();
    });

    this.initGL();
    this.bind();
    this.refreshButtons();

    // ComfyUI re-lays DOM widgets out on move, collapse, zoom and panel
    // changes, and does not always route that through onResize. Watch the box
    // itself and re-fit when it actually changes.
    if (window.ResizeObserver) {
      // Coalesce to one rAF: a drag emits a burst of these, and some of them
      // report a collapsed box mid-relayout.
      this._ro = new ResizeObserver(() => {
        if (this._pending) return;
        this._pending = requestAnimationFrame(() => {
          this._pending = 0;
          this.resize();
        });
      });
      this._ro.observe(this.stage);
    }
  }

  liveCamera() {
    return { position: this.eye(), target: this.centre.slice(), fov: this.fov };
  }

  cameraDirty() {
    if (!this.committed) return true;
    const a = this.liveCamera(), b = this.committed;
    const far = (p, q) => Math.max(...[0, 1, 2].map((i) => Math.abs(p[i] - q[i])));
    const scale = Math.max(this.radius, 1e-3) * 0.002;
    return far(a.position, b.position) > scale || far(a.target, b.target) > scale ||
           Math.abs(a.fov - b.fov) > 0.01;
  }

  commitCamera() {
    this.committed = this.liveCamera();
    this.sync();
    // Positive confirmation. The steady state after committing is "not dirty",
    // i.e. the button goes quiet - which on its own looks identical to nothing
    // having happened. Flash so the click is unmistakable.
    const b = this.setBtn;
    b.textContent = "Camera Set ✓";
    b.style.background = "#2f7d4f";
    b.style.borderColor = "rgba(130,235,170,.75)";
    b.style.color = "#eafff2";
    clearTimeout(this._flashTimer);
    this._flashTimer = setTimeout(() => this.refreshButtons(), 1200);
    this.draw();
  }

  fitView() {
    // Undo exactly the two things that strand a view: panning moved `centre`
    // off the splat, and zooming changed `dist`. Both are restored to the
    // framing the viewport opens with on a fresh load.
    //
    // The orbit angle is deliberately kept. Resetting it too would be a bigger
    // surprise than the problem being fixed - you asked to get back to the
    // splat, not to lose the direction you were looking from.
    this.centre = this.sceneCentre.slice();
    this.dist = this.radius / Math.tan((this.fov * Math.PI) / 360) * 1.15;
    this.sync();
    this.refreshButtons();
    this.draw();
    // Positive confirmation: with the angle preserved, a fit can otherwise look
    // like nothing happened when the view was only slightly off.
    const b = this.fitBtn;
    b.textContent = "Fitted ✓";
    b.style.background = "#2f7d4f";
    b.style.borderColor = "rgba(130,235,170,.75)";
    b.style.color = "#eafff2";
    clearTimeout(this._fitTimer);
    this._fitTimer = setTimeout(() => this.refreshButtons(), 1000);
  }

  recallCamera() {
    if (!this.committed) return;
    const off = sub(this.committed.position, this.committed.target);
    this.dist = len(off) || this.dist;
    const yp = yawPitchFromDir(off);
    this.yaw = yp.yaw; this.pitch = yp.pitch;
    this.centre = this.committed.target.slice();
    this.fov = this.committed.fov;
    this.refreshButtons();
    this.draw();
  }

  refreshButtons() {
    if (!this.setBtn) return;
    const dirty = this.cameraDirty();
    this.setBtn.style.background = dirty ? "#3b5b8c" : "#2a2d36";
    this.setBtn.style.borderColor = dirty ? "rgba(140,180,255,.55)" : "rgba(255,255,255,.18)";
    this.setBtn.style.color = "#dfe2ea";
    this.setBtn.textContent = dirty ? "Set Camera •" : "Set Camera";
    this.recallBtn.disabled = !this.committed;
    this.recallBtn.style.opacity = this.committed ? "1" : ".45";
    // Fit is always available - it is the escape hatch, so it must never be the
    // thing that is greyed out when the view is lost.
    if (this.fitBtn) {
      this.fitBtn.textContent = "Fit";
      this.fitBtn.style.background = "#2a2d36";
      this.fitBtn.style.borderColor = "rgba(255,255,255,.18)";
      this.fitBtn.style.color = "#dfe2ea";
      this.fitBtn.disabled = false;
      this.fitBtn.style.opacity = "1";
    }
  }

  initGL() {
    // preserveDrawingBuffer: the browser is free to clear the drawing buffer
    // after compositing, and dragging the node forces exactly that - the
    // viewport would go black until something happened to trigger a redraw.
    const gl = this.canvas.getContext("webgl2",
      { antialias: true, alpha: false, preserveDrawingBuffer: true });
    this.gl = gl;
    if (!gl) { this.status.textContent = "WebGL2 unavailable - viewport disabled"; return; }
    const sh = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    this.prog = p;
    this.uViewProj = gl.getUniformLocation(p, "uViewProj");
    this.uPointSize = gl.getUniformLocation(p, "uPointSize");
    this.uViewportH = gl.getUniformLocation(p, "uViewportH");
    this.vao = gl.createVertexArray();
    this.posBuf = gl.createBuffer();
    this.colBuf = gl.createBuffer();
    gl.enable(gl.DEPTH_TEST);
  }

  async load(info) {
    this.sceneCentre = (info.centre || [0, 0, 0]).slice();
    this.centre = this.sceneCentre.slice();
    this.radius = info.radius || 1;
    this.pointSize = info.point_size ?? 2.6;
    this.bg = hexToRgb(info.background || "#111318");
    if (Array.isArray(info.lights) && info.lights.length) {
      this.lights = info.lights.map((l, i) => ({ ...DEFAULT_LIGHTS[i], ...l }));
    }
    if (info.camera?.position && info.camera?.target) {
      this.committed = {
        position: info.camera.position.slice(),
        target: info.camera.target.slice(),
        fov: info.camera.fov ?? this.fov,
      };
      this.recallCamera();                       // open on the committed framing
    } else {
      this.dist = this.radius / Math.tan((this.fov * Math.PI) / 360) * 1.15;
      this.committed = this.liveCamera();        // first run: adopt the default view
    }
    if (info.show_crop && info.crop) {
      const sz = info.crop.size;
      this.crop = { centre: (info.crop.centre || this.sceneCentre).slice(),
                    size: Array.isArray(sz) ? sz.slice() : [sz ?? 1.6, sz ?? 1.6, sz ?? 1.6],
                    shape: info.crop.shape || 'box' };
    } else {
      this.crop = null;
    }
    this.refreshButtons();

    const gl = this.gl;
    if (!gl) return;
    const res = await fetch(`/api/view?filename=${encodeURIComponent(info.file)}&type=temp`);
    const buf = await res.arrayBuffer();
    const magic = new TextDecoder().decode(new Uint8Array(buf, 0, 4));
    if (magic !== "SRV1") { this.status.textContent = "viewport: unexpected data"; return; }
    const n = new Uint32Array(buf, 4, 1)[0];
    const pos = new Float32Array(buf, 8, n * 3);
    const col = new Uint8Array(buf, 8 + n * 12, n * 4);
    this.count = n;

    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.posBuf);
    gl.bufferData(gl.ARRAY_BUFFER, pos, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.colBuf);
    gl.bufferData(gl.ARRAY_BUFFER, col, gl.STATIC_DRAW);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 4, gl.UNSIGNED_BYTE, true, 0, 0);
    gl.bindVertexArray(null);
    this.status.textContent =
      `${n.toLocaleString()} points · left-drag orbit · right-drag pan · scroll zoom · drag a light handle`;
    this.sync();
    this.draw();
  }

  eye() {
    const d = dirFromYawPitch(this.yaw, this.pitch);
    return add(this.centre, mul(d, this.dist));
  }

  viewProj() {
    const w = this.canvas.width || 1, h = this.canvas.height || 1;
    return mat4mul(perspective(this.fov, w / h, Math.max(this.radius * 0.005, 1e-3),
                               this.radius * 200 + 100),
                   lookAt(this.eye(), this.centre, [0, 1, 0]));
  }

  /** Where a light's handle sits in the world: anchored to the splat, at its
   *  own distance, so panning the camera never drags the lights with it. */
  handleWorld(i) {
    const l = this.lights[i];
    const d = dirFromYawPitch(l.yaw, l.pitch);
    const dist = l.dist > 0 ? l.dist : this.radius * 1.45;
    return add(this.sceneCentre, mul(d, dist));
  }

  handleScreen(i) {
    const p = project(this.viewProj(), this.handleWorld(i));
    if (p.w <= 0) return null;                       // behind the camera
    return {
      x: (p.x / p.w * 0.5 + 0.5) * this.canvas.width,
      y: (0.5 - p.y / p.w * 0.5) * this.canvas.height,
    };
  }

  cropHalf() {
    const s2 = this.crop?.size || [1.6, 1.6, 1.6];
    return this.crop?.shape === "sphere"
      ? [s2[0] / 2, s2[0] / 2, s2[0] / 2]
      : [s2[0] / 2, s2[1] / 2, s2[2] / 2];
  }

  /** One square handle on each +face of the box; dragging it resizes that axis. */
  cropHandles() {
    if (!this.crop) return [];
    const h = this.cropHalf(), c = this.crop.centre;
    return AXES.map((a, i) => {
      const world = add(c, mul(a.v, h[i]));
      const s2 = this.toScreen(world);
      return s2 ? { axis: i, v: a.v, col: a.col, world, screen: s2 } : null;
    }).filter(Boolean);
  }

  pickCropHandle(mx, my) {
    const tol = 11 * this.hitScale();
    for (const hh of this.cropHandles()) {
      if (Math.hypot(mx - hh.screen.x, my - hh.screen.y) < tol) return hh;
    }
    return null;
  }

  /** Resize one axis by however far the cursor moved along that axis on screen. */
  dragCropSize(mx, my) {
    const d = this.drag;
    const i = d.axisIndex;
    const c = this.crop.centre;
    const a = AXES[i].v;
    const o = this.toScreen(c);
    const tip = this.toScreen(add(c, a));
    if (!o || !tip) return;
    const sx = tip.x - o.x, sy = tip.y - o.y;
    const slen = Math.hypot(sx, sy);
    if (slen < 1e-3) return;                       // axis points at the camera
    const proj = ((mx - d.x) * sx + (my - d.y) * sy) / slen;   // px along the axis
    const world = proj / slen;                     // 1 world unit == slen px
    d.x = mx; d.y = my;
    const size = this.crop.size.slice();
    // the handle sits at half the size, so it moves half as fast as the face
    size[i] = Math.max(0.02, size[i] + world * 2);
    if (this.crop.shape === "sphere") { size[1] = size[0] = size[2] = size[i]; }
    this.crop.size = size;
  }

  /** Canvas pixels per on-screen pixel.
   *
   * Hit tolerances have to be expressed in screen space. The canvas backing
   * store is a fixed size while the graph zoom scales how big it appears, so a
   * radius measured in canvas pixels shrinks as you zoom out - at the 0.32 zoom
   * these workflows open at, a 7px arrow target becomes about two real pixels
   * and is essentially impossible to click.
   */
  hitScale() {
    const b = this.canvas.getBoundingClientRect();
    return b.width > 1 ? this.canvas.width / b.width : 1;
  }

  pick(mx, my) {
    let best = -1, bestD = 18 * this.hitScale();
    for (let i = 0; i < 3; i++) {
      const s = this.handleScreen(i);
      if (!s) continue;
      const d = Math.hypot(mx - s.x, my - s.y);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // --- translate gizmo -----------------------------------------------------

  toScreen(world) {
    const p = project(this.viewProj(), world);
    if (p.w <= 0) return null;
    return { x: (p.x / p.w * 0.5 + 0.5) * this.canvas.width,
             y: (0.5 - p.y / p.w * 0.5) * this.canvas.height };
  }

  /** Which gizmos are on screen: the orbit pivot, plus the selected light. */
  gizmos() {
    // Order matters: pickAxis takes the first hit, and these gizmos routinely sit
    // on top of each other - the crop starts at the scene centre, which is also
    // where the orbit pivot begins. Crop and light are the ones you came to move,
    // so they win; the pivot is the fallback.
    const out = [];
    if (this.crop) out.push({ kind: "crop", origin: this.crop.centre.slice() });
    if (this.selected >= 0 && this.lights[this.selected])
      out.push({ kind: "light", i: this.selected, origin: this.handleWorld(this.selected) });
    // The pivot gizmo is hidden while the crop is shown. Both start at the scene
    // centre, so you would be looking at two identical XYZ gizmos stacked on the
    // same point with no way to tell which one you are about to grab. Panning
    // still works by dragging, so nothing is lost.
    if (!this.crop) out.push({ kind: "pivot", origin: this.centre.slice() });
    return out;
  }

  axisArms(origin) {
    const scale = this.radius * 0.42;
    const o = this.toScreen(origin);
    if (!o) return null;
    return AXES.map((a) => {
      const tip = this.toScreen(add(origin, mul(a.v, scale)));
      return tip ? { ...a, o, tip, scale } : null;
    }).filter(Boolean);
  }

  pickAxis(mx, my) {
    const tol = 10 * this.hitScale();          // ~10 real pixels at any zoom
    for (const g of this.gizmos()) {
      const arms = this.axisArms(g.origin);
      if (!arms) continue;
      for (const a of arms) {
        if (distToSeg(mx, my, a.o.x, a.o.y, a.tip.x, a.tip.y) < tol) {
          return { ...g, axis: a.v, col: a.col, arm: a };
        }
      }
    }
    return null;
  }

  /** Move the dragged gizmo along its axis by the cursor's motion. */
  dragAxis(mx, my) {
    const d = this.drag;
    const arms = this.axisArms(d.origin);
    if (!arms) return;
    const arm = arms.find((a) => a.v === d.axis) || arms[0];
    const sx = arm.tip.x - arm.o.x, sy = arm.tip.y - arm.o.y;
    const slen = Math.hypot(sx, sy);
    if (slen < 1e-3) return;                      // axis points at the camera
    const proj = ((mx - d.x) * sx + (my - d.y) * sy) / slen;   // px along the arm
    const world = mul(d.axis, (proj / slen) * arm.scale);
    d.x = mx; d.y = my;

    if (d.kind === "crop" && this.crop) {
      this.crop.centre = add(this.crop.centre, world);
      d.origin = this.crop.centre.slice();
    } else if (d.kind === "pivot") {
      this.centre = add(this.centre, world);
      d.origin = this.centre.slice();
    } else {
      // Genuine positional move: the handle travels along the axis and keeps
      // wherever that puts it. Re-projecting onto a fixed sphere - which is what
      // this did first - meant an X drag barely shifted anything, so the arrows
      // felt dead. Direction and distance are both derived from the new spot.
      const moved = add(d.origin, world);
      const off = sub(moved, this.sceneCentre);
      const dist = len(off);
      if (dist < this.radius * 0.05) return;      // don't let it collapse into the subject
      const yp = yawPitchFromDir(off);
      const l = this.lights[d.i];
      l.yaw = Math.round(yp.yaw * 10) / 10;
      l.pitch = Math.round(yp.pitch * 10) / 10;
      l.dist = Math.min(dist, this.radius * 40);
      d.origin = this.handleWorld(d.i);
    }
  }

  sync() {
    // the graph renders from the *committed* camera, not the live preview
    const cam = this.committed || this.liveCamera();
    this.stateWidget.value = JSON.stringify({
      camera: { position: cam.position, target: cam.target, fov: cam.fov },
      lights: this.lights,
      crop: this.crop ? { centre: this.crop.centre, size: this.crop.size } : undefined,
    });
    // Invalidating the graph canvas on every pointermove makes LiteGraph re-lay
    // the DOM widgets out mid-drag, which is what let a transient measurement
    // shrink the buffer. The widget value is already set; the repaint can wait
    // until the drag ends.
    if (!this.drag) this.node.graph?.setDirtyCanvas(true, true);
  }

  bind() {
    const c = this.canvas;
    const xy = (e) => {
      const b = c.getBoundingClientRect();
      if (b.width > 1 && b.height > 1)
        return [((e.clientX - b.left) / b.width) * c.width,
                ((e.clientY - b.top) / b.height) * c.height];
      return [(e.offsetX || 0), (e.offsetY || 0)];
    };
    c.addEventListener("contextmenu", (e) => e.preventDefault());
    c.addEventListener("pointerdown", (e) => {
      const [mx, my] = xy(e);
      // middle button pans too: some setups swallow right-drag for their own
      // context menu before it ever reaches us
      const wantsPan = e.button === 1 || e.button === 2 || e.shiftKey;
      if (!wantsPan && e.button === 0 && this.crop) {
        const rh = this.pickCropHandle(mx, my);
        if (rh) {
          this.drag = { mode: "cropsize", axisIndex: rh.axis, x: mx, y: my };
          try { c.setPointerCapture(e.pointerId); } catch (err) { /* synthetic */ }
          c.style.cursor = "nwse-resize";
          this.draw();
          e.stopPropagation(); e.preventDefault();
          return;
        }
      }
      // A fat grab dot at the crop centre. The axis arrows are thin targets, and
      // free movement in the view plane is what you want most of the time.
      if (!wantsPan && e.button === 0 && this.crop) {
        const cs = this.toScreen(this.crop.centre);
        if (cs && Math.hypot(mx - cs.x, my - cs.y) < 16 * this.hitScale()) {
          this.drag = { mode: "cropfree", x: mx, y: my };
          try { c.setPointerCapture(e.pointerId); } catch (err) { /* synthetic */ }
          c.style.cursor = "grabbing";
          this.draw();
          e.stopPropagation(); e.preventDefault();
          return;
        }
      }
      // A pan gesture must never grab a light or an arrow, or shift-dragging
      // over a handle silently moves the light instead of the camera.
      const ax = (!wantsPan && e.button === 0) ? this.pickAxis(mx, my) : null;
      if (ax) {
        this.drag = { mode: "axis", kind: ax.kind, i: ax.i, axis: ax.axis,
                      origin: ax.origin, x: mx, y: my };
      } else {
        const hit = (!wantsPan && e.button === 0) ? this.pick(mx, my) : -1;
        if (hit >= 0) this.selected = hit;
        else if (!wantsPan && e.button === 0) this.selected = -1;
        this.drag = { mode: hit >= 0 ? "light" : (wantsPan ? "pan" : "orbit"),
                      light: hit, x: mx, y: my };
      }
      try { c.setPointerCapture(e.pointerId); } catch (err) { /* synthetic pointer */ }
      c.style.cursor = this.drag.mode === "orbit" ? "move" : "grabbing";
      this.draw();
      e.stopPropagation(); e.preventDefault();
    });
    c.addEventListener("pointermove", (e) => {
      const [mx, my] = xy(e);
      if (!this.drag) {
        const overAxis = !!this.pickAxis(mx, my);
        const h = this.pick(mx, my);
        if (h !== this.hover || overAxis !== this.hoverAxis) {
          this.hover = h; this.hoverAxis = overAxis;
          c.style.cursor = overAxis ? "crosshair" : (h >= 0 ? "grab" : "default");
          this.draw();
        }
        return;
      }
      if (this.drag.mode === "axis") {
        this.dragAxis(mx, my);
        this.sync(); this.refreshButtons(); this.draw();
        e.stopPropagation(); e.preventDefault();
        return;
      }
      if (this.drag.mode === "cropsize" && this.crop) {
        this.dragCropSize(mx, my);
        this.sync(); this.draw();
        e.stopPropagation(); e.preventDefault();
        return;
      }
      if (this.drag.mode === "cropfree" && this.crop) {
        // slide the region across the view plane at its own depth
        const eye = this.eye();
        const fwd = norm(sub(this.centre, eye));
        const right = norm(cross(fwd, [0, 1, 0]));
        const up = cross(right, fwd);
        const depth = Math.max(dot(sub(this.crop.centre, eye), fwd), 1e-3);
        const k = (2 * depth * Math.tan((this.fov * Math.PI) / 360)) / this.canvas.height;
        this.crop.centre = add(this.crop.centre,
          add(mul(right, (mx - this.drag.x) * k), mul(up, -(my - this.drag.y) * k)));
        this.drag.x = mx; this.drag.y = my;
        this.sync(); this.draw();
        e.stopPropagation(); e.preventDefault();
        return;
      }
      const dx = mx - this.drag.x, dy = my - this.drag.y;
      this.drag.x = mx; this.drag.y = my;
      if (this.drag.mode === "orbit") {
        this.yaw -= dx * 0.35;
        this.pitch = Math.max(-89, Math.min(89, this.pitch + dy * 0.35));
      } else if (this.drag.mode === "pan") {
        const f = norm(sub(this.centre, this.eye()));
        const r = norm(cross(f, [0, 1, 0]));
        const u = cross(r, f);
        const k = (this.dist * Math.tan((this.fov * Math.PI) / 360) * 2) / this.canvas.height;
        this.centre = add(this.centre, add(mul(r, -dx * k), mul(u, dy * k)));
      } else {
        // drag a light: aim it from the subject toward the cursor's ray
        const i = this.drag.light;
        const vp = this.viewProj();
        const eye = this.eye();
        const ndcX = (mx / this.canvas.width) * 2 - 1;
        const ndcY = 1 - (my / this.canvas.height) * 2;
        const anchor = this.sceneCentre;
        const fwd = norm(sub(anchor, eye));
        const right = norm(cross(fwd, [0, 1, 0]));
        const up = cross(right, fwd);
        const t = Math.tan((this.fov * Math.PI) / 360);
        const aspect = this.canvas.width / Math.max(this.canvas.height, 1);
        const ray = norm(add(fwd, add(mul(right, ndcX * t * aspect), mul(up, ndcY * t))));
        // intersect the ray with the sphere the handles live on
        const oc = sub(eye, anchor);
        const R = this.lights[i].dist > 0 ? this.lights[i].dist : this.radius * 1.45;
        const b = 2 * dot(oc, ray);
        const cq = dot(oc, oc) - R * R;
        const disc = b * b - 4 * cq;
        let hitPoint;
        if (disc >= 0) {
          const s = (-b - Math.sqrt(disc)) / 2;
          hitPoint = add(eye, mul(ray, s > 0 ? s : (-b + Math.sqrt(disc)) / 2));
        } else {
          // ray misses: fall back to the closest point on the sphere
          const closest = add(eye, mul(ray, Math.max(dot(sub(anchor, eye), ray), 0)));
          hitPoint = add(anchor, mul(norm(sub(closest, anchor)), R));
        }
        const yp = yawPitchFromDir(sub(hitPoint, anchor));
        this.lights[i].yaw = Math.round(yp.yaw * 10) / 10;
        this.lights[i].pitch = Math.round(yp.pitch * 10) / 10;
      }
      this.sync();
      this.refreshButtons();
      this.draw();
      e.stopPropagation(); e.preventDefault();
    });
    const end = (e) => {
      if (!this.drag) return;
      this.drag = null;
      c.style.cursor = "default";
      try { c.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
      this.sync();            // now that the drag is over, repaint the graph once
      this.refreshButtons();
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("dblclick", (e) => {
      const [mx, my] = xy(e);
      const i = this.pick(mx, my);
      if (i < 0) return;
      this.lights[i].on = !this.lights[i].on;
      this.sync(); this.draw();
      e.stopPropagation(); e.preventDefault();
    });
    c.addEventListener("wheel", (e) => {
      this.dist = Math.max(this.radius * 0.05,
                           Math.min(this.radius * 60, this.dist * (1 + Math.sign(e.deltaY) * 0.12)));
      this.sync(); this.refreshButtons(); this.draw();
      e.stopPropagation(); e.preventDefault();
    }, { passive: false });
  }

  draw() {
    const gl = this.gl, c = this.canvas;
    if (!gl || !c.width || !c.height) return;
    gl.viewport(0, 0, c.width, c.height);
    gl.clearColor(this.bg[0], this.bg[1], this.bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    const vp = this.viewProj();
    if (this.count) {
      gl.useProgram(this.prog);
      gl.uniformMatrix4fv(this.uViewProj, false, new Float32Array(vp));
      gl.uniform1f(this.uPointSize, this.pointSize);
      gl.uniform1f(this.uViewportH, c.height);
      gl.bindVertexArray(this.vao);
      gl.drawArrays(gl.POINTS, 0, this.count);
      gl.bindVertexArray(null);
    }
    this.drawOverlay();
  }

  // handles are drawn on a 2D canvas stacked over the GL one
  drawOverlay() {
    if (!this.ov || this.ov.width !== this.canvas.width || this.ov.height !== this.canvas.height) {
      if (!this.ov) {
        this.ov = document.createElement("canvas");
        this.ov.style.cssText =
          "position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;";
        this.stage.appendChild(this.ov);
      }
      this.ov.width = this.canvas.width; this.ov.height = this.canvas.height;
    }
    const g = this.ov.getContext("2d");
    g.clearRect(0, 0, this.ov.width, this.ov.height);
    const centre = project(this.viewProj(), this.sceneCentre);
    const cs = centre.w > 0 ? {
      x: (centre.x / centre.w * 0.5 + 0.5) * this.ov.width,
      y: (0.5 - centre.y / centre.w * 0.5) * this.ov.height } : null;

    // where the render camera actually sits, so a committed framing stays
    // visible while you orbit away from it
    if (this.committed && this.cameraDirty()) {
      const p = project(this.viewProj(), this.committed.position);
      if (p.w > 0) {
        const px = (p.x / p.w * 0.5 + 0.5) * this.ov.width;
        const py = (0.5 - p.y / p.w * 0.5) * this.ov.height;
        if (cs) {
          g.beginPath(); g.moveTo(px, py); g.lineTo(cs.x, cs.y);
          g.strokeStyle = "rgba(120,170,255,.45)";
          g.setLineDash([5, 4]); g.lineWidth = 1; g.stroke(); g.setLineDash([]);
        }
        g.beginPath();
        g.rect(px - 6, py - 4.5, 12, 9);
        g.fillStyle = "rgba(60,90,150,.75)";
        g.fill();
        g.strokeStyle = "rgba(150,190,255,.95)"; g.lineWidth = 1.5; g.stroke();
        g.fillStyle = "rgba(180,205,255,.95)";
        g.font = "9px sans-serif"; g.textAlign = "center";
        g.fillText("render cam", px, py - 9);
      }
    }

    for (let i = 0; i < 3; i++) {
      const s = this.handleScreen(i);
      if (!s) continue;
      const lt = this.lights[i];
      const col = lt.on ? lt.color : "rgba(130,130,140,.4)";

      // An arrow from the handle toward the subject: the shaft shows where the
      // light is, the head shows which way it throws - the same read as an axis
      // gizmo, and it stays legible when the handle is edge-on to the camera.
      if (cs) {
        const dx = cs.x - s.x, dy = cs.y - s.y;
        const L = Math.hypot(dx, dy);
        if (L > 22) {
          const ux = dx / L, uy = dy / L;
          const start = 9;                       // clear of the handle disc
          const tipT = Math.min(L - 6, L * 0.72);
          const headLen = 11, headW = 6.5;
          const tipX = s.x + ux * tipT, tipY = s.y + uy * tipT;
          const baseX = tipX - ux * headLen, baseY = tipY - uy * headLen;

          g.beginPath();
          g.moveTo(s.x + ux * start, s.y + uy * start);
          g.lineTo(baseX, baseY);
          g.strokeStyle = col;
          g.globalAlpha = lt.on ? 0.95 : 0.4;
          g.lineWidth = 2.5; g.lineCap = "round";
          g.stroke();

          g.beginPath();                          // solid arrowhead
          g.moveTo(tipX, tipY);
          g.lineTo(baseX - uy * headW, baseY + ux * headW);
          g.lineTo(baseX + uy * headW, baseY - ux * headW);
          g.closePath();
          g.fillStyle = col;
          g.fill();
          g.globalAlpha = 1;
        }
      }

      g.beginPath();
      g.arc(s.x, s.y, this.hover === i ? 9 : 7, 0, Math.PI * 2);
      g.fillStyle = col;
      g.fill();
      g.lineWidth = 2;
      g.strokeStyle = lt.on ? "rgba(255,255,255,.9)" : "rgba(255,255,255,.35)";
      g.stroke();
      g.fillStyle = lt.on ? "rgba(255,255,255,.9)" : "rgba(255,255,255,.4)";
      g.font = "bold 10px sans-serif"; g.textAlign = "center";
      g.fillText(SLOTS[i] + (this.selected === i ? " *" : ""), s.x, s.y - 13);
    }

    // the crop region, so you can see exactly what will survive the cut
    if (this.crop) {
      const c = this.crop.centre, hv = this.cropHalf(), h = hv[0];
      g.strokeStyle = "rgba(255,215,120,.95)";
      g.lineWidth = 1.6;
      if (this.crop.shape === "sphere") {
        // three great circles read as a sphere without needing a mesh
        for (let axis = 0; axis < 3; axis++) {
          g.beginPath();
          let started = false;
          for (let k = 0; k <= 48; k++) {
            const a = (k / 48) * Math.PI * 2;
            const o = [Math.cos(a) * h, Math.sin(a) * h, 0];
            const p3 = axis === 0 ? [c[0], c[1] + o[0], c[2] + o[1]]
                     : axis === 1 ? [c[0] + o[0], c[1], c[2] + o[1]]
                                  : [c[0] + o[0], c[1] + o[1], c[2]];
            const s = this.toScreen(p3);
            if (!s) { started = false; continue; }
            if (!started) { g.moveTo(s.x, s.y); started = true; } else g.lineTo(s.x, s.y);
          }
          g.stroke();
        }
      } else {
        const corner = (i) => [c[0] + (i & 1 ? hv[0] : -hv[0]),
                               c[1] + (i & 2 ? hv[1] : -hv[1]),
                               c[2] + (i & 4 ? hv[2] : -hv[2])];
        const EDGES = [[0,1],[2,3],[4,5],[6,7],[0,2],[1,3],[4,6],[5,7],
                       [0,4],[1,5],[2,6],[3,7]];
        for (const [a, b2] of EDGES) {
          const p1 = this.toScreen(corner(a)), p2 = this.toScreen(corner(b2));
          if (!p1 || !p2) continue;
          g.beginPath(); g.moveTo(p1.x, p1.y); g.lineTo(p2.x, p2.y); g.stroke();
        }
      }
      // square resize handles, one per axis, sitting on the +face
      for (const hh of this.cropHandles()) {
        const r2 = 6 * this.hitScale();
        g.beginPath();
        g.rect(hh.screen.x - r2, hh.screen.y - r2, r2 * 2, r2 * 2);
        g.fillStyle = hh.col; g.fill();
        g.lineWidth = 1.5; g.strokeStyle = "rgba(255,255,255,.85)"; g.stroke();
        const s3 = (this.crop.size || [0, 0, 0])[hh.axis];
        g.fillStyle = "rgba(255,255,255,.85)";
        g.font = "9px sans-serif"; g.textAlign = "center";
        g.fillText(AXES[hh.axis].name + " " + s3.toFixed(2),
                   hh.screen.x, hh.screen.y - r2 - 4);
      }

      const cs2 = this.toScreen(c);
      if (cs2) {
        const rr = 9 * this.hitScale();          // the fat grab dot
        g.beginPath(); g.arc(cs2.x, cs2.y, rr, 0, Math.PI * 2);
        g.fillStyle = "rgba(255,215,120,.9)"; g.fill();
        g.lineWidth = 2; g.strokeStyle = "rgba(40,30,10,.85)"; g.stroke();
        g.fillStyle = "rgba(255,225,150,.95)";
        g.font = "bold 10px sans-serif"; g.textAlign = "center";
        g.fillText("crop", cs2.x, cs2.y - rr - 5);
      }
    }

    // XYZ translate arrows: on the orbit pivot always, and on the selected light
    for (const gz of this.gizmos()) {
      const arms = this.axisArms(gz.origin);
      if (!arms) continue;
      for (const a of arms) {
        const dx = a.tip.x - a.o.x, dy = a.tip.y - a.o.y;
        const L = Math.hypot(dx, dy);
        if (L < 14) continue;                    // axis nearly parallel to view
        const ux = dx / L, uy = dy / L;
        const head = 10, hw = 5.5;
        const bx = a.tip.x - ux * head, by = a.tip.y - uy * head;
        g.beginPath();
        g.moveTo(a.o.x, a.o.y); g.lineTo(bx, by);
        g.strokeStyle = a.col; g.lineWidth = 2.5; g.lineCap = "round"; g.stroke();
        g.beginPath();
        g.moveTo(a.tip.x, a.tip.y);
        g.lineTo(bx - uy * hw, by + ux * hw);
        g.lineTo(bx + uy * hw, by - ux * hw);
        g.closePath(); g.fillStyle = a.col; g.fill();
        g.fillStyle = a.col; g.font = "bold 10px sans-serif"; g.textAlign = "center";
        g.fillText(a.name, a.tip.x + ux * 8, a.tip.y + uy * 8 + 3);
      }
      g.beginPath();
      g.arc(arms[0].o.x, arms[0].o.y, 3, 0, Math.PI * 2);
      g.fillStyle = gz.kind === "pivot" ? "rgba(220,220,230,.9)" : "rgba(255,255,255,.85)";
      g.fill();
    }
  }

  /**
   * Match the backing store to however large CSS has actually made the canvas.
   * Never sets the display size - that is the stylesheet's job - so a momentary
   * zero-width layout during a node drag cannot shrink the viewport for good.
   * 16:9 throughout, to match an HD render: the viewport exists to decide
   * framing, so its aspect has to equal the render's.
   */
  /**
   * Size from the NODE's width, which is authoritative and stable. The measured
   * DOM box is unusable for this: it collapses transiently while LiteGraph
   * re-lays the graph out (every frame of a drag), and it is scaled by the graph
   * zoom, so trusting it either locks in a tiny buffer or fights the zoom.
   * 16:9 throughout, to match an HD render.
   */
  resize() {
    const nodeW = this.node?.size?.[0];
    const basis = nodeW > 60 ? nodeW - 30 : 480;
    const cssW = Math.max(300, Math.round(basis));
    const cssH = Math.round((cssW * 9) / 16);
    // definite display size, so nothing depends on the container resolving one
    if (this.canvas.style.width !== cssW + "px") {
      this.canvas.style.width = cssW + "px";
      this.canvas.style.height = cssH + "px";
      this.stage.style.width = cssW + "px";
      this.stage.style.height = cssH + "px";
    }
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(cssW * dpr), height = Math.round(cssH * dpr);
    if (this.canvas.width === width && this.canvas.height === height) { this.draw(); return; }
    this.canvas.width = width; this.canvas.height = height;
    this.draw();
  }
}

app.registerExtension({
  name: "SplatRelight.Viewport",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "SplatViewport") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const rv = onNodeCreated?.apply(this, arguments);
      const st = this.widgets?.find((w) => w.name === "view_state");
      if (!st) return rv;
      // A multiline STRING input builds a real <textarea>. Setting widget.type
      // to "hidden" stops LiteGraph reserving space but leaves that element in
      // the DOM, where it renders over the node - so hide the element too, and
      // again after the frontend has finished creating it.
      const conceal = () => {
        st.type = "hidden";
        st.hidden = true;
        st.computeSize = () => [0, -4];
        const el = st.element;
        if (el) {
          el.style.display = "none";
          el.style.pointerEvents = "none";
          if (el.parentElement && el.parentElement !== this.domElement) {
            el.parentElement.style.display = "none";
          }
        }
      };
      conceal();
      requestAnimationFrame(conceal);
      setTimeout(conceal, 250);

      const vp = new Viewport(this, st);
      this.splatViewport = vp;
      this.addDOMWidget("splat_viewport", "SPLAT_VIEWPORT", vp.el, {
        serialize: false, hideOnZoom: false, getMinHeight: () => 300,
      });
      this.size[0] = Math.max(this.size[0], 720);
      this.size[1] = Math.max(this.size[1], 660);
      requestAnimationFrame(() => vp.resize());
      return rv;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const r = onResize?.apply(this, arguments);
      this.splatViewport?.resize();
      return r;
    };

    // the node reports its payload after each run; that is when we load geometry
    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (message) {
      const r = onExecuted?.apply(this, arguments);
      const info = message?.splat_viewport?.[0];
      if (info && this.splatViewport) {
        this.splatViewport.resize();
        this.splatViewport.load(info);
      }
      return r;
    };
  },
});
