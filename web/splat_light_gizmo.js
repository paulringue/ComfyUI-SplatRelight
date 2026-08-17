/**
 * Draggable light gizmo for ComfyUI-SplatRelight.
 *
 * No ComfyUI splat viewer lets you move lights inside it - they are all
 * camera-orbit previews - so this adds the control that is missing: a lighting
 * ball you drag handles on, shaded live by the rig you are actually building.
 *
 * Deliberately plain canvas 2D. A three.js viewport would mean bundling ~600 KB
 * into a pack that otherwise has zero dependencies, and for aiming lights a
 * shaded sphere tells you what you need to know - which way each light points
 * and what the falloff looks like - at 60fps with a few thousand pixels.
 *
 * Conventions match relight_core: the direction TO a light is
 *   [cos(pitch)sin(yaw), sin(pitch), cos(pitch)cos(yaw)]
 * and yaw/pitch mean the same thing as they do on Create Camera Info, so the
 * viewer sits at +Z looking down -Z.
 */

import { app } from "../../scripts/app.js";

const SLOTS = ["key", "fill", "rim"];
const BALL_PAD = 10;
// The front hemisphere fills the disc; the back hemisphere is mapped onto an
// annulus just outside it, so a light can be swung behind the subject and its
// handle stays visible and grabbable instead of vanishing under the silhouette.
const BACK_SPAN = 0.45;

function dirFromYawPitch(yawDeg, pitchDeg) {
  const y = (yawDeg * Math.PI) / 180;
  const p = (pitchDeg * Math.PI) / 180;
  return [Math.cos(p) * Math.sin(y), Math.sin(p), Math.cos(p) * Math.cos(y)];
}

function yawPitchFromDir(d) {
  const n = Math.hypot(d[0], d[1], d[2]) || 1;
  const v = [d[0] / n, d[1] / n, d[2] / n];
  return {
    yaw: (Math.atan2(v[0], v[2]) * 180) / Math.PI,
    pitch: (Math.asin(Math.max(-1, Math.min(1, v[1]))) * 180) / Math.PI,
  };
}

function hexToRgb(h) {
  const s = (h || "#ffffff").replace("#", "");
  if (s.length !== 6) return [1, 1, 1];
  return [
    parseInt(s.slice(0, 2), 16) / 255,
    parseInt(s.slice(2, 4), 16) / 255,
    parseInt(s.slice(4, 6), 16) / 255,
  ];
}

const DEFAULTS = [
  { on: true, yaw: 40, pitch: 35 },
  { on: true, yaw: -60, pitch: 10 },
  { on: false, yaw: 180, pitch: 25 },
];

function parseState(raw) {
  try {
    const v = JSON.parse(raw);
    if (Array.isArray(v) && v.length === 3) {
      return v.map((l, i) => ({
        on: !!l.on,
        yaw: Number.isFinite(l.yaw) ? l.yaw : DEFAULTS[i].yaw,
        pitch: Number.isFinite(l.pitch) ? l.pitch : DEFAULTS[i].pitch,
      }));
    }
  } catch (e) {
    /* fall through to defaults */
  }
  return DEFAULTS.map((d) => ({ ...d }));
}

class LightBall {
  constructor(node, widget) {
    this.node = node;
    this.widget = widget;
    this.state = parseState(widget.value);
    this.selected = 0;
    this.dragging = -1;

    const wrap = document.createElement("div");
    wrap.style.cssText =
      "display:flex;flex-direction:column;align-items:center;gap:4px;padding:4px 0;";
    const canvas = document.createElement("canvas");
    canvas.style.cssText = "border-radius:6px;cursor:grab;touch-action:none;";
    const hint = document.createElement("div");
    hint.style.cssText =
      "font-size:10px;line-height:1.35;opacity:.65;text-align:center;font-family:sans-serif;";
    hint.textContent =
      "drag a handle to aim · drag outside the ring = behind · double-click = on/off";
    wrap.appendChild(canvas);
    wrap.appendChild(hint);

    this.el = wrap;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { willReadFrequently: true });
    this.bind();
  }

  // colour / intensity live on sibling widgets so they stay editable as numbers
  lightStyle(i) {
    const c = this.node.widgets?.find((w) => w.name === `${SLOTS[i]}_color`);
    const s = this.node.widgets?.find((w) => w.name === `${SLOTS[i]}_intensity`);
    return {
      color: c ? c.value : "#ffffff",
      intensity: s ? Number(s.value) || 0 : 1,
    };
  }

  sync() {
    this.widget.value = JSON.stringify(this.state);
    this.node.graph?.setDirtyCanvas(true, true);
  }

  // --- geometry -----------------------------------------------------------
  metrics() {
    const w = this.canvas.width;
    // leave room for the back-hemisphere annulus, or its handles get clipped
    const r = (w / 2 - BALL_PAD) / (1 + BACK_SPAN);
    return { cx: w / 2, cy: this.canvas.height / 2, r };
  }

  handlePos(i) {
    const { cx, cy, r } = this.metrics();
    const d = dirFromYawPitch(this.state[i].yaw, this.state[i].pitch);
    const rho = Math.hypot(d[0], d[1]);          // distance from the view axis
    const unit = rho > 1e-6 ? [d[0] / rho, d[1] / rho] : [0, 1];
    // front: radius = rho. back: pushed out into the annulus, reaching
    // 1 + BACK_SPAN when the light is directly behind (rho = 0).
    const k = d[2] >= 0 ? rho : 1 + BACK_SPAN * (1 - rho);
    const px = d[2] >= 0 ? d[0] : unit[0] * k;
    const py = d[2] >= 0 ? d[1] : unit[1] * k;
    return [cx + px * r, cy - py * r, d[2] >= 0];
  }

  pick(mx, my) {
    for (let i = 0; i < 3; i++) {
      const [hx, hy] = this.handlePos(i);
      if (Math.hypot(mx - hx, my - hy) <= 11) return i;
    }
    return -1;
  }

  setFromMouse(i, mx, my) {
    const { cx, cy, r } = this.metrics();
    let x = (mx - cx) / r;
    let y = -(my - cy) / r;
    const len = Math.hypot(x, y);
    let z;
    if (len <= 1) {
      z = Math.sqrt(Math.max(0, 1 - len * len)); // front hemisphere
    } else {
      // outside the disc: swing round behind. rho runs 1 -> 0 across the
      // annulus, so z runs 0 -> -1 and the mapping stays continuous at len = 1.
      const u = Math.min((len - 1) / BACK_SPAN, 1);
      const rho = 1 - u;
      const dir = [x / len, y / len];
      x = dir[0] * rho;
      y = dir[1] * rho;
      z = -Math.sqrt(Math.max(0, 1 - rho * rho));
    }
    const { yaw, pitch } = yawPitchFromDir([x, y, z]);
    this.state[i].yaw = Math.round(yaw * 10) / 10;
    this.state[i].pitch = Math.round(pitch * 10) / 10;
    this.sync();
    this.draw();
  }

  // --- interaction --------------------------------------------------------
  bind() {
    const c = this.canvas;
    // Map a pointer event to canvas pixels. The bounding rect can legitimately
    // come back zero-sized - a collapsed node, a hidden tab, a canvas that has
    // not been laid out yet - and dividing by it yields NaN, which silently
    // swallows the drag. Fall back to the element-relative offsets.
    const xy = (e) => {
      const b = c.getBoundingClientRect();
      if (b.width > 1 && b.height > 1) {
        return [
          ((e.clientX - b.left) / b.width) * c.width,
          ((e.clientY - b.top) / b.height) * c.height,
        ];
      }
      const sx = c.width / (c.clientWidth || c.width);
      const sy = c.height / (c.clientHeight || c.height);
      return [(e.offsetX || 0) * sx, (e.offsetY || 0) * sy];
    };
    c.addEventListener("pointerdown", (e) => {
      const [mx, my] = xy(e);
      const i = this.pick(mx, my);
      if (i < 0) return;
      this.dragging = i;
      this.selected = i;
      c.setPointerCapture(e.pointerId);
      c.style.cursor = "grabbing";
      e.stopPropagation();
      e.preventDefault();
      this.draw();
    });
    c.addEventListener("pointermove", (e) => {
      if (this.dragging < 0) return;
      const [mx, my] = xy(e);
      this.setFromMouse(this.dragging, mx, my);
      e.stopPropagation();
      e.preventDefault();
    });
    const end = (e) => {
      if (this.dragging < 0) return;
      this.dragging = -1;
      c.style.cursor = "grab";
      try {
        c.releasePointerCapture(e.pointerId);
      } catch (err) {
        /* pointer already released */
      }
    };
    c.addEventListener("pointerup", end);
    c.addEventListener("pointercancel", end);
    c.addEventListener("dblclick", (e) => {
      const [mx, my] = xy(e);
      const i = this.pick(mx, my);
      if (i < 0) return;
      this.state[i].on = !this.state[i].on;
      this.sync();
      this.draw();
      e.stopPropagation();
      e.preventDefault();
    });
  }

  // --- drawing ------------------------------------------------------------
  draw() {
    const c = this.canvas;
    const ctx = this.ctx;
    if (!c.width || !c.height) return;
    const { cx, cy, r } = this.metrics();

    ctx.clearRect(0, 0, c.width, c.height);
    ctx.fillStyle = "#15161a";
    ctx.fillRect(0, 0, c.width, c.height);

    // outer edge of the "behind" annulus
    ctx.beginPath();
    ctx.arc(cx, cy, r * (1 + BACK_SPAN), 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(255,255,255,.10)";
    ctx.setLineDash([3, 4]);
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.setLineDash([]);

    // shade the sphere with the actual rig: ambient + sum of N.L per light
    const lights = [];
    for (let i = 0; i < 3; i++) {
      if (!this.state[i].on) continue;
      const st = this.lightStyle(i);
      lights.push({
        d: dirFromYawPitch(this.state[i].yaw, this.state[i].pitch),
        c: hexToRgb(st.color),
        i: st.intensity,
      });
    }
    const size = Math.ceil(r * 2) + 2;
    // Rendered offscreen and composited with drawImage. putImageData *replaces*
    // pixels including alpha rather than blending, so writing it straight to the
    // canvas punches a transparent square through the background.
    if (!this.off || this.off.width !== size) {
      this.off = document.createElement("canvas");
      this.off.width = this.off.height = size;
      this.offCtx = this.off.getContext("2d");
    }
    const img = this.offCtx.createImageData(size, size);
    const px = img.data;
    for (let yy = 0; yy < size; yy++) {
      const ny = 1 - (2 * (yy + 0.5)) / size;
      for (let xx = 0; xx < size; xx++) {
        const nx = (2 * (xx + 0.5)) / size - 1;
        const o = (yy * size + xx) * 4;
        const q = nx * nx + ny * ny;
        if (q > 1) {
          px[o + 3] = 0;
          continue;
        }
        const nz = Math.sqrt(1 - q);
        let R = 0.06, G = 0.065, B = 0.08; // ambient
        for (const L of lights) {
          const ndl = nx * L.d[0] + ny * L.d[1] + nz * L.d[2];
          if (ndl <= 0) continue;
          const k = (ndl * L.i) / Math.PI;
          R += k * L.c[0];
          G += k * L.c[1];
          B += k * L.c[2];
        }
        // Reinhard so a strong key does not just clip to white
        px[o] = 255 * Math.min(1, R / (1 + R)) ** (1 / 2.2) * 1;
        px[o + 1] = 255 * Math.min(1, G / (1 + G)) ** (1 / 2.2);
        px[o + 2] = 255 * Math.min(1, B / (1 + B)) ** (1 / 2.2);
        px[o + 3] = 255;
      }
    }
    this.offCtx.putImageData(img, 0, 0);
    ctx.drawImage(this.off, Math.round(cx - size / 2), Math.round(cy - size / 2));

    // handles
    for (let i = 0; i < 3; i++) {
      const [hx, hy, front] = this.handlePos(i);
      const st = this.lightStyle(i);
      const on = this.state[i].on;
      ctx.beginPath();
      ctx.arc(hx, hy, i === this.selected ? 8 : 6.5, 0, Math.PI * 2);
      ctx.fillStyle = on ? st.color : "rgba(120,120,130,.35)";
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = front ? "rgba(255,255,255,.85)" : "rgba(255,255,255,.35)";
      if (!front) ctx.setLineDash([2, 3]);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = on ? "rgba(255,255,255,.85)" : "rgba(255,255,255,.35)";
      ctx.font = "10px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(SLOTS[i], hx, hy - 12);
    }
  }

  resize(width) {
    const w = Math.max(220, Math.floor(width));
    this.canvas.width = w;
    this.canvas.height = w;
    this.canvas.style.width = w + "px";
    this.canvas.style.height = w + "px";
    this.draw();
  }
}

app.registerExtension({
  name: "SplatRelight.LightGizmo",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== "SplatLightGizmo") return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const rv = onNodeCreated?.apply(this, arguments);
      const hidden = this.widgets?.find((w) => w.name === "lights");
      if (!hidden) return rv;

      // keep the raw JSON as the serialised value, but stop it taking up space
      hidden.type = "hidden";
      hidden.computeSize = () => [0, -4];

      const ball = new LightBall(this, hidden);
      this.splatLightBall = ball;
      this.addDOMWidget("light_gizmo", "SPLAT_LIGHT_GIZMO", ball.el, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 300,
      });

      // redraw when colour / intensity change so the ball reflects the rig
      for (const w of this.widgets || []) {
        if (/_(color|intensity)$/.test(w.name)) {
          const prev = w.callback;
          w.callback = function () {
            const r = prev?.apply(this, arguments);
            ball.draw();
            return r;
          };
        }
      }
      this.size[0] = Math.max(this.size[0], 300);
      this.size[1] = Math.max(this.size[1], 520);
      requestAnimationFrame(() => ball.resize(this.size[0] - 40));
      return rv;
    };

    const onResize = nodeType.prototype.onResize;
    nodeType.prototype.onResize = function (size) {
      const r = onResize?.apply(this, arguments);
      this.splatLightBall?.resize(size[0] - 40);
      return r;
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      const b = this.splatLightBall;
      if (b) {
        b.state = parseState(b.widget.value);
        requestAnimationFrame(() => b.resize(this.size[0] - 40));
      }
      return r;
    };
  },
});
