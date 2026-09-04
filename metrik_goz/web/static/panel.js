/*
  metrik-goz panel — the browser side.

  The flow is three steps: photo → reference → object. There is no mode button;
  dragging on the canvas draws whatever step you are on. If the reference is not
  complete it draws the reference, otherwise the object box.

  Nothing is measured here. The canvas collects pixel coordinates, the server
  measures. Writing the same computation in two places means that one day the two
  will drift apart.
*/
"use strict";

const DATA = JSON.parse(document.getElementById("server-data").textContent);
const $ = (id) => document.getElementById(id);
const STYLE = getComputedStyle(document.documentElement);
const color = (name, fallback) => (STYLE.getPropertyValue(name).trim() || fallback);

const REF_RULES = {
  scale: { points: 2, shape: "line",
    hint: "Drag between the two ends of the reference — if it is round, across its widest part.",
    note: "A coin is the handiest: being round, its diameter is its diameter at any angle. " +
          "Put it next to the object you are measuring, on top of it if you can." },
  rectangle: { points: 4, shape: "quad",
    hint: "Click the four corners of the rectangle in order: top left → top right → bottom right → bottom left.",
    note: "Four corners DO correct perspective. In a tilted photo this is the only right way." },
  aruco: { points: 4, shape: "quad", automatic: true,
    hint: "Searching for the ArUco marker…",
    note: "The most accurate route: corners are read at sub-pixel accuracy and perspective is corrected." },
};

const state = {
  image: null, img: null,
  ref: { type: "scale", points: [], label: null },
  object: { corners: [] },
  forceReference: false,
  view: { z: 1, x: 0, y: 0 },
  result: null, demo: null,
  drawing: null, dragging: null, panning: null,
  space: false, busy: false,
};

const canvas = $("canvas");
const ctx = canvas.getContext("2d");
const magnifier = $("magnifier");
const mctx = magnifier.getContext("2d");
const wrap = $("canvas-wrap");

const refRule = () => REF_RULES[state.ref.type];
const refDone = () => state.ref.points.length === refRule().points;
const objectDone = () => state.object.corners.length === 4;
const target = () => (!refDone() || state.forceReference ? "reference" : "object");

/* ================================================================ view */
function resizeCanvas() {
  const dpr = window.devicePixelRatio || 1;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  if (!w || !h) return;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function fitView() {
  if (!state.image) return;
  const w = wrap.clientWidth, h = wrap.clientHeight;
  const z = Math.min(w / state.image.width, h / state.image.height) * 0.94;
  state.view = { z, x: (w - state.image.width * z) / 2,
                 y: (h - state.image.height * z) / 2 };
  zoomBadge();
  draw();
}

function zoomBy(factor, centerCss) {
  if (!state.image) return;
  const view = state.view;
  const next = Math.min(40, Math.max(0.02, view.z * factor));
  const m = centerCss || { x: wrap.clientWidth / 2, y: wrap.clientHeight / 2 };
  view.x = m.x - (m.x - view.x) * (next / view.z);   // keep the point under the cursor fixed
  view.y = m.y - (m.y - view.y) * (next / view.z);
  view.z = next;
  zoomBadge();
  draw();
}

const zoomBadge = () => { $("zoom-badge").textContent = Math.round(state.view.z * 100) + "%"; };
const toScreen = (p) => ({ x: p[0] * state.view.z + state.view.x,
                           y: p[1] * state.view.z + state.view.y });
const cssPosition = (event) => {
  const rect = canvas.getBoundingClientRect();
  return { x: event.clientX - rect.left, y: event.clientY - rect.top };
};
const toImage = (c) => [(c.x - state.view.x) / state.view.z,
                        (c.y - state.view.y) / state.view.z];

/* ================================================================ drawing */
function draw() {
  const w = wrap.clientWidth, h = wrap.clientHeight;
  ctx.clearRect(0, 0, w, h);
  if (!state.img) return;

  const view = state.view;
  ctx.save();
  ctx.translate(view.x, view.y);
  ctx.scale(view.z, view.z);
  ctx.imageSmoothingEnabled = view.z < 3;
  ctx.drawImage(state.img, 0, 0);
  ctx.restore();

  const blue = color("--blue", "#2a78d6");
  const orange = color("--orange", "#eb6834");
  drawShape(state.ref.points, blue, refRule().shape, refLabel());
  drawShape(state.object.corners, orange, "quad", null);
  if (state.drawing) drawShape(state.drawing.points, state.drawing.color,
                               state.drawing.shape, null, true);
  drawEdgeSizes();
}

function refLabel() {
  if (state.ref.type !== "scale" || state.ref.points.length < 2) return null;
  const mm = Number($("length-mm").value);
  return isFinite(mm) ? number(mm, 2) + " mm" : null;
}

function drawShape(points, stroke, shape, label, temporary) {
  if (!points.length) return;
  const p = points.map(toScreen);
  ctx.save();
  ctx.strokeStyle = stroke;
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  if (temporary) ctx.setLineDash([6, 4]);

  if (p.length > 1) {
    ctx.beginPath();
    ctx.moveTo(p[0].x, p[0].y);
    for (let i = 1; i < p.length; i++) ctx.lineTo(p[i].x, p[i].y);
    if (shape === "quad" && p.length > 2) {
      ctx.closePath();
      ctx.fillStyle = stroke + "1f";
      ctx.fill();
    }
    ctx.stroke();
  }
  ctx.setLineDash([]);

  if (shape === "line" && p.length === 2) {
    // Perpendicular ticks at the ends: make it clear where the measurement stops.
    const dx = p[1].x - p[0].x, dy = p[1].y - p[0].y;
    const n = Math.hypot(dx, dy) || 1;
    const ux = (-dy / n) * 7, uy = (dx / n) * 7;
    [p[0], p[1]].forEach((q) => {
      ctx.beginPath();
      ctx.moveTo(q.x - ux, q.y - uy);
      ctx.lineTo(q.x + ux, q.y + uy);
      ctx.stroke();
    });
    if (label) textBox(label, (p[0].x + p[1].x) / 2, (p[0].y + p[1].y) / 2 - 16, stroke);
  }
  if (!temporary) p.forEach((q) => drawHandle(q, stroke));
  ctx.restore();
}

function drawHandle(p, fill) {
  ctx.beginPath();
  ctx.arc(p.x, p.y, 5.5, 0, Math.PI * 2);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 2;
  ctx.strokeStyle = color("--surface", "#fff");
  ctx.stroke();
}

function textBox(text, x, y, fill) {
  ctx.save();
  ctx.font = "600 12px ui-monospace, Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  const w = ctx.measureText(text).width;
  ctx.fillStyle = color("--surface", "#fff");
  ctx.globalAlpha = 0.92;
  ctx.fillRect(x - w / 2 - 5, y - 9, w + 10, 18);
  ctx.globalAlpha = 1;
  ctx.fillStyle = fill;
  ctx.fillText(text, x, y);
  ctx.restore();
}

function drawEdgeSizes() {
  const s = state.result;
  if (!s || s.type !== "box" || !objectDone() || !s.box) return;
  const p = state.object.corners.map(toScreen);
  const orange = color("--orange", "#eb6834");
  s.box.edges_mm.forEach((mm, i) => {
    const a = p[i], b = p[(i + 1) % 4];
    textBox(number(mm) + " mm", (a.x + b.x) / 2, (a.y + b.y) / 2, orange);
  });
}

function drawMagnifier(css) {
  if (!state.img || state.panning) { magnifier.style.display = "none"; return; }
  const ZOOM = 6, SIZE = 132, point = toImage(css), half = SIZE / (2 * ZOOM);
  mctx.save();
  mctx.clearRect(0, 0, SIZE, SIZE);
  mctx.beginPath();
  mctx.arc(SIZE / 2, SIZE / 2, SIZE / 2, 0, Math.PI * 2);
  mctx.clip();
  mctx.fillStyle = color("--surface2", "#eee");
  mctx.fillRect(0, 0, SIZE, SIZE);
  mctx.imageSmoothingEnabled = false;
  mctx.drawImage(state.img, point[0] - half, point[1] - half, 2 * half, 2 * half, 0, 0, SIZE, SIZE);
  mctx.strokeStyle = color("--orange", "#eb6834");
  mctx.lineWidth = 1;
  mctx.beginPath();
  mctx.moveTo(SIZE / 2, SIZE / 2 - 12); mctx.lineTo(SIZE / 2, SIZE / 2 + 12);
  mctx.moveTo(SIZE / 2 - 12, SIZE / 2); mctx.lineTo(SIZE / 2 + 12, SIZE / 2);
  mctx.stroke();
  mctx.restore();
  magnifier.style.display = "block";
  magnifier.style.left = (css.x > wrap.clientWidth - 170 ? css.x - 150 : css.x + 18) + "px";
  magnifier.style.top = Math.min(Math.max(css.y - 66, 6), wrap.clientHeight - 138) + "px";
}

/* ================================================================ canvas events */
function nearbyHandle(css) {
  const lists = [state.object.corners, state.ref.points];
  for (const list of lists) {
    if (list === state.ref.points && refRule().automatic) continue;
    for (let i = list.length - 1; i >= 0; i--) {
      const s = toScreen(list[i]);
      if (Math.hypot(s.x - css.x, s.y - css.y) <= 11) return { list, i };
    }
  }
  return null;
}

canvas.addEventListener("pointerdown", (event) => {
  if (!state.img) return;
  const css = cssPosition(event);
  if (event.button === 1 || event.button === 2 || state.space) {
    state.panning = { css, view: { ...state.view } };
    canvas.classList.add("panning");
    canvas.setPointerCapture(event.pointerId);
    event.preventDefault();
    return;
  }
  if (event.button !== 0) return;

  const handle = nearbyHandle(css);
  if (handle) {
    state.dragging = { ...handle, moved: false };
    canvas.setPointerCapture(event.pointerId);
    return;
  }

  const point = toImage(css);
  if (target() === "reference") {
    if (refRule().automatic) { status("The ArUco corners are found automatically."); return; }
    if (state.ref.type === "scale") {
      state.drawing = { kind: "ref-line", shape: "line", start: point,
                        points: [point, point], color: color("--blue", "#2a78d6") };
      canvas.setPointerCapture(event.pointerId);
    } else {
      // Rectangle corners are clicked one by one: it is a free quadrilateral, not a box.
      if (state.ref.points.length >= 4) state.ref.points.length = 0;
      state.ref.points.push(point);
      state.forceReference = state.ref.points.length < 4;
      changed();
    }
  } else {
    state.drawing = { kind: "object", shape: "quad", start: point,
                      points: boxCorners(point, point),
                      color: color("--orange", "#eb6834") };
    canvas.setPointerCapture(event.pointerId);
  }
});

const boxCorners = (a, b) => [[a[0], a[1]], [b[0], a[1]], [b[0], b[1]], [a[0], b[1]]];

canvas.addEventListener("pointermove", (event) => {
  const css = cssPosition(event);
  if (state.panning) {
    state.view.x = state.panning.view.x + (css.x - state.panning.css.x);
    state.view.y = state.panning.view.y + (css.y - state.panning.css.y);
    draw();
  } else if (state.dragging) {
    state.dragging.list[state.dragging.i] = toImage(css);
    state.dragging.moved = true;
    draw();
  } else if (state.drawing) {
    const p = toImage(css);
    state.drawing.points = state.drawing.shape === "line"
      ? [state.drawing.start, p] : boxCorners(state.drawing.start, p);
    draw();
  }
  drawMagnifier(css);
});

function endDrag(event) {
  if (state.panning) { state.panning = null; canvas.classList.remove("panning"); }

  if (state.drawing) {
    const d = state.drawing;
    state.drawing = null;
    const a = toScreen(d.points[0]), b = toScreen(d.points[d.shape === "line" ? 1 : 2]);
    if (Math.hypot(b.x - a.x, b.y - a.y) < 8) {
      status("Too small — press and drag.");
      draw();
    } else if (d.kind === "ref-line") {
      state.ref.points = d.points;
      state.forceReference = false;
      changed();
    } else {
      state.object.corners = d.points;
      changed();
    }
  }

  if (state.dragging) {
    const moved = state.dragging.moved;
    state.dragging = null;
    if (moved) changed();
  }
  if (event && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
}
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);
canvas.addEventListener("contextmenu", (event) => event.preventDefault());
canvas.addEventListener("pointerleave", () => { magnifier.style.display = "none"; });
canvas.addEventListener("wheel", (event) => {
  if (!state.img) return;
  event.preventDefault();
  zoomBy(Math.exp(-event.deltaY * 0.0022), cssPosition(event));
}, { passive: false });

window.addEventListener("keydown", (event) => {
  const typing = /^(INPUT|SELECT|TEXTAREA)$/.test(document.activeElement.tagName);
  if (event.code === "Space" && !typing) {
    state.space = true; canvas.classList.add("pan"); event.preventDefault(); return;
  }
  if (typing) return;
  if (event.key.toLowerCase() === "f") fitView();
  else if (event.key === "Escape") { state.drawing = null; draw(); }
});
window.addEventListener("keyup", (event) => {
  if (event.code === "Space") { state.space = false; canvas.classList.remove("pan"); }
});
window.addEventListener("resize", resizeCanvas);

/* ---------------------------------------------------------------- files */
["dragenter", "dragover"].forEach((name) => window.addEventListener(name, (event) => {
  if (!event.dataTransfer || ![...event.dataTransfer.types].includes("Files")) return;
  event.preventDefault();
  $("drop").classList.remove("is-hidden");
  $("drop").classList.add("over");
}));
["dragleave", "drop"].forEach((name) => window.addEventListener(name, (event) => {
  if (name === "dragleave" && event.relatedTarget) return;
  $("drop").classList.remove("over");
  if (state.image) $("drop").classList.add("is-hidden");
}));
window.addEventListener("drop", (event) => {
  const file = event.dataTransfer && event.dataTransfer.files[0];
  if (file) { event.preventDefault(); uploadImage(file); }
});
window.addEventListener("paste", (event) => {
  const item = [...(event.clipboardData ? event.clipboardData.items : [])]
    .find((x) => x.type.startsWith("image/"));
  if (item) uploadImage(item.getAsFile());
});

/* ================================================================ interface bindings */
$("choose-file").onclick = $("choose-file-big").onclick =
  $("new-measurement").onclick = () => $("file").click();
$("file").onchange = (event) => event.target.files[0] && uploadImage(event.target.files[0]);
$("fit").onclick = fitView;
$("zoom-in").onclick = () => zoomBy(1.3);
$("zoom-out").onclick = () => zoomBy(1 / 1.3);
$("aruco-find").onclick = findAruco;
$("ref-redraw").onclick = () => {
  state.ref.points = [];
  state.ref.label = null;
  state.forceReference = true;
  changed();
};
$("object-redraw").onclick = () => { state.object.corners = []; changed(); };

// The uncertainty card in the sidebar mirrors the "advanced" settings.
function uncertaintyCard() {
  const confidence = Number($("confidence").value);
  $("side-confidence").textContent = percent(confidence, 0);
  $("side-bar").style.width = `${(confidence * 100).toFixed(0)}%`;
  $("side-note").textContent =
    `σ ${number(Number($("sigma").value), 1)} px · ${$("mc-n").value} MC samples`;
}
["sigma", "confidence", "mc-n"].forEach((id) => $(id).addEventListener("input", uncertaintyCard));

document.querySelectorAll("[data-sample]").forEach((button) =>
  button.onclick = () => loadSample(button.dataset.sample));

$("ref-types").onclick = (event) => {
  const button = event.target.closest("[data-ref]");
  if (!button || button.dataset.ref === state.ref.type) return;
  state.ref = { type: button.dataset.ref, points: [], label: null };
  state.forceReference = true;
  $("sigma").value = button.dataset.ref === "aruco" ? DATA.sigmas.aruco : DATA.sigmas.manual;
  changed();
  if (button.dataset.ref === "aruco" && state.image) findAruco();
};

$("length-name").onchange = () => {
  const option = $("length-name").selectedOptions[0];
  if (option.dataset.mm) $("length-mm").value = option.dataset.mm;
  else $("length-mm").focus();
  changed();
};
$("length-mm").oninput = () => {
  // Once a number is typed by hand the ready-made choice no longer applies.
  const option = $("length-name").selectedOptions[0];
  if (option && option.dataset.mm && Number(option.dataset.mm) !== Number($("length-mm").value)) {
    $("length-name").value = "";
  }
  draw();
};
$("length-mm").onchange = changed;
$("object-name").onchange = () => {
  const name = $("object-name").value;
  const custom = !name;
  $("rect-sizes").hidden = !custom;
  if (!custom && DATA.objects[name]) {
    $("rect-w").value = DATA.objects[name][0];
    $("rect-h").value = DATA.objects[name][1];
  }
  changed();
};
["rect-w", "rect-h", "aruco-edge", "sigma", "confidence", "mc-n"].forEach(
  (id) => { $(id).onchange = changed; });

/* ================================================================ state flow */
let timer = null;

function changed() {
  refreshInterface();
  clearTimeout(timer);
  if (state.image && refDone() && objectDone()) {
    timer = setTimeout(runMeasurement, 60);      // one request once the drag ends
  } else {
    state.result = null;
    clearResult();
    draw();
  }
}

function refreshInterface() {
  document.querySelectorAll("[data-ref]").forEach((button) =>
    button.classList.toggle("selected", button.dataset.ref === state.ref.type));
  ["scale", "rectangle", "aruco"].forEach((name) =>
    $("ref-" + name).hidden = state.ref.type !== name);

  const rule = refRule();
  const hasPhoto = !!state.image;
  const active = !hasPhoto ? "photo" : target() === "reference" ? "ref" : "object";

  [["step-photo", "photo", hasPhoto],
   ["step-ref", "ref", refDone()],
   ["step-object", "object", objectDone()]].forEach(([id, short, done]) => {
    const el = $(id);
    el.classList.toggle("done", done);
    el.classList.toggle("active", active === short && !done);
    $(short + "-check").hidden = !done;
    // The flow list in the sidebar shows the same state.
    const nav = document.querySelector(`[data-flow="${short}"]`);
    if (nav) {
      nav.classList.toggle("done", done);
      nav.classList.toggle("active", active === short && !done);
    }
  });

  $("photo-note").textContent = hasPhoto
    ? `${state.image.name} · ${state.image.width}×${state.image.height} px`
    : "Drag-and-drop and pasting with ⌘V work too.";
  $("choose-file").textContent = hasPhoto ? "Choose another photo…" : "Choose a photo…";

  const refNote = state.ref.label && refDone()
    ? `${state.ref.label}${state.result && state.result.reference.pixel_length
        ? ` · ${Math.round(state.result.reference.pixel_length)} px in the image` : ""}`
    : rule.note;
  $("ref-note").textContent = refNote;
  $("ref-note").classList.toggle("highlight", refDone());
  $("ref-redraw").hidden = !state.ref.points.length || rule.automatic;

  $("object-note").textContent = objectDone()
    ? "You can drag the corners to correct them; the measurement refreshes instantly."
    : "Draw a box over the object you want to measure (press and drag).";
  $("object-redraw").hidden = !objectDone();

  const hint = $("hint");
  if (!hasPhoto) hint.hidden = true;
  else if (target() === "reference" && !rule.automatic) {
    hint.hidden = false;
    hint.textContent = state.ref.type === "rectangle"
      ? `${rule.hint}  (${state.ref.points.length}/4)` : rule.hint;
  } else if (!objectDone()) {
    hint.hidden = false;
    hint.textContent = "Draw a box over the object.";
  } else hint.hidden = true;

  zoomBadge();
  draw();
}

const status = (message) => { $("status-text").textContent = message; };

/* ================================================================ server */
async function request(path, options) {
  const response = await fetch(path, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.error || `Server error (${response.status}).`);
  return body;
}

async function uploadImage(file) {
  status("Uploading the photo…");
  try {
    const form = new FormData();
    form.append("file", file);
    await attachImage(await request("/api/image", { method: "POST", body: form }));
    state.demo = null;
    status("Ready.");
    if (refRule().automatic) findAruco();
  } catch (error) { showError(error.message); status("Upload failed."); }
}

function attachImage(info) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      state.image = info;
      state.img = img;
      state.ref.points = [];
      state.ref.label = null;
      state.object.corners = [];
      state.result = null;
      state.forceReference = false;
      $("drop").classList.add("is-hidden");
      clearResult();
      fitView();
      refreshInterface();
      resolve();
    };
    img.onerror = () => reject(new Error("The photo could not be opened in the browser."));
    img.src = info.url;
  });
}

async function findAruco() {
  if (!state.image) return;
  status("Looking for an ArUco marker…");
  try {
    const found = await request("/api/aruco", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_id: state.image.id,
                             edge_mm: Number($("aruco-edge").value) }),
    });
    state.ref = { type: "aruco", points: found.corners, label: found.label };
    state.forceReference = false;
    $("sigma").value = found.sigma_px;
    status(found.label + " found.");
    changed();
  } catch (error) {
    showError(error.message);
    status("No marker found.");
  }
}

async function loadSample(name) {
  status("Generating the sample scene…");
  try {
    const scene = await request("/api/sample", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await attachImage(scene);
    state.demo = scene;
    document.querySelector(`[data-ref="${scene.reference.type}"]`).classList.add("selected");
    state.ref.type = scene.reference.type;
    if (scene.reference.name) $("length-name").value = scene.reference.name;
    if (scene.reference.length_mm) $("length-mm").value = scene.reference.length_mm;
    $("sigma").value = DATA.sigmas.manual;
    state.ref.points = (scene.hint.reference || []).map((p) => [p[0], p[1]]);
    state.object.corners = (scene.hint.box || []).map((p) => [p[0], p[1]]);
    status(scene.description);
    changed();
  } catch (error) { showError(error.message); status("The sample could not be loaded."); }
}

function requestBody() {
  const ref = { type: state.ref.type };
  if (state.ref.type === "scale") {
    ref.length_mm = Number($("length-mm").value);
    ref.name = $("length-name").value || null;
    ref.points = state.ref.points;
  } else if (state.ref.type === "rectangle") {
    const name = $("object-name").value;
    if (name) { ref.type = "object"; ref.object = name; }
    else {
      ref.width_mm = Number($("rect-w").value);
      ref.height_mm = Number($("rect-h").value);
    }
    ref.corners = state.ref.points;
  } else {
    ref.edge_mm = Number($("aruco-edge").value);
    ref.corners = state.ref.points;
    ref.label = state.ref.label;
  }
  return {
    image_id: state.image.id,
    reference: ref,
    measurement: { type: "box", points: state.object.corners },
    sigma_px: Number($("sigma").value),
    confidence: Number($("confidence").value),
    mc_n: Number($("mc-n").value),
  };
}

async function runMeasurement() {
  if (state.busy) { timer = setTimeout(runMeasurement, 80); return; }
  state.busy = true;
  $("result").classList.add("busy");
  try {
    const s = await request("/api/measure", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(requestBody()),
    });
    state.result = s;
    state.ref.label = s.reference.label;
    if (state.ref.type === "aruco" && s.reference.points)
      state.ref.points = s.reference.points;
    renderResult(s);
    status(`${s.measurements[0].text}  ·  ${s.duration_ms.toFixed(0)} ms`);
    $("ref-note").textContent = `${s.reference.label}` +
      (s.reference.pixel_length ? ` · ${Math.round(s.reference.pixel_length)} px in the image` : "");
    draw();
  } catch (error) {
    state.result = null;
    showError(error.message);
    status("The measurement failed.");
  } finally {
    state.busy = false;
    $("result").classList.remove("busy");
  }
}

/* ================================================================ result */
const number = (v, digits = 1) => v.toLocaleString("en-US",
  { minimumFractionDigits: digits, maximumFractionDigits: digits });
const percent = (ratio, digits = 1) => number(ratio * 100, digits) + "%";

function el(tag, className, content) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (typeof content === "string") node.textContent = content;
  else if (Array.isArray(content)) node.append(...content);
  return node;
}

function clearCards() {
  $("metric-name").textContent = "MEASURE";
  $("metric-value").textContent = "—";
  $("metric-note").textContent = "waiting for a measurement";
  $("margin-value").textContent = "—";
  $("margin-note").textContent = "standard deviation";
  $("interval-value").textContent = "—";
  $("interval-note").textContent = "the end of this interval decides";
  $("interval-track").hidden = true;
  $("ref-value").textContent = "—";
  $("ref-card-note").textContent = "not marked yet";
  $("ref-chips").innerHTML = "";
  $("canvas-chips").hidden = true;
}

function fillCards(s) {
  const m = s.measurement;
  $("metric-name").textContent = m.name.toUpperCase();
  $("metric-value").innerHTML = "";
  $("metric-value").append(document.createTextNode(number(m.value)),
                           el("small", "", m.unit));
  $("metric-note").textContent = `${s.type} · ${m.method}`;

  $("margin-value").innerHTML = "";
  $("margin-value").append(document.createTextNode(`± ${number(m.std)}`),
                           el("small", "", m.unit));
  $("margin-note").textContent = `relative ${percent(m.relative_error, 2)}`;

  $("interval-value").textContent = `${number(m.low)} – ${number(m.high)}`;
  $("interval-note").textContent = `${percent(m.confidence, 0)} confidence · ${m.unit}`;
  // Where the marker sits: the real position of the point estimate in the interval.
  const width = m.high - m.low;
  $("interval-track").hidden = !(width > 0);
  if (width > 0) {
    const ratio = Math.min(1, Math.max(0, (m.value - m.low) / width));
    $("interval-marker").style.left = `${(ratio * 100).toFixed(1)}%`;
  }

  $("ref-value").innerHTML = "";
  $("ref-value").append(document.createTextNode(number(s.homography.scale_mm_px, 3)),
                        el("small", "", "mm/px"));
  $("ref-card-note").textContent = s.reference.label;
  const chips = $("ref-chips");
  chips.innerHTML = "";
  [s.homography.model, `RMS ${number(s.homography.rms_px, 2)} px`,
   `σ ${number(s.reference.sigma_px, 1)} px`]
    .forEach((text) => chips.append(el("span", "chip", text)));

  $("canvas-chips").hidden = false;
  $("chip-sigma").textContent = `σ ${number(s.reference.sigma_px, 1)} px`;
  $("chip-rms").textContent = `RMS ${number(s.homography.rms_px, 2)} px`;
  $("chip-scale").textContent = `${number(s.homography.scale_mm_px, 3)} mm/px`;
}

function clearResult() {
  clearCards();
  $("result-body").innerHTML = "";
  $("result-body").append(el("p", "result-empty",
    !state.image ? "Upload a photo first."
    : !refDone() ? "Mark the reference."
    : "Draw a box over the object."));
}

function showError(text) {
  clearCards();
  $("result-body").innerHTML = "";
  $("result-body").append(el("div", "error-box", text));
}

function measureBlock(m, truth) {
  const block = el("div", "measure");
  block.append(el("div", "name", m.name));
  block.append(el("div", "value", [document.createTextNode(number(m.value)),
                                   el("small", "", m.unit)]));
  block.append(el("div", "margin", `± ${number(m.std)}`));
  block.append(el("div", "interval", `${number(m.low)} – ${number(m.high)}`));
  if (truth != null) {
    const inside = truth >= m.low && truth <= m.high;
    block.append(el("div", "truth" + (inside ? "" : " outside"),
      `true ${number(truth)} ${inside ? "✓" : "✗"}`));
  }
  return block;
}

function renderResult(s) {
  fillCards(s);
  const body = $("result-body");
  body.innerHTML = "";
  const truth = (name) => (state.demo && state.demo.truth && state.demo.truth[name]
    ? state.demo.truth[name].value : null);

  const lengths = s.measurements.filter((m) => m.unit === "mm");
  const pair = el("div", "measure-pair");
  lengths.slice(0, 2).forEach((m) => pair.append(measureBlock(m, truth(m.name))));
  body.append(pair);

  const area = s.measurements.find((m) => m.name === "area");
  const extra = el("div", "extra");
  if (area) {
    const t = truth("area");
    extra.append(el("div", "", [el("b", "", "area "),
      document.createTextNode(`${number(area.value)} ± ${number(area.std)} ${area.unit}` +
        (t != null ? `  (true ${number(t)})` : ""))]));
  }
  if (s.box) {
    const edges = s.box.edges_mm;
    extra.append(el("div", "", [el("b", "", "edges "),
      document.createTextNode(edges.map((v) => number(v)).join(" · ") + " mm")]));
    if (s.box.rectangularity > 0.01) {
      extra.append(el("div", "", [el("b", "", "opposite edge mismatch "),
        document.createTextNode(percent(s.box.rectangularity))]));
    }
  }
  extra.append(el("div", "", [el("b", "", "reference "),
    document.createTextNode(`${s.reference.label} · ${s.homography.model}` +
      ` · ${number(s.homography.scale_mm_px, 3)} mm/px`)]));
  body.append(extra);

  if (s.warnings && s.warnings.length) {
    const list = el("ul", "warning-list collapsed");
    s.warnings.forEach((w) => list.append(el("li", w.level, w.text)));
    body.append(list);
    if (s.warnings.length > 2) {
      const button = el("button", "warning-toggle", `${s.warnings.length - 2} more notes`);
      button.onclick = () => {
        const collapsed = list.classList.toggle("collapsed");
        button.textContent = collapsed ? `${s.warnings.length - 2} more notes` : "show less";
      };
      body.append(button);
    }
  }
}

/* ================================================================ start-up */
new ResizeObserver(resizeCanvas).observe(wrap);
resizeCanvas();
refreshInterface();
uncertaintyCard();
clearResult();
