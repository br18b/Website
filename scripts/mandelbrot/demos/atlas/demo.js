'use strict';

const state = {
  data: null,
  worker: null,
  workerReady: false,
  rendererType: null,
  webgl: null,
  renderToken: 0,
  renderRaf: 0,
  pendingInteracting: false,
  settleTimer: null,
  centerX: 0,
  centerY: 0,
  spanX: 3,
  hover: null,
  selected: null,
  pointer: null,
  drag: null,
  animation: null,
  spatial: null,
  componentsByPeriod: new Map(),
  baseReady: false,
};

const els = {};

function clamp(value, lo, hi) { return Math.max(lo, Math.min(hi, value)); }
function easeInOut(t) { return t < .5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; }

function cleanMinus(text) {
  return String(text).replaceAll('-', '−');
}

function formatScientificHTML(value, significantDigits = 3) {
  if (!Number.isFinite(value)) return 'unknown';
  if (value === 0) return '0';
  const exponent = Math.floor(Math.log10(Math.abs(value)));
  const mantissa = value / Math.pow(10, exponent);
  const mantissaText = cleanMinus(Number(mantissa.toPrecision(significantDigits)).toString());
  return `${mantissaText} × 10<sup>${cleanMinus(exponent)}</sup>`;
}

function formatCompactHTML(value, significantDigits = 3) {
  if (!Number.isFinite(value)) return 'unknown';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude < 0.1 || magnitude >= 1.0e5) {
    return formatScientificHTML(value, significantDigits);
  }
  return cleanMinus(Number(value.toPrecision(significantDigits)).toString());
}

function formatArea(value) {
  return formatCompactHTML(value, 3);
}

function formatPercent(fraction) {
  if (!Number.isFinite(fraction)) return 'unknown';
  const percentage = 100 * fraction;
  if (percentage === 0) return '0%';
  return `${formatCompactHTML(percentage, 3)}%`;
}

function formatCount(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 'many';
  if (Math.abs(number) < 1.0e6) return Math.round(number).toLocaleString('en-US');
  return formatScientificHTML(number, 4);
}

function formatFixed(value, decimalPlaces) {
  if (!Number.isFinite(value)) return 'unknown';
  const threshold = 0.5 * Math.pow(10, -decimalPlaces);
  const cleaned = Math.abs(value) < threshold ? 0 : value;
  return cleanMinus(cleaned.toFixed(decimalPlaces));
}

function formatComplex(x, y, decimalPlaces) {
  const sign = y < 0 ? '−' : '+';
  return `${formatFixed(x, decimalPlaces)} ${sign} ${formatFixed(Math.abs(y), decimalPlaces)}i`;
}

function currentAspect() {
  if (els.overlay?.width > 0 && els.overlay?.height > 0) {
    return els.overlay.width / els.overlay.height;
  }
  const view = state.data?.view;
  return view ? view.width / view.height : 4 / 3;
}

function maximumSpanX() {
  const view = state.data.view;
  const xRange = view.xmax - view.xmin;
  const yRangeAsX = (view.ymax - view.ymin) * currentAspect();
  return Math.min(Number(view.max_span_x ?? Infinity), xRange, yRangeAsX);
}

function constrainedView(centerX, centerY, spanX) {
  const view = state.data.view;
  const maxSpan = maximumSpanX();
  const minSpan = Math.min(Number(view.min_span_x), maxSpan);
  const boundedSpan = clamp(spanX, minSpan, maxSpan);
  const halfX = boundedSpan / 2;
  const halfY = boundedSpan / currentAspect() / 2;
  const minCenterX = view.xmin + halfX;
  const maxCenterX = view.xmax - halfX;
  const minCenterY = view.ymin + halfY;
  const maxCenterY = view.ymax - halfY;
  return {
    centerX: minCenterX <= maxCenterX
      ? clamp(centerX, minCenterX, maxCenterX)
      : (view.xmin + view.xmax) / 2,
    centerY: minCenterY <= maxCenterY
      ? clamp(centerY, minCenterY, maxCenterY)
      : (view.ymin + view.ymax) / 2,
    spanX: boundedSpan,
  };
}

function applyView(centerX, centerY, spanX) {
  const bounded = constrainedView(centerX, centerY, spanX);
  state.centerX = bounded.centerX;
  state.centerY = bounded.centerY;
  state.spanX = bounded.spanX;
}

function pointerCoordinateDecimals() {
  const unitsPerPixel = state.spanX / Math.max(1, els.overlay.width);
  return clamp(Math.ceil(-Math.log10(unitsPerPixel)) + 1, 3, 15);
}

function componentCoordinateDecimals(component) {
  const size = Math.max(
    Number(component.size ?? 0),
    Math.sqrt(Math.max(0, Number(component.area ?? 0)) / Math.PI),
    1.0e-15,
  );
  return clamp(Math.ceil(-Math.log10(size)) + 2, 3, 15);
}

function zoomLevel() {
  return maximumSpanX() / state.spanX;
}

function componentSize(component) {
  const stored = Number(component.size ?? component.radius ?? 0);
  const area = Number(component.area ?? 0);
  if (Number.isFinite(area) && area > 0) {
    if (component.shape === 'circle') return Math.sqrt(area / Math.PI);
    if (component.shape === 'cardioid') return Math.sqrt(area / (6 * Math.PI));
  }
  return Number.isFinite(stored) && stored > 0 ? stored : 0;
}

function componentShapeCenter(component) {
  const center = component.center;
  if (!Array.isArray(component.shapeCenter)) return center;

  // A malformed polygon fit must never turn a microscopic component into a
  // screen-sized hover target.  Real fitted offsets are typically a fraction
  // of the component size; reject origins displaced by more than two sizes.
  const size = componentSize(component);
  const dx = Number(component.shapeCenter[0]) - Number(center[0]);
  const dy = Number(component.shapeCenter[1]) - Number(center[1]);
  if (!(size > 0) || !Number.isFinite(dx) || !Number.isFinite(dy)
      || Math.hypot(dx, dy) > 2 * size) {
    return center;
  }
  return component.shapeCenter;
}

function reportHeight() {
  if (window.self === window.top) return;
  const height = Math.ceil(document.documentElement.scrollHeight);
  window.parent.postMessage({ type: 'iframe-height', height }, '*');
}

function canvasPoint(event) {
  const rect = els.overlay.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * els.overlay.width / rect.width,
    y: (event.clientY - rect.top) * els.overlay.height / rect.height,
    cssX: event.clientX - rect.left,
    cssY: event.clientY - rect.top,
  };
}

function spanY() { return state.spanX * els.overlay.height / els.overlay.width; }

function screenToWorld(point) {
  return {
    x: state.centerX + (point.x / els.overlay.width - 0.5) * state.spanX,
    y: state.centerY + (0.5 - point.y / els.overlay.height) * spanY(),
  };
}

function worldToScreen(x, y) {
  return {
    x: (0.5 + (x - state.centerX) / state.spanX) * els.overlay.width,
    y: (0.5 - (y - state.centerY) / spanY()) * els.overlay.height,
  };
}

function pointInPolygon(x, y, flat) {
  let inside = false;
  const count = flat.length / 2;
  for (let i = 0, j = count - 1; i < count; j = i++) {
    const xi = flat[2 * i], yi = flat[2 * i + 1];
    const xj = flat[2 * j], yj = flat[2 * j + 1];
    const crosses = ((yi > y) !== (yj > y)) &&
      (x < (xj - xi) * (y - yi) / ((yj - yi) || Number.EPSILON) + xi);
    if (crosses) inside = !inside;
  }
  return inside;
}

function insideMainCardioid(x, y) {
  const dx = x - 0.25;
  const q = dx * dx + y * y;
  return q * (q + dx) <= 0.25 * y * y;
}

function cardioidWorldPoint(component, t) {
  const shapeCenter = componentShapeCenter(component);
  const size = componentSize(component);
  const angle = Number(component.angle || 0);
  const xi = Number(component.xi || 0);
  const cosine = Math.cos(angle), sine = Math.sin(angle);
  const slant = 1 - xi * Math.sin(t);
  const qx = (2 * Math.cos(t) - Math.cos(2 * t)) * slant;
  const qy = (2 * Math.sin(t) - Math.sin(2 * t)) * slant;
  return [
    shapeCenter[0] + size * (cosine * qx - sine * qy),
    shapeCenter[1] + size * (sine * qx + cosine * qy),
  ];
}

function cardioidOutlineFlat(component, samples = 192) {
  const points = [];
  for (let i = 0; i < samples; i++) {
    const point = cardioidWorldPoint(component, 2 * Math.PI * i / samples);
    points.push(point[0], point[1]);
  }
  return points;
}

function insideAnalyticComponent(component, x, y) {
  const shapeCenter = componentShapeCenter(component);
  const cx = shapeCenter[0], cy = shapeCenter[1];
  const size = componentSize(component);
  if (!(size > 0)) return false;
  const dx = x - cx, dy = y - cy;
  if (component.shape === 'circle') {
    return dx * dx + dy * dy <= size * size;
  }
  if (component.shape === 'cardioid') {
    const xi = Number(component.xi || 0);
    if (Math.abs(xi) > 1e-15) {
      if (!component._slantedOutline) {
        component._slantedOutline = cardioidOutlineFlat(component, 192);
      }
      return pointInPolygon(x, y, component._slantedOutline);
    }
    const angle = Number(component.angle || 0);
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    const localX = (cosine * dx + sine * dy) / size;
    const localY = (-sine * dx + cosine * dy) / size;
    const shiftedX = localX - 1;
    const q = shiftedX * shiftedX + localY * localY;
    return q * (q + 4 * shiftedX) <= 4 * localY * localY;
  }
  return false;
}

function pointInComponent(component, x, y) {
  if (component.shape === 'polygon') return pointInPolygon(x, y, component.points);
  return insideAnalyticComponent(component, x, y);
}

function componentBounds(component) {
  if (component._bbox) return component._bbox;
  if (Array.isArray(component.bbox) && component.bbox.length === 4) {
    component._bbox = component.bbox;
    return component._bbox;
  }
  const shapeCenter = componentShapeCenter(component);
  const cx = shapeCenter[0], cy = shapeCenter[1];
  const size = componentSize(component);
  if (component.shape === 'circle') {
    component._bbox = [cx - size, cx + size, cy - size, cy + size];
    return component._bbox;
  }
  if (component.shape === 'cardioid') {
    const xi = Number(component.xi || 0);
    if (Math.abs(xi) > 1e-15) {
      const flat = cardioidOutlineFlat(component, 256);
      let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
      for (let i = 0; i < flat.length; i += 2) {
        xmin = Math.min(xmin, flat[i]); xmax = Math.max(xmax, flat[i]);
        ymin = Math.min(ymin, flat[i + 1]); ymax = Math.max(ymax, flat[i + 1]);
      }
      component._bbox = [xmin, xmax, ymin, ymax];
      return component._bbox;
    }
    const angle = Number(component.angle || 0);
    const cosine = Math.cos(angle), sine = Math.sin(angle);
    let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
    // Exact axis-aligned extrema.  Besides the cusp t=0, dx/dt=0 at
    // 3t = pi - 2*angle + 2*pi*k and dy/dt=0 at
    // 3t = -2*angle + 2*pi*k.
    const candidates = [0];
    for (let k = 0; k < 3; k++) {
      candidates.push((Math.PI - 2 * angle + 2 * Math.PI * k) / 3);
      candidates.push((-2 * angle + 2 * Math.PI * k) / 3);
    }
    for (const t of candidates) {
      const qx = 2 * Math.cos(t) - Math.cos(2 * t);
      const qy = 2 * Math.sin(t) - Math.sin(2 * t);
      const x = cx + size * (cosine * qx - sine * qy);
      const y = cy + size * (sine * qx + cosine * qy);
      xmin = Math.min(xmin, x); xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y); ymax = Math.max(ymax, y);
    }
    const padding = Math.max(1e-15, 1e-12 * size);
    component._bbox = [xmin - padding, xmax + padding, ymin - padding, ymax + padding];
    return component._bbox;
  }
  const flat = component.points || [];
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < flat.length; i += 2) {
    xmin = Math.min(xmin, flat[i]); xmax = Math.max(xmax, flat[i]);
    ymin = Math.min(ymin, flat[i + 1]); ymax = Math.max(ymax, flat[i + 1]);
  }
  component._bbox = [xmin, xmax, ymin, ymax];
  return component._bbox;
}

function buildSpatialIndex() {
  const componentIndices = [];
  state.componentsByPeriod = new Map();
  state.data.components.forEach((component, index) => {
    if (!state.componentsByPeriod.has(component.period)) {
      state.componentsByPeriod.set(component.period, []);
    }
    state.componentsByPeriod.get(component.period).push(component);
    componentIndices.push(index);
  });
  const bounds = [-2.1, 0.55, -1.35, 1.35];
  const nx = 96, ny = 72;
  const cells = Array.from({ length: nx * ny }, () => []);
  const ix = x => clamp(Math.floor((x - bounds[0]) / (bounds[1] - bounds[0]) * nx), 0, nx - 1);
  const iy = y => clamp(Math.floor((y - bounds[2]) / (bounds[3] - bounds[2]) * ny), 0, ny - 1);
  for (const index of componentIndices) {
    const [xmin, xmax, ymin, ymax] = componentBounds(state.data.components[index]);
    for (let gy = iy(ymin); gy <= iy(ymax); gy++) {
      for (let gx = ix(xmin); gx <= ix(xmax); gx++) cells[gy * nx + gx].push(index);
    }
  }
  state.spatial = { bounds, nx, ny, cells };
}

function hitComponent(x, y) {
  const { bounds, nx, ny, cells } = state.spatial;
  if (x < bounds[0] || x > bounds[1] || y < bounds[2] || y > bounds[3]) return null;
  const gx = clamp(Math.floor((x - bounds[0]) / (bounds[1] - bounds[0]) * nx), 0, nx - 1);
  const gy = clamp(Math.floor((y - bounds[2]) / (bounds[3] - bounds[2]) * ny), 0, ny - 1);
  let best = null;
  for (const index of cells[gy * nx + gx]) {
    const component = state.data.components[index];
    const [xmin, xmax, ymin, ymax] = componentBounds(component);
    if (x < xmin || x > xmax || y < ymin || y > ymax) continue;
    if (pointInComponent(component, x, y)) {
      if (!best || component.period > best.period) best = component;
    }
  }
  return best;
}

function escapesMandelbrot(x, y, iterations = 350) {
  if (insideMainCardioid(x, y)) return false;
  const dx = x + 1;
  if (dx * dx + y * y <= 0.0625) return false;
  let zr = 0, zi = 0;
  for (let i = 0; i < iterations; i++) {
    const zr2 = zr * zr, zi2 = zi * zi;
    zi = 2 * zr * zi + y;
    zr = zr2 - zi2 + x;
    if (zr * zr + zi * zi > 16) return true;
  }
  return false;
}

function componentPath(ctx, component) {
  ctx.beginPath();
  if (component.shape === 'cardioid') {
    const samples = 220;
    for (let i = 0; i <= samples; i++) {
      const t = 2 * Math.PI * i / samples;
      const point = cardioidWorldPoint(component, t);
      const p = worldToScreen(point[0], point[1]);
      if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
    }
    ctx.closePath();
    return;
  }
  if (component.shape === 'circle') {
    const size = componentSize(component);
    const shapeCenter = componentShapeCenter(component);
    const center = worldToScreen(shapeCenter[0], shapeCenter[1]);
    const edge = worldToScreen(shapeCenter[0] + size, shapeCenter[1]);
    ctx.arc(center.x, center.y, Math.abs(edge.x - center.x), 0, 2 * Math.PI);
    return;
  }
  const flat = component.points;
  for (let i = 0; i < flat.length; i += 2) {
    const p = worldToScreen(flat[i], flat[i + 1]);
    if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
  }
  ctx.closePath();
}

function paintComponent(ctx, component, fill, stroke, width) {
  componentPath(ctx, component);
  ctx.fillStyle = fill;
  ctx.strokeStyle = stroke;
  ctx.lineWidth = width;
  ctx.fill();
  ctx.stroke();
}

function componentVisible(component) {
  const halfX = state.spanX / 2;
  const halfY = spanY() / 2;
  const viewMinX = state.centerX - halfX;
  const viewMaxX = state.centerX + halfX;
  const viewMinY = state.centerY - halfY;
  const viewMaxY = state.centerY + halfY;
  const [xmin, xmax, ymin, ymax] = componentBounds(component);
  return xmax >= viewMinX && xmin <= viewMaxX && ymax >= viewMinY && ymin <= viewMaxY;
}

function drawOverlay() {
  const ctx = els.overlayCtx;
  ctx.clearRect(0, 0, els.overlay.width, els.overlay.height);
  const style = state.data.interaction;

  if (state.hover) {
    // With tens of thousands of discovered components, painting every visible
    // component of the same period turns hover into an unreadable web of
    // circles and cardioids.  Highlight only the component under the pointer.
    paintComponent(ctx, state.hover, style.active_fill, style.active_stroke, style.active_line_width);
  }
  if (state.selected && state.selected.id !== state.hover?.id) {
    paintComponent(ctx, state.selected, style.active_fill, style.active_stroke, style.active_line_width);
  }
}

function updateInfo() {
  const component = state.selected || state.hover;
  if (!component) {
    els.info.innerHTML = '<div class="info-empty">Move over a known component to inspect it.</div>';
    return;
  }
  const summary = state.data.periodSummary.find(row => row.period === component.period);
  const total = Number(state.data.mandelbrotAreaReference);
  const completePeriod = summary?.complete !== false && component.index >= 0;
  const componentLine = completePeriod
    ? `Component ${component.index + 1}/${formatCount(summary.count)}`
    : `Discovered component · one of ${formatCount(summary?.count ?? component.periodCount)}`;
  const periodAreaLabel = completePeriod ? "Period total" : "Known period area";
  const periodArea = Number(summary?.knownTotalArea ?? summary?.totalArea ?? component.area);
  const centerDecimals = componentCoordinateDecimals(component);
  els.info.innerHTML = `
    <div class="info-card">
      <div class="info-label">Component</div>
      <div class="info-value">${componentLine}<br>Period ${component.period}</div>
    </div>
    <div class="info-card">
      <div class="info-label">Center</div>
      <div class="info-value">${formatComplex(component.center[0], component.center[1], centerDecimals)}</div>
    </div>
    <div class="info-card info-card-wide">
      <div class="info-label">Area</div>
      <div class="info-value">This component: ${formatArea(component.area)}, ${formatPercent(component.area / total)} of the total<br>${periodAreaLabel}: ${formatArea(periodArea)}, ${formatPercent(periodArea / total)} of the total</div>
    </div>`;
}

function updateTooltip(point) {
  if (!state.hover || !point || state.drag?.moved) {
    els.tooltip.style.display = 'none';
    return;
  }
  const summary = state.data.periodSummary.find(row => row.period === state.hover.period);
  const label = state.hover.index >= 0 && summary?.complete !== false
    ? `component ${state.hover.index + 1}/${formatCount(summary.count)}`
    : `discovered component`;
  els.tooltip.innerHTML = `Period ${state.hover.period} ${label}<br>Area: ${formatArea(state.hover.area)}`;
  els.tooltip.style.display = 'block';
  const stageRect = els.stage.getBoundingClientRect();
  const tooltipRect = els.tooltip.getBoundingClientRect();
  let left = point.cssX + 14;
  let top = point.cssY + 14;
  if (left + tooltipRect.width > stageRect.width - 8) left = point.cssX - tooltipRect.width - 14;
  if (top + tooltipRect.height > stageRect.height - 8) top = point.cssY - tooltipRect.height - 14;
  els.tooltip.style.left = `${Math.max(8, left)}px`;
  els.tooltip.style.top = `${Math.max(8, top)}px`;
}

function updateCoordinateReadout(point = state.pointer) {
  if (!point) {
    els.coordinate.textContent = '';
    return;
  }
  const world = screenToWorld(point);
  const decimals = pointerCoordinateDecimals();
  els.coordinate.textContent = `c = ${formatComplex(world.x, world.y, decimals)}  ·  zoom ${zoomLevel().toFixed(1)}`;
}

function updateHover(point) {
  state.pointer = point;
  const world = screenToWorld(point);
  state.hover = hitComponent(world.x, world.y);
  updateCoordinateReadout(point);
  updateTooltip(point);
  updateInfo();
  drawOverlay();
}

function requestCpuRender(interacting) {
  if (!state.workerReady) return;
  const passes = interacting ? state.data.render.interaction_passes : state.data.render.passes;
  const token = ++state.renderToken;
  state.worker.postMessage({
    type: 'render',
    token,
    width: els.fractal.width,
    height: els.fractal.height,
    centerX: state.centerX,
    centerY: state.centerY,
    spanX: state.spanX,
    passes,
  });
  els.status.textContent = interacting ? 'CPU preview…' : 'CPU refining…';
}

function renderWebGLFrame() {
  state.renderRaf = 0;
  if (!state.webgl) return;
  const interacting = state.pendingInteracting;
  const scale = interacting
    ? Number(state.data.render.interaction_resolution_scale ?? 0.5)
    : Number(state.data.render.idle_resolution_scale ?? 1.0);
  try {
    state.webgl.render({
      displayWidth: els.overlay.width,
      displayHeight: els.overlay.height,
      centerX: state.centerX,
      centerY: state.centerY,
      spanX: state.spanX,
      resolutionScale: scale,
    });
    const percent = Math.round(100 * Math.max(0.1, Math.min(1.0, scale)));
    els.status.textContent = interacting
      ? `WebGL2 GPU · ${percent}% preview`
      : 'WebGL2 GPU · full resolution';
    state.baseReady = true;
  } catch (error) {
    console.error(error);
    els.status.textContent = 'WebGL2 render failed';
  }
}

function requestRender(interacting) {
  if (!state.data) return;
  clearTimeout(state.settleTimer);

  if (state.rendererType === 'webgl2') {
    state.pendingInteracting = interacting;
    if (!state.renderRaf) state.renderRaf = requestAnimationFrame(renderWebGLFrame);
  } else if (state.rendererType === 'cpu') {
    requestCpuRender(interacting);
  }

  if (interacting) {
    state.settleTimer = setTimeout(
      () => requestRender(false),
      state.data.render.settle_delay_ms,
    );
  }
}

function cancelAnimation() {
  if (state.animation) cancelAnimationFrame(state.animation.raf);
  state.animation = null;
}

function animateView(targetX, targetY, targetSpan) {
  cancelAnimation();
  const target = constrainedView(targetX, targetY, targetSpan);
  targetX = target.centerX;
  targetY = target.centerY;
  targetSpan = target.spanX;
  const start = performance.now();
  const fromX = state.centerX, fromY = state.centerY, fromSpan = state.spanX;
  const duration = state.data.interaction.transition_ms;
  state.animation = { raf: 0 };
  const step = now => {
    if (!state.animation) return;
    const t = clamp((now - start) / duration, 0, 1);
    const u = easeInOut(t);
    const nextSpan = Math.exp(
      Math.log(fromSpan) + (Math.log(targetSpan) - Math.log(fromSpan)) * u,
    );
    // Keep the target's screen-space motion synchronized with the zoom. A
    // plain linear centre interpolation makes the scale collapse first and
    // the pan appear to lag comically behind it.
    const remainingScreenOffset = (1 - u) * (nextSpan / fromSpan);
    applyView(
      targetX + (fromX - targetX) * remainingScreenOffset,
      targetY + (fromY - targetY) * remainingScreenOffset,
      nextSpan,
    );
    updateCoordinateReadout();
    requestRender(true);
    drawOverlay();
    if (t < 1) state.animation.raf = requestAnimationFrame(step);
    else {
      state.animation = null;
      requestRender(false);
    }
  };
  state.animation.raf = requestAnimationFrame(step);
}

function frameComponent(component) {
  const [xmin, xmax, ymin, ymax] = componentBounds(component);
  const fraction = state.data.interaction.component_fit_fraction;
  const aspect = els.overlay.width / els.overlay.height;
  const neededX = (xmax - xmin) / fraction;
  const neededYAsX = (ymax - ymin) * aspect / fraction;
  const targetSpan = clamp(
    Math.max(neededX, neededYAsX),
    state.data.view.min_span_x,
    maximumSpanX(),
  );
  animateView((xmin + xmax) / 2, (ymin + ymax) / 2, targetSpan);
}

function resetView(animated = true) {
  const view = state.data.view;
  const x = (view.xmin + view.xmax) / 2;
  const y = (view.ymin + view.ymax) / 2;
  const span = maximumSpanX();
  if (animated) animateView(x, y, span);
  else {
    applyView(x, y, span);
    requestRender(false); drawOverlay();
  }
}

function resizeCanvas() {
  const rect = els.stage.getBoundingClientRect();
  const desiredAspect = state.data.view.width / state.data.view.height;
  const dpr = Math.min(window.devicePixelRatio || 1, state.data.render.device_pixel_ratio_cap);
  let width = Math.min(state.data.render.max_width, Math.max(1, Math.round(rect.width * dpr)));
  let height = Math.round(width / desiredAspect);
  if (height > state.data.render.max_height) {
    height = state.data.render.max_height;
    width = Math.round(height * desiredAspect);
  }
  els.stage.style.aspectRatio = `${width} / ${height}`;
  if (els.overlay.width !== width || els.overlay.height !== height) {
    els.overlay.width = width;
    els.overlay.height = height;
  }
  if (state.rendererType === 'cpu' &&
      (els.fractal.width !== width || els.fractal.height !== height)) {
    els.fractal.width = width;
    els.fractal.height = height;
  }
  applyView(state.centerX, state.centerY, state.spanX);
  updateCoordinateReadout();
  drawOverlay();
  requestRender(false);
  reportHeight();
}

function clearDragState() {
  if (state.drag) {
    try {
      if (els.overlay.hasPointerCapture(state.drag.pointerId)) {
        els.overlay.releasePointerCapture(state.drag.pointerId);
      }
    } catch (_) {
      // The browser may already have released capture while losing focus.
    }
  }
  state.drag = null;
  els.overlay.classList.remove('dragging');
}

function bindEvents() {
  els.overlay.addEventListener('pointerdown', event => {
    if (!event.isPrimary) return;
    clearDragState();
    cancelAnimation();
    els.overlay.setPointerCapture(event.pointerId);
    const point = canvasPoint(event);
    state.drag = {
      pointerId: event.pointerId,
      startPoint: point,
      anchorWorld: screenToWorld(point),
      moved: false,
    };
    els.overlay.classList.add('dragging');
    updateTooltip(null);
  });

  els.overlay.addEventListener('pointermove', event => {
    const point = canvasPoint(event);
    if (state.drag && state.drag.pointerId === event.pointerId) {
      const dx = point.x - state.drag.startPoint.x;
      const dy = point.y - state.drag.startPoint.y;
      const threshold = state.data.interaction.drag_threshold_pixels * els.overlay.width / els.overlay.getBoundingClientRect().width;
      if (Math.hypot(dx, dy) > threshold) state.drag.moved = true;
      if (state.drag.moved) {
        const sy = spanY();
        applyView(
          state.drag.anchorWorld.x - (point.x / els.overlay.width - .5) * state.spanX,
          state.drag.anchorWorld.y - (.5 - point.y / els.overlay.height) * sy,
          state.spanX,
        );
        state.pointer = point;
        updateCoordinateReadout(point);
        state.hover = null;
        requestRender(true);
        drawOverlay();
        updateInfo();
        return;
      }
    }
    updateHover(point);
  });

  const finishPointer = event => {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    const point = canvasPoint(event);
    const moved = state.drag.moved;
    clearDragState();
    if (!moved) {
      const world = screenToWorld(point);
      const hit = hitComponent(world.x, world.y);
      if (hit) {
        state.selected = hit;
        state.hover = hit;
        frameComponent(hit);
      } else if (escapesMandelbrot(
        world.x,
        world.y,
        state.data.interaction.click_test_iterations,
      )) {
        state.selected = null;
      }
      updateInfo();
      drawOverlay();
    } else {
      requestRender(false);
      updateHover(point);
    }
  };
  els.overlay.addEventListener('pointerup', finishPointer);
  els.overlay.addEventListener('pointercancel', () => {
    clearDragState();
    requestRender(false);
  });
  els.overlay.addEventListener('lostpointercapture', () => {
    if (state.drag) clearDragState();
  });

  els.overlay.addEventListener('pointerleave', () => {
    if (!state.drag) {
      state.hover = null;
      state.pointer = null;
      els.tooltip.style.display = 'none';
      els.coordinate.textContent = '';
      updateInfo();
      drawOverlay();
    }
  });

  els.overlay.addEventListener('wheel', event => {
    event.preventDefault();
    cancelAnimation();
    const point = canvasPoint(event);
    const anchor = screenToWorld(point);
    const factor = Math.exp(event.deltaY * state.data.interaction.wheel_zoom_speed);
    const nextSpan = clamp(
      state.spanX * factor,
      state.data.view.min_span_x,
      maximumSpanX(),
    );
    const nextSpanY = nextSpan * els.overlay.height / els.overlay.width;
    applyView(
      anchor.x - (point.x / els.overlay.width - .5) * nextSpan,
      anchor.y - (.5 - point.y / els.overlay.height) * nextSpanY,
      nextSpan,
    );
    requestRender(true);
    updateHover(point);
  }, { passive: false });

  window.addEventListener('blur', () => {
    clearDragState();
    state.hover = null;
    updateInfo();
    drawOverlay();
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) clearDragState();
  });
  els.overlay.addEventListener('pointerenter', event => {
    if (!state.drag && event.isPrimary) updateHover(canvasPoint(event));
  });

  document.getElementById('resetView').addEventListener('click', () => {
    state.selected = null;
    resetView(true);
    updateInfo();
  });
  window.addEventListener('resize', resizeCanvas);
}

async function setup() {
  els.stage = document.getElementById('stage');
  els.fractal = document.getElementById('fractalCanvas');
  els.overlay = document.getElementById('overlayCanvas');
  els.overlayCtx = els.overlay.getContext('2d');
  els.status = document.getElementById('status');
  els.coordinate = document.getElementById('coordinate');
  els.tooltip = document.getElementById('tooltip');
  els.info = document.getElementById('info');

  const response = await fetch(`data.json?v=${Date.now()}`, { cache: 'no-store' });
  if (!response.ok) throw new Error(`data.json: HTTP ${response.status}`);
  state.data = await response.json();
  buildSpatialIndex();

  const view = state.data.view;
  applyView(
    (view.xmin + view.xmax) / 2,
    (view.ymin + view.ymax) / 2,
    maximumSpanX(),
  );

  const requestedRenderer = String(state.data.render.renderer || 'auto').toLowerCase();
  if (requestedRenderer !== 'cpu') {
    try {
      if (!window.MandelbrotWebGLRenderer) throw new Error('WebGL renderer script did not load.');
      state.webgl = new window.MandelbrotWebGLRenderer(els.fractal, state.data);
      state.webgl.onContextLost = () => {
        els.status.textContent = 'WebGL context lost';
      };
      state.webgl.onContextRestored = () => {
        els.status.textContent = 'WebGL context restored';
        requestRender(false);
      };
      state.rendererType = 'webgl2';
      els.stage.dataset.renderer = 'webgl2';
    } catch (error) {
      console.warn('WebGL2 initialization failed; using CPU worker fallback.', error);
      if (requestedRenderer === 'webgl2') {
        console.warn('render.renderer requested WebGL2 explicitly, but fallback remains enabled.');
      }
      const replacement = els.fractal.cloneNode(false);
      els.fractal.replaceWith(replacement);
      els.fractal = replacement;
    }
  }

  if (state.rendererType !== 'webgl2') {
    state.rendererType = 'cpu';
    els.stage.dataset.renderer = 'cpu';
    els.fractalCtx = els.fractal.getContext('2d');
    state.worker = new Worker(`mandelbrot_worker.js?v=${state.data.assetVersion}`);
    state.worker.onmessage = event => {
      const message = event.data;
      if (message.type === 'ready') {
        state.workerReady = true;
        resizeCanvas();
        return;
      }
      if (message.type === 'frame') {
        if (message.token !== state.renderToken) {
          message.bitmap.close();
          return;
        }
        els.fractalCtx.imageSmoothingEnabled = false;
        els.fractalCtx.clearRect(0, 0, els.fractal.width, els.fractal.height);
        els.fractalCtx.drawImage(message.bitmap, 0, 0, els.fractal.width, els.fractal.height);
        message.bitmap.close();
        state.baseReady = true;
        els.status.textContent = message.scale === 1
          ? 'CPU fallback · full resolution'
          : `CPU fallback · preview 1/${message.scale}`;
      }
      if (message.type === 'done' && message.token === state.renderToken) {
        els.status.textContent = 'CPU fallback · ready';
      }
      if (message.type === 'error') {
        console.error(message.message);
        els.status.textContent = 'CPU fallback render error';
      }
    };
    state.worker.postMessage({ type: 'init', data: state.data });
  } else {
    resizeCanvas();
  }

  bindEvents();
  updateInfo();
  reportHeight();
}

setup().catch(error => {
  console.error(error);
  const status = document.getElementById('status');
  if (status) status.textContent = `failed to load: ${error.message}`;
});
