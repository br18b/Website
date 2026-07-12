(() => {
  const state = {
    data: null,
    selected: null,
    hoverKind: 'none', // none | contour | interior | exterior-unavailable
    showAll: false,
    hoverWindow: null,
    openWindow: null,
    pointerInOverview: false,
    overviewPointer: null,
    embedded: document.documentElement.classList.contains('is-embedded'),
    lastReportedHeight: 0,
    reportFrame: 0,
  };
  const STYLE = Object.freeze({
    faintOverviewRgba: "rgba(0,0,0,0.70)",
    faintZoomRgba: "rgba(0,0,0,0.28)",
    selectedContourRgba: "rgba(0,0,0,1.0)",
    faintWidth: 1.15,
    faintCount: 24,
    selectedOverviewWidth: 2.2,
    selectedZoomWidth: 2.8,

    windowIdleStrokeRgba: "rgba(255,52,72,0.13)",
    windowIdleFillRgba: "rgba(255,52,72,0.05)",
    windowNormalStrokeRgba: "rgba(255,52,72,0.50)",
    windowNormalFillRgba: "rgba(255,52,72,0.1)",
    windowHoverStrokeRgba: "rgba(255,52,72,1.00)",
    windowHoverFillRgba: "rgba(255,52,72,0.2)",
    windowIdleLineWidth: 1.0,
    windowNormalLineWidth: 2.0,
    windowHoverLineWidth: 3.0,
    windowCornerRadius: 0.0,
    windowHoverDistancePixels: 0.0,
    windowProximityRangeBoxWidths: 6.0,
    windowProximityFalloff: "smoothstep",

    windowShowHoverLabel: true,
    windowLabelPrefix: "",
    windowLabelFont: "700 18px system-ui, -apple-system, Segoe UI, sans-serif",
    windowLabelBackgroundRgba: "rgba(5,12,26,0.78)",
    windowLabelTextRgba: "rgba(255,255,255,1.00)",
    windowLabelPaddingX: 8.0,
    windowLabelHeight: 25.0,
    windowLabelCornerRadius: 10.0,
    windowLabelGap: 8.0,
  });

  const els = {};

  function fmtG(g) {
    if (!Number.isFinite(g) || g <= 0) return '';
    return `Selected contour: G ≈ ${g.toExponential(3)}`;
  }

  function readoutText() {
    if (state.hoverKind === 'contour' && state.selected != null) {
      const contour = state.data.contours[state.selected];
      return contour ? fmtG(contour.G) : '';
    }
    if (state.hoverKind === 'interior') {
      return 'Interior — no contour to draw';
    }
    if (state.hoverKind === 'exterior-unavailable') {
      return 'Exterior — no sampled contour';
    }
    return '';
  }

  function clearHoverSelection() {
    state.selected = null;
    state.hoverKind = 'none';
  }

  function reportEmbeddedHeight() {
    if (!state.embedded || !els.demo) return;
    if (state.reportFrame) cancelAnimationFrame(state.reportFrame);

    state.reportFrame = requestAnimationFrame(() => {
      state.reportFrame = 0;
      const height = Math.ceil(els.demo.getBoundingClientRect().height);
      if (!(height > 0) || Math.abs(height - state.lastReportedHeight) < 1) return;

      state.lastReportedHeight = height;
      window.parent.postMessage(
        { type: 'lazy-demo:resize', height },
        window.location.origin,
      );
    });
  }

  function lockEmbeddedViewport() {
    if (!state.embedded) return;

    const blockMiddleMouse = evt => {
      if (evt.button !== 1) return;
      evt.preventDefault();
      evt.stopPropagation();
    };

    document.addEventListener('mousedown', blockMiddleMouse, true);
    document.addEventListener('auxclick', blockMiddleMouse, true);
    window.addEventListener('scroll', () => {
      if (window.scrollX !== 0 || window.scrollY !== 0) {
        window.scrollTo(0, 0);
      }
    }, { passive: true });
  }

  function clamp01(value) {
    return Math.max(0, Math.min(1, value));
  }

  function smoothstep01(value) {
    const t = clamp01(value);
    return t * t * (3 - 2 * t);
  }

  function proximityFalloff(value) {
    return STYLE.windowProximityFalloff === 'linear'
      ? clamp01(value)
      : smoothstep01(value);
  }

  function parseCssColor(value) {
    const text = String(value).trim();

    const rgba = text.match(/^rgba?\(\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)\s*,\s*([+-]?[\d.]+)(?:\s*,\s*([+-]?[\d.]+)\s*)?\)$/i);
    if (rgba) {
      return {
        r: Number(rgba[1]),
        g: Number(rgba[2]),
        b: Number(rgba[3]),
        a: rgba[4] == null ? 1 : Number(rgba[4]),
      };
    }

    const hex = text.match(/^#([0-9a-f]{3}|[0-9a-f]{4}|[0-9a-f]{6}|[0-9a-f]{8})$/i);
    if (hex) {
      let digits = hex[1];
      if (digits.length === 3 || digits.length === 4) {
        digits = digits.split('').map(ch => ch + ch).join('');
      }
      return {
        r: parseInt(digits.slice(0, 2), 16),
        g: parseInt(digits.slice(2, 4), 16),
        b: parseInt(digits.slice(4, 6), 16),
        a: digits.length === 8 ? parseInt(digits.slice(6, 8), 16) / 255 : 1,
      };
    }

    console.warn(`Unsupported marker color ${text}; using transparent black.`);
    return { r: 0, g: 0, b: 0, a: 0 };
  }

  function mixNumber(a, b, amount) {
    return a + (b - a) * amount;
  }

  function mixColor(a, b, amount) {
    const ca = parseCssColor(a);
    const cb = parseCssColor(b);
    const t = clamp01(amount);
    return `rgba(${Math.round(mixNumber(ca.r, cb.r, t))},${Math.round(mixNumber(ca.g, cb.g, t))},${Math.round(mixNumber(ca.b, cb.b, t))},${mixNumber(ca.a, cb.a, t).toFixed(4)})`;
  }

  function assetUrl(src) {
    const version = state.data?.assetVersion;
    if (!version) return src;
    const sep = src.includes('?') ? '&' : '?';
    return `${src}${sep}v=${encodeURIComponent(version)}`;
  }

  function loadImage(src, label = src) {
    const relativeUrl = assetUrl(src);
    const url = new URL(relativeUrl, window.location.href).href;

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.decoding = 'async';
      img.onload = () => resolve(img);
      img.onerror = () => {
        reject(new Error(`Failed to load ${label}: ${url}`));
      };
      img.src = url;
    });
  }

  function canvasPoint(evt, canvas) {
    const r = canvas.getBoundingClientRect();
    return {
      x: (evt.clientX - r.left) * canvas.width / r.width,
      y: (evt.clientY - r.top) * canvas.height / r.height,
    };
  }

  function getHoverFromMap(ctx, x, y, w, h) {
    if (!(x >= 0 && x < w && y >= 0 && y < h)) {
      return { kind: 'none', index: null };
    }
    const ix = Math.max(0, Math.min(w - 1, Math.floor(x)));
    const iy = Math.max(0, Math.min(h - 1, Math.floor(y)));
    const d = ctx.getImageData(ix, iy, 1, 1).data;
    const status = d[2];
    if (status === 1) return { kind: 'interior', index: null };
    if (status === 2) return { kind: 'exterior-unavailable', index: null };

    const index = d[0] + 256 * d[1];
    if (index >= 65535 || !state.data.contours[index]) {
      return { kind: 'exterior-unavailable', index: null };
    }
    return { kind: 'contour', index };
  }

  function applyHover(hit) {
    state.hoverKind = hit.kind;
    state.selected = hit.kind === 'contour' ? hit.index : null;
  }

  function viewToPixel(view, x, y) {
    return {
      x: (x - view.xmin) / (view.xmax - view.xmin) * view.width,
      y: (view.ymax - y) / (view.ymax - view.ymin) * view.height,
    };
  }

  function windowRect(win) {
    const ov = state.data.views.overview;
    const p0 = viewToPixel(ov, win.xmin, win.ymax);
    const p1 = viewToPixel(ov, win.xmax, win.ymin);
    return { x: p0.x, y: p0.y, w: p1.x - p0.x, h: p1.y - p0.y };
  }

  function pointInRect(p, r) {
    return p.x >= r.x && p.x <= r.x + r.w && p.y >= r.y && p.y <= r.y + r.h;
  }

  function distanceToRect(p, r) {
    const dx = Math.max(r.x - p.x, 0, p.x - (r.x + r.w));
    const dy = Math.max(r.y - p.y, 0, p.y - (r.y + r.h));
    return Math.hypot(dx, dy);
  }

  function findNearWindow(p) {
    let best = null;
    let bestD = Infinity;
    for (const win of state.data.windows) {
      const r = windowRect(win);
      const d = pointInRect(p, r) ? 0 : distanceToRect(p, r);
      if (d < bestD) {
        bestD = d;
        best = win;
      }
    }
    return bestD <= STYLE.windowHoverDistancePixels ? best : null;
  }

  function windowProximity(win, pointer) {
    if (!pointer) return 0;

    const rect = windowRect(win);
    const distance = distanceToRect(pointer, rect);
    const boxWidth = Math.max(1, Math.abs(rect.w));
    const range = STYLE.windowProximityRangeBoxWidths * boxWidth;

    if (!(range > 0)) {
      return pointInRect(pointer, rect) ? 1 : 0;
    }

    return proximityFalloff(1 - distance / range);
  }

  function drawPolylines(ctx, polys, color, width) {
    if (!polys) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const flat of polys) {
      if (!flat || flat.length < 4) continue;
      ctx.beginPath();
      ctx.moveTo(flat[0], flat[1]);
      for (let i = 2; i < flat.length; i += 2) {
        ctx.lineTo(flat[i], flat[i + 1]);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  function selectedContourFor(viewName) {
    if (state.selected == null) return null;
    const contour = state.data.contours[state.selected];
    if (!contour) return null;
    if (viewName === 'overview') return contour.overview;
    return contour.windows?.[viewName] ?? null;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function drawWindowMarkers(ctx) {
    ctx.save();

    for (const win of state.data.windows) {
      const r = windowRect(win);
      const proximity = state.pointerInOverview
        ? windowProximity(win, state.overviewPointer)
        : 0;
      const pointerInside = !!(
        state.overviewPointer && pointInRect(state.overviewPointer, r)
      );
      const hovered = state.hoverWindow?.id === win.id && pointerInside;

      let strokeStyle = mixColor(
        STYLE.windowIdleStrokeRgba,
        STYLE.windowNormalStrokeRgba,
        proximity,
      );
      let fillStyle = mixColor(
        STYLE.windowIdleFillRgba,
        STYLE.windowNormalFillRgba,
        proximity,
      );
      let lineWidth = mixNumber(
        STYLE.windowIdleLineWidth,
        STYLE.windowNormalLineWidth,
        proximity,
      );

      if (hovered) {
        strokeStyle = STYLE.windowHoverStrokeRgba;
        fillStyle = STYLE.windowHoverFillRgba;
        lineWidth = STYLE.windowHoverLineWidth;
      }

      ctx.lineWidth = lineWidth;
      ctx.strokeStyle = strokeStyle;
      ctx.fillStyle = fillStyle;

      if (STYLE.windowCornerRadius > 0) {
        roundRect(
          ctx,
          r.x,
          r.y,
          r.w,
          r.h,
          Math.min(STYLE.windowCornerRadius, r.w / 2, r.h / 2),
        );
        ctx.fill();
        ctx.stroke();
      } else {
        ctx.fillRect(r.x, r.y, r.w, r.h);
        ctx.strokeRect(r.x, r.y, r.w, r.h);
      }

      if (hovered && STYLE.windowShowHoverLabel) {
        const label = `${STYLE.windowLabelPrefix}${win.label}`;
        ctx.font = STYLE.windowLabelFont;

        const metrics = ctx.measureText(label);
        const padX = STYLE.windowLabelPaddingX;
        const bw = metrics.width + 2 * padX;
        const bh = STYLE.windowLabelHeight;
        const bx = Math.max(
          8,
          Math.min(
            ctx.canvas.width - bw - 8,
            r.x + r.w * 0.5 - bw * 0.5,
          ),
        );
        const by = Math.max(8, r.y - bh - STYLE.windowLabelGap);

        ctx.fillStyle = STYLE.windowLabelBackgroundRgba;
        roundRect(
          ctx,
          bx,
          by,
          bw,
          bh,
          Math.min(STYLE.windowLabelCornerRadius, bh / 2),
        );
        ctx.fill();

        ctx.fillStyle = STYLE.windowLabelTextRgba;
        const textY = by + (bh + 10) / 2;
        ctx.fillText(label, bx + padX, textY);
      }
    }

    ctx.restore();
  }


  function faintContourIndices() {
    const total = state.data.contours.length;
    const requested = Math.max(1, Math.round(STYLE.faintCount));
    const count = Math.min(total, requested);
    if (count <= 1) return total > 0 ? [0] : [];

    const indices = [];
    for (let i = 0; i < count; i += 1) {
      const index = Math.round(i * (total - 1) / (count - 1));
      if (indices[indices.length - 1] !== index) indices.push(index);
    }
    return indices;
  }

  function redrawOverview() {
    const { ctx, canvas, bg } = els.overview;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bg, 0, 0, canvas.width, canvas.height);

    if (state.openWindow) {
      ctx.save();
      ctx.fillStyle = 'rgba(4, 8, 20, .42)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.restore();
    }

    if (state.showAll) {
      for (const i of faintContourIndices()) {
        drawPolylines(ctx, state.data.contours[i].overview, STYLE.faintOverviewRgba, STYLE.faintWidth);
      }
    }

    if (!state.openWindow && state.selected != null) {
      drawPolylines(ctx, selectedContourFor('overview'), STYLE.selectedContourRgba, STYLE.selectedOverviewWidth);
    }

    drawWindowMarkers(ctx);
  }

  function redrawZoom() {
    const win = state.openWindow;
    if (!win || !els.zoom.bg || !els.zoom.view) return;

    const { ctx, canvas, bg, view } = els.zoom;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(bg, 0, 0, view.width, view.height);

    if (state.showAll) {
      for (const i of faintContourIndices()) {
        drawPolylines(ctx, state.data.contours[i].windows?.[win.id], STYLE.faintZoomRgba, STYLE.faintWidth);
      }
    }

    if (state.selected != null) {
      drawPolylines(ctx, selectedContourFor(win.id), STYLE.selectedContourRgba, STYLE.selectedZoomWidth);
    }
  }

  function redraw() {
    if (!state.data) return;
    redrawOverview();
    redrawZoom();
    const text = readoutText();
    els.readout.textContent = state.openWindow ? '' : text;
    els.zoomReadout.textContent = state.openWindow ? text : '';
  }

  function fitZoomCanvas() {
    if (!state.openWindow || !els.zoom.view) return;
    const wrap = els.zoomWrap;
    const view = els.zoom.view;

    // clientWidth/clientHeight are layout dimensions and are not shrunk by the
    // pop-out transform animation. getBoundingClientRect() is transformed and
    // made the canvas approximately 8% of the intended size on first opening.
    const availableWidth = wrap.clientWidth;
    const availableHeight = wrap.clientHeight;
    if (!(availableWidth > 0 && availableHeight > 0)) return;

    const scale = Math.min(availableWidth / view.width, availableHeight / view.height);
    els.zoom.canvas.width = view.width;
    els.zoom.canvas.height = view.height;
    els.zoom.canvas.style.width = `${Math.max(1, view.width * scale)}px`;
    els.zoom.canvas.style.height = `${Math.max(1, view.height * scale)}px`;
  }

  function openWindow(win, evt) {
    state.openWindow = win;
    state.hoverWindow = win;
    clearHoverSelection();
    els.demo.classList.add('zoom-open');
    els.overlay.setAttribute('aria-hidden', 'false');
    els.zoomTitle.textContent = win.label;

    const stageRect = els.stage.getBoundingClientRect();
    const p = evt
      ? { x: evt.clientX - stageRect.left, y: evt.clientY - stageRect.top }
      : { x: stageRect.width / 2, y: stageRect.height / 2 };
    els.zoomShell.style.setProperty('--origin-x', `${p.x}px`);
    els.zoomShell.style.setProperty('--origin-y', `${p.y}px`);

    const view = state.data.views.windows[win.id];
    els.zoom.bg = els.images[win.id];
    els.zoom.mapCtx = els.maps[win.id];
    els.zoom.view = view;

    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        fitZoomCanvas();
        redraw();
        reportEmbeddedHeight();
      });
    });
  }

  function closeWindow() {
    state.openWindow = null;
    clearHoverSelection();
    els.demo.classList.remove('zoom-open');
    els.overlay.setAttribute('aria-hidden', 'true');
    redraw();
    reportEmbeddedHeight();
  }

  async function setup() {
    els.demo = document.getElementById('boundaryDemo');
    els.stage = document.getElementById('demoStage');
    els.readout = document.getElementById('gReadout');
    els.zoomReadout = document.getElementById('zoomReadout');
    els.zoomTitle = document.getElementById('zoomTitle');
    els.overlay = document.getElementById('zoomOverlay');
    els.zoomShell = document.getElementById('zoomShell');
    els.zoomWrap = document.querySelector('.zoom-canvas-wrap');

    lockEmbeddedViewport();

    const response = await fetch(`data.json?v=${Date.now()}`, { cache: 'no-store' });
    if (!response.ok) throw new Error(`data.json: HTTP ${response.status}`);
    state.data = await response.json();

    els.images = {};
    els.maps = {};

    const ovView = state.data.views.overview;
    const ovCanvas = document.getElementById('overviewCanvas');
    ovCanvas.width = ovView.width;
    ovCanvas.height = ovView.height;
    els.overview = {
      canvas: ovCanvas,
      ctx: ovCanvas.getContext('2d'),
      bg: await loadImage(ovView.background, 'overview background image'),
      view: ovView,
    };

    const ovMap = await loadImage(ovView.indexMap, 'overview hover index map');
    const ovMapCanvas = document.createElement('canvas');
    ovMapCanvas.width = ovView.width;
    ovMapCanvas.height = ovView.height;
    const ovMapCtx = ovMapCanvas.getContext('2d', { willReadFrequently: true });
    ovMapCtx.drawImage(ovMap, 0, 0);
    els.overview.mapCtx = ovMapCtx;

    for (const win of state.data.windows) {
      const view = state.data.views.windows[win.id];
      els.images[win.id] = await loadImage(view.background, `${win.label} background image`);
      const mapImg = await loadImage(view.indexMap, `${win.label} hover index map`);
      const mapCanvas = document.createElement('canvas');
      mapCanvas.width = view.width;
      mapCanvas.height = view.height;
      const mapCtx = mapCanvas.getContext('2d', { willReadFrequently: true });
      mapCtx.drawImage(mapImg, 0, 0);
      els.maps[win.id] = mapCtx;
    }

    const zoomCanvas = document.getElementById('zoomCanvas');
    els.zoom = {
      canvas: zoomCanvas,
      ctx: zoomCanvas.getContext('2d'),
      bg: null,
      mapCtx: null,
      view: null,
    };

    ovCanvas.addEventListener('mouseenter', evt => {
      state.pointerInOverview = true;
      state.overviewPointer = canvasPoint(evt, ovCanvas);
      state.hoverWindow = findNearWindow(state.overviewPointer);
      ovCanvas.style.cursor = state.hoverWindow ? 'pointer' : 'crosshair';
      clearHoverSelection();
      redraw();
    });
    ovCanvas.addEventListener('mouseleave', () => {
      state.pointerInOverview = false;
      state.overviewPointer = null;
      state.hoverWindow = null;
      ovCanvas.style.cursor = 'crosshair';
      clearHoverSelection();
      redraw();
    });
    ovCanvas.addEventListener('mousemove', evt => {
      const p = canvasPoint(evt, ovCanvas);
      state.overviewPointer = p;
      state.hoverWindow = findNearWindow(p);
      ovCanvas.style.cursor = state.hoverWindow ? 'pointer' : 'crosshair';
      if (!state.openWindow) {
        applyHover(getHoverFromMap(els.overview.mapCtx, p.x, p.y, ovView.width, ovView.height));
      }
      redraw();
    });
    ovCanvas.addEventListener('click', evt => {
      const p = canvasPoint(evt, ovCanvas);
      const win = findNearWindow(p);
      if (win) openWindow(win, evt);
    });

    zoomCanvas.addEventListener('mouseenter', () => {
      clearHoverSelection();
      redraw();
    });
    zoomCanvas.addEventListener('mousemove', evt => {
      if (!state.openWindow || !els.zoom.mapCtx || !els.zoom.view) return;
      const p = canvasPoint(evt, zoomCanvas);
      applyHover(getHoverFromMap(
        els.zoom.mapCtx,
        p.x,
        p.y,
        els.zoom.view.width,
        els.zoom.view.height,
      ));
      redraw();
    });
    zoomCanvas.addEventListener('mouseleave', () => {
      clearHoverSelection();
      redraw();
    });

    document.getElementById('zoomClose').addEventListener('click', closeWindow);
    document.addEventListener('keydown', evt => {
      if (evt.key === 'Escape' && state.openWindow) closeWindow();
    });

    const btn = document.getElementById('showAllBtn');
    btn.addEventListener('click', () => {
      state.showAll = !state.showAll;
      btn.textContent = state.showAll ? 'Hide faint contours' : 'Show faint contours';
      redraw();
    });

    els.zoomShell.addEventListener('transitionend', evt => {
      if (state.openWindow && (evt.propertyName === 'transform' || evt.propertyName === 'opacity')) {
        fitZoomCanvas();
        redraw();
      }
    });

    if ('ResizeObserver' in window) {
      const observer = new ResizeObserver(() => {
        if (state.openWindow) {
          fitZoomCanvas();
          redraw();
        }
      });
      observer.observe(els.zoomWrap);
      els.zoomResizeObserver = observer;

      if (state.embedded) {
        const embedObserver = new ResizeObserver(reportEmbeddedHeight);
        embedObserver.observe(els.demo);
        els.embedResizeObserver = embedObserver;
      }
    }

    window.addEventListener('resize', () => {
      if (state.openWindow) fitZoomCanvas();
      redraw();
      reportEmbeddedHeight();
    });

    clearHoverSelection();
    redraw();
    reportEmbeddedHeight();
    if (document.fonts?.ready) {
      document.fonts.ready.then(reportEmbeddedHeight);
    }
  }

  setup().catch(err => {
    console.error(err);
    const r = document.getElementById('gReadout');
    if (r) r.textContent = 'failed to load demo data';
  });
})();
