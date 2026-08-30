'use strict';

let DATA = null;
let latestToken = 0;

const LOG10_2 = Math.log10(2);
const nextTick = () => new Promise(resolve => setTimeout(resolve, 0));

function insideCardioid(x, y) {
  const dx = x - 0.25;
  const q = dx * dx + y * y;
  return q * (q + dx) <= 0.25 * y * y;
}

function insidePeriod2(x, y) {
  const dx = x + 1.0;
  return dx * dx + y * y <= 0.0625;
}

function knownInterior(x, y) {
  // Fitted component shapes are excellent display and hit-test geometry, but
  // they are approximations.  Using them as a render shortcut could paint a
  // tiny exterior sliver black.  Keep only the two exact analytic tests here.
  return insideCardioid(x, y) || insidePeriod2(x, y);
}

function colorPixel(cr, ci, out, offset) {
  const interior = DATA.color.interiorRgb;
  if (knownInterior(cr, ci)) {
    out[offset] = interior[0];
    out[offset + 1] = interior[1];
    out[offset + 2] = interior[2];
    out[offset + 3] = 255;
    return;
  }

  let zr = 0.0;
  let zi = 0.0;
  const escape2 = DATA.render.escape_radius * DATA.render.escape_radius;
  const stabilityTol = DATA.render.potential_stability_tolerance || 1.0e-4;
  const stabilitySteps = Math.max(1, DATA.render.potential_stability_steps | 0);
  const postEscapeMaxSteps = Math.max(1, DATA.render.post_escape_max_steps | 0);
  let escapedAt = 0;
  let abs2 = 0.0;
  let log10G = 0.0;
  let previousLog10G = Number.NaN;
  let stableCount = 0;

  for (let iteration = 1; iteration <= DATA.render.max_iter; iteration++) {
    const zr2 = zr * zr;
    const zi2 = zi * zi;
    zi = 2.0 * zr * zi + ci;
    zr = zr2 - zi2 + cr;
    abs2 = zr * zr + zi * zi;

    if (!escapedAt && abs2 > escape2) {
      escapedAt = iteration;
    }
    if (!escapedAt) continue;

    if (!Number.isFinite(abs2)) {
      log10G = Math.log10(DATA.color.gmax);
      break;
    }

    if (abs2 > 1.0) {
      const logAbs = 0.5 * Math.log(abs2);
      log10G = Math.log10(logAbs) - iteration * LOG10_2;
    } else {
      log10G = Math.log10(DATA.color.gmax);
    }

    if (Number.isFinite(previousLog10G) && Math.abs(log10G - previousLog10G) <= stabilityTol) {
      stableCount += 1;
    } else {
      stableCount = 0;
    }
    previousLog10G = log10G;

    if (stableCount >= stabilitySteps || (iteration - escapedAt) >= postEscapeMaxSteps) {
      break;
    }
  }

  if (!escapedAt) {
    out[offset] = interior[0];
    out[offset + 1] = interior[1];
    out[offset + 2] = interior[2];
    out[offset + 3] = 255;
    return;
  }

  const lo = Math.log10(DATA.color.gmin);
  const hi = Math.log10(DATA.color.gmax);
  let t = (log10G - lo) / (hi - lo);
  t = Math.max(0, Math.min(1, t));
  const boundaryMapping = String(DATA.color.mapping || 'boundary').toLowerCase() === 'boundary';
  const palettePosition = boundaryMapping
    ? Math.pow(1 - t, DATA.color.gamma)
    : Math.pow(t, DATA.color.gamma);
  const index = Math.max(0, Math.min(255, Math.round(255 * palettePosition)));
  const rgb = DATA.color.palette[index];
  out[offset] = rgb[0];
  out[offset + 1] = rgb[1];
  out[offset + 2] = rgb[2];
  out[offset + 3] = 255;
}

async function renderRequest(request) {
  const { token, width, height, centerX, centerY, spanX, passes } = request;
  const chunkRows = Math.max(1, DATA.render.worker_chunk_rows | 0);

  for (const scale of passes) {
    if (token !== latestToken) return;
    const renderWidth = Math.max(1, Math.ceil(width / scale));
    const renderHeight = Math.max(1, Math.ceil(height / scale));
    const spanY = spanX * height / width;
    const image = new ImageData(renderWidth, renderHeight);
    const out = image.data;

    for (let y0 = 0; y0 < renderHeight; y0 += chunkRows) {
      if (token !== latestToken) return;
      const y1 = Math.min(renderHeight, y0 + chunkRows);
      for (let py = y0; py < y1; py++) {
        const ci = centerY + (0.5 - (py + 0.5) / renderHeight) * spanY;
        for (let px = 0; px < renderWidth; px++) {
          const cr = centerX + ((px + 0.5) / renderWidth - 0.5) * spanX;
          colorPixel(cr, ci, out, 4 * (py * renderWidth + px));
        }
      }
      await nextTick();
    }

    if (token !== latestToken) return;
    const canvas = new OffscreenCanvas(renderWidth, renderHeight);
    canvas.getContext('2d').putImageData(image, 0, 0);
    const bitmap = canvas.transferToImageBitmap();
    postMessage({ type: 'frame', token, scale, bitmap }, [bitmap]);
  }
  if (token === latestToken) postMessage({ type: 'done', token });
}

onmessage = event => {
  const message = event.data;
  if (message.type === 'init') {
    DATA = message.data;
    postMessage({ type: 'ready' });
    return;
  }
  if (message.type === 'render') {
    latestToken = message.token;
    renderRequest(message).catch(error => {
      postMessage({ type: 'error', token: message.token, message: String(error?.stack || error) });
    });
  }
};
