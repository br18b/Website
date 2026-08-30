'use strict';

/* global WebGL2RenderingContext */

class MandelbrotWebGLRenderer {
  static HARD_MAX_ITER = 4096;
  static HARD_POST_ESCAPE_STEPS = 64;

  constructor(canvas, data) {
    this.canvas = canvas;
    this.data = data;
    this.lost = false;
    this.gl = canvas.getContext('webgl2', {
      alpha: false,
      antialias: false,
      depth: false,
      stencil: false,
      preserveDrawingBuffer: false,
      powerPreference: 'high-performance',
    });
    if (!this.gl) throw new Error('WebGL2 is unavailable.');

    this._onContextLost = event => {
      event.preventDefault();
      this.lost = true;
      if (typeof this.onContextLost === 'function') this.onContextLost();
    };
    this._onContextRestored = () => {
      this.lost = false;
      this._initialize();
      if (typeof this.onContextRestored === 'function') this.onContextRestored();
    };
    canvas.addEventListener('webglcontextlost', this._onContextLost, false);
    canvas.addEventListener('webglcontextrestored', this._onContextRestored, false);

    this._initialize();
  }

  _compileShader(type, source) {
    const gl = this.gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader) || 'Unknown shader compilation error';
      gl.deleteShader(shader);
      throw new Error(log);
    }
    return shader;
  }

  _createProgram(vertexSource, fragmentSource) {
    const gl = this.gl;
    const vertex = this._compileShader(gl.VERTEX_SHADER, vertexSource);
    const fragment = this._compileShader(gl.FRAGMENT_SHADER, fragmentSource);
    const program = gl.createProgram();
    gl.attachShader(program, vertex);
    gl.attachShader(program, fragment);
    gl.linkProgram(program);
    gl.deleteShader(vertex);
    gl.deleteShader(fragment);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(program) || 'Unknown shader link error';
      gl.deleteProgram(program);
      throw new Error(log);
    }
    return program;
  }

  _initialize() {
    const gl = this.gl;

    const vertexSource = `#version 300 es
      precision highp float;
      const vec2 POSITIONS[3] = vec2[3](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
      );
      out vec2 vUv;
      void main() {
        vec2 position = POSITIONS[gl_VertexID];
        vUv = 0.5 * position + 0.5;
        gl_Position = vec4(position, 0.0, 1.0);
      }
    `;

    const fragmentSource = `#version 300 es
      precision highp float;
      precision highp int;

      in vec2 vUv;
      out vec4 fragColor;

      uniform vec2 uResolution;
      uniform vec2 uCenterRe;
      uniform vec2 uCenterIm;
      uniform float uSpanX;
      uniform int uMaxIter;
      uniform float uEscape2;
      uniform float uStabilityTolerance;
      uniform int uStabilitySteps;
      uniform int uPostEscapeMaxSteps;
      uniform float uLogGMin;
      uniform float uLogGMax;
      uniform float uGamma;
      uniform int uBoundaryMapping;
      uniform vec3 uInteriorRgb;
      uniform sampler2D uPalette;
      uniform int uUseDoubleSingle;

      const int HARD_MAX_ITER = ${MandelbrotWebGLRenderer.HARD_MAX_ITER};
      const int HARD_POST_ESCAPE_STEPS = ${MandelbrotWebGLRenderer.HARD_POST_ESCAPE_STEPS};
      const float LOG10_2 = 0.30102999566398119521;
      const float SPLITTER = 4097.0;

      vec2 dsNormalize(float hi, float lo) {
        float sum = hi + lo;
        return vec2(sum, lo - (sum - hi));
      }

      vec2 dsAdd(vec2 a, vec2 b) {
        float sum = a.x + b.x;
        float virtualB = sum - a.x;
        float error = (a.x - (sum - virtualB)) + (b.x - virtualB);
        error += a.y + b.y;
        return dsNormalize(sum, error);
      }

      vec2 dsNeg(vec2 a) {
        return vec2(-a.x, -a.y);
      }

      vec2 dsSub(vec2 a, vec2 b) {
        return dsAdd(a, dsNeg(b));
      }

      vec2 dsMul(vec2 a, vec2 b) {
        float product = a.x * b.x;

        float conA = a.x * SPLITTER;
        float aHi = conA - (conA - a.x);
        float aLo = a.x - aHi;
        float conB = b.x * SPLITTER;
        float bHi = conB - (conB - b.x);
        float bLo = b.x - bHi;

        float error = ((aHi * bHi - product) + aHi * bLo + aLo * bHi) + aLo * bLo;
        error += a.x * b.y + a.y * b.x;
        return dsNormalize(product, error);
      }

      vec2 dsMulFloat(vec2 a, float b) {
        return dsMul(a, vec2(b, 0.0));
      }

      float dsValue(vec2 value) {
        return value.x + value.y;
      }

      bool insideAnalyticInterior(float x, float y) {
        float dx = x - 0.25;
        float q = dx * dx + y * y;
        if (q * (q + dx) <= 0.25 * y * y) return true;
        dx = x + 1.0;
        return dx * dx + y * y <= 0.0625;
      }

      float potentialAfterEscape(vec2 z, vec2 c, int escapedAt) {
        float log10G = uLogGMax;
        float previous = 1.0e30;
        int stable = 0;
        int iteration = escapedAt;

        for (int extra = 0; extra < HARD_POST_ESCAPE_STEPS; ++extra) {
          float abs2 = dot(z, z);
          if (!isinf(abs2) && !isnan(abs2) && abs2 > 1.0) {
            float logAbs = 0.5 * log(abs2);
            log10G = log(logAbs) / log(10.0) - float(iteration) * LOG10_2;
          } else {
            return uLogGMax;
          }

          if (abs(log10G - previous) <= uStabilityTolerance) {
            stable += 1;
          } else {
            stable = 0;
          }
          if (stable >= uStabilitySteps) break;
          if (extra + 1 >= uPostEscapeMaxSteps) break;
          if (iteration >= uMaxIter) break;

          previous = log10G;
          float zr2 = z.x * z.x;
          float zi2 = z.y * z.y;
          float nextY = 2.0 * z.x * z.y;
          z = vec2(zr2 - zi2 + c.x, nextY + c.y);
          iteration += 1;
        }
        return log10G;
      }

      bool iterateFloat(vec2 c, out float log10G) {
        vec2 z = vec2(0.0);
        for (int i = 0; i < HARD_MAX_ITER; ++i) {
          if (i >= uMaxIter) break;
          float zr2 = z.x * z.x;
          float zi2 = z.y * z.y;
          z = vec2(zr2 - zi2 + c.x, 2.0 * z.x * z.y + c.y);
          float abs2 = dot(z, z);
          if (abs2 > uEscape2) {
            log10G = potentialAfterEscape(z, c, i + 1);
            return true;
          }
        }
        return false;
      }

      bool iterateDoubleSingle(vec2 cRe, vec2 cIm, out float log10G) {
        vec2 zr = vec2(0.0);
        vec2 zi = vec2(0.0);
        for (int i = 0; i < HARD_MAX_ITER; ++i) {
          if (i >= uMaxIter) break;
          vec2 zr2 = dsMul(zr, zr);
          vec2 zi2 = dsMul(zi, zi);
          vec2 nextIm = dsAdd(dsMulFloat(dsMul(zr, zi), 2.0), cIm);
          vec2 nextRe = dsAdd(dsSub(zr2, zi2), cRe);
          zr = nextRe;
          zi = nextIm;

          vec2 z = vec2(dsValue(zr), dsValue(zi));
          float abs2 = dot(z, z);
          if (abs2 > uEscape2) {
            log10G = potentialAfterEscape(z, vec2(dsValue(cRe), dsValue(cIm)), i + 1);
            return true;
          }
        }
        return false;
      }

      void main() {
        float aspect = uResolution.y / uResolution.x;
        float offsetX = (vUv.x - 0.5) * uSpanX;
        float offsetY = (vUv.y - 0.5) * uSpanX * aspect;

        vec2 cRe = dsAdd(uCenterRe, vec2(offsetX, 0.0));
        vec2 cIm = dsAdd(uCenterIm, vec2(offsetY, 0.0));
        float cr = dsValue(cRe);
        float ci = dsValue(cIm);

        if (uUseDoubleSingle == 0 && insideAnalyticInterior(cr, ci)) {
          fragColor = vec4(uInteriorRgb, 1.0);
          return;
        }

        float log10G = uLogGMin;
        bool escaped;
        if (uUseDoubleSingle != 0) {
          escaped = iterateDoubleSingle(cRe, cIm, log10G);
        } else {
          escaped = iterateFloat(vec2(cr, ci), log10G);
        }

        if (!escaped) {
          fragColor = vec4(uInteriorRgb, 1.0);
          return;
        }

        float t = clamp((log10G - uLogGMin) / (uLogGMax - uLogGMin), 0.0, 1.0);
        float palettePosition = uBoundaryMapping != 0
          ? pow(1.0 - t, uGamma)
          : pow(t, uGamma);
        vec3 color = texture(uPalette, vec2(palettePosition, 0.5)).rgb;
        fragColor = vec4(color, 1.0);
      }
    `;

    this.program = this._createProgram(vertexSource, fragmentSource);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);

    this.uniforms = {};
    const names = [
      'uResolution', 'uCenterRe', 'uCenterIm', 'uSpanX', 'uMaxIter',
      'uEscape2', 'uStabilityTolerance', 'uStabilitySteps',
      'uPostEscapeMaxSteps', 'uLogGMin', 'uLogGMax', 'uGamma',
      'uBoundaryMapping', 'uInteriorRgb', 'uPalette', 'uUseDoubleSingle',
    ];
    for (const name of names) {
      this.uniforms[name] = gl.getUniformLocation(this.program, name);
    }

    this.paletteTexture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.paletteTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const palette = new Uint8Array(this.data.color.palette.flat());
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGB8,
      this.data.color.palette.length,
      1,
      0,
      gl.RGB,
      gl.UNSIGNED_BYTE,
      palette,
    );

    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
  }

  _splitDouble(value) {
    const high = Math.fround(value);
    return [high, value - high];
  }

  render({ displayWidth, displayHeight, centerX, centerY, spanX, resolutionScale = 1.0 }) {
    if (this.lost) return false;
    const gl = this.gl;
    const scale = Math.max(0.1, Math.min(1.0, resolutionScale));
    const width = Math.max(1, Math.round(displayWidth * scale));
    const height = Math.max(1, Math.round(displayHeight * scale));

    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width;
      this.canvas.height = height;
    }

    const [centerReHi, centerReLo] = this._splitDouble(centerX);
    const [centerImHi, centerImLo] = this._splitDouble(centerY);
    const render = this.data.render;
    const doubleSingleThreshold = Number(render.webgl_double_single_threshold ?? 1.0e-5);
    const useDoubleSingle = spanX <= doubleSingleThreshold ? 1 : 0;

    gl.viewport(0, 0, width, height);
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);

    gl.uniform2f(this.uniforms.uResolution, width, height);
    gl.uniform2f(this.uniforms.uCenterRe, centerReHi, centerReLo);
    gl.uniform2f(this.uniforms.uCenterIm, centerImHi, centerImLo);
    gl.uniform1f(this.uniforms.uSpanX, spanX);
    gl.uniform1i(
      this.uniforms.uMaxIter,
      Math.max(1, Math.min(MandelbrotWebGLRenderer.HARD_MAX_ITER, render.max_iter | 0)),
    );
    const escapeRadius = Math.max(2.0, Number(render.escape_radius ?? 2.0));
    gl.uniform1f(this.uniforms.uEscape2, escapeRadius * escapeRadius);
    gl.uniform1f(
      this.uniforms.uStabilityTolerance,
      Math.max(1.0e-8, Number(render.potential_stability_tolerance ?? 1.0e-4)),
    );
    gl.uniform1i(
      this.uniforms.uStabilitySteps,
      Math.max(1, Math.min(16, render.potential_stability_steps | 0)),
    );
    gl.uniform1i(
      this.uniforms.uPostEscapeMaxSteps,
      Math.max(1, Math.min(
        MandelbrotWebGLRenderer.HARD_POST_ESCAPE_STEPS,
        render.post_escape_max_steps | 0,
      )),
    );
    gl.uniform1f(this.uniforms.uLogGMin, Math.log10(this.data.color.gmin));
    gl.uniform1f(this.uniforms.uLogGMax, Math.log10(this.data.color.gmax));
    gl.uniform1f(this.uniforms.uGamma, Number(this.data.color.gamma));
    gl.uniform1i(
      this.uniforms.uBoundaryMapping,
      String(this.data.color.mapping || 'boundary').toLowerCase() === 'boundary' ? 1 : 0,
    );
    const interior = this.data.color.interiorRgb;
    gl.uniform3f(
      this.uniforms.uInteriorRgb,
      interior[0] / 255,
      interior[1] / 255,
      interior[2] / 255,
    );
    gl.uniform1i(this.uniforms.uUseDoubleSingle, useDoubleSingle);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.paletteTexture);
    gl.uniform1i(this.uniforms.uPalette, 0);

    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.flush();
    return true;
  }

  destroy() {
    this.canvas.removeEventListener('webglcontextlost', this._onContextLost, false);
    this.canvas.removeEventListener('webglcontextrestored', this._onContextRestored, false);
    const gl = this.gl;
    if (this.paletteTexture) gl.deleteTexture(this.paletteTexture);
    if (this.vao) gl.deleteVertexArray(this.vao);
    if (this.program) gl.deleteProgram(this.program);
  }
}

window.MandelbrotWebGLRenderer = MandelbrotWebGLRenderer;
