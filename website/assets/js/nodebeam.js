//console.log("🚀 nodebeam.js loaded!");
document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("pendulum-canvas");
  const ctx = canvas.getContext("2d");

  let cols, rows, spacing = 50;
  let grid = [];
  let mouse = { x: -9999, y: -9999 };
  let prevMouse = { x: mouse.x, y: mouse.y };
  let mouseSpeed = 0;
  let lastUpdate = performance.now();
  let mouseReady = false;

  function resize() {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
  
    cols = Math.floor(canvas.width / spacing);
    rows = Math.floor(canvas.height / spacing);
  
    grid = [];
    for (let y = 0; y < rows; y++) {
      grid[y] = [];
      for (let x = 0; x < cols; x++) {
        const px = x * spacing + spacing / 2;
        const py = y * spacing + spacing / 2;
        grid[y][x] = {
          x: px,
          y: py,
          x0: px,
          y0: py,
          vx: 0,
          vy: 0
        };
      }
    }
  }
  

  window.addEventListener("resize", resize);
  resize();

  canvas.addEventListener("mousemove", e => {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
  
    if (!mouseReady) {
      mouseReady = true;
      prevMouse.x = x;
      prevMouse.y = y;
      mouse.x = x;
      mouse.y = y;
      console.log("Initial mouse move at:", x, y);
      return;
    }
  
    mouse.x = x;
    mouse.y = y;
    console.log("Mouse moved to:", x, y);
  });
  
  setInterval(() => {
    console.log("Mouse:", mouse.x.toFixed(2), mouse.y.toFixed(2));
  }, 1000);
  
  const k = 0.1;
  const damping = 0.05;
  let repulsion = 80;
  const influenceRadius = 150;

  function step() {
    const now = performance.now();
    const dt = (now - lastUpdate);
    lastUpdate = now;


    repulsion = 0;
    if (mouseReady) {
        // Compute mouse speed
        const dx = mouse.x - prevMouse.x;
        const dy = mouse.y - prevMouse.y;
        const speed = Math.sqrt(dx * dx + dy * dy) / (dt || 0.016); // 1 <-> 100 px / sec
        mouseSpeed = 0.5 * mouseSpeed + 0.5 * speed;

        prevMouse.x = mouse.x;
        prevMouse.y = mouse.y;

        repulsion = mouseSpeed;
    }

    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < cols; x++) {
        const p = grid[y][x];
        let fx = -k * (p.x - p.x0);
        let fy = -k * (p.y - p.y0);

        const dx = p.x - mouse.x;
        const dy = p.y - mouse.y;
        const dist2 = dx * dx + dy * dy;

        if (dist2 < influenceRadius * influenceRadius) {
          const dist = Math.sqrt(dist2);
          const x = dist2 / (influenceRadius * influenceRadius);
          const strength = repulsion * (1 - x)**2;
          fx += strength * (dx / dist);
          fy += strength * (dy / dist);
        }

        p.vx = (p.vx + fx) * (1 - damping);
        p.vy = (p.vy + fy) * (1 - damping);
        p.x += p.vx;
        p.y += p.vy;
      }
    }
  }

  function draw() {
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const maxStrain = 0.3;

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
        const p = grid[y][x];

        if (x < cols - 1) drawLink(p, grid[y][x + 1], spacing);           // right
        if (y < rows - 1) drawLink(p, grid[y + 1][x], spacing);           // down

        if (x < cols - 1 && y < rows - 1) {
        drawLink(p, grid[y + 1][x + 1], 1.41 * spacing);                       // down-right diagonal
        }
        if (x < cols - 1 && y > 0) {
        drawLink(p, grid[y - 1][x + 1], 1.41 * spacing);                       // up-right diagonal
        }
    }
  }

  function drawLink(a, b, defSpacing) {
  const dx = b.x - a.x;
  const dy = b.y - a.y;
  const dist = Math.sqrt(dx * dx + dy * dy);
  const strain = (dist - defSpacing) / defSpacing;
  const absStrain = Math.abs(strain);

  // Clamp for extreme cases
  const s = Math.max(-maxStrain, Math.min(strain, maxStrain));

  // Color: compression (red) to extension (blue)
  const red = s < 0 ? 255 : Math.round(255 * (1 - s / maxStrain));
  const blue = s > 0 ? 255 : Math.round(255 * (1 + s / maxStrain));
  const alpha = 0.4 * Math.min(1, absStrain / maxStrain);

  ctx.strokeStyle = `rgba(${red},0,0,${alpha})`;

  // Line width grows with strain
  ctx.lineWidth = 0.5 + 2.5 * Math.min(1, absStrain / maxStrain);

  ctx.beginPath();
  ctx.moveTo(a.x, a.y);
  ctx.lineTo(b.x, b.y);
  ctx.stroke();
}


  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const p = grid[y][x];

      // distance to mouse
      const dx = p.x - mouse.x;
      const dy = p.y - mouse.y;
      const dist2 = dx * dx + dy * dy;

      let r = 0;
      if (dist2 < influenceRadius * influenceRadius) {
        const dist = Math.sqrt(dist2);
        const x = dist2 / (influenceRadius * influenceRadius);
        const strength = 5 * repulsion * (1 - x) * (1 - x);
        r = Math.min(strength, 20.0)

        ctx.beginPath();
        ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255, 140, 0, 0.8)";
        ctx.fill();
      }
    }
  }
}


  function animate() {
    step();
    draw();
    requestAnimationFrame(animate);
  }

  animate();
});