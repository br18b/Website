document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-plot-carousel]').forEach((carousel) => {
    const img = carousel.querySelector('[data-plot-image]');
    const prev = carousel.querySelector('[data-plot-prev]');
    const next = carousel.querySelector('[data-plot-next]');
    if (!img) return;

    const values = (img.dataset.values || '').split(',').map((value) => value.trim()).filter(Boolean);
    const folder = (img.dataset.folder || '').replace(/\/$/, '');
    const prefix = img.dataset.prefix || '';
    const ext = img.dataset.ext || 'png';
    let index = 0;

    const update = () => {
      if (!values.length) return;
      img.src = `${folder}/${prefix}${values[index]}.${ext}`;
    };

    prev?.addEventListener('click', () => {
      index = (index - 1 + values.length) % values.length;
      update();
    });

    next?.addEventListener('click', () => {
      index = (index + 1) % values.length;
      update();
    });
  });
});
