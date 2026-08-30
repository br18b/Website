document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-gallery]').forEach((container) => {
    const gallery = container.querySelector('.scroll-gallery');
    const prev = container.querySelector('[data-gallery-prev]');
    const next = container.querySelector('[data-gallery-next]');

    if (!gallery) return;

    const step = () => {
      const firstImage = gallery.querySelector('img');
      return firstImage ? firstImage.clientWidth + 12 : Math.max(220, gallery.clientWidth * 0.75);
    };

    prev?.addEventListener('click', () => {
      gallery.scrollBy({ left: -step(), behavior: 'smooth' });
    });

    next?.addEventListener('click', () => {
      gallery.scrollBy({ left: step(), behavior: 'smooth' });
    });
  });
});
