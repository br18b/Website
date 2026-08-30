document.addEventListener('DOMContentLoaded', () => {
  const storageKey = 'rabatinb.postFilter';
  const validModes = new Set(['all', 'featured']);
  const scopes = document.querySelectorAll('[data-post-filter-scope]');

  if (!scopes.length) return;

  const readMode = () => {
    try {
      const stored = window.localStorage.getItem(storageKey);
      return validModes.has(stored) ? stored : 'all';
    } catch (error) {
      return 'all';
    }
  };

  const saveMode = (mode) => {
    try {
      window.localStorage.setItem(storageKey, mode);
    } catch (error) {
      // Storage can be unavailable in some private/browser modes. The filter still works for this page view.
    }
  };

  const applyMode = (mode) => {
    scopes.forEach((scope) => {
      scope.dataset.postFilterMode = mode;

      scope.querySelectorAll('[data-post-filter-button]').forEach((button) => {
        const active = button.dataset.postFilterButton === mode;
        button.classList.toggle('is-active', active);
        button.setAttribute('aria-pressed', active ? 'true' : 'false');
      });

      scope.querySelectorAll('[data-post-featured]').forEach((card) => {
        const isFeatured = card.dataset.postFeatured === 'true';
        card.hidden = mode === 'featured' && !isFeatured;
      });
    });
  };

  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-post-filter-button]');
    if (!button) return;

    const mode = button.dataset.postFilterButton;
    if (!validModes.has(mode)) return;

    saveMode(mode);
    applyMode(mode);
  });

  applyMode(readMode());
});
