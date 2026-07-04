document.addEventListener('DOMContentLoaded', () => {
  if (window.hljs) {
    window.hljs.highlightAll();
  }

  document.querySelectorAll('pre').forEach((pre) => {
    if (pre.closest('.code-copy-wrapper')) return;

    const wrapper = document.createElement('div');
    wrapper.className = 'code-copy-wrapper';

    const button = document.createElement('button');
    button.className = 'copy-code-button';
    button.type = 'button';
    button.textContent = 'Copy';

    pre.parentNode.insertBefore(wrapper, pre);
    wrapper.appendChild(button);
    wrapper.appendChild(pre);

    button.addEventListener('click', async () => {
      const text = pre.innerText.trim();
      try {
        await navigator.clipboard.writeText(text);
        button.textContent = 'Copied';
        setTimeout(() => { button.textContent = 'Copy'; }, 1600);
      } catch (error) {
        button.textContent = 'Failed';
        setTimeout(() => { button.textContent = 'Copy'; }, 1600);
      }
    });
  });
});
