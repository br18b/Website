document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-lazy-demo]").forEach((demo) => {
    const mount = demo.querySelector("[data-lazy-demo-mount]");
    const buttons = demo.querySelectorAll("[data-lazy-demo-button]");
    const src = demo.dataset.src;
    const title = demo.dataset.title || "Interactive visualization";
    const loadingText = demo.dataset.loadingText || "Loading demo…";

    if (!mount || !src || buttons.length === 0) return;

    const setButtonsLoading = () => {
      buttons.forEach((button) => {
        button.disabled = true;
        button.textContent = loadingText;
      });
    };

    const setButtonsLoaded = () => {
      buttons.forEach((button) => {
        button.disabled = true;
        button.textContent = "Demo loaded";
      });
    };

    const loadDemo = () => {
      if (demo.dataset.loaded === "true" || demo.dataset.loaded === "loading") {
        return;
      }

      demo.dataset.loaded = "loading";
      demo.classList.add("is-loading");
      setButtonsLoading();

      // Give the browser one paint cycle to show the loader before Plotly starts working.
      requestAnimationFrame(() => {
        window.setTimeout(() => {
          const iframe = document.createElement("iframe");

          iframe.className = "lazy-demo__iframe";
          iframe.title = title;
          iframe.src = src;
          iframe.loading = "eager";
          iframe.setAttribute("allowfullscreen", "");

          iframe.addEventListener(
            "load",
            () => {
              demo.dataset.loaded = "true";
              demo.classList.remove("is-loading");
              demo.classList.add("is-loaded");
              setButtonsLoaded();
            },
            { once: true }
          );

          mount.replaceChildren(iframe);
        }, 120);
      });
    };

    buttons.forEach((button) => {
      button.addEventListener("click", loadDemo);
    });
  });
});