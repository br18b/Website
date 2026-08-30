document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-lazy-demo]").forEach((demo) => {
    const mount = demo.querySelector("[data-lazy-demo-mount]");
    const buttons = demo.querySelectorAll("[data-lazy-demo-button]");
    const src = demo.dataset.src;
    const title = demo.dataset.title || "Interactive visualization";
    const loadingText = demo.dataset.loadingText || "Loading demo…";
    const fitContent = demo.dataset.fitContent === "true";
    const fitMinHeight = Number.parseFloat(demo.dataset.fitMinHeight || "360");
    const fitMaxHeight = Number.parseFloat(demo.dataset.fitMaxHeight || "1200");
    let iframe = null;

    const applyFittedHeight = (rawHeight) => {
      if (!fitContent || !Number.isFinite(rawHeight) || rawHeight <= 0) return;
      const minHeight = Number.isFinite(fitMinHeight) ? fitMinHeight : 360;
      const maxHeight = Number.isFinite(fitMaxHeight) ? fitMaxHeight : 1200;
      const height = Math.ceil(Math.max(minHeight, Math.min(maxHeight, rawHeight)));
      demo.style.setProperty("--lazy-demo-height", `${height}px`);
      demo.style.setProperty("--lazy-demo-mobile-height", `${height}px`);
    };

    const onResizeMessage = (event) => {
      if (!fitContent || !iframe || event.source !== iframe.contentWindow) return;
      if (event.origin !== window.location.origin) return;
      if (!event.data || event.data.type !== "lazy-demo:resize") return;
      applyFittedHeight(Number(event.data.height));
    };

    if (fitContent) {
      window.addEventListener("message", onResizeMessage);
    }

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
          iframe = document.createElement("iframe");

          iframe.className = "lazy-demo__iframe";
          iframe.title = title;
          iframe.src = src;
          iframe.loading = "eager";
          iframe.setAttribute("allowfullscreen", "");
          if (fitContent) {
            iframe.setAttribute("scrolling", "no");
            iframe.style.overflow = "hidden";
          }

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