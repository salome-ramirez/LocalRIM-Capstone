/* Stepper: one stage at a time, for "How the labels were made" (01).
   Previous/Next buttons, numbered step markers (one tab stop, roving focus), Left/Right/Home/End anywhere in the stepper,
   swipe on the screenshot (touch and pen only; vertical scrolling is left alone), a polite live announcement of the new stage.
   No autoplay. The crossfade is CSS (casefile.css); under reduced motion it is instant. Inactive slides are inert and hidden from
   assistive technology, but stay in the page. */
(function () {
  document.querySelectorAll("[data-stepper]").forEach(root => {
    const slides = [...root.querySelectorAll(".st-slide")];
    const dots = [...root.querySelectorAll(".st-dot")];
    const prev = root.querySelector(".st-prev"), next = root.querySelector(".st-next");
    const status = root.querySelector("[data-st-status]");
    const n = slides.length;
    let cur = 0;

    function go(i, opt = {}) {
      i = Math.max(0, Math.min(n - 1, i));
      if (i === cur) return;
      slides[cur].classList.remove("is-current");
      slides[cur].setAttribute("aria-hidden", "true");
      slides[cur].inert = true;
      slides[i].classList.add("is-current");
      slides[i].removeAttribute("aria-hidden");
      slides[i].inert = false;
      const img = slides[i].querySelector("img");
      if (img && img.loading === "lazy") img.loading = "eager";          // make sure it is there before the fade ends
      dots.forEach((d, k) => {
        if (k === i) d.setAttribute("aria-current", "step"); else d.removeAttribute("aria-current");
        d.tabIndex = k === i ? 0 : -1;
      });
      prev.setAttribute("aria-disabled", String(i === 0));
      next.setAttribute("aria-disabled", String(i === n - 1));
      cur = i;
      if (status) status.textContent = slides[i].getAttribute("aria-label");
      if (opt.focusDot) dots[i].focus();
      // warm the neighbour so the next fade never shows an empty frame
      const nb = slides[i + 1] && slides[i + 1].querySelector("img");
      if (nb && nb.loading === "lazy") nb.loading = "eager";
    }

    prev.addEventListener("click", () => go(cur - 1));
    next.addEventListener("click", () => go(cur + 1));
    dots.forEach((d, k) => d.addEventListener("click", () => go(k)));

    root.addEventListener("keydown", e => {
      if (e.altKey || e.ctrlKey || e.metaKey) return;
      const onDot = e.target.classList && e.target.classList.contains("st-dot");
      const map = { ArrowLeft: cur - 1, ArrowRight: cur + 1, Home: 0, End: n - 1 };
      if (!(e.key in map)) return;
      e.preventDefault();
      go(map[e.key], { focusDot: onDot });
    });

    // swipe: horizontal intent only, on the screenshot; distance or a quick flick
    let sx = 0, sy = 0, st = 0, id = null;
    root.querySelector(".st-slides").addEventListener("pointerdown", e => {
      if (e.pointerType === "mouse" || !e.target.closest(".st-shot")) return;
      id = e.pointerId; sx = e.clientX; sy = e.clientY; st = performance.now();
    });
    root.querySelector(".st-slides").addEventListener("pointerup", e => {
      if (e.pointerId !== id) return;
      id = null;
      const dx = e.clientX - sx, dy = e.clientY - sy, v = Math.abs(dx) / Math.max(1, performance.now() - st);
      if (Math.abs(dx) < Math.abs(dy) * 1.5) return;               // mostly vertical: a scroll, not a swipe
      if (Math.abs(dx) >= 48 || (Math.abs(dx) >= 16 && v > 0.11)) go(dx < 0 ? cur + 1 : cur - 1);
    });
    root.querySelector(".st-slides").addEventListener("pointercancel", () => { id = null; });
  });
})();
