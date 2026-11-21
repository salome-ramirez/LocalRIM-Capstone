/* The app page: chapter links seek the demo video.
   Every link with data-t (seconds) jumps the video there and plays it; links outside the player bring it into view first.
   The chapter the video is in is marked aria-current. Without JavaScript the chapter links open the file at #t=. */
(function () {
  const v = document.getElementById("demo");
  if (!v) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const chapters = [...document.querySelectorAll(".chapters a[data-t]")];
  const starts = chapters.map(a => +a.dataset.t);

  function seek(t) {
    const go = () => { v.currentTime = t; const p = v.play(); if (p) p.catch(() => {}); };
    if (v.readyState >= 1) go(); else { v.addEventListener("loadedmetadata", go, { once: true }); v.load(); }
  }

  document.addEventListener("click", e => {
    const a = e.target.closest("a[data-t]");
    if (!a) return;
    e.preventDefault();
    if (!a.closest(".chapters")) v.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
    seek(+a.dataset.t);
  });

  function mark() {
    let i = -1;
    starts.forEach((s, k) => { if (v.currentTime >= s - 0.25) i = k; });
    chapters.forEach((a, k) => { if (k === i) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
  }
  v.addEventListener("timeupdate", mark);
  v.addEventListener("seeked", mark);
})();

/* The workflow figure fades in by row once, the first time it comes into view (like Figure 3.9's schematic).
   Hidden only after this script runs, so the figure shows without it; reduced motion: no animation. */
(() => {
  const fl = document.querySelector(".flow");
  if (!fl || !("IntersectionObserver" in window) || matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  fl.classList.add("anim");
  const io = new IntersectionObserver(es => { if (es.some(e => e.isIntersecting)) { fl.classList.add("in"); io.disconnect(); } }, { threshold: .2 });
  io.observe(fl);
})();
