/* Landing. Every number from data/site_data.json. */
(() => {
  "use strict";
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const get = (o, p) => p.split(".").reduce((x, k) => (x == null ? x : x[k]), o);
  const pct = v => `${Math.round(v * 100)}%`;
  const int = v => Math.round(v).toLocaleString("en-US");
  const short = n => n.replace(/\s*\(.*\)\s*$/, "");
  const FMT = { pct, int, num2: v => v.toFixed(2), date: v => String(v).replace(/\s+\d{1,2}:\d{2}$/, "") };
  let D;
  const CALC = {
    "archive.millions": () => (D.sample.funnel[0].count / 1e6).toFixed(2),
    "rag.vs": () => `${pct(D.final_test.rag.rag_top1)} vs ${pct(D.final_test.rag.classifier_top1)}`,
    "scale.seconds": () => `${D.scale_run.timing.reduce((a, t) => a + t.seconds, 0).toFixed(1)} s`,
    "read.backbone": () => short(D.ocr.engines.names[D.ocr.engines.backbone]),
    "read.vision": () => short(D.ocr.engines.names.D),
    "cutoff": () => D.final_test.cutoff.toFixed(2),
    "final.person": () => int(D.final_test.pages - D.final_test.auto_filed),
    "scale.share": () => pct(D.scale_run.auto_filed / D.scale_run.pages),   // Stage 6 card (428 of 1,000)
  };
  const fmt = (v, f) => (f && FMT[f] ? FMT[f](v) : typeof v === "number" ? int(v) : v);

  $$("[data-set-theme]").forEach(b => b.addEventListener("click", () => {
    const t = b.dataset.setTheme;
    if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
    $$("[data-set-theme]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  }));
  const q = new URLSearchParams(location.search);
  if (q.get("theme")) document.documentElement.dataset.theme = q.get("theme");

  // The process path is traced once across the six stages when the route first comes into view
  const route = document.querySelector(".route-wrap");
  if (route) {
    if (matchMedia("(prefers-reduced-motion: reduce)").matches || !("IntersectionObserver" in window)) route.classList.add("is-static");
    else { const io = new IntersectionObserver(([e]) => { if (e.isIntersecting) { route.classList.add("is-traced"); io.disconnect(); } }, { threshold: 0.5 }); io.observe(route); }
  }

  fetch("data/site_data.json", { cache: "no-store" })
    .then(r => r.json())
    .then(d => {
      D = d;
      $$("[data-field]").forEach(el => { const v = get(D, el.dataset.field); if (v != null) el.textContent = fmt(v, el.dataset.fmt); });
      // Data-sum="a-b" or "a+b", as in site.js (Stage 3's "the other 81 pages")
      $$("[data-sum]").forEach(el => { const m = el.dataset.sum.match(/^(.+?)([+-])(.+)$/); if (!m) return; const x = get(D, m[1]), y = get(D, m[3]);
        if (typeof x === "number" && typeof y === "number") el.textContent = fmt(m[2] === "+" ? x + y : x - y, "int"); });
      $$("[data-calc]").forEach(el => { const fn = CALC[el.dataset.calc]; if (fn) el.textContent = fn(); });
      // The key numbers' full detail goes in the shared tooltip (charts-ui.js), built from the data
      const F = D.final_test, A = F.scores.auto_accuracy;
      const TIPS = {
        auto: () => `Against a ${pct(F.target)} standard (95% interval ${pct(A.low)} to ${pct(A.high)})`,
        rag: () => `On the ${int(F.rag.review_pages)} pages sent to a person`,
        time: () => `On a box of ${int(D.scale_run.pages)} new pages`,
      };
      $$("[data-tip-calc]").forEach(el => { const fn = TIPS[el.dataset.tipCalc]; if (fn) el.setAttribute("data-tip", fn()); });
    })
    .catch(() => document.querySelector("main").insertAdjacentHTML("afterbegin", "<p><b>The data did not load.</b> Serve this folder over http (port 8001) and reload.</p>"));
})();
