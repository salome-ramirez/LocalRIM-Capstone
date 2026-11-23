/* Chart interaction shared by every page.
   1. One tooltip for every chart mark: hover (mouse), tap (touch), and keyboard (each chart is one
      tab stop; arrow keys move from mark to mark, Escape closes). Marks carry their text in data-tip;
      the older chart code's SVG <title> elements are adopted into data-tip automatically.
   2. One draw-in per chart, never repeated, done within 400 ms, the first time a chart comes into view,
      moving the way its form reads. Redraws (resizing, switching options) never replay it.
   Under prefers-reduced-motion the tooltip appears without movement and charts are drawn at once. */
(() => {
  "use strict";
  const HOSTS = ":is([data-chart], [data-tips])";   // [data-tips] = a group of non-chart marks (the Overview's key numbers)
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");

  /* Tooltip */
  const tip = document.createElement("div");
  tip.className = "chart-tip"; tip.setAttribute("role", "tooltip"); tip.id = "chart-tip";
  const live = document.createElement("p");
  live.className = "vh"; live.setAttribute("aria-live", "polite");
  document.body.append(tip, live);
  let current = null, pinned = false;

  function adopt() {
    document.querySelectorAll(`${HOSTS} svg title`).forEach(t => {
      const p = t.parentElement;
      if (!p) return;
      // a chart's own title names the whole chart: it becomes the chart's label, not a mark
      if (p.tagName.toLowerCase() === "svg") { if (!p.hasAttribute("aria-label")) p.setAttribute("aria-label", t.textContent.trim()); p.setAttribute("role", "img"); }
      else if (!p.hasAttribute("data-tip") && t.textContent.trim()) p.setAttribute("data-tip", t.textContent.trim());
      t.remove();
    });
    // record types (Appendix 01): their counts use the same tooltip as every other mark
    document.querySelectorAll(`${HOSTS} [data-count]`).forEach(el => {
      const name = el.firstElementChild ? el.firstElementChild.textContent : "";
      el.setAttribute("data-tip", `${name}: ${el.getAttribute("data-count")}`);
      el.removeAttribute("data-count"); el.removeAttribute("tabindex");
    });
    document.querySelectorAll(`${HOSTS} [title]`).forEach(el => { el.setAttribute("data-tip", el.getAttribute("title")); el.removeAttribute("title"); });
    document.querySelectorAll(HOSTS).forEach(h => {
      const has = !!h.querySelector("[data-tip]");
      if (has && !h.hasAttribute("tabindex")) {
        h.tabIndex = 0;
        h.setAttribute("role", "group");
        if (!h.hasAttribute("aria-label")) h.setAttribute("aria-label", "Chart. Use the arrow keys to read each mark.");
      }
    });
  }
  const marks = host => [...host.querySelectorAll("[data-tip]")].filter(m => m.getClientRects().length);

  function show(el, announce) {
    if (current && current !== el) current.classList.remove("tip-focus");
    current = el;
    tip.textContent = el.getAttribute("data-tip");
    tip.classList.add("is-on");
    const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect(), pad = 8;
    let x = r.left + r.width / 2 - t.width / 2, y = r.top - t.height - pad;
    if (y < pad) y = r.bottom + pad;                                   // flip below near the top
    x = Math.max(pad, Math.min(x, innerWidth - t.width - pad));        // keep inside the screen
    tip.style.left = `${Math.round(x)}px`; tip.style.top = `${Math.round(y)}px`;
    if (announce) { el.classList.add("tip-focus"); live.textContent = tip.textContent; }
  }
  function hide() {
    tip.classList.remove("is-on");
    if (current) current.classList.remove("tip-focus");
    current = null; pinned = false;
  }

  document.addEventListener("pointerover", e => {
    if (e.pointerType !== "mouse" || pinned) return;
    const m = e.target.closest && e.target.closest(`${HOSTS} [data-tip]`);
    if (m) show(m, false);
  });
  document.addEventListener("pointerout", e => {
    if (e.pointerType !== "mouse" || pinned) return;
    const m = e.target.closest && e.target.closest(`${HOSTS} [data-tip]`);
    if (m && (!e.relatedTarget || !m.contains(e.relatedTarget))) hide();
  });
  document.addEventListener("pointerdown", e => {
    const m = e.target.closest && e.target.closest(`${HOSTS} [data-tip]`);
    if (m && e.pointerType !== "mouse") { if (current === m && pinned) hide(); else { show(m, false); pinned = true; } return; }
    if (!m && pinned) hide();
  });
  document.addEventListener("keydown", e => {
    const host = document.activeElement && document.activeElement.matches && document.activeElement.matches(`${HOSTS}[tabindex]`) ? document.activeElement : null;
    if (!host) return;
    if (e.key === "Escape") { hide(); return; }
    const list = marks(host); if (!list.length) return;
    const i = current && list.includes(current) ? list.indexOf(current) : -1;
    let j = null;
    if (e.key === "ArrowRight" || e.key === "ArrowDown") j = Math.min(list.length - 1, i + 1);
    if (e.key === "ArrowLeft" || e.key === "ArrowUp") j = Math.max(0, i - 1);
    if (e.key === "Home") j = 0;
    if (e.key === "End") j = list.length - 1;
    if (j == null) return;
    e.preventDefault(); pinned = true; show(list[j], true);
  });
  document.addEventListener("focusout", e => { if (e.target.matches && e.target.matches(`${HOSTS}[tabindex]`)) hide(); });
  addEventListener("scroll", () => { if (current && !pinned) hide(); }, { passive: true });

  /* One draw-in per chart, by chart form
     Each form moves the way its data reads; all within 400 ms, once, none under reduced motion.
     squares: rows fill top to bottom · dots: rows appear top to bottom, each dot settling 4 px ·
     slopes and dumbbells: each line draws from its first value to its second, then its dots appear ·
     bars and strips: grow from zero along the value axis · tables, heatmaps, schematics: rows fade in
     top to bottom · anything else: a plain top-to-bottom reveal. (Was one left-to-right reveal for all.) */
  const FORM = {
    squares: ["pageGroups", "groupCheck", "orientationFlow", "batesUnits", "routingUnits", "testUnits", "units", "coverage"],
    dots: ["kCheck", "funsdScatter", "readabilityStrips", "stamp", "modelLadder", "textOptions", "ragLadder", "finalIntervals", "uncertaintyDots", "engines", "certainty"],
    slopes: ["roundsSlope", "ocrSlope", "deptGain", "certaintyGap"],
    bars: ["cascade", "outcomesStrip", "deptBars", "pairs", "scaleStrip", "scaleTime", "batches"],   // settling (small multiples) has no bars: plain reveal
    rows: ["reviewHeatmap", "confusion", "scheme"],
  };
  const formOf = name => Object.keys(FORM).find(k => FORM[k].includes(name)) || "plain";
  const EASE = "cubic-bezier(.23, 1, .32, 1)";
  const byRow = els => {                                   // group marks by their top edge, top to bottom
    const rows = new Map();
    els.forEach(e => { const r = e.getBoundingClientRect(); if (!r.width && !r.height) return;
      const k = Math.round(r.top / 4); (rows.get(k) || rows.set(k, []).get(k)).push(e); });
    return [...rows.keys()].sort((a, b) => a - b).map(k => rows.get(k));
  };
  const stagger = (rows, frames, each = 200, extra = {}) => {   // whole chart done by 400 ms
    const step = rows.length > 1 ? Math.min(40, (400 - each) / (rows.length - 1)) : 0;
    rows.forEach((row, i) => row.forEach(e => e.animate(frames, { duration: each, delay: i * step, easing: EASE, fill: "backwards", ...extra })));
  };
  const fill = e => { const f = getComputedStyle(e).fill; return f && f !== "none" && !/rgba\(0, 0, 0, 0\)/.test(f); };
  const shapes = (h, sel) => [...h.querySelectorAll(sel)].filter(e => {
    const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0 && r.width < h.clientWidth * .98;   // skip backgrounds
  });
  const ANIM = {
    squares: h => stagger(byRow(shapes(h, "svg rect, .units > *, [data-tip]:not(svg *)").filter(e => e.tagName !== "rect" || fill(e) || e.getAttribute("stroke"))),
                          [{ opacity: 0 }, { opacity: 1 }]),
    dots: h => stagger(byRow(shapes(h, "svg circle")), [{ opacity: 0, transform: "translateY(-4px)" }, { opacity: 1, transform: "none" }]),
    slopes: h => {
      const lines = shapes(h, "svg line, svg path, svg polyline").filter(e => parseFloat(getComputedStyle(e).strokeWidth) >= 1.5);
      lines.forEach(e => { const len = e.getTotalLength ? e.getTotalLength() : 0; if (!len) return;
        e.animate([{ strokeDasharray: `${len} ${len}`, strokeDashoffset: len }, { strokeDasharray: `${len} ${len}`, strokeDashoffset: 0 }],
                  { duration: 300, easing: EASE }); });
      shapes(h, "svg circle").forEach(e => e.animate([{ opacity: 0 }, { opacity: 1 }], { duration: 160, delay: 240, easing: EASE, fill: "backwards" }));
    },
    bars: h => {
      const bars = shapes(h, "svg rect").filter(fill);
      bars.forEach(e => { e.style.transformBox = "fill-box"; e.style.transformOrigin = "left center"; });
      stagger(byRow(bars), [{ transform: "scaleX(0)" }, { transform: "scaleX(1)" }], 320);
      setTimeout(() => bars.forEach(e => { e.style.transformBox = ""; e.style.transformOrigin = ""; }), 450);
    },
    rows: h => {
      const tr = shapes(h, "tr");
      stagger(tr.length ? tr.map(r => [r]) : byRow(shapes(h, "svg rect, .scheme > *, svg text")), [{ opacity: 0 }, { opacity: 1 }]);
    },
    plain: h => { h.classList.add("draw-in"); setTimeout(() => h.classList.remove("draw-in"), 450); },
  };
  const seen = new WeakSet(), visible = new WeakSet();
  const play = h => {
    if (seen.has(h) || !visible.has(h) || !h.firstElementChild) return;   // wait until it is in view and drawn
    seen.add(h); if (io) io.unobserve(h);
    if (reduce.matches || !Element.prototype.animate) return;
    try { ANIM[formOf(h.getAttribute("data-chart"))](h); } catch (e) { /* motion is never worth breaking a chart */ }
  };
  const io = "IntersectionObserver" in window ? new IntersectionObserver(entries => entries.forEach(en => {
    if (en.isIntersecting) visible.add(en.target); else visible.delete(en.target);
    play(en.target);
  }), { threshold: 0.2 }) : null;
  const watch = () => document.querySelectorAll("[data-chart]").forEach(h => { if (io && !seen.has(h)) { io.observe(h); play(h); } });

  let queued = false;
  const refresh = () => { if (queued) return; queued = true; requestAnimationFrame(() => { queued = false; adopt(); watch(); }); };
  new MutationObserver(refresh).observe(document.body, { childList: true, subtree: true });
  refresh();
  /* A decision reason of one or two lines uses the width it has; longer ones keep 40 rem */
  const fitShort = () => document.querySelectorAll(".decisions details[open] p").forEach(p => {
    p.classList.remove("fit");
    const lh = parseFloat(getComputedStyle(p).lineHeight) || 22;
    if (p.getBoundingClientRect().height > 0 && p.getBoundingClientRect().height <= lh * 2.2) p.classList.add("fit");
  });
  let fitT = 0; const fitSoon = () => { clearTimeout(fitT); fitT = setTimeout(fitShort, 120); };
  document.addEventListener("toggle", fitSoon, true); addEventListener("resize", fitSoon); addEventListener("load", fitSoon);
  new MutationObserver(fitSoon).observe(document.body, { childList: true, subtree: true });
  /* The masthead's Reference menu closes on Escape or a click elsewhere */
  document.addEventListener("click", e => document.querySelectorAll(".nav-ref[open]").forEach(d => { if (!d.contains(e.target)) d.open = false; }));
  document.addEventListener("keydown", e => { if (e.key !== "Escape") return; document.querySelectorAll(".nav-ref[open]").forEach(d => { d.open = false; d.querySelector("summary").focus(); }); });
})();

/* Text fitting.
A lead-in with nothing beside it that fits in three lines at the full width uses the full width (.fw).
Decision IDs, figure numbers, numbers with separators or units and file IDs are wrapped in .nb so they never
   break or hyphenate. Both run once the text is bound and again when content is added (decision lists, charts). */
(() => {
  const RX = /(\(?(?:D-\d+|Figures? \d+(?:\.\d+)?(?:[–-]\d+(?:\.\d+)?)?|\d{1,3}(?:,\d{3})+(?:\s(?:pages|records|documents|words))?|\d+(?:\.\d+)?\s?(?:%|s|ms|px)(?![A-Za-z])|\d+\.\d+(?:[–-]\d+\.\d+)?|\b[A-Z]{2}\d{6,}\b|\b[a-z]{4}\d{4}\b)\)?[.,;:]?)/g;   // brackets and the punctuation after stay with the token
  const SKIP = "svg, script, style, code, .nb, [data-field], [data-sum], [data-share], [data-calc], [data-derived], .rim-mono";
  function noBreak(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, { acceptNode: n =>
      !n.nodeValue.trim() || !n.parentElement || n.parentElement.closest(SKIP) ? NodeFilter.FILTER_REJECT : NodeFilter.FILTER_ACCEPT });
    const nodes = []; while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(n => {
      // a text run directly inside a flex or grid box is its own item: a wrapper there would become a separate item with the box's gap
      if (/flex|grid/.test(getComputedStyle(n.parentElement).display)) return;
      const t = n.nodeValue; RX.lastIndex = 0; if (!RX.test(t)) return; RX.lastIndex = 0;
      const frag = document.createDocumentFragment(); let last = 0, m;
      while ((m = RX.exec(t))) {
        if (m.index > last) frag.append(t.slice(last, m.index));
        const s = document.createElement("span"); s.className = "nb"; s.textContent = m[0]; frag.append(s); last = m.index + m[0].length;
      }
      if (last < t.length) frag.append(t.slice(last));
      n.replaceWith(frag);
    });
  }
  function fullWidthLeads() {
    if (!document.body.classList.contains("grid")) return;
    document.querySelectorAll("main section > p.muted, main section > p:not([class]), main section > p.fw").forEach(p => {
      p.classList.add("fw");
      const lh = parseFloat(getComputedStyle(p).lineHeight) || 28;
      if (p.getBoundingClientRect().height / lh > 3.4) p.classList.remove("fw");
    });
  }
  // A paragraph under three lines is short text: left-aligned, no hyphenation (measured at its justified setting)
  function shortParas() {
    document.querySelectorAll("main p").forEach(p => {
      p.classList.remove("short");
      const cs = getComputedStyle(p);
      if (cs.textAlign !== "justify" && cs.hyphens !== "auto") return;
      const lh = parseFloat(cs.lineHeight) || parseFloat(cs.fontSize) * 1.5;
      if (p.getBoundingClientRect().height / lh < 2.6) p.classList.add("short");
    });
  }
  let t = null;
  const run = () => { const m = document.querySelector("main"); if (!m) return; noBreak(m); fullWidthLeads(); shortParas(); };
  const later = () => { clearTimeout(t); t = setTimeout(run, 150); };
  addEventListener("load", () => { run(); setTimeout(run, 1200); setTimeout(run, 3000);
    const m = document.querySelector("main"); if (m) new MutationObserver(later).observe(m, { childList: true, subtree: true }); });
  let rw = null; addEventListener("resize", () => { clearTimeout(rw); rw = setTimeout(() => { fullWidthLeads(); shortParas(); }, 200); });
})();
