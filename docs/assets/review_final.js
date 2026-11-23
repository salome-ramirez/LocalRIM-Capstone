/* 03 Filing and review, second half: review suggestions (RAG),
   the one-time final test and the scale run. Figures 3.9–3.14.
   Adds to the shared CHARTS table in site.js and reuses its helpers (svg, el,
   widthOf, emptyNote). Every number comes from site_data.json (blocks rag,
   final_test, scale_run, written by build_site_data.py). Counts and rates only:
   no page IDs, page text or model reasons. */

(() => {
  const NS = "http://www.w3.org/2000/svg";
  const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;
  const f2 = v => v == null ? "—" : Number(v).toFixed(2);

  function E(parent, tag, attrs = {}, text) {
    const { stroke, ...rest } = attrs;
    const n = el(parent, tag, rest, text);
    if (stroke != null) n.style.stroke = stroke;
    return n;
  }
  function tip(node, text) {
    const t = document.createElementNS(NS, "title");
    t.textContent = text;
    node.append(t);
    return node;
  }
  const bold = (s, x, y, text, extra = {}) => {
    const t = E(s, "text", { x, y, class: "svg-label", ...extra }, text);
    t.style.fontWeight = 600;
    return t;
  };
  function need(host, block, msg) {
    if (!block) { emptyNote(host, msg || "Not run yet."); return false; }
    return true;
  }
  function axisX(s, x0, x1, y, min, max, ticks, fmtTick) {
    const sx = v => x0 + (x1 - x0) * (Math.max(min, Math.min(max, v)) - min) / (max - min);
    ticks.forEach(v => {
      E(s, "line", { x1: sx(v), x2: sx(v), y1: 0, y2: y, class: "svg-grid" });
      E(s, "text", { x: sx(v), y: y + 15, "text-anchor": "middle", class: "svg-text" }, fmtTick(v));
    });
    return sx;
  }
  // Wilson 95% interval for a share k of n (used only where no bootstrap interval exists)
  function wilson(k, n) {
    const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
    const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [c - h, c + h];
  }
  // Same encoding as a wrongly filed square in Figures 3.8 and 3.12
  const wrongStyle = n => { n.style.fill = "var(--panel)"; n.style.stroke = "var(--accent)"; n.style.strokeWidth = 1.6; };
  // Projected range (Figure 3.13): light fill with a dashed "wrong department" edge
  const bandStyle = n => { n.style.fill = "color-mix(in srgb, var(--deep) 28%, var(--panel))"; n.style.stroke = "var(--accent)"; n.style.strokeWidth = 1.5; n.style.strokeDasharray = "4 3"; };
  const hoursPer1000 = t => t.reduce((a, r) => a + r.seconds, 0) * 1000 / 3600;

  Object.assign(CHARTS, {

    /* Opening tiles with the final-test numbers (shown in place of the cross-validated tiles) */
    finalTiles(host, d) {
      const f = d.final_test, r = d.final_test && d.final_test.rag, sr = d.scale_run;
      if (!need(host, f, "The final test has not been scored yet.")) return;
      const tiles = [
        [`Macro F1 on the ${f.pages}-page final test`, f2(f.scores.macro_f1.value), `Macro F1 on ${f.pages} pages labeled blind and scored once (${f2(f.scores.macro_f1.low)}–${f2(f.scores.macro_f1.high)})`],
        ["Final-test pages qualified for automatic filing", pct(f.auto_filed / f.pages), `${f.auto_filed} of ${f.pages} test pages, ${pct(f.auto_correct / f.auto_filed)} of them correct, against a ${pct(f.target)} target`],
        ["Review-suggestion accuracy on pages routed for review (final test)", pct(r.rag_top1), `On test pages sent to a person, vs ${pct(r.classifier_top1)} for the classifier's first choice`],
        [sr ? `Total processing time for ${sr.pages.toLocaleString("en-US")} new pages` : "Total processing time for the new pages", sr ? `${hoursPer1000(sr.timing).toFixed(1)} h` : "—", sr ? `On one ordinary PC; ${pct(sr.auto_filed / sr.pages)} filed automatically` : "Scale run not run yet"],
      ];
      const wrap = document.createElement("div");
      wrap.className = "tiles";
      wrap.innerHTML = tiles.map(([k, v, s]) => `<div data-tip="${s}"><strong>${v}</strong><small>${k}</small><span class="tile-sub">${s}</span></div>`).join("");   // Number first, short label under it; the description is the tooltip
      host.append(wrap);
    },

    /* Figure 3.9. The case file a reviewer receives (schematic; no real page content) */
    ragCase(host, d) {
      if (!need(host, d.rag, "Run rag_retrieve.py and rag_suggest.py, then build_site_data.py.")) return;
      const k = d.rag.neighbors;
      host.innerHTML = `
        <div class="casefile">
          <div class="cf-in">
            <div class="cf-tab">The case file</div>
            <div class="cf-card"><b>The page</b><span>Its OCR text, own Bates removed</span></div>
            <div class="cf-card"><b>${k} similar filed pages</b><span>Found by embedding similarity among the 250 labeled pages, each with where it was filed</span></div>
            <div class="cf-card"><b>The filing manual</b><span>One card per department: its question, definition, hints and record types; rules R1–R10</span></div>
            <div class="cf-card"><b>The classifier's two best guesses</b><span>With how sure it is of each</span></div>
          </div>
          <div class="cf-mid"><span class="cf-arrow" aria-hidden="true">→</span><span class="cf-chip">${d.rag.model.replace(" (local, Ollama)", "")}<br><small>on the same PC</small></span></div>
          <div class="cf-out">
            <div class="cf-tab">The suggestion</div>
            <div class="cf-card cf-key"><b>Record type</b><span>Department › series › record type</span></div>
            <div class="cf-card"><b>Second choice</b><span>From a different department</span></div>
            <div class="cf-card"><b>One-line reason</b><span>Why the page belongs there</span></div>
            <div class="cf-card"><b>Sources used</b><span>Which similar pages and rules it relied on</span></div>
          </div>
        </div>`;
    },

    /* Figure 3.10. Ladder: what each piece of the case file adds (cross-validated review queue) */
    ragLadder(host, d) {
      if (!need(host, d.rag)) return;
      const rows = d.rag.approaches;
      const W = widthOf(host), labelW = Math.min(250, Math.round(W * 0.38)), valueW = 60, rowH = 46, top = 8;
      const H = top + rows.length * rowH + 26;
      const s = svg(W, H, `Right first choice and right within two choices, ${d.rag.review_pages} review pages, for each approach`);
      const sx = axisX(s, labelW, W - valueW, H - 22, .5, 1, [.5, .6, .7, .8, .9, 1], v => `${Math.round(v * 100)}%`);
      rows.forEach((r, i) => {
        const y = top + i * rowH + 20, lead = i === rows.length - 1;
        const label = bold(s, 0, y + 4, r.name.replace(" (with classifier's top two)", ""));
        if (!lead) label.style.fontWeight = 400;
        if (lead) E(s, "text", { x: 0, y: y + 19, class: "svg-text" }, "with the classifier's two guesses");
        E(s, "line", { x1: sx(r.top1), x2: sx(r.top2), y1: y, y2: y, stroke: "var(--dusty)", "stroke-width": 2 });
        tip(E(s, "circle", { cx: sx(r.top2), cy: y, r: 6, fill: "var(--panel)", stroke: "var(--dusty-edge)", "stroke-width": 1.6 }), `${r.name}: right within two choices ${pct(r.top2)}`);
        tip(E(s, "circle", { cx: sx(r.top1), cy: y, r: lead ? 9 : 7, class: lead ? "svg-deep" : "svg-dusty" }), `${r.name}: right first choice ${pct(r.top1)}`);
        E(s, "text", { x: W, y: y + 5, "text-anchor": "end", class: "svg-value" }, pct(r.top1));
      });
      host.append(s);
    },

    /* Figure 3.11. What held on the blind test: final-test value with its interval vs cross-validation */
    finalIntervals(host, d) {
      const f = d.final_test, c = d.classifier, rag = d.rag;
      if (!need(host, f)) return;
      const qr = f.rag, k = Math.round(qr.rag_top1 * qr.review_pages);
      const ragCI = wilson(k, qr.review_pages);
      const cvAuto = c ? c.routing.accuracy_auto_filed : null;
      const cvRag = rag ? rag.approaches[rag.approaches.length - 1].top1 : null;
      const rows = [
        { name: "Macro F1, sentence embeddings", sub: "macro F1, embeddings + LR", t: f.scores.macro_f1, cv: c && c.embeddings.hybrid.macro_f1 },
        { name: "Macro F1, TF-IDF", sub: "macro F1, TF-IDF + LR", t: f.scores.tfidf_macro_f1, cv: c && c.tfidf.hybrid.macro_f1 },
        { name: "Auto-filed pages correct", sub: `${f.auto_correct} of ${f.auto_filed} at cutoff ${f.cutoff}`, t: f.scores.auto_accuracy, cv: cvAuto, target: f.target },
        { name: "Suggestion right", sub: `${qr.review_pages} pages sent to a person`, t: { value: qr.rag_top1, low: ragCI[0], high: ragCI[1] }, cv: cvRag, base: qr.classifier_top1 },
        { name: "Whole pipeline", sub: "Hypothetical overall accuracy if every review suggestion were accepted without correction", t: f.scores.pipeline_accuracy, cv: null },
      ];
      const W = widthOf(host), labelW = Math.min(240, Math.round(W * 0.37)), valueW = 60, rowH = 50, top = 10;
      // A long subtitle (the hypothetical whole-pipeline row) wraps inside the label column
      // each extra line adds 15 px to its row. About 6.4 px per character at the chart label size.
      // Row names wrap the same way (bold, about 7.4 px per character)
      const wrapTo = (t, cw) => { const fit = Math.max(10, Math.floor((labelW - 12) / cw));
        return t.split(" ").reduce((ls, w) => { const l = ls[ls.length - 1]; if (l && (l + " " + w).length <= fit) ls[ls.length - 1] = l + " " + w; else ls.push(w); return ls; }, []); };
      rows.forEach(r => { r.names = wrapTo(r.name, 7.4); r.lines = wrapTo(r.sub, 6.4); r.extra = 15 * (r.names.length - 1 + r.lines.length - 1); });
      const ys = []; rows.reduce((acc, r, i) => { ys[i] = acc; return acc + rowH + r.extra; }, top + 20);
      const H = ys[rows.length - 1] + rowH - 20 + rows[rows.length - 1].extra + 26, min = .3, max = 1;
      const s = svg(W, H, "Final-test results with 95% intervals, compared with cross-validation");
      const sx = axisX(s, labelW, W - valueW, H - 22, min, max, [.3, .4, .5, .6, .7, .8, .9, 1], v => v.toFixed(1));
      rows.forEach((r, i) => {
        const y = ys[i];
        r.names.forEach((l, k) => bold(s, 0, y + 2 + 15 * k, l));
        const subY = y + 17 + 15 * (r.names.length - 1);
        r.lines.forEach((l, k) => E(s, "text", { x: 0, y: subY + 15 * k, class: "svg-text" }, l));
        if (r.target) {
          E(s, "line", { x1: sx(r.target), x2: sx(r.target), y1: y - 16, y2: y + 16, class: "svg-threshold" });
          E(s, "text", { x: sx(r.target) + 4, y: y - 9, class: "svg-text" }, `${pct(r.target)} target`);
        }
        E(s, "line", { x1: sx(r.t.low), x2: sx(r.t.high), y1: y, y2: y, stroke: "var(--deep)", "stroke-width": 3, opacity: .5 });
        [r.t.low, r.t.high].forEach(v => E(s, "line", { x1: sx(v), x2: sx(v), y1: y - 5, y2: y + 5, stroke: "var(--deep)", "stroke-width": 2, opacity: .5 }));
        if (r.base != null) tip(E(s, "circle", { cx: sx(r.base), cy: y, r: 5, class: "svg-soft", stroke: "var(--dusty-edge)" }), `Classifier's first choice on the same pages: ${pct(r.base)}`);
        if (r.cv != null) tip(E(s, "circle", { cx: sx(r.cv), cy: y, r: 7, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, "stroke-dasharray": "2 2" }), `Practice (cross-validation): ${f2(r.cv)}`);
        tip(E(s, "circle", { cx: sx(r.t.value), cy: y, r: 7, class: "svg-deep" }), `Final test: ${f2(r.t.value)} (95% interval ${f2(r.t.low)}–${f2(r.t.high)})`);
        E(s, "text", { x: W, y: y + 5, "text-anchor": "end", class: "svg-value" }, f2(r.t.value));
      });
      host.append(s);
    },

    /* Figure 3.12. 150 test pages, one square each, plus the excluded pages */
    testUnits(host, d) {
      const f = d.final_test;
      if (!need(host, f)) return;
      const u = f.units.concat(Array(f.excluded.pages).fill([0, 0, 1]));
      const W = widthOf(host), sq = 14, gap = 3, step = sq + gap, cols = Math.max(10, Math.floor(W / step));
      const H = Math.ceil(u.length / cols) * step + 4;
      const s = svg(W, H, `${f.pages} test pages and ${f.excluded.pages} excluded pages: filed automatically or sent to a person`);
      u.forEach(([a, c, x], i) => {
        const px = (i % cols) * step, py = Math.floor(i / cols) * step;
        const q = E(s, "rect", { x: px, y: py, width: sq, height: sq, rx: 2, class: a ? "svg-deep" : "svg-soft" });
        if (a && !c) { q.setAttribute("class", "svg-wrong"); q.style.fill = "var(--panel)"; }
        if (x) { q.setAttribute("class", ""); q.style.fill = "var(--panel)"; q.style.stroke = "var(--muted)"; q.style.strokeDasharray = "2 2"; }
        tip(q, x ? "Excluded page (not usable), sent to a person" : a ? (c ? "Filed automatically, correct" : "Filed automatically, wrong department") : "Sent to a person");
      });
      host.append(s);
    },

    /* Figure 3.13. Where a box of 1,000 new pages goes */
    scaleStrip(host, d) {
      const r = d.scale_run;
      if (!need(host, r, "Run scale_run.py, then build_site_data.py.")) return;
      const W = widthOf(host), barY = 44, barH = 34;
      const sx = v => W * v / r.pages;
      // Misfilings are a projection with a 95% interval, drawn as a range, not one count
      const aw = sx(r.auto_filed), lo = Math.round(r.projected_wrong.low), hi = Math.round(r.projected_wrong.high);
      const items = [
        { name: "Filed automatically, likely right", n: r.auto_filed - hi, cls: "svg-deep", op: 1, w: sx(r.auto_filed - hi), prefix: "at least " },
        { name: "Filed automatically, right or misfiled (projected range)", n: hi - lo, cls: "", op: 1, w: sx(hi - lo), band: true },
        { name: "Filed automatically, likely misfiled (projected)", n: lo, cls: "", op: 1, w: sx(lo), prefix: "at least ", wrong: true },
        { name: "Not confident enough", n: r.reasons["below cutoff"] || 0, cls: "svg-soft", op: 1 },
        { name: "Too little text", n: r.reasons["under 20 tokens"] || 0, cls: "svg-dusty", op: 1 },
        { name: "No readable text", n: r.reasons["no readable text"] || 0, cls: "svg-dusty", op: .5 },
      ];
      items.slice(2).forEach(it => { it.w = sx(it.n); });
      // Legend rows (wrap by an approximate text width)
      const legend = [];
      let lx = 0, ly = 0;
      items.forEach(it => {
        const text = `${it.name} · ${it.prefix || ""}${it.n}`, w = 22 + text.length * 6.6;
        if (w > W) {
          // An item wider than the chart wraps onto its own lines
          if (lx > 0) { lx = 0; ly += 22; }
          const lines = []; let line = "";
          text.split(" ").forEach(word => { const t = line ? `${line} ${word}` : word; if (22 + t.length * 6.6 > W && line) { lines.push(line); line = word; } else line = t; });
          lines.push(line);
          legend.push({ ...it, text, lines, x: 0, y: ly });
          ly += 22 + (lines.length - 1) * 16; lx = 0;
          return;
        }
        if (lx + w > W && lx > 0) { lx = 0; ly += 22; }
        legend.push({ ...it, text, x: lx, y: ly });
        lx += w + 16;
      });
      const legY = barY + barH + 30, H = legY + ly + 14;
      const s = svg(W, H, `${r.pages} new pages: filed automatically or sent to a person, and why`);
      let x = 0;
      items.forEach(it => {
        const seg = tip(E(s, "rect", { x: x + (x ? 1 : 0), y: barY, width: Math.max(1.5, it.w - (x ? 1 : 0)), height: barH, class: it.cls, opacity: it.op }),
          `${it.name}: ${it.prefix || ""}${it.n} pages`);
        if (it.wrong) wrongStyle(seg);
        if (it.band) bandStyle(seg);
        x += it.w;
      });
      E(s, "line", { x1: aw, x2: aw, y1: barY - 8, y2: barY + barH + 8, stroke: "var(--ink)", "stroke-width": 1.5 });
      bold(s, 0, 15, `Filed automatically · ${r.auto_filed}`);
      E(s, "text", { x: 0, y: 32, class: "svg-text" }, `${pct(r.auto_filed / r.pages)} of the pages`);
      bold(s, W, 15, W < 560 ? `Sent to a person · ${r.to_a_person}` : `Sent to a person with a suggestion · ${r.to_a_person}`, { "text-anchor": "end" });   // Shorter on narrow charts
      E(s, "text", { x: W, y: 32, "text-anchor": "end", class: "svg-text" }, `${pct(r.to_a_person / r.pages)} of the pages`);
      legend.forEach(it => {
        const sw = E(s, "rect", { x: it.x, y: legY + it.y - 10, width: 12, height: 12, class: it.cls, opacity: it.op });
        if (it.wrong) wrongStyle(sw);
        if (it.band) bandStyle(sw);
        (it.lines || [it.text]).forEach((l, k) => E(s, "text", { x: it.x + 18, y: legY + it.y + k * 16, class: "svg-text" }, l));
      });
      host.append(s);
    },

    /* Figure 3.14. Time per page by stage */
    scaleTime(host, d) {
      const r = d.scale_run;
      if (!need(host, r)) return;
      const NAMES = {
        "OCR, PaddleOCR (every page)": ["Read the page", "PaddleOCR, every page"],
        "OCR, vision model (hard pages only, spread over all)": ["Re-read hard pages", "vision model, hard pages only"],
        "Embedding": ["Represent its meaning", "Qwen3 embedding"],
        "Classifier and routing": ["Classify and route", "logistic regression"],
        "RAG suggestion (pages to a person only, spread over all)": ["Write the suggestion", "Qwen3 8B, only for pages sent to a person"],
      };
      const rows = r.timing.map(t => ({ ...t, name: (NAMES[t.stage] || [t.stage, ""])[0], sub: (NAMES[t.stage] || ["", ""])[1] }));
      const total = rows.reduce((a, t) => a + t.seconds, 0);
      // On narrow charts the stage name and its engine sit above the bar
      const W = widthOf(host), narrow = W < 560, labelW = narrow ? 0 : Math.min(250, Math.round(W * 0.38)), valueW = 70, rowH = narrow ? 56 : 40, top = 6;
      const H = top + rows.length * rowH + 50;
      const s = svg(W, H, "Seconds per page for each stage, averaged over 1,000 pages");
      const max = Math.ceil(Math.max(...rows.map(t => t.seconds)));
      const sx = axisX(s, labelW, W - valueW, H - 44, 0, max, Array.from({ length: max + 1 }, (_, i) => i), v => `${v} s`);
      rows.forEach((t, i) => {
        const y = top + i * rowH + 16, lead = t.seconds === Math.max(...rows.map(q => q.seconds));
        const lab = bold(s, 0, y, t.name);
        if (!lead) lab.style.fontWeight = 400;
        E(s, "text", { x: 0, y: y + 15, class: "svg-text" }, t.sub);
        const by = narrow ? y + 22 : y - 8;
        tip(E(s, "rect", { x: labelW, y: by, width: Math.max(2, sx(t.seconds) - labelW), height: narrow ? 12 : 16, class: lead ? "svg-deep" : "svg-dusty" }), `${t.name}: ${t.seconds.toFixed(3)} s per page`);
        E(s, "text", { x: W, y: narrow ? y : y + 4, "text-anchor": "end", class: "svg-value" },
          t.seconds < 0.001 ? "< 1 ms" : `${t.seconds < 0.1 ? t.seconds.toFixed(2) : t.seconds.toFixed(1)} s`);
      });
      E(s, "text", { x: W, y: H - 2, "text-anchor": "end", class: "svg-label" }, `Total ${total.toFixed(1)} s per page · about ${(total * 1000 / 3600).toFixed(1)} hours per 1,000 pages`);
      host.append(s);
    },
  });
})();
