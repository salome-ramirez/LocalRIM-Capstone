/* Filing and review chapter (03) charts.
   Adds the classifier figures to the shared CHARTS table in site.js and reuses its
   helpers (svg, el, widthOf, fmt, emptyNote). Every number comes from
   site_data.json (classifier block, written by build_site_data.py from the
   cross-validation outputs in results/classifier). Counts and scores only. */

(() => {
  const NS = "http://www.w3.org/2000/svg";
  const cls = d => d.classifier;
  const DEPT = { AFF: "Corporate affairs", FIN: "Finance", LEG: "Legal and admin.", MKT: "Marketing", QSH: "Quality", RND: "Research" };
  const f2 = v => v == null ? "—" : Number(v).toFixed(2);
  const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;

  // Like el() in site.js, but CSS variables also work for stroke (attributes can't hold var()).
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
  function need(host, d) {
    if (!cls(d)) { emptyNote(host, "Not run yet. Run the classifier scripts, then build_site_data.py."); return false; }
    return true;
  }
  const bold = (s, x, y, text, extra = {}) => {
    const t = E(s, "text", { x, y, class: "svg-label", ...extra }, text);
    t.style.fontWeight = 600;
    return t;
  };
  function axisX(s, x0, x1, y, min, max, ticks, fmtTick) {
    const sx = v => x0 + (x1 - x0) * (v - min) / (max - min);
    ticks.forEach(v => {
      E(s, "line", { x1: sx(v), x2: sx(v), y1: 0, y2: y, class: "svg-grid" });
      E(s, "text", { x: sx(v), y: y + 15, "text-anchor": "middle", class: "svg-text" }, fmtTick(v));
    });
    return sx;
  }

  Object.assign(CHARTS, {

    /* Opening tiles */
    reviewTiles(host, d) {
      if (!need(host, d)) return;
      const c = cls(d), e = c.embeddings, r = c.routing;
      const gain = e.A ? e.hybrid.macro_f1 - e.A.macro_f1 : null;
      const tiles = [
        ["Filing score", f2(e.hybrid.macro_f1), `Macro F1 across six departments, ${c.folds}-fold cross-validation on ${c.pages} labeled pages`],
        ["Filed automatically", pct(r.auto_filed / r.pages), `${r.auto_filed} of ${r.pages} pages, ${pct(r.accuracy_auto_filed)} of them correct, at a ${pct(r.target)} target set in advance`],
        ["Gained from better OCR", gain != null ? `+${Math.round(gain * 100)} pts` : "—", "Macro F1, chosen text vs the baseline Tesseract reading"],
        ["Held back", String(c.final_test_pages), "Pages labeled blind and kept aside for one final test"],
      ];
      const wrap = document.createElement("div");
      wrap.className = "tiles";
      wrap.innerHTML = tiles.map(([k, v, s]) => `<div><small>${k}</small><strong>${v}</strong><span>${s}</span></div>`).join("");
      host.append(wrap);
    },

    /* Figure 3.1. Ladder: what each step adds */
    modelLadder(host, d) {
      if (!need(host, d)) return;
      const c = cls(d);
      const rows = [
        { label: "Always guess Marketing", sub: "the largest department", v: c.tfidf.majority, kind: "floor" },
        { label: "Bates stamp alone", sub: "a shortcut, removed from the text", v: c.bates_probe, kind: "warn" },
        { label: "TF-IDF", sub: "weighted word counts + logistic regression", v: c.tfidf.hybrid, kind: "mid" },
        { label: "Sentence embeddings", sub: "Qwen3-Embedding-0.6B + logistic regression", v: c.embeddings.hybrid, kind: "lead" },
      ].filter(r => r.v);
      const W = widthOf(host), labelW = Math.min(230, Math.round(W * 0.36)), valueW = 70, rowH = 52, top = 8;
      const H = top + rows.length * rowH + 26;
      const s = svg(W, H, "Macro F1 for each step, from a floor to the sentence-embedding model");
      const sx = axisX(s, labelW, W - valueW, H - 22, 0, 1, [0, .25, .5, .75, 1], v => v.toFixed(2));
      let prev = null;
      rows.forEach((r, i) => {
        const y = top + i * rowH + 22, cx = sx(r.v.macro_f1);
        bold(s, 0, y - 4, r.label);
        E(s, "text", { x: 0, y: y + 13, class: "svg-text" }, r.sub);
        if (prev) E(s, "path", { d: `M${prev.x} ${prev.y + 8} V${y} H${cx - 9}`, fill: "none", class: "svg-grid", "stroke-dasharray": "3 3" });
        E(s, "line", { x1: sx(r.v.macro_f1 - r.v.macro_f1_sd), x2: sx(Math.min(1, r.v.macro_f1 + r.v.macro_f1_sd)), y1: y, y2: y,
                        stroke: r.kind === "warn" ? "var(--accent)" : "var(--deep)", "stroke-width": 2, opacity: .45 });
        const dot = E(s, "circle", { cx, cy: y, r: r.kind === "lead" ? 9 : 7 });
        if (r.kind === "warn") { dot.style.fill = "var(--panel)"; dot.style.stroke = "var(--accent)"; dot.style.strokeWidth = 2; dot.style.strokeDasharray = "3 2"; }
        else dot.setAttribute("class", r.kind === "lead" ? "svg-deep" : r.kind === "mid" ? "svg-dusty" : "svg-soft");
        tip(dot, `${r.label}: macro F1 ${f2(r.v.macro_f1)} ± ${f2(r.v.macro_f1_sd)} across folds`);
        E(s, "text", { x: W, y: y + 5, "text-anchor": "end", class: "svg-value" }, f2(r.v.macro_f1));
        if (r.kind !== "warn") prev = { x: cx, y };
      });
      host.append(s);
    },

    /* Figure 3.2. Strip: which text to read */
    textOptions(host, d) {
      if (!need(host, d)) return;
      const t = cls(d).tfidf;
      const opt = d.ocr && d.ocr.engines && d.ocr.engines.text_options;
      const fromD = opt && opt.hybrid_from_D != null ? `GLM-OCR on ${opt.hybrid_from_D} hard pages` : "GLM-OCR on the hard pages";
      const rows = [["hybrid", "Hybrid", fromD], ["C", "PaddleOCR only", "the reading engine"],
                    ["D", "GLM-OCR only", "the vision model"], ["both", "Both joined", "two readings side by side"]].filter(r => t[r[0]]);
      // Figure 3.2 is drawn as tall as Figure 3.3 beside it (250 px); the rows spread to fill it
      const W = widthOf(host), labelW = Math.min(230, Math.round(W * 0.38)), valueW = 60, top = 6, H = Math.max(250, top + rows.length * 44 + 26), rowH = (H - top - 26) / rows.length;
      const s = svg(W, H, "Macro F1 per fold for each text option");
      const sx = axisX(s, labelW, W - valueW, H - 22, .4, .7, [.4, .5, .6, .7], v => v.toFixed(1));
      rows.forEach(([k, name, sub], i) => {
        const y = top + i * rowH + 18, v = t[k], lead = k === "hybrid";
        bold(s, 0, y, name);
        E(s, "text", { x: 0, y: y + 15, class: "svg-text" }, sub);
        v.folds.forEach((f, j) => tip(E(s, "circle", { cx: sx(Math.max(.4, Math.min(.7, f))), cy: y + 4, r: 4, class: lead ? "svg-dusty" : "svg-soft", opacity: .9 }), `Fold ${j + 1}: ${f2(f)}`));
        E(s, "line", { x1: sx(v.macro_f1), x2: sx(v.macro_f1), y1: y - 8, y2: y + 16, stroke: lead ? "var(--deep)" : "var(--muted)", "stroke-width": lead ? 3 : 2 });
        E(s, "text", { x: W, y: y + 8, "text-anchor": "end", class: "svg-value" }, f2(v.macro_f1));
      });
      host.append(s);
    },

    /* Figure 3.3. Slope: did better OCR file better? */
    ocrSlope(host, d) {
      if (!need(host, d)) return;
      const c = cls(d);
      const lines = [["Embeddings", c.embeddings.A, c.embeddings.hybrid, "var(--deep)", null], ["TF-IDF", c.tfidf.A, c.tfidf.hybrid, "var(--dusty)", "6 4"]].filter(l => l[1] && l[2]);
      if (!lines.length) { emptyNote(host, "Run the baseline-OCR comparison (RQ3)."); return; }
      const W = widthOf(host), H = 250, padT = 30, padB = 34, x0 = Math.min(110, W * .22), x1 = W - Math.min(190, W * .36);
      const min = .4, max = .75, sy = v => padT + (H - padT - padB) * (1 - (v - min) / (max - min));
      const s = svg(W, H, "Macro F1 with the baseline OCR and with the chosen text, for each model");
      [x0, x1].forEach(x => E(s, "line", { x1: x, x2: x, y1: padT - 10, y2: H - padB + 6, class: "svg-axis" }));
      bold(s, x0, 14, "Baseline OCR", { "text-anchor": "middle" });
      E(s, "text", { x: x0, y: H - 12, "text-anchor": "middle", class: "svg-text" }, "Tesseract, 150 dpi");
      bold(s, x1, 14, "Chosen text", { "text-anchor": "middle" });
      E(s, "text", { x: x1, y: H - 12, "text-anchor": "middle", class: "svg-text" }, "PaddleOCR + GLM-OCR hybrid");
      lines.forEach(([name, a, b, color, dash]) => {
        E(s, "line", { x1: x0, y1: sy(a.macro_f1), x2: x1, y2: sy(b.macro_f1), stroke: color, "stroke-width": 3, "stroke-dasharray": dash });
        [[x0, a, "end", -12], [x1, b, "start", 12]].forEach(([x, v, anchor, dx]) => {
          E(s, "circle", { cx: x, cy: sy(v.macro_f1), r: 6, fill: color });
          E(s, "text", { x: x + dx, y: sy(v.macro_f1) + 4, "text-anchor": anchor, class: "svg-value" }, f2(v.macro_f1));
        });
        // On narrow charts the label would run past the right edge, so it ends at the edge, below the value
        const lbl = `${name}, +${Math.round((b.macro_f1 - a.macro_f1) * 100)} pts`, room = W - (x1 + 52) >= lbl.length * 7;
        E(s, "text", room ? { x: x1 + 52, y: sy(b.macro_f1) + 4, class: "svg-label" } : { x: W, y: sy(b.macro_f1) + (name === "Embeddings" ? -12 : 20), "text-anchor": "end", class: "svg-label" }, lbl);
      });
      host.append(s);
    },

    /* Figure 3.4. Dumbbell: department gains */
    deptGain(host, d) {
      if (!need(host, d)) return;
      const p = cls(d).per_department;
      if (!p.tfidf || !p.embeddings) { emptyNote(host, "Run tfidf_errors.py for both models."); return; }
      const rows = Object.keys(DEPT).filter(k => p.tfidf[k] != null && p.embeddings[k] != null)
        .map(k => ({ k, a: p.tfidf[k], b: p.embeddings[k] })).sort((x, y) => (y.b - y.a) - (x.b - x.a));
      const W = widthOf(host), labelW = Math.min(170, Math.round(W * 0.3)), valueW = 64, rowH = 34, top = 6;
      const H = top + rows.length * rowH + 26;
      const s = svg(W, H, "F1 per department with TF-IDF and with sentence embeddings");
      const sx = axisX(s, labelW, W - valueW, H - 22, .2, .9, [.2, .4, .6, .8], v => v.toFixed(1));
      rows.forEach((r, i) => {
        const y = top + i * rowH + 16;
        E(s, "text", { x: 0, y: y + 4, class: "svg-label" }, DEPT[r.k]);
        E(s, "line", { x1: sx(r.a), x2: sx(r.b), y1: y, y2: y, stroke: "var(--dusty)", "stroke-width": 3 });
        tip(E(s, "circle", { cx: sx(r.a), cy: y, r: 6, class: "svg-soft", stroke: "var(--dusty-edge)" }), `${DEPT[r.k]}, TF-IDF: ${f2(r.a)}`);
        tip(E(s, "circle", { cx: sx(r.b), cy: y, r: 7, class: "svg-deep" }), `${DEPT[r.k]}, sentence embeddings: ${f2(r.b)}`);
        const g = r.b - r.a;
        E(s, "text", { x: W, y: y + 4, "text-anchor": "end", class: "svg-value" }, `${g >= 0 ? "+" : "−"}${Math.abs(g).toFixed(2)}`);
      });
      host.append(s);
    },

    /* Figure 3.6. Confusion heatmap, with the largest confusions ranked beside it */
    confusion(host, d) {
      if (!need(host, d) || !cls(d).confusion) return;
      const c = cls(d).confusion, n = c.labels.length;
      const SHORT = { AFF: "Corp. aff.", FIN: "Finance", LEG: "Legal", MKT: "Mktg.", QSH: "Quality", RND: "Research" };
      const W = widthOf(host), wide = W >= 820, narrow = W < 560, labelW = 130, top = narrow ? 70 : 36;   // Angled headers on narrow charts
      const gridW = wide ? Math.round(W * 0.58) - labelW : W - labelW - 4;
      const cw = Math.floor(gridW / n), ch = 46, H0 = top + n * ch + 8;
      const pairs = [];
      c.labels.forEach((lab, i) => c.counts[i].forEach((v, j) => { if (c.guesses[j] !== lab && v) pairs.push({ lab, g: c.guesses[j], v }); }));
      pairs.sort((x, y) => y.v - x.v);
      const top5 = pairs.slice(0, 6);
      const listH = wide ? 0 : 30 + top5.length * 34;
      const H = H0 + listH;
      const s = svg(W, H, "Department labeled by hand (rows) against the embedding model's guess (columns), with the largest confusions ranked");
      E(s, "text", { x: labelW, y: 8, class: "svg-text" }, "Model's guess →");
      c.guesses.forEach((g, j) => narrow
        ? E(s, "text", { x: labelW + j * cw + cw / 2, y: top - 6, "text-anchor": "start", class: "svg-text", transform: `rotate(-45 ${labelW + j * cw + cw / 2} ${top - 6})` }, SHORT[g] || g)
        : E(s, "text", { x: labelW + j * cw + cw / 2, y: 26, "text-anchor": "middle", class: "svg-text" }, SHORT[g] || g));
      c.labels.forEach((lab, i) => {
        const total = c.counts[i].reduce((a, b) => a + b, 0) || 1;
        E(s, "text", { x: labelW - 10, y: top + i * ch + ch / 2 + 4, "text-anchor": "end", class: "svg-label" }, DEPT[lab] || lab);
        c.counts[i].forEach((v, j) => {
          const share = v / total, diag = c.guesses[j] === lab;
          const r = E(s, "rect", { x: labelW + j * cw + 1, y: top + i * ch + 1, width: cw - 2, height: ch - 2, class: "svg-deep", opacity: v ? .12 + .88 * share : .04 });
          if (diag) { r.style.stroke = "var(--ink)"; r.style.strokeWidth = 1.5; }
          tip(r, `${DEPT[lab] || lab} filed as ${DEPT[c.guesses[j]] || c.guesses[j]}: ${v} of ${total} pages`);
          const t = E(s, "text", { x: labelW + j * cw + cw / 2, y: top + i * ch + ch / 2 + 4, "text-anchor": "middle", class: share > .45 ? "svg-on" : "svg-value" }, v || "");
          if (share <= .45) t.style.fontWeight = 400;
        });
      });
      // Ranked list of the largest off-diagonal cells
      const lx = wide ? labelW + n * cw + 48 : 0, ly = wide ? top : H0 + 20, lw = wide ? W - lx : W;
      bold(s, lx, ly - 14, "Largest confusions");
      const maxV = top5.length ? top5[0].v : 1, barX = lx, barW = Math.max(60, lw - 60);
      top5.forEach((p, k) => {
        const y = ly + 8 + k * (wide ? (n * ch - 8) / top5.length : 34);
        E(s, "text", { x: lx, y: y + 10, class: "svg-label" }, `${DEPT[p.lab]} → filed as ${DEPT[p.g]}`);
        E(s, "rect", { x: barX, y: y + 17, width: barW, height: 8, class: "svg-soft", opacity: .5 });
        E(s, "rect", { x: barX, y: y + 17, width: Math.max(2, barW * p.v / maxV), height: 8, class: "svg-deep" });
        E(s, "text", { x: lx + lw, y: y + 10, "text-anchor": "end", class: "svg-value" }, `${p.v} pages`);
      });
      host.append(s);
    },

    /* Figure 3.6. Hard for a person, hard for the model */
    certaintyGap(host, d) {
      if (!need(host, d)) return;
      const c = cls(d).certainty;
      if (!c.tfidf || !c.embeddings) return;
      const rows = [["sure", "Pages I marked sure"], ["unsure", "Pages I marked unsure"]];
      // Figure 3.5 is drawn as tall as Figure 3.4 beside it (6 departments x 34 px + 32); its two rows spread to fill it
      const W = widthOf(host), labelW = Math.min(190, Math.round(W * 0.42)), valueW = 52, top = 6, H = Math.max(top + 6 * 34 + 26, top + rows.length * 46 + 26), rowH = (H - top - 26) / rows.length;
      const s = svg(W, H, "Accuracy on pages the annotator marked sure or unsure, for each model");
      const sx = axisX(s, labelW, W - valueW, H - 22, 0, 1, [0, .5, 1], v => `${v * 100}%`);
      rows.forEach(([k, name], i) => {
        const y = top + i * rowH + 18, a = c.tfidf[k].accuracy, b = c.embeddings[k].accuracy;
        E(s, "text", { x: 0, y: y + 4, class: "svg-label" }, name);
        E(s, "text", { x: 0, y: y + 19, class: "svg-text" }, `${c.embeddings[k].pages} pages`);
        E(s, "line", { x1: sx(a), x2: sx(b), y1: y, y2: y, stroke: "var(--dusty)", "stroke-width": 3 });
        tip(E(s, "circle", { cx: sx(a), cy: y, r: 6, class: "svg-soft", stroke: "var(--dusty-edge)" }), `TF-IDF: ${pct(a)}`);
        tip(E(s, "circle", { cx: sx(b), cy: y, r: 7, class: "svg-deep" }), `Sentence embeddings: ${pct(b)}`);
        E(s, "text", { x: W, y: y + 4, "text-anchor": "end", class: "svg-value" }, pct(b));
      });
      host.append(s);
    },

    /* Trade-off curve. No page draws it; the chapter's cutoff figure (Figure 3.7) is drawn by casefile.js */
    routingCurve(host, d) {
      if (!need(host, d)) return;
      const r = cls(d).routing;
      const W = widthOf(host), H = 300, padL = 52, padR = 24, padT = 16, padB = 44;
      const min = .5, sx = v => padL + (W - padL - padR) * v, sy = v => padT + (H - padT - padB) * (1 - (v - min) / (1 - min));
      const s = svg(W, H, "Share of pages filed automatically against the accuracy of those pages, for each confidence cutoff");
      [0, .25, .5, .75, 1].forEach(v => { E(s, "line", { x1: sx(v), x2: sx(v), y1: padT, y2: H - padB, class: "svg-grid" });
        E(s, "text", { x: sx(v), y: H - padB + 16, "text-anchor": "middle", class: "svg-text" }, `${v * 100}%`); });
      [.5, .6, .7, .8, .9, 1].forEach(v => { E(s, "line", { x1: padL, x2: W - padR, y1: sy(v), y2: sy(v), class: "svg-grid" });
        E(s, "text", { x: padL - 8, y: sy(v) + 4, "text-anchor": "end", class: "svg-text" }, `${Math.round(v * 100)}%`); });
      E(s, "text", { x: (padL + W - padR) / 2, y: H - 6, "text-anchor": "middle", class: "svg-text" }, "Share of pages filed automatically →");
      E(s, "line", { x1: padL, x2: W - padR, y1: sy(r.target), y2: sy(r.target), class: "svg-threshold" });
      E(s, "text", { x: W - padR, y: sy(r.target) - 6, "text-anchor": "end", class: "svg-text" }, `${pct(r.target)} target, set in advance`);
      [[r.curve_tfidf, "var(--dusty)", "TF-IDF", "6 4"], [r.curve_embeddings, "var(--deep)", "Embeddings", null]].forEach(([pts, color, name, dash]) => {
        const p = (pts || []).filter(q => q.auto_filed > 0 && q.accuracy_auto_filed != null);
        if (!p.length) return;
        E(s, "path", { d: p.map((q, i) => `${i ? "L" : "M"}${sx(q.coverage)} ${sy(Math.max(min, q.accuracy_auto_filed))}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.5, "stroke-dasharray": dash });
        p.forEach(q => tip(E(s, "circle", { cx: sx(q.coverage), cy: sy(Math.max(min, q.accuracy_auto_filed)), r: 3.5, fill: color }),
          `${name}, cutoff ${q.threshold}: ${pct(q.coverage)} filed automatically, ${pct(q.accuracy_auto_filed)} correct`));
        const last = p[0];
        E(s, "text", { x: sx(last.coverage) - 6, y: sy(Math.max(min, last.accuracy_auto_filed)) + 16, "text-anchor": "end", class: "svg-label" }, name);
      });
      const fx = sx(r.auto_filed / r.pages), fy = sy(r.accuracy_auto_filed);
      const ring = E(s, "circle", { cx: fx, cy: fy, r: 9, fill: "none" });
      ring.style.stroke = "var(--ink)"; ring.style.strokeWidth = 2;
      tip(ring, `Cutoff chosen on training folds only: ${r.auto_filed} of ${r.pages} filed automatically, ${pct(r.accuracy_auto_filed)} correct`);
      E(s, "text", { x: fx + 14, y: fy + 26, class: "svg-label" }, `Cross-validation: ${pct(r.auto_filed / r.pages)} filed, ${pct(r.accuracy_auto_filed)} correct`);
      host.append(s);
    },

    /* Figure 3.8. 250 pages, one square each */
    routingUnits(host, d) {
      if (!need(host, d)) return;
      const r = cls(d).routing, u = r.units || [];
      const W = widthOf(host), sq = 14, gap = 3, step = sq + gap, cols = Math.max(10, Math.floor(W / step));
      const H = Math.ceil(u.length / cols) * step + 4;
      const s = svg(W, H, `${r.pages} labeled pages: filed automatically or sent to a person`);
      u.forEach(([a, c], i) => {
        const x = (i % cols) * step, y = Math.floor(i / cols) * step;
        const q = E(s, "rect", { x, y, width: sq, height: sq, rx: 2, class: a ? "svg-deep" : "svg-soft" });
        if (a && !c) { q.setAttribute("class", "svg-wrong"); q.style.fill = "var(--panel)"; }
        tip(q, a ? (c ? "Filed automatically, correct" : "Filed automatically, wrong department") : "Sent to a person");
      });
      host.append(s);
    },
  });
})();
