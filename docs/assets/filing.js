/* Filing system chapter (01) charts.
   Adds figures to the shared CHARTS table in site.js and reuses its helpers
   (svg, el, widthOf, fmt, emptyNote, viewOf). Batch filters keep working:
   redraw in site.js calls these by name with host.dataset.batch set. */

(() => {
  const NS = "http://www.w3.org/2000/svg";
  const pct = v => `${Math.round(v * 100)}%`;
  const num = v => Number(v).toLocaleString("en-US");
  const PAIR_H = 250;   // Figures 1.2 and 1.3 share their chart height
  function tip(node, text) {
    const t = document.createElementNS(NS, "title");
    t.textContent = text;
    node.append(t);
    return node;
  }
  const bold = (s, x, y, text, attrs = {}) => {
    const t = el(s, "text", { x, y, class: "svg-label", ...attrs }, text);
    t.style.fontWeight = 600;
    return t;
  };

  Object.assign(CHARTS, {

    /* 1. Filter cascade: archive to labeled set, each step narrower */
    cascade(host, d) {
      const steps = d.sample.funnel;
      // On phones the number and name sit above each bar, so long names are not cut off on the left
      const W = widthOf(host), narrow = W < 560, rowH = narrow ? 66 : 48, H = steps.length * rowH + 4;
      const labelW = narrow ? 0 : Math.min(200, Math.round(W * 0.3)), gapL = narrow ? 0 : 16, x0 = labelW + gapL;
      const max = steps[0].count, minW = 10;
      const barMax = (W - x0) * (W < 560 ? 0.5 : 0.62);
      const s = svg(W, H, "From the archive to the labeled set");
      // Width on a log scale so every step stays visible; the number carries the exact size.
      const w = v => minW + (barMax - minW) * Math.log10(Math.max(v, 1)) / Math.log10(max);
      el(s, "line", { x1: x0 - 0.5, x2: x0 - 0.5, y1: 0, y2: H, class: "svg-grid" });
      steps.forEach((st, i) => {
        const y = i * rowH + 4, bw = w(st.count), mid = narrow ? y + 48 : y + 15, by = narrow ? y + 34 : y + 2;
        const last = i === steps.length - 1;
        // Name and exact size, right-aligned against the bars' common start
        const n = narrow ? el(s, "text", { x: 0, y: y + 13, class: "svg-value" }, num(st.count)) : el(s, "text", { x: labelW, y: mid - 1, "text-anchor": "end", class: "svg-value" }, num(st.count));
        if (narrow) el(s, "text", { x: 0, y: y + 28, class: "svg-text" }, st.label);
        else el(s, "text", { x: labelW, y: mid + 15, "text-anchor": "end", class: "svg-text" }, st.label);
        tip(el(s, "rect", { x: x0, y: by, width: bw, height: 26, rx: 3, class: last ? "svg-deep" : i < 3 ? "svg-soft" : "svg-dusty" }),
            `${st.label}: ${num(st.count)}`);
        if (i > 0) {
          // What each step did to the previous one, in words, right at the end of its bar
          const prev = steps[i - 1].count, share = st.count / prev;
          const note = {
            "Public": `${pct(share)} public`,
            "Public, scanned paper": `${pct(share)} scanned paper`,
            "Drawn at random": `1 in ${num(Math.round(1 / share))} drawn`,
            "Reviewed": `reviewed until ${num(d.labels.target)} valid`,
            "Valid labels": `${pct(share)} of reviewed`,
          }[st.label] || `${pct(share)} of previous`;
          const t = el(s, "text", { x: x0 + bw + 12, y: mid + 4, class: "svg-text" }, note);
          t.style.fill = "var(--accent)";
        }
      });
      host.append(s);
    },

    /* 2. Uncertainty across labeling rounds: slope with annotation */
    roundsSlope(host, d) {
      const sk = d.taxonomy.skims || [];
      if (sk.length < 2) return emptyNote(host, "Pilot rounds not found");
      // Each axis label wraps onto two lines under its own point; the dashed segment's note is in the legend row
      const pts = [
        { label: ["Generic", "categories"], name: "Generic categories", sub: `pilot · ${sk[0].valid != null ? `${sk[0].valid} valid labels` : `${sk[0].pages} pages`}`, v: sk[0].unsure },
        { label: ["Philip Morris", "departments"], name: "Philip Morris departments", sub: `pilot · ${sk[1].valid != null ? `${sk[1].valid} valid labels` : `${sk[1].pages} pages`}`, v: sk[1].unsure },
        { label: ["Final taxonomy,", "PM sample"], name: "Final taxonomy, PM sample", sub: `first pages · ${d.labels.valid} valid labels`, v: d.labels.uncertain_share },
      ];
      const W = widthOf(host), narrow = W < 480, H = PAIR_H, L = 46, R = 30, T = 30, B = narrow ? 88 : 74;
      const s = svg(W, H, "Share of labels marked uncertain in each labeling round");
      const x = i => L + (W - L - R) * (i / (pts.length - 1));
      const y = v => T + (H - T - B) * (1 - v / 0.5);
      [0, 0.25, 0.5].forEach(v => {
        el(s, "line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "svg-grid" });
        el(s, "text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "svg-text" }, pct(v));
      });
      el(s, "line", { x1: x(0), y1: y(pts[0].v), x2: x(1), y2: y(pts[1].v), class: "svg-slope" });
      el(s, "line", { x1: x(1), y1: y(pts[1].v), x2: x(2), y2: y(pts[2].v), class: "svg-slope dashed" });
      pts.forEach((p, i) => {
        tip(el(s, "circle", { cx: x(i), cy: y(p.v), r: 7, class: i === 1 ? "svg-deep" : "svg-dusty" }),
            `${p.name} (${p.sub}): ${pct(p.v)} uncertain`);
        const anchor = i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle";
        const ax = i === 0 ? x(i) - 7 : i === pts.length - 1 ? x(i) + 7 : x(i);
        bold(s, x(i), y(p.v) - 14, pct(p.v), { "text-anchor": anchor });
        p.label.forEach((line, k) => el(s, "text", { x: ax, y: H - B + 20 + k * 15, "text-anchor": anchor, class: "svg-label" }, line));
        (narrow ? p.sub.split(" · ") : [p.sub]).forEach((line, k) =>      // narrow charts: the sub-label on two lines too
          el(s, "text", { x: ax, y: H - B + 52 + k * 14, "text-anchor": anchor, class: "svg-text" }, line));
      });
      host.append(s);
    },

    /* 3. What happened to each reviewed document: one horizontal part-to-whole strip, and under it
          a list with one row per outcome (mark, name, count), so the small outcomes are read in the list, not as
          slivers. The chart keeps Figure 1.2's height (PAIR_H): rows are capped, never stretched. */
    outcomesStrip(host, d) {
      const parts = viewOf(d, host).outcomes.filter(o => o.count > 0);
      const total = parts.reduce((a, o) => a + o.count, 0);
      if (!total) return emptyNote(host, "No documents reviewed yet");
      const labeled = parts.filter(o => o.label.startsWith("Labeled")).reduce((a, o) => a + o.count, 0);
      const W = widthOf(host), H = PAIR_H;
      const s = svg(W, H, "Outcome of every reviewed document");
      bold(s, 0, 14, `${num(labeled)} of ${num(total)} labeled (${pct(labeled / total)}); ${num(total - labeled)} excluded with a reason`);
      const styles = o => o.label === "Labeled, certain" ? { class: "svg-deep" }
        : o.label === "Labeled, uncertain" ? { class: "svg-dusty" }
        : { class: "svg-soft" };
      // the strip
      const sy = 28, sh = 30;
      let x = 0;
      parts.forEach(o => {
        const w = W * o.count / total;
        tip(el(s, "rect", { x, y: sy, width: Math.max(w - 1.5, 1), height: sh, ...styles(o) }), `${o.label}: ${num(o.count)}`);
        x += w;
      });
      // the list: one row per outcome, in strip order
      const ly = sy + sh + 18, rowH = Math.min(30, (H - ly - 4) / parts.length);
      parts.forEach((o, i) => {
        const y = ly + i * rowH;
        if (i) el(s, "line", { x1: 0, x2: W, y1: y, y2: y, class: "svg-grid" });
        el(s, "rect", { x: 0, y: y + rowH / 2 - 6, width: 12, height: 12, rx: 1.5, ...styles(o) });
        el(s, "text", { x: 20, y: y + rowH / 2 + 4.5, class: "svg-label" }, o.label);
        el(s, "text", { x: W, y: y + rowH / 2 + 4.5, "text-anchor": "end", class: "svg-value" }, num(o.count));
      });
      host.append(s);
    },

    /* 4. Labels per department: certain + uncertain (labeling complete, no projections) */
    deptBars(host, d) {
      const v = viewOf(d, host);
      // Same order in every batch view: largest department across all labels first.
      const rank = Object.fromEntries(d.labels.by_department.map(r => [r.id, -(r.certain + r.uncertain)]));
      const rows = [...v.by_department].sort((p, q) => rank[p.id] - rank[q.id]);
      if (!rows.length) return emptyNote(host, "No labels yet");
      const total = rows.reduce((a, r) => a + r.certain + r.uncertain, 0);
      // Names go above the bars when they would not fit beside them
      const W = widthOf(host), narrow = W < 560, rowH = narrow ? 44 : 30, H = rows.length * rowH + 6;
      const s = svg(W, H, "Labels per department");
      const x0 = narrow ? 0 : Math.min(210, W * .36), x1 = W - (narrow ? 90 : 110);
      const m = Math.max(...rows.map(r => r.certain + r.uncertain), 1);
      const sc = n => (x1 - x0) * n / m;
      rows.forEach((r, i) => {
        const y = i * rowH + 4, n = r.certain + r.uncertain;
        const by = narrow ? y + 20 : y + 4;
        el(s, "text", { x: 0, y: narrow ? y + 12 : y + 16, class: "svg-label" }, r.name);
        tip(el(s, "rect", { x: x0, y: by, width: Math.max(0, sc(r.certain)), height: 16, class: "svg-deep" }), `${r.name}: ${r.certain} certain`);
        tip(el(s, "rect", { x: x0 + sc(r.certain), y: by, width: Math.max(0, sc(r.uncertain)), height: 16, class: "svg-dusty" }), `${r.name}: ${r.uncertain} marked uncertain`);
        bold(s, x0 + sc(n) + 8, by + 12, String(n));
        el(s, "text", { x: W, y: by + 12, "text-anchor": "end", class: "svg-text" }, total ? pct(n / total) : "");
      });
      host.append(s);
    },
  });
})();
