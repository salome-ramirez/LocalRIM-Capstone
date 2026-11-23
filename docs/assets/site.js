/* LocalRIM site script.
   1. Adds the shared header (file tabs) and footer to every page.
   2. Loads data/site_data.json (written by build_site_data.py) and fills
      every element with a data-field attribute, e.g. <b data-field="labels.valid">.
   3. Draws the charts named in data-chart attributes, as plain SVG (no library).
   Pages work from any folder because every path is relative. */

const PAGES = [
  ["dashboard.html", "Dashboard"],
  ["filing-system.html", "01 Filing system"],
  ["reading.html", "02 Reading the pages"],
  ["review.html", "03 Filing and review"],
  ["findings.html", "Findings"],
];
const FOOTER = "Client archive · Philip Morris USA records, UCSF Industry Documents Library, public through the 1998 Master Settlement Agreement and the 2006 U.S. v. Philip Morris ruling.";
const SVGNS = "http://www.w3.org/2000/svg";

/* Shell */
function buildShell() {
  // A page with its own masthead keeps it; only the footer is added.
  if (document.querySelector(".masthead")) {
    const m = document.querySelector("main");
    if (m && !m.querySelector("footer.site")) { const f = document.createElement("footer"); f.className = "site"; f.textContent = FOOTER; m.append(f); }
    return;
  }
  const here = location.pathname.split("/").pop() || "index.html";
  const header = document.createElement("header");
  header.className = "site";
  header.innerHTML =
    `<a class="brand" href="index.html">LocalRIM <span>/ Notes from the File Room</span></a>` +
    `<nav class="tabs" aria-label="Project files">` +
    PAGES.map(([href, name]) => `<a href="${href}"${href === here ? ' aria-current="page"' : ""}>${name}</a>`).join("") +
    `</nav>`;
  document.body.prepend(header);
  const main = document.querySelector("main");
  if (!main.querySelector("footer.site")) {
    const footer = document.createElement("footer");
    footer.className = "site";
    footer.textContent = FOOTER;
    main.append(footer);
  }
}

/* Data binding */
const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj);
const fmt = (v, kind) => {
  if (v == null || Number.isNaN(v)) return "—";
  if (kind === "pct") return `${Math.round(v * 100)}%`;
  if (kind === "pct1") return `${(v * 100).toFixed(1)}%`;
  if (kind === "dec2") return Number(v).toFixed(2);
  // A difference with its sign ("+0.31", "−0.01"); a share difference in points ("10"); a small count as a word ("seven")
  if (kind === "signed2") { const s = Number(v).toFixed(2); return Number(s) < 0 ? `−${s.slice(1)}` : `+${s}`; }
  if (kind === "pts") return String(Math.round(Number(v) * 100));
  if (kind === "word") { const w = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]; return w[v] || Number(v).toLocaleString("en-US"); }
  // ISO date ("2026-11-23") → "Nov 23, 2026"; UTC so the day never shifts
  if (kind === "date") return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(new Date(`${v}T00:00:00Z`));
  if (typeof v === "number") return v.toLocaleString("en-US");
  return String(v);
};
function bindFields(data) {
  document.querySelectorAll("[data-field]").forEach(el => {
    el.textContent = fmt(get(data, el.dataset.field), el.dataset.format);
  });
  document.querySelectorAll("[data-progress]").forEach(el => {
    const [num, den] = el.dataset.progress.split("/").map(p => get(data, p));
    el.style.width = `${Math.min(100, (100 * (num || 0)) / (den || 1))}%`;
  });
  // Data-sum="a+b" shows the sum of two data fields (01: documents labeled by hand = training + final test)
  document.querySelectorAll("[data-sum]").forEach(el => {          // "a+b" or "a-b"
    const m = el.dataset.sum.match(/^(.+?)([+-])(.+)$/);
    const x = m && get(data, m[1].trim()), y = m && get(data, m[3].trim());
    el.textContent = typeof x === "number" && typeof y === "number" ? fmt(m[2] === "+" ? x + y : x - y, el.dataset.format) : "—";
  });
  // Findings' "every 100 pages": data-hundred="share" shows round(100 x share), or the rest of 100 with data-rest
  // data-flex sets the same count as a flex weight, so the strip's parts come from the data
  document.querySelectorAll("[data-hundred], [data-flex]").forEach(el => {
    const v = get(data, el.dataset.hundred || el.dataset.flex);
    if (typeof v !== "number") return;
    const n = Math.round(100 * v), k = el.hasAttribute("data-rest") ? 100 - n : n;
    if (el.dataset.hundred) el.textContent = k; else el.style.flex = `${k} 1 0`;
  });
  // Data-share="a/b" shows a/b as a whole percentage (02: Bates numbers found)
  document.querySelectorAll("[data-share]").forEach(el => {
    const [a, b] = el.dataset.share.split("/").map(p => get(data, p.trim()));
    el.textContent = typeof a === "number" && b ? `${Math.round(100 * a / b)}%` : "—";
  });
  // Figure 3.14 — minutes for a box of the scale run's size at its seconds per page
  document.querySelectorAll('[data-derived="scale-minutes"]').forEach(el => {
    const r = data.scale_run;
    el.textContent = r && r.timing ? fmt(Math.round(r.timing.reduce((a, t) => a + t.seconds, 0) * r.pages / 60)) : "—";
  });
  // Figure 3.14 — hours for the scale run and seconds per page, from the same timing
  document.querySelectorAll('[data-derived="scale-hours"], [data-derived="scale-seconds"]').forEach(el => {
    const r = data.scale_run, s = r && r.timing ? r.timing.reduce((a, t) => a + t.seconds, 0) : null;
    el.textContent = s == null ? "—" : el.dataset.derived === "scale-hours" ? (s * r.pages / 3600).toFixed(1) : s.toFixed(1);
  });
  // The app page's assistant timing, "18–19 s" from each round's median seconds
  document.querySelectorAll('[data-derived="assistant-seconds"]').forEach(el => {
    const r = data.archive_check && data.archive_check.rounds, m = r ? r.map(x => x.median_seconds).filter(v => typeof v === "number") : [];
    if (!m.length) { el.textContent = "—"; return; }
    const lo = Math.round(Math.min(...m)), hi = Math.round(Math.max(...m));
    el.textContent = lo === hi ? `${lo} s` : `${lo}–${hi} s`;
  });
  // Figure 0 caption: replacements = drawn − labeling pool − scale batch (D-42)
  document.querySelectorAll('[data-derived="sample-replacements"]').forEach(el => {
    const s = data.sample || {}, r = data.scale_run;
    el.textContent = s.labeling_pool && r ? fmt(s.drawn - s.labeling_pool - r.pages, el.dataset.format) : "—";
  });
  // Speed ratios: data-ratio="path/path" gives the rounded ratio as a word
  // so every page states the same comparison with the same calculated number.
  const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
  document.querySelectorAll("[data-ratio]").forEach(el => {
    const [num, den] = el.dataset.ratio.split("/").map(p => get(data, p));
    const n = num && den ? Math.round(num / den) : null;
    el.textContent = n == null ? "—" : WORDS[n] || n.toLocaleString("en-US");
  });
  const updated = document.querySelector("[data-updated]");
  const asOf = v => String(v).replace(/,\s*\d{4}/, "").replace(/\s+\d{1,2}:\d{2}$/, "");
  if (updated && updated.dataset.updated === "dev-test" && data.data_as_of && data.final_test && data.final_test.labels_as_of)   // 01 · Intake
    updated.textContent = `Development labels as of ${asOf(data.data_as_of)}; test labels ${asOf(data.final_test.labels_as_of)}`;
  else if (updated && updated.dataset.updated === "dev-final" && data.data_as_of && data.final_test && data.final_test.labels_as_of)   // Project log
    updated.textContent = `Development labels ${asOf(data.data_as_of)}; final-test labels ${String(data.final_test.labels_as_of).replace(/\s+\d{1,2}:\d{2}$/, "")}`;
  else if (updated && (data.data_as_of || data.updated)) updated.textContent = data.data_as_of ? `Labels as of ${String(data.data_as_of).replace(/\s+\d{1,2}:\d{2}$/, "")}` : `Data updated ${String(data.updated).replace(/\s+\d{1,2}:\d{2}$/, "")}`;   // Date only
}

/* SVG helpers */
// Charts are drawn at the container's real width, so text stays 12-13px
// whether the chart sits in a full-width panel or a half-width column.
const widthOf = host => Math.max(300, Math.round(host.getBoundingClientRect().width) || 760);
function svg(w, h, title) {
  const s = document.createElementNS(SVGNS, "svg");
  s.setAttribute("viewBox", `0 0 ${w} ${h}`);
  s.setAttribute("width", w);
  s.setAttribute("height", h);
  s.setAttribute("role", "img");
  const t = document.createElementNS(SVGNS, "title");
  t.textContent = title;
  s.append(t);
  return s;
}
function el(parent, tag, attrs = {}, text) {
  const n = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null) continue;
    if (k === "fill" && String(v).startsWith("var(")) n.style.fill = v;   // CSS variables need style, not attributes
    else n.setAttribute(k, v);
  }
  if (text != null) n.textContent = text;
  parent.append(n);
  return n;
}
function emptyNote(host, text) {
  const d = document.createElement("div");
  d.className = "empty-plot";
  d.textContent = text;
  host.append(d);
}

/* Horizontal bars. rows: [{label, value, cls, display}] */
function hbars(host, rows, { title, max, threshold, thresholdLabel, log = false, labelW = 210, valueW = 90 } = {}) {
  // On narrow charts the label goes above its bar instead of being overdrawn by it
  const narrow = widthOf(host) < 560;
  labelW = narrow ? 0 : Math.min(labelW, Math.round(widthOf(host) * 0.42));
  const W = widthOf(host), rowH = narrow ? 40 : 24, top = threshold != null ? 20 : 6, H = top + rows.length * rowH + 6;
  const s = svg(W, H, title);
  const x0 = labelW, x1 = W - valueW;
  const m = max ?? Math.max(...rows.map(r => r.value), 1);
  const scale = v => x0 + (x1 - x0) * (log ? Math.log10(Math.max(v, 1)) / Math.log10(m) : v / m);
  rows.forEach((r, i) => {
    const y = top + i * rowH;
    const by = narrow ? y + 20 : y + 4;
    el(s, "text", { x: 0, y: narrow ? y + 13 : y + 16, class: "svg-label" }, r.label);
    el(s, "rect", { x: x0, y: by, width: x1 - x0, height: 14, class: "svg-soft", opacity: .45 });
    el(s, "rect", { x: x0, y: by, width: Math.max(1, scale(r.value) - x0), height: 14, class: r.cls || "svg-dusty" });
    el(s, "text", { x: W, y: by + 12, "text-anchor": "end", class: "svg-value" }, r.display ?? fmt(r.value));
  });
  if (threshold != null) {
    const tx = scale(threshold);
    el(s, "line", { x1: tx, x2: tx, y1: 12, y2: H, class: "svg-threshold" });
    el(s, "text", { x: tx + 4, y: 11, class: "svg-text" }, thresholdLabel || "");
  }
  host.append(s);
}

/* Batch view
   A chart host with data-batch="Batch 2" shows that batch; no value = all labels. */
function viewOf(d, host) {
  const b = host.dataset.batch ? (d.labels.by_batch || []).find(x => x.batch === host.dataset.batch) : null;
  return b ? { batch: b, by_department: b.by_department, outcomes: b.outcomes, pairs: b.pairs, type_counts: b.type_counts }
           : { batch: null, by_department: d.labels.by_department, outcomes: d.labels.outcomes,
               pairs: d.labels.pairs, type_counts: d.labels.type_counts || {} };
}

/* Charts */
const CHARTS = {
  /* Dashboard: valid labels over time against the target */
  progress(host, d) {
    const pts = d.labels.progress || [];
    const target = d.labels.target;
    if (!pts.length) return emptyNote(host, "No labels yet");
    const W = widthOf(host), H = 220, L = 44, R = 20, T = 14, B = 34;
    const s = svg(W, H, "Valid labels over time");
    const y = v => H - B - (H - T - B) * v / target;
    const x = i => L + (W - L - R) * (pts.length === 1 ? .5 : i / (pts.length - 1));
    [0, .25, .5, .75, 1].forEach(f => {
      el(s, "line", { x1: L, x2: W - R, y1: y(target * f), y2: y(target * f), class: "svg-grid" });
      el(s, "text", { x: L - 8, y: y(target * f) + 4, "text-anchor": "end", class: "svg-text" }, Math.round(target * f));
    });
    el(s, "line", { x1: L, x2: W - R, y1: y(target), y2: y(target), class: "svg-threshold" });
    el(s, "text", { x: W - R, y: y(target) - 6, "text-anchor": "end", class: "svg-text" }, `Target ${target}`);
    el(s, "polyline", { points: pts.map((p, i) => `${x(i)},${y(p.valid)}`).join(" "), class: "svg-line" });
    pts.forEach((p, i) => {
      el(s, "circle", { cx: x(i), cy: y(p.valid), r: 4, class: "svg-deep" });
      el(s, "text", { x: x(i), y: y(p.valid) - 9, "text-anchor": "middle", class: "svg-value" }, p.valid);
      el(s, "text", { x: x(i), y: H - 12, "text-anchor": "middle", class: "svg-text" }, p.date.slice(5));
    });
    host.append(s);
  },

  /* Dashboard, Figure 0: the labeling pool, batch by batch.
     Only the pool (sample.labeling_pool, D-42) is drawn; the rest of the sample
     (replacements, scale batch) is described in the caption. Each row: the bar,
     then its counts on a line underneath, so labels never run off narrow screens. */
  batches(host, d) {
    const all = d.labels.batches || [];
    const pool = d.sample && d.sample.labeling_pool;
    if (!all.length) return emptyNote(host, "No sample yet");
    if (!pool) return emptyNote(host, "The labeling pool size is not in the site data yet. Rebuild site_data.json.");
    const rows = all.slice(0, Math.ceil(pool / 100));
    const W = widthOf(host), rowH = 48, H = rows.length * rowH, x0 = Math.min(80, W * .22), x1 = W;
    const s = svg(W, H, `The labeling pool, ${pool} documents in batches of 100: labeled, excluded, skipped or not reviewed`);
    const parts = [
      ["certain", { class: "svg-deep" }, "labeled, certain"],
      ["uncertain", { class: "svg-dusty" }, "labeled, uncertain"],
      ["set_aside", { class: "svg-soft" }, "excluded"],
      ["skipped", { fill: "var(--panel)", stroke: "var(--accent)", "stroke-width": 1.5 }, "skipped (scan failed to download)"],
      ["remaining", { class: "svg-empty" }, "not reviewed"],
    ];
    rows.forEach((r, i) => {
      const y = i * rowH + 2, total = parts.reduce((a, [k]) => a + (r[k] || 0), 0) || 1;
      el(s, "text", { x: 0, y: y + 15, class: "svg-label" }, r.batch);
      let x = x0;
      parts.forEach(([k, attrs, name]) => {
        const w = (x1 - x0) * (r[k] || 0) / total;
        if (w > 0) {
          const q = el(s, "rect", { x, y: y + 2, width: w, height: 18, ...attrs });
          if (attrs.stroke) q.style.stroke = attrs.stroke;
          const t = document.createElementNS(SVGNS, "title");
          t.textContent = `${r.batch}: ${r[k]} ${name}`;
          q.append(t);
        }
        x += w;
      });
      const valid = r.certain + r.uncertain;
      el(s, "text", { x: x0, y: y + 37, class: valid ? "svg-value" : "svg-text" },
        [valid ? `${valid} valid` : "", r.set_aside ? `${r.set_aside} excluded` : "",
         r.skipped ? `${r.skipped} skipped` : "", r.remaining ? `${r.remaining} not reviewed` : ""].filter(Boolean).join(" · "));
    });
    host.append(s);
  },

  /* 01: how the sample was drawn, archive to labels (log scale) */
  funnel(host, d) {
    const f = d.sample.funnel;
    hbars(host, f.map(r => ({ label: r.label, value: r.count })), {
      title: "From the archive to the labeled set", log: true, max: f[0].count, labelW: 250, valueW: 100,
    });
  },

  /* 01: uncertainty under each taxonomy version */
  skims(host, d) {
    hbars(host, d.taxonomy.skims.map(r => ({
      label: `${r.label} · ${r.pages} pages`, value: r.unsure, display: fmt(r.unsure, "pct"),
      cls: r.current ? "svg-dusty" : "svg-soft",
    })), { title: "Share of labels marked uncertain", max: 1, labelW: 290 });
  },

  /* 01: labels per department, certain + uncertain, with projection to target */
  departments(host, d) {
    // Batch filter: host.dataset.batch picks one batch; empty = all labels.
    const v = viewOf(d, host), sel = v.batch;
    const rows = v.by_department;
    if (!d.labels.valid) return emptyNote(host, "No labels yet");
    const W = widthOf(host), rowH = 28, top = 6, H = top + rows.length * rowH + 4;
    const s = svg(W, H, "Labels per department");
    const x0 = Math.min(200, W * .36), x1 = W - (W < 500 ? 110 : 170);
    const m = Math.max(...rows.map(r => r.certain + r.uncertain), 1);
    const sc = v => (x1 - x0) * v / m;
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      el(s, "text", { x: 0, y: y + 17, class: "svg-label" }, r.name);
      el(s, "rect", { x: x0, y: y + 5, width: Math.max(0, sc(r.certain)), height: 15, class: "svg-deep" });
      el(s, "rect", { x: x0 + sc(r.certain), y: y + 5, width: Math.max(0, sc(r.uncertain)), height: 15, class: "svg-dusty" });
      el(s, "text", { x: x0 + sc(r.certain + r.uncertain) + 6, y: y + 17, class: "svg-value" }, r.certain + r.uncertain);
      if (sel) return;   // projections only make sense for all labels
      const low = r.projected < d.labels.low_count_rule;
      el(s, "text", { x: W, y: y + 17, "text-anchor": "end", class: low ? "svg-value" : "svg-text",
        fill: low ? "var(--accent)" : null }, `${low ? "▲ " : ""}${r.projected} projected`);
    });
    host.append(s);
  },

  /* 01: what happened to every reviewed document */
  outcomes(host, d) {
    const parts = viewOf(d, host).outcomes.filter(o => o.count > 0);
    const total = parts.reduce((a, o) => a + o.count, 0);
    if (!total) return emptyNote(host, "No documents reviewed yet");
    const W = widthOf(host), H = 34;
    const s = svg(W, H, "Outcome of reviewed documents");
    const cls = ["svg-deep", "svg-dusty", "svg-soft", "svg-empty", "svg-soft", "svg-empty"];
    let x = 0;
    parts.forEach((o, i) => {
      const w = W * o.count / total;
      el(s, "rect", { x, y: 4, width: w, height: 22, class: cls[i % cls.length] });
      if (w > 34) el(s, "text", { x: x + w / 2, y: 20, "text-anchor": "middle", class: "svg-value",
        fill: i === 0 ? "var(--on)" : null }, o.count);
      x += w;
    });
    host.append(s);
    const lg = document.createElement("div");
    lg.className = "legend";
    lg.innerHTML = parts.map((o, i) =>
      `<span><i style="background:${["var(--deep)", "var(--dusty)", "var(--soft)", "transparent", "var(--soft)", "transparent"][i % 6]};outline:1px solid var(--line)"></i>${o.label} · ${o.count}</span>`).join("");
    host.append(lg);
  },

  /* 01: uncertainty as a dot plot: one dot per department on 0-100%,
     dot size = labels, dashed line = overall share */
  uncertaintyDots(host, d) {
    const v = viewOf(d, host);
    const rows = v.by_department.map(r => ({ ...r, n: r.certain + r.uncertain })).filter(r => r.n > 0)
      .sort((a, b) => b.uncertain / b.n - a.uncertain / a.n);
    if (!rows.length) return emptyNote(host, "No labels yet");
    const all = v.batch ? v.batch.unsure_share : d.labels.uncertain_share;
    const W = widthOf(host), rowH = 30, T = 24, B = 26, H = T + rows.length * rowH + B;
    const x0 = Math.min(210, W * .34), x1 = W - 90;
    const xs = p => x0 + (x1 - x0) * p;
    const maxN = Math.max(...rows.map(r => r.n));
    const s = svg(W, H, "Share of each department's labels marked uncertain");
    [0, .25, .5, .75, 1].forEach(t => {
      el(s, "line", { x1: xs(t), x2: xs(t), y1: T - 6, y2: H - B, class: "svg-grid" });
      el(s, "text", { x: xs(t), y: H - 8, "text-anchor": "middle", class: "svg-text" }, `${t * 100}%`);
    });
    if (all != null) {
      el(s, "line", { x1: xs(all), x2: xs(all), y1: T - 10, y2: H - B, class: "svg-threshold" });
      el(s, "text", { x: xs(all) + 5, y: T - 12, class: "svg-text" }, `All departments ${Math.round(all * 100)}%`);
    }
    rows.forEach((r, i) => {
      const y = T + i * rowH + rowH / 2, p = r.uncertain / r.n;
      el(s, "line", { x1: x0, x2: x1, y1: y, y2: y, class: "svg-grid", "stroke-dasharray": "1 4" });
      el(s, "text", { x: 0, y: y + 4, class: "svg-label" }, r.name);
      el(s, "circle", { cx: xs(p), cy: y, r: 4 + 8 * Math.sqrt(r.n / maxN), class: r.n < 10 ? "svg-soft" : "svg-deep",
        "fill-opacity": .9 });
      el(s, "text", { x: W, y: y + 4, "text-anchor": "end", class: "svg-value" }, `${Math.round(p * 100)}% of ${r.n}`);
    });
    host.append(s);
  },

  /* 01: uncertainty as a department x batch grid, darker = more uncertain */
  uncertaintyGrid(host, d) {
    const batches = d.labels.by_batch || [];
    if (!batches.length) return emptyNote(host, "No labels yet");
    const cols = [...batches.map(b => ({ name: b.batch, rows: b.by_department })),
                  { name: "All labels", rows: d.labels.by_department }];
    const cell = (r) => {
      const n = r.certain + r.uncertain;
      if (!n) return `<td class="ugrid-empty">—</td>`;
      const p = r.uncertain / n;
      const a = (0.08 + 0.72 * p).toFixed(2);
      return `<td style="background: color-mix(in srgb, var(--deep) ${Math.round(a * 100)}%, var(--panel)); color: ${p > .8 ? "var(--on)" : "var(--ink)"}"
        title="${Math.round(p * 100)}% of ${n} labels">${Math.round(p * 100)}%<small>${n}</small></td>`;
    };
    host.innerHTML = `<div class="scroll"><table class="ugrid"><thead><tr><th>Department</th>${cols.map(c => `<th>${c.name}</th>`).join("")}</tr></thead><tbody>` +
      d.labels.by_department.map((dep, i) =>
        `<tr><th scope="row">${dep.name}</th>${cols.map(c => cell(c.rows[i])).join("")}</tr>`).join("") +
      `</tbody></table></div>`;
  },

  /* 01: did the department mix settle? one small chart per department */
  settling(host, d) {
    const st = d.labels.settling;
    if (!st || st.steps.length < 2) return emptyNote(host, "Not enough labels yet");
    const n = st.steps, last = n[n.length - 1];
    const peak = Math.max(...Object.values(st.series).flat(), .05);
    const top = Math.ceil(peak * 10) / 10;
    const grid = document.createElement("div");
    grid.className = "settle-grid";
    d.labels.by_department.forEach(dep => {
      const ys = st.series[dep.id] || [];
      const cell = document.createElement("div");
      cell.className = "settle-cell";
      const end = ys[ys.length - 1] || 0;
      cell.innerHTML = `<div class="settle-head"><span>${dep.name}</span><b>${Math.round(end * 100)}%</b></div>`;
      const W = 220, H = 70, L = 2, R = 2, T = 6, B = 6;
      const s = svg(W, H, `${dep.name}: share of labels as labeling went on`);
      s.removeAttribute("width"); s.style.width = "100%"; s.style.height = "auto";
      const xs = v => L + (W - L - R) * (v - n[0]) / (last - n[0] || 1);
      const yv = v => H - B - (H - T - B) * v / top;
      (st.batch_ends || []).forEach(e => el(s, "line", { x1: xs(e), x2: xs(e), y1: T, y2: H - B, class: "svg-grid", "stroke-dasharray": "2 3" }));
      el(s, "line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "svg-grid" });
      el(s, "path", { d: ys.map((v, i) => `${i ? "L" : "M"}${xs(n[i]).toFixed(1)},${yv(v).toFixed(1)}`).join(""), class: "svg-line" });
      el(s, "circle", { cx: xs(last), cy: yv(end), r: 3, class: "svg-deep" });
      cell.append(s);
      grid.append(cell);
    });
    host.append(grid);
  },

  /* 01: which departments share pages (primary -> secondary) */
  pairs(host, d) {
    const p = viewOf(d, host).pairs || [];
    if (!p.length) return emptyNote(host, "No secondary classifications yet");
    hbars(host, p.map(r => ({ label: `${r.a} ↔ ${r.b}`, value: r.count })), {
      title: "Department pairs that share pages", labelW: 290,
    });
  },

  /* 02: spot-check, Tesseract confidence vs a person's rating */
  spotcheck(host, d) {
    const rows = d.ocr.spotcheck || [];
    if (!rows.length) return emptyNote(host, "Spot-check not run");
    const W = widthOf(host), rowH = 24, top = 20, H = top + rows.length * rowH + 22, x0 = Math.min(150, W * .3), x1 = W - 120;
    const s = svg(W, H, "OCR spot-check: confidence and readability");
    const sc = v => x0 + (x1 - x0) * v / 100;
    [0, 25, 50, 75, 100].forEach(v => {
      el(s, "line", { x1: sc(v), x2: sc(v), y1: top - 4, y2: H - 20, class: "svg-grid" });
      el(s, "text", { x: sc(v), y: H - 4, "text-anchor": "middle", class: "svg-text" }, v);
    });
    el(s, "line", { x1: sc(70), x2: sc(70), y1: 4, y2: H - 20, class: "svg-threshold" });
    el(s, "text", { x: sc(70) + 4, y: 12, class: "svg-text" }, "70 review threshold to test");
    const names = { 1: "Poor", 2: "Usable", 3: "Good" };
    rows.forEach((r, i) => {
      const y = top + i * rowH + 12;
      el(s, "text", { x: 0, y: y + 4, class: "svg-label" }, r.page_type);
      if (r.rating === 3) el(s, "circle", { cx: sc(r.conf), cy: y, r: 6, class: "svg-deep" });
      else if (r.rating === 2) el(s, "circle", { cx: sc(r.conf), cy: y, r: 6, class: "svg-dusty" });
      else el(s, "rect", { x: sc(r.conf) - 6, y: y - 6, width: 12, height: 12, fill: "var(--panel)", stroke: "var(--accent)", "stroke-width": 2 });
      el(s, "text", { x: W, y: y + 4, "text-anchor": "end", class: "svg-text" }, `${r.conf.toFixed(1)} · ${names[r.rating]}`);
    });
    host.append(s);
  },

  /* 02: FUNSD, confidence vs share of true words recovered, one dot per form */
  funsd(host, d) {
    const pts = d.ocr.funsd.points || [];
    if (!pts.length) return emptyNote(host, "FUNSD evaluation not run");
    const W = widthOf(host), H = 300, L = 50, R = 16, T = 12, B = 40;
    const s = svg(W, H, "OCR confidence vs word recall on FUNSD forms");
    const xs = v => L + (W - L - R) * (v - 50) / 50;
    const ys = v => H - B - (H - T - B) * v;
    [50, 60, 70, 80, 90, 100].forEach(v => {
      el(s, "line", { x1: xs(v), x2: xs(v), y1: T, y2: H - B, class: "svg-grid" });
      el(s, "text", { x: xs(v), y: H - B + 16, "text-anchor": "middle", class: "svg-text" }, v);
    });
    [0, .25, .5, .75, 1].forEach(v => {
      el(s, "line", { x1: L, x2: W - R, y1: ys(v), y2: ys(v), class: "svg-grid" });
      el(s, "text", { x: L - 8, y: ys(v) + 4, "text-anchor": "end", class: "svg-text" }, `${v * 100}%`);
    });
    el(s, "text", { x: (L + W) / 2, y: H - 4, "text-anchor": "middle", class: "svg-text" }, "Tesseract mean confidence per form");
    el(s, "text", { x: 12, y: (T + H - B) / 2, transform: `rotate(-90 12 ${(T + H - B) / 2})`, "text-anchor": "middle", class: "svg-text" }, "True words recovered");
    pts.forEach(p => el(s, "circle", { cx: xs(Math.max(50, p.conf)), cy: ys(p.recall), r: 5, class: "svg-dusty", "fill-opacity": .85 }));
    host.append(s);
  },

  /* 02: pilot, mean confidence by original page type (RVL pages) */
  pilot(host, d) {
    const rows = d.ocr.pilot_by_type || [];
    hbars(host, rows.map(r => ({ label: r.type, value: r.conf, display: r.conf.toFixed(1),
      cls: r.conf < 70 ? "svg-soft" : "svg-dusty" })), {
      title: "Mean OCR confidence by page type", max: 100, threshold: 70,
      thresholdLabel: "70 review threshold to test", labelW: 190,
    });
  },
};

/* Taxonomy coverage (HTML, not SVG) */
function coverage(host, d) {
  const counts = viewOf(d, host).type_counts;
  const html = d.taxonomy.tree.map(dep => {
    const series = dep.series.map(se =>
      `<div class="series">${se.name}</div>` + se.types.map(t0 => ({ ...t0, count: counts[t0.id] || 0 })).map(t =>
        // One square per label, no cap (rows wrap); the count shows on hover or
        // Keyboard focus for every type in use.
        `<div class="type${t.count ? "" : " unused"}"${t.count ? ` tabindex="0" data-count="${t.count} label${t.count === 1 ? "" : "s"}"` : ""}><span>${t.name}</span>` +
        `<span class="pips" aria-label="${t.count} labels">${t.count ? "<i></i>".repeat(t.count) : "—"}</span></div>`
      ).join("")).join("");
    return `<article><h3>${dep.name}</h3>${series}</article>`;
  }).join("");
  host.innerHTML = html;
}

/* Tables built from data */
function pipeline(host, d) {
  /* Statuses come from site_data.json (pipeline) */
  host.innerHTML = d.pipeline.map(p =>
    `<li class="${p.status}"><b>${p.step}</b><span>${{ done: "Complete", active: "In progress", next: "Next", later: "Later" }[p.status]} · ${p.value}</span></li>`).join("");
}
function ledger(host, d) {
  const rows = host.dataset.limit ? d.decisions.slice(-host.dataset.limit) : d.decisions;
  host.innerHTML = rows.slice().reverse().map(r =>
    `<tr><td>${r.id}</td><td>${r.date}</td><td>${r.decision}</td><td>${r.why}</td></tr>`).join("");
}

/* Batch filter
   <div data-batch-filter="HOST-ID"> holds buttons for one chart; each chart
   filters on its own. <p data-batch-note="HOST-ID"> shows the selection's numbers. */
function redraw(host, data) {
  host.replaceChildren();
  const name = host.dataset.chart;
  if (name === "coverage") coverage(host, data);
  else CHARTS[name](host, data);
}
function batchFilter(data) {
  const batches = data.labels.by_batch || [];
  const pct = v => v == null ? "—" : `${Math.round(v * 100)}%`;
  document.querySelectorAll("[data-batch-filter]").forEach(bar => {
    const id = bar.dataset.batchFilter;
    const host = document.getElementById(id);
    const note = document.querySelector(`[data-batch-note="${id}"]`);
    const options = [["", "All labels"], ...batches.map(b => [b.batch, b.batch])];
    bar.innerHTML = options.map(([v, t]) =>
      `<button type="button" data-value="${v}" aria-pressed="${v === "" ? "true" : "false"}">${t}</button>`).join("");
    const show = v => {
      host.dataset.batch = v;
      redraw(host, data);
      bar.querySelectorAll("button").forEach(b => b.setAttribute("aria-pressed", String(b.dataset.value === v)));
      const b = batches.find(x => x.batch === v), l = data.labels;
      if (note) note.textContent = b
        ? `${b.batch}: ${b.valid} valid of ${b.reviewed} reviewed · ${pct(b.unsure_share)} uncertain · ${b.excluded} excluded · ${b.skipped} skipped`
        : `All labels: ${l.valid} valid of ${l.reviewed} reviewed · ${pct(l.uncertain_share)} uncertain · ${l.excluded} excluded · ${l.skipped} skipped`;
      document.querySelectorAll(`[data-projection-legend="${id}"]`).forEach(x => { x.hidden = Boolean(b); });
    };
    bar.addEventListener("click", e => { const btn = e.target.closest("button"); if (btn) show(btn.dataset.value); });
    show("");
  });
}

/* Start */
buildShell();
// Draw only after every deferred chapter script (review.js, ...) has added its charts.
const domReady = new Promise(res => document.readyState === "complete" ? res() : document.addEventListener("DOMContentLoaded", res));
fetch("data/site_data.json", { cache: "no-store" })
  .then(r => r.json())
  .then(data => domReady.then(() => data))
  .then(data => {
    bindFields(data);
    // Hosts marked data-cf are drawn by casefile.js (the shared case-file charts)
    document.querySelectorAll("[data-chart]:not([data-cf])").forEach(host => {
      const name = host.dataset.chart;
      try {
        if (name === "coverage") coverage(host, data);
        else if (name === "pipeline") pipeline(host, data);
        else if (name === "ledger") ledger(host, data);
        else CHARTS[name](host, data);
      } catch (e) { console.error(name, e); emptyNote(host, "Chart unavailable"); }
    });
    try { batchFilter(data); } catch (e) { console.error("batch filter", e); }
  })
  .catch(() => {
    document.querySelectorAll("[data-chart]:not([data-cf])").forEach(h => emptyNote(h, "Data file not found. Run build_site_data.py, then preview with python -m http.server."));
  });
