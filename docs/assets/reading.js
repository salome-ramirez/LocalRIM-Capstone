/* Reading chapter charts (02 Reading the pages), .
   Adds ten figures to the shared CHARTS table in site.js and reuses its helpers
   (svg, el, widthOf, fmt, emptyNote). Every number comes from site_data.json
   (ocr.engines, written by build_site_data.py from ocr_compare.py's tables).
   Engine identity is carried by labels; color only marks emphasis. */

(() => {
  const NS = "http://www.w3.org/2000/svg";
  const eng = d => d.ocr && d.ocr.engines;
  const ENGINES = ["A", "B", "C", "D", "E"];
  const SHORT = { A: "Tesseract, 150 dpi", B: "Tesseract, full res.", C: "PaddleOCR", D: "GLM-OCR", E: "UCSF text layer" };
  const GROUP_ORDER = ["1", "4", "3b", "3a", "2"];          // most pages to fewest
  const HARD = new Set(["2", "3a", "3b"]);
  const pct = v => `${Math.round(v * 100)}%`;
  const groupName = (d, id) => (eng(d).scan_groups.find(g => g.id === id) || {}).name || id;

  function tip(node, text) {
    const t = document.createElementNS(NS, "title");
    t.textContent = text;
    node.append(t);
    return node;
  }
  function need(host, d, key) {
    const e = eng(d);
    if (!e || !e[key] || (Array.isArray(e[key]) && !e[key].length)) {
      emptyNote(host, "Not run yet. Run ocr_compare.py, then build_site_data.py.");
      return false;
    }
    return true;
  }
  // Long SVG lines wrap on narrow charts instead of running past the edge
  function wrapText(s, x, y, text, maxW, cls = "svg-text", lh = 16, charW = 6.6) {
    const words = String(text).split(" "), lines = [];
    let line = "";
    words.forEach(w => { const t = line ? `${line} ${w}` : w; if (t.length * charW > maxW && line) { lines.push(line); line = w; } else line = t; });
    if (line) lines.push(line);
    lines.forEach((l, i) => el(s, "text", { x, y: y + i * lh, class: cls }, l));
    return lines.length;
  }
  const lineCount = (text, maxW, charW = 6.6) => {
    let n = 1, line = "";
    String(text).split(" ").forEach(w => { const t = line ? `${line} ${w}` : w; if (t.length * charW > maxW && line) { n++; line = w; } else line = t; });
    return n;
  };
  const boldText = (s, x, y, text, cls = "svg-label", extra = {}) => {
    const t = el(s, "text", { x, y, class: cls, ...extra }, text);
    t.style.fontWeight = 600;   // the class sets font; weight must be a style to win
    return t;
  };

  Object.assign(CHARTS, {

    /* 1. Opening tiles */
    readingTiles(host, d) {
      if (!need(host, d, "analyses")) return;
      const e = eng(d), c = e.analyses.C || {}, f = (e.funsd || []).find(r => r.engine === "C") || {};
      const tiles = [
        ["archival pages processed", (c.pages_expected || 0).toLocaleString("en-US"), "First page of every document in the sample"],
        ["text sources compared", String(Object.keys(e.analyses).length), "Tesseract at two resolutions, PaddleOCR, a vision-language model, UCSF's text layer"],
        ["word recall by PaddleOCR on benchmark forms with known text", f.word_recall != null ? pct(f.word_recall) : "—", ""],   // Description dropped, the label says it
        ["pages with Bates numbers recovered by PaddleOCR", c.pages_read ? pct(c.pages_with_own_bates_found / c.pages_read) : "—", ""],
      ];
      const wrap = document.createElement("div");
      wrap.className = "tiles";
      // In the chapter opening's margin each tile's description is its tooltip (still read by screen readers)
      wrap.innerHTML = tiles.map(([k, v, s]) => s ? `<div data-tip="${s}"><strong>${v}</strong><small>${k}</small><span class="tile-sub">${s}</span></div>` : `<div><strong>${v}</strong><small>${k}</small></div>`).join("");   // Number first, its short label under it
      host.append(wrap);
    },

    /* 11b. How many groups? A check done afterwards: inertia and silhouette for k = 2-8,
       with k = 4 (used) and the final five groups marked. Data: ocr.k_check (build_site_data.py). */
    kCheck(host, d) {
      const kc = d.ocr && d.ocr.k_check;
      if (!kc || !kc.tried || !kc.tried.length) {
        emptyNote(host, "This check appears after the next rebuild in the original project (ocr.k_check from outputs/page_groups_k_check.csv).");
        return;
      }
      const W = widthOf(host), side = W >= 560, gap = 28, pw = side ? (W - gap) / 2 : W, ph = 220;
      const s = svg(W, side ? ph : ph * 2 + gap, "Inertia and silhouette for two to eight scan groups, with four groups (used) and the final five groups marked");
      const ks = kc.tried.map(r => r.k), k0 = Math.min(...ks), k1 = Math.max(...ks);
      const used = kc.tried.find(r => r.used), fin = kc.final;
      const panel = (ox, oy, key, title, lo, hi, ticks, fmtv, withFinal) => {
        const L = 40, R = 14, T = 34, B = 34;
        const sx = k => ox + L + (pw - L - R) * (k - k0) / (k1 - k0), sy = v => oy + T + (ph - T - B) * (1 - (v - lo) / (hi - lo));
        boldText(s, ox, oy + 14, title);
        ticks.forEach(v => { el(s, "line", { x1: ox + L, x2: ox + pw - R, y1: sy(v), y2: sy(v), class: "svg-grid" }); el(s, "text", { x: ox + L - 6, y: sy(v) + 4, "text-anchor": "end", class: "svg-text" }, fmtv(v)); });
        ks.forEach(k => el(s, "text", { x: sx(k), y: oy + ph - B + 18, "text-anchor": "middle", class: "svg-text" }, String(k)));
        el(s, "text", { x: ox + pw - R, y: oy + ph - 2, "text-anchor": "end", class: "svg-text" }, "groups (k)");
        if (fin) {
          el(s, "line", { x1: sx(fin.groups), x2: sx(fin.groups), y1: oy + T - 6, y2: oy + ph - B, stroke: "var(--muted)", "stroke-dasharray": "3 3" });
        }
        el(s, "path", { d: kc.tried.map((r, i) => `${i ? "L" : "M"}${sx(r.k)} ${sy(r[key])}`).join(" "), fill: "none", stroke: "var(--second)", "stroke-width": 2.5 });
        kc.tried.forEach(r => tip(el(s, "circle", { cx: sx(r.k), cy: sy(r[key]), r: 4, fill: "var(--second)" }), `k = ${r.k}: ${title.toLowerCase()} ${fmtv(r[key])}; group sizes ${r.sizes.join(" / ")}`));
        if (used) {
          el(s, "circle", { cx: sx(used.k), cy: sy(used[key]), r: 8, fill: "none", stroke: "var(--ink)", "stroke-width": 2 });
          // The label goes where the line isn't. Try above, below, then above/below to either side
          // and keep the first box the curve (and the final-groups marker) doesn't cross.
          const px = sx(used.k), py = sy(used[key]), tw = 86, th = 14;
          const lineY = x => { for (let i = 1; i < kc.tried.length; i++) { const a = kc.tried[i - 1], b = kc.tried[i]; if (x >= sx(a.k) && x <= sx(b.k)) { const t = (x - sx(a.k)) / (sx(b.k) - sx(a.k)); return sy(a[key]) + t * (sy(b[key]) - sy(a[key])); } } return null; };
          const cands = [[px, py - 13, "middle"], [px, py + 24, "middle"], [px - 10, py - 13, "end"], [px - 10, py + 24, "end"], [px + 10, py - 13, "start"], [px + 10, py + 24, "start"]];
          const clear = ([x, y, a]) => { const x0 = a === "middle" ? x - tw / 2 : a === "end" ? x - tw : x, top = y - th + 2, bot = y + 3;
            if (x0 < ox + 2 || x0 + tw > ox + pw) return false;
            for (let xx = x0; xx <= x0 + tw; xx += 4) { const ly = lineY(xx); if (ly != null && ly > top - 4 && ly < bot + 4) return false; }
            if (withFinal && fin) { const fx = sx(fin.groups), fy = sy(fin.silhouette); if (fx + 11 < x0 + tw + 60 && fx > x0 - 10 && Math.abs(fy + 18 - y) < 16) return false; }
            return true; };
          const [lx, ly, la] = cands.find(clear) || cands[0];
          el(s, "text", { x: lx, y: ly, "text-anchor": la, class: "svg-value" }, `k = ${used.k}, used`);
        }
        if (withFinal && fin) {
          const x = sx(fin.groups), y = sy(fin.silhouette);
          tip(el(s, "path", { d: `M${x} ${y - 7}L${x + 7} ${y}L${x} ${y + 7}L${x - 7} ${y}Z`, fill: "var(--surface, #fff)", stroke: "var(--ink)", "stroke-width": 2 }), `Final ${fin.groups} groups: silhouette ${fmtv(fin.silhouette)}; sizes ${fin.sizes.join(" / ")}`);
          el(s, "text", { x: x + 11, y: y + 18, class: "svg-text" }, `final ${fin.groups} groups`);
        } else if (fin) {
          el(s, "text", { x: sx(fin.groups) + 5, y: oy + T - 2, class: "svg-text" }, `${fin.groups} groups`);
        }
      };
      const inert = kc.tried.map(r => r.inertia), maxI = Math.ceil(Math.max(...inert) / 1000) * 1000;
      panel(0, 0, "inertia", "Inertia (lower is tighter)", 0, maxI, [0, maxI / 2, maxI], v => Math.round(v).toLocaleString("en-US"), false);
      const sil = kc.tried.map(r => r.silhouette).concat(fin ? [fin.silhouette] : []);
      const sHi = Math.ceil(Math.max(...sil) * 10) / 10;
      panel(side ? pw + gap : 0, side ? 0 : ph + gap, "silhouette", "Silhouette (higher is better separated)", 0, sHi, [0, sHi / 2, sHi], v => v.toFixed(2), true);
      host.append(s);
    },

    /* 2. 1,400 pages sorted into scan groups (one square per page) */
    pageGroups(host, d) {
      if (!need(host, d, "scan_groups")) return;
      // On narrow charts the group name sits above its squares
      const W = widthOf(host), sq = 6, gap = 2, step = sq + gap, narrow = W < 560;
      const labelW = narrow ? 0 : Math.min(210, Math.round(W * 0.34)), head = narrow ? 36 : 0;
      const cols = Math.max(20, Math.floor((W - labelW) / step));
      const groups = GROUP_ORDER.map(id => eng(d).scan_groups.find(g => g.id === id)).filter(g => g && g.pages);
      let y = 4;
      const rows = groups.map(g => {
        const r = { g, y, h: narrow ? head + Math.ceil(g.pages / cols) * step : Math.max(Math.ceil(g.pages / cols) * step, 34) };
        y += r.h + 16;
        return r;
      });
      const s = svg(W, y, "Every page in the sample, sorted by how its scan looks");
      rows.forEach(({ g, y, h }) => {
        boldText(s, 0, y + 12, g.name);
        el(s, "text", { x: 0, y: y + 29, class: "svg-text" }, `${g.pages.toLocaleString("en-US")} pages · ${pct(g.pages / groups.reduce((a, x) => a + x.pages, 0))}`);
        const cls = HARD.has(g.id) ? "svg-deep" : "svg-dusty";
        for (let i = 0; i < g.pages; i++) {
          el(s, "rect", { x: labelW + (i % cols) * step, y: y + head + Math.floor(i / cols) * step, width: sq, height: sq, class: cls });
        }
        tip(el(s, "rect", { x: labelW, y: y + head, width: W - labelW, height: h - head, fill: "transparent" }),
            `${g.name}: ${g.pages} pages${HARD.has(g.id) ? " (hard to read)" : ""}`);
      });
      host.append(s);
    },

    /* 3. Did the groups hold up? One square per test page */
    groupCheck(host, d) {
      if (!need(host, d, "group_check")) return;
      const SHORTG = { "1": "clean typed", "2": "photo ad", "3a": "degraded scan", "3b": "graphics", "4": "clean report" };
      // One full-width strip per group, cut into one cell per checked page, so each
      // row reads as a share (4 of 6 vs 4 of 4) and every page stays visible. A wrong cell names the page's true
      // group inside it (replaces the separate "Correct groups" line). Narrow charts: name above, strip below.
      const W = widthOf(host), gap = 4, narrow = W < 560;
      const labelW = narrow ? 0 : Math.min(230, Math.round(W * 0.24));
      const cellH = narrow ? 40 : 34, rowH = narrow ? 50 + cellH : 52;
      const rows = GROUP_ORDER.map(id => eng(d).group_check.find(r => String(r.assigned_group) === id)).filter(Boolean);
      const s = svg(W, rows.length * rowH + 2, "Scan group check on the 22 test pages");
      rows.forEach((r, i) => {
        const y = i * rowH + 2;
        const moved = r.moved_to ? String(r.moved_to).split(", ") : [];
        const total = r.Correct + r.Wrong + r.Unsure;
        boldText(s, 0, y + 15, groupName(d, String(r.assigned_group)));
        el(s, "text", narrow ? { x: 0, y: y + 33, class: "svg-text" } : { x: 0, y: y + 32, class: "svg-text" }, `${r.Correct} of ${total} confirmed`);
        const marks = [...Array(r.Correct).fill("Correct"), ...Array(r.Unsure).fill("Unsure"), ...Array(r.Wrong).fill("Wrong")];
        const x0 = labelW + (narrow ? 0 : 12), stripW = W - x0, cy = narrow ? y + 44 : y + 2;
        const cellW = (stripW - gap * (marks.length - 1)) / marks.length;
        let w = 0;
        marks.forEach((m, j) => {
          const x = x0 + j * (cellW + gap);
          const attrs = m === "Correct" ? { class: "svg-deep" } : m === "Unsure" ? { class: "svg-soft" } : { fill: "transparent", class: "svg-wrong" };
          const label = m === "Correct" ? "right group" : m === "Unsure" ? "unsure" : "wrong group";
          const truth = m === "Wrong" && moved[w] ? moved[w++] : null;
          const looked = truth ? ` (correct group: ${groupName(d, truth).toLowerCase()})` : "";
          tip(el(s, "rect", { x: x + .75, y: cy + .75, width: Math.max(0, cellW - 1.5), height: cellH - 1.5, rx: 3, ...attrs }), `Test page: ${label}${looked}`);
          if (truth) {                                  // the true group, inside the cell, in text ink
            const name = SHORTG[truth] || groupName(d, truth).toLowerCase();
            const fits = name.length * 6.9 < cellW - 10;
            const words = fits ? [name] : name.split(" ");
            words.forEach((t, k) => el(s, "text", { x: x + cellW / 2, y: cy + cellH / 2 + 4 + (k - (words.length - 1) / 2) * 14,
              "text-anchor": "middle", class: "svg-text", "aria-hidden": "true" }, t));
          }
        });
      });
      host.append(s);
    },

    /* 4. Ground truth: correct words against speed */
    funsdScatter(host, d) {
      if (!need(host, d, "funsd")) return;
      const rows = eng(d).funsd;
      const W = widthOf(host), H = 320, L = 52, R = 24, T = 18, B = 44;
      const s = svg(W, H, "Word recall against seconds per form on FUNSD");
      const xMin = 0.2, xMax = 10, yMin = 0.6, yMax = 1;
      const x = v => L + (W - L - R) * (Math.log10(v) - Math.log10(xMin)) / (Math.log10(xMax) - Math.log10(xMin));
      const y = v => T + (H - T - B) * (1 - (v - yMin) / (yMax - yMin));
      for (let v = 0.6; v <= 1.0001; v += 0.1) {
        el(s, "line", { x1: L, x2: W - R, y1: y(v), y2: y(v), class: "svg-grid" });
        el(s, "text", { x: L - 8, y: y(v) + 4, "text-anchor": "end", class: "svg-text" }, pct(v));
      }
      [0.25, 0.5, 1, 2, 5, 10].forEach(v => {
        el(s, "line", { x1: x(v), x2: x(v), y1: H - B, y2: H - B + 5, class: "svg-axis" });
        el(s, "text", { x: x(v), y: H - B + 19, "text-anchor": "middle", class: "svg-text" }, `${v}s`);
      });
      el(s, "line", { x1: L, x2: W - R, y1: H - B, y2: H - B, class: "svg-axis" });
      el(s, "text", { x: (L + W - R) / 2, y: H - 4, "text-anchor": "middle", class: "svg-text" }, "Seconds per form (log scale): faster to the left");
      el(s, "text", { x: L, y: 11, class: "svg-text" }, "Word recall ↑");
      const placed = [];                         // Label boxes already drawn, so no two labels overlap
      rows.forEach(r => {
        const cx = x(r.seconds_per_form), cy = y(r.word_recall), lead = r.engine === eng(d).backbone;
        tip(el(s, "circle", { cx, cy, r: lead ? 9 : 7, class: lead ? "svg-deep" : "svg-dusty" }),
            `${SHORT[r.engine]}: word recall ${pct(r.word_recall)}, word error rate ${pct(r.wer)}, ${r.seconds_per_form}s per form`);
        const right = cx < W * 0.62;
        const tx = right ? cx + 14 : cx - 14, anchor = right ? "start" : "end";
        const sub = `recall ${pct(r.word_recall)} · WER ${pct(r.wer)}`, wBox = Math.max(SHORT[r.engine].length, sub.length) * 6.6;
        const x1 = right ? tx : tx - wBox, x2 = right ? tx + wBox : tx;
        let ly = cy - 2;
        while (placed.some(b => x1 < b[1] && b[0] < x2 && ly - 13 < b[3] && b[2] < ly + 18)) ly += 8;
        placed.push([x1, x2, ly - 13, ly + 18]);
        boldText(s, tx, ly, SHORT[r.engine], "svg-label", { "text-anchor": anchor });
        el(s, "text", { x: tx, y: ly + 16, "text-anchor": anchor, class: "svg-text" }, sub);
      });
      host.append(s);
    },

    /* 5. Who wins which kind of page: blind-review rank score, engine x group */
    reviewHeatmap(host, d) {
      if (!need(host, d, "review_by_group")) return;
      const rows = eng(d).review_by_group, ref = new Set(eng(d).reference || []);
      const W = widthOf(host), narrow = W < 560, labelW = narrow ? Math.min(84, Math.round(W * 0.24)) : Math.min(170, Math.round(W * 0.3)), headH = narrow ? 34 : 44, cellH = 40;
      const SHORTG = { "1": "clean typed", "2": "photo ad", "3a": "degraded", "3b": "graphics", "4": "clean report" };   // Narrow headers
      const cellW = (W - labelW) / GROUP_ORDER.length;
      const s = svg(W, headH + ENGINES.length * cellH + 2, "Blind review rank score by engine and scan group");
      GROUP_ORDER.forEach((g, j) => {
        const words = (narrow ? SHORTG[g] || groupName(d, g) : groupName(d, g)).split(" ");
        const half = Math.ceil(words.length / 2);
        [words.slice(0, half).join(" "), words.slice(half).join(" ")].filter(Boolean).forEach((line, k) =>
          el(s, "text", { x: labelW + (j + .5) * cellW, y: 15 + k * 14, "text-anchor": "middle", class: "svg-text" }, line));
      });
      const val = (e, g) => (rows.find(r => r.engine === e && String(r.group) === g) || {}).rank_score;
      const winner = {};
      GROUP_ORDER.forEach(g => {
        const cands = ENGINES.filter(e => !ref.has(e) && val(e, g) != null);
        const best = Math.max(...cands.map(e => val(e, g)));
        winner[g] = new Set(cands.filter(e => val(e, g) === best));
      });
      ENGINES.forEach((e, i) => {
        const y = headH + i * cellH;
        const nameLines = narrow ? SHORT[e].split(", ").map((t, k, a) => k < a.length - 1 ? t + "," : t) : [SHORT[e]];   // Narrow rows on two lines
        nameLines.forEach((t, k) => boldText(s, 0, y + 15 + k * 13, t));
        if (ref.has(e)) el(s, "text", { x: 0, y: y + 15 + nameLines.length * 13 + 2, class: "svg-text" }, "reference (D-51)");
        GROUP_ORDER.forEach((g, j) => {
          const v = val(e, g); if (v == null) return;
          const x = labelW + j * cellW;
          const r = el(s, "rect", { x: x + 2, y: y + 2, width: cellW - 4, height: cellH - 4, rx: 3, class: "svg-deep", opacity: (0.08 + 0.92 * v).toFixed(2) });
          tip(r, `${SHORT[e]} on ${groupName(d, g).toLowerCase()}: rank score ${v.toFixed(2)} (1 = ranked best on every page)`);
          if (winner[g].has(e)) el(s, "rect", { x: x + 2, y: y + 2, width: cellW - 4, height: cellH - 4, rx: 3, fill: "none", class: "svg-win" });
          el(s, "text", { x: x + cellW / 2, y: y + cellH / 2 + 5, "text-anchor": "middle", class: v > 0.55 ? "svg-on" : "svg-value" }, v.toFixed(2));
        });
      });
      host.append(s);
    },

    /* 6. The same pattern on all 1,400 pages: readability dot strips */
    readabilityStrips(host, d) {
      if (!need(host, d, "analyses")) return;
      const a = eng(d).analyses;
      const W = widthOf(host), labelW = Math.min(200, Math.round(W * 0.32)), R = 18, r = 10, laneH = 22, top = 26;
      const x = v => labelW + (W - labelW - R) * v;
      let y = top;
      const rows = GROUP_ORDER.map(g => {
        const pts = ENGINES.filter(e => a[e] && a[e].by_group && a[e].by_group[g])
          .map(e => ({ e, v: a[e].by_group[g].wordlike_share_median })).sort((p, q) => p.v - q.v);
        const laneEnd = [];
        pts.forEach(p => {
          let lane = 0;
          while (laneEnd[lane] != null && x(p.v) - laneEnd[lane] < 2 * r + 2) lane++;
          p.lane = lane; laneEnd[lane] = x(p.v);
        });
        const lanes = Math.max(1, ...pts.map(p => p.lane + 1));
        const row = { g, y, pts, h: Math.max(38, lanes * laneH + 10) };
        y += row.h + 8;
        return row;
      });
      const s = svg(W, y, "Median word-like share by scan group, all 1,400 pages");
      [0, 0.25, 0.5, 0.75, 1].forEach(v => {
        el(s, "line", { x1: x(v), x2: x(v), y1: top - 6, y2: y - 8, class: "svg-grid" });
        el(s, "text", { x: x(v), y: 12, "text-anchor": "middle", class: "svg-text" }, pct(v));
      });
      rows.forEach(({ g, y, pts }) => {
        boldText(s, 0, y + 16, groupName(d, g));
        pts.forEach(p => {
          const cy = y + 12 + p.lane * laneH, lead = p.e === "C" || p.e === "D";
          tip(el(s, "circle", { cx: x(p.v), cy, r, class: lead ? "svg-deep" : "svg-soft" }),
              `${SHORT[p.e]}: median word-like share ${pct(p.v)}`);
          el(s, "text", { x: x(p.v), y: cy + 4, "text-anchor": "middle", class: lead ? "svg-on svg-small" : "svg-value svg-small" }, p.e);
        });
      });
      host.append(s);
    },

    /* 7. File number read: 50 squares per engine, each = 2% of pages */
    batesUnits(host, d) {
      if (!need(host, d, "analyses")) return;
      const a = eng(d).analyses;
      const W = widthOf(host), narrow = W < 560;
      // The squares fill the space between the label and value columns; the values sit right-aligned
      // at the figure's right edge
      const labelW = narrow ? 0 : Math.min(170, Math.round(W * 0.3)), valueW = narrow ? 92 : 150, n = 50, perRow = narrow ? 25 : 50;
      const step = Math.max(5, (W - labelW - valueW - 12) / perRow), sq = Math.max(3, step - 2);
      const rowH = narrow ? 22 + 2 * step + 10 : Math.max(30, step + 14);
      const s = svg(W, ENGINES.length * rowH + 4, "Share of pages where each engine read the record's own Bates number");
      ENGINES.forEach((e, i) => {
        const r = a[e]; if (!r) return;
        const y = i * rowH + 4, rate = r.pages_with_own_bates_found / r.pages_read, filled = Math.round(rate * n);
        const top = narrow ? y + 22 : y + 5;
        boldText(s, 0, y + 15, SHORT[e]);
        for (let k = 0; k < n; k++) {
          el(s, "rect", { x: labelW + (k % perRow) * step, y: top + Math.floor(k / perRow) * step, width: sq, height: sq, class: k < filled ? (e === "C" ? "svg-deep" : "svg-dusty") : "svg-soft", opacity: k < filled ? 1 : .5 });
        }
        tip(el(s, "rect", { x: labelW, y: top - 2, width: perRow * step, height: (n / perRow) * step + 2, fill: "transparent" }),
            `${SHORT[e]}: ${r.pages_with_own_bates_found.toLocaleString("en-US")} of ${r.pages_read.toLocaleString("en-US")} pages`);
        if (narrow) {
          el(s, "text", { x: W, y: top + 10, "text-anchor": "end", class: "svg-value" }, `${pct(rate)}`);
          el(s, "text", { x: W, y: top + 26, "text-anchor": "end", class: "svg-text" }, `${r.pages_with_own_bates_found.toLocaleString("en-US")} pages`);
        } else {
          const vy = top + sq / 2 + 4;
          el(s, "text", { x: W, y: vy, "text-anchor": "end", class: "svg-text" }, `${r.pages_with_own_bates_found.toLocaleString("en-US")} pages`);
          el(s, "text", { x: W - 84, y: vy, "text-anchor": "end", class: "svg-value" }, `${pct(rate)}`);
        }
      });
      host.append(s);
    },

    /* 8. Orientation: what the model flagged, and how often it was right */
    orientationFlow(host, d) {
      const o = eng(d) && eng(d).orientation;
      if (!o) return emptyNote(host, "Not run yet.");
      const W = widthOf(host), sq = 12, step = 15;
      const cols = Math.max(10, Math.floor((W - 10) / step));
      const t1 = `Of ${o.pages.toLocaleString("en-US")} pages, PaddleOCR's page-orientation model flagged ${o.turn_suggested} for rotation.`;
      const t2 = "One square per flagged page. Each was read as scanned and as rotated; the more confident output was kept.";
      const n1 = lineCount(t1, W, 7), n2 = lineCount(t2, W), head = 14 + (n1 - 1) * 17 + 18 + (n2 - 1) * 16 + 12, shift = head - 44;
      const rowsN = Math.ceil(o.turn_suggested / cols), H = 46 + shift + rowsN * step + 34;
      const s = svg(W, H, "Pages the orientation model wanted to turn, and how many turns were right");
      wrapText(s, 0, 14, t1, W, "svg-label", 17, 7);
      wrapText(s, 0, 14 + (n1 - 1) * 17 + 18, t2, W);
      const g = el(s, "g", { transform: `translate(0 ${shift})` });
      for (let i = 0; i < o.turn_suggested; i++) {
        const right = i < o.turn_kept;
        tip(el(g, "rect", { x: (i % cols) * step, y: 44 + Math.floor(i / cols) * step, width: sq, height: sq, rx: 2,
          ...(right ? { class: "svg-deep" } : { fill: "transparent", class: "svg-wrong" }) }), right ? "Rotated output was more confident" : "Already upright: rotating would have been wrong");
      }
      const ly = 44 + rowsN * step + 22;
      el(g, "rect", { x: 0, y: ly - 10, width: sq, height: sq, rx: 2, class: "svg-deep" });
      el(g, "text", { x: 18, y: ly, class: "svg-text" }, `${o.turn_kept} rotations confirmed`);
      el(g, "rect", { x: 170, y: ly - 10, width: sq, height: sq, rx: 2, fill: "none", class: "svg-wrong" });
      el(g, "text", { x: 188, y: ly, class: "svg-text" }, `${o.turn_suggested - o.turn_kept} pages already upright`);
      host.append(s);
    },
  });
})();
