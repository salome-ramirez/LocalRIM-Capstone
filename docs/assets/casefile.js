/* The case file: the record's six stages with a stage switcher. Styles: assets/casefile.css.
   Every number comes from data/site_data.json. Charts are plain SVG or HTML, drawn
   when their stage is first shown and redrawn when their width changes. */
(() => {
  "use strict";
  const $ = (s, r = document) => r.querySelector(s), $$ = (s, r = document) => [...r.querySelectorAll(s)];
  const get = (o, p) => p.split(".").reduce((x, k) => (x == null ? x : x[k]), o);
  const NS = "http://www.w3.org/2000/svg";
  const E = (p, tag, attrs = {}, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    if (p) p.append(e);
    return e;
  };
  const H = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const pct = v => `${Math.round(v * 100)}%`;
  const int = v => Math.round(v).toLocaleString("en-US");
  const FMT = { pct, int, num2: v => v.toFixed(2), s: v => `${v.toFixed(2)} s` };
  const short = n => n.replace(/\s*\(.*\)\s*$/, "");
  const mq = matchMedia("(prefers-reduced-motion: reduce)");
  let forceReduce = false;
  const reduced = () => mq.matches || forceReduce;
  let D;

  const STAGES = ["intake", "capture", "preparation", "classification", "routing", "validation"];
  const NAMES = { intake: "Intake", capture: "Capture", preparation: "Preparation", classification: "Classification", routing: "Routing", validation: "Validation" };
  // Old stage names in links (#received, #read, ...) still open the right stage
  const OLD = { received: "intake", read: "capture", cleaned: "preparation", classified: "classification", routed: "routing", checked: "validation" };

  /* Named calculations (all from site_data.json) */
  const eng = () => D.ocr.engines;
  const WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve"];
  const word = n => WORDS[n] || int(n);
  const backbone = () => eng().backbone;
  const CALC = {
    "read.backbone": () => short(eng().names[backbone()]),
    "read.vision": () => short(eng().names.D),
    "speed.pages": () => word(Math.round(eng().analyses.D.seconds_mean / eng().analyses[backbone()].seconds_mean)),
    "speed.funsd": () => word(Math.round(eng().funsd.find(f => f.engine === "D").seconds_per_form / eng().funsd.find(f => f.engine === backbone()).seconds_per_form)),
    "read.recall": () => eng().funsd.find(f => f.engine === backbone()).word_recall,
    "read.sec": () => eng().analyses[backbone()].seconds_mean,
    "read.review": () => { const r = eng().review_overall.find(x => x.engine === backbone()); return `${int(r.good + r.usable)} of ${int(r.pages)}`; },
    "read.hard": () => eng().scan_groups.filter(g => g.id !== "1" && g.id !== "4").reduce((a, g) => a + g.pages, 0),
    "ocr.checked": () => eng().group_check.reduce((a, r) => a + r.Correct + r.Wrong + r.Unsure, 0),
    "final.ragVs": () => `${pct(D.final_test.rag.rag_top1)} vs ${pct(D.final_test.rag.classifier_top1)}`,
    "scale.auto": () => `${int(D.scale_run.auto_filed)} of ${int(D.scale_run.pages)}`,
    "sheet.read": () => `Text read by ${short(eng().names[backbone()])}; the hardest pages re-read by ${short(eng().names.D)}`,
    "sheet.routed": () => `Filed if confidence is ${D.final_test.cutoff.toFixed(2)} or more; otherwise to a person, with a suggestion`,
    "sheet.checked": () => `Pipeline scored once, on ${int(D.final_test.pages)} unseen pages`,
    "rcv.archive": () => `${(D.sample.funnel[0].count / 1e6).toFixed(2)} million`,
    "cls.depts": () => D.classifier.confusion.labels.length,
    "cls.total": () => D.classifier.confusion.counts.flat().reduce((a, b) => a + b, 0),
    "cls.topPair": () => { const p = pairs()[0]; return `${DEPT[p.a] || p.a} with ${DEPT[p.b] || p.b}`; },
    "rag.practiceVs": () => { const a = D.rag.approaches; return `${pct(a[a.length - 1].top1)} vs ${pct(a.find(x => x.name === "Classifier").top1)}`; },
    "scale.seconds": () => `${D.scale_run.timing.reduce((a, t) => a + t.seconds, 0).toFixed(1)} s`,
    "scale.note": () => {
      const r = D.scale_run, sc = r.spotcheck, w = r.projected_wrong;
      return `The misfilings are a projection from the final test, not a count: applying its accuracy of automatic filings, with its 95% interval, to the ${int(r.auto_filed)} filed pages gives between ${int(w.low)} and ${int(w.high)} misfiled. A spot check by hand agreed with ${int(sc.agree)} of ${int(sc.checked - sc.excluded)} filed pages. Source: LocalRIM scale run. Full figures: <a href="review.html">Filing and review</a>.`;
    },
    "final.ragNote": () => {
      const f = D.final_test.rag;
      return `On the final test the suggestion was right where the classifier was wrong on ${int(f.only_rag)} pages, and the reverse on ${int(f.only_classifier)} (paired test, p = ${f.p.toFixed(3)}). Source: LocalRIM evaluation of the suggestion, made by ${D.rag.model}. Full figures: <a href="review.html">Filing and review</a>.`;
    },
  };
  // A date field shows the date only, never the build time
  const fmtVal = (v, f) => (f === "date" && typeof v === "string" ? v.replace(/\s+\d{1,2}:\d{2}$/, "") : typeof v === "number" && f && FMT[f] ? FMT[f](v) : typeof v === "number" ? int(v) : v);

  function bind() {
    $$("[data-field]").forEach(el => { const v = get(D, el.dataset.field); if (v != null) el.textContent = fmtVal(v, el.dataset.fmt); });
    $$("[data-calc]").forEach(el => {
      const fn = CALC[el.dataset.calc]; if (!fn) return;
      const v = fn();
      if (el.dataset.calc === "final.ragNote" || el.dataset.calc === "scale.note") el.innerHTML = v; else el.textContent = fmtVal(v, el.dataset.fmt);
    });
    $$("[data-done-calc]").forEach(el => { el.dataset.done = CALC[el.dataset.doneCalc](); });
  }

  /* Decisions on file */
  function decisions() {
    $$("[data-decisions]").forEach(host => {
      host.dataset.decisions.split(",").forEach(id => {
        const d = D.decisions.find(x => x.id === id.trim());
        if (!d) return;
        const det = H("details"), sum = H("summary");
        sum.append(H("span", "did", d.id), H("span", "text", d.decision), H("span", "date", d.date));
        det.append(sum, H("p", null, d.why));
        host.append(det);
      });
      if (host.classList.contains("mdec")) return;   /* Margin lists stay compact; the full log is one click away in the masthead */
      const more = H("p", "sub");
      more.innerHTML = `${int(D.decisions.length)} decisions are logged across the project. <a href="dashboard.html#log">See the full log</a>.`;
      host.append(more);
    });
  }

  /* Charts */
  const widthOf = host => Math.floor(host.clientWidth);
  const hatchFill = "url(#hatch-legend)";
  const CHARTS = {
    /* Scan groups: pages per group, with the eye check beside each */
    groups(host) {
      const W = widthOf(host), narrow = W < 560;
      const groups = [...eng().scan_groups].sort((a, b) => b.pages - a.pages);
      const max = groups[0].pages, sq = 13, gap = 3;
      const checkW = 6 * (sq + gap), labelW = narrow ? 0 : 205, rowH = narrow ? 56 : 34;
      const barX = labelW, barW = W - labelW - checkW - 56;
      const s = E(null, "svg", { width: W, height: groups.length * rowH + 22, role: "img", "aria-label": "Pages per scan group, with the eye check for each group" });
      E(s, "text", { x: W - checkW, y: 12, class: "lbl muted" }, "Eye check");
      groups.forEach((g, i) => {
        const y0 = 22 + i * rowH, yb = narrow ? y0 + 24 : y0 + 8;
        E(s, "line", { x1: 0, x2: W, y1: y0, y2: y0, stroke: "var(--grid)" });
        E(s, "text", { x: 0, y: narrow ? y0 + 16 : y0 + 21, class: "lbl" }, g.name);
        const w = Math.max(2, barW * g.pages / max);
        E(s, "rect", { x: barX, y: yb, width: w, height: 18, fill: "var(--filed)", "data-tip": `${g.name}: ${int(g.pages)} pages` });
        E(s, "text", { x: barX + w + 6, y: yb + 14, class: "lbl val" }, int(g.pages));
        const c = eng().group_check.find(r => r.assigned_group === g.id);
        if (!c) return;
        const cells = [...Array(c.Correct).fill("c"), ...Array(c.Unsure).fill("u"), ...Array(c.Wrong).fill("w")];
        cells.forEach((k, j) => {
          const x = W - checkW + j * (sq + gap);
          const ck = { c: "belongs in this group", u: "hard to say", w: "belonged in another group" }[k];
          if (k === "w") E(s, "rect", { x: x + 1, y: yb + 3, width: sq - 2, height: sq - 2, rx: 2, fill: "none", stroke: "var(--wrong)", "stroke-width": 2, "data-tip": `${g.name}, page checked by eye: ${ck}` });
          else E(s, "rect", { x: x + .5, y: yb + 2.5, width: sq - 1, height: sq - 1, rx: 2, fill: k === "c" ? "var(--filed)" : hatchFill, stroke: k === "c" ? null : "var(--person-line)", "data-tip": `${g.name}, page checked by eye: ${ck}` });
        });
      });
      host.replaceChildren(s);
    },

    /* Five engines across five tests */
    engines(host) {
      const e = eng(), ids = Object.keys(e.names), A = e.analyses;
      const review = id => e.review_overall.find(r => r.engine === id);
      const funsd = id => e.funsd.find(f => f.engine === id);
      const maxSec = Math.ceil(Math.max(...ids.map(id => A[id].seconds_mean || 0)));
      const forms = e.funsd[0].forms;
      const tests = [
        { name: "Answer key", sub: `Words recovered on ${int(forms)} forms with known text`, v: id => funsd(id) && funsd(id).word_recall, label: pct, na: "Not run on the answer key", axis: ["0", "100%"] },
        { name: "Blind review", sub: `Pages rated good or usable, of ${int(review(ids[0]).pages)}`, v: id => { const r = review(id); return (r.good + r.usable) / r.pages; }, label: (v, id) => { const r = review(id); return `${int(r.good + r.usable)} of ${int(r.pages)}`; }, axis: ["0", "All"] },
        { name: "All pages", sub: `Pages read as mostly words, of ${int(A[ids[0]].pages_expected)}`, v: id => 1 - A[id].pages_mostly_junk / A[id].pages_expected, label: pct, axis: ["0", "100%"] },
        { name: "Own Bates number", sub: "Pages where the record's own number was found", v: id => A[id].pages_with_own_bates_found / A[id].pages_expected, label: pct, axis: ["0", "100%"] },
        { name: "Speed", sub: "Seconds per page on one PC; fewer is better", v: id => A[id].seconds_mean, pos: v => 1 - v / maxSec, label: v => `${v.toFixed(1)} s`, na: "Comes with the archive", axis: [`${maxSec} s`, "0 s"] },
      ];
      const wrap = H("div", "tests");
      tests.forEach(t => {
        const strip = H("div", "strip"), sh = H("div", "sh"), bar = H("div", "sbar");
        sh.append(H("b", null, t.name), H("span", null, t.sub));
        const vals = ids.map(id => [id, t.v(id)]);
        vals.filter(([, v]) => v != null).forEach(([id, v]) => {
          const p = t.pos ? t.pos(v) : v;
          const dot = H("span", "dot" + (e.reference.includes(id) ? " ref" : ""));
          dot.style.left = `${p * 100}%`; dot.dataset.e = id; dot.dataset.p = p; dot.dataset.label = t.label(v, id);
          dot.dataset.tip = `${short(e.names[id])}, ${t.name.toLowerCase()}: ${t.label(v, id)}`;
          bar.append(dot);
        });
        const na = H("span", "na"); na.dataset.text = t.na || ""; bar.append(na);
        const tag = H("span", "tag"); bar.append(tag);
        const ends = H("span", "ends"); ends.append(H("span", null, t.axis[0]), H("span", null, t.axis[1])); bar.append(ends);
        strip.append(sh, bar); wrap.append(strip);
      });
      host.replaceChildren(wrap);
      highlight(state.engine);
    },

    /* Blind review by engine and scan group */
    heat(host) {
      const e = eng(), ids = Object.keys(e.names);
      const order = ["1", "4", "3b", "3a", "2"].filter(g => e.scan_groups.some(s => s.id === g));
      const t = H("table", "heat");
      const cap = H("caption", "vh", "Blind review: pages rated good or usable, by engine (rows) and scan group (columns)");
      const head = H("tr"); head.append(H("td", "corner"));
      order.forEach(g => { const th = H("th", null, e.scan_groups.find(s => s.id === g).name); th.scope = "col"; head.append(th); });
      const thead = H("thead"); thead.append(head);
      const body = H("tbody");
      ids.forEach(id => {
        const tr = H("tr", e.reference.includes(id) ? "ref" : null);
        const th = H("th", null, short(e.names[id])); th.scope = "row"; tr.append(th);
        order.forEach(g => {
          const r = e.review_by_group.find(x => x.engine === id && x.group === g);
          const ok = r.good + r.usable, share = ok / r.pages;
          const td = H("td", null, `${int(ok)} of ${int(r.pages)}`);
          td.dataset.tip = `${short(e.names[id])} on ${e.scan_groups.find(x => x.id === g).name.toLowerCase()}: ${int(ok)} of ${int(r.pages)} pages good or usable`;
          if (share >= .5) td.className = "hi";
          if (share >= 1) { td.className = "top"; td.style.background = "var(--filed)"; }
          else td.style.setProperty("--s", share.toFixed(3));
          tr.append(td);
        });
        body.append(tr);
      });
      t.append(cap, thead, body);
      host.replaceChildren(t);
    },

    /* Cutoff trade-off with a slider */
    cutoff(host) {
      const pts = curve();
      const W = widthOf(host), narrow = W < 560;
      const padL = 44, padR = 18, padT = 26, padB = 50, Hh = narrow ? 250 : 290;
      const t0 = pts[0].threshold, t1 = pts[pts.length - 1].threshold;
      const sx = t => padL + (W - padL - padR) * (t - t0) / (t1 - t0), sy = v => padT + (Hh - padT - padB) * (1 - v);
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Share of pages filed automatically and their accuracy at each confidence cutoff" });
      [0, .25, .5, .75, 1].forEach(v => {
        E(s, "line", { x1: padL, x2: W - padR, y1: sy(v), y2: sy(v), stroke: "var(--grid)" });
        E(s, "text", { x: padL - 8, y: sy(v) + 4, "text-anchor": "end", class: "lbl muted" }, pct(v));
      });
      pts.forEach((p, i) => { if (!narrow || i % 2 === 0) E(s, "text", { x: sx(p.threshold), y: Hh - padB + 18, "text-anchor": "middle", class: "lbl muted" }, p.threshold.toFixed(2)); });
      E(s, "text", { x: padL, y: Hh - 8, class: "lbl muted" }, narrow ? "Confidence cutoff" : "Confidence cutoff: higher means only surer pages are filed");
      // frozen cutoff
      const fz = D.final_test.cutoff;
      E(s, "line", { x1: sx(fz), x2: sx(fz), y1: padT - 8, y2: Hh - padB, stroke: "var(--muted)", "stroke-dasharray": "2 3" });
      E(s, "text", { x: sx(fz) + (sx(fz) > W * .6 ? -6 : 6), y: padT - 12, "text-anchor": sx(fz) > W * .6 ? "end" : "start", class: "lbl muted" }, "Frozen for the final test");
      // standard
      const tg = D.classifier.routing.target;
      E(s, "line", { x1: padL, x2: W - padR, y1: sy(tg), y2: sy(tg), stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "5 4" });
      const lx = sx(pts[Math.min(2, pts.length - 1)].threshold);
      E(s, "text", { x: lx + 4, y: sy(tg) - 6, class: "lbl muted" }, `${pct(tg)} standard`);
      const path = (k, color) => E(s, "path", { d: pts.map((p, i) => `${i ? "L" : "M"}${sx(p.threshold)} ${sy(p[k])}`).join(" "), fill: "none", stroke: color, "stroke-width": 2.5, "stroke-linejoin": "round" });
      path("accuracy_auto_filed", "var(--second)");
      path("coverage", "var(--filed)");
      const cur = E(s, "g");
      host.replaceChildren(s);
      host._draw = idx => {
        cur.replaceChildren();
        const p = pts[idx], x = sx(p.threshold);
        E(cur, "line", { x1: x, x2: x, y1: padT, y2: Hh - padB, stroke: "var(--ink)", "stroke-width": 1 });
        [["accuracy_auto_filed", "var(--second)"], ["coverage", "var(--filed)"]].forEach(([k, c]) => {
          const y = sy(p[k]);
          E(cur, "circle", { cx: x, cy: y, r: 5.5, fill: c, stroke: "var(--paper)", "stroke-width": 2 });
          const cov = k === "coverage", right = cov ? x > padL + 50 : x < W - 70;
          E(cur, "text", { x: cov ? x + (right ? -9 : 9) : x + (right ? 9 : -9), y: y + (cov ? 18 : -9), "text-anchor": cov ? (right ? "end" : "start") : (right ? "start" : "end"), class: "lbl val", fill: c }, pct(p[k]));
        });
      };
      const input = $("#cut"), box = $("#cut-slider");
      input.max = String(pts.length - 1);
      input.style.marginLeft = `${padL - 8}px`;
      input.style.width = `${W - padL - padR + 16}px`;
      if (state.cut == null) state.cut = Math.max(0, pts.findIndex(p => Math.abs(p.threshold - fz) < 1e-9));
      state.cut = Math.min(state.cut, pts.length - 1);
      input.value = String(state.cut);
      box.hidden = false;
      updateCut();
    },

    /* Final test: one square per page; every block uses the same square size and as many
       squares per row as fit, so each block spans the figure width */
    units(host) {
      const f = D.final_test, u = f.units, W = widthOf(host), gap = 3;
      const filed = u.filter(x => x[0]).sort((a, b) => b[1] - a[1]);
      const person = u.filter(x => !x[0]).sort((a, b) => b[1] - a[1]);
      const cols = Math.max(10, Math.round((W + gap) / (16 + gap)));
      const sq = Math.max(12, (W - gap * (cols - 1)) / cols), step = sq + gap;
      const block = (title, note, items, draw) => {
        const b = H("div", "ublock");
        b.append(H("h4", null, title), H("span", "uh", note));
        const rows = Math.ceil(items.length / cols), used = Math.min(cols, items.length);
        const s = E(null, "svg", { width: used * step - gap, height: rows * step - gap, "aria-hidden": "true" });
        items.forEach((it, i) => draw(s, (i % cols) * step, Math.floor(i / cols) * step, it));
        b.append(s);
        return b;
      };
      const wrap = H("div", "units stacked");
      const wrong = filed.filter(x => !x[1]).length, pWrong = person.filter(x => !x[1]).length;
      wrap.append(
        block(`Filed automatically: ${int(filed.length)}`, `${int(filed.length - wrong)} in the right department, ${int(wrong)} in the wrong one`, filed, (s, x, y, it) => {
          if (it[1]) E(s, "rect", { x, y, width: sq, height: sq, rx: 2, fill: "var(--filed)", "data-tip": "Filed automatically, right department" });
          else E(s, "rect", { x: x + 1, y: y + 1, width: sq - 2, height: sq - 2, rx: 2, fill: "none", stroke: "var(--wrong)", "stroke-width": 2, "data-tip": "Filed automatically, wrong department" });
        }),
        block(`Sent to a person: ${int(person.length)}`, `The classifier's first choice was wrong on ${int(pWrong)} of these; filing them automatically would have misfiled them`, person, (s, x, y, it) => {
          E(s, "rect", { x: x + .5, y: y + .5, width: sq - 1, height: sq - 1, rx: 2, fill: hatchFill, stroke: "var(--person-line)", "data-tip": it[1] ? "Sent to a person; the classifier's first choice was right" : "Sent to a person; the classifier's first choice was wrong" });
          if (!it[1]) E(s, "circle", { cx: x + sq / 2, cy: y + sq / 2, r: sq * .2, fill: "var(--wrong)" });
        }),
        block(`Set aside: ${int(f.excluded.pages)}`, `Not usable as test pages; all ${int(f.excluded.to_a_person)} also go to a person`, Array(f.excluded.pages).fill(0), (s, x, y) => {
          E(s, "rect", { x: x + .75, y: y + .75, width: sq - 1.5, height: sq - 1.5, rx: 2, fill: "none", stroke: "var(--muted)", "stroke-dasharray": "2 2", "stroke-width": 1.5, "data-tip": "Set aside: not usable as a test page; sent to a person" });
        }),
      );
      const sr = H("p", "vh", `${int(filed.length)} filed automatically (${int(filed.length - wrong)} right, ${int(wrong)} wrong); ${int(person.length)} sent to a person, of which the classifier's first choice was wrong on ${int(pWrong)}; ${int(f.excluded.pages)} set aside.`);
      host.replaceChildren(wrap, sr);
    },

    /* RQ3: baseline text vs chosen text, both models */
    slope(host) {
      const c = D.classifier, e = eng(), W = widthOf(host), narrow = W < 560;
      const lines = [
        { name: "Sentence embeddings", a: c.embeddings.A, b: c.embeddings.hybrid, color: "var(--ink)", lead: true },
        { name: "TF-IDF", a: c.tfidf.A, b: c.tfidf.hybrid, color: "var(--second)" },
      ];
      const Hh = 260, padT = 30, padB = 46, x1 = narrow ? 44 : 150, x2 = W - (narrow ? 44 : 150);
      const lo = .4, hi = .75, sy = v => padT + (Hh - padT - padB) * (1 - (v - lo) / (hi - lo));
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Macro F1 with the baseline text and with the chosen text, for both models" });
      [.4, .5, .6, .7].forEach(v => { E(s, "line", { x1: 0, x2: W, y1: sy(v), y2: sy(v), stroke: "var(--grid)" }); E(s, "text", { x: 0, y: sy(v) - 4, class: "lbl muted" }, v.toFixed(1)); });
      [[x1, `${short(e.names.A)}`, "baseline text"], [x2, `${short(e.names[backbone()])} + ${short(e.names.D)}`, "chosen text"]].forEach(([x, a, b]) => {
        E(s, "line", { x1: x, x2: x, y1: padT - 6, y2: Hh - padB, stroke: "var(--rule)" });
        const anchor = narrow ? (x === x1 ? "start" : "end") : "middle", tx = narrow ? (x === x1 ? 0 : W) : x;
        E(s, "text", { x: tx, y: Hh - padB + 18, "text-anchor": anchor, class: "lbl strong" }, a);
        E(s, "text", { x: tx, y: Hh - padB + 33, "text-anchor": anchor, class: "lbl muted" }, b);
      });
      lines.forEach(l => {
        E(s, "line", { x1, x2, y1: sy(l.a.macro_f1), y2: sy(l.b.macro_f1), stroke: l.color, "stroke-width": l.lead ? 3 : 2.5 });
        [[x1, l.a, "baseline text"], [x2, l.b, "chosen text"]].forEach(([x, v, w]) => E(s, "circle", { cx: x, cy: sy(v.macro_f1), r: 5.5, fill: l.color, "data-tip": `${l.name}, ${w}: macro F1 ${v.macro_f1.toFixed(2)}` }));
        E(s, "text", { x: x1 - 10, y: sy(l.a.macro_f1) + 4, "text-anchor": "end", class: "lbl val" }, l.a.macro_f1.toFixed(2));
        E(s, "text", { x: x2 + 10, y: sy(l.b.macro_f1) + 4, class: "lbl val" }, `${l.b.macro_f1.toFixed(2)}`);
        const mx = (x1 + x2) / 2, my = (sy(l.a.macro_f1) + sy(l.b.macro_f1)) / 2;
        E(s, "text", { x: mx, y: my - 10, "text-anchor": "middle", class: l.lead ? "lbl strong" : "lbl" }, `${l.name}, +${Math.round((l.b.macro_f1 - l.a.macro_f1) * 100)} pts`);
      });
      host.replaceChildren(s);
    },

    /* Model ladder with fold spread */
    mladder(host) {
      const c = D.classifier, conf = c.confusion, sizes = conf.counts.map(r => r.reduce((a, b) => a + b, 0));
      const largest = DEPT[conf.labels[sizes.indexOf(Math.max(...sizes))]];
      foldLadder(host, [
        { name: "Always guess the largest department", sub: largest, v: c.tfidf.majority, kind: "ref" },
        { name: "File stamp alone", sub: "a shortcut, kept out of the model", v: c.bates_probe, kind: "ref" },
        { name: "TF-IDF", sub: "weighted word counts with logistic regression", v: c.tfidf.hybrid, kind: "mid" },
        { name: "Sentence embeddings", sub: "Qwen3-Embedding-0.6B with logistic regression", v: c.embeddings.hybrid, kind: "lead" },
      ], "Macro F1 per fold and mean for each model");
    },

    /* Preparation: what the stamp gives away, and what hiding it cost */
    stamp(host) {
      const c = D.classifier, conf = c.confusion, sizes = conf.counts.map(r => r.reduce((a, b) => a + b, 0));
      foldLadder(host, [
        { name: "Always guess the largest department", sub: DEPT[conf.labels[sizes.indexOf(Math.max(...sizes))]], v: c.tfidf.majority, kind: "ref" },
        { name: "File stamp alone", sub: "a probe given only the stamp", v: c.bates_probe, kind: "mid" },
        { name: "Embeddings, stamp kept", sub: "the text with its file number", v: c.embeddings.hybrid_with_bates, kind: "ref" },
        { name: "Embeddings, stamp removed", sub: "the model as used", v: c.embeddings.hybrid, kind: "lead" },
      ], "Macro F1 per fold and mean: floor, stamp alone, and the model with and without the stamp");
    },

    /* Preparation: one square per page the orientation model flagged */
    orient(host) {
      // squares sized so the grid spans the full figure width (about 18 px each, never under 12)
      const o = eng().orientation, gap = 3, W = widthOf(host);
      const cols = Math.max(10, Math.round((W + gap) / (18 + gap)));
      const sq = Math.max(12, (W - gap * (cols - 1)) / cols), step = sq + gap;
      const items = [...Array(o.turn_kept).fill(1), ...Array(o.turn_suggested - o.turn_kept).fill(0)];
      const rows = Math.ceil(items.length / cols);
      const s = E(null, "svg", { width: cols * step - gap, height: rows * step - gap, role: "img", "aria-label": `${o.turn_suggested} pages flagged for turning: ${o.turn_kept} turned, ${o.turn_suggested - o.turn_kept} left as scanned` });
      items.forEach((k, i) => {
        const x = (i % cols) * step, y = Math.floor(i / cols) * step;
        if (k) E(s, "rect", { x, y, width: sq, height: sq, rx: 2, fill: "var(--second)", "data-tip": "Turned: the turned reading was more confident" });
        else E(s, "rect", { x: x + .75, y: y + .75, width: sq - 1.5, height: sq - 1.5, rx: 2, fill: "none", stroke: "var(--ref)", "stroke-width": 1.5, "data-tip": "Left as scanned: the reading as scanned was more confident" });
      });
      host.replaceChildren(s);
    },

    /* Intake: the sample cascade on a log scale */
    cascade(host) {
      const f = D.sample.funnel, W = widthOf(host), narrow = W < 560;
      const labelW = narrow ? 0 : 190, padR = 90, rowH = narrow ? 50 : 34;
      const max = Math.log10(f[0].count), sx = v => (W - labelW - padR) * Math.log10(Math.max(1, v)) / max;
      const Hh = f.length * rowH + 22;
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Records at each step of the sample, from the whole collection to the labeled pages, log scale" });
      [1, 10, 100, 1e3, 1e4, 1e5, 1e6].forEach(v => {
        const x = labelW + sx(v);
        E(s, "line", { x1: x, x2: x, y1: 0, y2: f.length * rowH, stroke: "var(--grid)" });
        if (!narrow || [1, 100, 1e4, 1e6].includes(v)) E(s, "text", { x, y: Hh - 4, "text-anchor": "middle", class: "lbl muted" }, v >= 1e6 ? "1M" : v >= 1e3 ? `${v / 1e3}k` : String(v));
      });
      f.forEach((st, i) => {
        const y0 = i * rowH, yb = narrow ? y0 + 22 : y0 + 7;
        E(s, "text", { x: 0, y: narrow ? y0 + 15 : y0 + 22, class: i >= 3 ? "lbl strong" : "lbl" }, st.label);
        const w = Math.max(2, sx(st.count));
        E(s, "rect", { x: labelW, y: yb, width: w, height: 20, fill: i >= 3 ? "var(--ink)" : "var(--other)", "data-tip": `${st.label}: ${int(st.count)}` });
        E(s, "text", { x: labelW + w + 8, y: yb + 15, class: "lbl val" }, int(st.count));
      });
      host.replaceChildren(s);
    },

    /* Intake: share of labels marked unsure in each round */
    rounds(host) {
      const sk = D.taxonomy.skims, W = widthOf(host), narrow = W < 560;
      const pts = [
        { a: sk[0].label, b: `pilot, ${int(sk[0].pages)} pages`, v: sk[0].unsure },
        { a: sk[1].label, b: `pilot, ${int(sk[1].pages)} pages`, v: sk[1].unsure, key: true },
        { a: "Final filing system", b: `the sample, ${int(D.labels.reviewed)} pages`, v: D.labels.uncertain_share },
      ];
      const Hh = 250, padT = 34, padB = 52, L = narrow ? 30 : 60, R = narrow ? 30 : 60;
      const x = i => L + (W - L - R) * i / (pts.length - 1), y = v => padT + (Hh - padT - padB) * (1 - v / .5);
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Share of labels marked unsure in each labeling round" });
      [0, .25, .5].forEach(v => { E(s, "line", { x1: 0, x2: W, y1: y(v), y2: y(v), stroke: "var(--grid)" }); E(s, "text", { x: 0, y: y(v) - 4, class: "lbl muted" }, pct(v)); });
      E(s, "line", { x1: x(0), y1: y(pts[0].v), x2: x(1), y2: y(pts[1].v), stroke: "var(--ink)", "stroke-width": 2.5 });
      E(s, "line", { x1: x(1), y1: y(pts[1].v), x2: x(2), y2: y(pts[2].v), stroke: "var(--ref)", "stroke-width": 2, "stroke-dasharray": "5 4" });
      pts.forEach((p, i) => {
        const anchor = i === 0 ? "start" : i === pts.length - 1 ? "end" : "middle", tx = i === 0 ? Math.max(0, x(i) - 6) : i === pts.length - 1 ? Math.min(W, x(i) + 6) : x(i);
        E(s, "circle", { cx: x(i), cy: y(p.v), r: 6.5, fill: hatchFill, stroke: "var(--person-line)", "data-tip": `${p.a} (${p.b}): ${pct(p.v)} labeled unsure` });
        E(s, "text", { x: tx, y: y(p.v) - 14, "text-anchor": anchor, class: "lbl val" }, pct(p.v));
        E(s, "text", { x: tx, y: Hh - padB + 22, "text-anchor": anchor, class: p.key ? "lbl strong" : "lbl" }, narrow && i === 1 ? "PM departments" : p.a);
        E(s, "text", { x: tx, y: Hh - padB + 37, "text-anchor": anchor, class: "lbl muted" }, p.b);
      });
      host.replaceChildren(s);
    },

    /* Intake: labels per department, sure and unsure */
    labels(host) {
      const deps = [...D.labels.by_department].sort((a, b) => (b.certain + b.uncertain) - (a.certain + a.uncertain));
      const W = widthOf(host), narrow = W < 560, labelW = narrow ? 0 : 200, padR = 125, rowH = narrow ? 46 : 30;
      const max = Math.max(...deps.map(d => d.certain + d.uncertain)), sx = v => (W - labelW - padR) * v / max;
      const Hh = deps.length * rowH + 4;
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Hand labels per department, labeled sure and labeled unsure" });
      deps.forEach((d, i) => {
        const y0 = i * rowH, yb = narrow ? y0 + 20 : y0 + 6, n = d.certain + d.uncertain;
        E(s, "text", { x: 0, y: narrow ? y0 + 14 : y0 + 20, class: "lbl" }, d.name);
        const w1 = sx(d.certain), w2 = sx(d.uncertain);
        E(s, "rect", { x: labelW, y: yb, width: w1, height: 18, fill: "var(--filed)", "data-tip": `${d.name}: ${int(d.certain)} labeled sure` });
        E(s, "rect", { x: labelW + w1 + 1.5, y: yb + .5, width: Math.max(1, w2 - 1), height: 17, fill: hatchFill, stroke: "var(--person-line)", "data-tip": `${d.name}: ${int(d.uncertain)} labeled unsure` });
        E(s, "text", { x: labelW + w1 + w2 + 8, y: yb + 14, class: "lbl val" }, `${int(n)}`);
        E(s, "text", { x: labelW + w1 + w2 + 8 + (String(n).length * 8 + 6), y: yb + 14, class: "lbl muted" }, `${pct(d.uncertain / n)} unsure`);
      });
      host.replaceChildren(s);
    },

    /* Accuracy on pages labeled sure vs unsure */
    certainty(host) {
      const ce = D.classifier.certainty, W = widthOf(host), narrow = W < 560;
      const rows = [["Sentence embeddings", ce.embeddings, true], ["TF-IDF", ce.tfidf, false]];
      const labelW = narrow ? 0 : 170, padR = 20, rowH = narrow ? 64 : 48, lo = .3, hi = .9;
      const sx = v => labelW + 8 + (W - labelW - 8 - padR) * (v - lo) / (hi - lo);
      const Hh = rows.length * rowH + 26;
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Accuracy on pages labeled sure and unsure, for both models" });
      [.3, .5, .7, .9].forEach(v => { E(s, "line", { x1: sx(v), x2: sx(v), y1: 0, y2: rows.length * rowH, stroke: "var(--grid)" }); E(s, "text", { x: sx(v), y: Hh - 6, "text-anchor": "middle", class: "lbl muted" }, pct(v)); });
      rows.forEach(([name, v, lead], i) => {
        const y0 = i * rowH, y = narrow ? y0 + 42 : y0 + 24;
        E(s, "text", { x: 0, y: narrow ? y0 + 14 : y0 + 28, class: lead ? "lbl strong" : "lbl" }, name);
        E(s, "line", { x1: sx(v.unsure.accuracy), x2: sx(v.sure.accuracy), y1: y, y2: y, stroke: "var(--rule)", "stroke-width": 3 });
        E(s, "circle", { cx: sx(v.unsure.accuracy), cy: y, r: 7.5, fill: hatchFill, stroke: "var(--person-line)", "data-tip": `${name}, pages labeled unsure (${int(v.unsure.pages)}): ${pct(v.unsure.accuracy)} right` });
        E(s, "circle", { cx: sx(v.sure.accuracy), cy: y, r: 7.5, fill: "var(--filed)", "data-tip": `${name}, pages labeled sure (${int(v.sure.pages)}): ${pct(v.sure.accuracy)} right` });
        E(s, "text", { x: sx(v.unsure.accuracy), y: y - 13, "text-anchor": "middle", class: "lbl val" }, pct(v.unsure.accuracy));
        E(s, "text", { x: sx(v.sure.accuracy), y: y - 13, "text-anchor": "middle", class: "lbl val" }, pct(v.sure.accuracy));
      });
      host.replaceChildren(s);
    },

    /* Confusion: row share heatmap plus the ranked largest confusions */
    confusion(host) {
      const c = D.classifier.confusion, W = widthOf(host), narrow = W < 560;
      const all = pairs(), cut = all[Math.min(3, all.length - 1)].n;
      const top = all.filter(p => p.n >= cut), topKeys = new Set();
      // competition ranking: equal counts share a rank ("=2"), listed alphabetically within a tie
      top.forEach(p => { p.rank = 1 + all.filter(q => q.n > p.n).length; p.tied = all.filter(q => q.n === p.n).length > 1; });
      top.sort((x, y) => x.rank - y.rank || (DEPT[x.a] + DEPT[x.b]).localeCompare(DEPT[y.a] + DEPT[y.b]));
      top.forEach(p => { topKeys.add(p.a + ">" + p.b); topKeys.add(p.b + ">" + p.a); });
      const t = H("table", "heat conf");
      t.append(H("caption", "vh", "Share of each true department's pages (rows) by the department the classifier chose (columns)"));
      const head = H("tr"); const corner = H("td", "corner"); head.append(corner);
      c.guesses.forEach(g => { const th = H("th", null, narrow ? g : DEPT[g] || g); th.scope = "col"; th.title = DEPT[g] || g; head.append(th); });
      const thead = H("thead"); thead.append(head); const body = H("tbody");
      c.labels.forEach((lab, i) => {
        const tr = H("tr"), th = H("th", null, narrow ? lab : DEPT[lab] || lab); th.scope = "row"; th.title = DEPT[lab] || lab; tr.append(th);
        const n = c.counts[i].reduce((a, b) => a + b, 0);
        c.guesses.forEach((g, j) => {
          const v = c.counts[i][j], share = v / n, td = H("td", null, pct(share));
          td.dataset.tip = `${DEPT[lab] || lab} filed as ${DEPT[g] || g}: ${v} of ${n} pages`;
          td.style.setProperty("--s", share.toFixed(3));
          if (share >= .5) td.className = "hi";
          if (g !== lab && topKeys.has(lab + ">" + g) && v) td.classList.add("miss");
          tr.append(td);
        });
        body.append(tr);
      });
      t.append(thead, body);
      const list = H("ul", "ranked");
      top.forEach(p => {
        const li = H("li");
        li.append(H("span", "rank", `${p.tied ? "=" : ""}${p.rank}`), H("b", null, `${DEPT[p.a] || p.a} and ${DEPT[p.b] || p.b}: ${int(p.n)} pages`),
          H("span", null, `${int(p.ab)} ${DEPT[p.a] || p.a} pages filed as ${DEPT[p.b] || p.b}; ${int(p.ba)} the other way`));
        list.append(li);
      });
      const cap = H("p", "ranked-h", "Largest confusions, both directions together (= marks a tie)");
      const codes = H("p", "codes", c.labels.map(l => `${l}: ${DEPT[l] || l}`).join("; ") + ".");
      host.replaceChildren(...(narrow ? [t, codes] : [t]), cap, list);
    },

    /* What a reviewer receives: schematic only, no page content */
    scheme(host) {
      if (host._built) return;
      host._built = true;
      const r = D.rag, model = r.model.replace(/\s*\(.*\)\s*$/, "");
      const arrow = '<svg class="arrow" viewBox="0 0 28 14" aria-hidden="true"><path d="M1 7h24M19 2l6 5-6 5"/></svg>';
      const page = '<svg viewBox="0 0 34 44" aria-hidden="true"><path d="M1 1h23l9 9v33H1z" fill="url(#hatch-legend)" stroke="var(--person-line)"/><path d="M24 1v9h9" fill="none" stroke="var(--person-line)"/></svg>';
      host.innerHTML = `<div class="scheme">
        <div class="col"><b>The case file</b>
          <div class="box page">${page}<span><b>The page</b>Its text as read, own Bates number removed</span></div>
          <div class="box"><b>${int(r.neighbors)} similar filed pages</b>Found by meaning among the ${int(D.classifier.pages)} labeled pages, each with where it was filed</div>
          <div class="box"><b>The filing manual</b>One card per department: its question, definition, hints and record types</div>
          <div class="box"><b>The classifier's two highest-ranked predictions</b>With how sure it is of each</div>
        </div>
        ${arrow}
        <div class="model"><b>${model}</b>a language model on the same PC, about ${r.seconds_per_page.toFixed(0)} s a page</div>
        ${arrow}
        <div class="col"><b>The suggestion</b>
          <div class="box out"><b>Record type</b>Department, series and record type</div>
          <div class="box"><b>Second choice</b>From a different department</div>
          <div class="box"><b>One-line reason</b>Why the page belongs there</div>
          <div class="box"><b>Sources used</b>Which similar pages and rules it relied on</div>
        </div>
      </div>`;
    },

    /* Suggestion ladder: practice review pages only */
    rag(host) {
      const W = widthOf(host), narrow = W < 560;
      const rows = D.rag.approaches.map((a, i, all) => ({ name: a.name, t1: a.top1, t2: a.top2, key: i === all.length - 1 }));
      ladder(host, W, narrow, rows, .5, [.5, .6, .7, .8, .9, 1], "First-choice and top-two accuracy of each approach on the cross-validated review pages");
    },

    /* Final test against practice, with 95% intervals */
    intervals(host) {
      const f = D.final_test, c = D.classifier, r = D.rag, q = f.rag;
      const W = widthOf(host), narrow = W < 560;
      const ragCI = wilson(Math.round(q.rag_top1 * q.review_pages), q.review_pages);
      const rows = [
        { name: "Filing score, sentence embeddings", sub: "macro F1", t: f.scores.macro_f1, cv: c.embeddings.hybrid.macro_f1 },
        { name: "Filing score, TF-IDF", sub: "macro F1", t: f.scores.tfidf_macro_f1, cv: c.tfidf.hybrid.macro_f1 },
        { name: "Automatic filings right", sub: `${int(f.auto_correct)} of ${int(f.auto_filed)} at cutoff ${f.cutoff.toFixed(2)}`, t: f.scores.auto_accuracy, cv: c.routing.accuracy_auto_filed, target: f.target },
        { name: "Suggestion right first time", sub: `${int(q.review_pages)} pages sent to a person`, t: { value: q.rag_top1, low: ragCI[0], high: ragCI[1] }, cv: r.approaches[r.approaches.length - 1].top1 },
        { name: "Whole pipeline right", sub: "if every suggestion were accepted", t: f.scores.pipeline_accuracy, cv: null },
      ];
      const labelW = narrow ? 0 : 230, padR = 46, lo = .4, rowH = narrow ? 64 : 52;
      const sx = v => labelW + 8 + (W - labelW - 8 - padR) * (Math.max(lo, v) - lo) / (1 - lo);
      const Hh = rows.length * rowH + 28;
      const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": "Final-test results with 95% intervals, compared with cross-validation" });
      [.4, .5, .6, .7, .8, .9, 1].forEach(v => {
        E(s, "line", { x1: sx(v), x2: sx(v), y1: 0, y2: rows.length * rowH, stroke: "var(--grid)" });
        if (!narrow || Math.round(v * 10) % 2 === 0) E(s, "text", { x: sx(v), y: Hh - 6, "text-anchor": "middle", class: "lbl muted" }, v.toFixed(1));
      });
      rows.forEach((row, i) => {
        const y0 = i * rowH, y = narrow ? y0 + 46 : y0 + 26;
        E(s, "text", { x: 0, y: narrow ? y0 + 14 : y0 + 22, class: "lbl strong" }, row.name);
        E(s, "text", { x: narrow ? 0 : 0, y: narrow ? y0 + 29 : y0 + 38, class: "lbl muted" }, row.sub);
        if (row.target) {
          E(s, "line", { x1: sx(row.target), x2: sx(row.target), y1: y - 20, y2: y + 13, stroke: "var(--muted)", "stroke-width": 1.5, "stroke-dasharray": "3 2" });
          E(s, "text", { x: sx(row.target) - 4, y: y - 15, "text-anchor": "end", class: "lbl muted" }, `${pct(row.target)} standard`);
        }
        E(s, "line", { x1: sx(row.t.low), x2: sx(row.t.high), y1: y, y2: y, stroke: "var(--filed)", "stroke-width": 4, "stroke-linecap": "round", opacity: .35 });
        if (row.cv != null) E(s, "circle", { cx: sx(row.cv), cy: y, r: 6.5, fill: "none", stroke: "var(--second)", "stroke-width": 2, "data-tip": `${row.name}, practice (cross-validation): ${row.cv.toFixed(2)}` });
        E(s, "circle", { cx: sx(row.t.value), cy: y, r: 6, fill: "var(--filed)", "data-tip": `${row.name}, final test: ${row.t.value.toFixed(2)} (95% interval ${row.t.low.toFixed(2)} to ${row.t.high.toFixed(2)})` });
        E(s, "text", { x: W, y: y + 4, "text-anchor": "end", class: "lbl val" }, row.t.value.toFixed(2));
      });
      const sr = H("p", "vh", rows.map(r => `${r.name}: final test ${r.t.value.toFixed(2)} (95% interval ${r.t.low.toFixed(2)} to ${r.t.high.toFixed(2)})${r.cv != null ? `, practice ${r.cv.toFixed(2)}` : ""}`).join("; ") + ".");
      host.replaceChildren(s, sr);
    },

    /* A box of new pages: one part-to-whole strip; misfilings shown as a projected range, not a count */
    scale(host) {
      const r = D.scale_run, W = widthOf(host);
      const lo = Math.round(r.projected_wrong.low), hi = Math.round(r.projected_wrong.high);
      const parts = [
        { n: r.auto_filed - hi, kind: "filed", name: "Filed, likely right", label: `${int(r.auto_filed - hi)} or more` },
        { n: hi - lo, kind: "band", name: "Filed, could be right or misfiled", label: int(hi - lo) },
        { n: lo, kind: "wrong", name: "Filed, likely misfiled", label: `${int(lo)} or more` },
        { n: r.reasons["below cutoff"], kind: "person", name: "To a person: not confident enough", label: int(r.reasons["below cutoff"]), gap: true },
        { n: r.reasons["under 20 tokens"], kind: "person", name: "To a person: too little text", label: int(r.reasons["under 20 tokens"]), gap: true },
        { n: r.reasons["no readable text"], kind: "person", name: "To a person: no readable text", label: int(r.reasons["no readable text"]), gap: true },
      ];
      const barH = 34, top = 30, gap = 2, gaps = parts.filter(p => p.gap).length;
      const unit = (W - gap * gaps) / r.pages;
      const s = E(null, "svg", { width: W, height: top + barH + 4, "aria-hidden": "true" });
      const draw = (g, kind, x, y, w, h) => {
        if (kind === "filed") return E(g, "rect", { x, y, width: w, height: h, fill: "var(--filed)" });
        else if (kind === "band") { const b = E(g, "rect", { x: x + .75, y: y + .75, width: w - 1.5, height: h - 1.5, stroke: "var(--wrong)", "stroke-width": 1.5, "stroke-dasharray": "4 3" }); b.style.fill = "color-mix(in srgb, var(--filed) 28%, var(--surface))"; return b; }
        else if (kind === "wrong") return E(g, "rect", { x: x + 1, y: y + 1, width: w - 2, height: h - 2, fill: "none", stroke: "var(--wrong)", "stroke-width": 2 });
        return E(g, "rect", { x: x + .5, y: y + .5, width: w - 1, height: h - 1, fill: hatchFill, stroke: "var(--person-line)" });
      };
      let x = 0, bandX = 0;
      parts.forEach(p => {
        if (p.gap) x += gap;
        const w = Math.max(2, unit * p.n);
        if (p.kind === "band") bandX = x;
        draw(s, p.kind, x, top, w, barH).setAttribute("data-tip", `${p.name}: ${p.kind === "band" ? `${p.label} pages (projected range)` : p.label}`);
        x += w;
        if (p.kind === "wrong") {
          // bracket over the projected range: the last `hi` filed pages, of which `lo` to `hi` are likely misfiled
          const x2 = x;
          E(s, "path", { d: `M${bandX} ${top - 4}V${top - 10}H${x2}V${top - 4}`, fill: "none", stroke: "var(--wrong)", "stroke-width": 1.5 });
          const t = E(s, "text", { x: x2, y: top - 15, "text-anchor": x2 > W * .5 ? "end" : "start", class: "lbl val", fill: "var(--wrong)" }, `Likely misfiled: ${int(lo)} to ${int(hi)} (projected)`);
          if (x2 < W * .5) t.setAttribute("x", bandX);
        }
      });
      const list = H("ul", "legend scale-key");
      parts.forEach(p => {
        const li = H("li"), sw = E(null, "svg", { viewBox: "0 0 13 13", "aria-hidden": "true" });
        draw(sw, p.kind, 0, 0, 13, 13);
        li.append(sw, H("span", null, `${p.name}: `), H("b", null, p.label));
        list.append(li);
      });
      const sr = H("p", "vh", `${int(r.auto_filed)} of ${int(r.pages)} new pages filed automatically; projected from the final test, between ${int(lo)} and ${int(hi)} of them are likely misfiled. ${int(r.to_a_person)} went to a person.`);
      host.replaceChildren(s, list, sr);
    },
  };

  /* Department names for the codes in the classifier output (labels, not numbers) */
  const DEPT = { AFF: "Corporate affairs", FIN: "Finance", LEG: "Legal", MKT: "Marketing and sales", QSH: "Quality, safety and health", RND: "Research and development" };
  function pairs() {
    const c = D.classifier.confusion, out = [];
    c.labels.forEach((a, i) => c.labels.forEach((b, j) => {
      if (j <= i) return;
      const ab = c.counts[i][c.guesses.indexOf(b)], ba = c.counts[j][c.guesses.indexOf(a)];
      if (ab + ba) out.push({ a, b, ab, ba, n: ab + ba });
    }));
    return out.sort((x, y) => y.n - x.n);
  }
  function foldLadder(host, rows, label) {
    const W = widthOf(host), narrow = W < 560, labelW = narrow ? 0 : 250, padR = 44, rowH = narrow ? 66 : 50;
    const sx = v => labelW + 8 + (W - labelW - 8 - padR) * v;
    const Hh = rows.length * rowH + 26;
    const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": label });
    [0, .2, .4, .6, .8, 1].forEach(v => { E(s, "line", { x1: sx(v), x2: sx(v), y1: 0, y2: rows.length * rowH, stroke: "var(--grid)" }); E(s, "text", { x: sx(v), y: Hh - 6, "text-anchor": "middle", class: "lbl muted" }, v.toFixed(1)); });
    rows.forEach((r, i) => {
      const y0 = i * rowH, y = narrow ? y0 + 48 : y0 + 25;
      const color = r.kind === "lead" ? "var(--ink)" : r.kind === "mid" ? "var(--second)" : "var(--ref)";
      E(s, "text", { x: 0, y: narrow ? y0 + 14 : y0 + 21, class: r.kind === "lead" ? "lbl strong" : "lbl" }, r.name);
      E(s, "text", { x: 0, y: narrow ? y0 + 29 : y0 + 37, class: "lbl muted" }, r.sub);
      (r.v.folds || []).forEach((f, k) => E(s, "circle", { cx: sx(f), cy: y, r: 3.5, fill: "none", stroke: color, "stroke-width": 1.5, "data-tip": `${r.name}, fold ${k + 1}: macro F1 ${f.toFixed(2)}` }));
      E(s, "circle", { cx: sx(r.v.macro_f1), cy: y, r: 7, fill: color, "data-tip": `${r.name}: mean macro F1 ${r.v.macro_f1.toFixed(2)}${r.v.folds ? " over five folds" : ""}` });
      E(s, "text", { x: W, y: y + 4, "text-anchor": "end", class: "lbl val" }, r.v.macro_f1.toFixed(2));
    });
    const sr = H("p", "vh", rows.map(r => `${r.name}: mean macro F1 ${r.v.macro_f1.toFixed(2)}${r.v.folds ? `, folds ${r.v.folds.map(f => f.toFixed(2)).join(", ")}` : ""}`).join("; ") + ".");
    host.replaceChildren(s, sr);
  }
  function wilson(k, n) {
    const z = 1.96, p = k / n, d = 1 + z * z / n, c = (p + z * z / (2 * n)) / d;
    const h = z * Math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d;
    return [c - h, c + h];
  }
  function ladder(host, W, narrow, rows, lo, ticks, label) {
    const labelW = narrow ? 0 : 270, padR = 40;
    const sx = v => labelW + 10 + (W - labelW - 10 - padR) * (v - lo) / (1 - lo);
    const rowH = narrow ? 50 : 34, Hh = rows.length * rowH + 26;
    const s = E(null, "svg", { width: W, height: Hh, role: "img", "aria-label": label });
    ticks.forEach(v => {
      E(s, "line", { x1: sx(v), x2: sx(v), y1: 0, y2: rows.length * rowH, stroke: "var(--grid)" });
      E(s, "text", { x: sx(v), y: Hh - 6, "text-anchor": "middle", class: "lbl muted" }, pct(v));
    });
    rows.forEach((r, i) => {
      const y0 = i * rowH, c = r.key ? "var(--filed)" : "var(--ref)";
      const ty = narrow ? y0 + 14 : y0 + 21, my = narrow ? y0 + 34 : y0 + 17;
      E(s, "text", { x: 0, y: ty, class: r.key ? "lbl strong" : "lbl" }, r.name);
      E(s, "line", { x1: sx(r.t1), x2: sx(r.t2), y1: my, y2: my, stroke: c, "stroke-width": 3 });
      E(s, "circle", { cx: sx(r.t2), cy: my, r: 5, fill: "var(--paper)", stroke: c, "stroke-width": 2.5, "data-tip": `${r.name}: right department among the top two ${pct(r.t2)}` });
      E(s, "circle", { cx: sx(r.t1), cy: my, r: 6, fill: c, "data-tip": `${r.name}: right first choice ${pct(r.t1)}` });
      E(s, "text", { x: sx(r.t1) - 10, y: my + 4, "text-anchor": "end", class: "lbl val" }, pct(r.t1));
      E(s, "text", { x: sx(r.t2) + 10, y: my + 4, class: "lbl muted" }, pct(r.t2));
    });
    host.replaceChildren(s);
  }

  /* Helper tips (the guide reads these; nothing moves while the reader works) */
  const tip = (text, key) => document.dispatchEvent(new CustomEvent("localrim:tip", { detail: { text, key } }));
  let engineTipped = false, cutTipped = false;

  /* Engine highlight */
  const state = { stage: "capture", engine: null, model: "embeddings", cut: null };
  function highlight(id) {
    state.engine = id;
    const e = eng();
    $$(".sbar").forEach(bar => {
      const dot = $$(".dot", bar).find(d => d.dataset.e === id), tag = $(".tag", bar), na = $(".na", bar);
      $$(".dot", bar).forEach(d => d.classList.toggle("on", d === dot));
      if (dot) {
        const p = Number(dot.dataset.p);
        tag.hidden = false; na.hidden = true;
        tag.textContent = `${short(e.names[id])}: ${dot.dataset.label}`;
        tag.style.left = `${p * 100}%`;
        tag.className = "tag" + (p < .22 ? " l" : p > .78 ? " r" : "");
        // keep long labels inside the strip on narrow screens
        const br = bar.getBoundingClientRect(), tr = tag.getBoundingClientRect();
        if (tr.width && tr.right > br.right + 4) tag.className = "tag r";
        else if (tr.width && tr.left < br.left - 4) tag.className = "tag l";
        // still too wide (long engine name on a small phone): the value alone; the picker names the engine
        const t2 = tag.getBoundingClientRect();
        if (t2.width && (t2.right > br.right + 4 || t2.left < br.left - 4)) {
          tag.textContent = dot.dataset.label;
          tag.className = "tag" + (p < .22 ? " l" : p > .78 ? " r" : "");
        }
      } else { tag.hidden = true; na.hidden = false; na.textContent = na.dataset.text; /* engine named in the picker and readout; one line above the axis */ }
    });
    $$("#engine-pick button").forEach(b => { const on = b.dataset.e === id; b.setAttribute("aria-checked", String(on)); b.tabIndex = on ? 0 : -1; });
    const A = e.analyses[id], r = e.review_overall.find(x => x.engine === id), f = e.funsd.find(x => x.engine === id);
    const parts = [
      f ? `${pct(f.word_recall)} of answer-key words` : "not run on the answer key",
      `${int(r.good + r.usable)} of ${int(r.pages)} review pages good or usable`,
      `${pct(1 - A.pages_mostly_junk / A.pages_expected)} of all pages read as mostly words`,
      `own Bates number on ${pct(A.pages_with_own_bates_found / A.pages_expected)}`,
      A.seconds_mean != null ? `${A.seconds_mean.toFixed(2)} s a page` : "no reading time: the text comes with the archive",
    ];
    const out = $("#engine-readout");
    out.textContent = "";
    out.append(H("b", null, `${short(e.names[id])}: `), document.createTextNode(parts.join("; ") + "."));
  }
  function buildEnginePick() {
    const e = eng(), box = $("#engine-pick");
    Object.keys(e.names).forEach(id => {
      const b = H("button", null, short(e.names[id]));
      b.type = "button"; b.setAttribute("role", "radio"); b.dataset.e = id;
      b.addEventListener("click", () => { highlight(id); if (!engineTipped) { engineTipped = true; tip("The engine you pick is highlighted on every test; its numbers appear below the strips.", "engine"); } });
      box.append(b);
    });
    radioKeys(box, b => highlight(b.dataset.e));
    state.engine = backbone();
  }
  function radioKeys(box, act) {
    box.addEventListener("keydown", ev => {
      const bs = $$("[role=radio]", box), i = bs.indexOf(document.activeElement);
      if (i < 0) return;
      let j = null;
      if (ev.key === "ArrowRight" || ev.key === "ArrowDown") j = (i + 1) % bs.length;
      if (ev.key === "ArrowLeft" || ev.key === "ArrowUp") j = (i - 1 + bs.length) % bs.length;
      if (ev.key === "Home") j = 0;
      if (ev.key === "End") j = bs.length - 1;
      if (j == null) return;
      ev.preventDefault(); bs[j].focus(); act(bs[j]);
    });
  }

  /* Cutoff */
  const curve = () => {
    const r = D.classifier.routing, c = state.model === "tfidf" ? r.curve_tfidf : r.curve_embeddings;
    return c.filter(p => p.auto_filed > 0 && p.accuracy_auto_filed != null);
  };
  function updateCut() {
    const host = $('[data-chart="cutoff"]');
    if (!host._draw) return;
    const pts = curve(), p = pts[state.cut], tg = D.classifier.routing.target, n = D.classifier.routing.pages;
    host._draw(state.cut);
    const meets = p.accuracy_auto_filed >= tg;
    const txt = `Cutoff ${p.threshold.toFixed(2)}: ${int(p.auto_filed)} of ${int(n)} pages filed automatically (${pct(p.coverage)}), and ${pct(p.accuracy_auto_filed)} of them right.`;
    const out = $("#cut-readout");
    out.textContent = txt + " ";
    out.append(H("span", "verdict " + (meets ? "meets" : "misses"), meets ? `Meets the ${pct(tg)} standard.` : `Below the ${pct(tg)} standard.`));
    $("#cut").setAttribute("aria-valuetext", `Cutoff ${p.threshold.toFixed(2)}: ${int(p.auto_filed)} filed automatically, ${pct(p.accuracy_auto_filed)} right`);
  }
  function wireCutoff() {
    $("#cut").addEventListener("input", e => { state.cut = Number(e.target.value); updateCut(); });
    // after the first release, not while dragging
    $("#cut").addEventListener("change", () => { if (!cutTipped) { cutTipped = true; tip("The dashed line is the 90% standard, set before any result; the frozen cutoff is marked above the chart.", "cutoff"); } });
    const box = $("#model-pick");
    const pick = b => {
      state.model = b.dataset.model;
      $$("button", box).forEach(x => { const on = x === b; x.setAttribute("aria-checked", String(on)); x.tabIndex = on ? 0 : -1; });
      draw($('[data-chart="cutoff"]'), true);
      if (state.model === "tfidf") tip("In cross-validation, TF-IDF never reaches the 90% standard at any cutoff; sentence embeddings do.", "model");
    };
    $$("button", box).forEach(b => b.addEventListener("click", () => pick(b)));
    radioKeys(box, pick);
  }

  /* Drawing on demand */
  function draw(host, force) {
    const w = widthOf(host);
    if (!w || (!force && host._w === w)) return;
    host._w = w;
    CHARTS[host.dataset.chart](host);
  }
  // The same charts serve the landing's stage switcher (inside panels) and the chapters (data-cf hosts)
  const ro = new ResizeObserver(entries => entries.forEach(en => draw(en.target)));

  /* Mockup controls */
  $$("[data-set-theme]").forEach(b => b.addEventListener("click", () => {
    const t = b.dataset.setTheme;
    if (t === "auto") delete document.documentElement.dataset.theme; else document.documentElement.dataset.theme = t;
    $$("[data-set-theme]").forEach(x => x.setAttribute("aria-pressed", String(x === b)));
  }));
  const rm = $("#reduce");
  if (rm) rm.addEventListener("click", () => { forceReduce = !forceReduce; rm.setAttribute("aria-pressed", String(forceReduce)); document.documentElement.classList.toggle("reduce", forceReduce); });
  const q = new URLSearchParams(location.search);
  if (q.get("theme")) document.documentElement.dataset.theme = q.get("theme");
  if (q.get("reduce") === "1") { forceReduce = true; document.documentElement.classList.add("reduce"); if (rm) rm.setAttribute("aria-pressed", "true"); }

  fetch("data/site_data.json", { cache: "no-store" })
    .then(r => r.json())
    .then(d => {
      D = d;
      decisions();                        // site.js binds the text on chapter pages; here the shared charts are drawn
      if ($("#engine-pick")) buildEnginePick();
      if ($("#cut")) wireCutoff();
      $$("[data-cf]").forEach(h => { ro.observe(h); draw(h); });
    })
    .catch(() => {
      $$("[data-cf]").forEach(h => h.insertAdjacentHTML("afterbegin", '<p class="empty"><b>The data did not load.</b> Serve this folder over http (port 8001) and reload.</p>'));
    });
})();
