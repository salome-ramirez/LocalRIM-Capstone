/* File index: the decision log grouped by the "stage" field that
   build_site_data.py writes for every decision. Each stage shows its two key decisions;
   the rest sit in a <details> (keyboard and screen-reader friendly, found by Ctrl+F).
   Any decision with an unknown stage still appears, under "Not yet assigned to a stage". */
(() => {
  "use strict";
  const STAGES = [
    { id: "sample", name: "Sample", key: ["D-32", "D-42"], stage: "intake", stageName: "Intake", chapter: "01 · Intake", href: "filing-system.html" },
    { id: "labels", name: "Labels and filing system", key: ["D-33", "D-38"], stage: "intake", stageName: "Intake", chapter: "01 · Intake", href: "filing-system.html" },
    { id: "ocr", name: "Reading the pages (OCR)", key: ["D-44", "D-52"], stage: "capture", stageName: "Capture", chapter: "02 · Capture", href: "reading.html#capture" },
    { id: "classifier", name: "Classifier", key: ["D-59", "D-57"], stage: "classification", stageName: "Classification", chapter: "03 · Classification", href: "review.html#classification" },
    { id: "routing", name: "Routing and review suggestion", key: ["D-58", "D-61"], stage: "routing", stageName: "Routing", chapter: "03 · Routing", href: "review.html#routing" },
    { id: "final", name: "Final test and scale run", key: ["D-62", "D-63"], stage: "validation", stageName: "Validation", chapter: "03 · Validation", href: "review.html#validation" },
  ];
  const H = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
  const num = s => Number(String(s).replace(/\D/g, ""));
  const row = d => {
    const det = H("details"), sum = H("summary");
    sum.append(H("span", "did", d.id), H("span", "text", d.decision), H("span", "date", d.date));
    det.append(sum, H("p", null, d.why));
    return det;
  };

  function build(data) {
    const host = document.getElementById("log-stages");
    const all = data.decisions || [];
    const known = new Set(STAGES.map(s => s.id));
    const groups = STAGES.map(s => ({ ...s, items: all.filter(d => d.stage === s.id) }));
    const rest = all.filter(d => !known.has(d.stage));
    if (rest.length) groups.push({ id: "other", name: "Not yet assigned to a stage", key: [], items: rest });
    groups.filter(g => g.items.length).forEach(g => {
      const box = H("div", "decisions log-stage");
      box.id = `log-${g.id}`;
      const items = g.items.slice().sort((a, b) => num(a.id) - num(b.id));
      const key = items.filter(d => g.key.includes(d.id)), more = items.filter(d => !g.key.includes(d.id));
      box.append(H("h3", null, g.name));
      const sub = H("p", "sub");
      sub.textContent = `${items.length} decision${items.length === 1 ? "" : "s"}`;
      /* Each group links to its chapter section */
      if (g.href) { sub.append(". In the chapter: "); const a = H("a", null, g.chapter); a.href = g.href; sub.append(a); }
      box.append(sub);
      (key.length ? key : items.slice(0, 2)).forEach(d => box.append(row(d)));
      const rem = key.length ? more : items.slice(2);
      if (rem.length) {
        const det = H("details", "more"), s = H("summary", null, `Show ${rem.length} more`);
        det.append(s); rem.forEach(d => det.append(row(d)));
        det.addEventListener("toggle", () => { s.textContent = det.open ? `Hide ${rem.length} more` : `Show ${rem.length} more`; });
        box.append(det);
      }
      host.append(box);
    });
    document.getElementById("log-count").textContent = `${all.length} decisions in ${groups.filter(g => g.items.length).length} stages`;
    const btn = document.getElementById("log-open-all");
    btn.addEventListener("click", () => {
      const open = btn.getAttribute("aria-pressed") !== "true";
      host.querySelectorAll("details").forEach(d => { d.open = open; });
      btn.setAttribute("aria-pressed", String(open));
      btn.textContent = open ? "Close all" : "Open all";
    });
    if (location.hash.startsWith("#log")) document.getElementById(location.hash.slice(1))?.scrollIntoView();
  }

  fetch("data/site_data.json", { cache: "no-store" })
    .then(r => r.json())
    .then(build)
    .catch(() => { document.getElementById("log-stages").innerHTML = '<p class="empty"><b>The decision log did not load.</b> Serve this folder over http and reload.</p>'; });
})();
