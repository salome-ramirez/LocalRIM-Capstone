/* The helper.
   - Rests as the mascot alone in the lower corner; its line opens when clicked or tapped (or by itself
     only where the page has a free margin), so it never covers content.
   - Enters once per page: a short peek (12 px slide and fade, 240 ms) after the reader has scrolled a
     little, never on arrival. On the landing it flies in once along an arc when "Open the file" comes
     into view (the earlier mascot.js idea).
   - Pages can hand it a tip with document.dispatchEvent(new CustomEvent("localrim:tip", {detail})),
     detail = { text, key }: the line changes; a key's first tip adds one soft cue (a 2 degree tilt).
     Nothing moves while the reader drags or reads; there is no idle or continuous motion.
   - x hides it for that page (remembered in this browser); a "Show the helper again" link in the
     footer brings it back. Under prefers-reduced-motion it appears and leaves without any movement. */
(function () {
  const LINES = {
    // one line per page in the capstone voice
    "index.html": "Welcome to the file room. Each stage below opens its chapter.",
    "dashboard.html": "This is the project's paper trail: every decision is logged here, with its reason.",
    "filing-system.html": "Every record gets one office of record: the department that keeps the official copy.",
    "reading.html": "A page has to be read before it can be filed. Five ways of reading the pages were compared; PaddleOCR reads every page, and GLM-OCR helps on the hardest ones.",
    "review.html": "When the system isn't sure, it hands the page to a person, with a suggestion, instead of guessing.",
    "findings.html": "Four questions, their answers, and what the evidence can't support yet.",
  };
  const page = location.pathname.split("/").pop() || "index.html";
  const line = LINES[page];
  if (!line) return;
  const KEY = `localrim-helper-dismissed:${page}`;
  const reduce = matchMedia("(prefers-reduced-motion: reduce)");
  let dismissed = false;
  try { dismissed = !!localStorage.getItem(KEY); } catch (e) { /* storage blocked: show him */ }

  // Footer link that brings him back after he was hidden on this page
  function offerReturn() {
    if (document.querySelector(".helper-return")) return;
    const host = document.querySelector("footer.site-foot, footer.site, main") || document.body;
    const p = document.createElement("p");
    p.className = "helper-return";
    p.innerHTML = '<button type="button">Show the helper again</button>';
    p.querySelector("button").onclick = () => {
      try { localStorage.removeItem(KEY); } catch (e) { /* ignore */ }
      p.remove(); build(true);
    };
    host.append(p);
  }
  if (dismissed) { addEventListener("DOMContentLoaded", offerReturn); if (document.readyState !== "loading") offerReturn(); return; }

  function build(now) {
    const box = document.createElement("div");
    box.className = "helper";
    box.innerHTML = `<div class="helper-bubble" role="status" id="helper-line" hidden><span class="helper-text"></span><button class="helper-close" aria-label="Hide the helper on this page">×</button></div>` +
                    `<button type="button" class="helper-toggle" aria-expanded="false" aria-controls="helper-line" aria-label="Show the helper's tip"><img class="helper-mascot" src="assets/mascot_mulberry.png" alt=""></button>`;
    document.body.append(box);
    document.documentElement.classList.add("has-helper");
    const bubble = box.querySelector(".helper-bubble"), text = box.querySelector(".helper-text"), toggle = box.querySelector(".helper-toggle");
    text.textContent = line;
    const setOpen = open => { bubble.hidden = !open; toggle.setAttribute("aria-expanded", String(open)); };
    const room = () => { const m = document.querySelector("main"); return innerWidth - (m ? m.getBoundingClientRect().right : innerWidth) >= 300; };

    const leave = () => {
      if (box.classList.contains("is-leaving")) return;
      try { localStorage.setItem(KEY, "1"); } catch (e) { /* storage blocked */ }
      bubble.classList.add("is-fading");
      setTimeout(() => box.classList.add("is-leaving"), 200);
      setTimeout(() => { box.remove(); document.documentElement.classList.remove("has-helper"); offerReturn(); }, reduce.matches ? 300 : 1100);
    };
    box.querySelector(".helper-close").onclick = leave;
    toggle.onclick = () => setOpen(bubble.hidden);
    addEventListener("keydown", e => { if (e.key === "Escape" && !bubble.hidden) { setOpen(false); toggle.focus(); } });

    // Tips from the page: change the line; the first tip for a key gives one soft cue
    const cued = new Set();
    document.addEventListener("localrim:tip", e => {
      const d = e.detail || {};
      if (!d.text) return;
      text.textContent = d.text;
      if (d.key && !cued.has(d.key) && box.classList.contains("is-in") && !reduce.matches) {
        cued.add(d.key);
        box.classList.remove("is-cue"); void box.offsetWidth; box.classList.add("is-cue");
      }
    });

    const enter = mode => {
      if (box.classList.contains("is-in")) return;
      box.classList.add("is-in", mode === "fly" && !reduce.matches ? "is-flying" : "is-peeking");
      if (room()) setOpen(true);
    };
    if (now) { enter("peek"); return; }

    // Landing: fly in once when "Open the file" comes into view. Elsewhere: peek after a little scrolling.
    const openSection = page === "index.html" && document.querySelector(".open");
    if (openSection && "IntersectionObserver" in window) {
      const io = new IntersectionObserver(([en]) => { if (en.intersectionRatio >= 0.5) { io.disconnect(); enter("fly"); } }, { threshold: [0.5] });
      io.observe(openSection);
    } else {
      const onScroll = () => { if (scrollY > innerHeight * 0.25) { removeEventListener("scroll", onScroll); enter("peek"); } };
      addEventListener("scroll", onScroll, { passive: true });
      onScroll();
    }
  }
  if (document.readyState === "loading") addEventListener("DOMContentLoaded", () => build(false)); else build(false);
})();
