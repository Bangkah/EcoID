// Drives the real index.html <script> against a real server with a stub DOM (no browser needed).
// usage: node smoke.js <port> <image> <index.html>
const vm = require("vm"), fs = require("fs");
const [, , port, imgPath, htmlPath] = process.argv;
const js = fs.readFileSync(htmlPath, "utf8").match(/<script>([\s\S]*)<\/script>/)[1];
const els = {};
function stub(id, tag) {
  const e = { id, tag, children: [], _cls: new Set(), attrs: {}, dataset: {}, style: {}, value: "", textContent: "", src: "", open: false,
    classList: { toggle(c, on) { on === undefined ? (e._cls.has(c) ? e._cls.delete(c) : e._cls.add(c)) : (on ? e._cls.add(c) : e._cls.delete(c)); return !!on; },
                 add: (c) => e._cls.add(c), remove: (c) => e._cls.delete(c), contains: (c) => e._cls.has(c) },
    setAttribute(k, v) { e.attrs[k] = v; }, append(...k) { e.children.push(...k); }, replaceChildren(...k) { e.children = k; },
    showModal() { e.open = true; }, close() { e.open = false; },
    addEventListener() {}, setPointerCapture() {}, getBoundingClientRect: () => ({ width: 640, height: 420, left: 0, top: 0 }),
    offsetWidth: 240, offsetHeight: 160, remove() {}, click() {} };
  return e;
}
const doc = {
  getElementById: (id) => (els[id] ||= stub(id)),
  createElement: (t) => stub(null, t),
  createElementNS: (ns, t) => stub(null, t),
  addEventListener() {},
  body: stub('body'),
  createTextNode: (t) => ({ text: t }),
  querySelectorAll: () => [],
};
const base = "http://127.0.0.1:" + port;
const ctx = vm.createContext({
  document: doc, navigator: {}, console, Blob, URL, URLSearchParams, setTimeout, clearTimeout, Date, Number,
  confirm: () => true,
  window: { addEventListener() {} }, requestAnimationFrame: (f) => setTimeout(f, 0),
  fetch: (p, o) => fetch(p.startsWith("http") ? p : base + p, o),
});
const visible = (id) => !doc.getElementById(id)._cls.has("hidden");
const fail = (m) => { console.log(JSON.stringify({ ok: false, error: m })); process.exit(1); };
const text = (e) => [e.textContent, ...e.children.map((c) => (c.text ?? text(c)))].join(" ");

(async () => {
  vm.runInContext(js, ctx);
  await new Promise((r) => setTimeout(r, 300));                       // initial /api/model fetch
  const bytes = fs.readFileSync(imgPath);

  await vm.runInContext("submitImage", ctx)(new Blob([bytes], { type: "image/jpeg" }));
  if (!visible("s-result") || visible("s-capture")) fail("result step not shown");
  const cands = text(doc.getElementById("res-cands"));
  if (!cands.includes("Mangifera indica") || !cands.includes("Mangga")) fail("candidates not rendered: " + cands);
  if (visible("res-warn")) fail("low-confidence banner shown for a confident result");

  doc.getElementById("res-notes").value = "<b>leaf</b> ok";
  await vm.runInContext("saveDraft", ctx)("VERIFIED");
  if (!visible("s-saved")) fail("saved step not shown: " + text(doc.getElementById("result-err")));

  vm.runInContext("setTab", ctx)("history");                          // like clicking the History tab
  await new Promise((r) => setTimeout(r, 400));
  const list = doc.getElementById("hist-list");
  if (list.children.length < 1) fail("history empty");
  if (!text(doc.getElementById("hist-count")).includes("observation")) fail("count missing");
  const first = JSON.parse(await (await fetch(base + "/api/observations?limit=1")).text()).items[0];
  if (first.notes !== "<b>leaf</b> ok") fail("notes not stored verbatim");

  vm.runInContext("openDetail", ctx)(first);
  if (!doc.getElementById("dlg-detail").open) fail("detail dialog not opened");
  await vm.runInContext("patch", ctx)({ verification_status: "REJECTED" });
  if (!text(doc.getElementById("d-sub")).includes("Rejected")) fail("status change not reflected: " + text(doc.getElementById("d-sub")));
  await doc.getElementById("d-delete").onclick();
  const after = JSON.parse(await (await fetch(base + "/api/observations/" + first.id)).text());
  if (!after.error) fail("observation not deleted");

  // low-confidence rendering
  const gray = await (await fetch(base + "/api/identify", { method: "POST", headers: { "Content-Type": "image/png" },
    body: fs.readFileSync(imgPath.replace("plant.jpg", "gray.png")) })).json();
  console.log(JSON.stringify({ ok: true, gray_status: gray.result.status }));
})().catch((e) => fail(String(e && e.stack || e)));
