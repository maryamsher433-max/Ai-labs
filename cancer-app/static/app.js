// Frontend logic: build sliders from /api/info, call /api/predict on every change.
const $ = (id) => document.getElementById(id);
let info, values = {}, last = null, timer = null;
const HKEY = "tumour_history";

async function init() {
  try {
    info = await (await fetch("/api/info")).json();
    $("status").textContent = "API connected";
    $("status").className = "badge on";
  } catch (e) {
    $("status").textContent = "API offline";
    $("status").className = "badge off";
    return;
  }

  info.features.forEach((f) => {
    const step = (f.max - f.min) / 200;
    $("sliders").insertAdjacentHTML("beforeend",
      `<div class="row"><label>${f.name}<output id="o_${f.name}"></output></label>
       <input type="range" id="r_${f.name}" min="${f.min}" max="${f.max}" step="${step}"></div>`);
    $("r_" + f.name).addEventListener("input", (e) => {
      values[f.name] = +e.target.value;
      refresh();
    });
  });
  $("thead").innerHTML = "<th>#</th>" + info.features.map((f) => `<th>${f.name.replace("mean ", "")}</th>`).join("") + "<th>Result</th>";

  const cm = info.confusion_matrix;
  $("quality").innerHTML =
    `<p>Accuracy on ${info.test_size} unseen samples: <b>${(info.accuracy * 100).toFixed(1)}%</b></p>
     <table style="max-width:360px"><tr><th></th><th>Pred. malignant</th><th>Pred. benign</th></tr>
     <tr><th>Actually malignant</th><td>${cm[0][0]}</td><td>${cm[0][1]}</td></tr>
     <tr><th>Actually benign</th><td>${cm[1][0]}</td><td>${cm[1][1]}</td></tr></table>`;

  resetValues();
  renderHistory();
}

function resetValues() {
  info.features.forEach((f) => (values[f.name] = f.mean));
  refresh();
}

function randomValues() {
  info.features.forEach((f) => {
    // random value inside the middle 80% of the range
    values[f.name] = f.min + (f.max - f.min) * (0.1 + Math.random() * 0.6);
  });
  refresh();
}

function refresh() {
  info.features.forEach((f) => {
    $("r_" + f.name).value = values[f.name];
    $("o_" + f.name).textContent = (+values[f.name]).toFixed(3);
  });
  clearTimeout(timer);
  timer = setTimeout(predict, 80); // debounce so we don't spam the API
}

async function predict() {
  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ values }),
    });
    if (!res.ok) throw new Error(await res.text());
    last = await res.json();
    show(last);
  } catch (e) {
    $("label").textContent = "Error";
    $("prob").textContent = String(e).slice(0, 120);
  }
}

function show(r) {
  const benign = r.label === "Benign";
  $("label").textContent = r.label;
  $("label").style.color = benign ? "var(--ok)" : "var(--bad)";
  $("prob").textContent = `Benign ${(r.probability_benign * 100).toFixed(1)}%  ·  Malignant ${(r.probability_malignant * 100).toFixed(1)}%`;
  $("fill").style.width = r.probability_benign * 100 + "%";

  const mx = Math.max(1, ...r.contributions.map((c) => Math.abs(c.effect)));
  $("why").innerHTML = r.contributions.map((c) => {
    const w = (Math.abs(c.effect) / mx) * 50, up = c.effect >= 0;
    return `<span>${c.feature.replace("mean ", "")}</span>
      <div class="track"><b style="left:${up ? 50 : 50 - w}%;width:${w}%;background:${up ? "var(--ok)" : "var(--bad)"}"></b></div>
      <span style="text-align:right">${up ? "+" : ""}${c.effect.toFixed(2)}</span>`;
  }).join("");
}

// ---- history (saved in the browser) ----
const loadH = () => { try { return JSON.parse(localStorage.getItem(HKEY)) || []; } catch { return []; } };
const saveH = (h) => { try { localStorage.setItem(HKEY, JSON.stringify(h)); } catch {} };

function renderHistory() {
  const h = loadH();
  $("count").textContent = h.length ? `(${h.length})` : "";
  $("tbody").innerHTML = h.length
    ? h.map((row, i) => `<tr><td>${i + 1}</td>${info.features.map((f) => `<td>${(+row.v[f.name]).toFixed(2)}</td>`).join("")}
        <td class="tag ${row.label}">${row.label} ${(row.p * 100).toFixed(0)}%</td></tr>`).join("")
    : `<tr><td colspan="9" class="empty">Nothing saved yet. Click "Save to history".</td></tr>`;
}

$("reset").onclick = resetValues;
$("random").onclick = randomValues;
$("save").onclick = () => {
  if (!last) return;
  const h = loadH();
  h.push({ v: { ...values }, label: last.label, p: last.probability_benign >= 0.5 ? last.probability_benign : last.probability_malignant });
  saveH(h);
  renderHistory();
};
$("clear").onclick = () => { saveH([]); renderHistory(); };

init();
