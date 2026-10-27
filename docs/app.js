/* Breast mass classifier dashboard.
 * Loads model.json (exported by `python -m bcxai.train`) and recomputes the
 * logistic regression and its exact per-feature contributions in the browser.
 */
(() => {
  "use strict";

  const REPO_URL = "https://github.com/addiboak100-hue/breast-cancer-xai-dashboard";
  const TOP_N = 8;

  const GROUP_NOTES = {
    worst: "Worst: the mean of the three largest values in the image.",
    mean: "Mean: the average across all nuclei in the image.",
    error: "Standard error: how much the measurement varies across nuclei.",
  };

  const $ = (id) => document.getElementById(id);
  const state = {
    model: null,
    values: [],
    activeGroup: "worst",
    activeExample: null,
    edited: false,
    threshold: 0.5,
  };

  // ---------- helpers ----------

  const sigmoid = (v) => 1 / (1 + Math.exp(-v));

  function prettyName(raw) {
    let stat = "";
    let base = raw;
    if (raw.startsWith("mean ")) { stat = "mean"; base = raw.slice(5); }
    else if (raw.startsWith("worst ")) { stat = "worst"; base = raw.slice(6); }
    else if (raw.endsWith(" error")) { stat = "std. error"; base = raw.slice(0, -6); }
    const cap = base.charAt(0).toUpperCase() + base.slice(1);
    return { base: cap, stat, full: `${cap} (${stat})` };
  }

  function groupOf(raw) {
    if (raw.startsWith("mean ")) return "mean";
    if (raw.startsWith("worst ")) return "worst";
    return "error";
  }

  function fmtValue(v) {
    const a = Math.abs(v);
    if (a >= 100) return v.toFixed(0);
    if (a >= 10) return v.toFixed(1);
    if (a >= 1) return v.toFixed(2);
    if (a >= 0.1) return v.toFixed(3);
    return v.toFixed(4);
  }

  function fmtPct(p) {
    if (p < 0.01) return "<1%";
    if (p > 0.99) return ">99%";
    return `${Math.round(p * 100)}%`;
  }

  const fmtSigned = (v) => `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`;

  function predict(values) {
    const m = state.model;
    const contribs = m.features.map((f, i) => f.coef * ((values[i] - f.mean) / f.scale));
    const logit = m.model.intercept + contribs.reduce((a, b) => a + b, 0);
    return { contribs, logit, p: sigmoid(logit) };
  }

  // ---------- tooltip ----------

  const tip = $("tooltip");
  function showTip(html, evt) {
    tip.innerHTML = html;
    tip.hidden = false;
    moveTip(evt);
  }
  function moveTip(evt) {
    if (tip.hidden) return;
    let x, y;
    if (evt && "clientX" in evt && evt.clientX) {
      x = evt.clientX; y = evt.clientY;
    } else if (evt && evt.currentTarget) {
      const r = evt.currentTarget.getBoundingClientRect();
      x = r.left + r.width / 2; y = r.bottom;
    } else return;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > window.innerWidth - 8) left = x - w - 14;
    if (top + h > window.innerHeight - 8) top = y - h - 14;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  const hideTip = () => { tip.hidden = true; };

  function attachTip(el, htmlFn) {
    el.addEventListener("mouseenter", (e) => showTip(htmlFn(), e));
    el.addEventListener("mousemove", moveTip);
    el.addEventListener("mouseleave", hideTip);
    el.addEventListener("focus", (e) => showTip(htmlFn(), e));
    el.addEventListener("blur", hideTip);
  }

  // ---------- examples ----------

  function buildExamples() {
    const wrap = $("examples");
    const m = state.model;
    const list = [
      ...m.examples,
      { label: "Average case", true_class: null, values: m.features.map((f) => f.mean) },
    ];
    list.forEach((ex) => {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = ex.label;
      b.setAttribute("aria-pressed", "false");
      b.addEventListener("click", () => loadExample(ex, b));
      wrap.appendChild(b);
    });
    return list;
  }

  function loadExample(ex, button) {
    state.values = ex.values.slice();
    state.activeExample = ex;
    state.edited = false;
    document.querySelectorAll("#examples button").forEach((b) =>
      b.setAttribute("aria-pressed", String(b === button)));
    renderSliders();
    renderResult();
  }

  // ---------- sliders ----------

  function renderSliders() {
    const m = state.model;
    const host = $("sliders");
    host.innerHTML = "";
    $("group-note").textContent = GROUP_NOTES[state.activeGroup];
    host.setAttribute("aria-labelledby", `tab-${state.activeGroup}`);

    m.features.forEach((f, i) => {
      if (groupOf(f.name) !== state.activeGroup) return;
      const name = prettyName(f.name);
      const lo = Math.min(f.min, state.values[i]);
      const hi = Math.max(f.max, state.values[i]);
      const step = (hi - lo) / 400;

      const row = document.createElement("div");
      row.className = "slider-row";
      row.dataset.index = i;

      const id = `f-${i}`;
      const label = document.createElement("label");
      label.htmlFor = id;
      label.textContent = name.base;

      const track = document.createElement("div");
      track.className = "track";
      [["benign", f.median_benign], ["malignant", f.median_malignant]].forEach(([cls, v]) => {
        const t = document.createElement("span");
        t.className = `tick ${cls}`;
        t.style.left = `calc(8px + (100% - 16px) * ${(v - lo) / (hi - lo)})`;
        track.appendChild(t);
      });

      const range = document.createElement("input");
      range.type = "range";
      range.id = id;
      range.min = lo; range.max = hi; range.step = step;
      range.value = state.values[i];
      range.setAttribute("aria-describedby", `${id}-desc`);

      const desc = document.createElement("span");
      desc.id = `${id}-desc`;
      desc.className = "visually-hidden";
      desc.textContent = `Typical benign ${fmtValue(f.median_benign)}, typical malignant ${fmtValue(f.median_malignant)}`;

      const num = document.createElement("input");
      num.type = "number";
      num.step = "any";
      num.value = fmtValue(state.values[i]);
      num.setAttribute("aria-label", `${name.full} value`);

      range.addEventListener("input", () => {
        state.values[i] = parseFloat(range.value);
        num.value = fmtValue(state.values[i]);
        markEdited();
        renderResult();
      });
      num.addEventListener("change", () => {
        const v = parseFloat(num.value);
        if (Number.isFinite(v) && v >= 0) {
          state.values[i] = v;
          range.value = v;
          markEdited();
          renderResult();
        } else {
          num.value = fmtValue(state.values[i]);
        }
      });

      track.appendChild(range);
      row.append(label, track, num, desc);
      host.appendChild(row);
    });
  }

  function markEdited() {
    if (!state.edited) {
      state.edited = true;
      document.querySelectorAll("#examples button").forEach((b) =>
        b.setAttribute("aria-pressed", "false"));
    }
  }

  function selectTab(group, focus = false) {
    state.activeGroup = group;
    document.querySelectorAll(".tabs button").forEach((b) => {
      const on = b.dataset.group === group;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
      if (on && focus) b.focus();
    });
    renderSliders();
  }

  function setupTabs() {
    const tabs = [...document.querySelectorAll(".tabs button")];
    tabs.forEach((b, idx) => {
      b.tabIndex = b.getAttribute("aria-selected") === "true" ? 0 : -1;
      b.addEventListener("click", () => selectTab(b.dataset.group));
      b.addEventListener("keydown", (e) => {
        if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
        const next = tabs[(idx + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length];
        selectTab(next.dataset.group, true);
      });
    });
  }

  function jumpToFeature(i) {
    const f = state.model.features[i];
    selectTab(groupOf(f.name));
    const row = document.querySelector(`.slider-row[data-index="${i}"]`);
    if (!row) return;
    row.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
    row.classList.add("flash");
    setTimeout(() => row.classList.remove("flash"), 1400);
    row.querySelector('input[type="range"]').focus({ preventScroll: true });
  }

  // ---------- result ----------

  function renderResult() {
    const m = state.model;
    const { contribs, p } = predict(state.values);
    const malignant = p >= state.threshold;

    const label = $("verdict-label");
    label.textContent = malignant ? "Likely malignant" : "Likely benign";
    label.className = `verdict-label ${malignant ? "is-malignant" : "is-benign"}`;
    $("prob").textContent = fmtPct(p);
    $("meter-fill").style.width = `${Math.max(1, p * 100)}%`;
    $("meter-threshold").style.left = `${state.threshold * 100}%`;

    const truth = $("truth");
    const ex = state.activeExample;
    if (!state.edited && ex && ex.true_class) {
      const right = (ex.true_class === "malignant") === malignant;
      truth.textContent = `This is a real test-set case. Its confirmed diagnosis was ${ex.true_class}, so the model is ${right ? "correct" : "wrong"} here.`;
    } else if (state.edited) {
      truth.textContent = "Edited case. Move the measurements to see how the prediction responds.";
    } else {
      truth.textContent = "Every measurement set to the training-set average.";
    }

    renderContributions(contribs, p);
  }

  function renderContributions(contribs, p) {
    const m = state.model;
    const order = contribs.map((c, i) => i).sort((a, b) => Math.abs(contribs[b]) - Math.abs(contribs[a]));
    const top = order.slice(0, TOP_N);
    const rest = order.slice(TOP_N);
    const restSum = rest.reduce((s, i) => s + contribs[i], 0);

    const rows = top.map((i) => ({ i, value: contribs[i] }));
    rows.push({ i: null, value: restSum, count: rest.length });
    const maxAbs = Math.max(...rows.map((r) => Math.abs(r.value)), 0.5);

    const host = $("contrib");
    host.innerHTML = "";
    rows.forEach((r) => {
      const row = document.createElement("div");
      row.className = "bar-row";
      row.setAttribute("role", "listitem");

      const nameEl = document.createElement("div");
      nameEl.className = "bar-name";
      const plot = document.createElement("div");
      plot.className = "bar-plot";
      plot.innerHTML = '<span class="zero"></span>';

      const halfPct = (Math.abs(r.value) / maxAbs) * 34; // percent of full plot width
      const bar = document.createElement("span");
      const val = document.createElement("span");
      val.className = "bar-value";
      val.textContent = fmtSigned(r.value);
      if (r.value >= 0) {
        bar.className = "bar pos";
        bar.style.left = "50%";
        bar.style.width = `${halfPct}%`;
        val.style.left = `calc(50% + ${halfPct}% + 6px)`;
      } else {
        bar.className = "bar neg";
        bar.style.right = "50%";
        bar.style.width = `${halfPct}%`;
        val.style.right = `calc(50% + ${halfPct}% + 6px)`;
      }
      plot.append(bar, val);

      if (r.i === null) {
        row.classList.add("others");
        nameEl.textContent = `${r.count} others combined`;
        attachTip(row, () =>
          `<b>${r.count} other measurements</b><br>Combined push of ${fmtSigned(r.value)} on the log-odds scale.`);
        row.tabIndex = 0;
      } else {
        const f = m.features[r.i];
        const n = prettyName(f.name);
        nameEl.innerHTML = `${n.base} <small>${n.stat}</small>`;
        row.classList.add("clickable");
        row.tabIndex = 0;
        row.setAttribute("aria-label",
          `${n.full}: ${fmtValue(state.values[r.i])}, pushes ${r.value >= 0 ? "toward malignant" : "toward benign"} by ${Math.abs(r.value).toFixed(2)}. Press Enter to edit.`);
        attachTip(row, () => {
          const dir = r.value >= 0 ? "toward malignant" : "toward benign";
          return `<b>${n.full}: ${fmtValue(state.values[r.i])}</b><br>` +
            `Typical benign ${fmtValue(f.median_benign)}, typical malignant ${fmtValue(f.median_malignant)}.<br>` +
            `Pushes ${dir} by ${Math.abs(r.value).toFixed(2)} log-odds. Click to edit.`;
        });
        row.addEventListener("click", () => { hideTip(); jumpToFeature(r.i); });
        row.addEventListener("keydown", (e) => {
          if (e.key === "Enter" || e.key === " ") { e.preventDefault(); hideTip(); jumpToFeature(r.i); }
        });
      }
      row.append(nameEl, plot);
      host.appendChild(row);
    });

    // Plain-language summary.
    const pos = top.filter((i) => contribs[i] > 0);
    const neg = top.filter((i) => contribs[i] < 0);
    const base = sigmoid(m.model.intercept);
    const describe = (i) => {
      const f = m.features[i];
      return `${prettyName(f.name).full.toLowerCase()} of ${fmtValue(state.values[i])}`;
    };
    let text = `An average case starts at ${fmtPct(base)}. `;
    if (pos.length) text += `The strongest push toward malignant is ${describe(pos[0])}`;
    if (pos.length && neg.length) text += `; the strongest toward benign is ${describe(neg[0])}.`;
    else if (pos.length) text += ".";
    else if (neg.length) text += `The strongest push toward benign is ${describe(neg[0])}.`;
    if (Math.abs(p - state.threshold) < 0.1) text += " This one is close to the threshold, so treat it as uncertain.";
    $("why-summary").textContent = text;
  }

  // ---------- performance ----------

  function confusionAt(t) {
    let tp = 0, fp = 0, tn = 0, fn = 0;
    for (const [p, y] of state.model.metrics.test_predictions) {
      const pred = p >= t ? 1 : 0;
      if (pred === 1 && y === 1) tp++;
      else if (pred === 1 && y === 0) fp++;
      else if (pred === 0 && y === 0) tn++;
      else fn++;
    }
    return { tp, fp, tn, fn };
  }

  function renderPerformance() {
    const t = state.threshold;
    const { tp, fp, tn, fn } = confusionAt(t);
    const pct = (a, b) => (b ? `${(100 * a / b).toFixed(1)}%` : "–");
    $("threshold-out").textContent = `${Math.round(t * 100)}%`;

    const stats = [
      ["Sensitivity", pct(tp, tp + fn), `${tp} of ${tp + fn} malignant masses caught`],
      ["Specificity", pct(tn, tn + fp), `${tn} of ${tn + fp} benign masses cleared`],
      ["Precision", pct(tp, tp + fp), `of masses flagged, the share truly malignant`],
      ["ROC AUC", state.model.metrics.roc_auc.toFixed(3), "ranking quality, independent of threshold"],
    ];
    $("stats").innerHTML = stats.map(([k, v, note]) =>
      `<div class="stat"><dt>${k}</dt><dd>${v}<small>${note}</small></dd></div>`).join("");

    $("cm").innerHTML = `
      <caption>Test set outcomes at this threshold</caption>
      <thead><tr><td class="corner"></td><th scope="col">Predicted benign</th><th scope="col">Predicted malignant</th></tr></thead>
      <tbody>
        <tr><th scope="row">Actually benign</th>
          <td class="ok"><strong>${tn}</strong><span>correctly cleared</span></td>
          <td class="alarm"><strong>${fp}</strong><span>false alarms</span></td></tr>
        <tr><th scope="row">Actually malignant</th>
          <td class="miss"><strong>${fn}</strong><span>missed</span></td>
          <td class="ok"><strong>${tp}</strong><span>correctly flagged</span></td></tr>
      </tbody>`;
  }

  // ---------- global view ----------

  function renderImportance() {
    const feats = state.model.features.slice()
      .sort((a, b) => b.mean_abs_contribution - a.mean_abs_contribution)
      .slice(0, 10);
    const max = feats[0].mean_abs_contribution;
    const host = $("importance");
    host.innerHTML = "";
    feats.forEach((f) => {
      const n = prettyName(f.name);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.setAttribute("role", "listitem");
      row.tabIndex = 0;
      const w = (f.mean_abs_contribution / max) * 78;
      const dir = f.coef >= 0 ? "malignant" : "benign";
      row.innerHTML =
        `<div class="bar-name">${n.base} <small>${n.stat}</small></div>` +
        `<div class="bar-plot"><span class="bar ${f.coef >= 0 ? "pos" : "neg"}" style="width:${w}%"></span>` +
        `<span class="bar-value" style="left:calc(${w}% + 6px)">${f.mean_abs_contribution.toFixed(2)}</span></div>`;
      row.setAttribute("aria-label", `${n.full}: average push ${f.mean_abs_contribution.toFixed(2)}; higher values point toward ${dir}.`);
      attachTip(row, () =>
        `<b>${n.full}</b><br>Average push of ${f.mean_abs_contribution.toFixed(2)} log-odds per case.<br>` +
        `Higher values point toward ${dir} (weight ${fmtSigned(f.coef)} per standard deviation).`);
      host.appendChild(row);
    });
    const note = document.createElement("p");
    note.className = "hint";
    note.textContent = "Pink bars: higher values point toward malignant. Blue bars: higher values point toward benign.";
    host.appendChild(note);
  }

  function renderComparison() {
    const tbody = $("compare").querySelector("tbody");
    tbody.innerHTML = state.model.comparison.map((r) => `
      <tr class="${r.model === "Logistic regression" ? "chosen" : ""}">
        <td>${r.model}</td>
        <td>${r.roc_auc.toFixed(3)}</td>
        <td>${(r.recall * 100).toFixed(1)}%</td>
        <td>${(r.accuracy * 100).toFixed(1)}%</td>
      </tr>`).join("");
  }

  // ---------- boot ----------

  async function init() {
    $("repo-link").href = REPO_URL;
    try {
      const res = await fetch("model.json");
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      state.model = await res.json();
    } catch (err) {
      $("verdict-label").textContent = "Model file missing";
      $("why-summary").textContent =
        "model.json could not be loaded. Run `python -m bcxai.train`, then serve the docs folder over HTTP (for example `python -m http.server -d docs`).";
      return;
    }

    setupTabs();
    const list = buildExamples();
    const threshold = $("threshold");
    threshold.addEventListener("input", () => {
      state.threshold = parseFloat(threshold.value);
      renderPerformance();
      renderResult();
    });

    renderPerformance();
    renderImportance();
    renderComparison();

    const start = list.findIndex((e) => e.label === "Borderline");
    const buttons = document.querySelectorAll("#examples button");
    const i = start >= 0 ? start : 0;
    loadExample(list[i], buttons[i]);
  }

  init();
})();
