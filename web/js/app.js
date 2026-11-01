// Event-stream player for the When-to-Speak demo.
//
// Pure static JS: loads data/traces.json (events + gold labels + per-model
// decisions + full-eval metrics), replays the stream, and recomputes the
// metrics live with exactly the same formulas as training/src/evaluate.py.
//
// All user-facing strings come from js/i18n.js, so the whole page can switch
// between Chinese and English at runtime (button in the top-right corner).
//
// When the compiled WebLLM records land (F3.1), this file is the only place
// that needs a real inference branch; the UI and statistics stay unchanged.

(function () {
  "use strict";

  var ACTIONS = ["notify", "digest", "silence"];

  var state = {
    traces: null,
    modelId: "base",
    index: -1,
    timer: null,
    renderedCount: 0,
    gpu: "checking",
    loadError: null,
  };

  // Metric math lives in js/metrics.js (shared with the node verification step).
  var computeMetrics = window.WTSMetrics.computeMetrics;
  var i18n = window.WTSI18n;

  function t(key, vars) { return i18n.t(key, vars); }
  function sourceLabel(value) { return i18n.label("source", value); }
  function focusLabel(value) { return i18n.label("focus", value); }
  function actionLabel(value) { return i18n.label("action", value); }

  function metricsFor(events, modelId) {
    var records = events.map(function (entry) {
      return {
        gold: entry.gold,
        pred: entry.pred[modelId].decision,
        json_valid: entry.pred[modelId].json_valid,
      };
    });
    return computeMetrics(records);
  }

  // ---------------------------------------------------------------------
  // DOM helpers
  // ---------------------------------------------------------------------
  function el(id) { return document.getElementById(id); }

  function actionBadge(action) {
    return '<span class="action-badge action-' + action + '">' +
      actionLabel(action) + " · " + action + "</span>";
  }

  function errorType(gold, pred, jsonValid) {
    if (!jsonValid) return "invalid_json";
    if (gold.action === pred.action) {
      return gold.channel !== pred.channel ? "wrong_channel" : "exact";
    }
    if (pred.action === "notify") return "over_notify";
    if (gold.action === "notify") return "missed_notify";
    return "wrong_action";
  }

  function formatTime(iso) {
    // received_at is ISO-8601 with offset; show MM-DD HH:MM in local time.
    var dt = new Date(iso);
    if (isNaN(dt.getTime())) return iso.slice(5, 16).replace("T", " ");
    var pad = function (n) { return String(n).padStart(2, "0"); };
    return pad(dt.getMonth() + 1) + "-" + pad(dt.getDate()) + " " +
      pad(dt.getHours()) + ":" + pad(dt.getMinutes());
  }

  function modelTagline(model) {
    if (i18n.getLang() === "en" && model.tagline_en) return model.tagline_en;
    return model.tagline || model.id;
  }

  // ---------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------
  function renderModelTabs() {
    var container = el("modelSwitch");
    container.innerHTML = "";
    state.traces.models.forEach(function (model) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "model-tab" + (model.id === state.modelId ? " active" : "");
      button.setAttribute("role", "tab");
      button.setAttribute("aria-selected", model.id === state.modelId ? "true" : "false");
      button.innerHTML =
        '<span class="tab-label">' + model.label + "</span>" +
        '<span class="tab-tag">' + modelTagline(model) + "</span>";
      button.addEventListener("click", function () { switchModel(model.id); });
      container.appendChild(button);
    });
  }

  function switchModel(modelId) {
    if (modelId === state.modelId) return;
    state.modelId = modelId;
    renderModelTabs();
    rebuildStream();
    renderCurrent();
    renderMetrics();
    renderCompare();
  }

  function eventItem(entry, index) {
    var event = entry.event;
    var pred = entry.pred[state.modelId];
    var isMatch = errorType(entry.gold, pred.decision, pred.json_valid) === "exact";
    var div = document.createElement("div");
    div.className = "event-item" + (index === state.index ? " current" : "");
    div.innerHTML =
      '<div class="event-time">' + formatTime(event.received_at) + "</div>" +
      '<div class="event-main">' +
        '<div class="event-subject">' +
          '<span class="source-pill">' + sourceLabel(event.source) + "</span>" +
          escapeHtml(event.subject) +
        "</div>" +
        '<div class="event-meta">' + escapeHtml(event.sender) + " · " +
          focusLabel(event.context.user_focus) +
        "</div>" +
      "</div>" +
      "<div>" + actionBadge(pred.decision.action) +
        '<span class="match-dot ' + (isMatch ? "match-ok" : "match-bad") + '" title="' +
        (isMatch ? t("match.same") : t("match.diff")) + '"></span>' +
      "</div>";
    return div;
  }

  function rebuildStream() {
    var stream = el("stream");
    stream.innerHTML = "";
    state.renderedCount = 0;
    if (state.loadError) {
      stream.innerHTML = '<div class="stream-empty">' + t("stream.loadFail") + "</div>";
      return;
    }
    if (!state.traces || state.index < 0) {
      stream.innerHTML = '<div class="stream-empty">' + t("stream.empty") + "</div>";
      return;
    }
    appendUpTo(state.index, true);
  }

  function appendUpTo(index, rebuild) {
    var stream = el("stream");
    if (rebuild) stream.innerHTML = "";
    for (var i = state.renderedCount; i <= index; i += 1) {
      stream.appendChild(eventItem(state.traces.events[i], i));
    }
    state.renderedCount = Math.max(state.renderedCount, index + 1);
    var current = stream.querySelectorAll(".event-item");
    if (state.index >= 0 && current.length) {
      current[current.length - 1].classList.add("current");
      if (current.length > 1) current[current.length - 2].classList.remove("current");
    }
    stream.scrollTop = stream.scrollHeight;
  }

  function renderAutoButton() {
    el("autoBtn").textContent = state.timer ? t("controls.pause") : t("controls.auto");
  }

  function renderProgress() {
    var total = state.traces ? state.traces.events.length : 0;
    el("counter").textContent = Math.max(state.index + 1, 0) + " / " + total;
    el("progressFill").style.width = total ? ((state.index + 1) / total * 100) + "%" : "0%";
    el("stepBtn").disabled = !state.traces || state.index >= total - 1;
    el("autoBtn").disabled = !state.traces || state.index >= total - 1;
  }

  function renderCurrent() {
    var detail = el("eventDetail");
    var grid = el("decisionGrid");
    var rawBlock = el("rawBlock");
    var verdict = el("verdict");
    var sub = el("decisionSub");

    if (!state.traces || state.index < 0) {
      detail.className = "event-detail empty";
      detail.innerHTML = '<p class="empty-text">' + t("decision.empty") + "</p>";
      grid.classList.add("hidden");
      rawBlock.classList.add("hidden");
      verdict.classList.add("hidden");
      sub.textContent = t("decision.subIdle");
      return;
    }

    var entry = state.traces.events[state.index];
    var event = entry.event;
    var pred = entry.pred[state.modelId];
    var model = state.traces.models.find(function (m) { return m.id === state.modelId; });
    var kind = errorType(entry.gold, pred.decision, pred.json_valid);

    detail.className = "event-detail";
    detail.innerHTML =
      '<div class="event-headline"><h3>' + escapeHtml(event.subject) + "</h3></div>" +
      '<p class="event-snippet">' + escapeHtml(event.snippet) + "</p>" +
      '<div class="event-facts">' +
        '<span class="fact">' + sourceLabel(event.source) + "</span>" +
        '<span class="fact">' + escapeHtml(event.sender) + "</span>" +
        '<span class="fact">' + focusLabel(event.context.user_focus) + "</span>" +
        (event.mentions_user ? '<span class="fact">' + t("facts.mention") + "</span>" : "") +
        (event.is_direct ? '<span class="fact">' + t("facts.direct") + "</span>" : "") +
        (event.time_sensitive ? '<span class="fact">' + t("facts.timeSensitive") + "</span>" : "") +
        (event.context.duplicate_of ? '<span class="fact">' + t("facts.duplicate") + "</span>" : "") +
        '<span class="fact">' + t("facts.recent", { n: event.context.recent_notifications }) + "</span>" +
      "</div>";

    grid.classList.remove("hidden");
    rawBlock.classList.remove("hidden");
    el("modelCardLabel").textContent = t("decision.modelLabel", { label: model.label });

    renderDecisionBody(el("goldBody"), entry.gold, true);
    renderDecisionBody(el("modelBody"), pred.decision, pred.json_valid);

    el("rawOutput").textContent = pred.raw;

    verdict.classList.remove("hidden", "ok", "bad");
    verdict.classList.add(kind === "exact" ? "ok" : "bad");
    verdict.textContent = t("verdict." + kind) +
      (kind === "invalid_json" ? t("verdict.invalidSuffix") : "");

    sub.textContent = t("decision.subLive", {
      n: state.index + 1,
      time: formatTime(event.received_at),
    });
  }

  function renderDecisionBody(container, decision, jsonValidOrIsGold) {
    var valid = jsonValidOrIsGold;
    var warning = "";
    if (typeof valid === "boolean" && !valid) {
      warning = '<div class="decision-reason" style="color:var(--red)">' +
        t("decision.fallbackNote") + "</div>";
    }
    container.innerHTML =
      '<div class="decision-row"><span>action</span><b>' + decision.action + "</b></div>" +
      '<div class="decision-row"><span>channel</span><b>' + decision.channel + "</b></div>" +
      '<div class="decision-row"><span>urgency</span><b>' + decision.urgency + "</b></div>" +
      '<div class="decision-message">' + (decision.message ? escapeHtml(decision.message) :
        t("decision.silenceMessage")) + "</div>" +
      '<div class="decision-reason">' + escapeHtml(decision.reason) + "</div>" + warning;
  }

  function renderMetrics() {
    var metrics = state.traces
      ? metricsFor(state.traces.events.slice(0, state.index + 1), state.modelId)
      : null;
    // An empty replay (index = -1) has no meaningful metrics yet.
    if (metrics && metrics.total === 0) metrics = null;

    var tiles = [
      { key: "json_valid_rate", label: t("metric.json"), value: metrics ? metrics.json_valid_rate : 0, mode: "high", good: 0.95, warn: 0.9 },
      { key: "notify_precision", label: t("metric.precision"), value: metrics ? metrics.notify_precision : 0, mode: "high", good: 0.8, warn: 0.6 },
      { key: "notify_recall", label: t("metric.recall"), value: metrics ? metrics.notify_recall : 0, mode: "high", good: 0.9, warn: 0.75 },
      { key: "noise_rate", label: t("metric.noise"), value: metrics ? metrics.noise_rate : 0, mode: "low", good: 0.05, warn: 0.12 },
      { key: "miss_rate", label: t("metric.miss"), value: metrics ? metrics.miss_rate : 0, mode: "low", good: 0.05, warn: 0.1 },
      { key: "macro_f1", label: t("metric.macro"), value: metrics ? metrics.macro_f1 : 0, mode: "high", good: 0.9, warn: 0.75 },
    ];

    el("metricGrid").innerHTML = tiles.map(function (tile) {
      var cls = "";
      if (metrics) {
        if (tile.mode === "high") {
          cls = tile.value >= tile.good ? "good" : (tile.value >= tile.warn ? "warn" : "bad");
        } else {
          cls = tile.value <= tile.good ? "good" : (tile.value <= tile.warn ? "warn" : "bad");
        }
      }
      return '<div class="metric ' + cls + '">' +
        '<div class="m-label">' + tile.label + "</div>" +
        '<div class="m-value">' + (metrics ? formatPct(tile.value) : "—") + "</div>" +
        "</div>";
    }).join("");

    el("metricsSub").textContent = metrics
      ? t("metrics.subLive", { total: metrics.total, model: state.modelId })
      : t("metrics.subIdle");

    renderConfusion(metrics);
  }

  function formatPct(value) {
    return (value * 100).toFixed(1) + "%";
  }

  function renderConfusion(metrics) {
    var table = el("confusionTable");
    if (!metrics) { table.innerHTML = ""; return; }
    var header = "<tr><th>gold \\ pred</th>" + ACTIONS.map(function (a) {
      return "<th>" + a + "</th>";
    }).join("") + "</tr>";
    var rows = ACTIONS.map(function (gold) {
      var cells = ACTIONS.map(function (pred) {
        var count = metrics.confusion[gold + "->" + pred] || 0;
        return '<td class="' + (gold === pred ? "diag" : "") + '">' + count + "</td>";
      }).join("");
      return '<tr><td class="row-label">' + gold + "</td>" + cells + "</tr>";
    }).join("");
    table.innerHTML = header + rows;
  }

  function renderCompare() {
    var container = el("compareGrid");
    if (!state.traces) return;
    container.innerHTML = state.traces.models.map(function (model) {
      var metrics = state.traces.metrics[model.id];
      return '<div class="compare-card' + (model.id === state.modelId ? " active" : "") + '">' +
        '<div class="c-label">' + model.label + "</div>" +
        '<div class="c-sub">' + modelTagline(model) + "</div>" +
        '<div class="c-row"><span>' + t("compare.noise") + "</span><b>" + formatPct(metrics.noise_rate) + "</b></div>" +
        '<div class="c-row"><span>' + t("compare.miss") + "</span><b>" + formatPct(metrics.miss_rate) + "</b></div>" +
        '<div class="c-row"><span>' + t("compare.json") + "</span><b>" + formatPct(metrics.json_valid_rate) + "</b></div>" +
        "</div>";
    }).join("");
  }

  // ---------------------------------------------------------------------
  // Playback
  // ---------------------------------------------------------------------
  function step() {
    if (!state.traces || state.index >= state.traces.events.length - 1) return;
    state.index += 1;
    appendUpTo(state.index, state.renderedCount === 0);
    renderCurrent();
    renderMetrics();
    renderProgress();
  }

  function stopAuto() {
    if (state.timer) {
      clearInterval(state.timer);
      state.timer = null;
    }
    renderAutoButton();
  }

  function toggleAuto() {
    if (state.timer) { stopAuto(); return; }
    if (!state.traces) return;
    if (state.index >= state.traces.events.length - 1) return;
    state.timer = setInterval(function () {
      if (state.index >= state.traces.events.length - 1) { stopAuto(); return; }
      step();
    }, 900);
    renderAutoButton();
  }

  function reset() {
    stopAuto();
    state.index = -1;
    rebuildStream();
    renderCurrent();
    renderMetrics();
    renderProgress();
  }

  // ---------------------------------------------------------------------
  // Environment / fallback
  // ---------------------------------------------------------------------
  function renderBanner() {
    var banner = el("banner");
    var text = "";
    if (state.gpu === "unavailable") text = t("banner.noWebGPU");
    else if (state.gpu === "no-adapter") text = t("banner.noAdapter");
    if (state.loadError) {
      text = (text ? text + "<br>" : "") +
        t("banner.loadFail", { error: escapeHtml(state.loadError) });
    }
    if (text) {
      banner.innerHTML = text;
      banner.classList.remove("hidden");
    } else {
      banner.innerHTML = "";
      banner.classList.add("hidden");
    }
  }

  function renderGpu() {
    var badge = el("gpuBadge");
    badge.classList.remove("badge-ok", "badge-bad");
    if (state.gpu === "ok") {
      badge.textContent = t("gpu.ok");
      badge.classList.add("badge-ok");
      return;
    }
    badge.textContent = t("gpu." + state.gpu);
    if (state.gpu !== "checking") badge.classList.add("badge-bad");
  }

  function detectWebGPU() {
    state.gpu = "checking";
    renderGpu();
    if (!("gpu" in navigator)) {
      state.gpu = "unavailable";
      renderGpu();
      renderBanner();
      return;
    }
    navigator.gpu.requestAdapter().then(function (adapter) {
      state.gpu = adapter ? "ok" : "no-adapter";
      renderGpu();
      renderBanner();
    }).catch(function () {
      state.gpu = "error";
      renderGpu();
      renderBanner();
    });
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  // ---------------------------------------------------------------------
  // Localization
  // ---------------------------------------------------------------------
  function renderLangButton() {
    // The button shows the language you would switch to.
    el("langBtn").setAttribute("aria-label", t("lang.switch"));
    el("langBtn").setAttribute("title", t("lang.switch"));
  }

  function relocalize() {
    i18n.applyDom();
    renderLangButton();
    el("modelSwitch").setAttribute("aria-label", t("hero.switchAria"));
    renderGpu();
    renderBanner();
    renderAutoButton();

    if (!state.traces) {
      rebuildStream();
      renderCurrent();
      renderMetrics();
      renderProgress();
      return;
    }

    el("streamSub").textContent = t("stream.sub", { total: state.traces.events.length });
    el("modeBadge").textContent = state.traces._meta.mode === "mock" ? t("badge.mock") : t("badge.real");
    el("modeBadge").title = state.traces._meta.note || t("badge.mockTitle");
    renderModelTabs();
    rebuildStream();
    renderCurrent();
    renderMetrics();
    renderCompare();
    renderProgress();
  }

  // ---------------------------------------------------------------------
  // Init
  // ---------------------------------------------------------------------
  function init() {
    i18n.applyDom();
    renderLangButton();
    renderGpu();
    renderMetrics();

    el("langBtn").addEventListener("click", function () {
      i18n.setLang(i18n.getLang() === "zh" ? "en" : "zh");
    });
    i18n.onChange(relocalize);

    detectWebGPU();

    fetch("data/traces.json")
      .then(function (response) {
        if (!response.ok) throw new Error("HTTP " + response.status);
        return response.json();
      })
      .then(function (traces) {
        state.traces = traces;
        state.index = -1;
        state.modelId = traces.models[0].id;
        el("streamSub").textContent = t("stream.sub", { total: traces.events.length });
        el("modeBadge").textContent = traces._meta.mode === "mock" ? t("badge.mock") : t("badge.real");
        if (traces._meta.mode === "mock") {
          el("modeBadge").title = traces._meta.note;
        }
        renderModelTabs();
        rebuildStream();
        renderCurrent();
        renderMetrics();
        renderCompare();
        renderProgress();
      })
      .catch(function (error) {
        state.loadError = error.message;
        renderBanner();
        rebuildStream();
      });

    el("stepBtn").addEventListener("click", step);
    el("autoBtn").addEventListener("click", toggleAuto);
    el("resetBtn").addEventListener("click", reset);

    // Expose the metric function for verification (browser + node checks).
    window.WTS = {
      state: state,
      computeMetrics: computeMetrics,
      metricsFor: metricsFor,
    };
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
