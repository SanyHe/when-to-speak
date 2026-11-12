// Verifies that the browser metric formulas and the committed web traces
// match the offline Python evaluation report exactly.
//
// Covers both replay datasets:
//   web/data/traces.json        <- data/eval/eval_report[_hf].json
//   web/data/autoresearch.json  <- autoresearch/results/<trial>/metrics.json
//
// Run: node training/tests/verify_web_metrics.js
// Exits non-zero on any drift (used in the Pages workflow).

"use strict";

const fs = require("fs");
const path = require("path");
const root = path.resolve(__dirname, "..", "..");

const metricsLib = require(path.join(root, "web", "js", "metrics.js"));

let failed = false;
const checked = [];

function fail(message) {
  failed = true;
  console.error("FAIL: " + message);
}

function recordsFor(traces, modelId) {
  return traces.events.map(function (entry) {
    return {
      gold: entry.gold,
      pred: entry.pred[modelId].decision,
      json_valid: entry.pred[modelId].json_valid,
    };
  });
}

function sameJson(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}

// --- eval replay (mock or real checkpoints) -------------------------------
const traces = require(path.join(root, "web", "data", "traces.json"));

// The committed traces may come from deterministic mocks or from real
// checkpoints (mode "hf"); each mode has its own report to match against.
const mode = (traces._meta && traces._meta.mode) || "mock";
const reportFile = mode === "hf" ? "eval_report_hf.json" : "eval_report.json";
const report = require(path.join(root, "data", "eval", reportFile));

for (const model of traces.models) {
  const jsMetrics = metricsLib.computeMetrics(recordsFor(traces, model.id));
  const pyMetrics = report.metrics[model.id];

  if (!sameJson(jsMetrics, pyMetrics)) {
    fail("JS metrics differ from eval_report.json for model " + model.id);
  }
  if (!sameJson(traces.metrics[model.id], pyMetrics)) {
    fail("traces.json metrics are stale for model " + model.id);
  }
}
checked.push("web/data/traces.json vs data/eval/" + reportFile);

// --- AutoResearch dev replay ---------------------------------------------
const autoFile = path.join(root, "web", "data", "autoresearch.json");
if (fs.existsSync(autoFile)) {
  const auto = require(autoFile);
  for (const model of auto.models) {
    const jsMetrics = metricsLib.computeMetrics(recordsFor(auto, model.id));
    if (!sameJson(auto.metrics[model.id], jsMetrics)) {
      fail("autoresearch.json metrics are stale for model " + model.id);
    }
    const pyMetrics = require(path.join(
      root, "autoresearch", "results", model.trial, "metrics.json"
    )).metrics;
    if (!sameJson(jsMetrics, pyMetrics)) {
      fail("JS metrics differ from autoresearch/results for model " + model.id);
    }
  }
  checked.push("web/data/autoresearch.json vs autoresearch/results/<trial>/metrics.json");
}

if (failed) {
  process.exit(1);
}
console.log("OK: web metrics match the offline reports for all models.");
checked.forEach(function (line) { console.log("  - " + line); });
